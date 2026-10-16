#!/usr/bin/env python3
"""Phase 3a of the MICA 2026 benchmark: pull and filter the candidate pool.

What it does
  1. Asks the RCSB Search API for every cryo-EM entry released from --start to --freeze
     (inclusive), one week at a time. Server-side filters (--prefilter) are off by default:
     the audit in the first real run found an entry that passes every file filter but that
     the server-side filters removed. Every entry is therefore downloaded and judged on its
     own file.
  2. Downloads each entry's mmCIF file and applies the parts of decisions 3 and 4 in
     docs/SPEC.md that need no map: release window, single-particle, resolution <= 4.0 A,
     protein only, <= 700 modeled residues, <= 3 unique chain sequences, <= 5% UNK.
     One operational rule is added so that an entry means one map and one model: exactly
     one EMDB map must be linked to the entry. A file above 40 MB compressed is not parsed:
     it holds far more than 700 residues (filter file_size).
  3. Asks the EMDB API for the release date of that map and drops entries whose map was not
     released inside the window.
  4. Writes the pool files: CSV tables, a FASTA file, the frozen ID list and run_info.json.

CC_mask (map-model correlation, decision 4) is NOT applied here. It needs Phenix and the maps
and is done in Phase 3b, on the candidates this script writes.

Usage
  python pull_pool.py --freeze 2026-10-12 --out pool_out

Re-running continues from the cache in OUT/cache. Dependencies: requests and gemmi.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import platform
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import gemmi
from gemmi import cif

try:
    import resource  # Unix only; used to report peak memory
except ImportError:  # pragma: no cover
    resource = None

SCRIPT_VERSION = "1.1"
PARSER_VERSION = "1"

# Locked values from docs/SPEC.md (decisions 3 and 4). Command-line overrides exist for tests only
# and are recorded in run_info.json.
SPEC = {
    "window_start": "2026-01-01",
    "freeze": "2026-10-12",
    "max_resolution": 4.0,
    "max_modeled_residues": 700,
    "max_unique_sequences": 3,
    "max_unk_fraction": 0.05,
}

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
CIF_URLS = [
    "https://files.rcsb.org/download/{ID}.cif.gz",
    "https://files.wwpdb.org/pub/pdb/data/structures/all/mmCIF/{id}.cif.gz",
]
EMDB_URL = "https://www.ebi.ac.uk/emdb/api/entry/{EMD}"
MAP_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{EMD}/map/emd_{num}.map.gz"

# Optional server-side prefilters (--prefilter). They only reduce downloads, and each has a margin
# so that the exact check on the file decides. A prefilter is dropped if the API rejects it or
# returns zero hits. They are off by default (see the module docstring).
RESOLUTION_MARGIN = 4.1
RESIDUE_MARGIN = 800
MIN_PROBE_HITS = 20  # a zero-hit probe only counts as a sign of a wrong filter above this many entries

# A compressed mmCIF above this size is not parsed. mmCIF.gz compresses about 6 to 8 times, so it
# holds at least 150 MB of text, which is at least 40,000 residues; gemmi needs about 9 times the
# text size in memory, and the free Colab session has 12 GB.
MAX_GZ_BYTES = 40_000_000

# Ordered filters. The first failing one is recorded for each entry.
FILTERS = [
    ("file_size", "mmCIF file of at most 40 MB compressed (a larger file holds far more than 700 residues)"),
    ("window", "PDB model released inside the window"),
    ("single_particle", "Single-particle reconstruction"),
    ("resolution", "Resolution at most 4.0 A"),
    ("protein_only", "Protein only (every polymer entity is polypeptide(L))"),
    ("modeled_residues", "1 to 700 modeled residues in total"),
    ("unique_sequences", "At most 3 unique chain sequences"),
    ("unk_fraction", "At most 5% UNK residues"),
    ("one_emdb_map", "Exactly one EMDB map linked to the entry"),
    ("emdb_map_released", "EMDB map released inside the window (EMDB API)"),
]
STAGE1 = ("window", "single_particle", "resolution", "protein_only")

CANARIES = ["5A1A", "6VXX"]  # old, well-known single-particle entries; used to test the parser

COLUMNS = [
    "pdb_id", "emdb_id", "stratum", "release_date_pdb", "emdb_map_release", "emdb_status",
    "emdb_release_verified", "map_size_kb", "resolution_A", "recon_method",
    "n_polymer_entities", "poly_types", "n_modeled_residues", "n_unk", "unk_fraction",
    "n_chains", "n_unique_seq", "longest_entity", "longest_seq_len", "seqres_total_len",
    "entities", "n_emdb_links", "emdb_ids_all", "multi_recon_rows", "first_fail",
    "passes_cheap_filters", "cif_url", "map_url", "note", "error",
]

_LOG = None
_tls = threading.local()


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def log(msg):
    line = "[%s] %s" % (now_utc().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if _LOG is not None:
        _LOG.write(line + "\n")
        _LOG.flush()


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------
def session():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        kw = dict(total=5, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
        try:
            retry = Retry(allowed_methods=None, **kw)
        except TypeError:  # old urllib3
            retry = Retry(method_whitelist=False, **kw)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "mica-benchmark-pool/%s (academic benchmark)" % SCRIPT_VERSION})
        _tls.s = s
    return s


# --------------------------------------------------------------------------------------
# RCSB Search API
# --------------------------------------------------------------------------------------
class SearchRejected(Exception):
    """The Search API answered 400: an attribute or value is not understood."""


def terminal(attribute, operator, value):
    return {"type": "terminal", "service": "text",
            "parameters": {"attribute": attribute, "operator": operator, "value": value}}


OPTIONAL = {
    "single_particle": terminal("em_experiment.reconstruction_method", "exact_match", "SINGLE PARTICLE"),
    "resolution": terminal("rcsb_entry_info.resolution_combined", "less_or_equal", RESOLUTION_MARGIN),
    "protein_only": terminal("rcsb_entry_info.selected_polymer_entity_types", "exact_match", "Protein (only)"),
    "residues": terminal("rcsb_entry_info.deposited_modeled_polymer_monomer_count", "less_or_equal", RESIDUE_MARGIN),
}


PAGE_ROWS = 1000
# The base query has a few forms the API might want. calibrate_search() picks the first one it accepts.
DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT00:00:00Z")
METHOD_NODES = (
    terminal("exptl.method", "exact_match", "ELECTRON MICROSCOPY"),
    terminal("rcsb_entry_info.experimental_method", "exact_match", "EM"),
)
REQUEST_OPTIONS = (
    {"paginate": {"start": 0, "rows": PAGE_ROWS}, "results_content_type": ["experimental"]},
    {"return_all_hits": True, "results_content_type": ["experimental"]},
    {"return_all_hits": True},
)
SEARCH_CFG = {"date": 0, "method": 0, "ro": 0}  # indexes into the three tuples above


def _fmt(d):
    return dt.datetime(d.year, d.month, d.day).strftime(DATE_FORMATS[SEARCH_CFG["date"]])


def build_query(a, b, extra, rows=None):
    nodes = [
        terminal("rcsb_accession_info.initial_release_date", "greater_or_equal", _fmt(a)),
        terminal("rcsb_accession_info.initial_release_date", "less", _fmt(b)),
        METHOD_NODES[SEARCH_CFG["method"]],
    ] + list(extra)
    options = json.loads(json.dumps(REQUEST_OPTIONS[SEARCH_CFG["ro"]]))
    if rows is not None and "paginate" in options:
        options["paginate"]["rows"] = rows
    return {"query": {"type": "group", "logical_operator": "and", "nodes": nodes},
            "return_type": "entry", "request_options": options}


def _post(url, body):
    r = session().post(url, json=body, timeout=(15, 120))
    if r.status_code == 204:
        return [], 0
    if r.status_code == 400:
        raise SearchRejected(r.text[:300])
    r.raise_for_status()
    js = r.json()
    ids = [x["identifier"] if isinstance(x, dict) else str(x) for x in js.get("result_set", [])]
    return ids, js.get("total_count", len(ids))


def search_ids(url, a, b, extra):
    """IDs of cryo-EM entries released in [a, b). Splits the window if the API truncates."""
    ids, total = _post(url, build_query(a, b, extra))
    if total > len(ids):
        if (b - a).days <= 1:
            raise RuntimeError("search truncated for a single day %s: %d of %d hits" % (a, len(ids), total))
        mid = a + (b - a) // 2
        return search_ids(url, a, mid, extra) + search_ids(url, mid, b, extra)
    return ids


def calibrate_search(url, start, freeze):
    """Find the first form of the base query that the API accepts and that returns hits for the whole window."""
    last = "nothing tried"
    end = freeze + dt.timedelta(days=1)
    for m in range(len(METHOD_NODES)):
        for d in range(len(DATE_FORMATS)):
            for o in range(len(REQUEST_OPTIONS)):
                SEARCH_CFG.update({"date": d, "method": m, "ro": o})
                try:
                    _, total = _post(url, build_query(start, end, [], rows=1))
                except SearchRejected as e:
                    last = e.args[0]
                    continue
                if total > 0:
                    return dict(SEARCH_CFG), total
                last = "accepted but returned no hits"
    raise SearchRejected("no form of the base query was accepted with hits; last answer: %s" % last)


def week_windows(start, freeze):
    out, cur = [], start
    end = freeze + dt.timedelta(days=1)
    while cur < end:
        nxt = min(cur + dt.timedelta(days=7), end)
        out.append((cur, nxt))
        cur = nxt
    return out


def choose_prefilters(url, probe, n_base):
    """Probe each optional server-side filter on one busy week. Keep it only if the API accepts
    it and it still returns hits; the exact checks on the files decide in any case."""
    a, b = probe
    log("probe week %s to %s: %d cryo-EM entries" % (a, b, n_base))
    kept, dropped = [], []
    for key, node in OPTIONAL.items():
        try:
            n = len(search_ids(url, a, b, [node]))
        except SearchRejected as e:
            dropped.append({"filter": key, "why": "rejected by the API: " + e.args[0][:150]})
            continue
        if n == 0 and n_base >= MIN_PROBE_HITS:
            dropped.append({"filter": key, "why": "zero hits in the probe week"})
            continue
        kept.append(key)
    if kept:
        try:
            n_all = len(search_ids(url, a, b, [OPTIONAL[k] for k in kept]))
        except SearchRejected as e:
            dropped.append({"filter": "combination", "why": "rejected by the API: " + e.args[0][:150]})
            kept, n_all = [], 0
        if kept and n_all == 0 and n_base >= MIN_PROBE_HITS:
            dropped.append({"filter": "combination", "why": "zero hits in the probe week"})
            kept = []
    for d in dropped:
        log("search prefilter dropped: %(filter)s (%(why)s)" % d)
    log("search prefilters kept: %s" % (", ".join(kept) if kept else "none (every cryo-EM entry will be downloaded)"))
    return kept, dropped


# --------------------------------------------------------------------------------------
# mmCIF parsing
# --------------------------------------------------------------------------------------
def _col(block, tag):
    return [cif.as_string(x) for x in block.find_values(tag)]


def _is_float(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def norm_emd(code):
    m = re.search(r"(\d+)", code or "")
    return ("EMD-%s" % m.group(1)) if m else ""


def blank_record(pdb_id):
    rec = {c: "" for c in COLUMNS}
    rec.update({
        "pdb_id": pdb_id, "resolution_A": None, "n_modeled_residues": None, "n_unk": None,
        "unk_fraction": None, "n_chains": None, "n_unique_seq": None, "n_emdb_links": 0,
        "multi_recon_rows": False, "first_fail": None, "passes_cheap_filters": False,
        "emdb_release_verified": "", "emdb_ids": [], "entity_list": [], "exptl_method": "",
        "too_large": False,
    })
    return rec


def check(key, rec, spec):
    """True = passes, False = fails, None = the fields needed are not computed."""
    if key == "file_size":
        return not rec.get("too_large")
    if key == "window":
        d = rec.get("release_date_pdb")
        return None if not d else (spec["window_start"] <= d <= spec["freeze"])
    if key == "single_particle":
        return rec["recon_method"] == "SINGLE PARTICLE" and "ELECTRON MICROSCOPY" in rec["exptl_method"].upper()
    if key == "resolution":
        r = rec["resolution_A"]
        return r is not None and r <= spec["max_resolution"]
    if key == "protein_only":
        return rec["poly_types"] == "polypeptide(L)"
    if key == "modeled_residues":
        n = rec["n_modeled_residues"]
        return None if n is None else 1 <= n <= spec["max_modeled_residues"]
    if key == "unique_sequences":
        n = rec["n_unique_seq"]
        return None if n is None else 1 <= n <= spec["max_unique_sequences"]
    if key == "unk_fraction":
        f = rec["unk_fraction"]
        return None if f is None else f <= spec["max_unk_fraction"]
    if key == "one_emdb_map":
        return rec["n_emdb_links"] == 1
    if key == "emdb_map_released":
        d = rec.get("emdb_map_release")
        st = rec.get("emdb_status")
        if st and st.upper() not in ("REL", "RELEASED"):
            return False
        if not d:
            return None
        return spec["window_start"] <= d <= spec["freeze"]
    raise KeyError(key)


def first_failure(rec, spec, keys):
    for key, _ in FILTERS:
        if keys is not None and key not in keys:
            continue
        ok = check(key, rec, spec)
        if ok is False:
            return key
    return None


def parse_mmcif(text, pdb_id, spec, force_full=False):
    """Parse one entry. Atoms are read only if the entry passes the cheap header filters."""
    doc = cif.read_string(text)
    if len(doc) < 1:
        raise ValueError("no data block in the file")
    b = doc[0]
    rec = blank_record(pdb_id)
    rec["exptl_method"] = ";".join(_col(b, "_exptl.method"))
    recon = sorted({x.upper() for x in _col(b, "_em_experiment.reconstruction_method") if x not in ("?", ".")})
    rec["recon_method"] = ";".join(recon)
    res = [float(x) for x in _col(b, "_em_3d_reconstruction.resolution") if _is_float(x)]
    rec["resolution_A"] = max(res) if res else None
    n_recon = len(_col(b, "_em_3d_reconstruction.id"))
    rec["multi_recon_rows"] = n_recon > 1
    dates = [d for d in _col(b, "_pdbx_audit_revision_history.revision_date") if re.match(r"\d{4}-\d{2}-\d{2}", d)]
    rec["release_date_pdb"] = min(dates)[:10] if dates else ""

    ids = set()
    for dbid, code in zip(_col(b, "_database_2.database_id"), _col(b, "_database_2.database_code")):
        if dbid.upper() == "EMDB" and norm_emd(code):
            ids.add(norm_emd(code))
    for name, code in zip(_col(b, "_pdbx_database_related.db_name"), _col(b, "_pdbx_database_related.db_id")):
        if name.upper() == "EMDB" and norm_emd(code):
            ids.add(norm_emd(code))
    rec["emdb_ids"] = sorted(ids)
    rec["emdb_ids_all"] = ";".join(rec["emdb_ids"])
    rec["n_emdb_links"] = len(ids)
    if len(ids) == 1:
        rec["emdb_id"] = rec["emdb_ids"][0]

    poly_ids = _col(b, "_entity_poly.entity_id")
    poly_types = _col(b, "_entity_poly.type")
    poly_seqs = _col(b, "_entity_poly.pdbx_seq_one_letter_code_can")
    poly_seqs += [""] * (len(poly_ids) - len(poly_seqs))
    rec["n_polymer_entities"] = len(poly_ids)
    rec["poly_types"] = ";".join(sorted(set(poly_types)))
    seq_of = {e: re.sub(r"\s+", "", s) for e, s in zip(poly_ids, poly_seqs)}
    protein_ids = {e for e, t in zip(poly_ids, poly_types) if t == "polypeptide(L)"}

    fail = first_failure(rec, spec, STAGE1)
    if fail is None or force_full:
        fill_atoms(b, rec, protein_ids, seq_of)
    rec["first_fail"] = fail if fail is not None else first_failure(rec, spec, None)
    return rec


def fill_atoms(b, rec, protein_ids, seq_of):
    # gemmi columns are read lazily: no Python list is built for the (up to millions of) atoms
    asym = b.find_values("_atom_site.label_asym_id")
    ent = b.find_values("_atom_site.label_entity_id")
    seq = b.find_values("_atom_site.label_seq_id")
    comp = b.find_values("_atom_site.label_comp_id")
    model = b.find_values("_atom_site.pdbx_PDB_model_num")
    first_model = model[0] if len(model) else "1"
    residues = {}
    for a, e, s, c, m in zip(asym, ent, seq, comp, model):
        if m != first_model or s in (".", "?") or e not in protein_ids:
            continue
        key = (a, s)
        if key not in residues:
            residues[key] = (e, c)
    n = len(residues)
    n_unk = sum(1 for _, c in residues.values() if c == "UNK")
    rec["n_modeled_residues"] = n
    rec["n_unk"] = n_unk
    rec["unk_fraction"] = round(n_unk / n, 4) if n else None
    rec["n_chains"] = len({a for a, _ in residues})
    per_entity = {}
    for (a, _), (e, _c) in residues.items():
        d = per_entity.setdefault(e, {"chains": set(), "modeled": 0})
        d["chains"].add(a)
        d["modeled"] += 1
    ents = []
    for e, d in per_entity.items():
        s = seq_of.get(e, "")
        ents.append({"id": e, "len": len(s), "copies": len(d["chains"]), "modeled": d["modeled"], "seq": s})
    ents.sort(key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else 10 ** 9)
    rec["entity_list"] = ents
    rec["n_unique_seq"] = len({x["seq"] for x in ents}) if ents else 0
    rec["entities"] = "|".join("E%s:len=%d:copies=%d:modeled=%d" % (x["id"], x["len"], x["copies"], x["modeled"]) for x in ents)
    rec["seqres_total_len"] = sum(len(s) for s in {x["seq"] for x in ents})
    if ents:
        longest = max(ents, key=lambda x: (x["len"], -(int(x["id"]) if str(x["id"]).isdigit() else 10 ** 9)))
        rec["longest_entity"] = longest["id"]
        rec["longest_seq_len"] = longest["len"]


class TooLarge(Exception):
    """The compressed file is above MAX_GZ_BYTES and is not downloaded."""


def fetch_text(pdb_id, templates):
    last = "no source"
    for t in templates:
        url = t.format(ID=pdb_id.upper(), id=pdb_id.lower())
        try:
            r = session().get(url, timeout=(15, 180), stream=True)
            if r.status_code == 404:
                r.close()
                last = "404 at " + url
                continue
            r.raise_for_status()
            size = int(r.headers.get("Content-Length") or 0)
            if size > MAX_GZ_BYTES:
                r.close()
                raise TooLarge(size)
            data = r.content
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            return data.decode("utf-8", errors="replace")
        except TooLarge:
            raise
        except Exception as e:  # noqa: BLE001 - any failure moves on to the next source
            last = "%s: %s (%s)" % (type(e).__name__, str(e)[:150], url)
    raise RuntimeError(last)


def thresholds_key(spec):
    return json.dumps({k: spec[k] for k in sorted(spec)}, sort_keys=True)


def process_entry(pdb_id, cfg):
    path = os.path.join(cfg["cache"], "parsed", pdb_id + ".json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                js = json.load(f)
            if js.get("parser_version") == PARSER_VERSION and js.get("thresholds") == cfg["thr"]:
                return js["rec"], True
        except (OSError, ValueError):
            pass
    try:
        text = fetch_text(pdb_id, cfg["cif_urls"])
        rec = parse_mmcif(text, pdb_id, cfg["spec"])
        del text
    except TooLarge as e:
        rec = blank_record(pdb_id)
        rec["too_large"] = True
        rec["first_fail"] = "file_size"
        rec["note"] = "mmCIF.gz of %d MB, not parsed" % (int(e.args[0]) // 1_000_000)
    tmp = path + ".tmp%d" % threading.get_ident()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"parser_version": PARSER_VERSION, "thresholds": cfg["thr"], "rec": rec}, f)
    os.replace(tmp, path)
    return rec, False


# --------------------------------------------------------------------------------------
# EMDB API
# --------------------------------------------------------------------------------------
def find_key(obj, name):
    """First value stored under key `name` anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        for v in obj.values():
            r = find_key(v, name)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, name)
            if r is not None:
                return r
    return None


def first_leaf_string(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("valueOf_", "code", "value"):
            if k in obj:
                r = first_leaf_string(obj[k])
                if r:
                    return r
        for v in obj.values():
            r = first_leaf_string(v)
            if r:
                return r
    if isinstance(obj, list):
        for v in obj:
            r = first_leaf_string(v)
            if r:
                return r
    return None


def parse_emdb_json(js):
    out = {"map_release": None, "release_basis": None, "status": None, "size_kb": None}
    mr = find_key(js, "map_release")
    hr = find_key(js, "header_release")
    if isinstance(mr, str) and re.match(r"\d{4}-\d{2}-\d{2}", mr):
        out["map_release"], out["release_basis"] = mr[:10], "map_release"
    elif isinstance(hr, str) and re.match(r"\d{4}-\d{2}-\d{2}", hr):
        out["map_release"], out["release_basis"] = hr[:10], "header_release"
    out["status"] = first_leaf_string(find_key(js, "current_status"))
    sz = find_key(js, "size_kbytes")
    if isinstance(sz, (int, float)) or (isinstance(sz, str) and _is_float(sz)):
        out["size_kb"] = int(float(sz))
    return out


def fetch_emdb(emd, cfg):
    url = cfg["emdb_url"].format(EMD=emd, num=emd.split("-")[1])
    try:
        r = session().get(url, headers={"Accept": "application/json"}, timeout=(15, 60))
        if r.status_code != 200:
            return {"ok": False, "why": "HTTP %d" % r.status_code}
        info = parse_emdb_json(r.json())
        info["ok"] = bool(info["map_release"])
        if not info["ok"]:
            info["why"] = "no release date found in the JSON"
        return info
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": "%s: %s" % (type(e).__name__, str(e)[:120])}


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def finish_record(rec, spec):
    rec["passes_cheap_filters"] = rec.get("first_fail") is None and not rec.get("error")
    if rec.get("resolution_A") is not None and rec["passes_cheap_filters"]:
        rec["stratum"] = "le3.0" if rec["resolution_A"] <= 3.0 else "gt3.0_le4.0"
    if rec.get("emdb_id"):
        num = rec["emdb_id"].split("-")[1]
        rec["map_url"] = MAP_URL.format(EMD=rec["emdb_id"], num=num)
    rec["cif_url"] = "https://files.rcsb.org/download/%s.cif" % rec["pdb_id"]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})


def flow_table(records, n_all, n_selected, errors, prefilter_used=False):
    rows = [("all_cryoem", "Cryo-EM entries released in the window (RCSB search)", n_all, "")]
    if prefilter_used:
        rows.append(("server_prefilter", "Removed by the server-side prefilters (audited on a random sample)", n_selected, n_all - n_selected))
    remaining = n_selected - errors
    if errors:
        rows.append(("download_error", "Entries that could not be downloaded or parsed", remaining, errors))
    counts = Counter(r.get("first_fail") for r in records if not r.get("error"))
    for key, desc in FILTERS:
        remaining -= counts.get(key, 0)
        rows.append((key, desc, remaining, counts.get(key, 0)))
    return rows


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_canary(cfg):
    """Parse a few old, well-known single-particle entries with every filter switched off.
    Empty fields mean the parser does not understand the real files, so the run stops."""
    spec = dict(cfg["spec"])
    spec.update({"window_start": "0000-00-00", "freeze": "9999-99-99", "max_resolution": 99.0,
                 "max_modeled_residues": 10 ** 9})
    report, n_ok, n_unreachable, bad_any = [], 0, 0, False
    for pid in CANARIES:
        try:
            text = fetch_text(pid, cfg["cif_urls"])
        except Exception as e:  # noqa: BLE001
            n_unreachable += 1
            log("canary %s: could not download (%s)" % (pid, e))
            continue
        rec = parse_mmcif(text, pid, spec, force_full=True)
        line = ("%s: method=%s resolution=%s EMDB=%s entities=%s polymer_types=%s modeled_residues=%s unique_seq=%s released=%s"
                % (pid, rec["recon_method"], rec["resolution_A"], rec["emdb_ids_all"], rec["n_polymer_entities"],
                   rec["poly_types"], rec["n_modeled_residues"], rec["n_unique_seq"], rec["release_date_pdb"]))
        log("canary " + line)
        report.append(line)
        bad = []
        if not rec["recon_method"]:
            bad.append("reconstruction method not read")
        if rec["resolution_A"] is None:
            bad.append("resolution not read")
        if not rec["emdb_ids"]:
            bad.append("EMDB link not read")
        if not rec["n_modeled_residues"]:
            bad.append("modeled residues not counted")
        if not rec["entity_list"]:
            bad.append("no protein entity found")
        if not rec["release_date_pdb"]:
            bad.append("release date not read")
        if bad:
            bad_any = True
            log("CANARY FAILED for %s: %s" % (pid, "; ".join(bad)))
        else:
            n_ok += 1
    if n_unreachable == len(CANARIES):
        log("CANARY FAILED: none of the canary entries could be downloaded.")
        log("Check that Internet access is switched on for this notebook.")
        return False, report
    return (n_ok > 0 and not bad_any), report


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=SPEC["window_start"], help="first release date, inclusive")
    p.add_argument("--freeze", default=SPEC["freeze"], help="last release date, inclusive")
    p.add_argument("--out", default="pool_out", help="output folder")
    p.add_argument("--cache", default=None, help="cache folder (default: OUT/cache)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--emdb-threads", type=int, default=4)
    p.add_argument("--search-url", default=SEARCH_URL)
    p.add_argument("--cif-url", action="append", default=None, help="mmCIF URL template with {ID} or {id}; repeatable")
    p.add_argument("--emdb-url", default=EMDB_URL)
    p.add_argument("--no-canary", action="store_true")
    p.add_argument("--skip-emdb", action="store_true", help="skip the EMDB release-date check")
    p.add_argument("--limit", type=int, default=0, help="debug: process only the first N IDs")
    p.add_argument("--prefilter", action="store_true",
                   help="use server-side search filters to download fewer files (off by default: the audit of the "
                        "first real run found entries that they wrongly removed)")
    p.add_argument("--audit-n", type=int, default=100, help="with --prefilter: removed entries that are checked anyway")
    p.add_argument("--max-resolution", type=float, default=SPEC["max_resolution"])
    p.add_argument("--max-modeled-residues", type=int, default=SPEC["max_modeled_residues"])
    p.add_argument("--max-unique-sequences", type=int, default=SPEC["max_unique_sequences"])
    p.add_argument("--max-unk-fraction", type=float, default=SPEC["max_unk_fraction"])
    return p.parse_args(argv)


def main(argv=None):
    global _LOG
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    _LOG = open(os.path.join(args.out, "run.log"), "a", encoding="utf-8")
    try:
        return _run(args)
    finally:
        _LOG.close()
        _LOG = None


def _run(args):
    t0 = now_utc()
    out = args.out
    cache = args.cache or os.path.join(out, "cache")
    os.makedirs(os.path.join(cache, "parsed"), exist_ok=True)

    spec = dict(SPEC)
    spec.update({"window_start": args.start, "freeze": args.freeze, "max_resolution": args.max_resolution,
                 "max_modeled_residues": args.max_modeled_residues,
                 "max_unique_sequences": args.max_unique_sequences, "max_unk_fraction": args.max_unk_fraction})
    spec_is_locked = all(spec[k] == SPEC[k] for k in ("max_resolution", "max_modeled_residues",
                                                       "max_unique_sequences", "max_unk_fraction"))
    cfg = {"spec": spec, "thr": thresholds_key(spec), "cache": cache,
           "cif_urls": args.cif_url or CIF_URLS, "emdb_url": args.emdb_url}

    start = dt.date.fromisoformat(args.start)
    freeze = dt.date.fromisoformat(args.freeze)
    log("pull_pool %s | gemmi %s | python %s | window %s to %s inclusive" %
        (SCRIPT_VERSION, gemmi.__version__, platform.python_version(), start, freeze))
    if not spec_is_locked:
        log("WARNING: filter values differ from docs/SPEC.md. This run is a test, not the frozen pool.")
    today = now_utc().date()
    if today < freeze:
        log("Note: today (%s UTC) is before the freeze date. Entries released between now and %s are not in this pool." % (today, freeze))

    info = {"script_version": SCRIPT_VERSION, "parser_version": PARSER_VERSION, "started_utc": t0.isoformat(),
            "python": platform.python_version(), "platform": platform.platform(), "gemmi": gemmi.__version__,
            "requests": requests.__version__, "spec": spec, "spec_values_locked": spec_is_locked,
            "args": vars(args)}
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            info["script_sha256"] = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        pass

    # 1. canary
    if not args.no_canary:
        ok, report = run_canary(cfg)
        info["canary"] = report
        if not ok:
            log("Stopping: the parser could not read the canary entries. Send me this log.")
            with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, default=str)
            return 3
        log("canary OK")

    # 2. search
    windows = week_windows(start, freeze)
    try:
        cfg_found, n_window = calibrate_search(args.search_url, start, freeze)
        info["search_config"] = cfg_found
        log("search query form: method attribute %d, date format %d, request options %d (%d cryo-EM entries in the window)"
            % (cfg_found["method"], cfg_found["date"], cfg_found["ro"], n_window))
        base_by_week = [search_ids(args.search_url, a, b, []) for a, b in windows]
        n_base = [len(x) for x in base_by_week]
        if not any(n_base):
            raise RuntimeError("no cryo-EM entries found in any week; the search itself is not working")
        probe_i = max(range(len(windows)), key=lambda k: n_base[k])
        probe = windows[probe_i]
        if args.prefilter:
            kept, dropped = choose_prefilters(args.search_url, probe, n_base[probe_i])
        else:
            kept, dropped = [], []
            log("no server-side prefilters: every cryo-EM entry in the window is downloaded and judged on its file")
        extra = [OPTIONAL[k] for k in kept]
        info["search_prefilters_kept"] = kept
        info["search_prefilters_dropped"] = dropped
        info["example_query"] = build_query(probe[0], probe[1], extra)
        all_pairs, sel_pairs, weekly = [], [], []
        for (a, b), base in zip(windows, base_by_week):
            sel = search_ids(args.search_url, a, b, extra) if extra else base
            weekly.append((a.isoformat(), (b - dt.timedelta(days=1)).isoformat(), len(base), len(sel)))
            all_pairs.extend((i, a.isoformat()) for i in base)
            sel_pairs.extend((i, a.isoformat()) for i in sel)
    except (requests.RequestException, SearchRejected, RuntimeError) as e:
        log("Search failed: %s: %s" % (type(e).__name__, e))
        log("Check that Internet access is switched on for this notebook. If it is on, send me this log.")
        return 4
    all_em = sorted({i for i, _ in all_pairs})
    id_list = sorted({i for i, _ in sel_pairs})
    if len(all_em) != len(all_pairs):
        log("WARNING: %d duplicate IDs across weekly windows (removed)" % (len(all_pairs) - len(all_em)))
    if set(id_list) - set(all_em):
        log("WARNING: the prefiltered search returned IDs that the unfiltered search did not")
    removed = sorted(set(all_em) - set(id_list))
    week_of = {i: w for i, w in all_pairs}
    empty_weeks = [w[0] for w in weekly if w[2] == 0]
    if args.prefilter:
        log("search: %d cryo-EM entries in %d weekly windows; %d after the server-side prefilters"
            % (len(all_em), len(windows), len(id_list)))
    else:
        log("search: %d cryo-EM entries in %d weekly windows" % (len(all_em), len(windows)))
    if empty_weeks:
        log("note: no cryo-EM entries in the weeks starting %s" % ", ".join(empty_weeks))
    with open(os.path.join(out, "search_weeks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["window_start", "window_end_inclusive", "n_cryoem", "n_after_prefilters"])
        w.writerows(weekly)
    for name, ids in (("pool_ids_all_cryoem.txt", all_em), ("pool_ids.txt", id_list)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
    info["n_cryoem_in_window"] = len(all_em)
    info["n_after_prefilters"] = len(id_list)
    info["pool_ids_all_cryoem_sha256"] = sha256_text("\n".join(all_em) + "\n")
    info["pool_ids_sha256"] = sha256_text("\n".join(id_list) + "\n")
    if len(all_em) < 500 and spec_is_locked:
        log("WARNING: fewer than 500 cryo-EM entries found. Expect thousands for 2026. Check the log.")
    if args.limit:
        id_list = id_list[:args.limit]
        log("DEBUG: only the first %d IDs are processed" % args.limit)

    # 3. download and parse
    records, errors, n_cached = {}, [], 0
    done = 0
    t_dl = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        futs = {ex.submit(process_entry, i, cfg): i for i in id_list}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                rec, cached = fut.result()
                n_cached += 1 if cached else 0
                records[pid] = rec
            except Exception as e:  # noqa: BLE001
                err = blank_record(pid)
                err["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
                records[pid] = err
                errors.append(pid)
            done += 1
            if done % 100 == 0 or done == len(id_list):
                rate = done / max(1e-9, time.time() - t_dl)
                log("parsed %d/%d entries (%.1f/s, %d errors)" % (done, len(id_list), rate, len(errors)))
    info["n_cached"] = n_cached
    info["n_errors"] = len(errors)

    # prefilter audit: entries the server-side prefilters removed are sampled and checked anyway
    if removed and not args.limit and args.audit_n > 0:
        sample = sorted(random.Random(2026).sample(removed, min(args.audit_n, len(removed))))
        log("prefilter audit: checking %d of the %d entries removed by the prefilters" % (len(sample), len(removed)))
        would_pass, audit_errors = [], 0
        with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
            futs = {ex.submit(process_entry, i, cfg): i for i in sample}
            for fut in as_completed(futs):
                try:
                    rec, _ = fut.result()
                    if rec.get("first_fail") is None and not rec.get("error"):
                        would_pass.append(futs[fut])
                except Exception:  # noqa: BLE001
                    audit_errors += 1
        info["prefilter_audit"] = {"n_removed_by_prefilters": len(removed), "n_sampled": len(sample),
                                   "n_that_pass_the_file_filters": len(would_pass),
                                   "ids_that_pass": sorted(would_pass), "n_download_errors": audit_errors}
        if would_pass:
            log("WARNING: %d of %d sampled entries removed by the prefilters pass every file-based filter (%s). "
                "Run again without --prefilter." % (len(would_pass), len(sample), ", ".join(sorted(would_pass)[:10])))
        else:
            log("prefilter audit: none of the %d sampled entries would have passed (%d download errors)" % (len(sample), audit_errors))

    # release-date cross-check against the search week
    mism = 0
    for pid, rec in records.items():
        d = rec.get("release_date_pdb")
        wk = week_of.get(pid)
        if d and wk:
            wk_d = dt.date.fromisoformat(wk)
            if not (wk_d <= dt.date.fromisoformat(d) < wk_d + dt.timedelta(days=7)):
                mism += 1
    info["release_date_week_mismatches"] = mism
    if mism:
        log("note: %d entries have a first-revision date outside their search week" % mism)

    # 4. EMDB release check for the survivors of every earlier filter
    survivors = [r for r in records.values() if not r.get("error") and r.get("first_fail") is None]
    log("%d entries pass every file-based filter; checking their EMDB maps" % len(survivors))
    unverified = 0
    if not args.skip_emdb and survivors:
        emds = sorted({r["emdb_id"] for r in survivors})
        got = {}
        with ThreadPoolExecutor(max_workers=max(1, args.emdb_threads)) as ex:
            futs = {ex.submit(fetch_emdb, e, cfg): e for e in emds}
            for fut in as_completed(futs):
                got[futs[fut]] = fut.result()
        for r in survivors:
            g = got.get(r["emdb_id"], {"ok": False, "why": "not queried"})
            if g.get("ok"):
                r["emdb_map_release"] = g["map_release"]
                r["emdb_status"] = g.get("status") or ""
                r["map_size_kb"] = g.get("size_kb")
                r["emdb_release_verified"] = "yes" if g.get("release_basis") == "map_release" else "header_date_only"
                if first_failure(r, spec, ("emdb_map_released",)):
                    r["first_fail"] = "emdb_map_released"
            else:
                r["emdb_release_verified"] = "no: " + g.get("why", "")
                unverified += 1
    elif args.skip_emdb:
        for r in survivors:
            r["emdb_release_verified"] = "skipped"
            unverified += 1
    info["n_emdb_unverified"] = unverified
    if survivors and unverified > 0.05 * len(survivors):
        log("WARNING: the EMDB release date could not be checked for %d of %d candidates. Re-run later or check by hand." %
            (unverified, len(survivors)))

    # 5. write
    ordered = [records[i] for i in sorted(records)]
    for r in ordered:
        finish_record(r, spec)
    write_csv(os.path.join(out, "pool_all_entries.csv"), ordered)
    cands = [r for r in ordered if r["passes_cheap_filters"]]
    write_csv(os.path.join(out, "pool_candidates.csv"), cands)
    flow = flow_table(ordered, len(all_em), len(id_list), len(errors), prefilter_used=args.prefilter)
    with open(os.path.join(out, "pool_flow.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["step", "description", "n_remaining", "n_removed_at_this_step"])
        w.writerows(flow)
    with open(os.path.join(out, "pool_candidate_sequences.fasta"), "w", encoding="utf-8") as f:
        for r in cands:
            for e in r["entity_list"]:
                f.write(">%s_E%s len=%d copies=%d longest=%d\n%s\n" % (
                    r["pdb_id"], e["id"], e["len"], e["copies"], 1 if e["id"] == r["longest_entity"] else 0, e["seq"]))
    if errors:
        with open(os.path.join(out, "errors.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["pdb_id", "error"])
            for pid in sorted(errors):
                w.writerow([pid, records[pid]["error"]])
    strata = Counter(r["stratum"] for r in cands)
    info["n_candidates"] = len(cands)
    info["candidates_per_stratum"] = dict(strata)
    info["flow"] = [list(x) for x in flow]
    info["n_file_size_skipped"] = sum(1 for r in ordered if r.get("too_large"))
    if resource is not None:  # kilobytes on Linux
        info["peak_memory_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    info["finished_utc"] = now_utc().isoformat()
    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)

    log("---- flow ----")
    for step, desc, remaining, removed in flow:
        log("%-18s remaining %6s  removed %6s  %s" % (step, remaining, removed, desc))
    log("candidates: %d (<=3.0 A: %d, 3.0-4.0 A: %d)" % (len(cands), strata.get("le3.0", 0), strata.get("gt3.0_le4.0", 0)))
    log("files not parsed because of their size: %d | peak memory: %s MB" % (info["n_file_size_skipped"], info.get("peak_memory_mb", "n/a")))
    log("pool_ids_all_cryoem.txt sha256: %s" % info["pool_ids_all_cryoem_sha256"])
    log("pool_ids.txt sha256: %s" % info["pool_ids_sha256"])
    if errors:
        log("%d entries failed to download or parse (see errors.csv). Run the same command again to retry only those." % len(errors))
        return 2
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
