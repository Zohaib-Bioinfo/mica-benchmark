#!/usr/bin/env python3
"""Phase 3c: how close is each selected entry to proteins that were in the PDB before 2026? (docs/SPEC.md decision 9)

Every unique chain sequence of every selected entry is searched with MMseqs2 against all protein chains of PDB entries
released before 2026-01-01. The entry's value is the highest sequence identity found over its chains. Two numbers are
kept, because identity over a short fragment says little about the whole chain:

  max_id_to_pre2026_pdb   highest identity among alignments that cover at least 50% of the query chain (0 if none).
                          This is the number of the spec; the sensitivity analysis keeps the entries at or below 0.30.
  max_id_any_coverage     highest identity among all alignments, whatever they cover.

Identity is the fraction of identical residues in the alignment (0 to 1). The reference is built from the wwPDB file
pdb_seqres.txt.gz minus every entry that the RCSB search lists as released on or after 2026-01-01 (all methods),
duplicates removed. Checks that stop the run: the pool's cryo-EM entries and the selected entries must all be in the
excluded list, and the reference must not be small.

    python novelty_check.py --entries data/entries_2026.csv --fasta data/pool/pool_candidate_sequences.fasta \\
        --cryoem-ids data/pool/pool_ids_all_cryoem.txt --out novelty_out
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
import shutil
import subprocess
import sys
import tempfile
import time

import requests

SCRIPT_VERSION = "1.0"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
SEQRES_URL = "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
CUTOFF = dt.date(2026, 1, 1)
MIN_QCOV = 0.5
MIN_REFERENCE_CHAINS = 100_000
M8_FIELDS = "query,target,fident,qcov,tcov,alnlen,evalue,bits"
OUT_COLUMNS = ["pdb_id", "n_chain_seqs", "max_id_to_pre2026_pdb", "best_hit", "best_hit_query", "best_hit_qcov",
               "max_id_any_coverage", "best_any_hit", "best_any_hit_query", "best_any_hit_qcov"]
UA = "mica-benchmark-2026 (academic benchmark; contact via GitHub Zohaib-Bioinfo)"


# --------------------------------------------------------------------------------------
# RCSB: entries released from 2026 on
# --------------------------------------------------------------------------------------
def terminal(attribute, operator, value):
    return {"type": "terminal", "service": "text", "parameters": {"attribute": attribute, "operator": operator, "value": value}}


def post_window(url, a, b, retries=4):
    body = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        terminal("rcsb_accession_info.initial_release_date", "greater_or_equal", a.isoformat()),
        terminal("rcsb_accession_info.initial_release_date", "less", b.isoformat())]},
        "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": 1000}, "results_content_type": ["experimental"]}}
    last = ""
    for attempt in range(retries):
        try:
            r = requests.post(url, json=body, timeout=(15, 120), headers={"User-Agent": UA})
            if r.status_code == 204:
                return [], 0
            if r.status_code == 400:
                raise RuntimeError("the RCSB search did not accept the query: %s" % r.text[:300])
            r.raise_for_status()
            js = r.json()
            ids = [x["identifier"] if isinstance(x, dict) else str(x) for x in js.get("result_set", [])]
            return ids, js.get("total_count", len(ids))
        except requests.RequestException as e:
            last = str(e)[:200]
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("the RCSB search failed after %d attempts: %s" % (retries, last))


def window_ids(url, a, b):
    ids, total = post_window(url, a, b)
    if total > len(ids):
        if (b - a).days <= 1:
            raise RuntimeError("search truncated for the single day %s: %d of %d hits" % (a, len(ids), total))
        mid = a + (b - a) // 2
        return window_ids(url, a, mid) + window_ids(url, mid, b)
    return ids


def released_since(url, start, through):
    """Every PDB entry (all methods) with an initial release date in [start, through)."""
    out, cur = set(), start
    while cur < through:
        nxt = min(cur + dt.timedelta(days=7), through)
        out.update(i.upper() for i in window_ids(url, cur, nxt))
        cur = nxt
    return out


# --------------------------------------------------------------------------------------
# The reference: pdb_seqres without the 2026 entries
# --------------------------------------------------------------------------------------
def download(url, dest, retries=3):
    last = ""
    for attempt in range(retries):
        part = dest + ".part"
        try:
            with requests.get(url, stream=True, timeout=(15, 120), headers={"User-Agent": UA}) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            os.replace(part, dest)
            return
        except (requests.RequestException, OSError) as e:
            last = str(e)[:200]
            time.sleep(3 * (attempt + 1))
        finally:
            if os.path.exists(part):
                os.remove(part)
    raise RuntimeError("download of %s failed after %d attempts: %s" % (url, retries, last))


def build_reference(seqres_gz, exclude, out_fasta):
    """Writes the protein chains of entries not in `exclude`, each distinct sequence once. Returns the counts."""
    stats = {"protein_chains_in_seqres": 0, "chains_of_excluded_entries": 0, "reference_unique_sequences": 0}
    seen = set()
    header = None
    with gzip.open(seqres_gz, "rt", encoding="utf-8", errors="replace") as f, open(out_fasta, "w", encoding="utf-8") as out:
        for line in f:
            if line.startswith(">"):
                header = line[1:].split()
                continue
            if header is None or len(header) < 2 or header[1] != "mol:protein":
                header = None
                continue
            seq = line.strip()
            name, header = header[0], None
            pid = name.split("_")[0].upper()
            stats["protein_chains_in_seqres"] += 1
            if pid in exclude:
                stats["chains_of_excluded_entries"] += 1
                continue
            if seq in seen:
                continue
            seen.add(seq)
            out.write(">%s\n%s\n" % (name, seq))
    stats["reference_unique_sequences"] = len(seen)
    return stats


# --------------------------------------------------------------------------------------
# Queries and search
# --------------------------------------------------------------------------------------
def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def query_sequences(all_seqs, pdb_ids):
    """{entity name: sequence} of every entity of the given entries; each distinct sequence of an entry once."""
    out, missing = {}, []
    for pid in pdb_ids:
        mine = {n: s for n, s in all_seqs.items() if n.split("_E")[0] == pid}
        if not mine:
            missing.append(pid)
            continue
        seen = set()
        for n in sorted(mine):
            if mine[n] not in seen:
                seen.add(mine[n])
                out[n] = mine[n]
    if missing:
        raise ValueError("no sequences for %s in the FASTA file" % ", ".join(missing))
    return out


def run_search(queries, ref_fasta, binary, threads, workdir):
    qfa = os.path.join(workdir, "queries.fasta")
    with open(qfa, "w", encoding="utf-8") as f:
        for n in sorted(queries):
            f.write(">%s\n%s\n" % (n, queries[n]))
    m8 = os.path.join(workdir, "hits.m8")
    tmp = os.path.join(workdir, "mmseqs_tmp")
    cmd = [binary, "easy-search", qfa, ref_fasta, m8, tmp, "-s", "7.5", "-e", "1e-3", "--max-seqs", "2000",
           "--format-output", M8_FIELDS, "--threads", str(threads), "-v", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("mmseqs failed (exit %d):\n%s\n%s" % (r.returncode, r.stdout[-1500:], r.stderr[-1500:]))
    hits = {}
    with open(m8, encoding="utf-8") as f:
        for line in f:
            q, t, fid, qcov, tcov, alnlen, ev, bits = line.rstrip("\n").split("\t")[:8]
            if "_" in t:
                t = t.split("_", 1)[0].upper() + "_" + t.split("_", 1)[1]          # 101m_A -> 101M_A
            hits.setdefault(q, []).append({"target": t, "id": float(fid), "qcov": float(qcov), "alnlen": int(alnlen), "evalue": float(ev)})
    return cmd, hits


def entry_values(pdb_ids, queries, hits):
    rows = []
    for pid in pdb_ids:
        names = [n for n in queries if n.split("_E")[0] == pid]
        best, best_any = (0.0, "", "", 0.0), (0.0, "", "", 0.0)
        for n in names:
            for h in hits.get(n, []):
                cand = (h["id"], h["target"], n, h["qcov"])
                if cand[0] > best_any[0]:
                    best_any = cand
                if h["qcov"] >= MIN_QCOV and cand[0] > best[0]:
                    best = cand
        rows.append({"pdb_id": pid, "n_chain_seqs": len(names),
                     "max_id_to_pre2026_pdb": "%.3f" % best[0], "best_hit": best[1], "best_hit_query": best[2],
                     "best_hit_qcov": "%.2f" % best[3] if best[1] else "",
                     "max_id_any_coverage": "%.3f" % best_any[0], "best_any_hit": best_any[1], "best_any_hit_query": best_any[2],
                     "best_any_hit_qcov": "%.2f" % best_any[3] if best_any[1] else ""})
    return rows


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def update_entries(path, values):
    rows = read_csv(path)
    with open(path, newline="", encoding="utf-8") as f:
        columns = next(csv.reader(f))
    if "max_id_to_pre2026_pdb" not in columns:
        columns.append("max_id_to_pre2026_pdb")
    for r in rows:
        r["max_id_to_pre2026_pdb"] = values[r["pdb_id"]]
    tmp = path + ".tmp"
    write_csv(tmp, columns, rows)
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entries", required=True, help="entries_2026.csv (its max_id_to_pre2026_pdb column is filled in)")
    p.add_argument("--fasta", required=True, help="pool_candidate_sequences.fasta")
    p.add_argument("--out", required=True)
    p.add_argument("--cryoem-ids", default=None, help="pool_ids_all_cryoem.txt: all of them must be in the excluded list")
    p.add_argument("--mmseqs", default="mmseqs")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--through", default=None, help="last release date to exclude, default: a week from today")
    p.add_argument("--seqres", default=None, help="a local pdb_seqres.txt.gz (default: download)")
    p.add_argument("--search-url", default=SEARCH_URL)
    p.add_argument("--seqres-url", default=SEQRES_URL)
    p.add_argument("--min-reference-chains", type=int, default=MIN_REFERENCE_CHAINS)
    args = p.parse_args(argv)
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


def run(args):
    if shutil.which(args.mmseqs) is None and not os.path.exists(args.mmseqs):
        raise RuntimeError("mmseqs not found (%s)" % args.mmseqs)
    os.makedirs(args.out, exist_ok=True)
    entries = read_csv(args.entries)
    pdb_ids = [r["pdb_id"] for r in entries]
    if not pdb_ids:
        raise ValueError("%s has no entries" % args.entries)
    queries = query_sequences(read_fasta(args.fasta), pdb_ids)
    print("%d entries, %d distinct chain sequences to search" % (len(pdb_ids), len(queries)), flush=True)

    through = dt.date.fromisoformat(args.through) if args.through else dt.date.today() + dt.timedelta(days=7)
    exclude = released_since(args.search_url, CUTOFF, through)
    print("RCSB: %d entries released from %s to %s (all methods): they are left out of the reference" % (len(exclude), CUTOFF, through), flush=True)
    if args.cryoem_ids:
        with open(args.cryoem_ids, encoding="utf-8") as f:
            cryo = {ln.strip().upper() for ln in f if ln.strip()}
        lost = sorted(cryo - exclude)
        if lost:
            raise RuntimeError("%d cryo-EM entries of the pool are missing from the excluded list (e.g. %s); the all-methods search "
                               "is incomplete" % (len(lost), ", ".join(lost[:5])))
    still = sorted(set(pdb_ids) - exclude)
    if still:
        raise RuntimeError("selected entries are not in the excluded list: %s" % ", ".join(still[:5]))

    seqres = args.seqres
    if not seqres:
        seqres = os.path.join(args.out, "pdb_seqres.txt.gz")
        print("downloading pdb_seqres.txt.gz ...", flush=True)
        download(args.seqres_url, seqres)
    ref = os.path.join(args.out, "reference_pre2026.fasta")
    stats = build_reference(seqres, exclude, ref)
    print("reference: %(reference_unique_sequences)d distinct protein sequences (%(protein_chains_in_seqres)d protein chains in "
          "pdb_seqres, %(chains_of_excluded_entries)d of them from 2026 entries)" % stats, flush=True)
    if stats["reference_unique_sequences"] < args.min_reference_chains:
        raise RuntimeError("the reference has only %d distinct sequences (expected at least %d); is pdb_seqres complete?"
                           % (stats["reference_unique_sequences"], args.min_reference_chains))

    work = tempfile.mkdtemp(prefix="novelty_")
    try:
        cmd, hits = run_search(queries, ref, args.mmseqs, args.threads, work)
        r = subprocess.run([args.mmseqs, "version"], capture_output=True, text=True)
        version = (r.stdout or r.stderr).strip()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rows = entry_values(pdb_ids, queries, hits)
    write_csv(os.path.join(args.out, "novelty.csv"), OUT_COLUMNS, rows)
    update_entries(args.entries, {r["pdb_id"]: r["max_id_to_pre2026_pdb"] for r in rows})
    info = {"script_version": SCRIPT_VERSION, "mmseqs_version": version, "mmseqs_command": [c if not c.startswith(tempfile.gettempdir()) else "<tmp>/" + os.path.basename(c) for c in cmd],
            "min_query_coverage": MIN_QCOV, "cutoff_date": CUTOFF.isoformat(), "excluded_through": through.isoformat(),
            "n_excluded_entries": len(exclude), "excluded_ids_sha256": hashlib.sha256("\n".join(sorted(exclude)).encode()).hexdigest(),
            "seqres_sha256": sha256_file(seqres), "reference_fasta_sha256": sha256_file(ref), "n_queries": len(queries),
            "n_entries": len(pdb_ids), "entries_at_or_below_0.30": sum(1 for r in rows if float(r["max_id_to_pre2026_pdb"]) <= 0.30),
            "python": platform.python_version(), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **stats}
    with open(os.path.join(args.out, "novelty_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    if not args.seqres:
        os.remove(seqres)                  # 270 MB; its hash is in novelty_info.json
    os.remove(ref)                         # large; rebuilt from pdb_seqres and the excluded list
    vals = sorted(float(r["max_id_to_pre2026_pdb"]) for r in rows)
    print("novelty: %d entries; highest identity to a pre-2026 PDB chain (>=50%% of the chain covered): min %.2f, median %.2f, max %.2f; "
          "%d entries at or below 0.30" % (len(rows), vals[0], vals[len(vals) // 2], vals[-1], info["entries_at_or_below_0.30"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
