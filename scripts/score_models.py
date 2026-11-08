#!/usr/bin/env python3
"""Scores models against deposited models: protocol v1.0 of LOG.md (2026-10-24, "Scoring protocol").

Every model (the prediction) is compared with the deposited model of its entry (the native) in the same way for every method:

* US-align `-mm 1 -ter 1 -mol protein` (prediction = structure 1): tm_score (TM2, normalised by the deposited model), aligned_ca_length,
  ca_rmsd, seq_identity; tm_score_pred_norm (TM1) for information. The rotation matrix of this run puts the prediction into the frame of the deposited model.
* US-align `-mm 5` on one chain per structure (all chains of the prediction concatenated, all chains of the deposited model concatenated): the fragment-agnostic
  TM-score (tm_score_fns, ...). Fragment order, chain breaks and extra copies do not matter.
* `phenix.chain_comparison <native> <prediction>` on the prediction superposed by the matrix above (columns without a suffix) and on the prediction in its own
  frame (suffix _ff): ca_match = 100 * CLOSE / (CLOSE + FAR), ca_found = % FOUND, seq_match = SEQ MATCH(%), ca_rmsd_close; coverage = residues of the prediction /
  residues of the deposited model; ca_quality = ca_match * coverage.
* The same counts from a nearest-neighbour search (own_*): the fallback when Phenix cannot be used, and a check of the parsing.

All files that are handed to US-align (mm 5) and to Phenix are C-alpha traces written by this module (protein residues only, first model, first alternative
location), so that the formats and the extra atoms of the different methods cannot matter. The original file of the prediction is read by US-align for the
`-mm 1` run, as in the paper.

    python scripts/score_models.py one --usalign USalign --native 26DK.cif --pred model.pdb [--phenix-env env.sh] --workdir out/
    python scripts/score_models.py batch --usalign USalign --roots result1/ result2/ --entries data/entries_2026.csv --natives natives/ --out scores/ [--phenix-env env.sh]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROTOCOL_VERSION = "1.0"
CUTOFF = 3.0                 # phenix.chain_comparison max_dist (its default) and the cut-off of the own counts
USALIGN_TIMEOUT = 900
PHENIX_TIMEOUT = 900
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PRIMARY_KINDS = {"modelangelo", "e3_prealign", "emprot", "emodelx_af_all_atom", "mica_all_atom", "cryoatom1", "cryozeta"}
METHOD_OF_KIND = {"modelangelo": "modelangelo", "modelangelo_raw": "modelangelo", "e3_prealign": "e3", "emprot": "emprot", "emprot_denovo": "emprot", "emprot_fit": "emprot",
                  "emodelx_af_all_atom": "emodelx", "emodelx_af_ca": "emodelx", "cryoatom1": "cryoatom1", "cryoatom1_raw": "cryoatom1", "cryozeta": "cryozeta"}
MODEL_SUFFIXES = (".pdb", ".cif", ".ent", ".mmcif")
CC_COLUMNS = ["rmsd", "close_n", "far_n", "forward", "reverse", "mixed", "found_pct", "ca_score", "seq_match_pct", "seq_score", "mean_length", "fragments", "bad_connections"]

SCORE_COLUMNS = (["entry", "method", "model_kind", "primary", "model_file", "source", "status", "note", "pred_residues", "native_residues", "coverage",
                  "n_chains_pred", "n_chains_native", "n_chains_paired", "longest_chain_pred", "us_residues_pred",
                  "tm_score", "tm_score_pred_norm", "ca_rmsd", "aligned_ca_length", "seq_identity",
                  "tm_score_fns", "ca_rmsd_fns", "aligned_ca_length_fns", "seq_identity_fns",
                  "cc_status", "ca_match", "ca_found", "seq_match", "ca_rmsd_close", "ca_quality", "cc_close_n", "cc_far_n",
                  "cc_status_ff", "ca_match_ff", "ca_found_ff", "seq_match_ff", "ca_rmsd_close_ff", "ca_quality_ff",
                  "own_ca_match", "own_ca_found", "own_seq_match", "own_ca_match_ff", "own_ca_found_ff", "own_seq_match_ff", "seconds"])


# ---------------------------------------------------------------------------------------------------------------------------------- C-alpha traces

class Trace:
    """The C-alpha trace of a structure: a list of chains, each with its name, its residues (name, number, insertion code) and an (n, 3) array of coordinates."""

    def __init__(self, chains=None):
        self.chains = chains or []

    @property
    def n_residues(self):
        return sum(len(c["res"]) for c in self.chains)

    @property
    def n_chains(self):
        return len(self.chains)

    def xyz(self):
        return np.concatenate([c["xyz"] for c in self.chains]) if self.chains else np.zeros((0, 3))

    def names(self):
        return [r[0] for c in self.chains for r in c["res"]]

    def transformed(self, matrix):
        """x' = t + U x for every residue; `matrix` is the (t, U) of parse_matrix."""
        t, u = matrix
        return Trace([{"name": c["name"], "res": list(c["res"]), "xyz": c["xyz"] @ u.T + t} for c in self.chains])

    def merged(self):
        """All residues in one chain A, numbered 1 to n (the order of the file)."""
        names = self.names()
        if not names:
            return Trace([])
        return Trace([{"name": "A", "res": [(n, i + 1, " ") for i, n in enumerate(names)], "xyz": self.xyz().copy()}])


def is_amino_acid(name):
    import gemmi
    name = name.strip().upper()
    info = gemmi.find_tabulated_residue(name)
    if info is not None:
        return bool(info.is_amino_acid()) or name == "UNK"
    return name == "UNK"


def read_trace(path):
    """The C-alpha trace of the first model of a pdb or mmCIF file (gemmi): amino-acid residues that have a CA atom, first alternative location."""
    import gemmi
    st = gemmi.read_structure(path)
    if len(st) == 0:
        raise ValueError("the file has no model")
    chains = []
    for ch in st[0]:
        res, xyz = [], []
        for r in ch:
            if not is_amino_acid(r.name):
                continue
            a = r.find_atom("CA", "*")
            if a is None:
                continue
            icode = (r.seqid.icode or " ").strip() or " "
            res.append((r.name.strip(), int(r.seqid.num), icode))
            xyz.append((a.pos.x, a.pos.y, a.pos.z))
        if res:
            chains.append({"name": ch.name, "res": res, "xyz": np.array(xyz, dtype=float)})
    return Trace(chains)


def chain_labels(trace):
    names = [c["name"] for c in trace.chains]
    if all(len(n) == 1 for n in names) and len(set(names)) == len(names):
        return names
    return [CHAIN_IDS[i % len(CHAIN_IDS)] for i in range(len(names))]


def write_pdb(trace, path):
    """A PDB file with the C-alpha atoms of the trace. Chain names are kept if they are single characters and unique, otherwise they are A, B, C, ... in order."""
    labels = chain_labels(trace)
    lines, serial = [], 0
    for lab, c in zip(labels, trace.chains):
        for (rn, num, ic), (x, y, z) in zip(c["res"], c["xyz"]):
            if not -999 <= num <= 9999:
                raise ValueError("residue number %d does not fit the PDB format" % num)
            serial += 1
            lines.append("ATOM  %5d  CA  %3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s" % (serial % 100000, rn[:3], lab, num, ic, x, y, z, 1.0, 20.0, "C"))
        lines.append("TER")
    lines.append("END")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------------------------------------------------------------- US-align

TABLE_COLUMNS = ["PDBchain1", "PDBchain2", "TM1", "TM2", "RMSD", "ID1", "ID2", "IDali", "L1", "L2", "Lali"]


def parse_usalign_tabular(text):
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != len(TABLE_COLUMNS):
            raise ValueError("a US-align line has %d fields, %d expected: %r" % (len(parts), len(TABLE_COLUMNS), line[:200]))
        row = dict(zip(TABLE_COLUMNS, parts))
        for k in ("TM1", "TM2", "RMSD", "ID1", "ID2", "IDali"):
            row[k] = float(row[k])
        for k in ("L1", "L2", "Lali"):
            row[k] = int(row[k])
        rows.append(row)
    return rows


def parse_matrix(text):
    """(t, U) of the rotation matrix file of US-align (`-m`): X = t + U x rotates structure 1 onto structure 2."""
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*([012])\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$", line)
        if m:
            rows.append([float(m.group(i)) for i in (2, 3, 4, 5)])
    if len(rows) != 3:
        raise ValueError("no rotation matrix in the US-align matrix file")
    a = np.array(rows)
    return a[:, 0].copy(), a[:, 1:].copy()


def chain_pairs(row):
    """Chain pairs of a `-mm 1` row: the i-th chain of structure 1 is paired with the i-th chain of structure 2 (empty names are unpaired chains)."""
    c1 = row["PDBchain1"].split(":")[1:]
    c2 = row["PDBchain2"].split(":")[1:]
    return [(a, b) for a, b in zip(c1, c2) if a and b]


def run_usalign(usalign, pred, native, mm, ter, workdir, tag, timeout=USALIGN_TIMEOUT):
    """One US-align run; returns the table row (a dict) with the matrix under "matrix"; raises RuntimeError with the reason."""
    matrix = os.path.join(workdir, tag + "_matrix.txt")
    if os.path.exists(matrix):
        os.remove(matrix)
    cmd = [usalign, pred, native, "-mol", "protein", "-mm", str(mm), "-ter", str(ter), "-outfmt", "2", "-m", matrix]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
    except subprocess.TimeoutExpired:
        raise RuntimeError("US-align (-mm %d) did not finish within %d seconds" % (mm, timeout))
    except OSError as e:
        raise RuntimeError("US-align could not be started: %s" % e)
    with open(os.path.join(workdir, tag + "_usalign.txt"), "w", encoding="utf-8") as f:
        f.write("# %s\n%s%s" % (" ".join(cmd), r.stdout, ("\n" + r.stderr) if r.stderr else ""))
    if r.returncode != 0:
        raise RuntimeError("US-align (-mm %d) stopped with exit code %d: %s" % (mm, r.returncode, (r.stderr or r.stdout).strip()[-300:]))
    try:
        rows = parse_usalign_tabular(r.stdout)
    except ValueError as e:
        raise RuntimeError("US-align (-mm %d): %s" % (mm, e))
    if len(rows) != 1:
        raise RuntimeError("US-align (-mm %d) gave %d result lines, 1 expected. Output: %s" % (mm, len(rows), r.stdout.strip()[-300:]))
    row = rows[0]
    try:
        with open(matrix, encoding="utf-8", errors="replace") as f:
            row["matrix"] = parse_matrix(f.read())
    except (OSError, ValueError):
        row["matrix"] = None
    return row


# ---------------------------------------------------------------------------------------------------------------------------------- phenix.chain_comparison

_ROW = re.compile(r"^\s*([A-Za-z_][^\s]*)\s+((?:[-+]?\d+(?:\.\d+)?\s+){12}[-+]?\d+(?:\.\d+)?)\s*$")


def parse_chain_comparison(text):
    """The result row of phenix.chain_comparison (label and 13 numbers: RMSD, CLOSE N, FAR N, FORWARD, REVERSE, MIXED, % FOUND, CA SCORE, SEQ MATCH(%), SEQ SCORE,
    MEAN LENGTH, FRAGMENTS, BAD CONNECTIONS) as a dict, with the residues of the target and of the query that it used."""
    rows = [m for m in (_ROW.match(line) for line in text.splitlines()) if m]
    if not rows:
        raise ValueError("no result row in the output of phenix.chain_comparison")
    d = dict(zip(CC_COLUMNS, [float(x) for x in rows[0].group(2).split()]))
    d["label"] = rows[0].group(1)
    d["n_rows"] = len(rows)
    used = re.search(r"Used residues in target:\s*(\d+)\s+In query:\s*(\d+)", text)
    d["used_target"], d["used_query"] = (int(used.group(1)), int(used.group(2))) if used else (None, None)
    return d


def run_chain_comparison(phenix_env, target, query, out_txt, timeout=PHENIX_TIMEOUT):
    """phenix.chain_comparison target query (the target comes first, as in the Phenix documentation); the whole output is kept in out_txt."""
    cmd = ["bash", "-c", 'source "$0" && phenix.chain_comparison "$1" "$2"', phenix_env, target, query]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace", cwd=os.path.dirname(os.path.abspath(out_txt)))
    except subprocess.TimeoutExpired:
        raise RuntimeError("phenix.chain_comparison did not finish within %d seconds" % timeout)
    text = r.stdout + (("\n" + r.stderr) if r.stderr else "")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# phenix.chain_comparison %s %s\n%s" % (target, query, text))
    if r.returncode != 0:
        raise RuntimeError("phenix.chain_comparison stopped with exit code %d: %s" % (r.returncode, text.strip()[-300:]))
    try:
        return parse_chain_comparison(text)
    except ValueError as e:
        raise RuntimeError("%s: %s" % (e, text.strip()[-300:]))


def cc_metrics(d, coverage):
    """ca_match (percentage of the query residues within the cut-off of a deposited residue), ca_found, seq_match, rmsd of the close residues, ca_quality."""
    close, far = d["close_n"], d["far_n"]
    ca_match = 100.0 * close / (close + far) if close + far > 0 else 0.0
    out = {"ca_match": round(ca_match, 3), "ca_found": d["found_pct"], "seq_match": d["seq_match_pct"], "ca_rmsd_close": d["rmsd"],
           "cc_close_n": int(close), "cc_far_n": int(far)}
    out["ca_quality"] = round(ca_match * coverage, 3) if coverage != "" else ""
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- own counts

def own_stats(query, target, cutoff=CUTOFF):
    """The counts of chain_comparison from a nearest-neighbour search: the share of the query residues within the cut-off of a target residue (ca_match), the share of the
    target residues within the cut-off of a query residue (ca_found) and the share of the close query residues whose residue name equals that of the nearest target residue."""
    from scipy.spatial import cKDTree
    q, t = query.xyz(), target.xyz()
    if len(q) == 0 or len(t) == 0:
        return {"ca_match": 0.0, "ca_found": 0.0, "seq_match": 0.0}
    d, idx = cKDTree(t).query(q)
    close = d <= cutoff
    dq, _ = cKDTree(q).query(t)
    qn, tn = query.names(), target.names()
    same = sum(1 for i in np.nonzero(close)[0] if qn[i] == tn[idx[i]])
    n_close = int(close.sum())
    return {"ca_match": round(100.0 * n_close / len(q), 3), "ca_found": round(100.0 * float((dq <= cutoff).sum()) / len(t), 3),
            "seq_match": round(100.0 * same / n_close, 3) if n_close else 0.0}


# ---------------------------------------------------------------------------------------------------------------------------------- one model

def _num(x, digits=4):
    return "" if x is None or x == "" else round(float(x), digits)


def score_model(usalign, phenix_env, native, pred_path, workdir, native_file=None):
    """All numbers of one model. `native` is the Trace of the deposited model and `native_file` its C-alpha file (written if it is None).
    Returns a dict with the keys of SCORE_COLUMNS that this function fills; the raw outputs of the programs are in workdir."""
    t0 = time.time()
    os.makedirs(workdir, exist_ok=True)
    out = {k: "" for k in SCORE_COLUMNS}
    out.update(status="ok", native_residues=native.n_residues, n_chains_native=native.n_chains)
    try:
        pred = read_trace(pred_path)
    except Exception as e:
        out.update(status="no_model", note=("the file cannot be read: %s" % e)[:300], seconds=round(time.time() - t0, 1))
        return out
    if pred.n_residues == 0:
        out.update(status="no_model", note="the file has no amino-acid residue with a CA atom", pred_residues=0, n_chains_pred=0, seconds=round(time.time() - t0, 1))
        return out
    if native_file is None:
        native_file = write_pdb(native, os.path.join(workdir, "native_ca.pdb"))
    coverage = round(pred.n_residues / float(native.n_residues), 4)
    out.update(pred_residues=pred.n_residues, coverage=coverage, n_chains_pred=pred.n_chains, longest_chain_pred=max(len(c["res"]) for c in pred.chains))
    notes = []
    # 1. US-align, multi-chain, as in the paper (the original file of the prediction)
    matrix = None
    try:
        row = run_usalign(usalign, pred_path, native_file, 1, 1, workdir, "mm1")
        out.update(tm_score=_num(row["TM2"]), tm_score_pred_norm=_num(row["TM1"]), ca_rmsd=_num(row["RMSD"], 2), aligned_ca_length=row["Lali"], seq_identity=_num(row["IDali"]),
                   n_chains_paired=len(chain_pairs(row)), us_residues_pred=row["L1"])
        matrix = row["matrix"]
        if matrix is None:
            notes.append("no rotation matrix: the superposed Phenix variant was not run")
        if row["L1"] != pred.n_residues:
            notes.append("US-align read %d residues, the trace has %d" % (row["L1"], pred.n_residues))
    except RuntimeError as e:
        out.update(status="no_alignment")
        notes.append(str(e))
    # 2. US-align, one chain per structure, fully non-sequential (fragment-agnostic)
    try:
        pm = write_pdb(pred.merged(), os.path.join(workdir, "pred_merged.pdb"))
        nm = write_pdb(native.merged(), os.path.join(workdir, "native_merged.pdb"))
        row = run_usalign(usalign, pm, nm, 5, 2, workdir, "fns")
        out.update(tm_score_fns=_num(row["TM2"]), ca_rmsd_fns=_num(row["RMSD"], 2), aligned_ca_length_fns=row["Lali"], seq_identity_fns=_num(row["IDali"]))
    except (RuntimeError, ValueError) as e:
        notes.append("fns: %s" % e)
    # 3. Chain comparison: superposed (matrix of the -mm 1 run) and in the own frame of the prediction
    variants = [("", pred.transformed(matrix) if matrix is not None else None, "superposed"), ("_ff", pred, "frame_fixed")]
    for suffix, trace, tag in variants:
        if trace is None:
            continue
        try:
            qfile = write_pdb(trace, os.path.join(workdir, "query_%s.pdb" % tag))
        except ValueError as e:
            notes.append("%s: %s" % (tag, e))
            continue
        own = own_stats(trace, native)
        out.update({"own_ca_match" + suffix: own["ca_match"], "own_ca_found" + suffix: own["ca_found"], "own_seq_match" + suffix: own["seq_match"]})
        if not phenix_env:
            out["cc_status" + suffix] = "not run"
            continue
        try:
            d = run_chain_comparison(phenix_env, native_file, qfile, os.path.join(workdir, "chain_comparison_%s.txt" % tag))
            m = cc_metrics(d, coverage)
            out.update({"ca_match" + suffix: m["ca_match"], "ca_found" + suffix: m["ca_found"], "seq_match" + suffix: m["seq_match"],
                        "ca_rmsd_close" + suffix: m["ca_rmsd_close"], "ca_quality" + suffix: m["ca_quality"], "cc_status" + suffix: "ok"})
            if suffix == "":
                out.update(cc_close_n=m["cc_close_n"], cc_far_n=m["cc_far_n"])
            if d["used_query"] is not None and d["used_query"] != pred.n_residues:
                notes.append("%s: Phenix used %d of %d residues of the prediction" % (tag, d["used_query"], pred.n_residues))
        except RuntimeError as e:
            out["cc_status" + suffix] = "failed"
            notes.append("%s: %s" % (tag, str(e)[:200]))
    out["note"] = " | ".join(n.replace("\n", " ") for n in notes)[:600]
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- many models

def model_kind(method, filename):
    n = os.path.basename(filename)
    low = n.lower()
    if method == "mica":
        if "real_space_refined" in low:
            return "mica_refined"
        if low.endswith("_all_atom_model.pdb"):
            return "mica_all_atom"
        if "(init)" in low:
            return "mica_init_ca"
        if low.endswith("_ca_model.pdb"):
            return "mica_ca"
        return "mica_other"
    if method == "modelangelo":
        return "modelangelo_raw" if "_raw" in low else "modelangelo"
    if method == "e3":
        return "e3_prealign" if "prealign" in low else "e3_other"
    if method == "emprot":
        if "denovo" in low:
            return "emprot_denovo"
        if "fit" in low:
            return "emprot_fit"
        return "emprot"
    if method == "emodelx":
        if low.endswith("_all_atom_model.pdb"):
            return "emodelx_af_all_atom"
        if low.endswith("_ca_model.pdb"):
            return "emodelx_af_ca"
        return "emodelx_other"
    if method == "cryoatom1":                                   # added 2026-11-01 (Prof. Cheng's request): <ENTRY>_cryoatom1.cif is the model, <ENTRY>_cryoatom1_raw.cif the one before pruning
        return "cryoatom1_raw" if low.endswith("_raw.cif") else "cryoatom1"
    if method == "cryozeta":                                    # CryoZeta-Final/<ENTRY>_sample_<i>.cif, ranked best first by the program: sample 0 is the model of the benchmark
        m = re.search(r"_sample_(\d+)\.cif$", low)
        if not m:
            return "cryozeta_other"
        return "cryozeta" if int(m.group(1)) == 0 else "cryozeta_s%d" % int(m.group(1))
    return method + "_other"


def collect_models(roots):
    """Models of result folders (`<root>/models/<method>/<ENTRY>/<file>`), oldest root first; a model that appears in several roots is taken from the last one.
    Returns (list of dicts, list of warnings)."""
    found, warnings = {}, []
    for root in roots:
        label = os.path.basename(os.path.normpath(root))
        for path in sorted(glob.glob(os.path.join(root, "models", "*", "*", "*"))):
            if not os.path.isfile(path) or not path.lower().endswith(MODEL_SUFFIXES):
                continue
            parts = os.path.normpath(path).split(os.sep)
            method, entry, fname = parts[-3], parts[-2].upper(), parts[-1]
            kind = model_kind(method, fname)
            key = (method, entry, kind, fname.lower())
            if key in found:
                warnings.append("%s %s %s: in %s and in %s (the last one is used)" % (method, entry, fname, found[key]["source"], label))
            found[key] = {"method": method, "entry": entry, "model_kind": kind, "primary": kind in PRIMARY_KINDS, "model_file": fname, "path": path, "source": label}
    return sorted(found.values(), key=lambda m: (m["entry"], m["method"], m["model_kind"], m["model_file"])), warnings


def prepare_natives(entries, cif_dir, out_dir):
    """Reads <cif_dir>/<ENTRY>.cif for every entry row (dict with pdb_id and modeled_residues), writes <out_dir>/<ENTRY>_native_ca.pdb and returns
    {entry: (Trace, path, note)}. The note says when the residues differ from `modeled_residues` of the entries table."""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for row in entries:
        entry = row["pdb_id"].upper()
        src = None
        for cand in (os.path.join(cif_dir, entry + ".cif"), os.path.join(cif_dir, entry.lower() + ".cif"), os.path.join(cif_dir, entry + ".pdb")):
            if os.path.exists(cand):
                src = cand
                break
        if src is None:
            raise FileNotFoundError("no deposited model for %s in %s" % (entry, cif_dir))
        trace = read_trace(src)
        path = write_pdb(trace, os.path.join(out_dir, entry + "_native_ca.pdb"))
        note = ""
        want = row.get("modeled_residues")
        if want not in (None, "") and int(float(want)) != trace.n_residues:
            note = "%d amino-acid residues with a CA atom, entries table says %s" % (trace.n_residues, want)
        out[entry] = (trace, path, note)
    return out


def write_scores(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


COORDINATE_FILES = ("query_superposed.pdb", "query_frame_fixed.pdb", "pred_merged.pdb", "native_merged.pdb", "native_ca.pdb")


def score_all(models, natives, usalign, phenix_env, out_dir, workers=4, log=print, only_primary=False, keep_coordinates=False, progress=None):
    """Scores every model of collect_models; returns the rows (and writes scores_long.csv and the raw outputs under out_dir/raw).
    The coordinate files that the scoring writes (traces of the predictions: for MICA they derive from the private AlphaFold predictions) are removed after each model
    unless keep_coordinates is set; progress(rows so far) is called after every model."""
    import threading
    jobs = [m for m in models if (m["primary"] or not only_primary)]
    os.makedirs(out_dir, exist_ok=True)
    rows = [None] * len(jobs)
    lock = threading.Lock()

    def work(i):
        m = jobs[i]
        wd = os.path.join(out_dir, "raw", m["entry"], os.path.splitext(m["model_file"])[0].replace("(", "_").replace(")", "_").replace("+", "_"))
        if m["entry"] not in natives:
            r = {k: "" for k in SCORE_COLUMNS}
            r.update(status="no_native", note="no deposited model for this entry")
            trace, native_file, native_note = None, None, ""
        else:
            trace, native_file, native_note = natives[m["entry"]]
            try:
                r = score_model(usalign, phenix_env, trace, m["path"], wd, native_file=native_file)
            except Exception as e:                    # a bug must not cost the other models
                r = {k: "" for k in SCORE_COLUMNS}
                r.update(status="no_alignment", note="unexpected error: %r" % (e,), native_residues=trace.n_residues)
            if not keep_coordinates:
                for name in COORDINATE_FILES:
                    try:
                        os.remove(os.path.join(wd, name))
                    except OSError:
                        pass
        r.update(entry=m["entry"], method=m["method"], model_kind=m["model_kind"], primary=m["primary"], model_file=m["model_file"], source=m["source"])
        if native_note:
            r["note"] = ("native: " + native_note + (" | " + r["note"] if r["note"] else ""))[:600]
        with lock:
            rows[i] = r
            log("[%d/%d] %-5s %-12s %-22s %-10s TM %-7s fNS %-7s CAmatch %-7s %ss" % (i + 1, len(jobs), m["entry"], m["method"], m["model_kind"], r["status"], r["tm_score"], r["tm_score_fns"], r["ca_match"], r["seconds"]))
            if progress:
                progress([x for x in rows if x is not None])
        return r

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, range(len(jobs))))
    write_scores(rows, os.path.join(out_dir, "scores_long.csv"))
    return rows


def read_entries(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one", help="score one model")
    one.add_argument("--usalign", required=True)
    one.add_argument("--native", required=True, help="deposited model (pdb or mmCIF)")
    one.add_argument("--pred", required=True)
    one.add_argument("--phenix-env", help="a shell file that makes the phenix.* programs available (not given: Phenix is not run)")
    one.add_argument("--workdir", required=True)
    bat = sub.add_parser("batch", help="score all the models of result folders")
    bat.add_argument("--usalign", required=True)
    bat.add_argument("--roots", nargs="+", required=True, help="result folders (with models/<method>/<ENTRY>/), oldest first")
    bat.add_argument("--entries", required=True, help="data/entries_2026.csv")
    bat.add_argument("--natives", required=True, help="folder with <ENTRY>.cif of the deposited models")
    bat.add_argument("--out", required=True)
    bat.add_argument("--phenix-env")
    bat.add_argument("--workers", type=int, default=4)
    bat.add_argument("--only-primary", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "one":
        native = read_trace(args.native)
        res = score_model(args.usalign, args.phenix_env, native, args.pred, args.workdir)
        print(json.dumps(res, indent=1))
        return 0 if res["status"] == "ok" else 2
    entries = read_entries(args.entries)
    natives = prepare_natives(entries, args.natives, os.path.join(args.out, "natives"))
    models, warnings = collect_models(args.roots)
    for w in warnings:
        print("WARNING:", w)
    rows = score_all(models, natives, args.usalign, args.phenix_env, args.out, args.workers, only_primary=args.only_primary)
    return 0 if all(r["status"] == "ok" for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
