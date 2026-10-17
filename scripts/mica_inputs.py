#!/usr/bin/env python3
"""Phase 4: the input folder that MICA expects, for one entry of the frozen list.

MICA reads (README, "Running on New Dataset", and utils/process_AF3_results.py):

    <root>/<PDB ID>/
        <pdb id>.fasta                         one record per unique chain sequence, header  >26DK_1|Chains A, B|...
        emd_<n>.map                            (put there by the notebook)
        AF3_results/<pdb id>_<k>/fold_<pdb id>_<k>_model_0.cif

The header must start with the PDB ID, one underscore, the number of the sequence, then "|Chains ..." (the number of chains is
the number of copies of that sequence in the entry). The third part of the file name (split at "_") must be the number k, and
the PDB ID of the first header must be the one of the files, or MICA finds no protein for the file and drops it silently.
AlphaFold Server returned the files as fold_<job name in lower case>_model_0.cif with the job names of data/af3/af3_jobs.csv
(for example fold_26dk_e1_model_0.cif); this script renames E1, E2, ... to 1, 2, ... and writes the FASTA file from the same table, so the
sequence MICA reads is the sequence that AlphaFold Server was given (letters outside the 20 amino acids were replaced by G there).

    python scripts/mica_inputs.py --jobs data/af3/af3_jobs.csv --models <folder with the 26 files> --pdb 26DK --out <root>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

SCRIPT_VERSION = "1.0"
CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def read_jobs(path, pdb_id):
    """The rows of af3_jobs.csv for one entry, in the order of the entities (E1, E2, ...)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["pdb_id"].upper() == pdb_id.upper()]
    if not rows:
        raise ValueError("%s: no job for %s" % (path, pdb_id))
    rows.sort(key=lambda r: int(r["entity"].lstrip("Ee")))
    numbers = [int(r["entity"].lstrip("Ee")) for r in rows]
    if numbers != list(range(1, len(rows) + 1)):
        raise ValueError("%s: the entities of %s are %s, expected 1 to %d without gaps" % (path, pdb_id, numbers, len(rows)))
    return rows


def find_model(models_dir, job_name):
    """Path of fold_<job name in lower case>_model_0.cif somewhere under models_dir (case does not matter)."""
    want = "fold_%s_model_0.cif" % job_name.lower()
    hits = []
    for root, _dirs, files in os.walk(models_dir):
        hits += [os.path.join(root, f) for f in files if f.lower() == want]
    if not hits:
        raise ValueError("no file %s under %s" % (want, models_dir))
    if len(hits) > 1:
        raise ValueError("%s occurs %d times under %s" % (want, len(hits), models_dir))
    return hits[0]


def mica_fasta(rows):
    """FASTA text with MICA's header format. The chain letters run through the entry (A, B for the first sequence, C, D for the next, ...)."""
    total = sum(int(r["copies_in_entry"]) for r in rows)
    if total > len(CHAIN_LETTERS):
        raise ValueError("%s has %d chains; MICA writes PDB files and cannot name more than %d" % (rows[0]["pdb_id"], total, len(CHAIN_LETTERS)))
    out, used = [], 0
    for k, r in enumerate(rows, 1):
        copies = int(r["copies_in_entry"])
        if copies < 1:
            raise ValueError("%s: copies_in_entry is %s" % (r["job_name"], r["copies_in_entry"]))
        letters = ", ".join(CHAIN_LETTERS[used:used + copies])
        used += copies
        pdb = r["pdb_id"].upper()
        if "_" in pdb or "|" in pdb:
            raise ValueError("the PDB ID %r contains _ or |" % pdb)
        out.append(">%s_%d|Chains %s|%s sequence %d|" % (pdb, k, letters, pdb, k))
        out.append(r["sequence_submitted"])
    return "\n".join(out) + "\n"


def build(jobs_csv, models_dir, pdb_id, out_root):
    """Writes <out_root>/<PDB ID>/ with the FASTA file and AF3_results; returns a dict that says what was written."""
    rows = read_jobs(jobs_csv, pdb_id)
    pdb = rows[0]["pdb_id"].upper()
    low = pdb.lower()
    folder = os.path.abspath(os.path.join(out_root, pdb))
    results = os.path.join(folder, "AF3_results")
    # MICA derives its other folders with AF3_results_path.replace("results", ...): the word must occur only once, in the last part
    if "results" in os.path.dirname(results):
        raise ValueError("the path %s contains the word 'results' before AF3_results; MICA would write into the wrong folders. Use another root folder" % folder)
    os.makedirs(results, exist_ok=True)
    fasta_path = os.path.join(folder, low + ".fasta")
    with open(fasta_path, "w", encoding="utf-8") as f:
        f.write(mica_fasta(rows))
    files = []
    for k, r in enumerate(rows, 1):
        src = find_model(models_dir, r["job_name"])
        d = os.path.join(results, "%s_%d" % (low, k))
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "fold_%s_%d_model_0.cif" % (low, k))
        shutil.copyfile(src, dst)
        files.append({"job_name": r["job_name"], "source": os.path.basename(src), "target": os.path.relpath(dst, folder)})
    return {"pdb_id": pdb, "folder": folder, "fasta": fasta_path, "af3_results": results, "files": files}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", required=True, help="data/af3/af3_jobs.csv")
    p.add_argument("--models", required=True, help="folder with the fold_*_model_0.cif files from AlphaFold Server")
    p.add_argument("--pdb", required=True, help="PDB ID of the entry, for example 26DK")
    p.add_argument("--out", required=True, help="root folder; the entry gets <out>/<PDB ID>/")
    args = p.parse_args(argv)
    try:
        info = build(args.jobs, args.models, args.pdb, args.out)
    except (ValueError, OSError, KeyError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
