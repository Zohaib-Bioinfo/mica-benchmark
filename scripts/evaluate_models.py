#!/usr/bin/env python3
"""Phase 4: scores one predicted model against the deposited model (docs/SPEC.md, decision 7), version 0.1 for the pilot.

What it computes with US-align (protein chains only, `-mol protein`, tabular output `-outfmt 2`; the prediction is structure 1 and the
deposited model is structure 2, so TM2 is the TM-score normalised by the length of the deposited model, the one that the US-align
output recommends):

    tm_score            TM2, normalised by the deposited model
    tm_score_pred_norm  TM1, normalised by the prediction (for reference only)
    ca_rmsd             RMSD over the aligned residues
    aligned_ca_length   number of aligned residues (Lali)
    seq_identity        identical residues among the aligned ones (IDali)
    pred_residues       protein residues in the prediction
    native_residues     protein residues in the deposited model (US-align L2)
    coverage            built residues / deposited residues; the deposited count is `modeled_residues` of data/entries_2026.csv

The Phenix part (Calpha match and sequence match from phenix.chain_comparison, and with them the Calpha quality score) is not here yet:
the pilot saves the raw Phenix output so that the exact command and the output lines can be fixed before the scripts are trusted.

    python scripts/evaluate_models.py --usalign <path to USalign> --pred model.pdb --native 26DK.cif --modeled-residues 191
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

SCRIPT_VERSION = "0.1"
TABLE_COLUMNS = ["PDBchain1", "PDBchain2", "TM1", "TM2", "RMSD", "ID1", "ID2", "IDali", "L1", "L2", "Lali"]


def parse_usalign_tabular(text):
    """The rows of `USalign -outfmt 2`: a list of dicts with numbers as floats (L1, L2 and Lali as integers)."""
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


def run_usalign(usalign, pred, native, timeout=900):
    """US-align of the prediction onto the deposited model; returns the one table row, or raises RuntimeError with the reason."""
    cmd = [usalign, pred, native, "-mol", "protein", "-outfmt", "2"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
    except subprocess.TimeoutExpired:
        raise RuntimeError("US-align did not finish within %d seconds" % timeout)
    except OSError as e:
        raise RuntimeError("US-align could not be started: %s" % e)
    if r.returncode != 0:
        raise RuntimeError("US-align stopped with exit code %d: %s" % (r.returncode, (r.stderr or r.stdout).strip()[-300:]))
    rows = parse_usalign_tabular(r.stdout)
    if len(rows) != 1:
        raise RuntimeError("US-align gave %d result lines, 1 expected. Output: %s" % (len(rows), r.stdout.strip()[-300:]))
    return rows[0]


def evaluate(usalign, pred, native, modeled_residues):
    """The numbers of one model. status is "ok", or the reason why no alignment exists (then the numbers are empty)."""
    out = {"status": "ok", "note": "", "tm_score": "", "tm_score_pred_norm": "", "ca_rmsd": "", "aligned_ca_length": "", "seq_identity": "",
           "pred_residues": "", "native_residues": "", "coverage": ""}
    try:
        row = run_usalign(usalign, pred, native)
    except (RuntimeError, ValueError) as e:
        out.update(status="no_alignment", note=str(e)[:300])
        return out
    out.update(tm_score=row["TM2"], tm_score_pred_norm=row["TM1"], ca_rmsd=row["RMSD"], aligned_ca_length=row["Lali"], seq_identity=row["IDali"],
               pred_residues=row["L1"], native_residues=row["L2"])
    if modeled_residues:
        out["coverage"] = round(row["L1"] / float(modeled_residues), 4)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usalign", required=True, help="path to the USalign program")
    p.add_argument("--pred", required=True, help="predicted model (pdb or cif)")
    p.add_argument("--native", required=True, help="deposited model (pdb or cif)")
    p.add_argument("--modeled-residues", type=int, default=0, help="modeled_residues of the entry in data/entries_2026.csv")
    p.add_argument("--out", help="write the numbers as json")
    args = p.parse_args(argv)
    res = evaluate(args.usalign, args.pred, args.native, args.modeled_residues)
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0 if res["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
