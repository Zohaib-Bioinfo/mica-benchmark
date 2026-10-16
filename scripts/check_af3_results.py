#!/usr/bin/env python3
"""Phase 4 check: the AlphaFold Server predictions that came back for the frozen entry list (docs/SPEC.md, decision 6).

Reads data/af3/af3_jobs.csv (what was submitted) and a folder with the files that AlphaFold Server returned
(fold_<job name in lower case>_model_0.cif; decision 6 uses model_0 only) and checks, for every job, that

  - exactly one such file exists and it can be read,
  - it holds exactly one chain, and every residue of it is an amino acid,
  - the sequence of that chain equals the sequence that was submitted, letter by letter,
  - every coordinate is a finite number and every pLDDT (the B-factor column of AlphaFold mmCIF files) lies between 0 and 100.

It writes one row per job (status, residues, mean pLDDT of the C-alpha atoms, share of residues with pLDDT of at least 70 and below 50,
sha256 of the file) and, with --manifest, the sha256 of every file used. The statistics and the hashes are public; the predictions themselves
are not (AlphaFold Server terms: non-commercial research use), so the folder with the files stays outside the public repo.
Files in the folder that belong to no job, and a file name that occurs twice, are reported and make the exit code 2.

    python scripts/check_af3_results.py --jobs data/af3/af3_jobs.csv --models <folder> --out data/af3/af3_results_check.csv --manifest data/af3/af3_files_sha256.txt

Exit code 0: all jobs ok. 2: a problem (the table says which job and why).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys

SCRIPT_VERSION = "1.0"
COLUMNS = ["job_no", "job_name", "pdb_id", "entity", "status", "note", "expected_length", "residues", "chains", "atoms",
           "mean_plddt_ca", "frac_ca_plddt_ge70", "frac_ca_plddt_lt50", "file", "sha256"]


def file_key(job_name):
    return "fold_%s_model_0.cif" % job_name.lower()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jobs(path):
    with open(path, newline="", encoding="utf-8") as f:
        jobs = list(csv.DictReader(f))
    if not jobs:
        raise ValueError("%s has no jobs" % path)
    names = [j["job_name"] for j in jobs]
    if len(set(n.lower() for n in names)) != len(names):
        raise ValueError("%s: a job name occurs twice" % path)
    return jobs


def find_models(folder):
    """({lower-case file name: path}, set of file names that occur more than once)."""
    found, twice = {}, set()
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(".cif"):
                key = name.lower()
                if key in found:
                    twice.add(key)
                found[key] = os.path.join(root, name)
    return found, twice


def inspect_model(path, expected):
    """Dict with status, note and the numbers of one model file; status is "ok" or the reason it is not."""
    import gemmi
    out = {"status": "ok", "note": "", "residues": "", "chains": "", "atoms": "", "mean_plddt_ca": "",
           "frac_ca_plddt_ge70": "", "frac_ca_plddt_lt50": ""}
    try:
        st = gemmi.read_structure(path)
        st.setup_entities()
    except Exception as e:                                   # gemmi raises RuntimeError / ValueError on a damaged file
        out.update(status="unreadable", note=str(e).strip().splitlines()[0][:150] if str(e).strip() else "cannot be read")
        return out
    if len(st) != 1:
        out.update(status="not_one_model", note="%d models in the file" % len(st))
        return out
    chains = [c for c in st[0] if len(c) > 0]
    out["chains"] = len(chains)
    if len(chains) != 1:
        out.update(status="not_one_chain", note="%d chains in the file, 1 expected" % len(chains))
        return out
    chain = chains[0]
    letters, plddt_ca, atoms, odd = [], [], 0, []
    for res in chain:
        info = gemmi.find_tabulated_residue(res.name)
        if info is None or not info.is_amino_acid():
            odd.append(res.name)
            continue
        letters.append(info.one_letter_code.upper())
        atoms += len(res)
        for a in res:
            vals = (a.pos.x, a.pos.y, a.pos.z, a.b_iso)
            if not all(math.isfinite(v) for v in vals):
                out.update(status="bad_numbers", note="non-finite coordinate or pLDDT in residue %s %s" % (res.name, res.seqid.num))
                return out
            if not 0.0 <= a.b_iso <= 100.0:
                out.update(status="bad_numbers", note="pLDDT %.1f outside 0 to 100 in residue %s %s" % (a.b_iso, res.name, res.seqid.num))
                return out
            if a.name == "CA":
                plddt_ca.append(a.b_iso)
    out["residues"], out["atoms"] = len(letters), atoms
    if odd:
        out.update(status="not_protein_only", note="residues that are not amino acids: %s" % ",".join(sorted(set(odd))))
        return out
    seq = "".join(letters)
    if seq != expected:
        k = next((i for i, (a, b) in enumerate(zip(seq, expected)) if a != b), min(len(seq), len(expected)))
        out.update(status="sequence_mismatch", note="model has %d residues, %d submitted; first difference at position %d" % (len(seq), len(expected), k + 1))
        return out
    if len(plddt_ca) != len(seq):
        out.update(status="no_ca", note="%d C-alpha atoms for %d residues" % (len(plddt_ca), len(seq)))
        return out
    n = float(len(plddt_ca))
    out["mean_plddt_ca"] = "%.2f" % (sum(plddt_ca) / n)
    out["frac_ca_plddt_ge70"] = "%.3f" % (sum(1 for v in plddt_ca if v >= 70.0) / n)
    out["frac_ca_plddt_lt50"] = "%.3f" % (sum(1 for v in plddt_ca if v < 50.0) / n)
    return out


def check(jobs, folder):
    """(rows, problems): one row per job, in job order, and a list of sentences about everything that is not in order."""
    found, twice = find_models(folder)
    rows, problems, used = [], [], set()
    for j in jobs:
        key = file_key(j["job_name"])
        row = {"job_no": j["job_no"], "job_name": j["job_name"], "pdb_id": j["pdb_id"], "entity": j["entity"],
               "expected_length": len(j["sequence_submitted"]), "file": "", "sha256": ""}
        if key in twice:
            row.update(status="file_twice", note="%s occurs more than once in the folder" % key)
        elif key not in found:
            row.update(status="missing", note="no file %s" % key)
        else:
            used.add(key)
            row.update(inspect_model(found[key], j["sequence_submitted"]))
            row["file"] = key
            row["sha256"] = sha256_of(found[key])
        for c in COLUMNS:
            row.setdefault(c, "")
        if row["status"] != "ok":
            problems.append("%s: %s (%s)" % (j["job_name"], row["status"], row["note"]))
        rows.append(row)
    for key in sorted(set(found) - used - twice):
        problems.append("%s: a file that belongs to no job" % key)
    for key in sorted(twice - {file_key(j["job_name"]) for j in jobs}):
        problems.append("%s: occurs more than once and belongs to no job" % key)
    return rows, problems


def write_outputs(rows, out_csv, manifest=None):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    if manifest:
        with open(manifest, "w", encoding="utf-8") as f:
            for r in rows:
                if r["sha256"]:
                    f.write("%s  %s\n" % (r["sha256"], r["file"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", required=True, help="data/af3/af3_jobs.csv")
    p.add_argument("--models", required=True, help="folder with the fold_*_model_0.cif files from AlphaFold Server")
    p.add_argument("--out", required=True, help="result table (csv)")
    p.add_argument("--manifest", help="optional: sha256 of every file used (text)")
    args = p.parse_args(argv)
    try:
        import gemmi  # noqa: F401
        jobs = read_jobs(args.jobs)
        if not os.path.isdir(args.models):
            raise ValueError("%s is not a folder" % args.models)
        rows, problems = check(jobs, args.models)
        write_outputs(rows, args.out, args.manifest)
    except (ValueError, KeyError, OSError, ImportError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    ok = [r for r in rows if r["status"] == "ok"]
    print("%d jobs, %d ok, %d problems" % (len(rows), len(ok), len(problems)))
    if ok:
        means = [float(r["mean_plddt_ca"]) for r in ok]
        print("mean pLDDT of the C-alpha atoms per job: lowest %.1f, median %.1f, highest %.1f" % (min(means), sorted(means)[len(means) // 2], max(means)))
    for line in problems:
        print("PROBLEM: " + line)
    print("written to %s" % args.out)
    return 2 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
