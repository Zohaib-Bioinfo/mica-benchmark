#!/usr/bin/env python3
"""Phase 4 input: the AlphaFold Server jobs for the frozen entry list (docs/SPEC.md, decision 6).

Decision 6: the same AF3 predictions (AlphaFold Server, model_0, one prediction per unique chain sequence) go to every method
that takes predicted structures. This script reads the frozen list (data/entries_2026.csv) and the entity sequences of the pool
(data/pool/pool_candidate_sequences.fasta) and writes, into one folder:

    af3_jobs.csv          one row per job: job name, entry, entity, length, copies in the entry, the sequence that is submitted
    af3_jobs.fasta        the same sequences, for pasting into the web form by hand
    af3_server_jobs.json  all jobs as one file for the button "Upload JSON" of AlphaFold Server (up to 100 jobs per file)

Jobs are ordered by the seeded order of the entries (seed_rank, then the stratum up to 3.0 A first), so that, if the daily quota runs short, the jobs
of the entries at the bottom of the seeded order are the ones left for the next day (decision 12). A sequence with a letter
outside the 20 standard amino acids cannot be submitted: each such letter is replaced by G and the replacement is written
in the column "substitution".

    python scripts/make_af3_jobs.py --entries data/entries_2026.csv --fasta data/pool/pool_candidate_sequences.fasta --out data/af3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

SCRIPT_VERSION = "1.0"
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 4                    # AlphaFold Server: shortest macromolecule chain
MAX_TOKENS = 5000              # AlphaFold Server: tokens per job
COLUMNS = ["job_no", "job_name", "pdb_id", "entity", "stratum", "seed_rank", "length", "copies_in_entry", "sequence_submitted", "substitution"]


def read_fasta(path):
    """{record id: (header line, sequence)}."""
    records, cur = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                cur = line[1:].split()[0]
                if cur in records:
                    raise ValueError("%s: record %s appears twice" % (path, cur))
                records[cur] = [line, ""]
            elif cur is not None:
                records[cur][1] += line.strip()
    return {k: (v[0], v[1]) for k, v in records.items()}


def header_value(header, key):
    for part in header.split():
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return ""


def clean(sequence):
    """(sequence the server accepts, note). Letters outside the 20 standard amino acids become G."""
    sequence = sequence.strip().upper()
    odd = sorted({c for c in sequence if c not in STANDARD})
    if not odd:
        return sequence, ""
    fixed = "".join(c if c in STANDARD else "G" for c in sequence)
    count = sum(1 for c in sequence if c not in STANDARD)
    return fixed, "%d residue(s) %s replaced by G" % (count, "/".join(odd))


def build_jobs(entries, records):
    jobs = []
    stratum_order = {"le3.0": 0, "gt3.0_le4.0": 1}
    for e in sorted(entries, key=lambda r: (int(r["seed_rank"]), stratum_order.get(r["stratum"], 9), r["pdb_id"])):
        ents = sorted((k for k in records if k.split("_")[0] == e["pdb_id"]), key=lambda k: int(k.split("_E")[1]))
        if not ents:
            raise ValueError("no sequence for %s in the FASTA file" % e["pdb_id"])
        if len(ents) != int(e["unique_chain_seqs"]):
            raise ValueError("%s: %d sequences in the FASTA file but unique_chain_seqs is %s" % (e["pdb_id"], len(ents), e["unique_chain_seqs"]))
        for k in ents:
            header, seq = records[k]
            sub, note = clean(seq)
            if len(sub) < MIN_LEN or len(sub) > MAX_TOKENS:
                raise ValueError("%s: length %d is outside what AlphaFold Server accepts (%d to %d)" % (k, len(sub), MIN_LEN, MAX_TOKENS))
            if header_value(header, "len") and int(header_value(header, "len")) != len(seq):
                raise ValueError("%s: header says len=%s but the sequence has %d residues" % (k, header_value(header, "len"), len(seq)))
            jobs.append({"job_name": k, "pdb_id": e["pdb_id"], "entity": k.split("_")[1], "stratum": e["stratum"], "seed_rank": e["seed_rank"],
                         "length": len(sub), "copies_in_entry": header_value(header, "copies"), "sequence_submitted": sub, "substitution": note})
    for i, j in enumerate(jobs, 1):
        j["job_no"] = i
    return jobs


def server_json(jobs):
    """The format of AlphaFold Server for 'Upload JSON': a list of jobs, one protein chain each."""
    return [{"name": j["job_name"], "modelSeeds": [], "sequences": [{"proteinChain": {"sequence": j["sequence_submitted"], "count": 1}}],
             "dialect": "alphafoldserver", "version": 1} for j in jobs]


def write_outputs(jobs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "af3_jobs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(jobs)
    with open(os.path.join(out_dir, "af3_jobs.fasta"), "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(">%s\n%s\n" % (j["job_name"], j["sequence_submitted"]))
    with open(os.path.join(out_dir, "af3_server_jobs.json"), "w", encoding="utf-8") as f:
        json.dump(server_json(jobs), f, indent=1)
        f.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entries", required=True, help="data/entries_2026.csv")
    p.add_argument("--fasta", required=True, help="data/pool/pool_candidate_sequences.fasta")
    p.add_argument("--out", required=True, help="output folder")
    args = p.parse_args(argv)
    try:
        with open(args.entries, newline="", encoding="utf-8") as f:
            entries = list(csv.DictReader(f))
        if not entries:
            raise ValueError("%s has no entries" % args.entries)
        jobs = build_jobs(entries, read_fasta(args.fasta))
    except (ValueError, KeyError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    write_outputs(jobs, args.out)
    changed = [j["job_name"] for j in jobs if j["substitution"]]
    print("%d entries, %d jobs (one per unique chain sequence); AlphaFold Server allows 30 jobs per day" % (len(entries), len(jobs)))
    if changed:
        print("sequence changed to fit the server: %s" % ", ".join(changed))
    print("written to %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
