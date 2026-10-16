#!/usr/bin/env python3
"""Offline tests for make_af3_jobs.py.

    python scripts/test_make_af3_jobs.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_af3_jobs as aj  # noqa: E402

ROOT = os.path.dirname(HERE)


def entry(pdb_id, rank, stratum, n_seq):
    return {"pdb_id": pdb_id, "seed_rank": str(rank), "stratum": stratum, "unique_chain_seqs": str(n_seq)}


def record(name, seq, copies=1):
    return name, ("%s len=%d copies=%d longest=0" % (name, len(seq), copies), seq)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_order_follows_the_seeded_order_and_every_sequence_gets_a_job(self):
        entries = [entry("9BBB", 2, "le3.0", 2), entry("9AAA", 1, "gt3.0_le4.0", 1), entry("9CCC", 1, "le3.0", 1)]
        records = dict([record("9BBB_E1", "ACDEFGHIK"), record("9BBB_E2", "LMNPQRSTV", 2), record("9AAA_E1", "WYACDEFG"), record("9CCC_E1", "KLMNPQRS"),
                        record("9ZZZ_E1", "AAAAAAAA")])
        jobs = aj.build_jobs(entries, records)
        self.assertEqual([j["job_name"] for j in jobs], ["9CCC_E1", "9AAA_E1", "9BBB_E1", "9BBB_E2"])
        self.assertEqual([j["job_no"] for j in jobs], [1, 2, 3, 4])
        self.assertEqual(jobs[3]["copies_in_entry"], "2")
        self.assertEqual(jobs[3]["length"], 9)

    def test_entities_are_sorted_by_number_not_as_text(self):
        records = dict(record("9AAA_E%d" % i, "ACDEFGHIK") for i in (10, 2, 1))
        jobs = aj.build_jobs([entry("9AAA", 1, "le3.0", 3)], records)
        self.assertEqual([j["entity"] for j in jobs], ["E1", "E2", "E10"])

    def test_a_letter_outside_the_20_is_replaced_and_noted(self):
        self.assertEqual(aj.clean("XYKPLGLAY"), ("GYKPLGLAY", "1 residue(s) X replaced by G"))
        self.assertEqual(aj.clean("acdefg"), ("ACDEFG", ""))
        self.assertEqual(aj.clean("AUBXA")[0], "AGGGA")
        jobs = aj.build_jobs([entry("9AAA", 1, "le3.0", 1)], dict([record("9AAA_E1", "XYKPLGLAY")]))
        self.assertEqual((jobs[0]["sequence_submitted"], jobs[0]["substitution"]), ("GYKPLGLAY", "1 residue(s) X replaced by G"))

    def test_inconsistent_input_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no sequence"):
            aj.build_jobs([entry("9AAA", 1, "le3.0", 1)], {})
        with self.assertRaisesRegex(ValueError, "unique_chain_seqs"):
            aj.build_jobs([entry("9AAA", 1, "le3.0", 2)], dict([record("9AAA_E1", "ACDEFGHIK")]))
        with self.assertRaisesRegex(ValueError, "outside what AlphaFold Server accepts"):
            aj.build_jobs([entry("9AAA", 1, "le3.0", 1)], dict([record("9AAA_E1", "ACD")]))
        name, (header, seq) = record("9AAA_E1", "ACDEFGHIK")
        with self.assertRaisesRegex(ValueError, "header says"):
            aj.build_jobs([entry("9AAA", 1, "le3.0", 1)], {name: (header.replace("len=9", "len=10"), seq)})

    def test_files_and_the_server_json(self):
        entries_path, fasta_path, out = (os.path.join(self.tmp.name, n) for n in ("entries.csv", "seqs.fasta", "out"))
        with open(entries_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pdb_id", "seed_rank", "stratum", "unique_chain_seqs"], lineterminator="\n")
            w.writeheader()
            w.writerows([entry("9AAA", 1, "le3.0", 1), entry("9BBB", 2, "le3.0", 2)])
        with open(fasta_path, "w") as f:
            for name, (header, seq) in (record("9AAA_E1", "ACDEFGHIK"), record("9BBB_E1", "LMNPQRSTV"), record("9BBB_E2", "XYKPLGLAY")):
                f.write(">%s\n%s\n" % (header, seq[:4]))
                f.write(seq[4:] + "\n")                                             # a wrapped line
        buf, err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            code = aj.main(["--entries", entries_path, "--fasta", fasta_path, "--out", out])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("3 jobs", buf.getvalue())
        self.assertIn("9BBB_E2", buf.getvalue())
        with open(os.path.join(out, "af3_jobs.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["job_name"] for r in rows], ["9AAA_E1", "9BBB_E1", "9BBB_E2"])
        self.assertEqual(rows[2]["sequence_submitted"], "GYKPLGLAY")
        with open(os.path.join(out, "af3_jobs.fasta")) as f:
            self.assertEqual(f.read(), ">9AAA_E1\nACDEFGHIK\n>9BBB_E1\nLMNPQRSTV\n>9BBB_E2\nGYKPLGLAY\n")
        with open(os.path.join(out, "af3_server_jobs.json")) as f:
            data = json.load(f)
        self.assertEqual(data[0], {"name": "9AAA_E1", "modelSeeds": [], "sequences": [{"proteinChain": {"sequence": "ACDEFGHIK", "count": 1}}],
                                   "dialect": "alphafoldserver", "version": 1})
        self.assertEqual(len(data), 3)

    def test_errors_in_the_command_line_tool_give_exit_code_2(self):
        err = io.StringIO()
        with redirect_stderr(err):
            code = aj.main(["--entries", os.path.join(self.tmp.name, "missing.csv"), "--fasta", "x", "--out", self.tmp.name])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err.getvalue())

    def test_the_frozen_list_of_the_repository(self):
        entries_path = os.path.join(ROOT, "data", "entries_2026.csv")
        fasta_path = os.path.join(ROOT, "data", "pool", "pool_candidate_sequences.fasta")
        if not (os.path.exists(entries_path) and os.path.exists(fasta_path)):
            self.skipTest("repository data not found")
        with open(entries_path, newline="") as f:
            entries = list(csv.DictReader(f))
        if not entries:
            self.skipTest("the entry list is not frozen yet")
        jobs = aj.build_jobs(entries, aj.read_fasta(fasta_path))
        self.assertEqual((len(entries), len(jobs)), (20, 26))
        self.assertEqual({j["pdb_id"] for j in jobs}, {e["pdb_id"] for e in entries})
        self.assertEqual([j["job_name"] for j in jobs if j["substitution"]], ["9V4D_E2"])
        self.assertEqual([j["job_name"] for j in jobs[:3]], ["9T5K_E1", "22FX_E1", "9JKK_E1"])
        self.assertTrue(all(set(j["sequence_submitted"]) <= aj.STANDARD for j in jobs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
