"""Tests for mica_inputs.py. Run from the scripts folder:  python -W error -m unittest test_mica_inputs"""
import csv
import os
import re
import shutil
import tempfile
import unittest

import mica_inputs as mi

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_JOBS = os.path.join(HERE, "..", "data", "af3", "af3_jobs.csv")
COLUMNS = ["job_no", "job_name", "pdb_id", "entity", "stratum", "seed_rank", "length", "copies_in_entry", "sequence_submitted", "substitution"]
JOBS = [("9AAA_E1", "9AAA", "E1", 2, "MKVLAGGW"), ("9AAA_E2", "9AAA", "E2", 1, "GYKPLGLAY"), ("9BBB_E1", "9BBB", "E1", 1, "ACDEFGHIK")]


class MicaInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="micain_")
        self.addCleanup(shutil.rmtree, self.tmp)
        self.jobs = os.path.join(self.tmp, "jobs.csv")
        with open(self.jobs, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(COLUMNS)
            for i, (name, pdb, ent, copies, seq) in enumerate(JOBS, 1):
                w.writerow([i, name, pdb, ent, "le3.0", i, len(seq), copies, seq, ""])
        self.models = os.path.join(self.tmp, "AlphaFold ServerJOBS")
        os.makedirs(self.models)
        for name, _p, _e, _c, seq in JOBS:
            with open(os.path.join(self.models, "fold_%s_model_0.cif" % name.lower()), "w") as f:
                f.write("data_%s\n" % name)
        self.out = os.path.join(self.tmp, "mica_input")

    def test_folder_layout_and_names(self):
        info = mi.build(self.jobs, self.models, "9AAA", self.out)
        base = os.path.join(self.out, "9AAA")
        self.assertTrue(os.path.isfile(os.path.join(base, "9aaa.fasta")))
        self.assertTrue(os.path.isfile(os.path.join(base, "AF3_results", "9aaa_1", "fold_9aaa_1_model_0.cif")))
        self.assertTrue(os.path.isfile(os.path.join(base, "AF3_results", "9aaa_2", "fold_9aaa_2_model_0.cif")))
        self.assertEqual([f["target"] for f in info["files"]], ["AF3_results/9aaa_1/fold_9aaa_1_model_0.cif", "AF3_results/9aaa_2/fold_9aaa_2_model_0.cif"])
        with open(os.path.join(base, "AF3_results", "9aaa_2", "fold_9aaa_2_model_0.cif")) as f:
            self.assertEqual(f.read(), "data_9AAA_E2\n")                # the file of job E2 went to number 2
        self.assertFalse(os.path.exists(os.path.join(self.out, "9BBB")))

    def test_headers_follow_the_rules_of_the_mica_readme(self):
        rows = mi.read_jobs(self.jobs, "9AAA")
        text = mi.mica_fasta(rows)
        self.assertEqual(text, ">9AAA_1|Chains A, B|9AAA sequence 1|\nMKVLAGGW\n>9AAA_2|Chains C|9AAA sequence 2|\nGYKPLGLAY\n")
        for line in text.splitlines():
            if line.startswith(">"):
                self.assertEqual(line.split("|")[0].count("_"), 1)       # exactly one underscore before the first bar
                self.assertTrue(re.match(r">([^|]+)\|", line))          # the pattern of process_AF3_results.py

    def test_what_mica_does_with_the_names(self):
        """The lines of process_AF3_results.py that turn a file name and a header into a protein id."""
        mi.build(self.jobs, self.models, "9AAA", self.out)
        base = os.path.join(self.out, "9AAA")
        with open(os.path.join(base, "9aaa.fasta")) as f:
            ids = [re.match(r">([^|]+)\|", ln).group(1) for ln in f if ln.startswith(">")]
        first = ids[0]
        for k in (1, 2):
            fname = "fold_9aaa_%d_model_0.cif" % k
            chain_index = fname.split("_")[2]
            protein_id = "%s_%s" % (first.split("_")[0], chain_index)
            self.assertIn(protein_id, ids)
        chains = "Chains A, B".replace("Chains ", "").strip().split(", ")
        self.assertEqual(len(chains), 2)

    def test_missing_model_is_an_error(self):
        os.remove(os.path.join(self.models, "fold_9aaa_e2_model_0.cif"))
        with self.assertRaises(ValueError) as cm:
            mi.build(self.jobs, self.models, "9AAA", self.out)
        self.assertIn("fold_9aaa_e2_model_0.cif", str(cm.exception))

    def test_unknown_entry_and_gap_in_entities(self):
        with self.assertRaises(ValueError):
            mi.read_jobs(self.jobs, "9ZZZ")
        with open(self.jobs, "a", newline="") as f:
            f.write("4,9CCC_E2,9CCC,E2,le3.0,4,5,1,MKVLA,\n")           # entity 2 without entity 1
        with self.assertRaises(ValueError):
            mi.read_jobs(self.jobs, "9CCC")

    def test_the_word_results_before_af3_results_is_refused(self):
        bad = os.path.join(self.tmp, "my_results", "inputs")
        with self.assertRaises(ValueError) as cm:
            mi.build(self.jobs, self.models, "9AAA", bad)
        self.assertIn("results", str(cm.exception))
        self.assertFalse(os.path.exists(bad))

    def test_too_many_chains_are_refused(self):
        rows = mi.read_jobs(self.jobs, "9BBB")
        rows[0]["copies_in_entry"] = "63"
        with self.assertRaises(ValueError):
            mi.mica_fasta(rows)

    def test_the_cli_returns_2_on_an_error(self):
        self.assertEqual(mi.main(["--jobs", self.jobs, "--models", self.models, "--pdb", "9ZZZ", "--out", self.out]), 2)

    def test_real_entries(self):
        """26DK has one sequence, 9OC0 two sequences with two copies each (data/af3/af3_jobs.csv)."""
        r26 = mi.read_jobs(REAL_JOBS, "26DK")
        self.assertEqual(mi.mica_fasta(r26).splitlines()[0], ">26DK_1|Chains A|26DK sequence 1|")
        r9 = mi.read_jobs(REAL_JOBS, "9OC0")
        self.assertEqual([ln for ln in mi.mica_fasta(r9).splitlines() if ln.startswith(">")],
                         [">9OC0_1|Chains A, B|9OC0 sequence 1|", ">9OC0_2|Chains C, D|9OC0 sequence 2|"])
        with open(REAL_JOBS, newline="") as f:
            pdbs = sorted({r["pdb_id"] for r in csv.DictReader(f)})
        self.assertEqual(len(pdbs), 20)
        for pdb in pdbs:                                                 # every entry gives a valid header set
            text = mi.mica_fasta(mi.read_jobs(REAL_JOBS, pdb))
            heads = [ln for ln in text.splitlines() if ln.startswith(">")]
            self.assertTrue(all(re.match(r">%s_\d+\|Chains [A-Za-z0-9, ]+\|" % pdb, h) for h in heads), pdb)


if __name__ == "__main__":
    unittest.main()
