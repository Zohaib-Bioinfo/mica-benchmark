#!/usr/bin/env python3
"""Offline tests for novelty_check.py: a local stand-in for the RCSB search and the pdb_seqres download.
The tests that search need MMseqs2 (MMSEQS environment variable or PATH) and are skipped without it.

    python scripts/test_novelty_check.py
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import http.server
import io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import novelty_check as nv  # noqa: E402

MMSEQS = os.environ.get("MMSEQS") or shutil.which("mmseqs")
AA = "ACDEFGHIKLMNPQRSTVWY"
RND = random.Random(11)


def rseq(n):
    return "".join(RND.choice(AA) for _ in range(n))


def mutate(s, frac):
    s = list(s)
    for i in RND.sample(range(len(s)), int(len(s) * frac)):
        s[i] = RND.choice([a for a in AA if a != s[i]])
    return "".join(s)


R1, R4 = rseq(300), rseq(280)
Q1 = mutate(R1, 0.05)                 # about 95% identical to R1, whole chain
Q2 = R1[:100] + rseq(200)             # 100% identical to R1 over a third of its length
Q3 = rseq(250)                        # unrelated
PRE2026 = {"1AAA_A": R1, "1AAA_B": R1, "2BBB_A": mutate(R1, 0.20), "3CCC_A": R1[:100], "4DDD_A": R4}
RELEASE = {"1AAA": "2005-01-01", "2BBB": "2012-05-05", "3CCC": "2019-09-09", "4DDD": "2025-12-30",
           "9NEW": "2026-02-03", "9QQ1": "2026-03-04", "9QQ2": "2026-05-06", "9XXX": "2026-07-07", "9YYY": "2026-10-01"}


class Mock:
    max_rows = None
    reject = False
    hidden = set()

    @classmethod
    def seqres(cls):
        lines = []
        for name, seq in PRE2026.items():
            pid, chain = name.split("_")
            lines += [">%s_%s mol:protein length:%d  PROTEIN" % (pid.lower(), chain, len(seq)), seq]
        lines += [">1aaa_C mol:na length:12  DNA", "ACGTACGTACGT", ">9new_A mol:protein length:%d  NEW" % len(Q1), Q1,
                  ">9xxx_A mol:protein length:%d  NEW" % len(Q3), Q3]
        return gzip.compress(("\n".join(lines) + "\n").encode())


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, Mock.seqres(), "application/gzip") if self.path == "/seqres.gz" else self.reply(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if Mock.reject:
            return self.reply(400, b"bad query")
        nodes = body["query"]["nodes"]
        a = next(n["parameters"]["value"] for n in nodes if n["parameters"]["operator"] == "greater_or_equal")
        b = next(n["parameters"]["value"] for n in nodes if n["parameters"]["operator"] == "less")
        ids = sorted(i for i, d in RELEASE.items() if a <= d < b and i not in Mock.hidden)
        total = len(ids)
        if Mock.max_rows:
            ids = ids[: Mock.max_rows]
        self.reply(200, json.dumps({"total_count": total, "result_set": [{"identifier": i, "score": 1.0} for i in ids]}).encode())


class NoveltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = "http://127.0.0.1:%d" % cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Mock.max_rows, Mock.reject, Mock.hidden = None, False, set()
        self.tmp = tempfile.mkdtemp(prefix="nov_test_")
        self.entries = os.path.join(self.tmp, "entries_2026.csv")
        with open(self.entries, "w", newline="") as f:
            f.write("emdb_id,pdb_id,resolution_A,max_id_to_pre2026_pdb,cc_mask\nEMD-1,9QQ1,2.5,,0.9\nEMD-2,9QQ2,3.5,,0.8\n")
        self.fasta = os.path.join(self.tmp, "seqs.fasta")
        with open(self.fasta, "w") as f:
            for pid, eid, seq in (("9QQ1", 1, Q1), ("9QQ1", 2, Q3), ("9QQ2", 1, Q2), ("9OTH", 1, R4)):
                f.write(">%s_E%d len=%d copies=1 longest=0\n%s\n" % (pid, eid, len(seq), seq))
        self.cryo = os.path.join(self.tmp, "cryo.txt")
        with open(self.cryo, "w") as f:
            f.write("9QQ1\n9QQ2\n9XXX\n")
        self.out = os.path.join(self.tmp, "out")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_nv(self, *extra, mmseqs=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = nv.main(["--entries", self.entries, "--fasta", self.fasta, "--out", self.out, "--search-url", self.base + "/search",
                            "--seqres-url", self.base + "/seqres.gz", "--through", "2026-10-13", "--min-reference-chains", "1",
                            "--mmseqs", mmseqs or MMSEQS or sys.executable, "--cryoem-ids", self.cryo] + list(extra))
        return code, out.getvalue(), err.getvalue()

    def rows(self, path):
        with open(path, newline="") as f:
            return {r["pdb_id"]: r for r in csv.DictReader(f)}

    # --- needs MMseqs2 -------------------------------------------------
    @unittest.skipUnless(MMSEQS, "no MMseqs2")
    def test_values_files_and_cleanup(self):
        code, out, err = self.run_nv()
        self.assertEqual(code, 0, err)
        nov = self.rows(os.path.join(self.out, "novelty.csv"))
        a, b = nov["9QQ1"], nov["9QQ2"]
        self.assertEqual(a["n_chain_seqs"], "2")
        self.assertGreaterEqual(float(a["max_id_to_pre2026_pdb"]), 0.93)
        self.assertLessEqual(float(a["max_id_to_pre2026_pdb"]), 0.97)
        self.assertEqual((a["best_hit"], a["best_hit_query"]), ("1AAA_A", "9QQ1_E1"))     # not the 2026 copy 9NEW_A
        self.assertEqual(a["best_hit_qcov"], "1.00")
        self.assertEqual(b["max_id_to_pre2026_pdb"], "0.000")                          # 100% over a third of the chain does not count
        self.assertEqual(b["best_hit"], "")
        self.assertEqual(b["max_id_any_coverage"], "1.000")
        self.assertIn(b["best_any_hit"], ("1AAA_A", "3CCC_A"))
        self.assertEqual(b["best_any_hit_qcov"], "0.33")
        ent = self.rows(self.entries)
        self.assertEqual(ent["9QQ1"]["max_id_to_pre2026_pdb"], a["max_id_to_pre2026_pdb"])
        self.assertEqual(ent["9QQ2"]["max_id_to_pre2026_pdb"], "0.000")
        self.assertEqual((ent["9QQ1"]["emdb_id"], ent["9QQ1"]["cc_mask"], ent["9QQ2"]["resolution_A"]), ("EMD-1", "0.9", "3.5"))
        with open(self.entries) as f:
            self.assertEqual(f.readline().strip(), "emdb_id,pdb_id,resolution_A,max_id_to_pre2026_pdb,cc_mask")
        with open(os.path.join(self.out, "novelty_info.json")) as f:
            info = json.load(f)
        self.assertEqual((info["n_excluded_entries"], info["n_queries"], info["n_entries"]), (5, 3, 2))
        self.assertEqual(info["entries_at_or_below_0.30"], 1)
        self.assertEqual(info["protein_chains_in_seqres"], 7)
        self.assertEqual(info["chains_of_excluded_entries"], 2)
        self.assertEqual(info["reference_unique_sequences"], 4)         # R1 (twice in 1AAA, once alone), mutated R1, fragment, R4
        self.assertEqual(sorted(os.listdir(self.out)), ["novelty.csv", "novelty_info.json"])
        self.assertIn("1 entries at or below 0.30", out)

    @unittest.skipUnless(MMSEQS, "no MMseqs2")
    def test_a_local_seqres_file_is_used_and_kept(self):
        local = os.path.join(self.tmp, "pdb_seqres.txt.gz")
        with open(local, "wb") as f:
            f.write(Mock.seqres())
        code, _, err = self.run_nv("--seqres", local, "--seqres-url", "http://127.0.0.1:1/none")
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.exists(local))

    # --- without MMseqs2 -----------------------------------------------
    def test_all_cryoem_entries_must_be_in_the_excluded_list(self):
        Mock.hidden = {"9XXX"}
        code, _, err = self.run_nv()
        self.assertEqual(code, 2)
        self.assertIn("missing from the excluded list", err)
        self.assertIn("9XXX", err)

    def test_selected_entries_must_be_in_the_excluded_list(self):
        RELEASE["9QQ2"] = "2025-11-11"
        try:
            code, _, err = self.run_nv("--cryoem-ids", os.devnull)
        finally:
            RELEASE["9QQ2"] = "2026-05-06"
        self.assertEqual(code, 2)
        self.assertIn("selected entries are not in the excluded list: 9QQ2", err)

    def test_a_small_reference_stops_the_run(self):
        code, _, err = self.run_nv("--min-reference-chains", "100000")
        self.assertEqual(code, 2)
        self.assertIn("only 4 distinct sequences", err)

    def test_a_rejected_search_is_reported(self):
        Mock.reject = True
        code, _, err = self.run_nv()
        self.assertEqual(code, 2)
        self.assertIn("did not accept the query", err)

    def test_missing_sequences_and_missing_program(self):
        with open(self.entries, "a") as f:
            f.write("EMD-3,9ZZZ,2.0,,0.9\n")
        code, _, err = self.run_nv()
        self.assertEqual(code, 2)
        self.assertIn("no sequences for 9ZZZ", err)
        code, _, err = self.run_nv(mmseqs="/nonexistent/mmseqs")
        self.assertEqual(code, 2)
        self.assertIn("mmseqs not found", err)

    # --- units ----------------------------------------------------------
    def test_a_truncated_window_is_split(self):
        Mock.max_rows = 2
        got = nv.released_since(self.base + "/search", dt.date(2026, 1, 1), dt.date(2026, 10, 13))
        self.assertEqual(got, {"9NEW", "9QQ1", "9QQ2", "9XXX", "9YYY"})
        got = nv.released_since(self.base + "/search", dt.date(2026, 2, 3), dt.date(2026, 2, 4))
        self.assertEqual(got, {"9NEW"})

    def test_reference_building(self):
        seqres = os.path.join(self.tmp, "s.gz")
        with open(seqres, "wb") as f:
            f.write(Mock.seqres())
        ref = os.path.join(self.tmp, "ref.fasta")
        stats = nv.build_reference(seqres, {"9NEW", "9XXX"}, ref)
        self.assertEqual(stats, {"protein_chains_in_seqres": 7, "chains_of_excluded_entries": 2, "reference_unique_sequences": 4})
        names = [ln[1:].strip() for ln in open(ref) if ln.startswith(">")]
        self.assertEqual(names, ["1aaa_A", "2bbb_A", "3ccc_A", "4ddd_A"])     # first name of a repeated sequence, no DNA

    def test_entry_values_follow_the_coverage_rule(self):
        queries = {"9QQ1_E1": "A", "9QQ1_E2": "A", "9QQ2_E1": "A", "9QQ3_E1": "A"}
        hits = {"9QQ1_E1": [{"target": "t1", "id": 0.40, "qcov": 0.9, "alnlen": 100, "evalue": 1e-9},
                            {"target": "t2", "id": 0.99, "qcov": 0.2, "alnlen": 20, "evalue": 1e-5}],
                "9QQ1_E2": [{"target": "t3", "id": 0.55, "qcov": 0.5, "alnlen": 100, "evalue": 1e-9}],
                "9QQ2_E1": [{"target": "t4", "id": 0.80, "qcov": 0.49, "alnlen": 100, "evalue": 1e-9}]}
        rows = {r["pdb_id"]: r for r in nv.entry_values(["9QQ1", "9QQ2", "9QQ3"], queries, hits)}
        self.assertEqual((rows["9QQ1"]["max_id_to_pre2026_pdb"], rows["9QQ1"]["best_hit"], rows["9QQ1"]["best_hit_query"]), ("0.550", "t3", "9QQ1_E2"))
        self.assertEqual((rows["9QQ1"]["max_id_any_coverage"], rows["9QQ1"]["best_any_hit"]), ("0.990", "t2"))
        self.assertEqual((rows["9QQ2"]["max_id_to_pre2026_pdb"], rows["9QQ2"]["max_id_any_coverage"]), ("0.000", "0.800"))
        self.assertEqual((rows["9QQ3"]["max_id_to_pre2026_pdb"], rows["9QQ3"]["max_id_any_coverage"], rows["9QQ3"]["best_hit"]), ("0.000", "0.000", ""))

    def test_identical_chains_of_an_entry_are_searched_once(self):
        allseq = {"9AAA_E1": "MKV", "9AAA_E2": "MKV", "9AAA_E3": "GGG", "9BBB_E1": "MKV"}
        self.assertEqual(nv.query_sequences(allseq, ["9AAA"]), {"9AAA_E1": "MKV", "9AAA_E3": "GGG"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
