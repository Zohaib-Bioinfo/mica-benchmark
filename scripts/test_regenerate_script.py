#!/usr/bin/env python3
"""Tests for regenerate.sh (the one command that rebuilds the analysis, figures, tables and PDF from results/raw/).

    python -W error -m unittest test_regenerate_script

`test_the_committed_tables_regenerate_identically` runs `regenerate.sh --check` (about 10 seconds): it analyses results/raw/ twice into a temporary folder and compares every table
with the committed one. It needs bash, numpy, pandas and scipy and is skipped when results/raw/ is not there.
"""
import os
import re
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, "regenerate.sh")


class TestRegenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SCRIPT, encoding="utf-8") as f:
            cls.text = f.read()

    def test_the_script_is_valid_bash(self):
        if not shutil.which("bash"):
            self.skipTest("bash is not installed")
        r = subprocess.run(["bash", "-n", SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_it_stops_at_the_first_error_and_changes_to_the_repository_root(self):
        self.assertIn("set -euo pipefail", self.text)
        self.assertIn('cd "$(dirname "$0")/.."', self.text)

    def test_every_python_script_and_input_that_it_names_exists(self):
        names = set(re.findall(r"python3 (scripts/[\w]+\.py)", self.text))
        self.assertGreaterEqual(len(names), 5)
        for n in names:
            self.assertTrue(os.path.exists(os.path.join(ROOT, n)), n)
        self.assertTrue(os.path.exists(os.path.join(ROOT, "data", "entries_2026.csv")))

    def test_the_steps_run_in_the_order_analysis_figures_tables_pdf(self):
        order = [self.text.index(s) for s in ("analyze results/nodock", "make_figures.py", "make_report_tables.py", "build_report.py")]
        self.assertEqual(order, sorted(order))
        self.assertLess(self.text.index("analyze results "), order[0])

    def test_the_sensitivity_analysis_is_kept_apart_from_the_main_one(self):
        self.assertIn("grep -v mica_gpu_nodock", self.text)                     # the main analysis does not read the session without docked input
        self.assertIn("--out results/nodock", self.text.replace('analyze results/nodock "$METHOD_STATUS $NODOCK_STATUS" "$SCORES_NODOCK"', "--out results/nodock"))

    @unittest.skipUnless(shutil.which("bash"), "bash is not installed")
    def test_a_scoring_session_without_scores_is_left_out(self):
        """scoring_cryozeta_d stopped on 7 November 2026 because batch d had built no model: its folder has no scoring/scores_long.csv. The list of score files must not name it
        (the analysis stopped with FileNotFoundError), and a folder with a NOT_A_RESULT.txt is left out as well, like the status tables of ingest_results.py --list."""
        import tempfile
        funcs = re.search(r"^scoring_folders\(\) \{.*?^scores\(\) \{.*?^\}", self.text, re.S | re.M)
        self.assertIsNotNone(funcs)
        with tempfile.TemporaryDirectory() as tmp:
            def make(name, scores=True, note=False):
                os.makedirs(os.path.join(tmp, name, "scoring"))
                if scores:
                    open(os.path.join(tmp, name, "scoring", "scores_long.csv"), "w").close()
                if note:
                    open(os.path.join(tmp, name, "NOT_A_RESULT.txt"), "w").close()
            make("scoring_a_result_20261101-1000")
            make("scoring_b_result_20261102-1000_INCOMPLETE", scores=False)
            make("scoring_c_result_20261103-1000", note=True)
            make("scoring_nodock_result_20261104-1000")
            make("scoring_cryo_result_20261030-1000")
            script = "set -euo pipefail\nRAW=%s\n%s\n" % (tmp, funcs.group(0))
            run = lambda *flt: subprocess.run(["bash", "-c", script + "scores " + " ".join(flt)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            main, nodock = run("grep", "-v", "nodock"), run("grep", "-E", "'scoring_nodock|scoring_cryo'")
            self.assertEqual(main.returncode, 0, main.stderr)
            self.assertEqual([os.path.relpath(x, tmp).split("/")[0] for x in main.stdout.split()], ["scoring_cryo_result_20261030-1000", "scoring_a_result_20261101-1000"])    # oldest first
            self.assertEqual(nodock.returncode, 0, nodock.stderr)
            self.assertEqual([os.path.relpath(x, tmp).split("/")[0] for x in nodock.stdout.split()], ["scoring_cryo_result_20261030-1000", "scoring_nodock_result_20261104-1000"])
            for x in main.stdout.split() + nodock.stdout.split():
                self.assertTrue(x.endswith("/scoring/scores_long.csv"), x)

    @unittest.skipUnless(os.path.isdir(os.path.join(ROOT, "results", "raw")) and shutil.which("bash"), "results/raw/ and bash are needed")
    def test_the_committed_tables_regenerate_identically(self):
        r = subprocess.run(["bash", SCRIPT, "--check"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("regenerate identically", r.stdout)


if __name__ == "__main__":
    unittest.main()
