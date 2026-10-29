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

    @unittest.skipUnless(os.path.isdir(os.path.join(ROOT, "results", "raw")) and shutil.which("bash"), "results/raw/ and bash are needed")
    def test_the_committed_tables_regenerate_identically(self):
        r = subprocess.run(["bash", SCRIPT, "--check"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("regenerate identically", r.stdout)


if __name__ == "__main__":
    unittest.main()
