"""Smoke test of make_figures.py: every figure is drawn from the tables of analyze_results.py on synthetic data (python3 -W error -m unittest test_make_figures)."""
import os
import tempfile
import unittest

import analyze_results as A
import make_figures as F
import test_analyze_results as T


class FigureTests(unittest.TestCase):
    def test_all_figures_are_drawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = T.Fixture(tmp)
            out = os.path.join(tmp, "results")
            self.assertEqual(A.main(["--entries", fx.entries, "--status"] + fx.status + ["--scores"] + fx.scores + ["--out", out, "--resamples", "200"]), 0)
            figs = os.path.join(tmp, "figures")
            self.assertEqual(F.main(["--results", out, "--out", figs]), 0)
            names = ("fig1_tm_score_per_entry", "fig2_paired_differences", "fig3_tm_score_pairs", "fig4_distributions", "figA1_all_metrics", "fig5_outcomes", "fig6_runtime",
                     "figA2_fragmentation", "figA3_resolution", "figA4_paired_differences_fragment_agnostic")
            for n in names:
                for ext in (".pdf", ".png"):
                    path = os.path.join(figs, n + ext)
                    self.assertTrue(os.path.exists(path) and os.path.getsize(path) > 2000, path)

    def test_no_results_no_figures_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(F.main(["--results", os.path.join(tmp, "nothing"), "--out", os.path.join(tmp, "figures")]), 0)
            self.assertFalse(os.path.exists(os.path.join(tmp, "figures")) and os.listdir(os.path.join(tmp, "figures")))


if __name__ == "__main__":
    unittest.main()
