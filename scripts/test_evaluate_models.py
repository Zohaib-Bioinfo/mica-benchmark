"""Tests for evaluate_models.py. Run from the scripts folder:  python -W error -m unittest test_evaluate_models

The tests that start the real US-align program run only when it is found: put its path into the environment variable USALIGN
or on the PATH (build: git clone https://github.com/pylelab/USalign; g++ -O3 -ffast-math -lm -o USalign USalign.cpp)."""
import os
import shutil
import tempfile
import unittest

import evaluate_models as ev

USALIGN = os.environ.get("USALIGN") or shutil.which("USalign")
HEADER = "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali\n"
ROW = "a.pdb:A\tb.cif:A\t0.3171\t0.1014\t5.66\t0.025\t0.006\t0.046\t197\t847\t109\n"
THREE = {"A": "ALA", "G": "GLY", "K": "LYS", "L": "LEU", "M": "MET", "V": "VAL"}


def pdb_text(seq, shift=0.0):
    """A straight chain of C-alpha atoms with 3.8 A steps in a gentle helix, as a PDB file."""
    import math
    lines = []
    for i, aa in enumerate(seq, 1):
        x, y, z = 2.3 * math.cos(i * 1.745) + shift, 2.3 * math.sin(i * 1.745), 1.5 * i
        lines.append("ATOM  %5d  CA  %s A%4d    %8.3f%8.3f%8.3f  1.00 50.00           C" % (i, THREE[aa], i, x, y, z))
    return "\n".join(lines) + "\nEND\n"


class ParseTests(unittest.TestCase):
    def test_one_row(self):
        rows = ev.parse_usalign_tabular(HEADER + ROW)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["TM1"], r["TM2"], r["RMSD"], r["IDali"]), (0.3171, 0.1014, 5.66, 0.046))
        self.assertEqual((r["L1"], r["L2"], r["Lali"]), (197, 847, 109))

    def test_empty_and_comment_lines_are_skipped(self):
        self.assertEqual(ev.parse_usalign_tabular("\n" + HEADER + "\n"), [])

    def test_wrong_number_of_fields_is_an_error(self):
        with self.assertRaises(ValueError):
            ev.parse_usalign_tabular(HEADER + "a\tb\t0.1\n")

    def test_missing_program_gives_no_alignment_not_a_crash(self):
        res = ev.evaluate("/nonexistent/USalign", "a.pdb", "b.pdb", 10)
        self.assertEqual(res["status"], "no_alignment")
        self.assertIn("could not be started", res["note"])
        self.assertEqual(res["tm_score"], "")


@unittest.skipUnless(USALIGN, "no USALIGN environment variable and no USalign on the PATH")
class RealUsalignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="evtest_")
        self.addCleanup(shutil.rmtree, self.tmp)

    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_identical_models_give_tm_score_1_and_full_coverage(self):
        seq = "MKVLAGGKVLAMKVLAGGKVLAMKVLAGGKVLA"
        a, b = self.write("a.pdb", pdb_text(seq)), self.write("b.pdb", pdb_text(seq))
        res = ev.evaluate(USALIGN, a, b, len(seq))
        self.assertEqual(res["status"], "ok", res["note"])
        self.assertAlmostEqual(res["tm_score"], 1.0, places=3)
        self.assertAlmostEqual(res["ca_rmsd"], 0.0, places=2)
        self.assertEqual(res["aligned_ca_length"], len(seq))
        self.assertAlmostEqual(res["seq_identity"], 1.0, places=3)
        self.assertEqual(res["coverage"], 1.0)

    def test_a_shorter_prediction_is_penalised_by_the_normalisation_and_by_coverage(self):
        seq = "MKVLAGGKVLAMKVLAGGKVLAMKVLAGGKVLA"                  # 33 residues in the deposited model
        native, pred = self.write("n.pdb", pdb_text(seq)), self.write("p.pdb", pdb_text(seq[:22]))
        res = ev.evaluate(USALIGN, pred, native, len(seq))
        self.assertEqual(res["status"], "ok", res["note"])
        self.assertEqual(res["pred_residues"], 22)
        self.assertEqual(res["native_residues"], 33)
        self.assertEqual(res["coverage"], round(22 / 33.0, 4))
        self.assertLess(res["tm_score"], res["tm_score_pred_norm"])   # TM2 (by the longer deposited model) is the smaller one
        self.assertAlmostEqual(res["tm_score_pred_norm"], 1.0, places=2)

    def test_an_empty_prediction_is_reported_not_raised(self):
        native, pred = self.write("n.pdb", pdb_text("MKVLAGGKVLAMKVLA")), self.write("p.pdb", "END\n")
        res = ev.evaluate(USALIGN, pred, native, 16)
        self.assertEqual(res["status"], "no_alignment")
        self.assertEqual(res["coverage"], "")


if __name__ == "__main__":
    unittest.main()
