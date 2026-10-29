#!/usr/bin/env python3
"""Tests for score_models.py (protocol v1.0 of LOG.md, 2026-10-24).

The end-to-end tests need the real US-align program (`USALIGN` environment variable, or `USalign` on the path) and skip without it.
Phenix is replaced by a stand-in `phenix.chain_comparison` (a Python program that prints the table of the real program, computed from the two files).

    USALIGN=/path/to/USalign python -W error -m unittest test_score_models
"""
from __future__ import annotations

import csv
import glob
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_models as sm  # noqa: E402

USALIGN = os.environ.get("USALIGN") or shutil.which("USalign") or shutil.which("USalign.exe")
REAL_RESULTS = os.environ.get("MICA_TEST_GB1", "")

AA = "ARNDCQEGHILKMFPSTWYV"
THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
         "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}

REAL_ZERO_OUTPUT = """\
Validating inputs
Using /kaggle/working/pilot/native/26DK.cif as target

Testing unique_part_of_target_only as True and False and
reporting results for whichever gives higher fraction matched.
Percent close with unique_part_of_target_only=True:     0.0
Percent close with unique_part_of_target_only=False:     0.0

Original residues in target: 191  In query: 113
Used residues in target:  191  In query: 113
CLOSE is within  3.0 A.

               ----ALL RESIDUES---  CLOSE RESIDUES ONLY    %
     MODEL     --CLOSE-    --FAR-- FORWARD REVERSE MIXED FOUND  CA                  SEQ
               RMSD   N      N       N       N      N         SCORE  SEQ MATCH(%)  SCORE  MEAN LENGTH  FRAGMENTS BAD CONNECTIONS

 Unique_target 0.00    0    113      0       0       0    0.0   0.00     0.0        0.00    0.0          1               0

Finished
"""

STUB_PY = r'''
import sys, numpy as np
from scipy.spatial import cKDTree
def read(path):
    xyz, names = [], []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54]))); names.append(line[17:20])
    return np.array(xyz), names
t, tn = read(sys.argv[1]); q, qn = read(sys.argv[2])
d, idx = cKDTree(t).query(q)
close = d <= 3.0
dq, _ = cKDTree(q).query(t)
nclose = int(close.sum()); nfar = len(q) - nclose
same = sum(1 for i in np.nonzero(close)[0] if qn[i] == tn[idx[i]])
rmsd = float(np.sqrt(np.mean(d[close] ** 2))) if nclose else 0.0
found = 100.0 * float((dq <= 3.0).sum()) / len(t)
print("Used residues in target:  %d  In query: %d" % (len(t), len(q)))
print("               RMSD   N      N       N       N      N         SCORE  SEQ MATCH(%)  SCORE  MEAN LENGTH  FRAGMENTS BAD CONNECTIONS")
print("")
print(" Unique_target %.2f %4d %6d %6d %6d %6d %6.1f %6.2f %8.1f %8.2f %6.1f %10d %14d" % (rmsd, nclose, nfar, nclose, 0, 0, found, 0.0, 100.0 * same / max(nclose, 1), 0.0, 0.0, 1, 0))
'''


def helix_chain(n, seed):
    r = np.random.default_rng(seed)
    pts, pos = [], np.zeros(3)
    for seg in range(0, n, 20):
        length = min(20, n - seg)
        tt = np.arange(length)
        h = np.stack([2.3 * np.cos(tt * 100 / 180 * np.pi), 2.3 * np.sin(tt * 100 / 180 * np.pi), 1.5 * tt], 1)
        q = r.normal(size=4)
        q /= np.linalg.norm(q)
        a, b, c, d = q
        rot = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)], [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])
        seg_pts = h @ rot.T
        seg_pts += pos - seg_pts[0] + (rot @ np.array([0, 0, 3.8]) if seg else 0)
        pts.append(seg_pts)
        pos = seg_pts[-1]
    return np.concatenate(pts)[:n], "".join(r.choice(list(AA), n))


def rot_z(P, deg, shift=(0, 0, 0)):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return P @ R.T + np.array(shift)


def write_chains(path, chains, extra_lines=()):
    lines, i = [], 1
    for cid, (P, seq) in chains:
        for k, (x, s) in enumerate(zip(P, seq), 1):
            lines.append("ATOM  %5d  CA  %3s %1s%4d    %8.3f%8.3f%8.3f  1.00 20.00           C" % (i, THREE[s], cid, k, x[0], x[1], x[2]))
            i += 1
        lines.append("TER")
    lines.extend(extra_lines)
    lines.append("END")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class Fixture:
    """native = chains A (120) and B (80); predictions in several shapes."""

    def __init__(self, tmp):
        self.tmp = tmp
        rng = np.random.default_rng(11)
        self.A, self.sa = helix_chain(120, 1)
        self.B, self.sb = helix_chain(80, 2)
        self.B = self.B - self.B.mean(0) + self.A.mean(0) + np.array([25.0, 0, 0])
        self.ctr = self.A.mean(0) + np.array([60.0, 0, 0])
        noise = lambda P, s=0.8: P + rng.normal(scale=s, size=P.shape)
        cop = lambda P, deg: rot_z(P - self.ctr, deg) + self.ctr
        self.noise, self.cop = noise, cop
        self.native = os.path.join(tmp, "native.pdb")
        write_chains(self.native, [("A", (self.A, self.sa)), ("B", (self.B, self.sb))])
        self.same = os.path.join(tmp, "pred_same.pdb")
        write_chains(self.same, [("A", (noise(self.A), self.sa)), ("B", (noise(self.B), self.sb))])
        chains = []
        for k, deg in enumerate((0, 120, 240)):
            chains.append(("ABCDEF"[2 * k], (noise(cop(self.A, deg)), self.sa)))
            chains.append(("ABCDEF"[2 * k + 1], (noise(cop(self.B, deg)), self.sb)))
        self.copies3 = os.path.join(tmp, "pred_3copies.pdb")
        write_chains(self.copies3, chains)
        self.copy2 = os.path.join(tmp, "pred_copy2_only.pdb")
        write_chains(self.copy2, [("X", (noise(cop(self.A, 120)), self.sa)), ("Y", (noise(cop(self.B, 120)), self.sb))])
        both = np.concatenate([noise(self.A), noise(self.B)])
        seq = self.sa + self.sb
        pieces = [(both[i:i + 30], seq[i:i + 30]) for i in range(0, 200, 30)]
        np.random.default_rng(5).shuffle(pieces)
        self.fragments = os.path.join(tmp, "pred_fragments.pdb")
        write_chains(self.fragments, [("ABCDEFGH"[i], p) for i, p in enumerate(pieces)])


def make_phenix_env(tmp):
    stub = os.path.join(tmp, "fake_chain_comparison.py")
    with open(stub, "w", encoding="utf-8") as f:
        f.write(STUB_PY)
    env = os.path.join(tmp, "phenix_env.sh")
    with open(env, "w", encoding="utf-8") as f:
        f.write('phenix.chain_comparison() { %s %s "$@"; }\n' % (sys.executable, stub))
    return env


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="score_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_trace_survives_writing_and_reading(self):
        fx = Fixture(self.tmp)
        t = sm.read_trace(fx.native)
        self.assertEqual((t.n_chains, t.n_residues, [len(c["res"]) for c in t.chains]), (2, 200, [120, 80]))
        out = sm.write_pdb(t, os.path.join(self.tmp, "again.pdb"))
        t2 = sm.read_trace(out)
        self.assertEqual(t2.names(), t.names())
        self.assertLess(float(np.abs(t2.xyz() - t.xyz()).max()), 0.0006)
        self.assertEqual([c["name"] for c in t2.chains], ["A", "B"])
        self.assertEqual(t2.chains[0]["res"][0][1:], (1, " "))

    def test_only_amino_acids_with_a_calpha_count(self):
        path = os.path.join(self.tmp, "mixed.pdb")
        lines = ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
                 "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 20.00           C",
                 "ATOM      3  CA AGLY A   2       3.000   1.000   0.000  0.50 20.00           C",
                 "ATOM      4  CA BGLY A   2       3.100   1.100   0.000  0.50 20.00           C",
                 "ATOM      5  CA  UNK A   3       4.500   1.000   0.000  1.00 20.00           C",
                 "ATOM      6  N   SER A   4       6.000   1.000   0.000  1.00 20.00           N",
                 "HETATM    7 CA    CA A 101      10.000  10.000  10.000  1.00 20.00          CA",
                 "HETATM    8  O   HOH A 102      11.000  10.000  10.000  1.00 20.00           O",
                 "HETATM    9  CA  HEM A 103      12.000  10.000  10.000  1.00 20.00           C",
                 "ATOM     10  C1' DA  B   1       1.000   2.000   3.000  1.00 20.00           C",
                 "TER", "END"]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        t = sm.read_trace(path)
        self.assertEqual((t.n_chains, t.names()), (1, ["ALA", "GLY", "UNK"]))
        self.assertAlmostEqual(float(t.xyz()[1][0]), 3.0, places=3)           # the first alternative location

    def test_a_file_without_residues_or_a_broken_file_is_refused(self):
        empty = os.path.join(self.tmp, "empty.pdb")
        with open(empty, "w", encoding="utf-8") as f:
            f.write("END\n")
        try:
            self.assertEqual(sm.read_trace(empty).n_residues, 0)
        except (ValueError, RuntimeError):
            pass                                                # either is fine: score_model turns both into no_model
        broken = os.path.join(self.tmp, "broken.cif")
        with open(broken, "w", encoding="utf-8") as f:
            f.write("this is not a structure\n")
        try:
            self.assertEqual(sm.read_trace(broken).n_residues, 0)
        except (ValueError, RuntimeError):
            pass

    def test_chain_names_are_made_unique_when_they_are_not(self):
        t = sm.Trace([{"name": n, "res": [("ALA", 1, " "), ("GLY", 2, " ")], "xyz": np.array([[0, 0, 0], [3.8, 0, 0]], float) + 10 * i} for i, n in enumerate(["A", "A", "AA"])])
        out = sm.write_pdb(t, os.path.join(self.tmp, "x.pdb"))
        with open(out, encoding="utf-8") as f:
            ids = [line[21] for line in f if line.startswith("ATOM")]
        self.assertEqual("".join(ids), "AABBCC")
        with self.assertRaises(ValueError):
            sm.write_pdb(sm.Trace([{"name": "A", "res": [("ALA", 10000, " ")], "xyz": np.zeros((1, 3))}]), os.path.join(self.tmp, "y.pdb"))

    def test_the_rotation_matrix_of_usalign_is_read_and_applied(self):
        text = ("------ The rotation matrix to rotate Structure_1 to Structure_2 ------\nm               t[m]        u[m][0]        u[m][1]        u[m][2]\n"
                "0       1.0000000000   0.0000000000  -1.0000000000   0.0000000000\n1       2.0000000000   1.0000000000   0.0000000000   0.0000000000\n"
                "2       3.0000000000   0.0000000000   0.0000000000   1.0000000000\n\nCode for rotating Structure 1 from (x,y,z) to (X,Y,Z):\n")
        t, u = sm.parse_matrix(text)
        tr = sm.Trace([{"name": "A", "res": [("ALA", 1, " ")], "xyz": np.array([[1.0, 2.0, 3.0]])}]).transformed((t, u))
        self.assertEqual(tr.xyz().tolist(), [[-1.0, 3.0, 6.0]])
        with self.assertRaises(ValueError):
            sm.parse_matrix("nothing here")

    def test_chain_pairs_follow_the_order_of_the_two_lists(self):
        row = {"PDBchain1": "pred.pdb:C:A:B:D", "PDBchain2": "native.pdb:A:B::"}
        self.assertEqual(sm.chain_pairs(row), [("C", "A"), ("A", "B")])

    def test_the_real_output_of_chain_comparison_is_parsed(self):
        d = sm.parse_chain_comparison(REAL_ZERO_OUTPUT)
        self.assertEqual((d["close_n"], d["far_n"], d["found_pct"], d["seq_match_pct"], d["used_target"], d["used_query"], d["label"], d["fragments"]),
                         (0, 113, 0.0, 0.0, 191, 113, "Unique_target", 1))
        self.assertEqual(sm.cc_metrics(d, 0.59)["ca_match"], 0.0)
        row = " Unique_target 1.12  96     17      90       4       2   50.3   44.80     71.9     0.52   8.4          12               3"
        d = sm.parse_chain_comparison("Used residues in target:  191  In query: 113\n" + row + "\n")
        m = sm.cc_metrics(d, 113 / 191.0)
        self.assertEqual((d["close_n"], d["far_n"], d["found_pct"], d["seq_match_pct"], d["rmsd"], d["bad_connections"]), (96, 17, 50.3, 71.9, 1.12, 3))
        self.assertAlmostEqual(m["ca_match"], 100 * 96 / 113.0, places=3)
        self.assertAlmostEqual(m["ca_quality"], 100 * 96 / 191.0, places=2)         # precision * coverage = close residues / deposited residues
        with self.assertRaises(ValueError):
            sm.parse_chain_comparison("Sorry: something went wrong\n")

    def test_own_counts(self):
        t = sm.Trace([{"name": "A", "res": [("ALA", i + 1, " ") for i in range(4)], "xyz": np.array([[3.8 * i, 0, 0] for i in range(4)], float)}])
        q = sm.Trace([{"name": "A", "res": [("ALA", 1, " "), ("GLY", 2, " "), ("ALA", 3, " ")], "xyz": np.array([[0.5, 0, 0], [3.8, 1.0, 0], [30, 0, 0]], float)}])
        o = sm.own_stats(q, t)
        self.assertEqual((o["ca_match"], o["ca_found"], o["seq_match"]), (round(100 * 2 / 3.0, 3), 50.0, 50.0))
        self.assertEqual(sm.own_stats(sm.Trace([]), t)["ca_match"], 0.0)

    def test_model_kinds_and_primary_files(self):
        k = sm.model_kind
        self.assertEqual([k("modelangelo", "9T5K_modelangelo.cif"), k("modelangelo", "9T5K_modelangelo_raw.cif"), k("e3", "9T5K_e3_prealign.pdb"), k("emprot", "output.cif"),
                          k("emprot", "output_denovo.cif"), k("emodelx", "9t5k_EModelX(+AF)_all_atom_model.pdb"), k("emodelx", "9t5k_EModelX(+AF)_ca_model.pdb"),
                          k("mica", "EMD-1_9t5k_MICA_all_atom_model.pdb"), k("mica", "EMD-1_9t5k_MICA_ca_model.pdb"), k("mica", "EMD-1_9t5k_MICA_all_atom_model_real_space_refined_000.pdb")],
                         ["modelangelo", "modelangelo_raw", "e3_prealign", "emprot", "emprot_denovo", "emodelx_af_all_atom", "emodelx_af_ca", "mica_all_atom", "mica_ca", "mica_refined"])
        self.assertEqual(sorted(sm.PRIMARY_KINDS), ["e3_prealign", "emodelx_af_all_atom", "emprot", "mica_all_atom", "modelangelo"])

    def test_models_are_collected_from_result_folders_and_the_last_folder_wins(self):
        for root, entry, files in (("gb1", "9T5K", ["9T5K_e3_prealign.pdb"]), ("gb1", "26DK", ["26DK_e3_prealign.pdb"]), ("gb2", "9T5K", ["9T5K_e3_prealign.pdb"])):
            d = os.path.join(self.tmp, root, "models", "e3", entry)
            os.makedirs(d, exist_ok=True)
            for f in files:
                open(os.path.join(d, f), "w", encoding="utf-8").close()
        d = os.path.join(self.tmp, "gb2", "models", "mica", "9t5k")
        os.makedirs(d)
        for f in ("a_9t5k_MICA_all_atom_model.pdb", "a_9t5k_MICA_ca_model.pdb", "notes.txt"):
            open(os.path.join(d, f), "w", encoding="utf-8").close()
        models, warnings = sm.collect_models([os.path.join(self.tmp, "gb1"), os.path.join(self.tmp, "gb2")])
        self.assertEqual([(m["entry"], m["method"], m["model_kind"], m["source"], m["primary"]) for m in models],
                         [("26DK", "e3", "e3_prealign", "gb1", True), ("9T5K", "e3", "e3_prealign", "gb2", True), ("9T5K", "mica", "mica_all_atom", "gb2", True), ("9T5K", "mica", "mica_ca", "gb2", False)])
        self.assertEqual(len(warnings), 1)


@unittest.skipUnless(USALIGN, "US-align is not available (set USALIGN)")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="score_e2e_")
        cls.fx = Fixture(cls.tmp)
        cls.env = make_phenix_env(cls.tmp)
        cls.native = sm.read_trace(cls.fx.native)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def score(self, pred, phenix=True, tag=None):
        wd = os.path.join(self.tmp, "w_" + (tag or os.path.basename(pred)))
        return sm.score_model(USALIGN, self.env if phenix else None, self.native, pred, wd)

    def test_a_model_that_is_the_deposited_model_with_noise(self):
        r = self.score(self.fx.same)
        self.assertEqual((r["status"], r["pred_residues"], r["native_residues"], r["coverage"], r["n_chains_pred"], r["n_chains_paired"]), ("ok", 200, 200, 1.0, 2, 2))
        self.assertGreater(r["tm_score"], 0.9)
        self.assertEqual((r["aligned_ca_length"], r["seq_identity"]), (200, 1.0))
        self.assertGreater(r["tm_score_fns"], 0.9)
        self.assertEqual((r["cc_status"], r["cc_status_ff"]), ("ok", "ok"))
        self.assertGreater(r["ca_match"], 99.0)
        self.assertGreater(r["seq_match"], 95.0)                # a close residue can be nearer to its neighbour in the helix than to itself
        self.assertAlmostEqual(r["ca_quality"], r["ca_match"] * r["coverage"], places=2)
        self.assertAlmostEqual(r["ca_match"], r["own_ca_match"], places=1)       # the stand-in and the own count compute the same thing
        self.assertAlmostEqual(r["ca_found"], r["own_ca_found"], delta=0.2)

    def test_extra_copies_do_not_lower_tm_score_or_the_quality_score(self):
        one, three = self.score(self.fx.same), self.score(self.fx.copies3)
        self.assertEqual((three["pred_residues"], three["coverage"], three["n_chains_pred"], three["n_chains_paired"]), (600, 3.0, 6, 2))
        self.assertAlmostEqual(three["tm_score"], one["tm_score"], delta=0.02)                 # normalised by the deposited model
        self.assertLess(three["tm_score_pred_norm"], 0.4)                                      # normalised by the prediction: penalised
        self.assertLess(three["ca_match"], 40.0)                                               # precision: two thirds of the residues belong to copies that are not deposited
        self.assertGreater(three["ca_quality"], 95.0)                                          # precision * coverage: the copies cancel
        self.assertGreater(three["ca_found"], 95.0)

    def test_a_copy_elsewhere_in_the_map_is_scored_after_superposition(self):
        r = self.score(self.fx.copy2)
        self.assertGreater(r["tm_score"], 0.9)
        self.assertGreater(r["ca_match"], 95.0)                  # superposed onto the deposited model
        self.assertLess(r["ca_match_ff"], 5.0)                   # in its own frame the copy is far from the deposited one
        self.assertLess(r["own_ca_match_ff"], 5.0)

    def test_a_model_cut_into_fragments_is_penalised_by_the_chain_wise_score_and_not_by_the_fragment_agnostic_one(self):
        r = self.score(self.fx.fragments)
        self.assertEqual((r["n_chains_pred"], r["n_chains_paired"]), (7, 2))               # one fragment for each of the two deposited chains
        self.assertLess(r["tm_score"], 0.35)
        self.assertGreater(r["tm_score_fns"], 0.9)
        self.assertGreater(r["ca_match"], 95.0)                  # Phenix does not care about chains

    def test_without_phenix_only_the_own_counts_are_there(self):
        r = self.score(self.fx.same, phenix=False, tag="nophenix")
        self.assertEqual((r["status"], r["cc_status"], r["cc_status_ff"], r["ca_match"]), ("ok", "not run", "not run", ""))
        self.assertGreater(r["own_ca_match"], 99.0)
        self.assertGreater(r["tm_score"], 0.9)

    def test_a_failing_phenix_is_a_note_and_not_a_lost_model(self):
        env = os.path.join(self.tmp, "failing_env.sh")
        with open(env, "w", encoding="utf-8") as f:
            f.write('phenix.chain_comparison() { echo "Sorry: no license"; return 3; }\n')
        r = sm.score_model(USALIGN, env, self.native, self.fx.same, os.path.join(self.tmp, "w_fail"))
        self.assertEqual((r["status"], r["cc_status"], r["cc_status_ff"], r["ca_match"]), ("ok", "failed", "failed", ""))
        self.assertIn("exit code 3", r["note"])
        self.assertGreater(r["tm_score"], 0.9)

    def test_a_phenix_table_that_cannot_be_read_is_a_failure_of_that_part_only(self):
        env = os.path.join(self.tmp, "odd_env.sh")
        with open(env, "w", encoding="utf-8") as f:
            f.write('phenix.chain_comparison() { echo "a table with another layout"; return 0; }\n')
        r = sm.score_model(USALIGN, env, self.native, self.fx.same, os.path.join(self.tmp, "w_odd"))
        self.assertEqual((r["status"], r["cc_status"]), ("ok", "failed"))
        self.assertIn("no result row", r["note"])

    def test_a_broken_us_align_is_no_alignment_and_a_broken_file_is_no_model(self):
        r = sm.score_model(os.path.join(self.tmp, "no_such_program"), self.env, self.native, self.fx.same, os.path.join(self.tmp, "w_noprog"))
        self.assertEqual((r["status"], r["tm_score"], r["tm_score_fns"]), ("no_alignment", "", ""))
        self.assertIn("could not be started", r["note"])
        self.assertGreater(r["own_ca_match_ff"], 99.0)           # what does not need US-align is still there
        broken = os.path.join(self.tmp, "broken_model.pdb")
        with open(broken, "w", encoding="utf-8") as f:
            f.write("not a structure\n")
        r = self.score(broken, tag="broken")
        self.assertEqual((r["status"], r["tm_score"]), ("no_model", ""))
        self.assertTrue(r["note"])

    def test_a_us_align_that_runs_out_of_time_is_no_alignment(self):
        slow = os.path.join(self.tmp, "slow_usalign.sh")
        with open(slow, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\nsleep 30\n")
        os.chmod(slow, os.stat(slow).st_mode | stat.S_IEXEC)
        old = sm.USALIGN_TIMEOUT
        try:
            import time
            wd = os.path.join(self.tmp, "w_slow")
            os.makedirs(wd, exist_ok=True)
            t0 = time.time()
            with self.assertRaises(RuntimeError) as cm:
                sm.run_usalign(slow, self.fx.same, self.fx.native, 1, 1, wd, "mm1", timeout=2)
            self.assertIn("did not finish within 2 seconds", str(cm.exception))
            self.assertLess(time.time() - t0, 20)
        finally:
            sm.USALIGN_TIMEOUT = old

    def test_score_all_writes_one_row_per_model_with_the_columns_of_the_protocol(self):
        root = os.path.join(self.tmp, "result1")
        for method, entry, name, src in (("e3", "9ZZZ", "9ZZZ_e3_prealign.pdb", self.fx.same), ("modelangelo", "9ZZZ", "9ZZZ_modelangelo.pdb", self.fx.fragments),
                                         ("modelangelo", "9ZZZ", "9ZZZ_modelangelo_raw.pdb", self.fx.fragments), ("emprot", "9ZZZ", "output.pdb", self.fx.copies3)):
            d = os.path.join(root, "models", method, entry)
            os.makedirs(d, exist_ok=True)
            shutil.copy(src, os.path.join(d, name))
        os.makedirs(os.path.join(self.tmp, "cifs"), exist_ok=True)
        shutil.copy(self.fx.native, os.path.join(self.tmp, "cifs", "9ZZZ.pdb"))
        natives = sm.prepare_natives([{"pdb_id": "9ZZZ", "modeled_residues": "200"}], os.path.join(self.tmp, "cifs"), os.path.join(self.tmp, "nat"))
        self.assertEqual(natives["9ZZZ"][2], "")
        models, warnings = sm.collect_models([root])
        self.assertEqual((len(models), warnings), (4, []))
        lines = []
        rows = sm.score_all(models, natives, USALIGN, self.env, os.path.join(self.tmp, "scores"), workers=3, log=lines.append)
        self.assertEqual(len(lines), 4)
        with open(os.path.join(self.tmp, "scores", "scores_long.csv"), newline="", encoding="utf-8") as f:
            got = list(csv.DictReader(f))
        self.assertEqual(list(got[0].keys()), sm.SCORE_COLUMNS)
        self.assertEqual([(g["method"], g["model_kind"], g["primary"], g["status"]) for g in got],
                         [("e3", "e3_prealign", "True", "ok"), ("emprot", "emprot", "True", "ok"), ("modelangelo", "modelangelo", "True", "ok"), ("modelangelo", "modelangelo_raw", "False", "ok")])
        self.assertTrue(all(g["entry"] == "9ZZZ" and g["source"] == "result1" for g in got))
        only = sm.score_all(models, natives, USALIGN, None, os.path.join(self.tmp, "scores2"), workers=1, log=lambda s: None, only_primary=True)
        self.assertEqual(len(only), 3)
        # a native whose residue count differs from the entries table is said in the note of every one of its models
        n2 = sm.prepare_natives([{"pdb_id": "9ZZZ", "modeled_residues": "190"}], os.path.join(self.tmp, "cifs"), os.path.join(self.tmp, "nat2"))
        self.assertIn("entries table says 190", n2["9ZZZ"][2])


@unittest.skipUnless(USALIGN and REAL_RESULTS and os.path.isdir(REAL_RESULTS), "the real models of gpu_batch1 are not available (set MICA_TEST_GB1)")
class RealModelsTests(unittest.TestCase):
    """The formats of the real models: ModelAngelo (mmCIF, many chains), EMProt (mmCIF, one long chain with many copies), EModelX (pdb), E3-CryoFold (pdb)."""

    def test_every_real_format_can_be_scored_against_a_stand_in_for_the_deposited_model(self):
        tmp = tempfile.mkdtemp(prefix="score_real_")
        try:
            env = make_phenix_env(tmp)
            models, _ = sm.collect_models([REAL_RESULTS])
            self.assertGreater(len(models), 50)
            picks = [m for m in models if m["entry"] in ("9T5K", "9TKA") and m["method"] in ("modelangelo", "emprot", "emodelx", "e3")]
            self.assertGreaterEqual(len(picks), 8)
            for entry in ("9T5K", "9TKA"):
                src = [m for m in models if m["entry"] == entry and m["model_kind"] == "emodelx_af_all_atom"][0]["path"]
                native = sm.read_trace(src)
                sm.write_pdb(native, os.path.join(tmp, entry + "_native.pdb"))
                for m in [p for p in picks if p["entry"] == entry]:
                    r = sm.score_model(USALIGN, env, native, m["path"], os.path.join(tmp, "w_%s_%s" % (entry, m["model_kind"])), native_file=os.path.join(tmp, entry + "_native.pdb"))
                    self.assertEqual(r["status"], "ok", (entry, m["model_kind"], r["note"]))
                    self.assertEqual((r["cc_status"], r["cc_status_ff"]), ("ok", "ok"), r["note"])
                    self.assertLess(abs(r["us_residues_pred"] - r["pred_residues"]), 0.1 * r["pred_residues"], (entry, m["model_kind"], r["note"]))   # US-align skips chains of 2 residues
                    self.assertTrue(r["tm_score_fns"] != "", (entry, m["model_kind"]))
                    self.assertGreaterEqual(r["tm_score_fns"] + 1e-9, 0.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
