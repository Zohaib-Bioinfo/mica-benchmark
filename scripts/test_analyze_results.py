"""Tests of analyze_results.py on synthetic data with known answers (no real result is used or needed).

Run:  python3 -W error -m unittest test_analyze_results
"""
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy import stats

import analyze_results as A

IDS = ["1A%02d" % i for i in range(1, 13)]
STRATA = ["le3.0"] * 6 + ["gt3.0_le4.0"] * 6
NOVELTY = [1.0, 0.0, 0.5, 1.0, 1.0, 0.9, 1.0, 0.2, 1.0, 1.0, 0.8, 1.0]       # 1A02 and 1A08 are at most 30%
OFFSET = {"mica": 0.0, "modelangelo": 0.10, "emodelx": 0.0, "emprot": -0.05, "e3": -0.20, "cryoatom1": -0.10, "cryozeta": 0.04}


def noise(i, k):
    return (((i + 1) * 37 + k * 13) % 11 - 5) / 500.0                          # between -0.01 and +0.01, no ties


def base_tm(i):
    return 0.50 + 0.03 * i


def entries_frame():
    return pd.DataFrame({"emdb_id": ["EMD-%d" % (50000 + i) for i in range(12)], "pdb_id": IDS, "stratum": STRATA, "resolution_A": [2.0 + 0.2 * i for i in range(12)],
                         "modeled_residues": [200 + 20 * i for i in range(12)], "max_id_to_pre2026_pdb": NOVELTY, "seed_rank": range(1, 13), "n_chains": [1] * 12})


def baseline_rows():
    """gpu_batch1 (first file) and gpu_batch2 (second file)."""
    cols = ["method", "entry", "status", "failed_step", "note", "attempt", "gpu", "started_utc", "seconds", "peak_gpu_mb", "min_ram_gb", "other_gpu_busy", "limit_s"]
    one, two = [], []
    for m in ("modelangelo", "emodelx", "emprot", "e3", "cryoatom1", "cryozeta"):
        for i, e in enumerate(IDS):
            row = [m, e, "ok", "", "", 1, i % 2, "2026-10-23T%02d:00:00Z" % (5 + i), 600 + 10 * i, 5000, 8.0, True, 10800]
            if m == "cryoatom1" and e == "1A07":                                  # the real session: four entries ran out of GPU memory (a result, not a crash)
                row[2:5] = ["failed", "build", "out of GPU memory (exit code 1; the error message of the program); the model file was not written"]
            if m == "e3" and e == "1A03":
                row[2:5] = ["failed", "prealign", "exit code 1; the model file was not written"]
            if m == "cryozeta" and e == "1A06":                                   # out of GPU memory in every attempt: a failure of the method
                row[2:5] = ["failed", "run", "out of GPU memory (exit code 2; the error message of the program); the model file was not written"]
            if m == "cryozeta" and e in ("1A10", "1A11", "1A12"):                 # the free GPU time ran out: not run is not a failure
                row[2:5] = ["not run", "", "not started: not enough time left in the session"]
                row[7], row[8] = "", ""
            if m == "emprot" and e == "1A05":
                row[2:5] = ["killed_corun", "build", "killed by the system (exit code -9)"]
            if (m, e) in (("e3", "1A12"), ("modelangelo", "1A12"), ("emprot", "1A12"), ("emodelx", "1A12")):
                row[2:5] = ["not run", "", ""]
                row[7], row[8] = "", ""
            one.append(row)
    two.append(["emprot", "1A05", "ok", "", "", 1, 0, "2026-10-24T08:00:00Z", 900, 5000, 8.0, False, 10800])
    two.append(["modelangelo", "1A12", "ok", "", "", 1, 0, "2026-10-24T09:00:00Z", 1200, 5000, 8.0, False, 10800])
    two.append(["emodelx", "1A12", "ok", "", "", 1, 0, "2026-10-24T10:00:00Z", 700, 5000, 8.0, False, 10800])
    two.append(["emprot", "1A12", "ok", "", "", 1, 0, "2026-10-24T11:00:00Z", 800, 5000, 8.0, False, 10800])
    return pd.DataFrame(one, columns=cols), pd.DataFrame(two, columns=cols)


def mica_rows():
    """mica_status.csv of a CPU session (first) and of the GPU session (second)."""
    cols = ["entry", "status", "failed_step", "note", "lane", "device", "windows", "p1_seconds", "dock_seconds", "p3_seconds", "seconds", "started_utc"]
    a, b = [], []
    for i, e in enumerate(IDS):
        row = [e, "ok", "", "", 0, "cpu", 100, 100, 1000, 500, 1700, "2026-10-24T%02d:00:00Z" % (6 + i)]
        if e == "1A02":
            row[1:4] = ["failed", "2_dock", "no domain could be docked"]
            row[8:11] = [900, "", 1000]
        if e == "1A11":
            row[1:4] = ["deferred", "", "program 3 needs a GPU"]
            row[9] = ""
            row[10] = 1100
        if e == "1A12":
            row[1:4] = ["not run", "time", "not started"]
            row[7:11] = ["", "", "", ""]
            row[11] = ""
        a.append(row)
    b.append(["1A11", "ok", "", "", 0, "cuda", 100, "", "", 30, 60, "2026-10-25T01:00:00Z"])
    return pd.DataFrame(a, columns=cols), pd.DataFrame(b, columns=cols)


def score_rows(drop=(), extra=()):
    """One primary row per method and entry that is ok, with TM-score = base + offset + noise."""
    cols = ["entry", "method", "model_kind", "primary", "model_file", "source", "status", "note", "pred_residues", "native_residues", "coverage", "tm_score", "tm_score_fns",
            "ca_rmsd", "aligned_ca_length", "seq_identity", "ca_match", "seq_match", "ca_quality", "ca_found", "ca_match_ff", "ca_quality_ff", "own_ca_match"]
    kinds = {"mica": "mica_all_atom", "modelangelo": "modelangelo", "emodelx": "emodelx_af_all_atom", "emprot": "emprot", "e3": "e3_prealign", "cryoatom1": "cryoatom1", "cryozeta": "cryozeta"}
    ok_entries = {"mica": [e for e in IDS if e not in ("1A02", "1A12")], "modelangelo": IDS, "emodelx": IDS, "emprot": IDS, "e3": [e for e in IDS if e not in ("1A03", "1A12")],
                  "cryoatom1": [e for e in IDS if e != "1A07"], "cryozeta": [e for e in IDS if e not in ("1A06", "1A10", "1A11", "1A12")]}
    rows = []
    for k, m in enumerate(A.METHODS):
        for i, e in enumerate(IDS):
            if e not in ok_entries[m] or (m, e) in drop:
                continue
            tm = round(base_tm(i) + OFFSET[m] + noise(i, k), 4)
            cov = 1.0 + (0.5 if (m == "emprot" and e == "1A04") else 0.0)
            rows.append([e, m, kinds[m], True, kinds[m] + ".pdb", "x", "ok", "", int(200 * cov), 200, cov, tm, min(1.0, tm + 0.05), 2.0, 180, 0.5, 100 * tm, 60 * tm,
                         100 * tm * cov, 100 * tm, 90 * tm, 90 * tm, 95 * tm])
    rows.append(["1A01", "mica", "mica_ca", False, "mica_ca.pdb", "x", "ok", "", 200, 200, 1.0, 0.1, 0.1, 2.0, 100, 0.5, 10, 10, 10, 10, 10, 10, 10])
    for r in extra:
        rows.append(r)
    df = pd.DataFrame(rows, columns=cols)
    return df


def write(df, path):
    df.to_csv(path, index=False)
    return path


class Fixture:
    def __init__(self, tmp, scores=None, status_overrides=None):
        self.tmp = tmp
        self.entries = write(entries_frame(), os.path.join(tmp, "entries.csv"))
        b1, b2 = baseline_rows()
        m1, m2 = mica_rows()
        os.makedirs(os.path.join(tmp, "gpu_batch1"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "gpu_batch2"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "mica_cpu_a"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "mica_gpu"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "scoring"), exist_ok=True)
        self.status = [write(b1, os.path.join(tmp, "gpu_batch1", "method_status.csv")), write(b2, os.path.join(tmp, "gpu_batch2", "method_status.csv")),
                       write(m1, os.path.join(tmp, "mica_cpu_a", "mica_status.csv")), write(m2, os.path.join(tmp, "mica_gpu", "mica_status.csv"))]
        self.scores = [write(score_rows() if scores is None else scores, os.path.join(tmp, "scoring", "scores_long.csv"))]

    def analyse(self, n_boot=300):
        return A.analyse(A.load_entries(self.entries), A.load_status(self.status), A.load_scores(self.scores), n_boot=n_boot, log=lambda *a: None)


class PureStatisticsTests(unittest.TestCase):
    def test_holm_known_example(self):
        adj = A.holm_adjust([0.01, 0.04, 0.03, 0.005])
        self.assertTrue(np.allclose(adj, [0.03, 0.06, 0.06, 0.02]))

    def test_holm_family_size_and_nan(self):
        adj = A.holm_adjust([0.01, float("nan"), 0.02, float("nan")], m=4)
        self.assertAlmostEqual(adj[0], 0.04)
        self.assertAlmostEqual(adj[2], 0.06)
        self.assertTrue(np.isnan(adj[1]) and np.isnan(adj[3]))
        self.assertEqual(A.holm_adjust([0.6, 0.9], m=2).tolist(), [1.0, 1.0])

    def test_paired_matches_scipy_and_sign(self):
        x = np.array([0.8, 0.7, 0.9, 0.65, 0.72, 0.88, 0.5, 0.6])
        y = np.array([0.6, 0.71, 0.5, 0.60, 0.70, 0.60, 0.5, 0.4])
        r = A.paired_stats(x, y, "k", "higher", n_boot=500)
        d = x - y
        self.assertEqual(r["n"], 8)
        self.assertAlmostEqual(r["mean_diff"], d.mean())
        self.assertGreater(r["mean_diff"], 0)                                      # method minus MICA
        nz = d[d != 0]
        self.assertEqual(r["n_nonzero"], 7)
        self.assertAlmostEqual(r["wilcoxon_p"], stats.wilcoxon(nz).pvalue)
        self.assertAlmostEqual(r["ttest_p"], stats.ttest_rel(x, y).pvalue)
        self.assertEqual((r["n_method_higher"], r["n_method_lower"], r["n_equal"]), (6, 1, 1))
        self.assertEqual((r["n_method_better"], r["n_method_worse"]), (6, 1))
        low = A.paired_stats(x, y, "k", "lower", n_boot=500)
        self.assertEqual((low["n_method_better"], low["n_method_worse"]), (1, 6))     # the direction of "better" flips the counts, not the difference
        self.assertAlmostEqual(low["mean_diff"], r["mean_diff"])
        ranks = stats.rankdata(np.abs(nz))
        self.assertAlmostEqual(r["rank_biserial"], (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
        self.assertAlmostEqual(r["cohens_dz"], d.mean() / d.std(ddof=1))
        self.assertLess(r["diff_ci_low"], r["mean_diff"])
        self.assertGreater(r["diff_ci_high"], r["mean_diff"])

    def test_all_positive_exact_p(self):
        x = np.arange(10) * 0.01 + 1.0
        r = A.paired_stats(x, np.zeros(10), "k", "higher", n_boot=200)
        self.assertAlmostEqual(r["wilcoxon_p"], 2 / 2 ** 10)                       # exact two-sided p of a sign pattern of ten pluses
        self.assertEqual(r["rank_biserial"], 1.0)

    def test_equal_samples_and_tiny_samples(self):
        r = A.paired_stats([0.5, 0.6, 0.7], [0.5, 0.6, 0.7], "k", "higher", n_boot=100)
        self.assertEqual(r["wilcoxon_p"], 1.0)
        self.assertTrue(math.isnan(r["ttest_p"]))
        self.assertEqual(r["n_equal"], 3)
        r = A.paired_stats([0.5], [0.4], "k", "higher", n_boot=100)
        self.assertEqual(r["n"], 1)
        self.assertTrue(math.isnan(r["diff_ci_low"]))
        r = A.paired_stats([], [], "k", "higher", n_boot=100)
        self.assertEqual(r["n"], 0)
        self.assertTrue(math.isnan(r["wilcoxon_p"]) and math.isnan(r["mean_diff"]))
        r = A.paired_stats([0.5, float("nan"), 0.7, 0.9], [0.4, 0.3, float("nan"), 0.5], "k", "higher", n_boot=100)
        self.assertEqual(r["n"], 2)                                                # pairs with a missing value are dropped

    def test_constant_difference_has_no_ttest(self):
        r = A.paired_stats([0.6, 0.7, 0.8], [0.5, 0.6, 0.7], "k", "higher", n_boot=100)
        self.assertTrue(math.isnan(r["ttest_p"]))
        self.assertAlmostEqual(r["mean_diff"], 0.1)

    def test_bootstrap_is_reproducible_and_depends_on_the_key(self):
        v = np.linspace(-0.2, 0.5, 15)
        a = A.bootstrap_ci(v, np.mean, "key one", n_boot=2000)
        self.assertEqual(a, A.bootstrap_ci(v, np.mean, "key one", n_boot=2000))
        self.assertNotEqual(a, A.bootstrap_ci(v, np.mean, "key two", n_boot=2000))
        self.assertLess(a[0], v.mean())
        self.assertGreater(a[1], v.mean())
        self.assertNotEqual(a, A.bootstrap_ci(v, np.mean, "key one", seed=7, n_boot=2000))
        self.assertTrue(all(math.isnan(x) for x in A.bootstrap_ci([1.0], np.mean, "k")))

    def test_bootstrap_ci_is_close_to_the_normal_interval(self):
        rng = np.random.default_rng(5)
        v = rng.normal(0.3, 0.1, size=40)
        lo, hi = A.bootstrap_ci(v, np.mean, "normal", n_boot=10000)
        se = v.std(ddof=1) / math.sqrt(len(v))
        self.assertAlmostEqual(lo, v.mean() - 1.96 * se, delta=0.01)
        self.assertAlmostEqual(hi, v.mean() + 1.96 * se, delta=0.01)

    def test_describe(self):
        d = A.describe([1.0, 2.0, 3.0, 4.0, float("nan")], "k", n_boot=300)
        self.assertEqual(d["n"], 4)
        self.assertAlmostEqual(d["mean"], 2.5)
        self.assertAlmostEqual(d["median"], 2.5)
        self.assertAlmostEqual(d["sd"], np.std([1, 2, 3, 4], ddof=1))
        self.assertTrue(math.isnan(A.describe([], "k")["mean"]))


class ReadingTests(unittest.TestCase):
    def test_scores_percent_and_later_file_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = score_rows()
            one = write(s, os.path.join(tmp, "a.csv"))
            s2 = s[(s["method"] == "e3") & (s["entry"] == "1A01")].copy()
            s2["tm_score"] = 0.123
            os.makedirs(os.path.join(tmp, "b"))
            two = write(s2, os.path.join(tmp, "b", "scores_long.csv"))
            out = A.load_scores([one, two])
            row = out[(out["method"] == "e3") & (out["entry"] == "1A01")]
            self.assertEqual(len(row), 1)
            self.assertAlmostEqual(float(row["tm_score"].iloc[0]), 0.123)
            self.assertAlmostEqual(float(out["seq_identity"].iloc[0]), 50.0)         # US-align's fraction becomes percent
            self.assertTrue(out["primary"].dtype == bool)
            self.assertEqual(int((~out["primary"]).sum()), 1)

    def test_status_files_are_told_apart_by_the_method_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            f = A.load_status(fx.status)
            self.assertEqual(set(f["method"]), set(A.METHODS))
            self.assertEqual(len(f[f["method"] == "mica"]), 13)
            self.assertTrue(set(f["file_order"]) == {0, 1, 2, 3})

    def test_groups(self):
        e = A.load_entries(write(entries_frame(), tempfile.mktemp(suffix=".csv")))
        g = A.groups_of(e)
        self.assertEqual(list(g), ["pooled", "le3.0", "gt3.0_le4.0", "novel_le30"])
        self.assertEqual(len(g["pooled"]), 12)
        self.assertEqual(g["le3.0"], IDS[:6])
        self.assertEqual(g["novel_le30"], ["1A02", "1A08"])


class MergeAndOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fx = Fixture(self.tmp.name)
        self.res = self.fx.analyse()

    def merged(self, method, entry):
        m = self.res["merged"]
        return m[(m["method"] == method) & (m["entry"] == entry)].iloc[0]

    def outcome(self, method, entry):
        o = self.res["outcomes"]
        return o[(o["method"] == method) & (o["entry"] == entry)].iloc[0]

    def test_a_later_result_replaces_a_run_that_was_not_a_result(self):
        m = self.merged("emprot", "1A05")
        self.assertEqual(m["status_raw"], "ok")
        self.assertEqual(m["history"], "killed_corun; ok")
        self.assertFalse(m["conflict"])
        self.assertEqual(self.merged("modelangelo", "1A12")["status_raw"], "ok")
        self.assertEqual(self.merged("e3", "1A12")["status_raw"], "not run")

    def test_mica_runtime_is_the_sum_of_the_stages_over_the_sessions(self):
        self.assertEqual(self.merged("mica", "1A01")["runtime_s"], 100 + 1000 + 500)
        self.assertEqual(self.merged("mica", "1A11")["runtime_s"], 100 + 1000 + 30)     # docking in the CPU session, program 3 in the GPU session
        self.assertEqual(self.merged("mica", "1A11")["status_raw"], "ok")
        self.assertEqual(self.merged("mica", "1A11")["history"], "deferred; ok")
        self.assertEqual(self.merged("e3", "1A01")["runtime_s"], 600)
        self.assertTrue(math.isnan(self.merged("e3", "1A12")["runtime_s"]))

    def test_outcomes(self):
        self.assertEqual(self.outcome("mica", "1A01")["outcome"], "model")
        self.assertEqual(self.outcome("mica", "1A02")["outcome"], "failed")
        self.assertIn("2_dock", self.outcome("mica", "1A02")["reason"])
        self.assertEqual(self.outcome("mica", "1A12")["outcome"], "not_run")
        self.assertEqual(self.outcome("e3", "1A03")["outcome"], "failed")
        self.assertIn("prealign", self.outcome("e3", "1A03")["reason"])
        self.assertEqual(self.outcome("e3", "1A12")["outcome"], "not_run")
        self.assertEqual(self.outcome("cryozeta", "1A06")["outcome"], "failed")                     # out of GPU memory in every attempt: a result of the method
        self.assertEqual(self.outcome("cryozeta", "1A10")["outcome"], "not_run")                    # no time left in the session: not a failure
        self.assertEqual(self.outcome("emprot", "1A05")["outcome"], "model")
        c = self.res["completion"]
        row = c[(c["group"] == "pooled") & (c["method"] == "mica")].iloc[0]
        self.assertEqual((row["n_model"], row["n_failed"], row["n_not_run"], row["n_unscorable"]), (10, 1, 1, 0))
        self.assertAlmostEqual(row["completion_of_list"], 10 / 12)
        self.assertAlmostEqual(row["completion_of_attempted"], 10 / 11)                  # entries that were not run are not counted as failures of the method

    def test_conflicting_results_are_reported_and_the_later_one_counts(self):
        f = A.load_status(self.fx.status)
        extra = pd.DataFrame([{"method": "e3", "entry": "1A03", "status": "ok", "started_utc": "2026-10-25T00:00:00Z", "seconds": "50", "file_order": 4}])
        f = pd.concat([f, extra], ignore_index=True).fillna("")
        f["_order"] = np.arange(len(f))
        m = A.merge_status(f)
        row = m[(m["method"] == "e3") & (m["entry"] == "1A03")].iloc[0]
        self.assertTrue(row["conflict"])
        self.assertEqual(row["status_raw"], "ok")
        qa = A.qa_checks(A.load_entries(self.fx.entries), f, m, A.load_scores(self.fx.scores), A.entry_outcomes(A.load_entries(self.fx.entries), m, A.load_scores(self.fx.scores)))
        self.assertTrue(any(l.startswith("CHECK") and "ok and a failed" in l and "e3/1A03" in l for l in qa))

    def test_a_metric_table_has_no_numbers_for_runs_without_a_model(self):
        m = self.res["metrics"]
        self.assertEqual(len(m), 12 * len(A.METHODS))
        self.assertEqual(m["tm_score"].notna().sum(), (m["outcome"] == "model").sum())
        self.assertTrue(m.loc[m["outcome"] != "model", "tm_score"].isna().all())
        self.assertEqual(set(m["outcome"]), {"model", "failed", "not_run"})

    def test_unscorable_outcomes(self):
        extra = [["1A09", "emprot", "emprot", True, "output.cif", "x", "no_alignment", "US-align failed", 100, 200] + [""] * 13]
        with tempfile.TemporaryDirectory() as tmp:
            s = score_rows(drop=[("emprot", "1A09"), ("emodelx", "1A07")], extra=extra)
            fx = Fixture(tmp, scores=s)
            res = fx.analyse()
        o = res["outcomes"]
        a = o[(o["method"] == "emprot") & (o["entry"] == "1A09")].iloc[0]
        b = o[(o["method"] == "emodelx") & (o["entry"] == "1A07")].iloc[0]
        self.assertEqual(a["outcome"], "unscorable")
        self.assertIn("no_alignment", a["reason"])
        self.assertEqual(b["outcome"], "unscorable")
        self.assertIn("no primary model file", b["reason"])
        self.assertTrue(any(l.startswith("CHECK") and "unscorable" in l for l in res["qa"]))
        z = res["zero"]
        z = z[(z["policy"] == "strict") & (z["group"] == "pooled") & (z["metric"] == "tm_score") & (z["method"] == "emodelx") & (z["role"] == "mean_over_entries")].iloc[0]
        self.assertEqual(z["n"], 12)                                                 # an unscorable run is a failure: it counts 0 (1A07), so all twelve entries are in
        self.assertAlmostEqual(z["mean"], sum(base_tm(i) + noise(i, 2) for i in range(12) if i != 6) / 12, places=3)


class FinalSetupTests(unittest.TestCase):
    """Rows of an earlier docking set-up are history, not the outcome (LOG.md 2026-10-27)."""

    @staticmethod
    def frame(rows):
        f = pd.DataFrame(rows).fillna("")
        for c in ("failed_step", "note", "started_utc", "seconds", "gpu", "device", "p1_seconds", "dock_seconds", "p3_seconds", "attempt", "peak_gpu_mb", "min_ram_gb",
                  "other_gpu_busy", "dock_profile", "source_file"):
            if c not in f.columns:
                f[c] = ""
        f["method"] = "mica"
        f["_order"] = np.arange(len(f))
        return f

    def merged(self, rows):
        m = A.merge_status(self.frame(rows))
        return {r.entry: r for r in m.itertuples()}

    def test_an_entry_that_waits_after_pass_3_is_not_run_and_not_failed(self):
        m = self.merged([{"entry": "AAAA", "status": "failed", "failed_step": "2_dock"}, {"entry": "AAAA", "status": "failed", "failed_step": "2_dock", "dock_profile": "pass2"},
                         {"entry": "AAAA", "status": "docked", "dock_profile": "pass3"}, {"entry": "AAAA", "status": "deferred", "dock_profile": "pass3"}])
        self.assertEqual(m["AAAA"].status_raw, "deferred")
        self.assertFalse(m["AAAA"].is_result)
        self.assertFalse(m["AAAA"].conflict)
        self.assertEqual((m["AAAA"].n_rows, m["AAAA"].n_earlier_setup_rows), (4, 2))
        self.assertEqual(m["AAAA"].history, "failed; failed; docked; deferred")

    def test_a_result_of_the_final_set_up_counts_and_an_earlier_failure_is_no_conflict(self):
        m = self.merged([{"entry": "BBBB", "status": "failed", "failed_step": "2_dock"}, {"entry": "BBBB", "status": "docked", "dock_profile": "pass3"},
                         {"entry": "BBBB", "status": "ok", "dock_profile": "pass3", "started_utc": "2026-10-27T02:00:00Z"}])
        self.assertEqual(m["BBBB"].status_raw, "ok")
        self.assertFalse(m["BBBB"].conflict)

    def test_two_results_of_the_final_set_up_are_a_conflict_and_the_later_one_counts(self):
        m = self.merged([{"entry": "CCCC", "status": "docked", "dock_profile": "pass3"},
                         {"entry": "CCCC", "status": "failed", "failed_step": "3_build", "dock_profile": "pass3", "started_utc": "2026-10-27T02:00:00Z"},
                         {"entry": "CCCC", "status": "ok", "dock_profile": "pass3", "started_utc": "2026-10-28T02:00:00Z"}])
        self.assertEqual(m["CCCC"].status_raw, "ok")
        self.assertTrue(m["CCCC"].conflict)

    def test_without_rows_of_the_final_set_up_the_old_rule_holds(self):
        m = self.merged([{"entry": "DDDD", "status": "failed", "failed_step": "2_dock"}, {"entry": "DDDD", "status": "deferred"}])
        self.assertEqual(m["DDDD"].status_raw, "failed")
        self.assertEqual(m["DDDD"].n_earlier_setup_rows, 0)


class TablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fx = Fixture(cls.tmp.name)
        cls.res = cls.fx.analyse(n_boot=500)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def paired(self, group, method, metric="tm_score"):
        p = self.res["paired"]
        return p[(p["group"] == group) & (p["method"] == method) & (p["metric"] == metric)].iloc[0]

    def test_pairs_use_the_entries_where_both_have_a_model(self):
        self.assertEqual(self.paired("pooled", "modelangelo")["n"], 10)             # MICA has 10 models
        self.assertEqual(self.paired("pooled", "e3")["n"], 9)                       # E3-CryoFold failed on 1A03
        self.assertEqual(self.paired("le3.0", "e3")["n"], 4)                        # 1A01, 1A04, 1A05, 1A06 (1A02 MICA failed, 1A03 E3 failed)
        self.assertEqual(self.paired("gt3.0_le4.0", "e3")["n"], 5)                  # 1A07 to 1A11 (1A12 not run)

    def test_differences_are_method_minus_mica(self):
        self.assertAlmostEqual(self.paired("pooled", "modelangelo")["mean_diff"], 0.10, delta=0.01)
        self.assertAlmostEqual(self.paired("pooled", "e3")["mean_diff"], -0.20, delta=0.01)
        self.assertAlmostEqual(self.paired("pooled", "emprot")["mean_diff"], -0.05, delta=0.01)
        self.assertLess(abs(self.paired("pooled", "emodelx")["mean_diff"]), 0.01)

    def test_exact_p_values_and_holm(self):
        ma, e3, ep, ex, ca, cz = (self.paired("pooled", m) for m in ("modelangelo", "e3", "emprot", "emodelx", "cryoatom1", "cryozeta"))
        self.assertAlmostEqual(ma["wilcoxon_p"], 2 / 2 ** 10)
        self.assertAlmostEqual(e3["wilcoxon_p"], 2 / 2 ** 9)
        self.assertAlmostEqual(ep["wilcoxon_p"], 2 / 2 ** 10)
        self.assertAlmostEqual(ca["wilcoxon_p"], 2 / 2 ** 9)                          # CryoAtom: nine pairs (it failed on 1A07, MICA on 1A02 and 1A12), all lower than MICA
        self.assertAlmostEqual(cz["wilcoxon_p"], 2 / 2 ** 7)                          # CryoZeta: seven pairs (it failed on 1A06 and was not run on 1A10 to 1A12; MICA failed on 1A02), all higher than MICA
        self.assertAlmostEqual(ma["wilcoxon_p_holm"], 6 * 2 / 2 ** 10)                # the smallest p (tied) times the six pairs compared
        self.assertAlmostEqual(ep["wilcoxon_p_holm"], 6 * 2 / 2 ** 10)
        self.assertAlmostEqual(e3["wilcoxon_p_holm"], max(6 * 2 / 2 ** 10, 4 * 2 / 2 ** 9))   # the step-down maximum (E3-CryoFold and CryoAtom are tied)
        self.assertAlmostEqual(cz["wilcoxon_p_holm"], 2 * 2 / 2 ** 7)                 # fifth in the order of the p-values: two pairs are left to compare
        self.assertAlmostEqual(ca["wilcoxon_p_holm"], e3["wilcoxon_p_holm"])
        self.assertGreater(ex["wilcoxon_p"], 0.05)
        for r in (ma, e3, ep, ex, ca, cz):
            self.assertGreaterEqual(r["wilcoxon_p_holm"], r["wilcoxon_p"])
            self.assertLessEqual(r["wilcoxon_p_holm"], 1.0)
        self.assertEqual((ma["n_method_higher"], ma["n_method_lower"]), (10, 0))
        self.assertEqual(ma["rank_biserial"], 1.0)
        self.assertEqual(e3["rank_biserial"], -1.0)

    def test_the_bootstrap_interval_of_a_clear_effect_excludes_zero(self):
        ma = self.paired("pooled", "modelangelo")
        self.assertGreater(ma["diff_ci_low"], 0.05)
        self.assertLess(ma["diff_ci_high"], 0.15)
        e3 = self.paired("pooled", "e3")
        self.assertLess(e3["diff_ci_high"], -0.15)

    def test_direction_of_better(self):
        r = self.paired("pooled", "e3", "tm_score")
        self.assertEqual((r["better"], r["n_method_better"], r["n_method_worse"]), ("higher", 0, 9))
        r = self.paired("pooled", "e3", "ca_rmsd")
        self.assertEqual(r["better"], "lower")
        self.assertEqual(self.paired("pooled", "e3", "coverage")["better"], "")

    def test_every_group_metric_and_method_has_a_row(self):
        p = self.res["paired"]
        self.assertEqual(len(p), 4 * len(A.ALL_METRICS) * (len(A.METHODS) - 1))   # 4 groups, metrics, every other method compared with MICA
        s = self.res["summary"]
        self.assertEqual(len(s), 4 * len(A.METHODS) * len(A.ALL_METRICS))

    def test_summary_values(self):
        s = self.res["summary"]
        r = s[(s["group"] == "pooled") & (s["method"] == "modelangelo") & (s["metric"] == "tm_score")].iloc[0]
        values = [base_tm(i) + 0.10 + noise(i, 1) for i in range(12)]
        self.assertEqual(r["n_model"], 12)
        self.assertAlmostEqual(r["mean"], np.mean(values), places=3)
        self.assertAlmostEqual(r["median"], np.median(values), places=3)
        self.assertLess(r["mean_ci_low"], r["mean"])
        self.assertGreater(r["mean_ci_high"], r["mean"])
        novel = s[(s["group"] == "novel_le30") & (s["method"] == "mica") & (s["metric"] == "tm_score")].iloc[0]
        self.assertEqual(novel["n_model"], 1)                                      # 1A02 is a MICA failure, 1A08 is the only model

    def test_failures_as_zero(self):
        z = self.res["zero"]

        def get(policy, method, role="mean_over_entries", group="pooled", metric="tm_score"):
            return z[(z["policy"] == policy) & (z["method"] == method) & (z["role"] == role) & (z["group"] == group) & (z["metric"] == metric)].iloc[0]

        models = [base_tm(i) - 0.20 + noise(i, 4) for i in range(11) if i != 2]       # E3-CryoFold: models on 1A01..1A11 except 1A03
        strict = get("strict", "e3")
        self.assertEqual(strict["n"], 11)                                           # ten models and the failure on 1A03; 1A12 was not run
        self.assertAlmostEqual(strict["mean"], sum(models) / 11, places=3)
        allm = get("all_missing", "e3")
        self.assertEqual(allm["n"], 12)
        self.assertAlmostEqual(allm["mean"], sum(models) / 12, places=3)
        mica = get("strict", "mica")
        mica_models = [base_tm(i) + noise(i, 0) for i in range(11) if i != 1]       # 1A02 failed (0), 1A12 not run
        self.assertEqual(mica["n"], 11)
        self.assertAlmostEqual(mica["mean"], sum(mica_models) / 11, places=3)
        pair = get("strict", "e3", role="paired_vs_mica")
        self.assertEqual(pair["n"], 11)
        self.assertLess(pair["mean_diff"], 0)
        self.assertEqual(get("all_missing", "e3", role="paired_vs_mica")["n"], 12)
        # CryoZeta (partial method): the three entries that were not run are not failures. They are left out in the strict policy and count as zero in the other one
        cz_models = [base_tm(i) + 0.04 + noise(i, 6) for i in range(9) if i != 5]
        czs, cza = get("strict", "cryozeta"), get("all_missing", "cryozeta")
        self.assertEqual((czs["n"], cza["n"]), (9, 12))                             # eight models and the failure on 1A06 / the same and the three entries not run
        self.assertAlmostEqual(czs["mean"], sum(cz_models) / 9, places=3)
        self.assertAlmostEqual(cza["mean"], sum(cz_models) / 12, places=3)
        self.assertEqual(get("strict", "cryozeta", role="paired_vs_mica")["n"], 9)
        self.assertEqual(get("all_missing", "cryozeta", role="paired_vs_mica")["n"], 12)

    def test_variants_and_correlations_and_runtime(self):
        v = self.res["variants"]
        self.assertEqual(set(v[v["method"] == "mica"]["model_kind"]), {"mica_all_atom", "mica_ca"})
        self.assertTrue(bool(v[v["model_kind"] == "mica_all_atom"]["primary"].iloc[0]))
        self.assertFalse(bool(v[v["model_kind"] == "mica_ca"]["primary"].iloc[0]))
        c = self.res["correlations"]
        r = c[(c["method"] == "modelangelo") & (c["x"] == "resolution_A") & (c["y"] == "tm_score")].iloc[0]
        self.assertEqual(r["n"], 12)
        self.assertGreater(r["spearman_rho"], 0.9)                                  # TM-score rises with the entry index in the synthetic data, and so does the resolution value
        rt = self.res["runtime"]
        self.assertEqual(rt[rt["method"] == "mica"]["n_runs"].iloc[0], 11)           # ten models and the failed docking
        self.assertAlmostEqual(rt[rt["method"] == "modelangelo"]["min_min"].iloc[0], 10.0)

    def test_qa_passes_on_clean_inputs_except_the_planted_conflict_free_state(self):
        text = "\n".join(self.res["qa"])
        self.assertIn("all checks passed", text)
        self.assertNotIn("CHECK", text)


class EndToEndTests(unittest.TestCase):
    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            out = os.path.join(tmp, "results")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = A.main(["--entries", fx.entries, "--status"] + fx.status + ["--scores"] + fx.scores + ["--out", out, "--resamples", "200"])
            self.assertEqual(rc, 0)
            for name in ("metrics_per_entry.csv", "entry_outcomes.csv", "completion.csv", "summary_by_method.csv", "paired_vs_mica.csv", "sensitivity_failures_as_zero.csv",
                         "variants.csv", "correlations.csv", "runtime.csv", "status_merged.csv", "qa.txt", "provenance.json"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            m = pd.read_csv(os.path.join(out, "metrics_per_entry.csv"))
            self.assertEqual(list(m.columns[:len(A.SPEC_COLUMNS)]), A.SPEC_COLUMNS)    # the header that the repo has had since decision 7
            self.assertEqual(len(m), 12 * len(A.METHODS))
            self.assertEqual(set(m["status"]), {"model", "failed", "not_run"})
            row = m[(m["method"] == "modelangelo") & (m["pdb_id"] == "1A01")].iloc[0]
            self.assertAlmostEqual(row["tm_score"], round(base_tm(0) + 0.10 + noise(0, 1), 4), places=4)
            self.assertAlmostEqual(row["runtime_min"], 10.0)
            with open(os.path.join(out, "provenance.json")) as f:
                prov = json.load(f)
            self.assertEqual((prov["seed"], prov["bootstrap_resamples"]), (2026, 200))
            self.assertEqual(len(prov["inputs"]["scores"]), 1)
            self.assertIn("all checks passed", buf.getvalue())

    def test_the_tables_do_not_depend_on_the_order_of_the_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            a = fx.analyse(n_boot=300)
            s = score_rows().sample(frac=1.0, random_state=3)
            write(s, fx.scores[0])
            b = fx.analyse(n_boot=300)
            pd.testing.assert_frame_equal(a["paired"].reset_index(drop=True), b["paired"].reset_index(drop=True))

    def test_qa_flags_what_does_not_fit(self):
        extra = [["ZZZZ", "mica", "mica_all_atom", True, "x.pdb", "x", "ok", "", 1, 1, 1, 0.5, 0.5, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1],
                 ["1A02", "mica", "mica_all_atom", True, "x.pdb", "x", "ok", "", 1, 1, 1, 0.5, 0.5, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1]]     # a model of a run that failed
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, scores=score_rows(extra=extra))
            res = fx.analyse()
        text = "\n".join(res["qa"])
        self.assertIn("CHECK scores: every entry is in the frozen list (not in the list: ZZZZ)", text)
        self.assertIn("CHECK scores: no model of a run that is not ok (mica/1A02)", text)
        self.assertIn("2 point(s) to look at", text)

    def test_no_scores_at_all_still_gives_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            entries = A.load_entries(fx.entries)
            res = A.analyse(entries, A.load_status(fx.status), A.load_scores([]), n_boot=100, log=lambda *a: None)
            self.assertEqual(set(res["outcomes"]["outcome"]), {"unscorable", "failed", "not_run"})
            self.assertEqual(len(res["metrics"]), 12 * len(A.METHODS))
            A.write_all(res, os.path.join(tmp, "out"), {"entries": [fx.entries], "status": fx.status, "scores": []})


class DockModeTests(unittest.TestCase):
    """LOG.md, 2026-10-26: the entries are also grouped by how far MICA's docking got (column dock_mode of mica_status.csv)."""

    def test_groups_use_the_last_value_that_is_not_empty(self):
        frame = pd.DataFrame({"method": ["mica"] * 7 + ["modelangelo"], "entry": ["A", "B", "A", "C", "D", "B", "E", "F"],
                              "dock_mode": ["full", "none", "", "partial", "", "full", "none", "full"], "_order": range(8)})
        g = A.dock_mode_groups(frame)
        self.assertEqual(g, {"mica_docked": ["A", "B", "C"], "mica_nodock": ["E"]})            # B: the later "full" replaces "none"; D never had a mode; F is a baseline row
        self.assertEqual(A.dock_mode_groups(frame.drop(columns="dock_mode")), {})
        self.assertEqual(A.dock_mode_groups(frame.iloc[0:0]), {})
        self.assertEqual(A.dock_mode_groups(None), {})

    def test_the_analysis_has_a_row_for_each_dock_mode_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            t = pd.read_csv(fx.status[2], dtype=str, keep_default_na=False)
            t["dock_mode"] = ["full"] * 4 + ["partial"] + ["none"] * 7                         # 1A01 to 1A05 docked, 1A06 to 1A12 not
            t.to_csv(fx.status[2], index=False)
            res = fx.analyse(n_boot=200)
            p, s = res["paired"], res["summary"]
            self.assertEqual(set(s["group"]), {"pooled", "le3.0", "gt3.0_le4.0", "novel_le30", "mica_docked", "mica_nodock"})
            def n(group, method):
                return int(p[(p["group"] == group) & (p["method"] == method) & (p["metric"] == "tm_score")].iloc[0]["n"])
            self.assertEqual(n("mica_docked", "modelangelo"), 4)          # 1A01, 1A03, 1A04, 1A05 (MICA has no model for 1A02)
            self.assertEqual(n("mica_nodock", "modelangelo"), 6)          # 1A06 to 1A11 (MICA has no model for 1A12)
            self.assertEqual(n("pooled", "modelangelo"), 10)

    def test_without_the_column_there_are_no_such_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = Fixture(tmp).analyse(n_boot=100)
            self.assertFalse({"mica_docked", "mica_nodock"} & set(res["summary"]["group"]))


if __name__ == "__main__":
    unittest.main()
