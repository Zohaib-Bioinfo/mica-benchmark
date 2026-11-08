"""Tests of make_report_tables.py on small synthetic inputs. The real data folder is used only for files that do not change (pool flow, methods table).

Run:  python3 -W error -m unittest test_make_report_tables
"""
import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout

import pandas as pd

import analyze_results as A
import make_report_tables as T
import test_analyze_results as TA

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DOCK_COLUMNS = ["entry", "model", "attempt", "setting", "settings", "timeout_s", "seconds", "rc", "timed_out", "skipped", "docked", "local_cc", "log"]


def dock_row(entry, model, attempt, setting, seconds, docked, cc="", timed_out=False, skipped=False):
    return [entry, model, attempt, setting, "{}", 2400, seconds, 0 if docked else 1, timed_out, skipped, docked, cc, "x.log"]


def dock_frame(rows):
    return pd.DataFrame(rows, columns=DOCK_COLUMNS)


def make_raw(tmp, zips):
    """zips: (notebook, stamp, {file name: DataFrame | dict | text}); writes the folders and INDEX.csv as ingest_results.py does."""
    raw = os.path.join(tmp, "raw")
    os.makedirs(raw)
    index = []
    for notebook, stamp, files in zips:
        folder = "%s_result_%s" % (notebook, stamp)
        os.makedirs(os.path.join(raw, folder))
        for name, content in files.items():
            path = os.path.join(raw, folder, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if isinstance(content, pd.DataFrame):
                content.to_csv(path, index=False)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(content) if isinstance(content, dict) else content)
        index.append({"folder": folder, "notebook": notebook, "stamp": stamp})
    pd.DataFrame(index).to_csv(os.path.join(raw, "INDEX.csv"), index=False)
    return raw


# AAAA has three domains, BBBB two. Pass 1: one docked. Pass 2: nothing (all instant failures). Pass 3: two of three docked in AAAA, nothing in BBBB.
PASS1 = dock_frame([dock_row("AAAA", "a1.pdb", 1, "mica", 100, True, "0.60"), dock_row("AAAA", "a2.pdb", 1, "mica", 2700, False, timed_out=True),
                    dock_row("AAAA", "a3.pdb", 1, "mica", 2700, False, timed_out=True), dock_row("BBBB", "b1.pdb", 1, "mica", 2700, False, timed_out=True),
                    dock_row("BBBB", "b2.pdb", 1, "mica", 2700, False, timed_out=True)])
PASS2 = dock_frame([dock_row(e, m, 1, "res6_quick", 1, False) for e, m in (("AAAA", "a1.pdb"), ("AAAA", "a2.pdb"), ("AAAA", "a3.pdb"), ("BBBB", "b1.pdb"), ("BBBB", "b2.pdb"))])
PASS3 = dock_frame([dock_row("AAAA", "a1.pdb", 1, "res6_quick", 600, True, "0.55"),
                    dock_row("AAAA", "a2.pdb", 1, "res6_quick", 2400, False, timed_out=True), dock_row("AAAA", "a2.pdb", 2, "res10_quick", 800, True, "0.45"),
                    dock_row("AAAA", "a3.pdb", 1, "res6_quick", 2400, False, timed_out=True), dock_row("AAAA", "a3.pdb", 2, "res10_quick", 1500, False, timed_out=True),
                    dock_row("BBBB", "b1.pdb", 1, "res6_quick", 2400, False, timed_out=True), dock_row("BBBB", "b1.pdb", 2, "res10_quick", 1500, False, timed_out=True),
                    dock_row("BBBB", "b2.pdb", 1, "res6_quick", 2400, False, timed_out=True), dock_row("BBBB", "b2.pdb", 2, "res10_quick", 1500, False, timed_out=True)])
STATUS3 = pd.DataFrame([["AAAA", "docked", "", "", 3, 2, "partial"], ["BBBB", "docked", "", "", 2, 0, "none"]], columns=["entry", "status", "failed_step", "note", "domains", "domains_docked", "dock_mode"])
STATUS_GPU = pd.DataFrame([["AAAA", "ok", "", "", "", "", "", 4000.0], ["BBBB", "failed", "3_run", "program 3 stopped", "", "", "", 100.0]],
                          columns=["entry", "status", "failed_step", "note", "domains", "domains_docked", "dock_mode", "seconds"])


def zips():
    return [("mica_cpu_a", "20261024-1331", {"docking_domains.csv": PASS1, "run_info.json": {"mica_commit": "aaaaaaaaaaaaaaaa1111", "phenix_installer": "Phenix-2.2.1-6174-Linux-x86_64.sh",
                                                                                          "usalign_commit": "bbbb", "cpu_cores": 4, "ram_gb": 31.3}}),
            ("mica_dock2_a", "20261025-2038", {"docking_domains.csv": PASS2, "run_info.json": {"mica_commit": "aaaaaaaaaaaaaaaa1111"}}),
            ("mica_dock3_a", "20261026-2000", {"docking_domains.csv": PASS3, "mica_status.csv": STATUS3, "run_info.json": {"mica_commit": "cccccccccccc2222"}}),
            ("gpu_batch3", "20261025-2341", {"run_info.json": {"methods": {"modelangelo": {"commit": "dddddddddddd"}, "emprot": {"commit": "eeeeeeeeeeee"}}, "gpu": "Tesla T4, 15360 MiB"}}),
            ("mica_gpu", "20261027-1200", {"mica_status.csv": STATUS_GPU})]


class DockingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = make_raw(self.tmp.name, zips())
        self.dom = T.docking_passes(self.raw)

    def tearDown(self):
        self.tmp.cleanup()

    def test_modes(self):
        self.assertEqual([T.dock_mode(0, 3), T.dock_mode(2, 3), T.dock_mode(3, 3), T.dock_mode(0, 0)], ["none", "partial", "full", "none"])

    def test_one_row_per_pass_entry_and_domain(self):
        d = self.dom
        self.assertEqual(sorted(d["pass"].unique()), [1, 2, 3])
        self.assertEqual(d.groupby("pass").size().to_dict(), {1: 5, 2: 5, 3: 5})
        a2 = d[(d["pass"] == 3) & (d["model"] == "a2.pdb")].iloc[0]
        self.assertTrue(a2["docked"])
        self.assertEqual(a2["setting"], "res10_quick")                  # the second rung docked it
        self.assertEqual(a2["local_cc"], 0.45)
        self.assertEqual(a2["seconds"], 3200)                           # both rungs count
        self.assertTrue(a2["timed_out"])                                # the first rung ran into its limit
        a3 = d[(d["pass"] == 3) & (d["model"] == "a3.pdb")].iloc[0]
        self.assertFalse(a3["docked"])
        self.assertEqual(a3["setting"], "")

    def test_a_skipped_attempt_does_not_count_as_time(self):
        rows = dock_frame([dock_row("AAAA", "a1.pdb", 1, "mica", 2700, False, timed_out=True), dock_row("AAAA", "a1.pdb", 2, "plain", 0, False, skipped=True)])
        with tempfile.TemporaryDirectory() as tmp:
            d = T.docking_passes(make_raw(tmp, [("mica_cpu_a", "20261024-1331", {"docking_domains.csv": rows})]))
        self.assertEqual(d.iloc[0]["seconds"], 2700)

    def test_notebooks_that_are_not_docking_passes_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = T.docking_passes(make_raw(tmp, [("gpu_batch1", "20261023-1416", {"docking_domains.csv": PASS1})]))
        self.assertTrue(d.empty)

    def test_macros(self):
        m = T.Macros()
        T.docking_macros(m, self.dom)
        v = m.items
        self.assertEqual((v["DockDomainsOne"], v["DockDockedOne"], v["DockTimedOutOne"]), ("5", "1", "4"))
        self.assertEqual((v["DockDockedTwo"], v["DockTimedOutTwo"], v["DockEntriesNoneTwo"]), ("0", "0", "2"))
        self.assertEqual((v["DockDockedThree"], v["DockTimedOutThree"], v["DockEntriesNoneThree"], v["DockEntriesFullThree"], v["DockEntriesThree"]), ("2", "3", "1", "0", "2"))
        self.assertEqual((v["DockCcMedOne"], v["DockCcMedTwo"], v["DockCcMedThree"]), ("0.60", "n/a", "0.50"))
        self.assertEqual(v["DockMinutesMedThree"], "65")                 # per domain: 10, 53, 65, 65, 65 min
        self.assertEqual((v["DockModeFull"], v["DockModePartial"], v["DockModeNone"]), ("0", "1", "1"))
        self.assertEqual((v["DockRungOne"], v["DockRungTwo"]), ("1", "1"))
        self.assertEqual((v["DockBoth"], v["DockCcDiffMed"]), ("1", "$-$0.05"))     # a1 docked in pass 1 (0.60) and pass 3 (0.55)

    def test_no_docking_files_no_macros(self):
        m = T.Macros()
        T.docking_macros(m, pd.DataFrame())
        self.assertEqual(m.items, {})

    def test_table(self):
        entries = pd.DataFrame({"pdb_id": ["AAAA", "BBBB", "CCCC"]})
        text = T.docking_table(self.dom, entries)
        lines = {l.split(" & ")[0]: l for l in text.splitlines() if " & " in l}
        self.assertEqual(lines["AAAA"].split(" & ")[:3], ["AAAA", "3", "1 / 3"])
        self.assertIn("2 / 3", lines["AAAA"])
        self.assertTrue(lines["AAAA"].rstrip(" \\\\").endswith("partial"))
        self.assertTrue(lines["BBBB"].rstrip(" \\\\").endswith("none"))
        self.assertTrue(all(c.strip(" \\") == "--" for c in lines["CCCC"].split(" & ")[1:]))
        self.assertIn("All & 5 & 1 / 5", text)
        self.assertEqual(text.count("\\begin{tabular}"), 1)


class PinsTests(unittest.TestCase):
    def test_pins_and_macros(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = make_raw(tmp, zips())
            pins = T.collect_pins(raw)
            self.assertEqual(pins["MICA"], "aaaaaaaaaaaaaaaa1111; cccccccccccc2222")          # two different commits are shown, oldest first
            self.assertEqual(pins["ModelAngelo"], "dddddddddddd")
            self.assertEqual(pins["EMProt"], "eeeeeeeeeeee")
            m = T.Macros()
            T.commit_macros(m, raw)
            v = m.items
            self.assertEqual((v["CommitMica"], v["CommitMicaPins"], v["CommitMicaFull"]), ("aaaaaaaaaa", "2", "aaaaaaaaaaaaaaaa1111"))
            self.assertEqual((v["CommitAngelo"], v["CommitAngeloPins"], v["CommitEmprot"], v["CommitUsalign"]), ("dddddddddd", "1", "eeeeeeeeee", "bbbb"))
            self.assertEqual((v["PhenixVersion"], v["PhenixInstaller"]), ("2.2.1-6174", "Phenix-2.2.1-6174-Linux-x86_64.sh"))
            table = T.pins_table(raw)
            self.assertIn("\\texttt{aaaaaaaaaaaaaaaa1111; cccccccccccc2222}", table)
            self.assertIn("Phenix-2.2.1-6174-Linux-x86\\_64.sh", table)
            m2 = T.Macros()
            T.environment_macros(m2, raw)
            self.assertEqual((m2.items["CpuCores"], m2.items["RamGb"], m2.items["GpuName"], m2.items["GpuMemory"]), ("4", "31.3", "Tesla T4", "15360 MiB"))

    def test_without_raw_results_there_is_nothing_to_pin(self):
        m = T.Macros()
        T.commit_macros(m, "")
        self.assertEqual(m.items, {})
        self.assertIn("No run information", T.pins_table(""))


class FixedFilesTests(unittest.TestCase):
    def test_pool_flow(self):
        text = T.pool_flow_table(os.path.join(REPO, "data"))
        self.assertIn("6\\,116", text)
        self.assertIn("Resolution at most 4.0 \\AA{}", text)
        self.assertIn("5\\% UNK", text)
        self.assertEqual(text.count("\\\\"), 12)                         # header + 11 steps
        m = T.Macros()
        T.dataset_macros(m, os.path.join(REPO, "data"))
        self.assertEqual((m.items["PoolAll"], m.items["PoolFinal"]), ("6\\,116", "484"))

    def test_methods_screen(self):
        csv = os.path.join(REPO, "methods", "methods_table.csv")
        text = T.methods_screen_table(csv)
        self.assertTrue(text.startswith("\\begin{longtable}") and text.rstrip().endswith("\\end{longtable}"))
        t = pd.read_csv(csv, dtype=str).fillna("")
        self.assertEqual(text.count(" \\\\\n"), len(t) + 2)                # one row per method + the header row twice (first page and continuation pages)
        m = T.Macros()
        T.methods_macros(m, csv)
        self.assertEqual(int(m.items["NScreened"]), int((t["included"] != "baseline").sum()))
        self.assertEqual(int(m.items["NScreenPassed"]) + int(m.items["NScreenReserve"]) + int(m.items["NScreenExcluded"]), int(m.items["NScreened"]))
        for x in re.findall(r"[^\x00-\x7f]", text):
            self.fail("a non-ASCII character in the screening table: %r" % x)

    def test_missing_files_are_reported_in_the_table_not_as_a_crash(self):
        self.assertIn("not found", T.pool_flow_table("/nonexistent"))
        self.assertIn("not found", T.methods_screen_table("/nonexistent.csv"))


class MicaModesTests(unittest.TestCase):
    def test_one_row_per_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = make_raw(tmp, zips())
            entries = pd.DataFrame({"pdb_id": ["AAAA", "BBBB", "CCCC"]})
            metrics = pd.DataFrame({"pdb_id": ["AAAA", "AAAA", "BBBB"], "method": ["mica", "modelangelo", "mica"], "tm_score": [0.612, 0.7, float("nan")]})
            outcomes = pd.DataFrame({"entry": ["AAAA", "BBBB", "CCCC"], "method": ["mica"] * 3, "outcome": ["model", "failed", "not_run"]})
            text = T.mica_modes_table(raw, entries, metrics, outcomes)
        rows = {l.split(" & ")[0]: [c.strip(" \\") for c in l.split(" & ")] for l in text.splitlines() if " & " in l and not l.startswith("Entry")}
        self.assertEqual(rows["AAAA"], ["AAAA", "3", "2", "partial", "ok", "0.61"])            # the mode and the counts come from the docking session, the status from the GPU session
        self.assertEqual(rows["BBBB"], ["BBBB", "2", "0", "none", "failed", "F"])
        self.assertEqual(rows["CCCC"], ["CCCC", "--", "--", "--", "--", "N"])

    def test_the_sensitivity_sessions_do_not_decide_and_a_rerun_shows_both_results(self):
        nodock = pd.DataFrame([["AAAA", "failed", "3_run", "x", "", "", "none", 1.0]], columns=["entry", "status", "failed_step", "note", "domains", "domains_docked", "dock_mode", "seconds"])
        rerun = pd.DataFrame([["BBBB", "ok", "", "", "", "", "", 90.0]], columns=["entry", "status", "failed_step", "note", "domains", "domains_docked", "dock_mode", "seconds"])
        with tempfile.TemporaryDirectory() as tmp:
            raw = make_raw(tmp, zips() + [("mica_gpu_nodock", "20261027-1300", {"mica_status.csv": nodock}), ("mica_gpu2_b", "20261028-0100", {"mica_status.csv": rerun})])
            entries = pd.DataFrame({"pdb_id": ["AAAA", "BBBB"]})
            metrics = pd.DataFrame({"pdb_id": ["AAAA", "BBBB"], "method": ["mica", "mica"], "tm_score": [0.9, 0.8]})
            outcomes = pd.DataFrame({"entry": ["AAAA", "BBBB"], "method": ["mica"] * 2, "outcome": ["model", "model"]})
            text = T.mica_modes_table(raw, entries, metrics, outcomes)
        rows = {l.split(" & ")[0]: [c.strip(" \\") for c in l.split(" & ")] for l in text.splitlines() if " & " in l and not l.startswith("Entry")}
        self.assertEqual(rows["AAAA"][4], "ok")                                   # the nodock session said failed: it is another analysis
        self.assertEqual(rows["BBBB"][3:5], ["none", "failed $\\to$ ok"])            # failed in the first GPU session, ok in the second


class DockEffectTests(unittest.TestCase):
    ENTRIES = pd.DataFrame({"pdb_id": ["AAAA", "BBBB", "CCCC", "DDDD"], "resolution_A": [2.0, 2.5, 3.1, 3.5]})
    DOM = pd.DataFrame({"pass": [3, 3, 3, 3, 3], "entry": ["AAAA", "AAAA", "BBBB", "CCCC", "CCCC"], "docked": [True, True, False, True, False]})

    @staticmethod
    def metrics(values):
        return pd.DataFrame({"pdb_id": ["AAAA", "BBBB", "CCCC", "DDDD"], "method": ["mica"] * 4, "tm_score": values})

    def test_pairs_difference_and_numbers(self):
        text, m = T.dock_effect(self.metrics([0.95, 0.90, 0.80, float("nan")]), self.metrics([0.90, 0.90, 0.85, 0.70]), self.DOM, self.ENTRIES)
        rows = {l.split(" & ")[0]: [c.strip(" \\") for c in l.split(" & ")] for l in text.splitlines() if " & " in l and not l.startswith("Entry")}
        self.assertEqual(rows["AAAA"], ["AAAA", "2.00", "full", "0.950", "0.900", "+0.050"])
        self.assertEqual(rows["BBBB"][2:], ["none", "0.900", "0.900", "0.000"])
        self.assertEqual(rows["CCCC"][2], "partial")
        self.assertNotIn("DDDD", rows)                                   # no result with docked input: not a pair
        self.assertEqual((m["N"], m["Higher"], m["Lower"], m["Equal"]), ("3", "1", "1", "1"))
        self.assertEqual((m["Mean"], m["Med"], m["P"]), ("0.000", "0.000", "1.000"))
        self.assertEqual((m["MeanDocked"], m["MeanNoDock"]), ("0.883", "0.883"))
        self.assertEqual(text.count("\\begin{tabular}"), 1)

    def test_the_sign_is_docked_minus_no_docked_input(self):
        _, m = T.dock_effect(self.metrics([0.95, 0.96, 0.97, 0.98]), self.metrics([0.90, 0.90, 0.90, 0.90]), self.DOM, self.ENTRIES)
        self.assertEqual((m["N"], m["Higher"], m["Lower"]), ("4", "4", "0"))
        self.assertTrue(float(m["Mean"]) > 0.05)
        self.assertTrue(float(m["Lo"]) <= float(m["Mean"]) <= float(m["Hi"]))

    def test_no_pair_gives_a_sentence_and_n_a(self):
        text, m = T.dock_effect(pd.DataFrame(), self.metrics([0.9] * 4), self.DOM, self.ENTRIES)
        self.assertIn("No entry has a result", text)
        self.assertEqual(set(m.values()), {"n/a"})
        text, m = T.dock_effect(self.metrics([0.9] * 4), pd.DataFrame(), pd.DataFrame(), self.ENTRIES)
        self.assertEqual(m["N"], "n/a")


class ExtraMacroTests(unittest.TestCase):
    """The numbers that the text quotes beyond the tables (extra_macros), on small synthetic frames."""
    ENTRIES = pd.DataFrame({"pdb_id": ["AAAA", "BBBB", "CCCC", "DDDD"]})

    @staticmethod
    def summary():
        rows = []
        for method, tm, fns in (("mica", [0.8, 0.9, 0.99], 0.95), ("modelangelo", [0.2, 0.9, 1.0], 0.9)):
            rows.append({"group": "pooled", "method": method, "metric": "tm_score", "n": 3, "mean": sum(tm) / 3, "median": sorted(tm)[1], "q1": min(tm), "q3": max(tm), "min": min(tm),
                         "max": max(tm), "mean_ci_low": 0.1, "mean_ci_high": 0.9})
            rows.append(dict(rows[-1], metric="tm_score_fns", mean=fns, median=fns))
        return pd.DataFrame(rows)

    @staticmethod
    def paired():
        base = {"group": "pooled", "metric": "tm_score_fns", "method": "modelangelo", "n": 3, "mean_diff": -0.05, "diff_ci_low": -0.1, "diff_ci_high": 0.0, "wilcoxon_p": 0.25,
                "wilcoxon_p_holm": 0.5, "n_method_higher": 1, "n_method_lower": 2}
        return pd.DataFrame([base, dict(base, group="le3.0", n=1)])

    @staticmethod
    def metrics():
        rows = []
        for e, tm in (("AAAA", 0.99), ("BBBB", 0.95), ("CCCC", 0.85), ("DDDD", 0.80)):
            rows.append({"pdb_id": e, "method": "mica", "status": "model", "tm_score": tm})
        rows.append({"pdb_id": "AAAA", "method": "modelangelo", "status": "model", "tm_score": 0.30})
        rows.append({"pdb_id": "BBBB", "method": "modelangelo", "status": "failed", "tm_score": float("nan")})
        return pd.DataFrame(rows)

    def macros(self, **extra):
        res = {"summary": self.summary(), "paired": self.paired(), "metrics": self.metrics(), "runtime": pd.DataFrame(), "entries": self.ENTRIES}
        res.update(extra)
        m = T.Macros()
        T.extra_macros(m, res, extra.pop("dom", None))
        return m.items

    def test_minimum_maximum_and_the_fragment_agnostic_numbers(self):
        m = self.macros()
        self.assertEqual((m["TmMinMica"], m["TmMaxMica"], m["TmMinAngelo"]), ("0.80", "0.99", "0.20"))
        self.assertEqual((m["FnsMedMica"], m["FnsMeanAngelo"]), ("0.95", "0.90"))
        self.assertEqual((m["DiffFnsAngelo"], m["PwHolmFnsAngelo"], m["NPairsFnsAngelo"], m["HigherFnsAngelo"], m["LowerFnsAngelo"]), ("$-$0.05", "0.500", "3", "1", "2"))   # the pooled row, not the stratum

    def test_counts_above_and_below_a_tm_score_count_models_only(self):
        m = self.macros()
        self.assertEqual((m["NTmGeNinetyMica"], m["NTmGeNinetyFiveMica"], m["NTmLtNinetyMica"], m["NTmLtHalfMica"]), ("2", "2", "2", "0"))
        self.assertEqual((m["NTmLtHalfAngelo"], m["NTmGeNinetyAngelo"]), ("1", "0"))               # the failed run has no score and is not counted
        self.assertEqual(m["NTmLtHalfEthree"], "0")                                                  # a method without rows gets zeros, not a missing macro

    def test_failure_classes(self):
        out = pd.DataFrame({"entry": ["AAAA", "BBBB", "CCCC", "CCCC", "DDDD"], "method": ["mica", "mica", "e3", "e3", "emprot"], "outcome": ["failed", "failed", "failed", "failed", "model"],
                            "reason": ["3_build: exit code -9; x", "3_build: timeout after 7200 s", "prealign: exit code 1", "prealign: killed by the system (exit code -9, probably out of memory)", ""]})
        m = self.macros(outcomes=out)
        self.assertEqual((m["FailTotal"], m["FailKill"], m["FailTimeout"], m["FailCrash"]), ("4", "2", "1", "1"))                     # the run with a model is not a failure
        self.assertEqual((m["FailKillMica"], m["FailTimeoutMica"], m["FailCrashEthree"], m["FailCrashListEthree"], m["FailKillEmprot"]), ("1", "1", "1", "CCCC", "0"))

    def test_a_run_that_was_not_run_is_not_a_failure_and_the_lists_of_entries_by_outcome(self):
        out = pd.DataFrame({"entry": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"], "method": ["cryozeta"] * 6, "outcome": ["model", "failed", "failed", "not_run", "not_run", "model"],
                            "reason": ["", "run: killed by the system (exit code -9, probably out of memory; also when it ran alone)", "run: out of GPU memory (exit code 2)",
                                       "not run: not started: not enough time left in the session", "killed_corun: killed by the system (exit code -9)", ""]})
        m = self.macros(outcomes=out)
        self.assertEqual((m["FailTotal"], m["FailKill"], m["FailCrash"]), ("2", "1", "1"))                                           # the two entries not run are not counted
        self.assertEqual((m["FailKillListCryozeta"], m["FailCrashListCryozeta"], m["FailTimeoutListCryozeta"]), ("BBBB", "CCCC", "none"))
        self.assertEqual((m["ModelListCryozeta"], m["FailedListCryozeta"], m["NotRunListCryozeta"]), ("AAAA and FFFF", "BBBB and CCCC", "DDDD and EEEE"))
        self.assertEqual(m["ModelListMica"], "none")                                                                                   # a method without rows gets "none", not a missing macro

    def test_alphafold_prediction_macros(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "af3"))
            pd.DataFrame({"pdb_id": ["AAAA", "AAAA", "BBBB", "CCCC"], "entity": ["E1", "E2", "E1", "E1"], "mean_plddt_ca": [90.0, 62.4, 80.0, 97.5],
                          "frac_ca_plddt_lt50": [0.0, 0.428, 0.254, 0.0]}).to_csv(os.path.join(d, "af3", "af3_results_check.csv"), index=False)
            m = T.Macros()
            T.dataset_macros(m, d)
        self.assertEqual((m.items["AfChains"], m.items["AfPlddtMin"], m.items["AfPlddtMed"], m.items["AfPlddtMax"]), ("4", "62.4", "85.0", "97.5"))
        self.assertEqual(m.items["AfLowList"], "AAAA chain E2 (43\\,\\%), BBBB chain E1 (25\\,\\%) and AAAA chain E1 (0\\,\\%)")      # the three chains with most residues below 50

    def test_a_median_that_rounds_to_zero_has_no_sign(self):
        self.assertEqual((T.zero_free(-0.0004), T.zero_free(-0.04), T.zero_free(0.0), T.zero_free(-0.0004, 3)), ("0.00", "$-$0.04", "0.00", "0.000"))
        self.assertEqual(T.ci(-0.1, -0.001), "[$-$0.10, $-$0.00]")                                  # a confidence bound keeps its sign

    def test_runtime_and_correlation_macros(self):
        rt = pd.DataFrame({"method": ["mica"], "n_runs": [20], "median_min": [81.59], "min_min": [32.8], "max_min": [557.5], "total_h": [40.035]})
        cor = pd.DataFrame({"method": ["mica", "mica"], "x": ["resolution_A", "modeled_residues"], "x_label": ["", ""], "y": ["tm_score", "tm_score"], "n": [17, 17],
                            "spearman_rho": [-0.254, 0.034], "p": [0.325, 0.896]})
        m = self.macros(runtime=rt, correlations=cor)
        self.assertEqual((m["RtMedMica"], m["RtMinMica"], m["RtMaxMica"], m["RtTotMica"], m["RtNMica"]), ("82", "33", "558", "40.0", "20"))
        self.assertEqual((m["RhoResTmMica"], m["RhoPResTmMica"], m["RhoSizeTmMica"], m["RhoNResTmMica"]), ("$-$0.25", "0.325", "0.03", "17"))

    def test_reruns_modes_and_the_lowest_three(self):
        merged = pd.DataFrame({"method": ["mica"] * 3 + ["emprot"], "entry": ["CCCC", "AAAA", "DDDD", "AAAA"], "conflict": [True, False, True, True]})
        dom = pd.DataFrame({"pass": [3] * 6, "entry": ["AAAA", "BBBB", "BBBB", "CCCC", "CCCC", "DDDD"], "docked": [True, True, True, True, False, False]})
        m = self.macros(merged=merged, dom=dom)
        self.assertEqual((m["NRerunMica"], m["RerunMicaList"]), ("2", "CCCC and DDDD"))              # in the order of the frozen list, MICA only
        self.assertEqual((m["ModeFullN"], m["ModeFullModels"], m["ModeFullTmMean"]), ("2", "2", "0.97"))
        self.assertEqual((m["ModePartialN"], m["ModePartialModels"], m["ModePartialTmMin"]), ("1", "1", "0.85"))
        self.assertEqual((m["ModeNoneN"], m["ModeNoneModels"], m["ModeNoneFailed"], m["ModeNoneTmMean"]), ("1", "1", "0", "0.80"))      # DDDD docked nothing and has a model
        self.assertEqual((m["ModeFullFailed"], m["ModePartialFailed"]), ("0", "0"))
        self.assertEqual((m["ModeLowThreeList"], m["ModeLowThreePartial"]), ("DDDD, CCCC and BBBB", "1"))

    def test_the_first_attempt_numbers_and_the_zero_sensitivity(self):
        merged = pd.DataFrame({"method": ["mica", "mica"], "entry": ["CCCC", "DDDD"], "conflict": [True, True]})
        m = self.macros(merged=merged)
        self.assertEqual((m["FirstModels"], m["FirstTmMean"]), ("2", "0.97"))                       # AAAA 0.99 and BBBB 0.95: the two entries that were not rerun
        self.assertAlmostEqual(float(m["FirstZeroMean"]), 0.485, delta=0.006)                         # (0.99 + 0.95) / 4 entries, the reruns count as 0
        zero = pd.DataFrame([{"policy": "strict", "group": "pooled", "metric": "tm_score", "method": "mica", "role": "mean_over_entries", "n": 20, "mean": 0.796, "mean_ci_low": 0.64,
                              "mean_ci_high": 0.93},
                             {"policy": "strict", "group": "pooled", "metric": "tm_score", "method": "emprot", "role": "paired_vs_mica", "mean_diff": 0.115, "diff_ci_low": 0.03,
                              "diff_ci_high": 0.24, "wilcoxon_p": 0.000008, "wilcoxon_p_holm": 0.000031},
                             {"policy": "all_missing", "group": "pooled", "metric": "tm_score", "method": "mica", "role": "mean_over_entries", "n": 20, "mean": 0.1, "mean_ci_low": 0.0,
                              "mean_ci_high": 0.2}])
        m = self.macros(zero=zero)
        self.assertEqual((m["ZeroMeanMica"], m["ZeroLoMica"], m["ZeroNMica"]), ("0.80", "0.64", "20"))     # the strict policy, not the other one
        self.assertEqual((m["ZeroDiffEmprot"], m["ZeroPwHolmEmprot"]), ("0.12", "$<$0.001"))

    def test_chain_counts_and_the_smallest_identity_p(self):
        metrics = pd.DataFrame([{"pdb_id": "AAAA", "method": "modelangelo", "status": "model", "tm_score": 0.9, "n_chains_pred": 4, "n_chains_native": 1},
                                {"pdb_id": "BBBB", "method": "modelangelo", "status": "model", "tm_score": 0.8, "n_chains_pred": 67, "n_chains_native": 2},
                                {"pdb_id": "CCCC", "method": "modelangelo", "status": "failed", "tm_score": float("nan"), "n_chains_pred": 99, "n_chains_native": 9}])
        cor = pd.DataFrame({"method": ["mica", "emodelx", "mica"], "x": ["max_id_to_pre2026_pdb", "max_id_to_pre2026_pdb", "resolution_A"], "y": ["tm_score"] * 3, "n": [17, 18, 17],
                            "spearman_rho": [0.02, -0.42, -0.25], "p": [0.94, 0.08, 0.325]})
        m = self.macros(metrics=metrics, correlations=cor)
        self.assertEqual((m["ChainsMedAngelo"], m["ChainsMaxAngelo"], m["ChainsNativeMax"]), ("36", "67", "2"))      # the failed run does not count
        self.assertEqual(m["RhoIdentMinP"], "0.080")
        text = T.corr_table(cor)
        self.assertIn("MICA & 17 & $-$0.25 (0.325) & -- & 0.02 (0.940)", text)
        self.assertIn("EModelX(+AF) & 18 & -- & -- & $-$0.42 (0.080)", text)
        self.assertIn("Not available", T.corr_table(pd.DataFrame()))

    def test_the_second_highest_coverage_is_a_macro(self):
        metrics = pd.DataFrame([{"pdb_id": "AAAA", "method": "mica", "status": "model", "residue_coverage": 4.757},
                                {"pdb_id": "BBBB", "method": "mica", "status": "model", "residue_coverage": 1.1801},
                                {"pdb_id": "CCCC", "method": "mica", "status": "model", "residue_coverage": 1.0},
                                {"pdb_id": "DDDD", "method": "mica", "status": "failed", "residue_coverage": 9.0},       # a failed run does not count
                                {"pdb_id": "AAAA", "method": "emprot", "status": "model", "residue_coverage": 25.11}])  # one model only: no second value
        m = self.macros(metrics=metrics)
        self.assertEqual(m["CovSecondMica"], "1.18")
        self.assertNotIn("CovSecondEmprot", m)

    def test_nothing_to_quote_gives_no_macro_and_no_crash(self):
        m = T.Macros()
        T.extra_macros(m, {"summary": pd.DataFrame(), "paired": pd.DataFrame(), "metrics": pd.DataFrame(), "runtime": pd.DataFrame()}, None)
        self.assertEqual(m.items, {})

    def test_listing(self):
        self.assertEqual((T.listing(["a"]), T.listing(["a", "b"]), T.listing(["a", "b", "c"]), T.listing([])), ("a", "a and b", "a, b and c", "none"))

    def test_the_two_new_tables(self):
        var = pd.DataFrame([{"method": "mica", "model_kind": "mica_all_atom", "n_models": 17, "tm_score_median": 0.94, "tm_score_mean": 0.93, "tm_score_fns_median": 0.96,
                             "ca_match_median": 94.9, "coverage_median": 1.02, "ca_quality_median": 96.0}])
        text = T.variants_table(var)
        self.assertIn("MICA, all-atom model (scored) & 17 & 0.94 & 0.93 & 0.96 & 94.9 & 1.02 & 96.0", text)
        self.assertIn("No variants", T.variants_table(pd.DataFrame()))
        text = T.fns_table(self.summary(), self.paired())
        self.assertIn("MICA & 3 & 0.95 (0.80--0.99) & -- & -- & --", text)
        self.assertIn("ModelAngelo & 3 &", text)
        self.assertIn("0.500", text)
        self.assertIn("Not available", T.fns_table(pd.DataFrame(), pd.DataFrame()))


class SessionTests(unittest.TestCase):
    def test_one_row_per_zip_with_status_counts_and_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = make_raw(tmp, [("mica_gpu", "20261027-0426", {"mica_status.csv": pd.DataFrame({"entry": ["A", "B", "C"], "status": ["ok", "ok", "failed"]}),
                                                               "run_info.json": {"started_utc": "2026-10-27T02:04:03+00:00", "elapsed_minutes": 120, "complete": True, "gpu": "Tesla T4, 15360 MiB"}}),
                                 ("mica_gpu2_a", "20261027-1949_INCOMPLETE", {"mica_status.csv": pd.DataFrame({"entry": ["A"], "status": ["interrupted"]}),
                                                                             "run_info.json": {"elapsed_minutes": 60, "complete": False, "gpu": "Tesla T4"}}),
                                 ("scoring_base", "20261027-0235", {"scoring/scores_long.csv": "a,b\n1,2\n3,4\n", "run_info.json": {"elapsed_minutes": 30, "gpu": ""}})])
            text = T.sessions_table(raw)
            self.assertIn("\\texttt{mica\\_gpu} & 2026-10-27 02:04 & 120 & yes & ok 2, failed 1", text)
            self.assertIn("interrupted 1 (cut by the time budget)", text)
            self.assertIn("2 models scored", text)
            m = T.Macros()
            T.session_macros(m, raw)
            self.assertEqual((m.items["NSessions"], m.items["GpuSessionHours"], m.items["CpuSessionHours"]), ("3", "3.0", "0.5"))

    def test_a_session_that_stopped_at_a_check_is_not_called_cut_by_the_time_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = make_raw(tmp, [("scoring_x", "20261108-0459_INCOMPLETE", {"run_info.json": {"elapsed_minutes": 13, "complete": False, "gpu": "", "problem": "the inputs hold no model"}}),
                                 ("gpu_y", "20261108-0417_INCOMPLETE", {"mica_status.csv": pd.DataFrame({"entry": ["A"], "status": ["interrupted"]}),
                                                                       "run_info.json": {"elapsed_minutes": 60, "complete": False, "gpu": "Tesla T4"}})])
            text = T.sessions_table(raw)
            self.assertIn("-- (stopped by a check of the notebook)", text)
            self.assertIn("interrupted 1 (cut by the time budget)", text)
            self.assertEqual(text.count("cut by the time budget"), 1)

    def test_no_zip_gives_a_sentence_and_no_macro(self):
        self.assertIn("No result zip", T.sessions_table(""))
        m = T.Macros()
        T.session_macros(m, "")
        self.assertEqual(m.items, {})


class BuildTests(unittest.TestCase):
    def test_every_table_and_the_macros_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = TA.Fixture(tmp)
            results, out = os.path.join(tmp, "results"), os.path.join(tmp, "generated")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(A.main(["--entries", fx.entries, "--status"] + fx.status + ["--scores"] + fx.scores + ["--out", results, "--resamples", "200"]), 0)
            raw = make_raw(tmp, zips())
            files = T.build(results, raw, os.path.join(REPO, "data"), out)
            for name in ("tab_completion", "tab_summary", "tab_summary_strata", "tab_summary_all", "tab_paired", "tab_zero", "tab_runtime", "tab_per_entry", "tab_failures", "tab_entries",
                         "tab_docking", "tab_pins", "tab_summary_modes", "tab_paired_modes", "tab_methods_screen", "tab_pool_flow", "tab_mica_modes", "tab_summary_nodock",
                         "tab_paired_nodock", "tab_dock_effect"):
                self.assertIn(name + ".tex", files)
                self.assertTrue(os.path.getsize(os.path.join(out, name + ".tex")) > 0, name)
            with open(os.path.join(out, "numbers.tex"), encoding="utf-8") as f:
                lines = [l for l in f.read().splitlines() if l and not l.startswith("%")]
            names = {}
            for l in lines:
                mt = re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", l)
                self.assertIsNotNone(mt, l)
                names[mt.group(1)] = mt.group(2)
            for need in ("TmMedMica", "TmMedAngelo", "PwHolmTmAngelo", "DockDockedThree", "CommitMica", "PhenixVersion", "NScreened", "PoolAll", "CompListMica", "NEntries", "DockEffectN",
                         "DockEffectMean", "DockEffectP"):
                self.assertIn(need, names)
            self.assertNotIn("NdTmMedMica", names)                        # no analysis without docked input in this folder
            with open(os.path.join(out, "tab_dock_effect.tex"), encoding="utf-8") as f:
                self.assertIn("No entry has a result", f.read())
            with open(os.path.join(out, "tab_summary_nodock.tex"), encoding="utf-8") as f:
                self.assertIn("has not been run", f.read())
            for k, v in names.items():
                self.assertNotRegex(v, r"\bnan\b|None|inf", k)
            with open(os.path.join(out, "numbers.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f), names)

    def test_the_analysis_without_docked_input_gets_its_own_tables_and_macros(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = TA.Fixture(tmp)
            results, out = os.path.join(tmp, "results"), os.path.join(tmp, "generated")
            with redirect_stdout(io.StringIO()):
                for target in (results, os.path.join(results, "nodock")):
                    self.assertEqual(A.main(["--entries", fx.entries, "--status"] + fx.status + ["--scores"] + fx.scores + ["--out", target, "--resamples", "100"]), 0)
            T.build(results, make_raw(tmp, zips()), os.path.join(REPO, "data"), out)
            with open(os.path.join(out, "numbers.json"), encoding="utf-8") as f:
                names = json.load(f)
            self.assertEqual(names["NdTmMedMica"], names["TmMedMica"])        # the same input twice: the same numbers
            self.assertIn("NdPwHolmTmAngelo", names)
            self.assertEqual(names["DockEffectN"], "n/a")                      # the fixture's entries are not those of the real list, so nothing pairs up (the pairing has its own tests)
            with open(os.path.join(out, "tab_summary_nodock.tex"), encoding="utf-8") as f:
                self.assertIn("MICA, no docked input", f.read())

    def test_it_also_runs_without_any_raw_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = TA.Fixture(tmp)
            results, out = os.path.join(tmp, "results"), os.path.join(tmp, "generated")
            with redirect_stdout(io.StringIO()):
                A.main(["--entries", fx.entries, "--status"] + fx.status + ["--scores"] + fx.scores + ["--out", results, "--resamples", "100"])
            T.build(results, "", os.path.join(REPO, "data"), out)
            self.assertTrue(os.path.exists(os.path.join(out, "numbers.tex")))


if __name__ == "__main__":
    unittest.main()
