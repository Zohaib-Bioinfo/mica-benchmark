#!/usr/bin/env python3
"""Tests for make_mica_notebook.py and the notebooks it builds (MICA's own programs on the benchmark entries).

Fast tests: the notebooks build, every cell compiles, the embedded files equal the repository files, the settings are in the notebook, the shards and the list of what is left.

End-to-end tests run a whole notebook as a script against stand-ins (the ideas of test_pilot_notebook.py): a Phenix download server with a fake installer whose programs
`phenix.dock_in_map` and `phenix.real_space_refine` behave as the tests tell them, a local git repository that plays MICA (MICA's own utils/dock_in_map.py when
MICA_TEST_MICA_DIR is set, so that the docking driver runs against the real code; programs 1 and 3 are stand-ins; the stand-in of program 3 fails when two run at the same time,
as MICA's does), a file server for the map, the deposited model and the weights. They need

    MICA_TEST_ENV_PYTHON   a Python with Biopython, mrcfile, numpy, scipy (the MICA environment)
    MICA_TEST_AF3_DIR      the folder with the 26 AlphaFold Server files (private)
    MICA_TEST_MICA_DIR     a checkout of MICA at the pinned commit (only utils/dock_in_map.py is used)
    USALIGN                the compiled US-align program

    python -W error -m unittest test_mica_notebook
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_main_notebook as MN  # noqa: E402
import make_mica_notebook as mm  # noqa: E402
import test_dockdiag_notebook as DD  # noqa: E402  (the stand-ins of the docking diagnostic)
import test_pilot_notebook as T  # noqa: E402

ENV_PY, AF3_DIR, USALIGN = T.ENV_PY, T.AF3_DIR, T.USALIGN
MICA_SRC = os.environ.get("MICA_TEST_MICA_DIR")
USER, PASSWORD = T.USER, T.PASSWORD
TEST_ENTRIES = ["26DK", "9T5K", "22FX"]

# the stand-in of Phenix: the one of the docking diagnostic, with a docking that centres the model in the map and a refinement that takes time and writes MICA's file name
STUB_PHENIX = DD.STUB_PHENIX
_dock_old = '''    shift = 1.0 if kv.get("resolution") == "6.0" else 0.0
    with open(kv["search_model"]) as src, open(kv["pdb_out"], "w") as out:
        for line in src:
            if shift and line.startswith(("ATOM  ", "HETATM")):
                line = line[:30] + "%8.3f" % (float(line[30:38]) + shift) + line[38:]
            out.write(line)
    say("Final score 0.9")
'''
_dock_new = '''    lines = open(kv["search_model"]).read().splitlines()
    xyz = [(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in lines if l.startswith(("ATOM  ", "HETATM"))]
    mean = [sum(c[i] for c in xyz) / len(xyz) for i in range(3)]
    with open(kv["pdb_out"], "w") as out:
        for l in lines:
            if l.startswith(("ATOM  ", "HETATM")):
                x, y, z = (float(l[30:38]) - mean[0] + 48.0, float(l[38:46]) - mean[1] + 48.0, float(l[46:54]) - mean[2] + 48.0)
                l = l[:30] + "%8.3f%8.3f%8.3f" % (x, y, z) + l[54:]
            out.write(l + "\\n")
    say("Wrote placed model with CC_mask (local CC) = 0.61")
'''
assert STUB_PHENIX.count(_dock_old) == 1
STUB_PHENIX = STUB_PHENIX.replace(_dock_old, _dock_new)
_refine_old = '''elif prog == "phenix.real_space_refine":
    pass
'''
_refine_new = '''elif prog == "phenix.real_space_refine":
    time.sleep(float(os.environ.get("STUB_REFINE_SECONDS", "0.3")))
    if not os.environ.get("STUB_NO_REFINED"):
        shutil.copy(args[0], os.path.basename(args[0])[:-4] + "_real_space_refined_000.pdb")
'''
assert STUB_PHENIX.count(_refine_old) == 1
STUB_PHENIX = STUB_PHENIX.replace(_refine_old, _refine_new)

STUB_RUN = r'''
import argparse, json, os, re, shutil, subprocess, sys, time
p = argparse.ArgumentParser()
p.add_argument("-m", required=True); p.add_argument("-f", required=True); p.add_argument("-a", required=True)
p.add_argument("--run_pulchra", action="store_true"); p.add_argument("--pulchra_path"); p.add_argument("--run_phenix", action="store_true")
p.add_argument("--phenix_act"); p.add_argument("--resolution", type=float); p.add_argument("--device", default="cuda"); p.add_argument("--phenix_param", default="modules/phenix.eff")
a = p.parse_args()
assert os.path.exists("trained_models/MICA_best_model.pth"), "weights missing"
assert os.path.exists(a.pulchra_path) and os.access(a.pulchra_path, os.X_OK), "pulchra missing"
entry_dir = os.path.dirname(a.a)
docked_present = os.path.exists(os.path.join(entry_dir, os.path.basename(entry_dir) + "_af3_docked.pdb"))      # optional in MICA: without it the AlphaFold channels of the network are zeros
assert os.path.isdir(os.path.join(entry_dir, "AF3_structures")), "program 3 needs the processed structures"
bt = re.search(r"self[.]batch_threshold = ([0-9]+)", open("utils/predict.py").read()) if os.path.exists("utils/predict.py") else None
with open(os.path.join(entry_dir, "p3_call.json"), "w") as f:
    json.dump({"device": a.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "omp": os.environ.get("OMP_NUM_THREADS"), "cwd": os.getcwd(), "docked_present": docked_present, "batch_threshold": int(bt.group(1)) if bt else None}, f)
if os.environ.get("STUB_FAIL_STEP") == "3_build":
    print("boom in step 3"); sys.exit(1)
grids = os.path.join("output", "results", "predicted_grids")           # MICA's program 3 deletes and rewrites this folder: two at the same time break each other
if os.path.exists(grids):
    print("COLLISION: another program 3 is running in this folder"); sys.exit(7)
os.makedirs(grids)
time.sleep(float(os.environ.get("STUB_P3_SECONDS", "0.3")))
map_id = os.path.basename(a.m).split("emd_")[-1].split(".")[0]; pdb_id = os.path.basename(a.f).split(".fasta")[0]
src = os.path.join(entry_dir, "AF3_structures"); first = sorted(os.listdir(src))[0]
model = os.path.join(src, first, "ranked_0.pdb"); os.makedirs("output", exist_ok=True)
ca = "output/%s_%s_MICA_ca_model.pdb" % (map_id, pdb_id); full = "output/%s_%s_MICA_all_atom_model.pdb" % (map_id, pdb_id)
lines = open(model).read().splitlines()
open(ca, "w").write("\n".join(l for l in lines if l.startswith("ATOM") and l[12:16].strip() == "CA") + "\nEND\n")
shutil.copy(model, full)
shutil.rmtree(grids)
if a.run_phenix:                                                         # like MICA's modules/phenix.sh: the refinement runs in the background and run.py ends first
    cmd = 'bash modules/phenix.sh %s %s "phenix.real_space_refine %s %s %s resolution=%s"' % (os.path.abspath(a.phenix_act), os.path.abspath("output"), os.path.abspath(full), os.path.abspath(a.m), os.path.abspath(a.phenix_param), a.resolution)
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
'''

STUB_DOCK_FALLBACK = T.STUB_MICA["utils/dock_in_map.py"]


def stub_mica():
    files = dict(T.STUB_MICA)
    files["utils/process_AF3_results.py"] = DD.STUB_PROGRAM_1
    files["run.py"] = STUB_RUN
    files["utils/predict.py"] = T.STUB_MICA["utils/predict.py"].rstrip("\n") + "\n\n\nclass Predictor:\n    def __init__(self):\n        self.batch_threshold = 200          # as in MICA's utils/predict.py\n"
    if MICA_SRC:
        with open(os.path.join(MICA_SRC, "utils", "dock_in_map.py"), encoding="utf-8") as f:
            files["utils/dock_in_map.py"] = f.read()
    return files


FAKE_INSTALLER = DD.FAKE_INSTALLER


def committed(name):
    return os.path.join(HERE, name + ".ipynb")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="micabuild_")
        cls.entries = ["26DK", "9T5K", "22FX", "9WQ6"]
        cls.nbs = {}
        for kind in ("cpu", "gpu", "dock"):
            path = os.path.join(cls.tmp, kind + ".ipynb")
            cls.expected = mm.build(path, kind="gpu" if kind == "gpu" else "cpu", entries=cls.entries, stages=(1, 2) if kind == "dock" else None)
            cls.nbs[kind] = nbformat.read(path, as_version=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cell(self, kind, cid):
        return next(c for c in self.nbs[kind].cells if c["id"] == cid).source

    def test_cells_compile_and_have_no_placeholders(self):
        import re
        for kind, nb in self.nbs.items():
            for c in nb.cells:
                if c.cell_type == "code" and not c.source.startswith("%%writefile"):
                    compile(c.source, c["id"], "exec")
                self.assertEqual(re.findall(r"@[A-Z_0-9]+@", c.source), [], (kind, c["id"]))

    def test_cell_order(self):
        for kind, nb in self.nbs.items():
            ids = [c["id"] for c in nb.cells if c.cell_type == "code" and not c["id"].startswith("file-")]
            self.assertEqual(ids, ["settings", "step1", "step2-dirs", "step2-check", "robust", "step3", "step4", "step5", "step6", "step7", "final"], kind)

    def test_embedded_files_equal_the_repository_files(self):
        for kind, nb in self.nbs.items():
            got = {}
            for c in nb.cells:
                if c.source.startswith("%%writefile"):
                    first, text = c.source.split("\n", 1)
                    got[first.split(None, 1)[1].strip()] = text
            self.assertEqual(sorted(got), sorted(t for _s, t in mm.FILES))
            for src, target in mm.FILES:
                with open(os.path.join(ROOT, src), encoding="utf-8") as f:
                    text = f.read()
                self.assertEqual(got[target], text if text.endswith("\n") else text + "\n", target)
                self.assertEqual(self.expected[target], hashlib.sha256(got[target].encode()).hexdigest())
        self.assertIn("scripts/mica_dock_driver.py", [t for _s, t in mm.FILES])

    def test_the_gpu_is_required_only_by_the_gpu_notebook(self):
        self.assertIn("REQUIRE_GPU = False", self.cell("cpu", "step1"))
        self.assertIn("REQUIRE_GPU = True", self.cell("gpu", "step1"))
        self.assertIn("STAGES = (1, 2, 3)", self.cell("cpu", "step6"))
        self.assertIn("STAGES = (1, 2)", self.cell("dock", "step6"))
        self.assertIn("STAGES = (3,)", self.cell("gpu", "step6"))
        self.assertIn('P3_DEVICE = "auto"', self.cell("cpu", "step6"))
        self.assertIn('P3_DEVICE = "cuda"', self.cell("gpu", "step6"))

    def test_the_weights_are_fetched_when_program_3_runs(self):
        for kind in ("cpu", "gpu"):
            self.assertIn("if fetch_mica() and fetch_weights() and import_mica():", self.cell(kind, "step5"))

    def test_the_commands_of_mica_are_the_ones_of_the_pipeline(self):
        step6 = self.cell("cpu", "step6")
        self.assertIn('[ENV_PY, "utils/process_AF3_results.py", "-f", fasta, "-a", af3, "-d", "cpu"]', step6)
        self.assertIn('[ENV_PY, DRIVER, "-m", map_path, "-c", str(contour), "-r", str(resolution), "-f", fasta, "-a", af3, "--phenix_act", STATE["phenix_env"]]', step6)
        self.assertIn('cmd = [ENV_PY, "run.py", "-m", map_path, "-f", fasta, "-a", af3, "--run_pulchra", "--pulchra_path=modules/pulchra304/src/pulchra", "--run_phenix",', step6)
        self.assertIn('"--phenix_act=" + STATE["phenix_env"], "--resolution=" + str(resolution), "--device=" + device]', step6)
        pipeline = ("process_AF3_results.py", "dock_in_map.py", "run.py")
        for name in pipeline:
            self.assertTrue(name in step6 or name == "dock_in_map.py", name)

    def test_the_docking_attempts_and_limits_are_in_the_notebook(self):
        for kind in ("cpu", "dock", "gpu"):
            step6 = self.cell(kind, "step6")
            self.assertIn("DOCK_ATTEMPTS = " + repr(mm.DOCK_ATTEMPTS), step6)
            self.assertIn('"2_dock": %d' % mm.DOCK_STAGE_CAP_S, step6)
        a = mm.DOCK_ATTEMPTS
        self.assertEqual([x["name"] for x in a], ["res6_quick", "res10_quick"])
        self.assertEqual([x.get("search_resolution") for x in a], [6.0, 10.0])
        self.assertEqual([x["timeout_s"] for x in a], [2400, 1500])
        self.assertEqual([x.get("skip_after_timeout") for x in a], [None, None])
        for x in a:
            self.assertEqual(x["set"], {"dock_with_mr": "False", "quick": "True"})
        self.assertEqual(mm.DOCK_STAGE_CAP_S, 8 * 3600)
        # pass 1 stays in the file as the record of what ran on 2026-10-24
        self.assertEqual([x["name"] for x in mm.DOCK_ATTEMPTS_PASS1], ["no_mr_quick", "no_mr_thorough"])
        self.assertEqual([x["name"] for x in mm.DOCK_ATTEMPTS_PASS2], ["res6_quick", "res10_quick", "plain_quick"])
        self.assertEqual(mm.DOCK_ATTEMPTS_PASS1[0]["timeout_s"], 2700)

    def test_every_phenix_parameter_of_every_attempt_is_one_that_phenix_accepted(self):
        """Pass 2 lost a whole session cycle because `rigid_body_refinement_resolution` (a True/False parameter) was rejected: the fake Phenix of the tests cannot know that, so
        the parameters are checked against the command line of a call that really ran on Phenix 2.2.1 (the logs of pass 1 and 2 in results/raw, when they are there)."""
        worked = {"search_model", "map_file", "nproc", "resolution", "pdb_out", "dock_chains_individually", "sequence", "dock_with_mr", "quick"}
        sys.path.insert(0, HERE)
        import mica_dock_driver as driver
        for a in mm.DOCK_ATTEMPTS:
            used = set(a["set"]) | set(driver.derived_settings(a, 1.24)) | set(a["set"])
            self.assertLessEqual(used, worked, "%s uses %s" % (a["name"], sorted(used - worked)))
            self.assertEqual(set(driver.derived_settings(a, 1.24)) - {"resolution"}, set())
        import glob as _glob
        logs = sorted(_glob.glob(os.path.join(ROOT, "results", "raw", "mica_dock2_*", "docking_logs", "*", "*_attempt3.log")))
        if logs:
            keys, on = set(), False
            with open(logs[0], errors="replace") as f:
                for line in f:
                    if "Adding command-line PHIL" in line:
                        on = True
                    elif on and "=" in line:
                        keys.add(line.strip().split("=", 1)[0])
                    elif on and not line.strip() and keys:
                        break
            self.assertEqual(keys, worked, "the parameters of the real call that ran changed: %s" % sorted(keys ^ worked))

    def test_the_search_resolution_of_every_attempt_is_coarser_than_every_map(self):
        with open(os.path.join(ROOT, "data", "mica_map_geometry.csv"), newline="", encoding="utf-8") as f:
            res = [float(r["resolution"]) for r in csv.DictReader(f)]
        self.assertEqual(len(res), 20)
        self.assertLess(max(res), 6.0)

    def test_the_state_profile_of_the_notebooks(self):
        self.assertIn("DOCK_PROFILE = \"pass3\"", self.cell("dock", "step6"))
        self.assertIn("STATE_PROFILE = None", self.cell("dock", "step6"))
        self.assertIn("STATE_PROFILE = None", self.cell("cpu", "step6"))
        self.assertIn("STATE_PROFILE = 'pass3'", self.cell("gpu", "step6"))
        self.assertIn("profile `pass3`", self.cell("gpu", "intro"))
        self.assertIn("not run in this session", self.cell("dock", "intro"))
        self.assertIn("res6_quick: search at 6 A, at most 40 minutes", self.cell("dock", "intro"))
        self.assertIn("dock_profile", mm.SUMMARY)

    def test_the_four_cpu_sessions_of_pass_3_hold_every_entry_once_and_are_balanced(self):
        order = MN.seeded_order(MN.read_entries())
        got = sum(mm.PASS2_SHARDS.values(), [])
        self.assertEqual(sorted(got), sorted(order))
        self.assertEqual(len(got), 20)
        with open(os.path.join(ROOT, "data", "mica_map_geometry.csv"), newline="", encoding="utf-8") as f:
            domains = {r["entry"]: int(r["domains"]) for r in csv.DictReader(f)}
        totals = sorted(sum(domains[e] for e in v) for v in mm.PASS2_SHARDS.values())
        self.assertEqual(totals, [14, 15, 15, 15])
        for v in mm.PASS2_SHARDS.values():                               # the heaviest entry of a session starts first
            self.assertEqual(domains[v[0]], max(domains[e] for e in v))

    def test_the_gpu_session_runs_the_smallest_maps_first(self):
        order = mm.gpu_order(MN.seeded_order(MN.read_entries()))
        with open(os.path.join(ROOT, "data", "mica_map_geometry.csv"), newline="", encoding="utf-8") as f:
            windows = {r["entry"]: int(r["windows"]) for r in csv.DictReader(f)}
        self.assertEqual(sorted(order), sorted(windows))
        self.assertEqual([windows[e] for e in order], sorted(windows.values()))
        self.assertEqual((order[0], order[-1]), ("9VUH", "9VI4"))

    def test_the_settings_cell_has_the_hours(self):
        self.assertIn("HOURS = 10.5", self.cell("cpu", "settings"))
        self.assertIn("HOURS = 9.0", self.cell("gpu", "settings"))

    def test_the_sensitivity_notebook_is_the_gpu_notebook_plus_the_removal_of_the_docked_input(self):
        path = os.path.join(self.tmp, "nodock.ipynb")
        mm.build(path, kind="gpu", entries=self.entries, nodock=True)
        nb = nbformat.read(path, as_version=4)
        cell = lambda cid: next(c for c in nb.cells if c["id"] == cid).source
        self.assertNotIn("FORCE_NO_DOCK", self.cell("gpu", "step6"))                        # the main GPU notebook is untouched
        self.assertNotIn("remove_docked_input", self.cell("gpu", "step6"))
        for text in (cell("step6"),):
            self.assertIn("FORCE_NO_DOCK = True", text)
            self.assertIn("_stage_3_with_docking = stage_3", text)
            self.assertIn("def remove_docked_input(entry, folder):", text)
            self.assertLess(text.index("def stage_3(entry, tag"), text.index("_stage_3_with_docking = stage_3"))       # the original is defined before it is wrapped
            self.assertLess(text.index("def stage_3(entry, tag", text.index("_stage_3_with_docking = stage_3")), text.index("if STATE[\"ok\"] and STATE[\"usalign\"] and STATE[\"phenix_env\"]:\n    try:\n        check_af3_present"))
        self.assertIn("HOURS = 5.0", cell("settings"))
        self.assertIn("session `mica_gpu_nodock`", cell("intro"))
        self.assertIn("sensitivity session", cell("intro"))
        self.assertIn('"force_no_dock": True', cell("final") + cell("step7") + cell("step6"))
        compile(cell("step6"), "step6", "exec")
        with self.assertRaises(ValueError):
            mm.build(os.path.join(self.tmp, "x.ipynb"), kind="cpu", entries=self.entries, nodock=True)

    def test_the_pass_4_sessions_cover_the_twelve_entries_that_did_not_run_and_are_balanced(self):
        a, b = mm.PASS4_SHARDS["a"], mm.PASS4_SHARDS["b"]
        did_not_run = {"9WQ6", "22FX", "9V4D", "9TKA", "9VBD", "9WTX", "26DK", "9Z24", "9T5K", "9OC0", "13BD", "9VI4"}      # failed (OOM) or deferred in the GPU sessions of 2026-10-27
        self.assertEqual((set(a) | set(b), len(a) + len(b)), (did_not_run, 12))
        with open(os.path.join(ROOT, "data", "mica_map_geometry.csv"), newline="", encoding="utf-8") as f:
            windows = {r["entry"]: int(r["windows"]) for r in csv.DictReader(f)}
        wa, wb = sum(windows[e] for e in a), sum(windows[e] for e in b)
        self.assertEqual((wa, wb), (2185, 1935))
        self.assertLess(abs(wa - wb) / max(wa, wb), 0.15)
        self.assertEqual(mm.gpu_order(a), ["9WQ6", "9V4D", "9T5K", "9OC0", "9VI4"])
        self.assertEqual(mm.gpu_order(b)[-2:], ["9Z24", "13BD"])

    def test_the_batch_threshold_is_raised_only_in_the_notebooks_that_ask_for_it(self):
        self.assertNotIn("patch_batch_threshold", self.cell("gpu", "step6"))                # the notebooks of 2026-10-27 are unchanged
        for nodock in (False, True):
            path = os.path.join(self.tmp, "p4_%s.ipynb" % nodock)
            mm.build(path, kind="gpu", entries=self.entries, nodock=nodock, batch_threshold=mm.BATCH_THRESHOLD_NEW)
            nb = nbformat.read(path, as_version=4)
            text = next(c for c in nb.cells if c["id"] == "step6").source
            compile(text, "step6", "exec")
            self.assertIn("BATCH_THRESHOLD = 1000000", text)
            self.assertIn('old = "self.batch_threshold = 200"', text)
            self.assertLess(text.index("def patch_batch_threshold():"), text.index("patch_batch_threshold()\n            run_all()"))
            self.assertEqual(text.count("patch_batch_threshold()\n            run_all()"), 1)
            self.assertEqual("FORCE_NO_DOCK = True" in text, nodock)
            self.assertIn('"batch_threshold": globals().get("BATCH_THRESHOLD")', "\n".join(c.source for c in nb.cells if c.cell_type == "code"))

    def test_the_pass_5_session_runs_the_two_entries_that_the_time_budget_cut(self):
        self.assertEqual(mm.gpu_order(mm.PASS5_ENTRIES), ["9OC0", "9VI4"])
        self.assertTrue(set(mm.PASS5_ENTRIES) <= set(mm.PASS4_SHARDS["a"]))              # both were in mica_gpu2_a and were interrupted there
        path = os.path.join(self.tmp, "p5.ipynb")
        mm.build(path, kind="gpu", name="mica_gpu3", entries=mm.gpu_order(mm.PASS5_ENTRIES), hours=mm.PASS5_HOURS, batch_threshold=mm.BATCH_THRESHOLD_NEW)
        nb = nbformat.read(path, as_version=4)
        code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        text = next(c for c in nb.cells if c["id"] == "step6").source
        compile(text, "step6", "exec")
        self.assertIn("ENTRIES = ['9OC0', '9VI4']", text)
        self.assertIn("HOURS = 4.2", code)
        self.assertIn("BATCH_THRESHOLD = 1000000", text)
        self.assertNotIn("FORCE_NO_DOCK = True", text)                                      # the main session, with the docked input
        self.assertIn("mica_gpu3", nb.cells[0].source)

    def test_the_entries_are_in_the_notebook(self):
        self.assertIn("ENTRIES = " + repr(self.entries), self.cell("cpu", "step6"))

    def test_shards_share_the_seeded_order_round_robin(self):
        order = MN.seeded_order(MN.read_entries())
        shards = [mm.shard_entries(order, i, 3) for i in range(3)]
        self.assertEqual(sorted(sum(shards, [])), sorted(order))
        self.assertEqual([s[0] for s in shards], order[:3])
        self.assertEqual(shards[0], order[0::3])
        self.assertEqual(len(order), 20)
        self.assertEqual([len(s) for s in shards], [7, 7, 6])

    def test_what_is_left_leaves_out_ok_and_failed_only(self):
        path = os.path.join(self.tmp, "status.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entry", "status"])
            for e, s in (("A", "ok"), ("B", "failed"), ("C", "deferred"), ("D", "docked"), ("E", "interrupted"), ("F", "not run")):
                w.writerow([e, s])
        self.assertEqual(mm.pending_entries(list("ABCDEFG"), [path]), ["C", "D", "E", "F", "G"])
        self.assertEqual(mm.pending_entries(list("ABCDEFG"), [path], redo=["B"]), ["B", "C", "D", "E", "F", "G"])

    def test_the_committed_notebooks_are_current(self):
        # pass 1 (mica_cpu_a/b/c) and pass 2 (mica_dock2_a/b/c/d) are in as_run/: they are the files that ran and are not rebuilt
        for letter, entries in mm.PASS3_SHARDS.items():
            name = "mica_dock3_" + letter
            if not os.path.exists(committed(name)):
                continue
            fresh = os.path.join(self.tmp, name + ".ipynb")
            mm.build(fresh, kind="cpu", name=name, entries=entries, stages=(1, 2))
            a, b = nbformat.read(fresh, as_version=4), nbformat.read(committed(name), as_version=4)
            self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_mica_notebook.py --pass3 again (%s)" % name)
        if os.path.exists(committed("mica_gpu")):
            fresh = os.path.join(self.tmp, "mica_gpu.ipynb")
            mm.build(fresh, kind="gpu", name="mica_gpu", entries=mm.gpu_order(MN.seeded_order(MN.read_entries())))
            a, b = nbformat.read(fresh, as_version=4), nbformat.read(committed("mica_gpu"), as_version=4)
            self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_mica_notebook.py --kind gpu again")
        for variant, nodock in (("", False), ("nodock_", True)):
            for letter, entries in mm.PASS4_SHARDS.items():
                name = "mica_gpu2_%s%s" % (variant, letter)
                if not os.path.exists(committed(name)):
                    continue
                fresh = os.path.join(self.tmp, name + ".ipynb")
                mm.build(fresh, kind="gpu", name=name, entries=mm.gpu_order(entries), hours=mm.PASS4_HOURS["nodock" if nodock else "main"], nodock=nodock, batch_threshold=mm.BATCH_THRESHOLD_NEW)
                a, b = nbformat.read(fresh, as_version=4), nbformat.read(committed(name), as_version=4)
                self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_mica_notebook.py --pass4 again (%s)" % name)
        if os.path.exists(committed("mica_gpu3")):
            fresh = os.path.join(self.tmp, "mica_gpu3.ipynb")
            mm.build(fresh, kind="gpu", name="mica_gpu3", entries=mm.gpu_order(mm.PASS5_ENTRIES), hours=mm.PASS5_HOURS, batch_threshold=mm.BATCH_THRESHOLD_NEW)
            a, b = nbformat.read(fresh, as_version=4), nbformat.read(committed("mica_gpu3"), as_version=4)
            self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_mica_notebook.py --pass5 again")
        if os.path.exists(committed("mica_gpu_nodock")):
            fresh = os.path.join(self.tmp, "mica_gpu_nodock.ipynb")
            mm.build(fresh, kind="gpu", name="mica_gpu_nodock", entries=mm.gpu_order(MN.seeded_order(MN.read_entries())), nodock=True)
            a, b = nbformat.read(fresh, as_version=4), nbformat.read(committed("mica_gpu_nodock"), as_version=4)
            self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_mica_notebook.py --nodock again")


@unittest.skipUnless(ENV_PY and AF3_DIR and USALIGN and any("fold_26dk_e1_model_0.cif" in f for _r, _d, f in os.walk(AF3_DIR or "")),
                     "needs MICA_TEST_ENV_PYTHON, MICA_TEST_AF3_DIR (with the 26DK file) and USALIGN")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="micarun_")
        cif = [os.path.join(r, f) for r, _d, fs in os.walk(AF3_DIR) for f in fs if f == "fold_26dk_e1_model_0.cif"][0]
        with open(cif, "rb") as f:
            cif_bytes = f.read()
        mapfile = os.path.join(cls.root, "emd_80558.map")
        subprocess.run([ENV_PY, "-c", "import mrcfile, numpy; m = mrcfile.new(%r, overwrite=True); m.set_data(numpy.random.rand(96, 96, 96).astype('float32')); "
                        "m.voxel_size = 1.0; m.close()" % mapfile], check=True)
        with open(mapfile, "rb") as f:
            map_gz = gzip.compress(f.read())
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            data = b"stub weights" * 100
            ti = tarfile.TarInfo("trained_models/MICA_best_model.pth")
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
        cls.weights = buf.getvalue()
        md5 = hashlib.md5(cls.weights).hexdigest()
        cls.srv, cls.url = T.serve(T.Files)
        T.Files.routes = {"/map/emd.map.gz": map_gz, "/cif/native.cif": cif_bytes, "/weights/trained_models.tar.gz": cls.weights,
                          "/zenodo/api": json.dumps({"files": [{"key": "trained_models.tar.gz", "checksum": "md5:" + md5, "size": len(cls.weights)}]}).encode()}
        cls.phenix_srv, cls.phenix_url = T.serve(T.PhenixServer)
        T.PhenixServer.body = FAKE_INSTALLER.replace("@STUB@", STUB_PHENIX.replace("@PYTHON@", sys.executable)).encode()
        cls.entries_csv = os.path.join(cls.root, "entries.csv")
        with open(os.path.join(ROOT, "data", "entries_2026.csv"), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows, fields = list(reader), reader.fieldnames
        for r in rows:
            if r["pdb_id"] in TEST_ENTRIES:
                r["map_url"], r["cif_url"] = cls.url + "/map/emd.map.gz", cls.url + "/cif/native.cif"
        with open(cls.entries_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        cls.repo = os.path.join(cls.root, "mica_stub")
        for rel, text in stub_mica().items():
            path = os.path.join(cls.repo, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        os.chmod(os.path.join(cls.repo, "modules/pulchra304/src/pulchra"), 0o755)
        T.sh("git init -q -b main . && git add -A && git -c user.name=t -c user.email=t@t commit -q -m stub", cwd=cls.repo)
        cls.commit = T.sh("git rev-parse HEAD", cwd=cls.repo)
        T.sh("git config uploadpack.allowAnySHA1InWant true", cwd=cls.repo)                  # the notebook fetches the pinned commit by its hash, which is no longer the tip
        T.sh("sed -i 's/self.batch_threshold = 200/self.batch_threshold = 300/' utils/predict.py && git -c user.name=t -c user.email=t@t commit -q -am other", cwd=cls.repo)
        cls.commit_without_threshold = T.sh("git rev-parse HEAD", cwd=cls.repo)           # a later MICA version in which the line is different

    @classmethod
    def tearDownClass(cls):
        for s in (cls.srv, cls.phenix_srv):
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def build_script(self, tmp, kind="cpu", **kw):
        nb_path = os.path.join(tmp, "nb.ipynb")
        url = "%s/download?version=2.2.1-6174&file=%s" % (self.phenix_url, T.INSTALLER_NAME)
        args = dict(kind=kind, entries=["26DK"], phenix_url=url, mica_repo="file://" + self.repo, mica_commit=self.commit, entries_path=self.entries_csv, env_python=ENV_PY,
                    usalign_bin=USALIGN, net_hosts=(), weights_url=self.url + "/weights/trained_models.tar.gz", zenodo_api=self.url + "/zenodo/api", require_gpu=False)
        args.update(kw)
        mm.build(nb_path, **args)
        nb = nbformat.read(nb_path, as_version=4)
        for c in nb.cells:
            if c["id"] == "step3":
                c.source = c.source.replace("MIN_INSTALLER_MB = 50", "MIN_INSTALLER_MB = 0").replace("MIN_FREE_GB = 20", "MIN_FREE_GB = 0")
        script = os.path.join(tmp, "nb.py")
        T.to_script(nb, script)
        return script

    def env(self, tmp, **extra):
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), PHENIX_USER=USER, PHENIX_PASSWORD=PASSWORD,
                   MICA_AF3_DIR=AF3_DIR, MICA_BEAT="1", MICA_REFINE_WAIT="60")
        for k in [k for k in env if k.startswith("STUB_")]:
            env.pop(k)
        env.update(extra)
        return env

    def run_notebook(self, tmp=None, build=None, **extra_env):
        tmp = tmp or tempfile.mkdtemp(prefix="run_", dir=self.root)
        script = self.build_script(tmp, **(build or {}))
        r = subprocess.run([sys.executable, script], cwd=tmp, env=self.env(tmp, **extra_env), capture_output=True, text=True, timeout=900)
        work = os.path.join(tmp, "work")
        zips = sorted(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".zip")) if os.path.isdir(work) else []
        return r, zips, tmp

    @staticmethod
    def zip_files(path):
        with zipfile.ZipFile(path) as z:
            return {n.split("/", 1)[1]: z.read(n) for n in z.namelist()}

    @staticmethod
    def table(files, name):
        return list(csv.DictReader(io.StringIO(files[name].decode())))

    def status(self, files):
        return {r["entry"]: r for r in self.table(files, "mica_status.csv")}

    def entry_dir(self, tmp, entry):
        return os.path.join(tmp, "big", "MICA", "input", entry)

    def state_dir(self, tmp):
        return os.path.join(tmp, "work_mica_state")

    # ------------------------------------------------------------------------------------------------------------------------------
    def test_a_complete_run_on_the_cpu(self):
        r, zips, tmp = self.run_notebook()
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        self.assertIn("RESULT: OK", out, out[-4000:])
        self.assertEqual(len(zips), 1, zips)
        self.assertNotIn("_INCOMPLETE", zips[0])
        files = self.zip_files(zips[0])
        for needed in ("run.log", "state.json", "run_info.json", "mica_status.csv", "pending.csv", "docking_domains.csv", "metrics_usalign.csv", "logs/26DK_1_process_af3.log",
                       "logs/26DK_2_dock.log", "logs/26DK_3_build.log", "gpu/26DK_3_build.csv", "docking_logs/26DK/bench_docking.csv", "scoring/26DK_mica_all_atom_usalign.txt",
                       "models/mica/26DK/80558_26dk_MICA_all_atom_model.pdb"):
            self.assertIn(needed, files, sorted(files))
        names = sorted(files)
        self.assertFalse(any("AF3_" in n or n.endswith((".map", ".mrc", ".ccp4")) or "mica_state" in n for n in names), names)
        coordinates = [n for n in names if n.endswith((".pdb", ".cif"))]
        self.assertTrue(coordinates and all(n.startswith("models/mica/") for n in coordinates), coordinates)       # MICA's models only: no AlphaFold or docked AlphaFold coordinates
        self.assertFalse(any(PASSWORD.encode() in data for data in files.values()), "the password is in the zip")
        row = self.status(files)["26DK"]
        self.assertEqual((row["status"], row["device"], row["windows"], row["domains"], row["domains_docked"], row["failed_step"]), ("ok", "cpu", "8", "2", "2", ""), row)
        self.assertEqual(row["box_A"], "96.0x96.0x96.0")
        dock = self.table(files, "docking_domains.csv")
        self.assertEqual([(d["entry"], d["attempt"], d["setting"], d["docked"], d["local_cc"]) for d in dock], [("26DK", "1", "res6_quick", "True", "0.61")] * 2)
        self.assertEqual(json.loads(dock[0]["settings"]), {"dock_with_mr": "False", "quick": "True", "resolution": "6.0"})
        self.assertEqual((row["dock_profile"], row["dock_mode"]), ("pass3", "full"))
        self.assertEqual(self.table(files, "pending.csv"), [])
        kinds = {m["model_kind"]: m for m in self.table(files, "metrics_usalign.csv")}
        for kind in ("mica_ca", "mica_all_atom", "mica_refined", "af3_docked"):
            self.assertIn(kind, kinds, kinds.keys())
            self.assertEqual(kinds[kind]["method"], "mica")
            self.assertEqual(kinds[kind]["status"], "ok", kinds[kind])
        self.assertEqual(kinds["mica_refined"]["chain_comparison"], "ok")
        # MICA's docking was called with the settings of the notebook
        calls = open(os.path.join(self.entry_dir(tmp, "26DK"), "AF3_docked_models", "calls.txt")).read().splitlines()
        self.assertEqual(len(calls), 2)
        args = dict(a.split("=", 1) for a in calls[0].split())
        self.assertEqual((args["quick"], args["dock_with_mr"], args["dock_chains_individually"], args["resolution"]), ("True", "False", "True", "6.0"))
        self.assertNotIn("rigid_body_refinement_resolution", args)
        self.assertNotIn("density_radius", args)
        # program 3 on the CPU with the cores of the machine, in the MICA folder
        p3 = read_json(os.path.join(self.entry_dir(tmp, "26DK"), "p3_call.json"))
        self.assertEqual((p3["device"], p3["omp"], p3["cwd"]), ("cpu", str(os.cpu_count()), os.path.join(tmp, "big", "MICA")))
        # the private state is kept outside the zip, with the markers and the files that program 3 needs
        state = os.path.join(self.state_dir(tmp), "26DK")
        for needed in (".done_1_process_af3", ".done_2_dock", ".done_3_build", "dock_profile.json", "26DK_af3_docked.pdb", "AF3_structures", "AF3_domains", "AF3_docked_models", "docking_logs", "26dk.fasta"):
            self.assertTrue(os.path.exists(os.path.join(state, needed)), needed)
        for absent in ("AF3_results", "grids", "temp_maps"):
            self.assertFalse(os.path.exists(os.path.join(state, absent)), absent)
        self.assertFalse([f for f in os.listdir(state) if f.endswith(".map")])
        info = json.loads(files["run_info.json"].decode())
        self.assertTrue(info["complete"])
        self.assertEqual((info["stages"], info["program_3_device"], info["entries_in_this_notebook"]), ([1, 2, 3], "cpu", ["26DK"]))
        self.assertEqual(info["mica_commit"], self.commit)
        self.assertIn("the private AlphaFold files of 1 entries are here", out)

    def test_a_big_map_is_deferred_and_a_second_session_finishes_it_from_the_saved_state(self):
        r, zips, tmp = self.run_notebook(build=dict(cpu_max_windows=2))
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        self.assertIn("DEFERRED", out)
        self.assertIn("RESULT: OK", out)
        files = self.zip_files(zips[0])
        row = self.status(files)["26DK"]
        self.assertEqual((row["status"], row["domains_docked"], row["windows"]), ("deferred", "2", "8"), row)
        self.assertIn("needs a GPU session", row["note"])
        self.assertEqual([p["entry"] for p in self.table(files, "pending.csv")], ["26DK"])
        self.assertNotIn("logs/26DK_3_build.log", files)
        state = self.state_dir(tmp)
        self.assertTrue(os.path.exists(os.path.join(state, "26DK", ".done_2_dock")))
        self.assertFalse(os.path.exists(os.path.join(state, "26DK", ".done_3_build")))
        # the second session: program 3 only, the state of the first as an input
        r2, zips2, tmp2 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_test", device="cpu", stages=(3,)), MICA_STATE_DIRS=state)
        out2 = r2.stdout + r2.stderr
        self.assertEqual(r2.returncode, 0, out2[-4000:])
        self.assertIn("the state of an earlier session was restored from", out2)
        self.assertIn("RESULT: OK", out2)
        files2 = self.zip_files(zips2[0])
        row2 = self.status(files2)["26DK"]
        self.assertEqual((row2["status"], row2["failed_step"]), ("ok", ""), row2)
        self.assertTrue(row2["restored_from"].endswith("26DK"))
        self.assertNotIn("logs/26DK_2_dock.log", files2)
        self.assertNotIn("logs/26DK_1_process_af3.log", files2)
        self.assertFalse(os.path.exists(os.path.join(self.entry_dir(tmp2, "26DK"), "AF3_docked_models", "calls.txt")) and
                         len(open(os.path.join(self.entry_dir(tmp2, "26DK"), "AF3_docked_models", "calls.txt")).read().splitlines()) > 2, "the docking was repeated")
        self.assertIn("mica_refined", {m["model_kind"] for m in self.table(files2, "metrics_usalign.csv")})
        # without the state: not run, said clearly
        r3, zips3, tmp3 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_test", device="cpu", stages=(3,)))
        files3 = self.zip_files(zips3[0])
        row3 = self.status(files3)["26DK"]
        self.assertEqual((row3["status"], row3["failed_step"]), ("not run", "state"))
        self.assertIn("no docked state", row3["note"])
        self.assertIn("_INCOMPLETE", zips3[0])

    def test_the_state_of_another_docking_set_up_is_not_used_by_the_gpu_session(self):
        r, zips, tmp = self.run_notebook(build=dict(cpu_max_windows=2, dock_profile="old_setup"))
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-3000:])
        state = self.state_dir(tmp)
        r2, zips2, tmp2 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_test", device="cpu", stages=(3,)), MICA_STATE_DIRS=state)
        out2 = r2.stdout + r2.stderr
        self.assertEqual(r2.returncode, 0, out2[-3000:])
        self.assertIn("was not docked with the profile pass3", out2)
        row = self.status(self.zip_files(zips2[0]))["26DK"]
        self.assertEqual((row["status"], row["failed_step"]), ("not run", "state"), row)
        # and with the profile that the GPU notebook asks for it is used
        r3, zips3, tmp3 = self.run_notebook(build=dict(cpu_max_windows=2))
        r4, zips4, tmp4 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_test", device="cpu", stages=(3,)), MICA_STATE_DIRS=self.state_dir(tmp3))
        row4 = self.status(self.zip_files(zips4[0]))["26DK"]
        self.assertEqual((row4["status"], row4["dock_profile"], row4["dock_mode"]), ("ok", "pass3", "full"), row4)

    def test_the_gpu_session_uses_the_gpu_of_its_lane(self):
        r, zips, tmp = self.run_notebook(build=dict(cpu_max_windows=2))
        state = self.state_dir(tmp)
        fake_bin = os.path.join(tmp, "fakebin")
        os.makedirs(fake_bin)
        with open(os.path.join(fake_bin, "nvidia-smi"), "w") as f:
            f.write('#!/bin/sh\ncase "$*" in *"query-gpu=index"*) echo 0; echo 1;; *"-i "*) echo "100, 5";; *) echo "Tesla T4, 15360 MiB, 580.1";; esac\n')
        os.chmod(os.path.join(fake_bin, "nvidia-smi"), 0o755)
        r2, zips2, tmp2 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_test", stages=(3,)), MICA_STATE_DIRS=state, PATH=fake_bin + os.pathsep + os.environ["PATH"])
        out2 = r2.stdout + r2.stderr
        self.assertEqual(r2.returncode, 0, out2[-4000:])
        p3 = read_json(os.path.join(self.entry_dir(tmp2, "26DK"), "p3_call.json"))
        self.assertEqual((p3["device"], p3["cuda_visible_devices"]), ("cuda", "0"), p3)
        self.assertEqual(self.status(self.zip_files(zips2[0]))["26DK"]["status"], "ok")
        self.assertIn("2 GPU(s)", out2)

    def test_the_sensitivity_session_removes_the_docked_input_and_leaves_the_saved_state_alone(self):
        r, zips, tmp = self.run_notebook(build=dict(cpu_max_windows=2))
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-3000:])
        state = self.state_dir(tmp)
        saved = sorted(os.path.relpath(os.path.join(d, f), state) for d, _s, fs in os.walk(state) for f in fs)
        self.assertIn(os.path.join("26DK", "26DK_af3_docked.pdb"), saved)
        self.assertTrue(any(f.startswith(os.path.join("26DK", "AF3_docked_models")) and f.endswith("_docked.pdb") for f in saved))
        # the main GPU session sees the docked input; the sensitivity session does not
        r2, zips2, tmp2 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_test", device="cpu", stages=(3,)), MICA_STATE_DIRS=state)
        self.assertTrue(read_json(os.path.join(self.entry_dir(tmp2, "26DK"), "p3_call.json"))["docked_present"])
        row2 = self.status(self.zip_files(zips2[0]))["26DK"]
        self.assertEqual((row2["status"], row2["dock_mode"], row2["domains_docked"]), ("ok", "full", "2"), row2)
        r3, zips3, tmp3 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_nodock_test", device="cpu", stages=(3,), nodock=True), MICA_STATE_DIRS=state)
        out3 = r3.stdout + r3.stderr
        self.assertEqual(r3.returncode, 0, out3[-4000:])
        self.assertIn("sensitivity session: the docked AlphaFold models", out3)
        p3 = read_json(os.path.join(self.entry_dir(tmp3, "26DK"), "p3_call.json"))
        self.assertFalse(p3["docked_present"])
        files3 = self.zip_files(zips3[0])
        row3 = self.status(files3)["26DK"]
        self.assertEqual((row3["status"], row3["failed_step"], row3["dock_mode"], row3["domains"], row3["domains_docked"]), ("ok", "", "none", "2", "0"), row3)
        self.assertIn("mode full", row3["note"])
        self.assertIn("without docked AlphaFold input", row3["note"])
        self.assertIn("mica_refined", {m["model_kind"] for m in self.table(files3, "metrics_usalign.csv")})
        info = json.loads(files3["run_info.json"].decode())
        self.assertTrue(info["force_no_dock"])
        self.assertEqual(info["entries_in_this_notebook"], ["26DK"])
        after = sorted(os.path.relpath(os.path.join(d, f), state) for d, _s, fs in os.walk(state) for f in fs)
        self.assertEqual(after, saved, "the attached state was changed")

    def test_pass_4_raises_the_batch_threshold_of_mica_before_program_3_runs(self):
        r, zips, tmp = self.run_notebook(build=dict(cpu_max_windows=2))
        state = self.state_dir(tmp)
        r2, zips2, tmp2 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_test", device="cpu", stages=(3,)), MICA_STATE_DIRS=state)
        self.assertEqual(read_json(os.path.join(self.entry_dir(tmp2, "26DK"), "p3_call.json"))["batch_threshold"], 200)            # the notebooks of 2026-10-27
        r3, zips3, tmp3 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_test", device="cpu", stages=(3,), batch_threshold=1000000), MICA_STATE_DIRS=state)
        out3 = r3.stdout + r3.stderr
        self.assertEqual(r3.returncode, 0, out3[-4000:])
        self.assertIn("batch_threshold raised from 200 to 1000000", out3)
        self.assertEqual(read_json(os.path.join(self.entry_dir(tmp3, "26DK"), "p3_call.json"))["batch_threshold"], 1000000)
        files3 = self.zip_files(zips3[0])
        self.assertEqual(self.status(files3)["26DK"]["status"], "ok")
        self.assertEqual(json.loads(files3["run_info.json"].decode())["batch_threshold"], 1000000)
        with open(os.path.join(tmp3, "big", "MICA", "utils", "predict.py"), encoding="utf-8") as f:
            text = f.read()
        self.assertEqual((text.count("self.batch_threshold = 1000000"), text.count("self.batch_threshold = 200")), (1, 0))      # one line changed, nothing else
        # a MICA version without that line is not run at all: the notebook says so instead of running with the old threshold
        r4, zips4, tmp4 = self.run_notebook(build=dict(kind="gpu", name="mica_gpu_test", device="cpu", stages=(3,), batch_threshold=1000000, mica_commit=self.commit_without_threshold), MICA_STATE_DIRS=state)
        out4 = r4.stdout + r4.stderr
        self.assertIn("does not contain 'self.batch_threshold = 200' exactly once", out4)
        self.assertFalse(os.path.exists(os.path.join(self.entry_dir(tmp4, "26DK"), "p3_call.json")), "program 3 ran with the old threshold")

    def test_when_no_domain_can_be_docked_program_3_runs_without_docked_alphafold_input(self):
        r, zips, tmp = self.run_notebook(STUB_FAIL_ALL_DOCK="1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        self.assertIn("RESULT: OK", out)
        files = self.zip_files(zips[0])
        row = self.status(files)["26DK"]
        self.assertEqual((row["status"], row["failed_step"], row["dock_mode"], row["domains"], row["domains_docked"]), ("ok", "", "none", "2", "0"), row)
        self.assertIn("without docked AlphaFold input", row["note"])
        self.assertIn("logs/26DK_3_build.log", files)                    # program 3 ran
        p3 = read_json(os.path.join(self.entry_dir(tmp, "26DK"), "p3_call.json"))
        self.assertFalse(p3["docked_present"])
        dock = self.table(files, "docking_domains.csv")
        self.assertEqual(len(dock), 4)                                   # two domains, two attempts each
        self.assertEqual({d["docked"] for d in dock}, {"False"})
        self.assertEqual({d["setting"] for d in dock}, {"res6_quick", "res10_quick"})
        self.assertTrue(os.path.exists(os.path.join(self.entry_dir(tmp, "26DK"), ".done_2_dock")))
        self.assertEqual(self.table(files, "pending.csv"), [])

    def test_a_docking_attempt_that_runs_out_of_time_is_followed_by_the_next_and_nothing_is_left_running(self):
        attempts = [{"name": "slow", "set": {"quick": "False"}, "timeout_s": 2}, {"name": "fast", "set": {"quick": "True"}, "timeout_s": 60}]
        r, zips, tmp = self.run_notebook(build=dict(dock_attempts=attempts), STUB_SLOW_MICA="0.5")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        files = self.zip_files(zips[0])
        dock = self.table(files, "docking_domains.csv")
        self.assertEqual([(d["setting"], d["timed_out"], d["docked"]) for d in dock], [("slow", "True", "False"), ("fast", "False", "True")] * 2)
        self.assertEqual(self.status(files)["26DK"]["status"], "ok")
        left = subprocess.run(["pgrep", "-f", "phenix.dock_in_map.*quick=False"], capture_output=True, text=True).stdout.split()
        self.assertEqual(left, [], "a docking process of the stand-in is still running")

    def test_two_entries_never_run_program_3_at_the_same_time(self):
        r, zips, tmp = self.run_notebook(build=dict(entries=["26DK", "9T5K"], lanes=2), STUB_P3_SECONDS="3")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        self.assertNotIn("COLLISION", out)
        files = self.zip_files(zips[0])
        rows = self.status(files)
        self.assertEqual((rows["26DK"]["status"], rows["9T5K"]["status"]), ("ok", "ok"), rows)
        self.assertEqual(sorted({r["lane"] for r in rows.values()}), ["0", "1"])

    def test_mica_refinement_in_the_background_is_waited_for(self):
        r, zips, tmp = self.run_notebook(STUB_REFINE_SECONDS="6")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        files = self.zip_files(zips[0])
        self.assertIn("mica_refined", {m["model_kind"] for m in self.table(files, "metrics_usalign.csv")})
        self.assertIn("waited", out)
        self.assertIn("refined model found", out)
        self.assertIn("models/mica/26DK/80558_26dk_MICA_all_atom_model_real_space_refined_000.pdb", files)

    def test_a_refinement_that_writes_nothing_is_run_again_and_the_entry_still_counts(self):
        r, zips, tmp = self.run_notebook(STUB_NO_REFINED="1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        self.assertIn("running phenix.real_space_refine again, with its output kept", out)
        files = self.zip_files(zips[0])
        kinds = {m["model_kind"] for m in self.table(files, "metrics_usalign.csv")}
        self.assertIn("mica_all_atom", kinds)
        self.assertNotIn("mica_refined", kinds)
        self.assertEqual(self.status(files)["26DK"]["status"], "ok")

    def test_two_failures_of_program_3_in_a_row_switch_it_off_and_the_docking_goes_on(self):
        r, zips, tmp = self.run_notebook(build=dict(entries=TEST_ENTRIES, lanes=1), STUB_FAIL_STEP="3_build")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        self.assertIn("GUARD: program 3 failed on two entries in a row", out)
        files = self.zip_files(zips[0])
        rows = self.status(files)
        self.assertEqual([(rows[e]["status"], rows[e]["failed_step"]) for e in TEST_ENTRIES], [("failed", "3_build"), ("failed", "3_build"), ("deferred", "")])
        self.assertTrue(os.path.exists(os.path.join(self.state_dir(tmp), "22FX", ".done_2_dock")), "the docked state of the third entry must be kept")
        self.assertEqual([p["entry"] for p in self.table(files, "pending.csv")], ["22FX"])

    def test_two_quick_failures_in_program_1_stop_the_session(self):
        r, zips, tmp = self.run_notebook(build=dict(entries=TEST_ENTRIES, lanes=1), STUB_FAIL_STEP="1_process_af3")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        self.assertIn("GUARD: two entries in a row failed in 1_process_af3", out)
        files = self.zip_files(zips[0])
        rows = self.status(files)
        self.assertEqual([rows[e]["status"] for e in TEST_ENTRIES], ["failed", "failed", "not run"])
        self.assertEqual(rows["22FX"]["failed_step"], "guard")
        self.assertIn("_INCOMPLETE", zips[0])

    def test_the_end_of_the_time_budget_starts_nothing_and_says_so(self):
        r, zips, tmp = self.run_notebook(build=dict(entries=TEST_ENTRIES, hours=0.2))
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        files = self.zip_files(zips[0])
        rows = self.status(files)
        self.assertEqual({rows[e]["status"] for e in TEST_ENTRIES}, {"not run"})
        self.assertEqual({rows[e]["failed_step"] for e in TEST_ENTRIES}, {"time"})
        self.assertEqual(len(self.table(files, "pending.csv")), 3)
        self.assertIn("_INCOMPLETE", zips[0])
        self.assertIn("TO DO IN A LATER SESSION", out)

    def test_a_missing_private_dataset_stops_before_any_time_is_spent(self):
        empty = tempfile.mkdtemp(prefix="noaf3_", dir=self.root)
        r, zips, tmp = self.run_notebook(MICA_AF3_DIR=empty)
        out = r.stdout + r.stderr
        self.assertIn("the private AlphaFold dataset is missing", out, out[-3000:])
        self.assertIn("RESULT: STOPPED", out)
        self.assertNotIn("MICA program", out)
        self.assertEqual(len(zips), 1)
        self.assertIn("_INCOMPLETE", zips[0])


if __name__ == "__main__":
    unittest.main()
