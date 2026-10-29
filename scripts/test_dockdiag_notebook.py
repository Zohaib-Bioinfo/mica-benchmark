#!/usr/bin/env python3
"""Tests for make_dockdiag_notebook.py and the notebook it builds.

Fast tests: the notebook builds, every cell compiles, the embedded files equal the repository files, the plan is consistent, the committed notebook is current.

End-to-end tests run the whole notebook as a script against stand-ins (the same ideas as test_pilot_notebook.py): a Phenix download server with a fake
installer whose programs `phenix.dock_in_map`, `phenix.map_model_cc` and `phenix.chain_comparison` behave as the tests tell them (fast, slow, failing, refusing
`verbose=True`), a local git repository that plays MICA, and a file server for the map and the deposited model. They need

    MICA_TEST_ENV_PYTHON   a Python with Biopython, mrcfile, numpy (the MICA environment)
    MICA_TEST_AF3_DIR      the folder with the 26 AlphaFold Server files (private; entry 26DK is used)
    USALIGN                the compiled US-align program

    USALIGN=... MICA_TEST_ENV_PYTHON=... MICA_TEST_AF3_DIR=... python -W error -m unittest test_dockdiag_notebook
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
import tempfile
import unittest
import zipfile

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_dockdiag_notebook as dd  # noqa: E402
import make_pilot_notebook as mk  # noqa: E402
import test_pilot_notebook as T  # noqa: E402  (the stand-in servers and helpers of the first pilot)

COMMITTED = os.path.join(HERE, "dockdiag.ipynb")
ENV_PY, AF3_DIR, USALIGN = T.ENV_PY, T.AF3_DIR, T.USALIGN
USER, PASSWORD = T.USER, T.PASSWORD

STUB_PHENIX = r'''#!@PYTHON@
import os, shutil, sys, time
prog, args = os.path.basename(sys.argv[0]), sys.argv[1:]
kv = dict(a.split("=", 1) for a in args if "=" in a)


def say(text):
    print(text)
    sys.stdout.flush()


if prog == "phenix.version":
    say("Phenix version: 2.2.1-6174 (stub)")
elif prog == "phenix.dock_in_map":
    if "--show_defaults" in args or "--help" in args:
        say("quick = True\nnproc = 1\ntarget_boxes = None\nrun_in_boxes = True\nresolution = None\ndock_with_mr = True")
        sys.exit(0)
    assert os.path.exists(kv["map_file"]) and "sequence" in kv and os.path.exists(kv["search_model"])
    if "verbose" in kv and os.environ.get("STUB_NO_VERBOSE"):
        say("Sorry: unknown parameter: verbose")
        sys.exit(1)
    with open(os.path.join(os.path.dirname(kv["pdb_out"]), "calls.txt"), "a") as f:
        f.write(" ".join(args) + "\n")
    if os.environ.get("STUB_FAIL_ALL_DOCK") or (kv.get("nproc") == "4" and os.environ.get("STUB_FAIL_NPROC4")):
        say("Sorry: stub docking failure")
        sys.exit(2)
    say("Running in %s boxes" % kv.get("nproc"))
    if kv.get("quick") == "False" and kv.get("resolution") != "6.0" and os.environ.get("STUB_SLOW_MICA"):
        for i in range(1000):
            say("box %d of 2: searching rotation %d" % (1, i))
            time.sleep(float(os.environ["STUB_SLOW_MICA"]))
    time.sleep(float(os.environ.get("STUB_DOCK_SECONDS", "0.2")))
    shift = 1.0 if kv.get("resolution") == "6.0" else 0.0
    with open(kv["search_model"]) as src, open(kv["pdb_out"], "w") as out:
        for line in src:
            if shift and line.startswith(("ATOM  ", "HETATM")):
                line = line[:30] + "%8.3f" % (float(line[30:38]) + shift) + line[38:]
            out.write(line)
    say("Final score 0.9")
elif prog == "phenix.map_model_cc":
    if "--force" not in args:
        say("phenix.map_model_cc and phenix.model_map_cc are replaced by phenix.map_correlations. Add the keyword '--force' to use anyway")
        sys.exit(1)
    say("  CC_mask  : %s" % ("0.8000" if args[0].endswith("26DK.cif") else "0.6100"))
elif prog == "phenix.real_space_refine":
    pass
elif prog == "phenix.chain_comparison":
    if "--help" in args:
        say("usage: phenix.chain_comparison target.pdb query.pdb [options]")
    else:
        say("target %s query %s\nCA match: 100 of 191 (52.4 %%)" % (args[0], args[1]))
'''

STUB_PROGRAM_1 = r'''
import argparse, os, shutil, sys
from Bio.PDB import MMCIFParser, PDBIO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", required=True)
    p.add_argument("-a", required=True)
    p.add_argument("-d", default="cuda")
    a = p.parse_args()
    assert a.d == "cpu", "the diagnostic runs program 1 on the CPU"
    if os.environ.get("STUB_FAIL_STEP") == "1_process_af3":
        print("boom in step 1"); sys.exit(1)
    entry_dir = os.path.dirname(a.a)
    for line in open(a.f):
        if line.startswith(">"):
            h = line[1:].split("|")[0]; pdb, k = h.split("_")
            s = MMCIFParser(QUIET=True).get_structure("x", os.path.join(a.a, "%s_%s" % (pdb.lower(), k), "fold_%s_%s_model_0.cif" % (pdb.lower(), k)))
            out = os.path.join(entry_dir, "AF3_structures", h); os.makedirs(out, exist_ok=True)
            io = PDBIO(); io.set_structure(s); io.save(os.path.join(out, "ranked_0.pdb"))
            if not os.environ.get("STUB_NO_DOMAINS"):
                dom = os.path.join(entry_dir, "AF3_domains"); os.makedirs(dom, exist_ok=True)
                for i in (1, 2):
                    shutil.copy(os.path.join(out, "ranked_0.pdb"), os.path.join(dom, "fold_%s_%s_model_0_chain_A_domain_0%d.pdb" % (pdb.lower(), k, i)))
                open(os.path.join(dom, "fold_%s_%s_model_0_chain_A_domain.domains" % (pdb.lower(), k)), "w").write("x\t1\t113\n")


if __name__ == "__main__":
    main()
'''


def stub_mica():
    files = dict(T.STUB_MICA)
    files["utils/process_AF3_results.py"] = STUB_PROGRAM_1
    return files


FAKE_INSTALLER = T.FAKE_INSTALLER.replace("for p in phenix.dock_in_map phenix.real_space_refine phenix.chain_comparison;",
                                          "for p in phenix.dock_in_map phenix.real_space_refine phenix.chain_comparison phenix.map_model_cc;")
assert "phenix.map_model_cc" in FAKE_INSTALLER


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ddbuild_")
        cls.path = os.path.join(cls.tmp, "nb.ipynb")
        cls.expected = dd.build(cls.path)
        cls.nb = nbformat.read(cls.path, as_version=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cells_compile_and_have_no_placeholders(self):
        import re
        for c in self.nb.cells:
            if c.cell_type == "code" and not c.source.startswith("%%writefile"):
                compile(c.source, c["id"], "exec")
            self.assertEqual(re.findall(r"@[A-Z_0-9]+@", c.source), [], c["id"])

    def test_cell_order(self):
        ids = [c["id"] for c in self.nb.cells if c.cell_type == "code" and not c["id"].startswith("file-")]
        self.assertEqual(ids, ["step1", "step2-dirs", "step2-check", "robust", "step3", "step3b", "step4", "step5", "step6", "step7", "step8", "final"])

    def test_embedded_files_equal_the_repository_files(self):
        got = {}
        for c in self.nb.cells:
            if c.source.startswith("%%writefile"):
                first, text = c.source.split("\n", 1)
                got[first.split(None, 1)[1].strip()] = text
        self.assertEqual(sorted(got), sorted(t for _s, t in dd.FILES))
        for src, target in dd.FILES:
            with open(os.path.join(ROOT, src), encoding="utf-8") as f:
                text = f.read()
            self.assertEqual(got[target], text if text.endswith("\n") else text + "\n", target)
            self.assertEqual(self.expected[target], hashlib.sha256(got[target].encode()).hexdigest())

    def test_the_committed_notebook_is_current(self):
        if not os.path.exists(COMMITTED):
            self.skipTest("dockdiag.ipynb is not built yet")
        fresh = os.path.join(self.tmp, "fresh.ipynb")
        dd.build(fresh)
        a, b = nbformat.read(fresh, as_version=4), nbformat.read(COMMITTED, as_version=4)
        self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_dockdiag_notebook.py again")

    def test_the_weights_are_not_downloaded_and_the_gpu_is_not_required(self):
        step5 = next(c for c in self.nb.cells if c["id"] == "step5").source
        self.assertIn("if fetch_mica() and import_mica():", step5)
        self.assertNotIn("fetch_weights() and", step5)
        self.assertIn("REQUIRE_GPU = False", next(c for c in self.nb.cells if c["id"] == "step1").source)

    def test_the_plan_is_consistent(self):
        names = [s["name"] for s in dd.DOCK_PLAN]
        self.assertEqual(len(names), len(set(names)))
        for s in dd.DOCK_PLAN:
            self.assertIn("cap_min", s)
            for dep in ([s["after_failure_of"]] if "after_failure_of" in s else []) + s.get("needs_a_success_of", []):
                self.assertIn(dep, names)
                self.assertLess(next(x["group"] for x in dd.DOCK_PLAN if x["name"] == dep), s["group"], "a test must come after the tests it depends on")
        mica = next(s for s in dd.DOCK_PLAN if s["name"] == "B_mica_exact")
        self.assertEqual(mica["set"], {}, "the test that reproduces MICA must change nothing")

    def test_mica_exact_arguments_and_order(self):
        step7 = next(c for c in self.nb.cells if c["id"] == "step7").source
        self.assertIn('kv = {"search_model": DOCK["search_model"], "map_file": processed, "nproc": 2, "quick": "False", "resolution": DOCK["resolution"],', step7)
        self.assertIn('"dock_chains_individually": "True", "sequence": DOCK["fasta"]}', step7)

    def test_the_plan_is_in_the_notebook(self):
        step7 = next(c for c in self.nb.cells if c["id"] == "step7").source
        self.assertIn(repr(dd.DOCK_PLAN), step7)


@unittest.skipUnless(ENV_PY and AF3_DIR and USALIGN and any("fold_26dk_e1_model_0.cif" in f for _r, _d, f in os.walk(AF3_DIR or "")),
                     "needs MICA_TEST_ENV_PYTHON, MICA_TEST_AF3_DIR (with the 26DK file) and USALIGN")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="ddrun_")
        cif = [os.path.join(r, f) for r, _d, fs in os.walk(AF3_DIR) for f in fs if f == "fold_26dk_e1_model_0.cif"][0]
        with open(cif, "rb") as f:
            cif_bytes = f.read()
        mapfile = os.path.join(cls.root, "emd_80558.map")
        subprocess.run([ENV_PY, "-c", "import mrcfile, numpy; m = mrcfile.new(%r, overwrite=True); m.set_data(numpy.random.rand(48, 48, 48).astype('float32')); "
                        "m.voxel_size = 1.0; m.close()" % mapfile], check=True)
        with open(mapfile, "rb") as f:
            map_gz = gzip.compress(f.read())
        cls.srv, cls.url = T.serve(T.Files)
        T.Files.routes = {"/map/emd_80558.map.gz": map_gz, "/cif/26DK.cif": cif_bytes}
        cls.phenix_srv, cls.phenix_url = T.serve(T.PhenixServer)
        T.PhenixServer.body = FAKE_INSTALLER.replace("@STUB@", STUB_PHENIX.replace("@PYTHON@", sys.executable)).encode()
        cls.entries = os.path.join(cls.root, "entries.csv")
        with open(os.path.join(ROOT, "data", "entries_2026.csv"), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows, fields = list(reader), reader.fieldnames
        for r in rows:
            if r["pdb_id"] == "26DK":
                r["map_url"], r["cif_url"] = cls.url + "/map/emd_80558.map.gz", cls.url + "/cif/26DK.cif"
        with open(cls.entries, "w", newline="", encoding="utf-8") as f:
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

    @classmethod
    def tearDownClass(cls):
        for s in (cls.srv, cls.phenix_srv):
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def build_script(self, tmp, plan=None):
        nb_path = os.path.join(tmp, "nb.ipynb")
        url = "%s/download?version=2.2.1-6174&file=%s" % (self.phenix_url, T.INSTALLER_NAME)
        dd.build(nb_path, phenix_url=url, mica_repo="file://" + self.repo, mica_commit=self.commit, entries_path=self.entries, env_python=ENV_PY,
                 usalign_bin=USALIGN, net_hosts=(), plan=plan)
        nb = nbformat.read(nb_path, as_version=4)
        for c in nb.cells:
            if c["id"] == "step3":
                c.source = c.source.replace("MIN_INSTALLER_MB = 50", "MIN_INSTALLER_MB = 0").replace("MIN_FREE_GB = 20", "MIN_FREE_GB = 0")
        script = os.path.join(tmp, "nb.py")
        T.to_script(nb, script)
        return script, nb

    def env(self, tmp, **extra):
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), PHENIX_USER=USER, PHENIX_PASSWORD=PASSWORD,
                   MICA_AF3_DIR=AF3_DIR, MICA_BEAT="1")
        for k in [k for k in env if k.startswith("STUB_")]:
            env.pop(k)
        env.update(extra)
        return env

    def run_notebook(self, plan=None, tmp=None, **extra_env):
        tmp = tmp or tempfile.mkdtemp(prefix="run_", dir=self.root)
        script, nb = self.build_script(tmp, plan)
        r = subprocess.run([sys.executable, script], cwd=tmp, env=self.env(tmp, **extra_env), capture_output=True, text=True, timeout=900)
        work = os.path.join(tmp, "work")
        zips = sorted(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".zip")) if os.path.isdir(work) else []
        return r, zips, tmp

    @staticmethod
    def zip_files(path):
        with zipfile.ZipFile(path) as z:
            return {n.split("/", 1)[1]: z.read(n) for n in z.namelist()}

    @staticmethod
    def rows(files, name="dock_results.csv"):
        return {r["name"]: r for r in csv.DictReader(io.StringIO(files[name].decode()))}

    FAST_PLAN = [
        {"name": "A_quick_4cpu", "group": 1, "set": {"quick": "True", "nproc": 4}, "cap_min": 0.5, "note": "a"},
        {"name": "A2_quick_1cpu", "group": 2, "set": {"quick": "True", "nproc": 1}, "cap_min": 0.5, "after_failure_of": "A_quick_4cpu", "note": "a2"},
        {"name": "B_mica_exact", "group": 3, "set": {}, "cap_min": 0.5, "needs_a_success_of": ["A_quick_4cpu", "A2_quick_1cpu"], "note": "b"},
        {"name": "C_quick_false_res6", "group": 3, "set": {"resolution": 6.0}, "cap_min": 0.5, "needs_a_success_of": ["A_quick_4cpu", "A2_quick_1cpu"], "note": "c"},
    ]

    def test_a_complete_run(self):
        r, zips, tmp = self.run_notebook(self.FAST_PLAN)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out, out[-3000:])
        self.assertEqual(len(zips), 1, zips)
        self.assertNotIn("_INCOMPLETE", zips[0])
        self.assertFalse(os.path.exists(os.path.join(tmp, "work", "dockdiag_partial.zip")), "the partial zip must be replaced by the final one")
        files = self.zip_files(zips[0])
        for needed in ("run.log", "state.json", "run_info.json", "dock_results.csv", "phenix_help.txt", "phenix_dock_in_map_help.txt", "logs/26DK_1_process_af3.log",
                       "dock/A_quick_4cpu/phenix.log", "dock/B_mica_exact/phenix.log", "dock/C_quick_false_res6/phenix.log", "dock/A_quick_4cpu/A_quick_4cpu_docked.pdb",
                       "scoring/native_map_model_cc.txt", "scoring/B_mica_exact_chain_comparison.txt", "scoring/B_mica_exact_usalign.txt", "scoring/B_mica_exact_map_model_cc.txt",
                       "dock_inputs/fold_26dk_1_model_0_chain_A_domain_01.pdb"):
            self.assertIn(needed, files)
        self.assertFalse(any("AF3_results" in n or n.endswith((".map", ".mrc", ".ccp4")) for n in files), sorted(files))
        self.assertFalse(any(PASSWORD.encode() in data for data in files.values()), "the password is in the zip")
        rows = self.rows(files)
        self.assertEqual(list(rows), ["A_quick_4cpu", "A2_quick_1cpu", "B_mica_exact", "C_quick_false_res6"])
        self.assertEqual(rows["A2_quick_1cpu"]["skipped"], "True")
        for name in ("A_quick_4cpu", "B_mica_exact", "C_quick_false_res6"):
            self.assertEqual(rows[name]["docked"], "True", name)
            self.assertEqual(rows[name]["cc_mask"], "0.61", name)
            self.assertGreater(float(rows[name]["tm_score"]), 0.3, name)
            self.assertEqual(rows[name]["rc"], "0", name)
        self.assertEqual(rows["A_quick_4cpu"]["reference"], "A_quick_4cpu")
        self.assertTrue(rows["B_mica_exact"]["ca_distance_to_reference"].startswith("0.0 "), rows["B_mica_exact"]["ca_distance_to_reference"])
        self.assertTrue(rows["C_quick_false_res6"]["ca_distance_to_reference"].startswith("1.0 "), rows["C_quick_false_res6"]["ca_distance_to_reference"])
        # MICA's exact call: the stand-in recorded the arguments it was given
        calls = files["dock/B_mica_exact/calls.txt"].decode().strip().splitlines()
        self.assertEqual(len(calls), 1)
        args = dict(a.split("=", 1) for a in calls[0].split())
        self.assertEqual((args["nproc"], args["quick"], args["dock_chains_individually"], args["resolution"]), ("2", "False", "True", "3.5"))
        self.assertEqual(args["verbose"], "True")
        self.assertTrue(args["search_model"].endswith("_chain_A_domain_01.pdb"), args["search_model"])
        self.assertTrue(args["map_file"].endswith("initial_processed.mrc"))
        a_args = dict(a.split("=", 1) for a in files["dock/A_quick_4cpu/calls.txt"].decode().split())
        self.assertEqual((a_args["nproc"], a_args["quick"]), ("4", "True"))
        # numbers of the entry and the native control
        info = json.loads(files["run_info.json"].decode())
        self.assertTrue(info["complete"])
        self.assertEqual(info["mica_commit"], self.commit)
        self.assertEqual(info["state"]["native_cc_mask"]["measured_here"], 0.8)
        with open(self.entries, newline="", encoding="utf-8") as f:
            table = {r["pdb_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(info["state"]["native_cc_mask"]["in_entries_table"], table["26DK"]["cc_mask"])
        self.assertEqual(info["state"]["dock_inputs"]["domain_calpha_counts"]["fold_26dk_1_model_0_chain_A_domain_01.pdb"], 197)
        # progress lines of the heartbeat were printed while the tests ran, and the processed map was made like MICA makes it
        self.assertIn("processed map:", out)
        for line in ("phenix.dock_in_map search_model=", "docking test B_mica_exact finished after", "docking test C_quick_false_res6 finished after"):
            self.assertIn(line, out)

    def test_a_test_that_hits_its_time_limit_is_reported_and_the_run_still_completes(self):
        plan = [dict(s, cap_min=0.05) if s["name"] == "B_mica_exact" else s for s in self.FAST_PLAN]
        r, zips, tmp = self.run_notebook(plan, STUB_SLOW_MICA="0.5")
        out = r.stdout + r.stderr
        self.assertIn("STOPPED at its time limit of 0.05 min", out, out[-3000:])
        self.assertIn("RESULT: OK", out)
        self.assertIn("[dock B_mica_exact] running for", out)          # progress lines while the test was silent for seconds
        self.assertIn("box 1 of 2: searching rotation", out)           # and the last line of the Phenix log is shown in them
        files = self.zip_files(zips[0])
        rows = self.rows(files)
        self.assertEqual((rows["B_mica_exact"]["docked"], rows["B_mica_exact"]["timed_out"]), ("False", "True"))
        self.assertEqual(rows["C_quick_false_res6"]["docked"], "True")
        self.assertIn(b"searching rotation", files["dock/B_mica_exact/phenix.log"])
        self.assertEqual(rows["B_mica_exact"]["tm_score"], "")

    def test_the_second_test_runs_only_when_the_first_fails_and_the_longer_ones_follow_it(self):
        r, zips, tmp = self.run_notebook(self.FAST_PLAN, STUB_FAIL_NPROC4="1")
        out = r.stdout + r.stderr
        self.assertIn("RESULT: OK", out, out[-3000:])
        rows = self.rows(self.zip_files(zips[0]))
        self.assertEqual((rows["A_quick_4cpu"]["docked"], rows["A_quick_4cpu"]["rc"]), ("False", "2"))
        self.assertEqual((rows["A2_quick_1cpu"]["docked"], rows["A2_quick_1cpu"]["skipped"]), ("True", "False"))
        self.assertEqual(rows["B_mica_exact"]["docked"], "True")
        self.assertEqual(rows["A_quick_4cpu"]["reference"], "")
        self.assertEqual(rows["A2_quick_1cpu"]["reference"], "A2_quick_1cpu")

    def test_when_every_docking_fails_the_longer_tests_are_skipped_and_the_zip_says_so(self):
        r, zips, tmp = self.run_notebook(self.FAST_PLAN, STUB_FAIL_ALL_DOCK="1")
        out = r.stdout + r.stderr
        self.assertIn("is skipped: none of A_quick_4cpu, A2_quick_1cpu gave a docked model", out)
        self.assertIn("RESULT: OK", out)
        files = self.zip_files(zips[0])
        rows = self.rows(files)
        self.assertEqual([rows[n]["docked"] for n in rows], ["False"] * 4)
        self.assertEqual((rows["B_mica_exact"]["skipped"], rows["C_quick_false_res6"]["skipped"]), ("True", "True"))
        self.assertIn(b"stub docking failure", files["dock/A_quick_4cpu/phenix.log"])

    def test_a_phenix_that_refuses_verbose_is_asked_again_without_it(self):
        r, zips, tmp = self.run_notebook(self.FAST_PLAN, STUB_NO_VERBOSE="1")
        out = r.stdout + r.stderr
        self.assertIn("Phenix did not accept verbose=True", out)
        files = self.zip_files(zips[0])
        rows = self.rows(files)
        self.assertEqual(rows["B_mica_exact"]["docked"], "True")
        calls = files["dock/B_mica_exact/calls.txt"].decode().strip().splitlines()
        self.assertEqual(len(calls), 1, "only the second call reached the program; the first one was refused before it wrote anything useful")
        self.assertNotIn("verbose", calls[0])

    def test_a_failing_program_1_stops_the_run_and_packs_the_log(self):
        r, zips, tmp = self.run_notebook(self.FAST_PLAN, STUB_FAIL_STEP="1_process_af3")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-2000:])
        self.assertIn("MICA's program 1 gave no domain file", out)
        self.assertIn("boom in step 1", out)
        self.assertIn("RESULT: STOPPED", out)
        self.assertEqual(len(zips), 1)
        self.assertIn("_INCOMPLETE", zips[0])
        self.assertNotIn("docking test", out)

    def test_program_1_without_domain_files_stops_with_a_clear_message(self):
        r, zips, tmp = self.run_notebook(self.FAST_PLAN, STUB_NO_DOMAINS="1")
        out = r.stdout + r.stderr
        self.assertIn("no domain file was written", out)
        self.assertIn("RESULT: STOPPED", out)

    def test_a_wrong_password_stops_the_run_but_still_packs_the_log(self):
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        script, _ = self.build_script(tmp, self.FAST_PLAN)
        r = subprocess.run([sys.executable, script], cwd=tmp, env=self.env(tmp, PHENIX_PASSWORD="wrong-password-1"), capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        self.assertIn("did not accept the user name and password", out)
        self.assertIn("RESULT: STOPPED", out)
        zips = [f for f in os.listdir(os.path.join(tmp, "work")) if f.endswith(".zip")]
        self.assertEqual(len(zips), 1)
        files = self.zip_files(os.path.join(tmp, "work", zips[0]))
        self.assertFalse(any(b"wrong-password-1" in data for data in files.values()))

    def test_the_last_cell_alone_recovers_a_run_after_a_restart_of_the_kernel(self):
        # a run that is stopped by the time limit of the test, then "a fresh kernel": only the last cell runs
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        script, nb = self.build_script(tmp, self.FAST_PLAN)
        cut = os.path.join(tmp, "partial.py")
        T.to_script(nb, cut)
        with open(cut) as f:
            text = f.read()
        stop = text.index("# ---- cell step8")
        with open(cut, "w") as f:                       # everything before the scoring: the notebook "died" here
            f.write(text[:stop])
        r = subprocess.run([sys.executable, cut], cwd=tmp, env=self.env(tmp), capture_output=True, text=True, timeout=900)
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-2000:])
        work = os.path.join(tmp, "work")
        self.assertTrue(os.path.exists(os.path.join(work, "dockdiag_partial.zip")), "the partial zip was written after the docking tests")
        self.assertFalse([f for f in os.listdir(work) if f.startswith("dockdiag_result")])
        final_cell = next(c for c in nb.cells if c["id"] == "final").source
        fresh = os.path.join(tmp, "fresh_final.py")
        with open(fresh, "w") as f:
            f.write(final_cell)
        r2 = subprocess.run([sys.executable, fresh], cwd=tmp, env=self.env(tmp), capture_output=True, text=True, timeout=300)
        out = r2.stdout + r2.stderr
        self.assertEqual(r2.returncode, 0, out[-2000:])
        self.assertIn("RECOVERY: zip written", out)
        zips = [f for f in os.listdir(work) if f.startswith("dockdiag_result") and f.endswith(".zip")]
        self.assertEqual(len(zips), 1)
        self.assertTrue(zips[0].endswith("_RECOVERED.zip"))
        files = self.zip_files(os.path.join(work, zips[0]))
        self.assertIn("dock/B_mica_exact/phenix.log", files)
        self.assertIn("state.json", files)
        self.assertTrue(json.loads(files["run_info.json"].decode())["recovered"])

    def test_the_last_cell_in_an_empty_session_says_there_is_nothing_to_pack(self):
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        nb_path = os.path.join(tmp, "nb.ipynb")
        dd.build(nb_path, net_hosts=())
        nb = nbformat.read(nb_path, as_version=4)
        fresh = os.path.join(tmp, "fresh_final.py")
        with open(fresh, "w") as f:
            f.write(next(c for c in nb.cells if c["id"] == "final").source)
        env = {k: v for k, v in os.environ.items() if k != "MICA_WORK"}
        r = subprocess.run([sys.executable, fresh], cwd=tmp, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("RECOVERY: the work folder was not found", r.stdout)


if __name__ == "__main__":
    unittest.main()
