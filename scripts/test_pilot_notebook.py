#!/usr/bin/env python3
"""Tests for make_pilot_notebook.py and the notebook it builds.

Fast tests: the notebook builds, every cell compiles, the embedded files equal the repository files, the committed notebook is current, and the
helper functions of the notebook (run with timeout, secrets, monitor, EMDB header parsing, file search, download, chain_comparison call) work.

The end-to-end tests run the whole notebook as a script against stand-ins: a Phenix download server with a user name and password, a fake Phenix
installer, a local git repository that plays MICA (small programs with MICA's command lines), a file server for the map, the deposited model, the
weights and the Zenodo listing. They need three things from the environment and are skipped without them:

    MICA_TEST_ENV_PYTHON   a Python with Biopython, mrcfile and the packages of env/mica_requirements.txt (the MICA environment)
    MICA_TEST_AF3_DIR      the folder with the 26 AlphaFold Server files (private; entry 26DK is used)
    USALIGN                the compiled US-align program

    USALIGN=... MICA_TEST_ENV_PYTHON=... MICA_TEST_AF3_DIR=... python -W error -m unittest test_pilot_notebook
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import http.server
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import zipfile
from unittest import mock

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_pilot_notebook as mk  # noqa: E402

COMMITTED = os.path.join(HERE, "pilot_mica.ipynb")
ENV_PY = os.environ.get("MICA_TEST_ENV_PYTHON")
AF3_DIR = os.environ.get("MICA_TEST_AF3_DIR")
USALIGN = os.environ.get("USALIGN")
USER, PASSWORD = "pxuser-7", "s3cret-pass-987"
INSTALLER_NAME = "Phenix-2.2.1-6174-Linux-x86_64.sh"


class Kernel:
    """Runs the code cells of a built notebook in this process, in one namespace, like a kernel (a %%writefile cell writes its file)."""

    def __init__(self, test, **build_kw):
        self.tmp = tempfile.mkdtemp(prefix="pilot_")
        test.addCleanup(shutil.rmtree, self.tmp, True)
        self.nb_path = os.path.join(self.tmp, "nb.ipynb")
        mk.build(self.nb_path, require_gpu=False, net_hosts=(), **build_kw)
        self.nb = nbformat.read(self.nb_path, as_version=4)
        self.work, self.big = os.path.join(self.tmp, "work"), os.path.join(self.tmp, "big")
        patcher = mock.patch.dict(os.environ, {"MICA_WORK": self.work, "MICA_BIG": self.big})
        patcher.start()
        test.addCleanup(patcher.stop)
        cwd, path = os.getcwd(), list(sys.path)
        test.addCleanup(os.chdir, cwd)
        test.addCleanup(lambda: sys.path.__setitem__(slice(None), path))
        test.addCleanup(lambda: [sys.modules.pop(m, None) for m in ("mica_inputs", "evaluate_models")])
        self.ns = {"__name__": "__nb__"}

    def cell(self, cid):
        return next(c for c in self.nb.cells if c["id"] == cid)

    def exec_cell(self, cid):
        src = self.cell(cid).source
        if src.startswith("%%writefile"):
            first, text = src.split("\n", 1)
            path = first.split(None, 1)[1].strip()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            exec(compile(src, "cell " + cid, "exec"), self.ns)

    def run_until(self, last_id):
        for c in self.nb.cells:
            if c.cell_type == "code":
                self.exec_cell(c["id"])
            if c["id"] == last_id:
                break

    def load_step6_definitions(self):
        """Runs steps 1 and 2, then defines the functions of step 6 without running any entry."""
        self.run_until("step2-check")
        self.ns["STATE"]["ok"] = False
        self.ns["MICA_DIR"] = os.path.join(self.big, "MICA")
        self.ns["ENV_PY"] = sys.executable
        self.ns["MICA_ENV"] = {}
        self.exec_cell("step6")
        self.ns["STATE"]["ok"] = True


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pilotbuild_")
        cls.path = os.path.join(cls.tmp, "nb.ipynb")
        cls.expected = mk.build(cls.path)
        cls.nb = nbformat.read(cls.path, as_version=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cells_compile_and_have_no_placeholders(self):
        for c in self.nb.cells:
            if c.cell_type == "code" and not c.source.startswith("%%writefile"):
                compile(c.source, c["id"], "exec")
            self.assertEqual(__import__("re").findall(r"@[A-Z_]+@", c.source), [], c["id"])

    def test_embedded_files_equal_the_repository_files(self):
        got = {}
        for c in self.nb.cells:
            if c.source.startswith("%%writefile"):
                first, text = c.source.split("\n", 1)
                got[first.split(None, 1)[1].strip()] = text
        self.assertEqual(sorted(got), sorted(t for _s, t in mk.FILES))
        for src, target in mk.FILES:
            with open(os.path.join(ROOT, src), encoding="utf-8") as f:
                text = f.read()
            self.assertEqual(got[target], text if text.endswith("\n") else text + "\n", target)
            self.assertEqual(self.expected[target], hashlib.sha256(got[target].encode()).hexdigest())

    def test_the_committed_notebook_is_current(self):
        if not os.path.exists(COMMITTED):
            self.skipTest("pilot_mica.ipynb is not built yet")
        fresh = os.path.join(self.tmp, "fresh.ipynb")
        mk.build(fresh)
        a, b = nbformat.read(fresh, as_version=4), nbformat.read(COMMITTED, as_version=4)
        self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_pilot_notebook.py again")

    def test_pins_and_links_are_in_the_notebook(self):
        text = "\n".join(c.source for c in self.nb.cells)
        for needle in (mk.MICA_COMMIT, mk.USALIGN_COMMIT, mk.WEIGHTS_URL, mk.PHENIX_URL, "2.2.1-6174", mk.PY_VERSION, mk.UV_VERSION):
            self.assertIn(needle, text)
        self.assertEqual(len(mk.MICA_COMMIT), 40)
        self.assertEqual(len(mk.USALIGN_COMMIT), 40)

    def test_the_pinned_environment_file_is_complete(self):
        with open(os.path.join(ROOT, "env", "mica_requirements.txt"), encoding="utf-8") as f:
            pins = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        self.assertGreater(len(pins), 150)
        self.assertTrue(all("==" in p for p in pins))
        names = {p.split("==")[0].lower() for p in pins}
        for needed in ("torch", "numpy", "biopython", "open3d", "mrcfile", "scikit-image", "networkx", "freesasa", "atom3d"):
            self.assertIn(needed, names)
        self.assertIn("torch==1.8.1", pins)

    def test_bad_phenix_link_is_refused(self):
        with self.assertRaises(ValueError):
            mk.build(os.path.join(self.tmp, "x.ipynb"), phenix_url="https://example.org/installer.sh")


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.k = Kernel(self)
        self.k.run_until("step2-check")
        self.ns = self.k.ns

    def test_run_collects_output_and_merges_stderr(self):
        rc, last, timed = self.ns["run"](["bash", "-c", "echo out; echo err >&2; exit 3"], quiet=True)
        self.assertEqual((rc, last, timed), (3, ["out", "err"], False))

    def test_run_stops_a_program_that_takes_too_long(self):
        t0 = time.time()
        rc, last, timed = self.ns["run"](["bash", "-c", "echo start; sleep 30"], timeout=1, quiet=True)
        self.assertTrue(timed)
        self.assertLess(time.time() - t0, 15)
        self.assertNotEqual(rc, 0)
        self.assertEqual(last, ["start"])

    def test_run_can_remove_and_set_environment_variables(self):
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/nonexistent", "KEEP_ME": "1"}):
            rc, last, _ = self.ns["run"](["bash", "-c", 'echo "[$PYTHONPATH] [$KEEP_ME] [$NEW]"'], env={"PYTHONPATH": None, "NEW": "x"}, quiet=True)
        self.assertEqual(last, ["[] [1] [x]"])

    def test_run_prints_progress_bars_only_now_and_then_but_keeps_them_in_the_file(self):
        logfile = os.path.join(self.k.work, "bars.log")
        before = len(open(self.ns["LOG"]).read().splitlines())
        self.ns["run"](["bash", "-c", 'for i in $(seq 1 250); do echo "  $i%| bar $i"; done; echo done'], log_file=logfile)
        shown = open(self.ns["LOG"]).read().splitlines()[before:]
        self.assertEqual(len(open(logfile).read().splitlines()), 251)
        self.assertLessEqual(len([ln for ln in shown if "%|" in ln]), 4)
        self.assertIn("done", shown)

    def test_run_reports_a_program_that_cannot_be_started(self):
        rc, last, timed = self.ns["run"](["/nonexistent/program"], quiet=True)
        self.assertEqual(rc, 127)

    def test_secret_comes_from_the_environment_or_is_empty(self):
        with mock.patch.dict(os.environ, {"PHENIX_USER": " abc "}):
            self.assertEqual(self.ns["secret"]("PHENIX_USER", "?"), "abc")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOPE_SECRET", None)
            with mock.patch("getpass.getpass", side_effect=EOFError), mock.patch("builtins.input", side_effect=EOFError):
                self.assertEqual(self.ns["secret"]("NOPE_SECRET", "?"), "")
                self.assertEqual(self.ns["secret"]("NOPE_SECRET", "?", hidden=False), "")

    def test_monitor_works_without_a_gpu(self):
        path = os.path.join(self.k.work, "gpu", "m.csv")
        mon = self.ns["Monitor"](path, every=0.1)
        mon.start()
        time.sleep(0.6)
        info = mon.stop()
        lines = open(path).read().splitlines()
        self.assertEqual(lines[0], "time_utc,gpu_mem_used_mb,gpu_util_pct,ram_available_gb")
        self.assertGreaterEqual(len(lines), 3)
        self.assertGreaterEqual(info["samples"], 2)
        if not shutil.which("nvidia-smi"):
            self.assertEqual(info["peak_gpu_mb"], 0)
        self.assertIsNotNone(info["min_ram_available_gb"])

    def test_pick_big_dir_honours_the_test_setting(self):
        self.assertEqual(self.ns["BIG"], self.k.big)
        self.assertTrue(os.path.isdir(self.k.big))

    def test_the_hash_check_of_step_2_passes_and_detects_a_change(self):
        self.assertTrue(self.ns["STATE"]["ok"])
        with open("scripts/mica_inputs.py", "a") as f:
            f.write("# changed\n")
        self.k.exec_cell("step2-check")
        self.assertFalse(self.ns["STATE"]["ok"])
        self.assertIn("scripts/mica_inputs.py", self.ns["STATE"]["problem"])

    def test_step_1_stops_without_a_gpu_when_one_is_required(self):
        k = Kernel(self)
        k.nb_path = os.path.join(k.tmp, "gpu.ipynb")
        mk.build(k.nb_path, require_gpu=True, net_hosts=())
        k.nb = nbformat.read(k.nb_path, as_version=4)
        real_which = shutil.which
        with mock.patch("shutil.which", lambda name, *a, **kw: None if name == "nvidia-smi" else real_which(name, *a, **kw)):
            k.exec_cell("step1")
        self.assertFalse(k.ns["STATE"]["ok"])
        self.assertIn("no GPU", k.ns["STATE"]["problem"])


class Step6Tests(unittest.TestCase):
    def setUp(self):
        self.k = Kernel(self)
        self.k.load_step6_definitions()
        self.ns = self.k.ns

    XML = """<?xml version="1.0"?>
<emdb xmlns="https://x" version="3.0.9"><map format="CCP4"><contour_list>
 <contour primary="false"><level>0.5</level><source>emdb</source></contour>
 <contour primary="true"><level>0.0213</level><source>author</source></contour>
</contour_list></map>
<structure_determination_list><structure_determination><image_processing><final_reconstruction>
 <resolution units="A" res_type="BY AUTHOR">3.5</resolution></final_reconstruction></image_processing></structure_determination></structure_determination_list></emdb>"""

    def test_contour_and_resolution_from_the_emdb_header_file(self):
        self.assertEqual(self.ns["contour_from_header_xml"](self.XML), (0.0213, 3.5))

    def test_header_file_without_contour(self):
        level, res = self.ns["contour_from_header_xml"]("<emdb><map></map></emdb>")
        self.assertIsNone(level)
        self.assertIsNone(res)

    def test_classify_the_files_of_mica(self):
        c = self.ns["classify"]
        self.assertEqual(c("/o/80558_26dk_MICA_all_atom_model_real_space_refined_000.pdb"), "mica_refined")
        self.assertEqual(c("/o/80558_26dk_MICA_all_atom_model.pdb"), "mica_all_atom")
        self.assertEqual(c("/o/80558_26dk_MICA(init)_ca_model.pdb"), "mica_init_ca")
        self.assertEqual(c("/o/80558_26dk_MICA_ca_model.pdb"), "mica_ca")
        self.assertEqual(c("/o/80558_26dk_MICA_other.pdb"), "mica_other")

    def test_stage_outputs(self):
        so = self.ns["stage_output"]
        self.assertTrue(so("26DK", "/f", "1_process_af3")[0].endswith("AF3_structures/26DK_1/ranked_0.pdb"))
        self.assertEqual(so("26DK", "/f", "2_dock")[0], "/f/26DK_af3_docked.pdb")
        self.assertTrue(so("26DK", "/f", "3_build")[0].endswith("output/*_26dk_MICA_all_atom_model.pdb"))

    def test_the_four_programs_of_mica_get_the_arguments_of_the_pipeline_script(self):
        self.ns["STATE"]["phenix_env"] = "/p/phenix_env.sh"
        progs = dict(self.ns["mica_programs"]("26DK", "/in/26DK", "/in/26DK/emd_80558.map", 0.0213, 3.5))
        self.assertEqual(list(progs), ["1_process_af3", "2_dock", "3_build"])
        self.assertEqual(progs["1_process_af3"][1:], ["utils/process_AF3_results.py", "-f", "/in/26DK/26dk.fasta", "-a", "/in/26DK/AF3_results"])
        self.assertEqual(progs["2_dock"][1:], ["utils/dock_in_map.py", "-m", "/in/26DK/emd_80558.map", "-c", "0.0213", "-r", "3.5", "-f", "/in/26DK/26dk.fasta",
                                               "-a", "/in/26DK/AF3_results", "--phenix_act", "/p/phenix_env.sh"])
        self.assertEqual(progs["3_build"][1:], ["run.py", "-m", "/in/26DK/emd_80558.map", "-f", "/in/26DK/26dk.fasta", "-a", "/in/26DK/AF3_results", "--run_pulchra",
                                               "--pulchra_path=modules/pulchra304/src/pulchra", "--run_phenix", "--phenix_act=/p/phenix_env.sh", "--resolution=3.5",
                                               "--device=cuda"])

    def test_find_af3_dir_unpacks_the_zip_and_ignores_other_zips(self):
        jobs = self.ns["mica_inputs"].read_jobs("data/af3/af3_jobs.csv", "9OC0")
        names = ["fold_%s_model_0.cif" % r["job_name"].lower() for r in jobs]
        base = os.path.join(self.k.tmp, "inputs")
        os.makedirs(base)
        with zipfile.ZipFile(os.path.join(base, "AlphaFold_ServerJOBS.zip"), "w") as z:
            for n in names:
                z.writestr("AlphaFold ServerJOBS/" + n, "data_x\n")
        with zipfile.ZipFile(os.path.join(base, "other_stuff.zip"), "w") as z:
            z.writestr("x.txt", "x")
        with mock.patch.dict(os.environ, {"MICA_AF3_DIR": base}):
            found = self.ns["find_af3_dir"](jobs)
        self.assertTrue(found.startswith(os.path.join(self.k.work, "af3_unzipped")))
        for n in names:
            self.assertTrue(any(n in files for _r, _d, files in os.walk(found)), n)

    def test_find_af3_dir_says_what_to_do_when_nothing_is_found(self):
        jobs = self.ns["mica_inputs"].read_jobs("data/af3/af3_jobs.csv", "26DK")
        empty = os.path.join(self.k.tmp, "empty")
        os.makedirs(empty)
        with mock.patch.dict(os.environ, {"MICA_AF3_DIR": empty}):
            with self.assertRaises(RuntimeError) as cm:
                self.ns["find_af3_dir"](jobs)
        self.assertIn("fold_26dk_e1_model_0.cif", str(cm.exception))
        self.assertIn("Add Input", str(cm.exception))

    def test_download_retries_and_checks_the_size(self):
        calls = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                calls.append(self.path)
                if self.path == "/flaky" and calls.count("/flaky") == 1:
                    self.send_response(500)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = b"x" * (10 if self.path == "/tiny" else 5000)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        url = "http://127.0.0.1:%d" % srv.server_address[1]
        dest = os.path.join(self.k.tmp, "d", "flaky.bin")
        with mock.patch("time.sleep"):
            self.ns["download"](url + "/flaky", dest, 1000)
            self.assertEqual(os.path.getsize(dest), 5000)
            self.assertEqual(calls.count("/flaky"), 2)
            self.assertFalse(os.path.exists(dest + ".part"))
            with self.assertRaises(RuntimeError):
                self.ns["download"](url + "/tiny", os.path.join(self.k.tmp, "d", "tiny.bin"), 1000, tries=2)
            self.assertFalse(os.path.exists(os.path.join(self.k.tmp, "d", "tiny.bin")))

    def _fake_phenix(self, body):
        bindir = os.path.join(self.k.tmp, "fakephenix")
        os.makedirs(bindir, exist_ok=True)
        with open(os.path.join(bindir, "phenix.chain_comparison"), "w") as f:
            f.write("#!/bin/bash\n" + body)
        os.chmod(os.path.join(bindir, "phenix.chain_comparison"), 0o755)
        env = os.path.join(self.k.tmp, "phenix_env.sh")
        with open(env, "w") as f:
            f.write('export PATH="%s:$PATH"\n' % bindir)
        self.ns["STATE"]["phenix_env"] = env

    def test_chain_comparison_keeps_the_whole_output_and_the_command(self):
        self._fake_phenix('echo "target=$1 query=$2"; echo "CA match 190 of 191"\n')
        out = os.path.join(self.k.tmp, "cc.txt")
        status = self.ns["chain_comparison"]("/q/pred.pdb", "/t/native.cif", out)
        self.assertEqual(status, "ok")
        text = open(out).read()
        self.assertTrue(text.startswith("# phenix.chain_comparison /t/native.cif /q/pred.pdb\n"))      # target first, then query
        self.assertIn("target=/t/native.cif query=/q/pred.pdb", text)
        self.assertIn("CA match 190 of 191", text)

    def test_chain_comparison_retries_with_the_deposited_model_as_pdb(self):
        # the stand-in reads only .pdb files: the first call (mmCIF) fails, the second (converted by Biopython) works
        if not ENV_PY:
            self.skipTest("needs MICA_TEST_ENV_PYTHON (Biopython)")
        self._fake_phenix('case "$1" in *.pdb) echo "read $1"; exit 0;; *) echo "Sorry: cannot read $1"; exit 1;; esac\n')
        self.ns["ENV_PY"], self.ns["MICA_ENV"] = ENV_PY, {}
        cif = [os.path.join(r, f) for r, _d, fs in os.walk(AF3_DIR or "") for f in fs if f == "fold_26dk_e1_model_0.cif"]
        if not cif:
            self.skipTest("needs MICA_TEST_AF3_DIR")
        target = os.path.join(self.k.tmp, "native.cif")
        shutil.copy(cif[0], target)
        out = os.path.join(self.k.tmp, "cc.txt")
        status = self.ns["chain_comparison"]("/q/pred.pdb", target, out)
        self.assertEqual(status, "ok (deposited model as pdb)")
        text = open(out).read()
        self.assertIn("# phenix.chain_comparison %s /q/pred.pdb" % target, text)
        self.assertIn("# phenix.chain_comparison %s /q/pred.pdb" % target.replace(".cif", "_as.pdb"), text)
        self.assertIn("Sorry: cannot read", text)

    def test_chain_comparison_failure_is_reported_not_raised(self):
        self._fake_phenix('echo "Sorry: something"; exit 3\n')
        status = self.ns["chain_comparison"]("/q/p.pdb", "/t/n.cif", os.path.join(self.k.tmp, "cc.txt"))
        self.assertEqual(status, "failed (exit code 3)")


@unittest.skipUnless(os.environ.get("MICA_TEST_UV_CACHE"), "set MICA_TEST_UV_CACHE (a uv cache that holds the wheels) and MICA_TEST_UV_PYTHONS to run this slow test")
class EnvironmentBuildTest(unittest.TestCase):
    """Step 4 for real: uv installs Python 3.8.20 and the 171 pinned packages; the notebook then finds no difference to the pinned list."""

    def test_step_4_builds_the_pinned_environment(self):
        k = Kernel(self)
        extra = {"UV_CACHE_DIR": os.environ["MICA_TEST_UV_CACHE"], "UV_PYTHON_INSTALL_DIR": os.environ.get("MICA_TEST_UV_PYTHONS", "")}
        extra = {a: b for a, b in extra.items() if b}
        with mock.patch.dict(os.environ, extra):
            k.run_until("step2-check")
            k.exec_cell("step4")
        ns = k.ns
        self.assertTrue(ns["STATE"]["ok"], ns["STATE"]["problem"])
        self.assertTrue(os.path.exists(ns["ENV_PY"]))
        self.assertEqual(ns["STATE"]["env"]["differences_to_pinned_list"], [])
        self.assertEqual(ns["STATE"]["env"]["packages"], 171)
        self.assertTrue(os.path.exists(os.path.join(k.work, "mica_env_freeze.txt")))
        with mock.patch.dict(os.environ, extra):                       # a second call finds the environment and does not build it again
            k.exec_cell("step4")
        self.assertTrue(ns["STATE"]["ok"])


# ---------------------------------------------------------------- end to end, with stand-ins

STUB_PHENIX = r'''#!@PYTHON@
import os, shutil, sys
prog, args = os.path.basename(sys.argv[0]), sys.argv[1:]
if prog == "phenix.version":
    print("Phenix version: 2.2.1-6174 (stub)")
elif prog == "phenix.dock_in_map":
    kv = dict(a.split("=", 1) for a in args if "=" in a)
    assert os.path.exists(kv["map_file"]) and float(kv["resolution"]) > 0 and "sequence" in kv
    shutil.copy(kv["search_model"], kv["pdb_out"])
    print("Final score 0.9")
elif prog == "phenix.real_space_refine":
    if os.environ.get("STUB_REFINE_FAIL"):
        print("Sorry: stub refinement failure")
        sys.exit(1)
    model = [a for a in args if a.endswith(".pdb")][0]
    shutil.copy(model, os.path.splitext(os.path.basename(model))[0] + "_real_space_refined_000.pdb")
    print("refined")
elif prog == "phenix.chain_comparison":
    if "--help" in args:
        print("usage: phenix.chain_comparison target.pdb query.pdb [options]")
    else:
        if os.environ.get("STUB_CC_FAIL"):
            print("Sorry: stub failure")
            sys.exit(3)
        print("target %s query %s\nCA match: 190 of 191 (99.5 %%)" % (args[0], args[1]))
'''

FAKE_INSTALLER = r"""#!/bin/bash
while [ $# -gt 0 ]; do
  case "$1" in -p) PREFIX="$2"; shift ;; esac
  shift
done
mkdir -p "$PREFIX/bin"
cat > "$PREFIX/bin/phenix.version" <<'PYEOF'
@STUB@
PYEOF
for p in phenix.dock_in_map phenix.real_space_refine phenix.chain_comparison; do cp "$PREFIX/bin/phenix.version" "$PREFIX/bin/$p"; done
chmod +x "$PREFIX/bin/"phenix.*
echo "export PATH=\"$PREFIX/bin:\$PATH\"" > "$PREFIX/phenix_env.sh"
echo "installed (stub)"
exit 0
"""

STUB_MICA = {
    "utils/__init__.py": "",
    "utils/modeler.py": "", "utils/predict.py": "", "utils/create_grids.py": "", "utils/preprocessing.py": "",
    "utils/process_AF3_results.py": r'''
import argparse, os, sys
from Bio.PDB import MMCIFParser, PDBIO


def main():
    p = argparse.ArgumentParser(); p.add_argument("-f", required=True); p.add_argument("-a", required=True); a = p.parse_args()
    if os.environ.get("STUB_FAIL_STEP") == "1_process_af3":
        print("boom in step 1"); sys.exit(1)
    entry_dir = os.path.dirname(a.a)
    for line in open(a.f):
        if line.startswith(">"):
            h = line[1:].split("|")[0]; pdb, k = h.split("_")
            s = MMCIFParser(QUIET=True).get_structure("x", os.path.join(a.a, "%s_%s" % (pdb.lower(), k), "fold_%s_%s_model_0.cif" % (pdb.lower(), k)))
            out = os.path.join(entry_dir, "AF3_structures", h); os.makedirs(out, exist_ok=True)
            io = PDBIO(); io.set_structure(s); io.save(os.path.join(out, "ranked_0.pdb"))


if __name__ == "__main__":
    main()
''',
    "utils/dock_in_map.py": r'''
import argparse, os, shutil, subprocess, sys


def main():
    p = argparse.ArgumentParser()
    for flag in ("-m", "-c", "-r", "-f", "-a", "--phenix_act"):
        p.add_argument(flag, required=True)
    a = p.parse_args()
    float(a.c); float(a.r)
    if os.environ.get("STUB_FAIL_STEP") == "2_dock":
        print("boom in step 2"); sys.exit(1)
    entry_dir = os.path.dirname(a.a); name = os.path.basename(entry_dir)
    src = os.path.join(entry_dir, "AF3_structures"); first = sorted(os.listdir(src))[0]
    dd = os.path.join(entry_dir, "AF3_docked_models"); os.makedirs(dd, exist_ok=True)
    out = os.path.join(dd, first + "_docked.pdb")
    cmd = "bash -c 'source %s && phenix.dock_in_map search_model=%s map_file=%s nproc=1 quick=False resolution=%s pdb_out=%s dock_chains_individually=True sequence=%s'" % (
        a.phenix_act, os.path.join(src, first, "ranked_0.pdb"), a.m, a.r, out, a.f)
    rc = subprocess.run(cmd, shell=True).returncode
    if rc:
        sys.exit(rc)
    shutil.copy(out, os.path.join(entry_dir, name + "_af3_docked.pdb"))


if __name__ == "__main__":
    main()
''',
    "utils/emdb_extractor.py": r'''
import argparse
p = argparse.ArgumentParser(); p.add_argument("--emdb_id", required=True); a = p.parse_args()
print("Fetching data for EMDB ID: EMD-%s" % a.emdb_id); print("Contour Level: 0.0213"); print("Resolution: 3.5")
''',
    "run.py": r'''
import argparse, os, shutil, subprocess, sys
p = argparse.ArgumentParser()
p.add_argument("-m", required=True); p.add_argument("-f", required=True); p.add_argument("-a", required=True)
p.add_argument("--run_pulchra", action="store_true"); p.add_argument("--pulchra_path"); p.add_argument("--run_phenix", action="store_true")
p.add_argument("--phenix_act"); p.add_argument("--resolution", type=float); p.add_argument("--device", default="cuda"); p.add_argument("--phenix_param", default="modules/phenix.eff")
a = p.parse_args()
assert os.path.exists("trained_models/MICA_best_model.pth"), "weights missing"
assert os.path.exists(a.pulchra_path) and os.access(a.pulchra_path, os.X_OK), "pulchra missing"
assert a.device == "cuda"
if os.environ.get("STUB_FAIL_STEP") == "3_build":
    print("boom in step 3"); sys.exit(1)
map_id = os.path.basename(a.m).split("emd_")[-1].split(".")[0]; pdb_id = os.path.basename(a.f).split(".fasta")[0]
entry_dir = os.path.dirname(a.a); src = os.path.join(entry_dir, "AF3_structures"); first = sorted(os.listdir(src))[0]
model = os.path.join(src, first, "ranked_0.pdb"); os.makedirs("output", exist_ok=True)
ca = "output/%s_%s_MICA_ca_model.pdb" % (map_id, pdb_id); full = "output/%s_%s_MICA_all_atom_model.pdb" % (map_id, pdb_id)
lines = open(model).read().splitlines()
open(ca, "w").write("\n".join(l for l in lines if l.startswith("ATOM") and l[12:16].strip() == "CA") + "\nEND\n")
shutil.copy(model, full)
open("output/time_cost_%s_%s_MICA.csv" % (map_id, pdb_id), "w").write("step,time\ngetData,1\n")
if a.run_phenix:
    cmd = 'bash modules/phenix.sh %s %s "phenix.real_space_refine %s %s %s resolution=%s"' % (os.path.abspath(a.phenix_act), os.path.abspath("output"), os.path.abspath(full), os.path.abspath(a.m), os.path.abspath(a.phenix_param), a.resolution)
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
''',
    "modules/phenix.sh": "source $1\ncd $2\n$3",
    "modules/phenix.eff": "debug = False\n",
    "modules/pulchra304/src/pulchra": "#!/bin/sh\necho pulchra stub\n",
}


class Files(http.server.BaseHTTPRequestHandler):
    routes: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = Files.routes.get(self.path.split("?")[0])
        if body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class PhenixServer(http.server.BaseHTTPRequestHandler):
    body = b""

    def log_message(self, *a):
        pass

    def do_GET(self):
        import base64
        want = "Basic " + base64.b64encode(("%s:%s" % (USER, PASSWORD)).encode()).decode()
        if self.headers.get("Authorization") != want:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="phenix"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(PhenixServer.body)))
        self.end_headers()
        self.wfile.write(PhenixServer.body)


def serve(handler):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True, check=True).stdout.strip()


def to_script(nb, path):
    """The notebook as a plain Python script (a %%writefile cell becomes a write of its file)."""
    parts = []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        if c.source.startswith("%%writefile"):
            first, text = c.source.split("\n", 1)
            target = first.split(None, 1)[1].strip()
            parts.append("os.makedirs(os.path.dirname(%r) or '.', exist_ok=True)\nopen(%r, 'w', encoding='utf-8').write(%r)\n" % (target, target, text))
        else:
            parts.append("# ---- cell %s\n%s" % (c["id"], c.source))
    with open(path, "w", encoding="utf-8") as f:
        f.write("import os\n" + "\n".join(parts))


@unittest.skipUnless(ENV_PY and AF3_DIR and USALIGN and os.path.exists(os.path.join(AF3_DIR or "", "fold_26dk_e1_model_0.cif")) or
                     (ENV_PY and AF3_DIR and USALIGN and any("fold_26dk_e1_model_0.cif" in f for _r, _d, f in os.walk(AF3_DIR))),
                     "needs MICA_TEST_ENV_PYTHON, MICA_TEST_AF3_DIR (with the 26DK file) and USALIGN")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="pilote2e_")
        # the real AlphaFold Server file of 26DK doubles as the deposited model (served as 26DK.cif)
        cif = [os.path.join(r, f) for r, _d, fs in os.walk(AF3_DIR) for f in fs if f == "fold_26dk_e1_model_0.cif"][0]
        with open(cif, "rb") as f:
            cif_bytes = f.read()
        mapfile = os.path.join(cls.root, "emd_80558.map")
        subprocess.run([ENV_PY, "-c", "import mrcfile, numpy; m = mrcfile.new(%r, overwrite=True); m.set_data(numpy.random.rand(48, 48, 48).astype('float32')); "
                        "m.voxel_size = 1.0; m.close()" % mapfile], check=True)
        with open(mapfile, "rb") as f:
            map_gz = gzip.compress(f.read())
        # the weights and the Zenodo listing
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            data = b"stub weights" * 100
            ti = tarfile.TarInfo("trained_models/MICA_best_model.pth")
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
        cls.weights = buf.getvalue()
        md5 = hashlib.md5(cls.weights).hexdigest()
        cls.srv, cls.url = serve(Files)
        Files.routes = {"/map/emd_80558.map.gz": map_gz, "/cif/26DK.cif": cif_bytes, "/weights/trained_models.tar.gz": cls.weights,
                        "/zenodo/api": json.dumps({"files": [{"key": "trained_models.tar.gz", "checksum": "md5:" + md5, "size": len(cls.weights)}]}).encode()}
        cls.phenix_srv, cls.phenix_url = serve(PhenixServer)
        PhenixServer.body = FAKE_INSTALLER.replace("@STUB@", STUB_PHENIX.replace("@PYTHON@", sys.executable)).encode()
        # entries table with local links
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
        # the stand-in for MICA: a local git repository
        cls.repo = os.path.join(cls.root, "mica_stub")
        for rel, text in STUB_MICA.items():
            path = os.path.join(cls.repo, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        os.chmod(os.path.join(cls.repo, "modules/pulchra304/src/pulchra"), 0o755)
        sh("git init -q -b main . && git add -A && git -c user.name=t -c user.email=t@t commit -q -m stub", cwd=cls.repo)
        cls.commit = sh("git rev-parse HEAD", cwd=cls.repo)

    @classmethod
    def tearDownClass(cls):
        for s in (cls.srv, cls.phenix_srv):
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def run_notebook(self, extra_env=None, user=USER, password=PASSWORD, entries=("26DK",)):
        tmp = tempfile.mkdtemp(prefix="pilotrun_", dir=self.root)
        nb_path = os.path.join(tmp, "nb.ipynb")
        url = "%s/download?version=2.2.1-6174&file=%s" % (self.phenix_url, INSTALLER_NAME)
        mk.build(nb_path, entries=entries, phenix_url=url, mica_repo="file://" + self.repo, mica_commit=self.commit, weights_url=self.url + "/weights/trained_models.tar.gz",
                 zenodo_api=self.url + "/zenodo/api", entries_path=self.entries, require_gpu=False, env_python=ENV_PY, usalign_bin=USALIGN, net_hosts=())
        nb = nbformat.read(nb_path, as_version=4)
        for c in nb.cells:                                      # the fake installer is small
            if c["id"] == "step3":
                c.source = c.source.replace("MIN_INSTALLER_MB = 50", "MIN_INSTALLER_MB = 0").replace("MIN_FREE_GB = 20", "MIN_FREE_GB = 0")
        script = os.path.join(tmp, "nb.py")
        to_script(nb, script)
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), PHENIX_USER=user, PHENIX_PASSWORD=password,
                   MICA_AF3_DIR=AF3_DIR)
        env.pop("STUB_FAIL_STEP", None)
        env.update(extra_env or {})
        r = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
        zips = [os.path.join(tmp, "work", f) for f in os.listdir(os.path.join(tmp, "work")) if f.endswith(".zip")] if os.path.isdir(os.path.join(tmp, "work")) else []
        return r, zips, tmp

    def zip_info(self, path):
        with zipfile.ZipFile(path) as z:
            return {n: z.read(n) for n in z.namelist()}

    def test_a_complete_run(self):
        r, zips, tmp = self.run_notebook()
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out, out[-3000:])
        self.assertEqual(len(zips), 1)
        self.assertNotIn("_INCOMPLETE", zips[0])
        files = self.zip_info(zips[0])
        names = [n.split("/", 1)[1] for n in files]
        for needed in ("run.log", "run_info.json", "pilot_metrics.csv", "entry_status.csv", "machine.txt", "phenix_help.txt", "logs/26DK_1_process_af3.log",
                       "logs/26DK_3_build.log", "gpu/26DK_3_build.csv", "scoring/26DK_mica_refined_chain_comparison.txt", "scoring/26DK_mica_all_atom_usalign.txt",
                       "mica_env_freeze.txt"):
            if needed == "mica_env_freeze.txt":
                continue                                       # only written when the environment is built by the notebook
            self.assertIn(needed, names)
        self.assertTrue(any(n.startswith("pilot_result_") is False and n.endswith("_MICA_all_atom_model.pdb") for n in names))
        self.assertFalse(any("AF3_results" in n or n.endswith((".map", ".mrc")) for n in names), names)
        self.assertFalse(any(PASSWORD.encode() in data for data in files.values()), "the password is in the zip")
        rows = list(csv.DictReader(io.StringIO(files[[n for n in files if n.endswith("pilot_metrics.csv")][0]].decode())))
        kinds = {r["model_kind"]: r for r in rows}
        self.assertEqual(sorted(kinds), ["af3_docked", "af3_model_0", "mica_all_atom", "mica_ca", "mica_refined"])
        for k, row in kinds.items():
            self.assertEqual(row["status"], "ok", (k, row))
            self.assertEqual(row["chain_comparison"], "ok", k)
        self.assertGreater(float(kinds["mica_all_atom"]["tm_score"]), 0.99)
        self.assertGreater(float(kinds["mica_ca"]["tm_score"]), 0.9)
        info = json.loads(files[[n for n in files if n.endswith("run_info.json")][0]].decode())
        self.assertEqual(info["mica_commit"], self.commit)
        self.assertEqual(info["entries"]["26DK"]["status"], "ok")
        self.assertEqual(sorted(info["entries"]["26DK"]["steps"]), ["1_process_af3", "2_dock", "3_build"])
        self.assertEqual(info["entries"]["26DK"]["contour"], 0.0213)
        self.assertEqual(info["weights"]["md5"], info["weights"]["md5_listed_by_zenodo"])
        self.assertIn("2.2.1-6174", info["phenix_version_output"])
        self.assertEqual(len(info["phenix_installer_sha256"]), 64)

    def test_a_wrong_password_stops_the_run_but_still_packs_the_log(self):
        r, zips, tmp = self.run_notebook(password="wrong-password-1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-2000:])
        self.assertIn("did not accept the user name and password", out)
        self.assertIn("RESULT: STOPPED", out)
        self.assertEqual(len(zips), 1)
        self.assertIn("_INCOMPLETE", zips[0])
        files = self.zip_info(zips[0])
        self.assertTrue(any(n.endswith("run.log") for n in files))
        self.assertFalse(any(b"wrong-password-1" in data for data in files.values()))

    def test_a_missing_secret_says_where_to_put_it(self):
        r, zips, tmp = self.run_notebook(user="", password="")
        out = r.stdout + r.stderr
        self.assertIn("no Phenix user name or password found", out)
        self.assertIn("Add-ons -> Secrets", out)
        self.assertEqual(len(zips), 1)

    def test_a_failing_mica_program_is_reported_with_its_name_and_the_zip_is_incomplete(self):
        r, zips, tmp = self.run_notebook(extra_env={"STUB_FAIL_STEP": "3_build"})
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-2000:])
        self.assertIn("ENTRY 26DK STOPPED in 3_build: exit code 1", out)
        self.assertIn("boom in step 3", out)
        self.assertIn("AT LEAST ONE ENTRY DID NOT FINISH", out)
        files = self.zip_info(zips[0])
        self.assertIn("_INCOMPLETE", zips[0])
        status = list(csv.DictReader(io.StringIO(files[[n for n in files if n.endswith("entry_status.csv")][0]].decode())))
        self.assertEqual((status[0]["status"], status[0]["failed_step"]), ("failed", "3_build"))
        # the earlier programs are still scored: the docked model and the AlphaFold control
        rows = list(csv.DictReader(io.StringIO(files[[n for n in files if n.endswith("pilot_metrics.csv")][0]].decode())))
        self.assertEqual(sorted(r["model_kind"] for r in rows), ["af3_docked", "af3_model_0"])

    def test_a_silent_refinement_failure_is_run_again_with_its_output_kept(self):
        r, zips, tmp = self.run_notebook(extra_env={"STUB_REFINE_FAIL": "1"})
        out = r.stdout + r.stderr
        self.assertIn("the all-atom model exists but the refined model does not", out)
        self.assertIn("phenix.real_space_refine exit code 1", out)
        files = self.zip_info(zips[0])
        self.assertTrue(any(n.endswith("logs/26DK_refine_diagnosis.log") for n in files))
        self.assertIn(b"stub refinement failure", [data for n, data in files.items() if n.endswith("refine_diagnosis.log")][0])
        rows = list(csv.DictReader(io.StringIO(files[[n for n in files if n.endswith("pilot_metrics.csv")][0]].decode())))
        self.assertNotIn("mica_refined", [r["model_kind"] for r in rows])

    def test_a_failing_chain_comparison_does_not_stop_the_scoring(self):
        r, zips, tmp = self.run_notebook(extra_env={"STUB_CC_FAIL": "1"})
        out = r.stdout + r.stderr
        self.assertIn("RESULT: OK", out, out[-2000:])
        files = self.zip_info(zips[0])
        rows = list(csv.DictReader(io.StringIO(files[[n for n in files if n.endswith("pilot_metrics.csv")][0]].decode())))
        self.assertTrue(all(r["chain_comparison"] == "failed (exit code 3)" for r in rows))
        self.assertTrue(all(r["status"] == "ok" for r in rows))

    def test_a_second_run_in_the_same_folders_does_not_repeat_finished_programs(self):
        r, zips, tmp = self.run_notebook()
        self.assertIn("RESULT: OK", r.stdout)
        # run the same notebook script again in the same work folders
        script = os.path.join(tmp, "nb.py")
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), PHENIX_USER=USER, PHENIX_PASSWORD=PASSWORD, MICA_AF3_DIR=AF3_DIR)
        r2 = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
        out = r2.stdout + r2.stderr
        self.assertEqual(out.count("finished earlier in this session, not repeated"), 3, out[-2000:])
        self.assertIn("Phenix is already installed", out)
        self.assertIn("RESULT: OK", out)


if __name__ == "__main__":
    unittest.main()
