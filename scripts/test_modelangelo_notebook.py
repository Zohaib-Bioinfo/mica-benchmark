#!/usr/bin/env python3
"""Tests for make_modelangelo_notebook.py and the notebook it builds.

Fast tests: the notebook builds, every cell compiles, the embedded files equal the repository files, the pins are complete, the committed notebook is current.

End-to-end tests run the whole notebook as a script against stand-ins: a local git repository that plays ModelAngelo (a `model_angelo` package with the commands
`--version`, `setup_weights` and `build`, which behave as the tests tell them), a file server for the map and the deposited model, the real US-align. They need

    MICA_TEST_ENV_PYTHON   any Python 3 with numpy and mrcfile (it plays the ModelAngelo environment; the stand-in uses only the standard library)
    MICA_TEST_AF3_DIR      the folder with the 26 AlphaFold Server files (private; the 26DK file doubles as the deposited model and as the "built" model)
    USALIGN                the compiled US-align program

    USALIGN=... MICA_TEST_ENV_PYTHON=... MICA_TEST_AF3_DIR=... python -W error -m unittest test_modelangelo_notebook
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
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
import make_modelangelo_notebook as ma  # noqa: E402
import test_pilot_notebook as T  # noqa: E402

COMMITTED = os.path.join(HERE, "pilot_modelangelo.ipynb")
ENV_PY, AF3_DIR, USALIGN = T.ENV_PY, T.AF3_DIR, T.USALIGN
SIMPLE_IMPORT_TEST = "import numpy, mrcfile; print('stand-in environment: numpy', numpy.__version__)"

STUB_MA = {
    "model_angelo/__init__.py": '__version__ = "1.0.18-stub"\n',
    "model_angelo/apps/__init__.py": "",
    "model_angelo/apps/build.py": "",
    "model_angelo/utils/__init__.py": "",
    "model_angelo/utils/setup_weights.py": "",
    "model_angelo/model_definitions/nucleotides/c_alpha/model.py": "# stand-in\n",
    "model_angelo/__main__.py": r'''
import argparse, os, shutil, sys, time


def say(text):
    print(text)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="cmd")
    w = sub.add_parser("setup_weights"); w.add_argument("--bundle-name", default="original")
    b = sub.add_parser("build")
    b.add_argument("--volume-path", "-v", required=True); b.add_argument("--protein-fasta", "-pf", required=True); b.add_argument("--output-dir", "-o", default="output")
    b.add_argument("--device", "-d", default=None)
    a = parser.parse_args()
    if a.version:
        say("ModelAngelo 1.0.18-stub"); return
    if a.cmd == "setup_weights":
        if os.environ.get("STUB_NO_WEIGHTS"):
            say("Setting up bundle with name: %s for the first time." % a.bundle_name); say("RuntimeError: Checksum of downloaded file does not match expected checksum."); sys.exit(1)
        home = os.environ["TORCH_HOME"]
        d = os.path.join(home, "hub", "checkpoints", "model_angelo_v1.0", a.bundle_name)
        for rel, text in (("config.json", '{"gnn_infer_args": {"num_rounds": 3, "esm_model": "esm1b_t33_650M_UR50S"}}'), ("c_alpha/chkpt.torch", "x" * 2000000),
                          ("gnn/chkpt.torch", "y" * 3000000), ("success.txt", "ok")):
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
            open(os.path.join(d, rel), "w").write(text)
        os.makedirs(os.path.join(home, "hub", "checkpoints"), exist_ok=True)
        open(os.path.join(home, "hub", "checkpoints", "esm1b_t33_650M_UR50S.pt"), "w").write("z" * 4000000)
        say("Checking checksum of downloaded file."); say("Checksum of downloaded file matches expected checksum.")
        say("Successful!"); say("Path to model is %s" % d); return
    if a.cmd == "build":
        assert os.environ.get("TORCH_HOME"), "TORCH_HOME is not set"
        assert os.environ.get("MPLBACKEND") == "Agg", "MPLBACKEND=%r: pyplot cannot be imported without matplotlib-inline" % os.environ.get("MPLBACKEND")
        assert os.path.exists(a.volume_path) and a.volume_path.endswith(".mrc"), a.volume_path
        assert os.path.exists(a.protein_fasta) and a.device == "0", (a.protein_fasta, a.device)
        out = a.output_dir; os.makedirs(out, exist_ok=True)
        name = os.path.basename(os.path.normpath(out))
        open(os.path.join(out, "model_angelo.log"), "w").write("stand-in log\n")
        with open(os.path.join(os.path.dirname(os.path.dirname(out)), "build_calls.txt"), "a") as f:
            f.write(" ".join(sys.argv[1:]) + "\n")
        if os.environ.get("STUB_MA_FAIL"):
            say("Error in ModelAngelo: stand-in failure"); sys.exit(1)
        say("Initial C-alpha prediction")
        for i in range(int(os.environ.get("STUB_MA_STEPS", "1"))):
            say("GNN model refinement, round %d / 3 ... 40%%|####      | 4/10" % (i + 1)); time.sleep(float(os.environ.get("STUB_MA_SLOW", "0")))
        model = os.environ["STUB_MA_MODEL"]
        shutil.copy(model, os.path.join(out, name + ".cif")); shutil.copy(model, os.path.join(out, name + "_raw.cif"))
        os.makedirs(os.path.join(out, "entropy_scores"), exist_ok=True)
        shutil.copy(model, os.path.join(out, "entropy_scores", name + ".cif"))
        say("ModelAngelo build has been completed successfully!")


if __name__ == "__main__":
    main()
''',
}


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mabuild_")
        cls.path = os.path.join(cls.tmp, "nb.ipynb")
        cls.expected = ma.build(cls.path)
        cls.nb = nbformat.read(cls.path, as_version=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cells_compile_and_have_no_placeholders(self):
        for c in self.nb.cells:
            if c.cell_type == "code" and not c.source.startswith("%%writefile"):
                compile(c.source, c["id"], "exec")
            self.assertEqual(re.findall(r"@[A-Z_0-9]+@", c.source), [], c["id"])

    def test_cell_order(self):
        ids = [c["id"] for c in self.nb.cells if c.cell_type == "code" and not c["id"].startswith("file-")]
        self.assertEqual(ids, ["step1", "step2-dirs", "step2-check", "robust", "step3", "step4", "step5", "step6", "final"])

    def test_embedded_files_equal_the_repository_files(self):
        got = {}
        for c in self.nb.cells:
            if c.source.startswith("%%writefile"):
                first, text = c.source.split("\n", 1)
                got[first.split(None, 1)[1].strip()] = text
        self.assertEqual(sorted(got), sorted(t for _s, t in ma.FILES))
        for src, target in ma.FILES:
            with open(os.path.join(ROOT, src), encoding="utf-8") as f:
                text = f.read()
            self.assertEqual(got[target], text if text.endswith("\n") else text + "\n", target)
            self.assertEqual(self.expected[target], hashlib.sha256(got[target].encode()).hexdigest())

    def test_the_committed_notebook_is_current(self):
        if not os.path.exists(COMMITTED):
            self.skipTest("pilot_modelangelo.ipynb is not built yet")
        fresh = os.path.join(self.tmp, "fresh.ipynb")
        ma.build(fresh)
        a, b = nbformat.read(fresh, as_version=4), nbformat.read(COMMITTED, as_version=4)
        self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_modelangelo_notebook.py again")

    def test_pins_and_links_are_in_the_notebook(self):
        text = "\n".join(c.source for c in self.nb.cells)
        for needle in (ma.MA_COMMIT, ma.MA_REPO, ma.MA_PY_VERSION, ma.MA_BUNDLE, "setup_weights", "github.com/pylelab/USalign"):
            self.assertIn(needle, text)
        self.assertEqual(len(ma.MA_COMMIT), 40)

    def test_the_pinned_environment_file_is_complete(self):
        with open(os.path.join(ROOT, "env", "modelangelo_requirements.txt"), encoding="utf-8") as f:
            pins = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        self.assertTrue(all("==" in p for p in pins))
        names = {p.split("==")[0].lower() for p in pins}
        for needed in ("torch", "torchvision", "numpy", "biopython", "mrcfile", "einops", "fair-esm", "pyhmmer", "loguru", "psutil", "scipy", "pandas", "matplotlib", "tqdm"):
            self.assertIn(needed, names)
        self.assertIn("torch==2.9.1", pins)
        self.assertIn("numpy==1.26.4", pins)          # setup.py of ModelAngelo: numpy<2.0
        self.assertIn("fair-esm==1.0.3", pins)
        self.assertIn("pyhmmer==0.7.1", pins)
        self.assertEqual(len(pins), len(set(pins)))

    def test_the_command_is_the_one_of_the_readme_with_the_gpu_named(self):
        step6 = next(c for c in self.nb.cells if c["id"] == "step6").source
        self.assertIn('cmd = [ENV_PY, "-m", "model_angelo", "build", "-v", map_path, "-pf", fasta, "-o", out_dir, "--device", "0"]', step6)

    def test_a_gpu_is_required(self):
        self.assertIn("REQUIRE_GPU = True", next(c for c in self.nb.cells if c["id"] == "step1").source)


@unittest.skipUnless(ENV_PY and AF3_DIR and USALIGN and any("fold_26dk_e1_model_0.cif" in f for _r, _d, f in os.walk(AF3_DIR or "")),
                     "needs MICA_TEST_ENV_PYTHON, MICA_TEST_AF3_DIR (with the 26DK file) and USALIGN")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="marun_")
        cls.cif = [os.path.join(r, f) for r, _d, fs in os.walk(AF3_DIR) for f in fs if f == "fold_26dk_e1_model_0.cif"][0]
        with open(cls.cif, "rb") as f:
            cif_bytes = f.read()
        mapfile = os.path.join(cls.root, "emd_80558.map")
        subprocess.run([ENV_PY, "-c", "import mrcfile, numpy; m = mrcfile.new(%r, overwrite=True); m.set_data(numpy.random.rand(32, 32, 32).astype('float32')); "
                        "m.voxel_size = 1.0; m.close()" % mapfile], check=True)
        with open(mapfile, "rb") as f:
            map_gz = gzip.compress(f.read())
        cls.srv, cls.url = T.serve(T.Files)
        T.Files.routes = {"/map/emd_80558.map.gz": map_gz, "/cif/26DK.cif": cif_bytes}
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
        cls.repo = os.path.join(cls.root, "ma_stub")
        for rel, text in STUB_MA.items():
            path = os.path.join(cls.repo, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        T.sh("git init -q -b main . && git add -A && git -c user.name=t -c user.email=t@t commit -q -m stub", cwd=cls.repo)
        cls.commit = T.sh("git rev-parse HEAD", cwd=cls.repo)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def run_notebook(self, require_gpu=False, patch=None, **extra_env):
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        nb_path = os.path.join(tmp, "nb.ipynb")
        ma.build(nb_path, ma_repo="file://" + self.repo, ma_commit=self.commit, entries_path=self.entries, env_python=ENV_PY, usalign_bin=USALIGN, net_hosts=(),
                 require_gpu=require_gpu, import_test=SIMPLE_IMPORT_TEST)
        nb = nbformat.read(nb_path, as_version=4)
        for c in nb.cells:
            for old, new in (patch or {}).get(c["id"], []):
                self.assertIn(old, c.source)
                c.source = c.source.replace(old, new)
        script = os.path.join(tmp, "nb.py")
        T.to_script(nb, script)
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1", STUB_MA_MODEL=self.cif, MPLBACKEND="module://matplotlib_inline.backend_inline")
        for k in [k for k in env if k.startswith("STUB_") and k != "STUB_MA_MODEL"]:
            env.pop(k)
        env.update(extra_env)
        r = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
        work = os.path.join(tmp, "work")
        zips = sorted(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".zip")) if os.path.isdir(work) else []
        return r, zips, tmp

    @staticmethod
    def zip_files(path):
        with zipfile.ZipFile(path) as z:
            return {n.split("/", 1)[1]: z.read(n) for n in z.namelist()}

    def test_a_complete_run(self):
        r, zips, tmp = self.run_notebook(STUB_MA_STEPS="3", STUB_MA_SLOW="1.5")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out, out[-3000:])
        self.assertEqual(len(zips), 1, zips)
        self.assertNotIn("_INCOMPLETE", zips[0])
        self.assertFalse(os.path.exists(os.path.join(tmp, "work", "modelangelo_pilot_partial.zip")))
        files = self.zip_files(zips[0])
        for needed in ("run.log", "state.json", "run_info.json", "pilot_metrics.csv", "entry_status.csv", "machine.txt", "cpu_facts.txt", "modelangelo_config.json",
                       "inputs/26dk.fasta", "logs/26DK_modelangelo.log", "logs/setup_weights.log", "gpu/26DK_modelangelo.csv", "scoring/26DK_modelangelo_usalign.txt",
                       "scoring/26DK_modelangelo_raw_usalign.txt", "ma_out/26DK_modelangelo/26DK_modelangelo.cif", "ma_out/26DK_modelangelo/26DK_modelangelo_raw.cif",
                       "ma_out/26DK_modelangelo/model_angelo.log", "ma_out/26DK_modelangelo/entropy_scores/26DK_modelangelo.cif"):
            self.assertIn(needed, files)
        self.assertFalse(any(n.endswith((".map", ".mrc", ".ccp4")) or "AF3_results" in n for n in files), sorted(files))
        rows = {r["model_kind"]: r for r in csv.DictReader(io.StringIO(files["pilot_metrics.csv"].decode()))}
        self.assertEqual(sorted(rows), ["modelangelo", "modelangelo_raw"])
        for row in rows.values():
            self.assertEqual(row["status"], "ok")
            self.assertGreater(float(row["tm_score"]), 0.99)
        info = json.loads(files["run_info.json"].decode())
        self.assertTrue(info["complete"])
        self.assertEqual(info["modelangelo_commit"], self.commit)
        self.assertEqual(info["modelangelo_version"], "ModelAngelo 1.0.18-stub")
        self.assertEqual(info["bundle"], "nucleotides")
        self.assertTrue(info["weights"]["md5_checked_by_modelangelo"])
        self.assertGreater(info["weights"]["files_bytes"]["hub/checkpoints/esm1b_t33_650M_UR50S.pt"], 1000000)
        self.assertEqual(info["state"]["entries"]["26DK"]["status"], "ok")
        self.assertEqual(info["state"]["entries"]["26DK"]["steps"]["modelangelo"]["rc"], 0)
        self.assertEqual(info["state"]["entries"]["26DK"]["sequences"], [{"job": "26DK_E1", "length": 197, "copies": 1}])
        # the command, the environment of the program, the progress lines
        calls = files["build_calls.txt"].decode()
        self.assertRegex(calls, r"build -v .*/emd_80558\.mrc -pf .*/26dk\.fasta -o .*/ma_out/26DK_modelangelo --device 0")
        self.assertIn("[ModelAngelo] running for", out)
        self.assertIn("last line: GNN model refinement", out)
        self.assertIn("ModelAngelo at commit %s" % self.commit, out)
        self.assertIn("version printed by the program: ModelAngelo 1.0.18-stub", out)
        fasta = files["inputs/26dk.fasta"].decode()
        self.assertTrue(fasta.startswith(">26DK_1|Chains A|"))

    def test_a_failing_modelangelo_is_reported_with_the_last_lines_and_the_zip_is_incomplete(self):
        r, zips, tmp = self.run_notebook(STUB_MA_FAIL="1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-2000:])
        self.assertIn("ENTRY 26DK STOPPED in ModelAngelo: exit code 1", out)
        self.assertIn("stand-in failure", out)
        self.assertIn("RESULT: NOT COMPLETE (26DK: modelangelo)", out)
        self.assertEqual(len(zips), 1)
        self.assertIn("_INCOMPLETE", zips[0])
        files = self.zip_files(zips[0])
        status = list(csv.DictReader(io.StringIO(files["entry_status.csv"].decode())))
        self.assertEqual((status[0]["status"], status[0]["failed_step"]), ("failed", "modelangelo"))
        self.assertIn("ma_out/26DK_modelangelo/model_angelo.log", files)

    def test_a_run_that_takes_too_long_is_stopped_and_counted_as_failed(self):
        r, zips, tmp = self.run_notebook(patch={"step6": [("TIMEOUT = 3 * 3600", "TIMEOUT = 3")]}, STUB_MA_STEPS="20", STUB_MA_SLOW="1")
        out = r.stdout + r.stderr
        self.assertIn("ENTRY 26DK STOPPED in ModelAngelo: timeout", out)
        self.assertIn("_INCOMPLETE", zips[0])
        status = list(csv.DictReader(io.StringIO(self.zip_files(zips[0])["entry_status.csv"].decode())))
        self.assertIn("timeout", status[0]["note"])

    def test_weights_that_fail_their_checksum_stop_the_run_with_the_reason(self):
        r, zips, tmp = self.run_notebook(STUB_NO_WEIGHTS="1")
        out = r.stdout + r.stderr
        self.assertIn("the ModelAngelo weights could not be downloaded or did not pass ModelAngelo's own MD5 check", out)
        self.assertIn("RESULT: STOPPED", out)
        self.assertNotIn("=== ModelAngelo:", out)
        self.assertIn("_INCOMPLETE", zips[0])

    def test_without_a_gpu_the_notebook_says_what_to_switch_on(self):
        r, zips, tmp = self.run_notebook(require_gpu=True)
        out = r.stdout + r.stderr
        import shutil as _sh
        if _sh.which("nvidia-smi"):
            self.skipTest("this machine has a GPU")
        self.assertIn("this session has no GPU", out)
        self.assertIn("GPU T4 x2", out)
        self.assertIn("RESULT: STOPPED", out)
        self.assertEqual(len(zips), 1)

    def test_a_second_run_in_the_same_folders_reuses_the_environment_and_the_weights_and_runs_again(self):
        r, zips, tmp = self.run_notebook()
        self.assertIn("RESULT: OK", r.stdout)
        script = os.path.join(tmp, "nb.py")
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1", STUB_MA_MODEL=self.cif, MPLBACKEND="module://matplotlib_inline.backend_inline")
        r2 = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
        out = r2.stdout + r2.stderr
        self.assertIn("RESULT: OK", out, out[-2000:])
        self.assertIn("the Python environment already exists", out)


if __name__ == "__main__":
    unittest.main()
