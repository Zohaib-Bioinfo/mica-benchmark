#!/usr/bin/env python3
"""Tests for make_main_notebook.py and the notebook it builds (the production runs of ModelAngelo, E3-CryoFold, EMProt and EModelX(+AF) on many entries, two GPUs).

Fast tests: the notebook builds, every cell compiles, the embedded files equal the repository files, the commands are the ones of the READMEs, the list of tasks and the
order of the entries, the committed notebook is current.

End-to-end tests run the whole notebook as a script against stand-ins (local git repositories that play the four programs with the command line of the real ones, a file
server for the maps, the deposited model and the weights, the real US-align) and check the schedule: both GPUs are used, the two memory-hungry methods never overlap, a run
that the end of the session cuts is "interrupted" and not a failure, a method that crashes twice within three minutes is left alone, a program that the system kills while the
other GPU runs is run again alone. They need the same three things as test_methods_notebook.py:

    MICA_TEST_ENV_PYTHON   a Python 3 with numpy, mrcfile and biopython
    MICA_TEST_AF3_DIR      the folder with the 26 AlphaFold Server files (private)
    USALIGN                the compiled US-align program

    USALIGN=... MICA_TEST_ENV_PYTHON=... MICA_TEST_AF3_DIR=... python -W error -m unittest test_main_notebook
"""
from __future__ import annotations

import csv
import datetime
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
import make_main_notebook as M  # noqa: E402
import nbkit  # noqa: E402
import test_cryozeta_tools as TZ  # noqa: E402  (the stand-in programs of CryoZeta and the stand-in ColabFold server)
import test_methods_notebook as TM  # noqa: E402
import test_pilot_notebook as T  # noqa: E402

COMMITTED = os.path.join(HERE, "gpu_batch1.ipynb")
ENV_PY, AF3_DIR, USALIGN = T.ENV_PY, T.AF3_DIR, T.USALIGN

PRELUDE = r'''
import atexit, os, shutil, signal, sys, time

NAME = [""]


def timeline(tag):
    with open(os.environ["STUB_TIMELINE"], "a") as f:
        f.write("%s %s pid=%d gpu=%s t=%.3f\n" % (tag, NAME[0], os.getpid(), os.environ.get("CUDA_VISIBLE_DEVICES"), time.time()))


def calls(name):
    NAME[0] = name
    with open(os.environ["STUB_CALLS"], "a") as f:
        f.write(name + " " + " ".join(sys.argv[1:]) + "\n")
    timeline("START")
    atexit.register(lambda: timeline("END"))


def maybe_fail(name):
    if name in os.environ.get("STUB_FAIL", "").split(","):
        print("Error: stand-in failure of " + name)
        sys.stdout.flush()
        sys.exit(1)
    marker = os.path.join(os.environ.get("STUB_MARKERS", "/nonexistent"), name)
    if name in os.environ.get("STUB_KILL_ONCE", "").split(",") and not os.path.exists(marker):
        open(marker, "w").close()
        time.sleep(1.5)                                    # the other GPU is busy for longer than this
        os.kill(os.getpid(), signal.SIGKILL)               # what the out-of-memory killer does
    if name in os.environ.get("STUB_KILL_ALWAYS", "").split(","):
        time.sleep(1.5)
        os.kill(os.getpid(), signal.SIGKILL)


def work(name):
    for i in range(int(os.environ.get("STUB_STEPS", "1"))):
        print("%s step %d ... 40%%|####      | 4/10" % (name, i + 1))
        sys.stdout.flush()
        time.sleep(float(os.environ.get("STUB_SLOW_" + name.upper(), "0")))


def need_gpu_env():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1"), os.environ.get("CUDA_VISIBLE_DEVICES")
    assert os.environ.get("TORCH_HOME"), "TORCH_HOME is not set"
    assert not os.environ.get("VIRTUAL_ENV") and not os.environ.get("PYTHONPATH")
    assert os.environ.get("MPLBACKEND") == "Agg", "MPLBACKEND=%r" % os.environ.get("MPLBACKEND")
'''


def with_prelude(files):
    return {k: (v.replace(TM.PRELUDE, PRELUDE) if TM.PRELUDE in v else v) for k, v in files.items()}


STUB_E3 = with_prelude(TM.STUB_E3)
STUB_EMPROT = with_prelude(TM.STUB_EMPROT)
STUB_EMODELX = with_prelude(TM.STUB_EMODELX)
STUB_MODELANGELO = {
    "model_angelo/__init__.py": "__version__ = '1.0.18-stub'\n",
    "model_angelo/apps/__init__.py": "", "model_angelo/utils/__init__.py": "", "model_angelo/apps/build.py": "", "model_angelo/utils/setup_weights.py": "",
    "model_angelo/model_definitions/nucleotides/config.json": "{}\n",
    "model_angelo/__main__.py": PRELUDE + r'''
import argparse
p = argparse.ArgumentParser()
p.add_argument("--version", action="store_true")
sub = p.add_subparsers(dest="cmd")
w = sub.add_parser("setup_weights"); w.add_argument("--bundle-name", default="nucleotides")
b = sub.add_parser("build"); b.add_argument("-v", "--volume-path"); b.add_argument("-pf", "--protein-fasta"); b.add_argument("-o", "--output-dir"); b.add_argument("--device", default="0")
a = p.parse_args()
if a.version:
    print("ModelAngelo 1.0.18-stub")
    sys.exit(0)
bundle = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", "model_angelo_v1.0", "nucleotides")
if a.cmd == "setup_weights":
    assert os.environ.get("MPLBACKEND") == "Agg"
    os.makedirs(bundle, exist_ok=True)
    open(os.path.join(bundle, "config.json"), "w").write("{}\n")
    open(os.path.join(bundle, "stub.torch"), "wb").write(b"\0" * 1500000)
    print("Checksum of downloaded file matches expected checksum")
    print("Successful!")
    sys.exit(0)
assert a.cmd == "build"
calls("modelangelo")
need_gpu_env()
assert os.path.exists(os.path.join(bundle, "config.json")), "the weights are not installed"
assert os.path.exists(a.volume_path) and a.volume_path.endswith(".mrc") and os.path.exists(a.protein_fasta)
assert a.device == "0"
maybe_fail("modelangelo")
work("modelangelo")
out = os.path.normpath(a.output_dir)
name = os.path.basename(out)
os.makedirs(out, exist_ok=True)
shutil.copy(os.environ["STUB_MODEL_CIF"], os.path.join(out, name + ".cif"))
shutil.copy(os.environ["STUB_MODEL_CIF"], os.path.join(out, name + "_raw.cif"))
open(os.path.join(out, "model_angelo.log"), "w").write("stand-in log\n")
''',
}


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mainbuild_")
        cls.path = os.path.join(cls.tmp, "nb.ipynb")
        cls.expected = M.build(cls.path)
        cls.nb = nbformat.read(cls.path, as_version=4)
        cls.text = "\n".join(c.source for c in cls.nb.cells)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cell(self, cid):
        return next(c for c in self.nb.cells if c["id"] == cid).source

    def test_cells_compile_and_have_no_placeholders(self):
        for c in self.nb.cells:
            if c.cell_type == "code" and not c.source.startswith("%%writefile"):
                compile(c.source, c["id"], "exec")
            self.assertEqual(re.findall(r"@[A-Z_0-9]+@", c.source), [], c["id"])

    def test_cell_order(self):
        ids = [c["id"] for c in self.nb.cells if c.cell_type == "code" and not c["id"].startswith("file-")]
        self.assertEqual(ids, ["settings", "step1", "step2-dirs", "step2-check", "robust", "tools", "step4", "modelangelo", "e3", "emprot", "emodelx", "run", "summary", "final"])

    def test_the_only_setting_is_the_first_code_cell(self):
        first = next(c for c in self.nb.cells if c.cell_type == "code")
        self.assertEqual(first["id"], "settings")
        self.assertRegex(first.source, r"(?m)^HOURS = 9\.0$")
        self.assertRegex(first.source, r"(?m)^GUARD_MINUTES = 60$")

    def test_embedded_files_equal_the_repository_files(self):
        got = {}
        for c in self.nb.cells:
            if c.source.startswith("%%writefile"):
                first, text = c.source.split("\n", 1)
                got[first.split(None, 1)[1].strip()] = text
        self.assertEqual(sorted(got), sorted(t for _s, t in M.FILES))
        for src, target in M.FILES:
            with open(os.path.join(ROOT, src), encoding="utf-8") as f:
                text = f.read()
            self.assertEqual(got[target], text if text.endswith("\n") else text + "\n", target)
            self.assertEqual(self.expected[target], hashlib.sha256(got[target].encode()).hexdigest())
        for key in M.ORDER:
            self.assertIn(M.SPECS[key]["req"], got)

    def test_the_committed_notebook_is_current(self):
        if not os.path.exists(COMMITTED):
            self.skipTest("gpu_batch1.ipynb is not built yet")
        fresh = os.path.join(self.tmp, "fresh.ipynb")
        M.build(fresh)
        a, b = nbformat.read(fresh, as_version=4), nbformat.read(COMMITTED, as_version=4)
        self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_main_notebook.py --batch batch1 again")

    def test_the_four_methods_and_their_pins(self):
        self.assertEqual(M.ORDER, ["modelangelo", "e3", "emprot", "emodelx"])
        self.assertEqual(sorted(M.SPECS), sorted(M.ORDER))
        for key, spec in M.SPECS.items():
            self.assertRegex(spec["commit"], r"^[0-9a-f]{40}$", key)
            self.assertIn(spec["commit"], self.text)
            self.assertEqual(spec["timeout"], M.RUN_LIMIT_S)
        self.assertEqual(M.SPECS["modelangelo"]["commit"], "994945bdfa6e5368e0d62349a47792f4864eebc3")
        self.assertEqual(M.SPECS["e3"]["commit"], TM.mm.SPECS["e3"]["commit"])                    # the commits of the pilots
        self.assertEqual(M.SPECS["emprot"]["commit"], TM.mm.SPECS["emprot"]["commit"])
        self.assertEqual(M.SPECS["emodelx"]["commit"], TM.mm.SPECS["emodelx"]["commit"])
        self.assertEqual({k for k, s in M.SPECS.items() if s["uses_af3"]}, {"emprot", "emodelx"})   # decision 6 of the spec
        self.assertEqual(M.RUN_LIMIT_S, 10800)

    def test_the_commands_are_the_ones_of_the_readmes_with_the_gpu_named(self):
        ma, e3, emprot, emx, tools = self.cell("modelangelo"), self.cell("e3"), self.cell("emprot"), self.cell("emodelx"), self.cell("tools")
        self.assertIn('[env_py, "-m", "model_angelo", "build", "-v", ctx["map"], "-pf", ctx["fasta"], "-o", out, "--device", "0"]', ma)
        self.assertIn('"setup_weights", "--bundle-name", spec["bundle"]', ma)
        self.assertIn('name = "%s_modelangelo" % entry', ma)
        self.assertIn('out = os.path.join(RUN_ROOT, "modelangelo", entry, name)', ma)
        self.assertIn('(("prealign", "pre_align"),)', e3)                                         # only the default protocol
        self.assertNotIn('"denovo"', e3)
        self.assertIn('"--protocol", protocol, "--t", "0.1"', e3)
        self.assertIn('"-m", "emprot", "build", "--map", ctx["map"], "--seq", fasta_path, "--chain"] + chains + ["--device", "0", "--output", out]', emprot)
        self.assertIn('"run.py", "--protocol=temp_flex", "--EM_map", ctx["map"], "--fasta", fasta, "--template_dir", tdir, "--output_dir", out, "--run_pulchra"', emx)
        self.assertIn('env = dict(r["env"], CUDA_VISIBLE_DEVICES=str(gpu))', tools)                 # the GPU of the worker
        self.assertIn('MPLBACKEND": "Agg"', tools)

    def test_the_cells_of_the_methods_only_define_functions(self):
        for cid in ("modelangelo", "e3", "emprot", "emodelx"):
            self.assertNotIn("execute_method", self.cell(cid), cid)
            self.assertNotIn('if STATE["ok"]', self.cell(cid), cid)

    def test_a_gpu_is_required_and_no_secret_is_asked(self):
        self.assertIn("REQUIRE_GPU = True", self.cell("step1"))
        code = "\n".join(c.source for c in self.nb.cells if c.cell_type == "code" and not c.source.startswith("%%writefile"))
        self.assertNotIn('secret("PHENIX', code)                                                   # nbtools (a file cell) knows the name, to keep it out of zips; no cell asks for it
        self.assertNotIn("getpass.getpass(", "\n".join(self.cell(c) for c in ("settings", "tools", "run", "summary")))

    def test_the_predictions_themselves_are_not_kept_in_the_zip(self):
        tools = self.cell("tools")
        self.assertIn('PRIVATE_KINDS = ("emprot_fit",)', tools)
        self.assertIn("if kind in PRIVATE_KINDS:", tools)

    def test_the_two_memory_hungry_methods_are_kept_apart(self):
        tools = self.cell("tools")
        self.assertIn("EXPLOSIVE = ('e3', 'emodelx')", tools)
        self.assertIn('if key in EXPLOSIVE and any(b["key"] in EXPLOSIVE for g, b in BUSY.items() if g != gpu):', tools)

    def test_the_limits(self):
        tools = self.cell("tools")
        self.assertIn("RUN_LIMIT = 10800", tools)
        self.assertIn("min(HOURS_SETTING, 11.0)", tools)
        self.assertIn("HARD_STOP = T0 + SESSION_S - RESERVE_S", tools)


class GpuHandlingTests(unittest.TestCase):
    """The parts of the notebook that need a GPU, run against a fake nvidia-smi."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="gpuh_")
        nb = os.path.join(cls.tmp, "nb.ipynb")
        M.build(nb)
        cls.nb = nbformat.read(nb, as_version=4)
        cls.bin = os.path.join(cls.tmp, "bin")
        os.makedirs(cls.bin)
        cls.args = os.path.join(cls.tmp, "smi_args.txt")
        with open(os.path.join(cls.bin, "nvidia-smi"), "w") as f:
            f.write("#!/bin/sh\necho \"$@\" >> %s\ncase \"$*\" in\n  *--query-gpu=index*) printf '0\\n1\\n' ;;\n  *-i\\ 1*) echo '1234, 56' ;;\n  *) echo '99, 1' ;;\nesac\n" % cls.args)
        os.chmod(os.path.join(cls.bin, "nvidia-smi"), 0o755)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def source(self, cid):
        return next(c for c in self.nb.cells if c["id"] == cid).source

    def segment(self, cid, kind, name):
        import ast
        src = self.source(cid)
        for node in ast.parse(src).body:
            if isinstance(node, kind) and node.name == name:
                return ast.get_source_segment(src, node)
        raise KeyError(name)

    def test_the_monitor_of_a_lane_reads_the_gpu_of_that_lane(self):
        ns = {"os": os, "shutil": shutil, "datetime": __import__("datetime"), "threading": __import__("threading"), "subprocess": subprocess}
        import ast
        exec(self.segment("step1", ast.FunctionDef, "sh"), ns)
        exec(self.segment("step1", ast.ClassDef, "Monitor"), ns)
        exec(self.segment("tools", ast.ClassDef, "GpuMonitor"), ns)
        old = os.environ["PATH"]
        os.environ["PATH"] = self.bin + os.pathsep + old
        try:
            mon = ns["GpuMonitor"](os.path.join(self.tmp, "gpu1.csv"), 1)
            mon.sample()
            mon0 = ns["GpuMonitor"](os.path.join(self.tmp, "gpu0.csv"), 0)
            mon0.sample()
        finally:
            os.environ["PATH"] = old
        self.assertEqual((mon.peak_gpu_mb, mon.peak_util), (1234, 56))
        self.assertEqual((mon0.peak_gpu_mb, mon0.peak_util), (99, 1))
        calls = read_lines(self.args)
        self.assertTrue(any("-i 1 --query-gpu=memory.used,utilization.gpu" in c for c in calls), calls)
        self.assertTrue(any("-i 0 --query-gpu=memory.used,utilization.gpu" in c for c in calls), calls)

    def fake_smi(self):
        import contextlib

        @contextlib.contextmanager
        def ctx():
            old = os.environ["PATH"]
            os.environ["PATH"] = self.bin + os.pathsep + old
            try:
                yield
            finally:
                os.environ["PATH"] = old
        return ctx()

    def namespace(self, **extra):
        import ast
        ns = {"os": os, "shutil": shutil, "REQUIRE_GPU": True, "ENV_PY_GIVEN": None, "RUN_METHODS": ["modelangelo", "e3"], "GPU_TEST": "print('gpu test')", "log": lambda *a: None}
        exec(self.segment("tools", ast.FunctionDef, "detect_gpus"), ns)
        exec(self.segment("tools", ast.FunctionDef, "usable_lanes"), ns)
        ns.update(extra)
        return ns

    def test_two_gpus_are_found_and_tested_with_the_environment_of_the_first_method(self):
        calls = []

        def run(cmd, env=None, quiet=False):
            calls.append((cmd, env["CUDA_VISIBLE_DEVICES"]))
            return 0, ["ok"], False

        def sh(cmd, timeout=60):
            return "0\n1\n"

        ns = self.namespace(READY={"modelangelo": ("/env/ma/bin/python", "/src/ma")}, run=run, sh=sh, env_vars=lambda py, extra=None: dict(extra or {}))
        with self.fake_smi():
            self.assertEqual(ns["detect_gpus"](), [0, 1])
            self.assertEqual(ns["usable_lanes"](), [0, 1])
        self.assertEqual(calls, [(["/env/ma/bin/python", "-c", "print('gpu test')"], "0"), (["/env/ma/bin/python", "-c", "print('gpu test')"], "1")])

    def test_a_gpu_that_pytorch_cannot_use_is_not_used(self):
        def run(cmd, env=None, quiet=False):
            return (3, ["cuda available: False"], False) if env["CUDA_VISIBLE_DEVICES"] == "1" else (0, ["ok"], False)

        ns = self.namespace(READY={"modelangelo": ("/env/ma/bin/python", "/src/ma")}, run=run, sh=lambda cmd, timeout=60: "0\n1\n", env_vars=lambda py, extra=None: dict(extra or {}))
        with self.fake_smi():
            self.assertEqual(ns["usable_lanes"](), [0])

    def test_one_gpu_one_lane_and_no_gpu_test_without_the_real_requirement(self):
        ns = self.namespace(READY={}, run=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no test expected")), sh=lambda cmd, timeout=60: "0\n", env_vars=None)
        with self.fake_smi():
            self.assertEqual(ns["usable_lanes"](), [0])
        ns = self.namespace(READY={}, REQUIRE_GPU=False, run=None, sh=lambda cmd, timeout=60: "0\n1\n", env_vars=None)
        with self.fake_smi():
            self.assertEqual(ns["usable_lanes"](), [0, 1])               # the tests of the stand-in programs trust the GPUs that nvidia-smi lists


class OrderTests(unittest.TestCase):
    def rows(self):
        return M.read_entries()

    def test_the_order_interleaves_the_two_strata_in_the_seeded_order(self):
        rows = self.rows()
        order = M.seeded_order(rows)
        self.assertEqual(len(order), 20)
        self.assertEqual(sorted(order), sorted(r["pdb_id"] for r in rows))
        by = {r["pdb_id"]: r for r in rows}
        strata = [by[e]["stratum"] for e in order]
        self.assertEqual(strata[:4], ["le3.0", "gt3.0_le4.0", "le3.0", "gt3.0_le4.0"])
        for stratum in ("le3.0", "gt3.0_le4.0"):
            ranks = [int(by[e]["seed_rank"]) for e in order if by[e]["stratum"] == stratum]
            self.assertEqual(ranks, sorted(ranks), stratum)
        # the first 12 (the floor of decision 12) hold six entries of each stratum: dropping from the end keeps the two strata balanced
        self.assertEqual(sum(1 for s in strata[:12] if s == "le3.0"), 6)
        self.assertEqual(order[:4], ["9T5K", "9WQ6", "22FX", "9V4D"])

    def test_the_tasks_go_entry_by_entry(self):
        tasks = M.all_tasks(["A", "B"])
        self.assertEqual(tasks, [["A", "modelangelo"], ["A", "e3"], ["A", "emprot"], ["A", "emodelx"], ["B", "modelangelo"], ["B", "e3"], ["B", "emprot"], ["B", "emodelx"]])

    def test_pending_tasks_drop_the_finished_ones_and_keep_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = os.path.join(d, "a.csv"), os.path.join(d, "b.csv")
            for path, rows in ((p1, [("A", "e3", "ok"), ("A", "emprot", "failed"), ("A", "emodelx", "interrupted")]), (p2, [("B", "e3", "not run"), ("B", "modelangelo", "ok"), ("A", "modelangelo", "killed_corun")])):
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["method", "entry", "status"])
                    for e, m, s in rows:
                        w.writerow([m, e, s])
            left = M.pending_tasks(M.all_tasks(["A", "B"]), [p1, p2])
            self.assertEqual(left, [["A", "modelangelo"], ["A", "emodelx"], ["B", "e3"], ["B", "emprot"], ["B", "emodelx"]])

    def test_a_finished_task_can_be_forced_again(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "a.csv")
            with open(p1, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "entry", "status"])
                w.writerows([["e3", "A", "failed"], ["emprot", "A", "ok"]])
            left = M.pending_tasks(M.all_tasks(["A"]), [p1], redo=["A:e3"])
            self.assertEqual(left, [["A", "modelangelo"], ["A", "e3"], ["A", "emodelx"]])

    def test_unknown_tasks_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                M.build(os.path.join(d, "x.ipynb"), tasks=[["NOPE", "e3"]])
            with self.assertRaises(ValueError):
                M.build(os.path.join(d, "x.ipynb"), tasks=[["9T5K", "cryoatom"]])


@unittest.skipUnless(ENV_PY and AF3_DIR and USALIGN and any("fold_26dk_e1_model_0.cif" in f for _r, _d, f in os.walk(AF3_DIR or "")),
                     "needs MICA_TEST_ENV_PYTHON, MICA_TEST_AF3_DIR (with the 26DK file) and USALIGN")
class EndToEndTests(unittest.TestCase):
    ENTRIES = ["9T5K", "9WQ6", "22FX"]

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="mainrun_")
        cls.cif = [os.path.join(r, f) for r, _d, fs in os.walk(AF3_DIR) for f in fs if f == "fold_26dk_e1_model_0.cif"][0]
        with open(cls.cif, "rb") as f:
            cif_bytes = f.read()
        cls.pdb = os.path.join(cls.root, "model.pdb")
        subprocess.run([ENV_PY, "-c", "import sys; from Bio.PDB import MMCIFParser, PDBIO; s = MMCIFParser(QUIET=True).get_structure('m', sys.argv[1]); io = PDBIO(); "
                        "io.set_structure(s); io.save(sys.argv[2])", cls.cif, cls.pdb], check=True)
        mapfile = os.path.join(cls.root, "stub.map")
        subprocess.run([ENV_PY, "-c", "import mrcfile, numpy; m = mrcfile.new(%r, overwrite=True); m.set_data(numpy.random.rand(32, 32, 32).astype('float32')); "
                        "m.voxel_size = 1.0; m.close()" % mapfile], check=True)
        with open(mapfile, "rb") as f:
            map_gz = gzip.compress(f.read())
        blob = lambda n, seed: bytes((i * seed) % 251 for i in range(256)) * (n // 256)

        def zipped(files):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for name, data in files.items():
                    z.writestr(name, data)
            return buf.getvalue()

        def tgz(files):
            import tarfile
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as t:
                for name, data in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    t.addfile(info, io.BytesIO(data))
            return buf.getvalue()

        cls.srv, cls.url = T.serve(T.Files)
        cls.routes = {"/map/stub.map.gz": map_gz, "/cif/stub.cif": cif_bytes,
                      "/e3/models.zip": zipped({"models/model_weight.pth": blob(1500000, 3), "models/structure_model.pth": blob(1500000, 5)}),
                      "/emprot/weights.tgz": tgz({"weights/stub.pth": blob(1500000, 7)}),
                      "/emodelx/models.zip": zipped({"models/best_BB_model.ckpt": blob(1500000, 29), "models/best_CA_model.ckpt": blob(1500000, 31), "models/best_AA_model.ckpt": blob(1500000, 37)})}
        T.Files.routes = dict(cls.routes)
        cls.entries = os.path.join(cls.root, "entries.csv")
        with open(os.path.join(ROOT, "data", "entries_2026.csv"), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows, fields = list(reader), reader.fieldnames
        for r in rows:
            r["map_url"], r["cif_url"] = cls.url + "/map/stub.map.gz", cls.url + "/cif/stub.cif"
        with open(cls.entries, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        cls.repos, cls.commits = {}, {}
        for name, files in (("modelangelo", STUB_MODELANGELO), ("e3", STUB_E3), ("emprot", STUB_EMPROT), ("emodelx", STUB_EMODELX)):
            repo = os.path.join(cls.root, name + "_stub")
            for rel, text in files.items():
                path = os.path.join(repo, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(text)
            if name == "emprot":
                os.makedirs(os.path.join(repo, "emprot", "bin"), exist_ok=True)
                with open(os.path.join(repo, "emprot", "bin", "tool"), "wb") as f:
                    f.write(TM.ELF)
            T.sh("git init -q -b main . && git config core.fileMode true && git add -A && git -c user.name=t -c user.email=t@t commit -q -m stub", cwd=repo)
            T.sh("chmod 644 emprot/bin/tool 2>/dev/null; true", cwd=repo)
            cls.repos[name], cls.commits[name] = repo, T.sh("git rev-parse HEAD", cwd=repo)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def stub_specs(self):
        s = M.make_specs()
        for key in s:
            s[key].update(repo="file://" + self.repos[key], commit=self.commits[key])
        s["e3"].update(weights_url=self.url + "/e3/models.zip", weights_min_bytes=1000)
        s["emprot"].update(weights_url=self.url + "/emprot/weights.tgz", weights_min_bytes=1000)
        s["emodelx"].update(weights_url=self.url + "/emodelx/models.zip", weights_min_bytes=1000, gdrive_id=None)
        return s

    def run_notebook(self, entries=None, tasks=None, hours=9.0, gpus="2", routes=None, patch=None, run_methods=None, **extra_env):
        T.Files.routes = dict(self.routes, **(routes or {}))
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        nb_path = os.path.join(tmp, "nb.ipynb")
        M.build(nb_path, batch="test", entries=entries or self.ENTRIES, tasks=tasks, hours=hours, entries_path=self.entries, env_python=ENV_PY, usalign_bin=USALIGN, net_hosts=(),
                require_gpu=False, specs=self.stub_specs(), run_methods=run_methods)
        nb = nbformat.read(nb_path, as_version=4)
        for c in nb.cells:
            for old, new in (patch or {}).get(c["id"], []):
                self.assertIn(old, c.source)
                c.source = c.source.replace(old, new)
        script = os.path.join(tmp, "nb.py")
        T.to_script(nb, script)
        os.makedirs(os.path.join(tmp, "markers"))
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1", MICA_RETRY_WAIT="0.1", MICA_AF3_DIR=AF3_DIR,
                   MICA_TEST_GPUS=gpus, MICA_POLL="0.2", MICA_RESERVE_S="0", MICA_MIN_START_S="2", STUB_CALLS=os.path.join(tmp, "calls.txt"), STUB_TIMELINE=os.path.join(tmp, "timeline.txt"),
                   STUB_MARKERS=os.path.join(tmp, "markers"), STUB_MODEL_CIF=self.cif, STUB_MODEL_PDB=self.pdb, STUB_RNAFM_BYTES="1",
                   MPLBACKEND="module://matplotlib_inline.backend_inline")
        for k in [k for k in env if k.startswith("STUB_") and k not in ("STUB_CALLS", "STUB_TIMELINE", "STUB_MARKERS", "STUB_MODEL_CIF", "STUB_MODEL_PDB", "STUB_RNAFM_BYTES")]:
            env.pop(k)
        env.update(extra_env)
        r = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=900)
        work = os.path.join(tmp, "work")
        zips = sorted(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".zip")) if os.path.isdir(work) else []
        return r, zips, tmp

    @staticmethod
    def zip_files(path):
        with zipfile.ZipFile(path) as z:
            return {n.split("/", 1)[1]: z.read(n) for n in z.namelist()}

    @staticmethod
    def csv_rows(files, name):
        return list(csv.DictReader(io.StringIO(files[name].decode())))

    @staticmethod
    def timeline(tmp):
        """[(method, pid, gpu, start, end)] of the stand-in programs."""
        starts, ends = {}, {}
        for line in read_lines(os.path.join(tmp, "timeline.txt")):
            tag, name, pid, gpu, t = line.split()
            (starts if tag == "START" else ends)[(name, pid)] = (gpu.split("=")[1], float(t.split("=")[1]))
        return sorted((n, p, starts[(n, p)][0], starts[(n, p)][1], ends.get((n, p), (None, float("inf")))[1]) for (n, p) in starts)

    def test_a_complete_run_with_two_gpus(self):
        r, zips, tmp = self.run_notebook(STUB_STEPS="2", STUB_SLOW_MODELANGELO="0.6", STUB_SLOW_EMPROT="0.6", STUB_SLOW_E3="0.4", STUB_SLOW_EMODELX="0.4")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out, out[-3000:])
        self.assertEqual(len(zips), 1, zips)
        self.assertNotIn("_INCOMPLETE", zips[0])
        self.assertTrue(os.path.basename(zips[0]).startswith("gpu_test_result_"), zips)
        files = self.zip_files(zips[0])
        for entry in self.ENTRIES:
            for needed in ("models/modelangelo/%s/%s_modelangelo.cif" % (entry, entry), "models/modelangelo/%s/%s_modelangelo_raw.cif" % (entry, entry),
                           "models/e3/%s/%s_e3_prealign.pdb" % (entry, entry), "models/emprot/%s/output.cif" % entry, "models/emprot/%s/output_denovo.cif" % entry,
                           "models/emodelx/%s/%s_EModelX(+AF)_all_atom_model.pdb" % (entry, entry.lower()), "models/emodelx/%s/%s_EModelX(+AF)_ca_model.pdb" % (entry, entry.lower()),
                           "logs/%s_modelangelo_build.log" % entry, "logs/%s_e3_prealign.log" % entry, "logs/%s_emprot_build.log" % entry, "logs/%s_emodelx_temp_flex.log" % entry,
                           "gpu/%s_modelangelo_build.csv" % entry, "scoring/%s_modelangelo_usalign.txt" % entry, "scoring/%s_e3_prealign_usalign.txt" % entry,
                           "method_files/modelangelo/%s/build/model_angelo.log" % entry):
                self.assertIn(needed, files, (needed, sorted(files)[:60]))
            self.assertNotIn("models/emprot/%s/output_fit.cif" % entry, files)
        for needed in ("run.log", "state.json", "run_info.json", "metrics_usalign.csv", "method_status.csv", "pending.csv", "machine.txt", "cpu_facts.txt", "modelangelo_config.json"):
            self.assertIn(needed, files)
        self.assertFalse(any(n.endswith((".map", ".mrc", ".ccp4")) or "AF3_results" in n or "ranked_0" in n or n.endswith((".pth", ".ckpt", ".torch", ".tgz")) for n in files), sorted(files))
        status = self.csv_rows(files, "method_status.csv")
        self.assertEqual(sorted((r["entry"], r["method"]) for r in status), sorted((e, m) for e in self.ENTRIES for m in M.ORDER))
        self.assertTrue(all(r["status"] == "ok" for r in status), status)
        self.assertEqual(self.csv_rows(files, "pending.csv"), [])
        kinds = {}
        for r in self.csv_rows(files, "metrics_usalign.csv"):
            kinds.setdefault(r["entry"], []).append(r["model_kind"])
            self.assertEqual(r["status"], "ok")
            self.assertGreater(float(r["tm_score"]), 0.99)
        for entry in self.ENTRIES:
            self.assertEqual(sorted(kinds[entry]), ["e3_prealign", "emodelx_af_all_atom", "emodelx_af_ca", "emprot", "emprot_denovo", "emprot_fit", "modelangelo", "modelangelo_raw"])
        info = json.loads(files["run_info.json"].decode())
        self.assertTrue(info["complete"])
        self.assertEqual(info["lanes"], [0, 1])
        self.assertEqual(info["run_limit_s"], 10800)
        self.assertEqual(len(info["tasks"]), 12)
        for key in M.ORDER:
            self.assertEqual(info["methods"][key]["status"], "ready")
            self.assertEqual(info["methods"][key]["commit"], self.stub_specs()[key]["commit"])
        self.assertIn("1.0.18-stub", info["methods"]["modelangelo"]["version"])
        # both GPUs worked, and every task has its GPU, times and the overlap flag
        gpus = {s["gpu"] for s in status}
        self.assertEqual(gpus, {"0", "1"})
        self.assertTrue(all(s["started_utc"] and s["other_gpu_busy"] in ("True", "False") for s in status))
        tl = self.timeline(tmp)
        self.assertEqual({g for _n, _p, g, _a, _b in tl}, {"0", "1"})
        self.assertTrue(any(s["other_gpu_busy"] == "True" for s in status), "with two workers some runs must overlap")
        # the two methods that fill the memory never ran at the same time
        hungry = [(a, b) for n, _p, _g, a, b in tl if n in M.EXPLOSIVE]
        for i, (a1, b1) in enumerate(hungry):
            for a2, b2 in hungry[i + 1:]:
                self.assertFalse(a1 < b2 and a2 < b1, "E3-CryoFold and EModelX ran at the same time: %s %s" % ((a1, b1), (a2, b2)))
        # no program ran on a GPU that another program of the same moment used
        for n1, p1, g1, a1, b1 in tl:
            for n2, p2, g2, a2, b2 in tl:
                if p1 < p2 and g1 == g2:
                    self.assertFalse(a1 < b2 and a2 < b1, "two programs on GPU %s at the same time" % g1)
        # the commands
        calls = read_lines(os.path.join(tmp, "calls.txt"))
        by = {}
        for c in calls:
            by.setdefault(c.split(" ", 1)[0], []).append(c)
        self.assertEqual(sorted(by), ["e3", "emodelx", "emprot", "modelangelo"])
        for k in by:
            self.assertEqual(len(by[k]), 3, k)
        self.assertRegex(by["modelangelo"][0], r"^modelangelo build -v .*/input/9T5K/emd_55586\.mrc -pf .*/9t5k\.fasta -o .*/runs/modelangelo/9T5K/9T5K_modelangelo --device 0$")
        self.assertTrue(all("--protocol pre_align" in c for c in by["e3"]))
        self.assertFalse(any("denovo" in c for c in by["e3"]))
        # the order of the work: the tasks are started entry by entry
        firsts = [min(s["started_utc"] for s in status if s["entry"] == e) for e in self.ENTRIES]
        self.assertEqual(firsts, sorted(firsts))
        self.assertIn("GPUs used: [0, 1]", out)
        self.assertRegex(out, r"\[GPU[01] ModelAngelo 9T5K\] running for")
        # the weights are where the programs read them
        big = os.path.join(tmp, "big")
        self.assertTrue(os.path.exists(os.path.join(big, "torch_home", "hub", "checkpoints", "model_angelo_v1.0", "nucleotides", "config.json")))
        self.assertTrue(os.path.exists(os.path.join(big, "src_e3", "models", "model_weight.pth")))

    def test_the_end_of_the_budget_stops_new_runs_and_leaves_the_rest_pending(self):
        r, zips, tmp = self.run_notebook(hours=0.0055, STUB_SLOW_MODELANGELO="4", STUB_SLOW_E3="4", STUB_SLOW_EMPROT="4", STUB_SLOW_EMODELX="4")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = self.zip_files(zips[0])
        status = self.csv_rows(files, "method_status.csv")
        pending = self.csv_rows(files, "pending.csv")
        self.assertEqual(len(status), 12, status)
        done = [s for s in status if s["status"] == "ok"]
        self.assertGreaterEqual(len(done), 2, out[-2500:])
        self.assertGreaterEqual(len(pending), 3, "something must be left for the next session")
        self.assertEqual(len(done) + len(pending), 12)
        self.assertFalse(any(s["status"] == "failed" for s in status), "a task that did not run is not a failure: %s" % status)
        self.assertTrue(all(p["status"] in ("not run", "interrupted") for p in pending), pending)
        self.assertTrue(any("not enough time left" in p["note"] or "interrupted" in p["status"] for p in pending))
        self.assertIn("RESULT: OK", out)                                        # a session that ends on time is a normal session
        # what was done is the beginning of the list: entry by entry
        order = [(e, m) for e in self.ENTRIES for m in M.ORDER]
        done_set = {(s["entry"], s["method"]) for s in done}
        first_missing = next(i for i, t in enumerate(order) if t not in done_set)
        self.assertLessEqual(sum(1 for t in order[first_missing + 4:] if t in done_set), 0, "the entries are not run in order: %s" % sorted(done_set))
        info = json.loads(files["run_info.json"].decode())
        self.assertEqual(len(info["state"]["pending"]), len(pending))

    def test_a_run_that_the_end_of_the_session_cuts_is_interrupted_not_failed(self):
        r, zips, tmp = self.run_notebook(tasks=[["9T5K", "emprot"]], hours=0.0075, STUB_SLOW_EMPROT="60")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = self.zip_files(zips[0])
        status = self.csv_rows(files, "method_status.csv")
        self.assertEqual([(s["entry"], s["method"], s["status"]) for s in status], [("9T5K", "emprot", "interrupted")], out[-2000:])
        self.assertEqual([(p["entry"], p["method"], p["status"]) for p in self.csv_rows(files, "pending.csv")], [("9T5K", "emprot", "interrupted")])
        self.assertIn("INTERRUPTED by the end of the session (not a failure)", out)
        self.assertIn("logs/9T5K_emprot_build.log", files)
        self.assertEqual(self.csv_rows(files, "metrics_usalign.csv"), [])
        self.assertLess(float(status[0]["seconds"]), 40)

    def test_a_method_that_crashes_twice_within_the_guard_time_is_left_alone(self):
        r, zips, tmp = self.run_notebook(STUB_FAIL="e3")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = self.zip_files(zips[0])
        status = {(s["entry"], s["method"]): s for s in self.csv_rows(files, "method_status.csv")}
        e3 = [status[(e, "e3")] for e in self.ENTRIES]
        self.assertEqual([s["status"] for s in e3], ["failed", "failed", "not run"], e3)
        self.assertIn("exit code 1", e3[0]["note"])
        self.assertEqual(e3[2]["failed_step"], "guard")
        self.assertIn("not a result", e3[2]["note"])
        self.assertTrue(all(s["status"] == "ok" for (e, m), s in status.items() if m != "e3"))
        pending = self.csv_rows(files, "pending.csv")
        self.assertEqual([(p["entry"], p["method"]) for p in pending], [(self.ENTRIES[2], "e3")])
        self.assertIn("Error: stand-in failure of e3", out)                       # the last lines of the program are in the log
        self.assertIn("RESULT: OK", out)

    def test_the_guard_can_be_switched_off(self):
        r, zips, tmp = self.run_notebook(STUB_FAIL="e3", patch={"settings": [("GUARD_MINUTES = 60", "GUARD_MINUTES = 0")]})
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        rows = [s for s in self.csv_rows(self.zip_files(zips[0]), "method_status.csv") if s["method"] == "e3"]
        self.assertEqual([s["status"] for s in rows], ["failed"] * 3, rows)

    def test_a_failure_that_is_slow_is_a_result_and_does_not_stop_the_method(self):
        # a crash after more than the quick-crash time is a result of the method; the guard does not apply
        r, zips, tmp = self.run_notebook(STUB_FAIL="emodelx", MICA_QUICK_CRASH="0.0001")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        status = [s for s in self.csv_rows(self.zip_files(zips[0]), "method_status.csv") if s["method"] == "emodelx"]
        self.assertEqual([s["status"] for s in status], ["failed"] * 3, status)

    def test_a_method_whose_weights_cannot_be_downloaded_is_not_run_and_the_others_go_on(self):
        r, zips, tmp = self.run_notebook(routes={"/e3/models.zip": None})
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("E3-CryoFold STOPPED in its setup: the weights could not be downloaded", out)
        files = self.zip_files(zips[0])
        status = {(s["entry"], s["method"]): s for s in self.csv_rows(files, "method_status.csv")}
        self.assertEqual(len(status), 12)
        for e in self.ENTRIES:
            self.assertEqual(status[(e, "e3")]["status"], "not run")
            self.assertIn("the setup failed", status[(e, "e3")]["note"])
            for m in ("modelangelo", "emprot", "emodelx"):
                self.assertEqual(status[(e, m)]["status"], "ok", (e, m))
        self.assertEqual(len(self.csv_rows(files, "pending.csv")), 3)
        info = json.loads(files["run_info.json"].decode())
        self.assertEqual(info["methods"]["e3"]["status"], "setup failed")

    def test_without_the_alphafold_dataset_the_session_stops_at_once_and_spends_nothing(self):
        empty = tempfile.mkdtemp(prefix="noaf3_", dir=self.root)
        r, zips, tmp = self.run_notebook(entries=["9T5K"], MICA_AF3_DIR=empty)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("STOPPED: the private AlphaFold dataset is missing or is not the checked one", out)
        self.assertIn("RESULT: STOPPED", out)
        self.assertNotIn("setup of", out)                                       # not one environment was built
        self.assertFalse(os.path.exists(os.path.join(tmp, "calls.txt")))
        self.assertEqual(len(zips), 1)
        self.assertIn("_INCOMPLETE", zips[0])

    def test_a_wrong_alphafold_file_is_found_before_any_run(self):
        bad = tempfile.mkdtemp(prefix="badaf3_", dir=self.root)
        shutil.copytree(AF3_DIR, os.path.join(bad, "copy"))
        victim = [os.path.join(r, f) for r, _d, fs in os.walk(bad) for f in fs if f == "fold_9t5k_e1_model_0.cif"][0]
        with open(victim, "a") as f:
            f.write("# changed\n")
        r, zips, tmp = self.run_notebook(entries=["9T5K"], MICA_AF3_DIR=bad)
        out = r.stdout + r.stderr
        self.assertIn("is not the one that was checked", out)
        self.assertIn("RESULT: STOPPED", out)
        self.assertFalse(os.path.exists(os.path.join(tmp, "calls.txt")))

    def test_the_methods_that_do_not_use_the_predictions_run_without_the_dataset(self):
        empty = tempfile.mkdtemp(prefix="noaf3_", dir=self.root)
        r, zips, tmp = self.run_notebook(tasks=[["9T5K", "e3"], ["9T5K", "modelangelo"]], MICA_AF3_DIR=empty)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out)
        status = {s["method"]: s["status"] for s in self.csv_rows(self.zip_files(zips[0]), "method_status.csv")}
        self.assertEqual(status, {"e3": "ok", "modelangelo": "ok"})

    def test_a_program_killed_while_the_other_gpu_runs_is_run_again_alone(self):
        r, zips, tmp = self.run_notebook(entries=["9T5K"], STUB_KILL_ONCE="e3", STUB_SLOW_MODELANGELO="6", STUB_SLOW_EMPROT="3")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("KILLED while the other GPU ran; it is run again alone at the end", out)
        self.assertIn("run again alone: E3-CryoFold on 9T5K", out)
        files = self.zip_files(zips[0])
        status = {s["method"]: s for s in self.csv_rows(files, "method_status.csv")}
        self.assertEqual(status["e3"]["status"], "ok", status["e3"])
        self.assertEqual(status["e3"]["attempt"], "2")
        self.assertEqual(status["e3"]["other_gpu_busy"], "False")
        self.assertIn("models/e3/9T5K/9T5K_e3_prealign.pdb", files)
        info = json.loads(files["run_info.json"].decode())
        first = info["state"]["methods"]["e3"]["entries"]["9T5K"]["first_attempt"]
        self.assertEqual(first["status"], "killed_corun")
        self.assertEqual(self.csv_rows(files, "pending.csv"), [])

    def test_a_program_that_is_killed_also_when_alone_is_a_failure_of_the_method(self):
        r, zips, tmp = self.run_notebook(entries=["9T5K"], STUB_KILL_ALWAYS="e3", STUB_SLOW_MODELANGELO="6", STUB_SLOW_EMPROT="3")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        status = {s["method"]: s for s in self.csv_rows(self.zip_files(zips[0]), "method_status.csv")}
        self.assertEqual(status["e3"]["status"], "failed", status["e3"])
        self.assertEqual(status["e3"]["attempt"], "2")
        self.assertIn("exit code -9", status["e3"]["note"])
        self.assertIn("also when it ran alone", status["e3"]["note"])

    def test_a_kill_with_nobody_else_running_is_a_failure_at_once_and_not_a_crash_of_the_guard(self):
        r, zips, tmp = self.run_notebook(gpus="1", STUB_KILL_ALWAYS="e3")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        rows = [s for s in self.csv_rows(self.zip_files(zips[0]), "method_status.csv") if s["method"] == "e3"]
        self.assertEqual([s["status"] for s in rows], ["failed"] * 3, rows)                    # an out-of-memory kill is a result of the method: no guard
        self.assertTrue(all(s["attempt"] == "1" and "exit code -9" in s["note"] for s in rows))
        self.assertNotIn("KILLED while the other GPU ran", out)

    def test_one_gpu_means_one_worker(self):
        r, zips, tmp = self.run_notebook(entries=["9T5K", "9WQ6"], gpus="1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertEqual({g for _n, _p, g, _a, _b in self.timeline(tmp)}, {"0"})
        files = self.zip_files(zips[0])
        status = self.csv_rows(files, "method_status.csv")
        self.assertEqual(len(status), 8)
        self.assertTrue(all(s["status"] == "ok" and s["gpu"] == "0" and s["other_gpu_busy"] == "False" for s in status), status)

    def test_only_the_methods_that_have_tasks_are_set_up(self):
        r, zips, tmp = self.run_notebook(tasks=[["22FX", "emprot"], ["22FX", "modelangelo"]])
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = self.zip_files(zips[0])
        self.assertEqual(sorted((s["entry"], s["method"], s["status"]) for s in self.csv_rows(files, "method_status.csv")), [("22FX", "emprot", "ok"), ("22FX", "modelangelo", "ok")])
        self.assertNotIn("setup of E3-CryoFold", out)
        self.assertNotIn("setup of EModelX", out)
        self.assertFalse(os.path.exists(os.path.join(tmp, "big", "src_e3")))

    def test_a_second_run_in_the_same_folders_reuses_the_weights(self):
        r, zips, tmp = self.run_notebook(tasks=[["22FX", "modelangelo"]])
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-3000:])
        self.assertIn("downloading the weights: model_angelo setup_weights", r.stdout)
        # the same work and big folders again
        T.Files.routes = dict(self.routes)
        script = os.path.join(tmp, "nb.py")
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1", MICA_RETRY_WAIT="0.1", MICA_AF3_DIR=AF3_DIR, MICA_TEST_GPUS="2",
                   MICA_POLL="0.2", MICA_RESERVE_S="0", MICA_MIN_START_S="2", STUB_CALLS=os.path.join(tmp, "calls.txt"), STUB_TIMELINE=os.path.join(tmp, "timeline.txt"),
                   STUB_MARKERS=os.path.join(tmp, "markers"), STUB_MODEL_CIF=self.cif, STUB_MODEL_PDB=self.pdb)
        r2 = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(r2.returncode, 0, (r2.stdout + r2.stderr)[-3000:])
        self.assertNotIn("downloading the weights: model_angelo setup_weights", r2.stdout)
        self.assertIn("RESULT: OK", r2.stdout)


# ---------------------------------------------------------------------------------------------------------- CryoAtom v1.0.0: a notebook of its own
# the function of CryoAtom v1.0.0 (commit e2a31f2b, CryoAtom/utils/aa_probs_to_hmm.py, lines 177 to 186) that names the one temporary file of all processes, without its imports
UPSTREAM_AA_PROBS_TO_HMM = '''import os


def aa_logits_to_hmm(aa_logits: np.ndarray, base_dir: str = "/tmp") -> HMM:
    aa_probs = torch.from_numpy(aa_logits).softmax(dim=-1).numpy()
    tmp_path = os.path.join(base_dir, f"CryoAtom_temp.hmm")
    aa_probs_to_hmm_file("CryoAtom_search", aa_probs, tmp_path)

    with HMMFile(tmp_path) as hmm_file:
        hmm = hmm_file.read()

    os.remove(tmp_path)
    return hmm
'''
STUB_CRYOATOM = {
    "esm/__init__.py": "from . import pretrained\n",
    "esm/pretrained.py": r'''
import os, sys


def esm2_t33_650M_UR50D():
    with open(os.environ["STUB_ESM_CALLS"], "a") as f:
        f.write("download pid=%d gpu=%s\n" % (os.getpid(), os.environ.get("CUDA_VISIBLE_DEVICES")))
    if os.environ.get("STUB_ESM_FAIL"):
        print("urllib.error.URLError: stand-in network failure")
        sys.stdout.flush()
        sys.exit(1)
    folder = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "esm2_t33_650M_UR50D.pt"), "wb") as f:
        f.write(b"\0" * 1500000)
    print("stand-in ESM-2 weights written")
''',
    "CryoAtom/__init__.py": "__version__ = '1.0.0-stub'\n", "CryoAtom/build.py": "", "CryoAtom/utils/__init__.py": "",
    "CryoAtom/utils/aa_probs_to_hmm.py": UPSTREAM_AA_PROBS_TO_HMM,
    "CryoAtom/__main__.py": PRELUDE + r'''
import argparse
p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd")
b = sub.add_parser("build"); b.add_argument("-s", "--seq"); b.add_argument("-v", "--volume"); b.add_argument("-o", "--output")
a = p.parse_args()
assert a.cmd == "build"
calls("cryoatom1")
need_gpu_env()
here = os.path.dirname(os.path.abspath(__file__))
source = open(os.path.join(here, "utils", "aa_probs_to_hmm.py")).read()
assert 'f"CryoAtom_temp_{os.getpid()}.hmm"' in source and 'f"CryoAtom_temp.hmm"' not in source, "the temporary profile file is still shared by all processes"
for n in ("CryNet.pth", "SimpleUnet.pth"):
    assert os.path.getsize(os.path.join(here, "checkpoint", n)) > 1000, "the CryoAtom weights are not in place"
assert os.path.exists(os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", "esm2_t33_650M_UR50D.pt")), "the ESM-2 weights are not in place: the program would download them now"
assert os.path.exists(a.volume) and a.volume.endswith(".mrc") and os.path.exists(a.seq)
maybe_fail("cryoatom1")
if os.environ.get("STUB_OOM"):
    print("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 14.75 GiB total capacity; 12.90 GiB already allocated)")
    sys.stdout.flush()
    sys.exit(1)
work("cryoatom1")
out = os.path.normpath(a.output)
name = os.path.basename(out)
os.makedirs(out, exist_ok=True)
shutil.copy(os.environ["STUB_MODEL_CIF"], os.path.join(out, name + ".cif"))
shutil.copy(os.environ["STUB_MODEL_CIF"], os.path.join(out, name + "_raw.cif"))
open(os.path.join(out, "running_time.log"), "w").write("stand-in log\n")
''',
}

SYNTH_MODEL = r'''
import math, sys
from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure
s, m, c = Structure.Structure("m"), Model.Model(0), Chain.Chain("A")
s.add(m)
m.add(c)
serial = 0
for i in range(60):                                                  # an ideal helix of 60 alanines: a protein that US-align reads
    r = Residue.Residue((" ", i + 1, " "), "ALA", " ")
    a = math.radians(100 * i)
    x, y, z = 2.3 * math.cos(a), 2.3 * math.sin(a), 1.5 * i
    for name, dx, dy, dz in (("N", -0.6, -0.9, -0.5), ("CA", 0, 0, 0), ("C", 0.7, 0.9, 0.4), ("O", 0.9, 1.4, 1.4), ("CB", -0.5, 0.6, -0.8)):
        serial += 1
        r.add(Atom.Atom(name, (x + dx, y + dy, z + dz), 20.0, 1.0, " ", " %-3s" % name, serial, name[0]))
    c.add(r)
io = MMCIFIO()
io.set_structure(s)
io.save(sys.argv[1])
'''


def literal(nb, name):
    """The value of the top-level assignment `name = <literal>` in a code cell of the notebook."""
    import ast
    for c in nb.cells:
        if c.cell_type == "code" and not c.source.startswith("%%writefile"):
            for node in ast.parse(c.source).body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                    try:
                        return ast.literal_eval(node.value)
                    except ValueError:
                        continue
    raise KeyError(name)


def segment(nb, cid, kind, name):
    import ast
    src = next(c for c in nb.cells if c["id"] == cid).source
    for node in ast.parse(src).body:
        if isinstance(node, kind) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(name)


class CryoAtomStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cryobuild_")
        cls.path = os.path.join(cls.tmp, "nb.ipynb")
        cls.expected = M.build(cls.path, batch="cryoatom_test", hours=11.0, run_methods=["cryoatom1"])
        cls.nb = nbformat.read(cls.path, as_version=4)
        cls.text = "\n".join(c.source for c in cls.nb.cells)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cell(self, cid):
        return next(c for c in self.nb.cells if c["id"] == cid).source

    def test_cells_compile_and_have_no_placeholders(self):
        for c in self.nb.cells:
            if c.cell_type == "code" and not c.source.startswith("%%writefile"):
                compile(c.source, c["id"], "exec")
            self.assertEqual(re.findall(r"@[A-Z_0-9]+@", c.source), [], c["id"])

    def test_cell_order_has_one_method_and_the_steps_are_numbered_again(self):
        ids = [c["id"] for c in self.nb.cells if c.cell_type == "code" and not c["id"].startswith("file-")]
        self.assertEqual(ids, ["settings", "step1", "step2-dirs", "step2-check", "robust", "tools", "step4", "cryoatom", "run", "summary", "final"])
        self.assertIn("## Step 5. CryoAtom v1.0.0", self.cell("cryoatom-title"))
        self.assertIn("## Step 6. The runs", self.cell("run-title"))
        self.assertIn("## Step 7. Summary", self.cell("summary-title"))

    def test_the_setting_is_eleven_hours_and_the_text_matches(self):
        first = next(c for c in self.nb.cells if c.cell_type == "code")
        self.assertEqual(first["id"], "settings")
        self.assertRegex(first.source, r"(?m)^HOURS = 11\.0$")
        self.assertIn("less than 12 hours left this week", first.source)

    def test_no_alphafold_dataset_is_needed(self):
        specs = literal(self.nb, "SPECS")
        self.assertEqual(list(specs), ["cryoatom1"])
        self.assertFalse(specs["cryoatom1"]["uses_af3"])
        self.assertIn("no AlphaFold dataset is needed", self.cell("intro"))
        self.assertIn("no input dataset and no secret", self.cell("intro"))
        hosts = literal(self.nb, "NET_HOSTS")
        self.assertIn("yanglab.qd.sdu.edu.cn", " ".join(h[1] for h in hosts))
        self.assertIn("esm2_t33_650M_UR50D.pt", " ".join(h[1] for h in hosts))
        self.assertNotIn("zenodo", " ".join(h[1] for h in hosts))

    def test_the_command_is_the_readme_example_of_v1_0_0(self):
        spec = literal(self.nb, "SPECS")["cryoatom1"]
        self.assertEqual(spec["commit"], "e2a31f2b2ff21fae363490ac5f4a5e945513024c")
        self.assertEqual(spec["commit"], TM.mm.SPECS["cryoatom1"]["commit"])                      # the commit of the pilot
        self.assertEqual((spec["pkg"], spec["cli"]), ("CryoAtom", ["-s", "{fasta}", "-v", "{map}", "-o", "{out}"]))
        self.assertEqual(spec["ckpt_files"], ["CryNet.pth", "SimpleUnet.pth"])
        code = self.cell("cryoatom")
        self.assertIn('cmd = [env_py, "-m", spec["pkg"], "build"] + args', code)
        self.assertIn('name = "%s_%s" % (entry, spec["key"])', code)
        self.assertIn('env = dict(r["env"], CUDA_VISIBLE_DEVICES=str(gpu))', self.cell("tools"))   # the GPU of the worker
        self.assertNotIn("cryoatom2", self.text)                                                  # CryoAtom2 is not part of this notebook
        self.assertNotIn("CryoAtom2", self.cell("cryoatom"))

    def test_the_limit_and_the_time_model(self):
        spec = literal(self.nb, "SPECS")["cryoatom1"]
        self.assertEqual((spec["timeout"], spec["s_per_residue"]), (36000, 51))
        self.assertTrue(spec["oom_is_result"])                                                     # a GPU out-of-memory error is a result, not a crash of the installation
        self.assertIn('spec.get("oom_is_result")', self.cell("tools"))
        self.assertEqual((M.CRYOATOM_LIMIT_S, M.CRYOATOM_S_PER_RESIDUE), (36000, 51))
        tools = self.cell("tools")
        self.assertIn("RUN_LIMIT = 10800", tools)                                                  # the default of the other methods is not changed
        self.assertIn('run_limit = spec.get("timeout") or RUN_LIMIT', tools)
        self.assertIn("cut = limit < run_limit - 1", tools)
        self.assertIn("if HARD_STOP - now < need_s(key, entry):", tools)
        self.assertIn("Every run has a time limit of 10 hours", self.cell("intro"))
        sizes = literal(self.nb, "SIZES")
        self.assertEqual(len(sizes), 20)
        self.assertEqual(sizes["26DK"], 191)

    def test_the_esm_weights_are_fetched_once_before_the_two_gpus_start(self):
        code = self.cell("cryoatom")
        self.assertIn("def setup_cryoatom_session(spec, repo, env_py):", code)
        self.assertIn('SETUPS["cryoatom1"], PLANS["cryoatom1"] = setup_cryoatom_session, plan_cryoatom', code)
        esm = literal(self.nb, "SPECS")["cryoatom1"]["esm"]
        self.assertEqual(esm["call"], "esm2_t33_650M_UR50D")                                       # what protein.py of v1.0.0 calls
        self.assertGreaterEqual(esm["min_bytes"], 500000000)                                        # a file of a few hundred MB cannot be the model (about 2.5 GB)

    def test_one_line_of_the_source_is_changed_so_that_two_programs_do_not_share_a_file(self):
        code = self.cell("cryoatom")
        self.assertIn("def patch_cryoatom_tempfile(spec, repo):", code)
        self.assertIn("setup_cryoatom(spec, repo, env_py)\n    patch_cryoatom_tempfile(spec, repo)", code)       # right after the setup of the pilot, before anything is fetched
        self.assertIn('tmp_path = os.path.join(base_dir, f"CryoAtom_temp.hmm")', code)
        self.assertIn('tmp_path = os.path.join(base_dir, f"CryoAtom_temp_{os.getpid()}.hmm")', code)
        self.assertIn("**One line of the source is changed**", self.cell("cryoatom-title"))
        self.assertIn("cryoatom_source_patch.diff", self.cell("cryoatom-title"))

    def test_the_tasks_are_the_twenty_entries_in_the_seeded_order(self):
        tasks = literal(self.nb, "TASKS")
        self.assertEqual(tasks, [[e, "cryoatom1"] for e in M.seeded_order(M.read_entries())])
        self.assertEqual(len(tasks), 20)
        self.assertEqual([t[0] for t in tasks][:4], ["9T5K", "9WQ6", "22FX", "9V4D"])
        self.assertEqual(literal(self.nb, "RUN_METHODS"), ["cryoatom1"])
        self.assertEqual(literal(self.nb, "MAX_LANES"), 2)

    def test_the_embedded_files_are_the_common_ones_and_the_environment_of_cryoatom(self):
        got = {}
        for c in self.nb.cells:
            if c.source.startswith("%%writefile"):
                first, text = c.source.split("\n", 1)
                got[first.split(None, 1)[1].strip()] = text
        self.assertEqual(sorted(got), sorted([t for _s, t in M.BASE_FILES] + ["env/cryoatom1_requirements.txt"]))
        for target, text in got.items():
            with open(os.path.join(ROOT, target), encoding="utf-8") as f:
                want = f.read()
            self.assertEqual(text, want if want.endswith("\n") else want + "\n", target)
            self.assertEqual(self.expected[target], hashlib.sha256(text.encode()).hexdigest())

    def test_the_other_methods_are_not_in_the_notebook(self):
        code = "\n".join(c.source for c in self.nb.cells if c.cell_type == "code" and not c.source.startswith("%%writefile"))
        for name in ("setup_modelangelo", "plan_emprot", "plan_e3", "setup_emodelx", "model_angelo"):
            self.assertNotIn(name, code)

    def test_it_cannot_be_mixed_with_the_other_methods_and_the_other_notebooks_are_not_changed(self):
        with self.assertRaises(ValueError):
            M.build(os.path.join(self.tmp, "x.ipynb"), run_methods=["cryoatom1", "e3"])
        with self.assertRaises(ValueError):
            M.build(os.path.join(self.tmp, "x.ipynb"), tasks=[["9T5K", "cryoatom1"]])               # the four-method notebook does not know it
        self.assertEqual(M.ORDER, ["modelangelo", "e3", "emprot", "emodelx"])
        self.assertNotIn("cryoatom1", M.SPECS)


class CryoAtomLogicTests(unittest.TestCase):
    """need_s, take_task and the ESM-2 step, run on their own source text taken from the built notebook."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cryologic_")
        path = os.path.join(cls.tmp, "nb.ipynb")
        M.build(path, batch="cryoatom_test", hours=11.0, run_methods=["cryoatom1"])
        cls.nb = nbformat.read(path, as_version=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def namespace(self, **extra):
        import ast
        ns = {"SPECS": {"cryoatom1": {"s_per_residue": 51}, "e3": {}}, "SIZES": {"A": 100, "B": 400, "C": 1}, "RATES": {}, "DURATIONS": {}, "MIN_START_S": 900.0}
        ns.update(extra)
        exec(segment(self.nb, "tools", ast.FunctionDef, "need_s"), ns)
        return ns

    def test_before_a_run_has_finished_the_estimate_is_the_rate_of_the_pilot(self):
        ns = self.namespace()
        self.assertAlmostEqual(ns["need_s"]("cryoatom1", "B"), 1.25 * 51 * 400)
        self.assertAlmostEqual(ns["need_s"]("cryoatom1", "A"), 1.25 * 51 * 100)

    def test_a_tiny_entry_still_needs_the_minimum(self):
        self.assertEqual(self.namespace()["need_s"]("cryoatom1", "C"), 900.0)

    def test_the_largest_rate_of_the_session_replaces_the_pilot_rate(self):
        ns = self.namespace(RATES={"cryoatom1": [3.0, 9.0, 5.0]})
        self.assertAlmostEqual(ns["need_s"]("cryoatom1", "B"), 1.25 * 9.0 * 400)

    def test_a_method_without_a_time_model_keeps_the_rule_of_the_other_notebooks(self):
        ns = self.namespace(MIN_START_S=10.0, DURATIONS={"e3": [100, 300]})
        self.assertEqual(ns["need_s"]("e3", "B"), 250.0)
        self.assertEqual(ns["need_s"]("e3"), 250.0)
        self.assertEqual(self.namespace(MIN_START_S=10.0)["need_s"]("e3", "B"), 10.0)

    def test_an_entry_that_is_not_in_the_table_falls_back_to_the_mean_duration(self):
        ns = self.namespace(MIN_START_S=10.0, DURATIONS={"cryoatom1": [200, 400]})
        self.assertEqual(ns["need_s"]("cryoatom1", "ZZ"), 375.0)

    def take_ns(self, hours_left, **extra):
        import ast
        import threading
        import time
        ns = self.namespace(SCHED=threading.RLock(), time=time, BUSY={}, EXPLOSIVE=("e3", "emodelx"), HARD_STOP=time.time() + hours_left * 3600,
                            PENDING=[("9WFL", "cryoatom1"), ("22FX", "cryoatom1")], STATE={"methods": {"cryoatom1": {"status": "ready"}}})
        ns["SIZES"] = {"9WFL": 682, "22FX": 173}
        ns.update(extra)
        exec(segment(self.nb, "tools", ast.FunctionDef, "take_task"), ns)
        return ns

    def test_an_entry_that_does_not_fit_is_skipped_and_a_smaller_one_later_in_the_list_is_taken(self):
        ns = self.take_ns(5.0)                                               # 9WFL: 12.1 hours estimated, 22FX: 3.1 hours
        self.assertEqual(ns["take_task"](0), ("22FX", "cryoatom1"))
        self.assertEqual(ns["PENDING"], [("9WFL", "cryoatom1")])
        self.assertIsNone(ns["take_task"](1))                                # nothing more fits: this GPU is done

    def test_after_a_fast_run_the_big_entry_fits_in_the_same_time(self):
        ns = self.take_ns(5.0, RATES={"cryoatom1": [3.0]})                   # 1.25 x 3 x 682 = 2557 s
        self.assertEqual(ns["take_task"](0), ("9WFL", "cryoatom1"))

    def esm_ns(self, tmp, rcs, present=False):
        import ast
        calls, recorded, setups = [], [], []
        folder = os.path.join(tmp, "torch", "hub", "checkpoints")
        if present:
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "esm2_t33_650M_UR50D.pt"), "wb") as f:
                f.write(b"0" * 3000)
        outcomes = list(rcs)

        def run_watched(cmd, label, watch=(), **kw):
            rc = outcomes.pop(0)
            calls.append((cmd, kw))
            if rc == 0:
                os.makedirs(folder, exist_ok=True)
                with open(os.path.join(folder, "esm2_t33_650M_UR50D.pt"), "wb") as f:
                    f.write(b"0" * 3000)
            return rc, ["last line"], False

        ns = {"os": os, "TORCH_HOME": os.path.join(tmp, "torch"), "WORK": tmp, "log": lambda *a: None, "setup_cryoatom": lambda spec, repo, env_py: setups.append((spec["key"], repo, env_py)),
              "patch_cryoatom_tempfile": lambda spec, repo: setups.append(("patch", repo)),
              "run_watched": run_watched, "env_vars": lambda py: {"python": py}, "record_files": lambda *a: recorded.append(a)}
        exec(segment(self.nb, "cryoatom", ast.FunctionDef, "setup_cryoatom_session"), ns)
        return ns, calls, recorded, setups

    SPEC = {"key": "cryoatom1", "esm": {"call": "esm2_t33_650M_UR50D", "file": "esm2_t33_650M_UR50D.pt", "min_bytes": 1000}}

    def test_the_esm_weights_are_fetched_with_the_environment_of_the_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns, calls, recorded, setups = self.esm_ns(tmp, [0])
            ns["setup_cryoatom_session"](self.SPEC, "/src", "/env/bin/python")
            self.assertEqual(setups, [("cryoatom1", "/src", "/env/bin/python"), ("patch", "/src")])  # the setup of the pilot comes first, then the one-line change of the source
            self.assertEqual(len(calls), 1)
            cmd, kw = calls[0]
            self.assertEqual(cmd[:2], ["/env/bin/python", "-c"])
            self.assertIn("esm.pretrained.esm2_t33_650M_UR50D()", cmd[2])
            self.assertEqual((kw["cwd"], kw["env"]), ("/src", {"python": "/env/bin/python"}))
            self.assertEqual(recorded[0][0], "cryoatom1")
            self.assertEqual(recorded[0][2], "esm_weights")

    def test_weights_that_are_in_place_are_not_fetched_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns, calls, recorded, _ = self.esm_ns(tmp, [], present=True)
            ns["setup_cryoatom_session"](self.SPEC, "/src", "/env/bin/python")
            self.assertEqual(calls, [])
            self.assertEqual(len(recorded), 1)

    def test_a_second_attempt_follows_a_failed_one_and_two_failures_stop_the_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns, calls, _, _ = self.esm_ns(tmp, [1, 0])
            ns["setup_cryoatom_session"](self.SPEC, "/src", "/env/bin/python")
            self.assertEqual(len(calls), 2)
        with tempfile.TemporaryDirectory() as tmp:
            ns, calls, recorded, _ = self.esm_ns(tmp, [1, 1])
            with self.assertRaisesRegex(RuntimeError, "ESM-2 weights could not be downloaded"):
                ns["setup_cryoatom_session"](self.SPEC, "/src", "/env/bin/python")
            self.assertEqual((len(calls), recorded), (2, []))

    def test_a_spec_without_esm_only_runs_the_setup_of_the_pilot_and_the_change_of_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns, calls, recorded, setups = self.esm_ns(tmp, [])
            ns["setup_cryoatom_session"]({"key": "cryoatom1"}, "/src", "/env/bin/python")
            self.assertEqual((setups, calls, recorded), ([("cryoatom1", "/src", "/env/bin/python"), ("patch", "/src")], [], []))

    def patch_ns(self, tmp, text, diff="diff --git a/x b/x\n+changed"):
        import ast
        os.makedirs(os.path.join(tmp, "repo", "CryoAtom", "utils"))
        os.makedirs(os.path.join(tmp, "work"))
        path = os.path.join(tmp, "repo", "CryoAtom", "utils", "aa_probs_to_hmm.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logged, shells = [], []

        def sha256_file(p):
            with open(p, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        ns = {"os": os, "WORK": os.path.join(tmp, "work"), "log": lambda *a: logged.append(a[0]), "STATE": {"methods": {"cryoatom1": {}}},
              "sh": lambda cmd, timeout=60: shells.append(cmd) or diff, "sha256_file": sha256_file}
        exec(segment(self.nb, "cryoatom", ast.FunctionDef, "patch_cryoatom_tempfile"), ns)
        return ns, path, logged, shells

    def test_the_temporary_profile_file_is_named_after_the_process(self):
        # the shared file of two programs at the same time: /tmp/CryoAtom_temp.hmm in the source of v1.0.0, which the stand-in repeats word for word
        with tempfile.TemporaryDirectory() as tmp:
            ns, path, logged, shells = self.patch_ns(tmp, UPSTREAM_AA_PROBS_TO_HMM)
            ns["patch_cryoatom_tempfile"]({"key": "cryoatom1"}, os.path.join(tmp, "repo"))
            with open(path, encoding="utf-8") as f:
                after = f.read()
            self.assertEqual(after, UPSTREAM_AA_PROBS_TO_HMM.replace('f"CryoAtom_temp.hmm"', 'f"CryoAtom_temp_{os.getpid()}.hmm"'))   # nothing else changes
            self.assertEqual(after.count("os.getpid()"), 1)
            self.assertEqual(len(logged), 1)
            self.assertEqual(shells, ["git -C %s diff" % os.path.join(tmp, "repo")])
            with open(os.path.join(tmp, "work", "cryoatom_source_patch.diff"), encoding="utf-8") as f:
                self.assertIn("+changed", f.read())                                                    # the diff goes into the zip
            info = ns["STATE"]["methods"]["cryoatom1"]["source_patch"]
            self.assertEqual(info["file"], os.path.join("CryoAtom", "utils", "aa_probs_to_hmm.py"))
            self.assertEqual(info["file_sha256"], hashlib.sha256(after.encode()).hexdigest())
            ns["patch_cryoatom_tempfile"]({"key": "cryoatom1"}, os.path.join(tmp, "repo"))               # a second setup in the same folders changes nothing more
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), after)
            self.assertEqual(len(logged), 1)

    def test_a_source_without_that_line_stops_the_setup(self):
        for text in (UPSTREAM_AA_PROBS_TO_HMM.replace("CryoAtom_temp.hmm", "CryoAtom_other.hmm"),                  # another file name: not the pinned commit
                     UPSTREAM_AA_PROBS_TO_HMM + '\n' + 'tmp_path = os.path.join(base_dir, f"CryoAtom_temp.hmm")\n'):   # twice: the change would be ambiguous
            with tempfile.TemporaryDirectory() as tmp:
                ns, path, _logged, _shells = self.patch_ns(tmp, text)
                with self.assertRaisesRegex(RuntimeError, "not the pinned commit"):
                    ns["patch_cryoatom_tempfile"]({"key": "cryoatom1"}, os.path.join(tmp, "repo"))


@unittest.skipUnless(ENV_PY and USALIGN, "needs MICA_TEST_ENV_PYTHON (numpy, mrcfile, biopython) and USALIGN; the AlphaFold files are not needed")
class CryoAtomEndToEndTests(unittest.TestCase):
    """The whole notebook of CryoAtom as a script against a stand-in program (a local git repository), a stand-in of esm.pretrained, a file server and the real US-align."""
    ENTRIES = ["9T5K", "9WQ6", "22FX", "9V4D"]

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="cryorun_")
        cls.cif = os.path.join(cls.root, "model.cif")
        subprocess.run([ENV_PY, "-c", SYNTH_MODEL, cls.cif], check=True)
        with open(cls.cif, "rb") as f:
            cif_bytes = f.read()
        mapfile = os.path.join(cls.root, "stub.map")
        subprocess.run([ENV_PY, "-c", "import mrcfile, numpy; m = mrcfile.new(%r, overwrite=True); m.set_data(numpy.random.rand(32, 32, 32).astype('float32')); "
                        "m.voxel_size = 1.0; m.close()" % mapfile], check=True)
        with open(mapfile, "rb") as f:
            map_gz = gzip.compress(f.read())
        blob = lambda n, seed: bytes((i * seed) % 251 for i in range(256)) * (n // 256)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("checkpoints/CryNet.pth", blob(1500000, 3))
            z.writestr("checkpoints/SimpleUnet.pth", blob(1500000, 5))
        cls.srv, cls.url = T.serve(T.Files)
        cls.routes = {"/map/stub.map.gz": map_gz, "/cif/stub.cif": cif_bytes, "/cryoatom/checkpoints_v1.0.zip": buf.getvalue()}
        T.Files.routes = dict(cls.routes)
        cls.entries = os.path.join(cls.root, "entries.csv")
        with open(os.path.join(ROOT, "data", "entries_2026.csv"), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows, fields = list(reader), reader.fieldnames
        for r in rows:
            r["map_url"], r["cif_url"] = cls.url + "/map/stub.map.gz", cls.url + "/cif/stub.cif"
        with open(cls.entries, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        repos = {}
        for name, other_line in (("cryoatom_stub", False), ("cryoatom_stub_other_line", True)):          # the second one is not the pinned commit: the line that must be changed is not there
            repo = os.path.join(cls.root, name)
            for rel, text in STUB_CRYOATOM.items():
                path = os.path.join(repo, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(text.replace("CryoAtom_temp.hmm", "CryoAtom_other.hmm") if (other_line and rel.endswith("aa_probs_to_hmm.py")) else text)
            T.sh("git init -q -b main . && git add -A && git -c user.name=t -c user.email=t@t commit -q -m stub", cwd=repo)
            repos[name] = (repo, T.sh("git rev-parse HEAD", cwd=repo))
        cls.repo, cls.commit = repos["cryoatom_stub"]
        cls.repo_other_line, cls.commit_other_line = repos["cryoatom_stub_other_line"]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def stub_specs(self, **update):
        s = {"cryoatom1": M.cryoatom_spec()}
        s["cryoatom1"].update(repo="file://" + self.repo, commit=self.commit, weights_url=self.url + "/cryoatom/checkpoints_v1.0.zip", weights_min_bytes=1000, s_per_residue=0.001)
        s["cryoatom1"]["esm"] = dict(s["cryoatom1"]["esm"], min_bytes=1000)
        s["cryoatom1"].update(update)
        return s

    def run_notebook(self, tasks=None, hours=11.0, gpus="2", routes=None, spec_update=None, **extra_env):
        T.Files.routes = dict(self.routes, **(routes or {}))
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        nb_path = os.path.join(tmp, "nb.ipynb")
        M.build(nb_path, batch="test", entries=self.ENTRIES, tasks=tasks, hours=hours, entries_path=self.entries, env_python=ENV_PY, usalign_bin=USALIGN, net_hosts=(), require_gpu=False,
                specs=self.stub_specs(**(spec_update or {})), run_methods=["cryoatom1"])
        nb = nbformat.read(nb_path, as_version=4)
        script = os.path.join(tmp, "nb.py")
        T.to_script(nb, script)
        os.makedirs(os.path.join(tmp, "markers"))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("STUB_", "MICA_AF3"))}
        env.update(MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1", MICA_RETRY_WAIT="0.1", MICA_TEST_GPUS=gpus, MICA_POLL="0.2", MICA_RESERVE_S="0",
                   MICA_MIN_START_S="2", STUB_CALLS=os.path.join(tmp, "calls.txt"), STUB_TIMELINE=os.path.join(tmp, "timeline.txt"), STUB_MARKERS=os.path.join(tmp, "markers"),
                   STUB_ESM_CALLS=os.path.join(tmp, "esm_calls.txt"), STUB_MODEL_CIF=self.cif, MPLBACKEND="module://matplotlib_inline.backend_inline")
        env.update(extra_env)
        r = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=900)
        work = os.path.join(tmp, "work")
        zips = sorted(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".zip")) if os.path.isdir(work) else []
        return r, zips, tmp

    def test_a_complete_run_with_two_gpus_and_without_the_alphafold_dataset(self):
        r, zips, tmp = self.run_notebook(STUB_SLOW_CRYOATOM1="0.8")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out)
        files = EndToEndTests.zip_files(zips[-1])
        status = EndToEndTests.csv_rows(files, "method_status.csv")
        self.assertEqual(sorted((s["entry"], s["method"], s["status"]) for s in status), sorted((e, "cryoatom1", "ok") for e in self.ENTRIES))
        self.assertEqual({s["limit_s"] for s in status}, {"36000"})                       # the limit of CryoAtom, not the 3 hours of the other methods
        metrics = EndToEndTests.csv_rows(files, "metrics_usalign.csv")
        self.assertEqual(sorted((m["entry"], m["model_kind"]) for m in metrics), sorted((e, k) for e in self.ENTRIES for k in ("cryoatom1", "cryoatom1_raw")))
        self.assertTrue(all(float(m["tm_score"]) > 0.99 for m in metrics), metrics)
        for e in self.ENTRIES:
            self.assertIn("models/cryoatom1/%s/%s_cryoatom1.cif" % (e, e), files)
            self.assertIn("models/cryoatom1/%s/%s_cryoatom1_raw.cif" % (e, e), files)
            self.assertIn("logs/%s_cryoatom1_build.log" % e, files)
        self.assertIn("logs/cryoatom_esm2_weights.log", files)
        # the one changed line of the source: the diff is in the zip, the notebook recorded the file and its hash, and the programs ran with the changed file (the stand-in asserts it)
        diff = files["cryoatom_source_patch.diff"].decode()
        self.assertIn('+    tmp_path = os.path.join(base_dir, f"CryoAtom_temp_{os.getpid()}.hmm")', diff)
        self.assertIn('-    tmp_path = os.path.join(base_dir, f"CryoAtom_temp.hmm")', diff)
        self.assertEqual(diff.count("\n+") + diff.count("\n-"), 4, diff)                       # two header lines (+++ and ---) and the two lines of the change
        # the command of the README of v1.0.0, one run per entry
        calls = read_lines(os.path.join(tmp, "calls.txt"))
        self.assertEqual(len(calls), 4, calls)
        for line in calls:
            self.assertRegex(line, r"^cryoatom1 build -s \S+/[0-9a-z]{4}\.fasta -v \S+/emd_\d+\.mrc -o \S+/[0-9A-Z]{4}_cryoatom1$")
        self.assertEqual({line.rsplit("/", 1)[1].split("_")[0] for line in calls}, set(self.ENTRIES))
        # both GPUs, at the same time
        tl = EndToEndTests.timeline(tmp)
        self.assertEqual({g for _n, _p, g, _a, _b in tl}, {"0", "1"}, tl)
        self.assertTrue(any(a1 < b2 and a2 < b1 for _n1, _p1, g1, a1, b1 in tl for _n2, _p2, g2, a2, b2 in tl if g1 != g2), "no two programs ran at the same time: %s" % tl)
        # the ESM-2 weights were fetched once, by the setup, not by the two programs
        esm_calls = read_lines(os.path.join(tmp, "esm_calls.txt"))
        self.assertEqual(len(esm_calls), 1, esm_calls)
        info = json.loads(files["run_info.json"].decode())
        self.assertEqual(info["state"]["pending"], [])
        self.assertIn("esm_weights", info["state"]["methods"]["cryoatom1"])
        patch = info["state"]["methods"]["cryoatom1"]["source_patch"]
        self.assertEqual(patch["file"], "CryoAtom/utils/aa_probs_to_hmm.py")
        self.assertRegex(patch["file_sha256"], r"^[0-9a-f]{64}$")

    def test_an_entry_that_does_not_fit_in_what_is_left_is_skipped_and_a_smaller_one_runs(self):
        # 0.02 h = 72 s. At the start (the pilot rate of 0.12 s per residue) 9WFL (682 residues) is estimated at 1.25 x 0.12 x 682 = 102 s, 22FX (173 residues) at 26 s. 22FX takes about
        # 14 s, so its measured rate is about 0.08 s per residue and 9WFL is then estimated at about 70 s, more than the 50 s or so that are left: it stays for the next session
        r, zips, tmp = self.run_notebook(tasks=[["9WFL", "cryoatom1"], ["22FX", "cryoatom1"]], hours=0.02, spec_update={"s_per_residue": 0.12}, STUB_SLOW_CRYOATOM1="14")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = EndToEndTests.zip_files(zips[-1])
        status = {s["entry"]: s for s in EndToEndTests.csv_rows(files, "method_status.csv")}
        self.assertEqual((status["22FX"]["status"], status["9WFL"]["status"]), ("ok", "not run"), out[-2500:])
        pending = EndToEndTests.csv_rows(files, "pending.csv")
        self.assertEqual([(p["entry"], p["status"]) for p in pending], [("9WFL", "not run")])
        self.assertIn("not enough time left", pending[0]["note"])
        self.assertIn("RESULT: OK", out)                                                  # a session that ends on time is a normal session

    def test_the_seconds_per_residue_of_a_finished_run_replace_the_rate_of_the_pilot(self):
        # the pilot rate (1 s per residue) puts 9T5K (476 s) out of reach of the 360 s budget at the start; 22FX (216 s) fits and, once it has finished in about 2 s, 9T5K fits
        r, zips, tmp = self.run_notebook(tasks=[["22FX", "cryoatom1"], ["9T5K", "cryoatom1"]], hours=0.1, spec_update={"s_per_residue": 1.0}, STUB_SLOW_CRYOATOM1="1.5")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = EndToEndTests.zip_files(zips[-1])
        status = {s["entry"]: s for s in EndToEndTests.csv_rows(files, "method_status.csv")}
        self.assertEqual((status["22FX"]["status"], status["9T5K"]["status"]), ("ok", "ok"), out[-2500:])
        tl = sorted(EndToEndTests.timeline(tmp), key=lambda t: t[3])
        self.assertEqual(len(tl), 2)
        self.assertLessEqual(tl[0][4], tl[1][3], "the second run started before the first one ended: %s" % tl)
        self.assertEqual(tl[0][2], tl[1][2], "one GPU only: the other had nothing that fitted at the start")

    def test_a_run_that_the_end_of_the_session_cuts_is_interrupted_and_not_a_failure(self):
        r, zips, tmp = self.run_notebook(tasks=[["9T5K", "cryoatom1"]], hours=0.0075, STUB_SLOW_CRYOATOM1="60")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = EndToEndTests.zip_files(zips[-1])
        status = EndToEndTests.csv_rows(files, "method_status.csv")
        self.assertEqual([(s["entry"], s["method"], s["status"]) for s in status], [("9T5K", "cryoatom1", "interrupted")], out[-2000:])
        self.assertIn("INTERRUPTED by the end of the session (not a failure)", out)
        self.assertEqual([(p["entry"], p["status"]) for p in EndToEndTests.csv_rows(files, "pending.csv")], [("9T5K", "interrupted")])

    def test_esm_weights_that_cannot_be_fetched_stop_the_setup_before_any_run(self):
        r, zips, tmp = self.run_notebook(tasks=[["9T5K", "cryoatom1"], ["9WQ6", "cryoatom1"]], STUB_ESM_FAIL="1")
        out = r.stdout + r.stderr
        self.assertIn("CryoAtom v1.0.0 (code of the paper) STOPPED in its setup: the ESM-2 weights could not be downloaded", out)
        self.assertEqual(len(read_lines(os.path.join(tmp, "esm_calls.txt"))), 2)           # two attempts
        self.assertFalse(os.path.exists(os.path.join(tmp, "calls.txt")), "the program must not start without its weights")
        files = EndToEndTests.zip_files(zips[-1])
        status = EndToEndTests.csv_rows(files, "method_status.csv")
        self.assertEqual(sorted((s["entry"], s["status"]) for s in status), [("9T5K", "not run"), ("9WQ6", "not run")])
        self.assertEqual(len(EndToEndTests.csv_rows(files, "pending.csv")), 2)

    def test_a_source_that_is_not_the_pinned_one_stops_the_setup_before_any_run(self):
        # the line that two programs at the same time would share is not there: nothing runs (the file could be shared in a way that nobody sees)
        r, zips, tmp = self.run_notebook(tasks=[["9T5K", "cryoatom1"]], spec_update={"repo": "file://" + self.repo_other_line, "commit": self.commit_other_line})
        out = r.stdout + r.stderr
        self.assertIn("STOPPED in its setup: the line of CryoAtom that names its temporary profile file was not found", out)
        self.assertFalse(os.path.exists(os.path.join(tmp, "calls.txt")), "the program must not start")
        self.assertFalse(os.path.exists(os.path.join(tmp, "esm_calls.txt")), "no weights are fetched for a setup that stops")
        files = EndToEndTests.zip_files(zips[-1])
        self.assertEqual([(s["entry"], s["status"]) for s in EndToEndTests.csv_rows(files, "method_status.csv")], [("9T5K", "not run")])

    def test_a_program_that_fails_is_a_result_and_the_other_entries_go_on(self):
        r, zips, tmp = self.run_notebook(tasks=[["9T5K", "cryoatom1"], ["9WQ6", "cryoatom1"], ["22FX", "cryoatom1"]], STUB_FAIL="cryoatom1", MICA_QUICK_CRASH="0.0001")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        status = EndToEndTests.csv_rows(EndToEndTests.zip_files(zips[-1]), "method_status.csv")
        self.assertEqual([s["status"] for s in sorted(status, key=lambda s: s["entry"])], ["failed"] * 3, status)
        self.assertTrue(all("exit code 1" in s["note"] for s in status), status)

    def test_two_quick_crashes_in_a_row_still_stop_the_method_for_the_session(self):
        # the guard of the other notebooks is kept: a crash within the hour on two entries in a row is probably the installation, not a result
        r, zips, tmp = self.run_notebook(tasks=[["9T5K", "cryoatom1"], ["9WQ6", "cryoatom1"], ["22FX", "cryoatom1"]], STUB_FAIL="cryoatom1", gpus="1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = EndToEndTests.zip_files(zips[-1])
        status = EndToEndTests.csv_rows(files, "method_status.csv")
        self.assertEqual([s["status"] for s in status], ["failed", "failed", "not run"], status)
        self.assertEqual(status[2]["failed_step"], "guard")
        self.assertEqual([(p["entry"], p["status"]) for p in EndToEndTests.csv_rows(files, "pending.csv")], [("22FX", "not run")])

    def test_an_out_of_gpu_memory_error_is_a_result_and_does_not_trip_the_guard(self):
        # CryoAtom needs 13 to 14 GB of the 15 GB of a T4: for a large entry that may not be enough, and then the entry is a failure of the method, not a problem of the installation
        r, zips, tmp = self.run_notebook(tasks=[["9T5K", "cryoatom1"], ["9WQ6", "cryoatom1"], ["22FX", "cryoatom1"]], STUB_OOM="1", gpus="1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        status = EndToEndTests.csv_rows(EndToEndTests.zip_files(zips[-1]), "method_status.csv")
        self.assertEqual([s["status"] for s in status], ["failed"] * 3, status)
        self.assertTrue(all(s["note"].startswith("out of GPU memory") for s in status), status)
        self.assertEqual(len(read_lines(os.path.join(tmp, "calls.txt"))), 3)


# ---------------------------------------------------------------------------------------------------------- CryoZeta: a notebook of its own
COMMITTED_CRYOZETA = os.path.join(HERE, "gpu_cryozeta_d.ipynb")      # batches a, b and c (as_run/): a stopped on a missing data file, b finished 8 entries, c three more (LOG 2026-11-03 to 05)
CZ_B_STATUS = os.path.join(ROOT, "results", "raw", "gpu_cryozeta_b_result_20261103-1233", "method_status.csv")
CZ_C_STATUS = os.path.join(ROOT, "results", "raw", "gpu_cryozeta_c_result_20261104-1021", "method_status.csv")
CZ_D_ARGS = ["--batch", "cryozeta_d", "--methods", "cryozeta", "--hours", "11", "--entries", "9OC0", "9VI4", "9VBD", "9VUH", "9WFL", "9WME", "--alone-first", "9OC0:cryozeta", "--order-text",
             "from the smallest to the largest deposited model, except that 9OC0, which the system killed in the second session while the other GPU was busy, runs first and alone and 9WME, which the end of the first session cut after 4.9 hours, runs last"]
# the three programs of CryoZeta of the tests of the runner, plus: a model that US-align can read instead of the dummy text, and the time span of every call
STUB_CRYOZETA = r'''#!/usr/bin/env python3
import atexit, json, os, shutil, sys, time
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T0 = time.time()


def _finish():
    prog, a = os.path.basename(sys.argv[0]), sys.argv[1:]
    if prog == "cryozeta-combine" and os.environ.get("STUB_MODEL_CIF"):
        with open(a[a.index("--input-json-path") + 1]) as f:
            name = json.load(f)[0]["name"]
        final = os.path.join(a[a.index("--dump-dir") + 1], name, "CryoZeta-Final")
        for n in (os.listdir(final) if os.path.isdir(final) else []):
            if n.endswith(".cif"):
                shutil.copy(os.environ["STUB_MODEL_CIF"], os.path.join(final, n))
    with open(os.path.join(_HERE, "spans.jsonl"), "a") as f:
        f.write(json.dumps({"prog": prog, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "t0": _T0, "t1": time.time(), "pid": os.getpid()}) + "\n")


atexit.register(_finish)
if os.path.basename(sys.argv[0]) == "cryozeta-inference":
    time.sleep(float(os.environ.get("STUB_SLOW_CRYOZETA", "0")))
''' + TZ.STUB.split("\n", 1)[1]


class FlakyFiles(T.Files):
    """The file server of the tests, but it refuses (503) the first `fail[path]` requests of a path: a network error that goes away."""
    fail: dict = {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if FlakyFiles.fail.get(path, 0) > 0:
            FlakyFiles.fail[path] -= 1
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()


def jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class CryoZetaStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="czbuild_")
        cls.path = os.path.join(cls.tmp, "nb.ipynb")
        cls.expected = M.build(cls.path, batch="cryozeta_test", hours=11.0, run_methods=["cryozeta"])
        cls.nb = nbformat.read(cls.path, as_version=4)
        cls.text = "\n".join(c.source for c in cls.nb.cells)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cell(self, cid):
        return next(c for c in self.nb.cells if c["id"] == cid).source

    def test_cells_compile_and_have_no_placeholders(self):
        for c in self.nb.cells:
            if c.cell_type == "code" and not c.source.startswith("%%writefile"):
                compile(c.source, c["id"], "exec")
            self.assertEqual(re.findall(r"@[A-Z_0-9]+@", c.source), [], c["id"])

    def test_cell_order_has_one_method_and_the_steps_are_numbered(self):
        ids = [c["id"] for c in self.nb.cells if c.cell_type == "code" and not c["id"].startswith("file-")]
        self.assertEqual(ids, ["settings", "step1", "step2-dirs", "step2-check", "robust", "tools", "step4", "cryozeta", "run", "summary", "final"])
        self.assertIn("## Step 5. CryoZeta", self.cell("cryozeta-title"))
        self.assertIn("## Step 6. The runs", self.cell("run-title"))
        self.assertIn("## Step 7. Summary", self.cell("summary-title"))

    def test_the_setting_is_eleven_hours_and_the_text_matches(self):
        first = next(c for c in self.nb.cells if c.cell_type == "code")
        self.assertRegex(first.source, r"(?m)^HOURS = 11\.0$")
        self.assertIn("less than 12 hours left this week", first.source)

    def test_no_alphafold_dataset_is_needed_and_the_hosts_are_the_ones_of_the_setup(self):
        specs = literal(self.nb, "SPECS")
        self.assertEqual(list(specs), ["cryozeta"])
        self.assertFalse(specs["cryozeta"]["uses_af3"])
        self.assertIn("no AlphaFold dataset is needed", self.cell("intro"))
        self.assertIn("no input dataset and no secret", self.cell("intro"))
        hosts = {h[0]: h for h in literal(self.nb, "NET_HOSTS")}
        self.assertTrue(hosts["Hugging Face (CryoZeta weights)"][2] and hosts["conda-forge"][2] and hosts["GitHub"][2])           # without these the setup cannot work: stop before it starts
        self.assertFalse(hosts["ColabFold server"][2])
        self.assertIn("api.colabfold.com", hosts["ColabFold server"][1])
        self.assertNotIn("zenodo", " ".join(h[1] for h in hosts.values()))

    def test_the_pins(self):
        spec = literal(self.nb, "SPECS")["cryozeta"]
        self.assertEqual(spec["commit"], "c544c6bbd0366951d8b4e0bc4c0da547836f7b8f")
        self.assertEqual(spec["repo"], "https://github.com/kiharalab/CryoZeta")
        self.assertEqual((spec["pixi"]["version"], spec["pixi"]["env"]), ("0.41.1", "cu11"))
        self.assertRegex(spec["pixi"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("v0.41.1", spec["pixi"]["url"])
        self.assertEqual((spec["teaser"]["repo"], spec["teaser"]["commit"]), ("https://github.com/MIT-SPARK/TEASER-plusplus", "52a9c52ee7d4c838c5e8a75458c33178be5bfb70"))
        self.assertEqual((spec["teaser"]["cmake"], spec["teaser"]["pybind11"]), ("3.31.6", "3.0.1"))
        self.assertEqual(spec["hub"]["data_files"], {"ccd": "components.v20240608.cif", "ccd_rdkit": "components.v20240608.cif.rdkit_mol.pkl"})
        self.assertIn("CRYOZETA_ASSETS_DIR", self.cell("cryozeta"))
        self.assertIn("CRYOZETA_ASSETS_DIR", self.cell("cryozeta-title"))
        self.assertEqual(spec["hub"]["repo"], "KiharaLab/CryoZeta")
        self.assertEqual(spec["hub"]["files"], {"detection": "cryozeta-detection-v0.0.1.safetensors", "cryozeta": "cryozeta-v0.0.1.safetensors", "interpolate": "cryozeta-interpolate-v0.0.1.safetensors"})
        self.assertIn("blobs=true", spec["hub"]["api"])                                      # the size and the sha256 of every file
        for t in (self.cell("cryozeta"), self.cell("tools")):
            self.assertNotIn("pip install -q torch", t)

    def test_the_settings_of_a_run_are_those_of_the_demo_of_the_authors(self):
        spec = literal(self.nb, "SPECS")["cryozeta"]
        self.assertEqual(spec["run"], {"seed": 101, "n_sample": 5, "n_step": 20, "n_cycle": 10, "num_select": 5, "dtype": "bf16", "kernels": "off"})
        self.assertEqual(spec["colabfold"], "https://api.colabfold.com")

    def test_the_limit_and_the_time_model(self):
        spec = literal(self.nb, "SPECS")["cryozeta"]
        self.assertEqual((spec["timeout"], spec["s_per_residue"]), (21600, 15))
        self.assertTrue(spec["oom_is_result"])                                               # a GPU out-of-memory error is a result, not a crash of the installation
        self.assertEqual((M.CRYOZETA_LIMIT_S, M.CRYOZETA_S_PER_RESIDUE), (21600, 15))
        self.assertIn("Every run has a time limit of 6 hours", self.cell("intro"))
        self.assertRegex(self.cell("intro"), r"first estimate is\s+15 seconds per residue")
        self.assertIn("RUN_LIMIT = 10800", self.cell("tools"))                               # the default of the other methods is not changed
        self.assertEqual(literal(self.nb, "SIZES")["26DK"], 191)

    def test_the_tasks_are_the_twenty_entries_in_the_seeded_order(self):
        tasks = literal(self.nb, "TASKS")
        self.assertEqual(tasks, [[e, "cryozeta"] for e in M.seeded_order(M.read_entries())])
        self.assertEqual(len(tasks), 20)
        self.assertEqual(literal(self.nb, "RUN_METHODS"), ["cryozeta"])
        self.assertEqual(literal(self.nb, "MAX_LANES"), 2)

    def test_the_embedded_files_are_the_common_ones_and_the_runner(self):
        got = {}
        for c in self.nb.cells:
            if c.source.startswith("%%writefile"):
                first, text = c.source.split("\n", 1)
                got[first.split(None, 1)[1].strip()] = text
        self.assertEqual(sorted(got), sorted([t for _s, t in M.BASE_FILES] + ["scripts/cryozeta_tools.py"]))
        for target, text in got.items():
            with open(os.path.join(ROOT, target), encoding="utf-8") as f:
                want = f.read()
            self.assertEqual(text, want if want.endswith("\n") else want + "\n", target)
            self.assertEqual(self.expected[target], hashlib.sha256(text.encode()).hexdigest())

    def test_the_other_methods_are_not_in_the_notebook(self):
        code = "\n".join(c.source for c in self.nb.cells if c.cell_type == "code" and not c.source.startswith("%%writefile"))
        for name in ("setup_modelangelo", "plan_emprot", "plan_e3", "setup_emodelx", "model_angelo", "patch_cryoatom_tempfile", "setup_cryoatom"):
            self.assertNotIn(name, code)

    def test_the_run_goes_through_the_runner_of_the_repository_and_the_programs_of_the_demo(self):
        code = self.cell("cryozeta")
        self.assertIn('cmd = [sys.executable, os.path.join(WORK, "scripts", "cryozeta_tools.py"), "run", cfg_path]', code)
        self.assertIn('SETUPS["cryozeta"], PLANS["cryozeta"] = setup_cryozeta, plan_cryozeta', code)
        self.assertIn('"-e", name, "--frozen"', code)                                         # the environment exactly as the lock file says
        for word in ("cryozeta-detection", "cryozeta-inference", "cryozeta-combine"):
            self.assertIn(word, self.cell("cryozeta-title") + code)
        self.assertIn("cryozeta_source_patch.diff", code)

    def test_it_cannot_be_mixed_with_the_other_methods(self):
        for other in ("e3", "cryoatom1"):
            with self.assertRaises(ValueError):
                M.build(os.path.join(self.tmp, "x.ipynb"), run_methods=["cryozeta", other])
        self.assertEqual(M.ORDER, ["modelangelo", "e3", "emprot", "emodelx"])
        self.assertNotIn("cryozeta", M.SPECS)

    def test_the_command_line_gives_eleven_hours_by_default(self):
        out = os.path.join(self.tmp, "cli.ipynb")
        self.assertEqual(M.main(["--batch", "cz_cli", "--methods", "cryozeta", "--out", out]), 0)
        nb = nbformat.read(out, as_version=4)
        self.assertRegex(next(c for c in nb.cells if c.cell_type == "code").source, r"(?m)^HOURS = 11\.0$")
        self.assertEqual(len(literal(nb, "TASKS")), 20)

    def test_the_committed_notebook_is_current(self):
        if not (os.path.exists(COMMITTED_CRYOZETA) and os.path.exists(CZ_B_STATUS) and os.path.exists(CZ_C_STATUS)):
            self.skipTest("gpu_cryozeta_d.ipynb or the status tables of batches b and c are not there")
        fresh = os.path.join(self.tmp, "fresh.ipynb")
        self.assertEqual(M.main(CZ_D_ARGS + ["--done", CZ_B_STATUS, CZ_C_STATUS, "--out", fresh]), 0)
        a, b = nbformat.read(fresh, as_version=4), nbformat.read(COMMITTED_CRYOZETA, as_version=4)
        self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_main_notebook.py " + " ".join(CZ_D_ARGS) + " --done <the status tables of batches b and c> again")

    def test_a_continuation_has_its_order_its_rate_and_its_first_task_alone(self):
        status = os.path.join(self.tmp, "b_status.csv")
        with open(status, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["method", "entry", "status", "seconds"])
            w.writerow(["cryozeta", "22FX", "ok", "692"])                    # 173 residues: 4.0 s per residue
            w.writerow(["cryozeta", "9T5K", "ok", "3810"])                   # 381 residues: 10.0 s per residue
            w.writerow(["cryozeta", "9WQ6", "failed", "10"])                 # a failure is not a rate
            w.writerow(["cryozeta", "9WME", "interrupted", "17576"])         # nor is a run that the end of the session cut
        out = os.path.join(self.tmp, "cont.ipynb")
        self.assertEqual(M.main(["--batch", "cz_cont", "--methods", "cryozeta", "--entries", "26DK", "9WQ6", "22FX", "9T5K", "--done", status, "--redo", "9WQ6:cryozeta",
                                 "--alone-first", "26DK:cryozeta", "--out", out]), 0)
        nb = nbformat.read(out, as_version=4)
        self.assertEqual(literal(nb, "TASKS"), [["26DK", "cryozeta"], ["9WQ6", "cryozeta"]])      # 22FX and 9T5K are done; the failure of 9WQ6 is run again because of --redo
        self.assertEqual(literal(nb, "ALONE_FIRST"), [["26DK", "cryozeta"]])
        rate = literal(nb, "SPECS")["cryozeta"]["s_per_residue"]
        self.assertEqual(rate, round((692 + 3810) / (173 + 381)))                                 # all seconds over all residues, rounded: 8
        intro = nb.cells[0].source
        self.assertIn("The earlier session timed 2 runs of CryoZeta on this GPU: 4 to 10 seconds per residue", intro)
        self.assertIn("The first estimate is %d;" % rate, intro)
        self.assertIn("in an order that was written down before the session (LOG.md)", intro)
        self.assertNotIn("No run of CryoZeta has been timed", intro)
        self.assertEqual(M.measured_rates([status], "cryozeta", {"22FX": 173, "9T5K": 381, "9WQ6": 261, "9WME": 446})["n"], 2)
        self.assertIsNone(M.measured_rates([status], "e3", {"22FX": 173}))


class StartCheckTests(unittest.TestCase):
    """The check of the internet at the start of a session (probe_site and check_internet of the step 1 cell), run on its own source text against a fake `urlopen` and a
    fake `time.sleep`. Batch d of CryoZeta stopped after 25 seconds on 7 November 2026 because the check asked every site once and one TLS handshake of the EBI server
    timed out: the whole session was skipped. A site that does not answer is now asked again (4 rounds, 20, 40 and 60 s apart) before the session is stopped."""
    EBI, PYPI, COLAB = "https://ftp.ebi.ac.uk/x.xml", "https://pypi.org/simple/uv/", "https://api.colabfold.com/"
    HOSTS = (("EBI FTP", EBI, True), ("PyPI", PYPI, True), ("ColabFold server", COLAB, False))
    TLS = OSError("<urlopen error _ssl.c:1015: The handshake operation timed out>")

    @classmethod
    def setUpClass(cls):
        import ast
        tmp = tempfile.mkdtemp(prefix="czcheck_")
        try:
            path = os.path.join(tmp, "nb.ipynb")
            M.build(path, batch="cryozeta_test", hours=11.0, run_methods=["cryozeta"])
            nb = nbformat.read(path, as_version=4)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        cls.src = segment(nb, "step1", ast.FunctionDef, "probe_site") + "\n\n" + segment(nb, "step1", ast.FunctionDef, "check_internet")

    def run_check(self, answers, hosts=None):
        """`answers` maps a url to its outcomes in the order of the calls (the last one repeats): an exception is raised, 200 answers, any other number is an HTTP error."""
        import types
        import urllib.error
        from unittest import mock
        queue, asked, slept, lines = {u: list(o) for u, o in answers.items()}, [], [], []

        class Reply:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, n=-1):
                return b"ok"

        def fake_urlopen(req, timeout=None):
            asked.append((req.full_url, timeout))
            outcomes = queue[req.full_url]
            outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
            if isinstance(outcome, Exception):
                raise outcome
            if outcome == 200:
                return Reply()
            raise urllib.error.HTTPError(req.full_url, outcome, "stub", {}, io.BytesIO(b""))

        ns = {"NET_HOSTS": hosts or self.HOSTS, "log": lines.append, "time": types.SimpleNamespace(sleep=slept.append)}
        exec(compile(self.src, "start-check", "exec"), ns)
        with mock.patch("urllib.request.urlopen", fake_urlopen):
            missing = ns["check_internet"]()
        return missing, asked, slept, lines

    def test_a_tls_timeout_of_one_site_is_asked_again_and_the_session_goes_on(self):
        missing, asked, slept, lines = self.run_check({self.EBI: [self.TLS, 200], self.PYPI: [200], self.COLAB: [404]})
        self.assertEqual(missing, [])
        self.assertEqual([u for u, t in asked], [self.EBI, self.PYPI, self.COLAB, self.EBI])      # only the site that failed is asked again
        self.assertEqual(slept, [20])
        text = "\n".join(lines)
        self.assertIn("no answer (try 1 of 4): EBI FTP", text)
        self.assertRegex(text, r"reachable: EBI FTP\s+HTTP 200 \(at try 2\)")
        self.assertNotIn("NOT reachable", text)
        self.assertTrue(all(t >= 30 for u, t in asked), asked)                                    # the 20 s of the first version were too short for a slow handshake

    def test_an_http_answer_below_500_means_that_the_way_is_open(self):
        missing, asked, slept, lines = self.run_check({self.EBI: [200], self.PYPI: [200], self.COLAB: [404]})                 # the ColabFold root address answers 404
        self.assertEqual((missing, slept, len(asked)), ([], [], 3))
        self.assertRegex("\n".join(lines), r"reachable: ColabFold server\s+HTTP 404")
        missing, asked, slept, lines = self.run_check({self.EBI: [403], self.PYPI: [200], self.COLAB: [200]})
        self.assertEqual((missing, slept, len(asked)), ([], [], 3))

    def test_a_server_error_is_asked_again(self):
        missing, asked, slept, lines = self.run_check({self.EBI: [503, 503, 200], self.PYPI: [200], self.COLAB: [200]})
        self.assertEqual(missing, [])
        self.assertEqual(slept, [20, 40])
        self.assertEqual(sum(1 for u, t in asked if u == self.EBI), 3)

    def test_a_needed_site_that_never_answers_stops_the_session_after_two_minutes(self):
        missing, asked, slept, lines = self.run_check({self.EBI: [self.TLS], self.PYPI: [200], self.COLAB: [200]})
        self.assertEqual(missing, ["EBI FTP"])
        self.assertEqual(sum(1 for u, t in asked if u == self.EBI), 4)
        self.assertEqual(sum(1 for u, t in asked if u == self.PYPI), 1)
        self.assertEqual(slept, [20, 40, 60])
        self.assertIn("NOT reachable: EBI FTP", "\n".join(lines))
        self.assertIn("needed: the notebook stops", "\n".join(lines))

    def test_an_optional_site_that_never_answers_does_not_stop_the_session(self):
        missing, asked, slept, lines = self.run_check({self.EBI: [200], self.PYPI: [200], self.COLAB: [self.TLS]})
        self.assertEqual(missing, [])
        self.assertIn("optional: the notebook goes on", "\n".join(lines))

    def test_with_the_internet_off_every_needed_site_is_named_and_the_wait_is_short(self):
        off = OSError("[Errno -3] Temporary failure in name resolution")
        missing, asked, slept, lines = self.run_check({self.EBI: [off], self.PYPI: [off], self.COLAB: [off]})
        self.assertEqual(missing, ["EBI FTP", "PyPI"])
        self.assertEqual(slept, [20, 40, 60])                                                      # about two minutes of waiting, then the session stops before any hour is spent


class CryoZetaLogicTests(unittest.TestCase):
    """The functions of the CryoZeta cell, run on their own source text taken from the built notebook against fakes of the tools (no network, no GPU)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="czlogic_")
        path = os.path.join(cls.tmp, "nb.ipynb")
        M.build(path, batch="cryozeta_test", hours=11.0, run_methods=["cryozeta"])
        cls.nb = nbformat.read(path, as_version=4)
        cls.spec = M.cryozeta_spec()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.work = tempfile.mkdtemp(dir=self.tmp)

    def ns(self, *names, cell="cryozeta", **extra):
        import ast
        import glob
        ns = {"os": os, "json": json, "sys": sys, "shutil": shutil, "glob": glob, "log": lambda *a, **k: None, "BIG": self.work, "WORK": self.work, "sha256_file": TZ_sha256}
        ns.update(extra)
        for name in names:
            exec(segment(self.nb, cell, ast.FunctionDef, name), ns)
        return ns

    def test_pixi_is_checked_against_the_hash_of_the_release(self):
        data = b"p" * 100

        def fetch(url, dest, min_bytes, label, **kw):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            return True
        spec = {"pixi": dict(self.spec["pixi"], min_bytes=10, sha256=hashlib.sha256(data).hexdigest())}
        ns = self.ns("install_pixi", fetch=fetch, pixi_vars=lambda: {}, run=lambda cmd, **kw: (0, ["pixi 0.41.1"], False))
        exe = ns["install_pixi"](spec)
        self.assertTrue(os.access(exe, os.X_OK))
        shutil.rmtree(os.path.join(self.work, "pixi"))
        spec["pixi"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "does not have the sha256"):
            ns["install_pixi"](spec)
        self.assertFalse(os.path.exists(exe))

    def test_the_environment_is_installed_from_the_lock_file_and_a_source_without_it_stops(self):
        repo = os.path.join(self.work, "repo")
        os.makedirs(repo)
        for f in ("pixi.lock", "pyproject.toml", "env_setup.sh"):
            with open(os.path.join(repo, f), "w") as h:
                h.write("x")
        env_py = os.path.join(repo, ".pixi", "envs", "cu11", "bin", "python")
        seen = []

        def run_watched(cmd, label, **kw):
            seen.append((cmd, kw))
            os.makedirs(os.path.dirname(env_py), exist_ok=True)
            open(env_py, "w").close()
            return 0, [], False

        def run(cmd, **kw):
            seen.append((cmd, kw))
            return 0, ["ok"], False
        ns = self.ns("make_env_cryozeta", install_pixi=lambda spec: "/x/pixi", pixi_vars=lambda: {"PIXI_HOME": "/h"}, run_watched=run_watched, run=run, env_vars=lambda p, e=None: {}, IMPORT_TESTS=None,
                     CZ_PACKAGE_LIST="# CZ_PACKAGE_LIST_MARK")
        self.assertEqual(ns["make_env_cryozeta"](self.spec, repo), env_py)
        cmd, kw = seen[0]
        self.assertEqual(cmd, ["/x/pixi", "install", "-e", "cu11", "--frozen"])
        self.assertEqual((kw["cwd"], kw["env"]), (repo, {"PIXI_HOME": "/h"}))
        self.assertEqual(seen[-1][0][:2], [env_py, "-c"])                                    # the modules are imported with the program of the new environment
        listing = [c for c, _ in seen if c[0] == env_py and "CZ_PACKAGE_LIST_MARK" in c[2]]                 # the versions are listed by the program of the new environment, not by pixi
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0][3], os.path.join(self.work, "cryozeta_env_packages.txt"))
        os.remove(os.path.join(repo, "pixi.lock"))
        shutil.rmtree(os.path.join(repo, ".pixi"))
        with self.assertRaisesRegex(RuntimeError, "lacks pixi.lock"):
            ns["make_env_cryozeta"](self.spec, repo)

    def teaser_ns(self, makes_module=True):
        calls = []

        def git_checkout(repo, commit, folder):
            calls.append(("git", repo, commit, folder))
            os.makedirs(os.path.join(folder, "python", "teaserpp_python"), exist_ok=True)
            with open(os.path.join(folder, "python", "teaserpp_python", "__init__.py"), "w") as f:
                f.write("# init\n")
            return True

        def run(cmd, **kw):
            calls.append(("pip", cmd))
            tools = cmd[cmd.index("--target") + 1]
            os.makedirs(os.path.join(tools, "cmake", "data", "bin"), exist_ok=True)
            open(os.path.join(tools, "cmake", "data", "bin", "cmake"), "w").close()
            os.makedirs(os.path.join(tools, "pybind11", "share", "cmake", "pybind11"), exist_ok=True)
            return 0, [], False

        def run_watched(cmd, label, **kw):
            calls.append(("watched", cmd, kw))
            if "--build" in cmd and makes_module:
                pkg = os.path.join(cmd[cmd.index("--build") + 1], "python", "teaserpp_python")
                os.makedirs(pkg, exist_ok=True)
                open(os.path.join(pkg, "_teaserpp.cpython-312-x86_64-linux-gnu.so"), "w").close()
            return 0, [], False
        return self.ns("build_teaser", git_checkout=git_checkout, run=run, run_watched=run_watched), calls

    def test_teaser_is_built_at_the_pinned_commit_with_pinned_tools_and_the_headers_of_the_environment(self):
        ns, calls = self.teaser_ns()
        prefix = os.path.join(self.work, "env")
        root = ns["build_teaser"](self.spec, prefix)
        self.assertEqual(root, os.path.join(self.work, "teaser_plusplus"))
        self.assertEqual(calls[0][:3], ("git", "https://github.com/MIT-SPARK/TEASER-plusplus", "52a9c52ee7d4c838c5e8a75458c33178be5bfb70"))
        pip = next(c[1] for c in calls if c[0] == "pip")
        self.assertIn("cmake==3.31.6", pip)
        self.assertIn("pybind11==3.0.1", pip)
        configure, build = [c for c in calls if c[0] == "watched"]
        for flag in ("-DBUILD_PYTHON_BINDINGS=ON", "-DBUILD_TESTS=OFF", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_PREFIX_PATH=" + prefix, "-DPYTHON_EXECUTABLE=" + os.path.join(prefix, "bin", "python"),
                     "-Dpybind11_DIR=" + os.path.join(self.work, "teaser_tools", "pybind11", "share", "cmake", "pybind11")):
            self.assertIn(flag, configure[1])
        self.assertEqual(configure[1][0], os.path.join(self.work, "teaser_tools", "cmake", "data", "bin", "cmake"))
        self.assertIn("--build", build[1])
        self.assertEqual(configure[2]["env"], {"CC": None, "CXX": None, "PYTHONPATH": None})
        self.assertTrue(os.path.exists(os.path.join(root, "build", "python", "teaserpp_python", "__init__.py")))     # as env_setup.sh does
        ns["build_teaser"](self.spec, prefix)                                                 # the module is there: nothing is built again
        self.assertEqual(len([c for c in calls if c[0] == "watched"]), 2)

    def test_a_teaser_build_without_the_python_module_stops_the_setup(self):
        ns, _calls = self.teaser_ns(makes_module=False)
        with self.assertRaisesRegex(RuntimeError, "did not make the Python module"):
            ns["build_teaser"](self.spec, os.path.join(self.work, "env"))

    def test_the_conda_environment_gets_its_library_path_but_the_runner_does_not(self):
        import ast
        ns = {"os": os, "TORCH_HOME": "/t"}
        exec(segment(self.nb, "tools", ast.FunctionDef, "env_vars"), ns)
        ns["_env_vars_plain"] = ns["env_vars"]
        exec(segment(self.nb, "cryozeta", ast.FunctionDef, "env_vars"), ns)
        conda, venv = os.path.join(self.work, "conda"), os.path.join(self.work, "venv")
        os.makedirs(os.path.join(conda, "conda-meta"))
        os.makedirs(os.path.join(venv, "bin"))
        e = ns["env_vars"](os.path.join(conda, "bin", "python"))
        self.assertEqual(e["CONDA_PREFIX"], conda)
        self.assertTrue(e["LD_LIBRARY_PATH"].startswith(os.path.join(conda, "lib")))
        e = ns["env_vars"](os.path.join(venv, "bin", "python"))
        self.assertNotIn("CONDA_PREFIX", e)                                                   # the environments of the other methods are what they were
        self.assertNotIn("LD_LIBRARY_PATH", e)
        self.assertNotIn("LD_LIBRARY_PATH", ns["_env_vars_plain"](os.path.join(conda, "bin", "python")))

    def test_cz_env_is_what_the_activation_script_of_the_project_sets(self):
        ns = self.ns("cz_env", env_vars=lambda p, extra=None: dict(extra or {}, PYTHONPATH=None))
        e = ns["cz_env"]("/prefix", "/teaser")
        self.assertEqual((e["CUTLASS_PATH"], e["USE_OPM_CHUNKED"], e["CUDA_DEVICE_ORDER"]), ("/prefix", "1", "PCI_BUS_ID"))
        self.assertEqual((e["TEASERPP_DIR"], e["PYTHONPATH"]), ("/teaser", "/teaser/build/python"))

    def preflight(self, probe, rc_import=0, rc_data=0):
        """Runs cryozeta_preflight against a git repository with the detection file of CryoZeta and a fake GPU test; returns (namespace, repo, error)."""
        import cryozeta_tools as CZT
        repo = os.path.join(self.work, "repo")
        path = os.path.join(repo, CZT.DETECTION_FILE)
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            f.write("class D:\n    def load(self):\n        %s\n        %s\n" % (CZT.DETECTION_CHANGES[0][0], CZT.DETECTION_CHANGES[1][0]))
        T.sh("git init -q -b main . && git add -A && git -c user.name=t -c user.email=t@t commit -q -m stub", cwd=repo)

        def run(cmd, **kw):
            if "CZ_PREFLIGHT_MARK" in cmd[2]:
                return 0, [json.dumps(probe)] if probe is not None else ["no json"], False
            if "CZ_DATA_CHECK_MARK" in cmd[2]:
                calls.append(kw)
                return rc_data, [], False
            return rc_import, [], False
        sh = lambda cmd, timeout=60: subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        calls = self.data_check_calls = []
        ns = self.ns("cryozeta_preflight", CZT=CZT, CZ={"teaser": "/t", "assets": "/a"}, STATE={"methods": {"cryozeta": {}}}, REQUIRE_GPU=True, ENV_PY_GIVEN=None, run=run, sh=sh,
                     cz_env=lambda prefix, teaser=None, assets=None: {"ASSETS": assets}, CZ_PROGRAMS=(), CZ_PREFLIGHT="# CZ_PREFLIGHT_MARK", CZ_DATA_CHECK="# CZ_DATA_CHECK_MARK")
        return ns, repo

    GOOD = {"cuda_available": True, "gpu": "Tesla T4", "bf16_matmul": "ok", "bf16_conv3d": "ok", "bf16_autocast_conv3d": "ok"}

    def test_a_gpu_that_does_bfloat16_keeps_the_source_as_it_is(self):
        ns, repo = self.preflight(self.GOOD)
        ns["cryozeta_preflight"](self.spec, repo, "/env/bin/python", "/env")
        self.assertEqual(ns["CZ"]["dtype"], "bf16")
        self.assertEqual(T.sh("git status --porcelain", cwd=repo), "")
        self.assertFalse(ns["STATE"]["methods"]["cryozeta"]["preflight"]["detection_patched"])

    def test_a_gpu_that_cannot_do_bfloat16_in_the_detection_network_gets_the_two_line_change(self):
        ns, repo = self.preflight(dict(self.GOOD, bf16_conv3d="failed: no kernel"))
        ns["cryozeta_preflight"](self.spec, repo, "/env/bin/python", "/env")
        self.assertEqual(ns["CZ"]["dtype"], "bf16")                                           # only the detection network is changed
        diff = T.sh("git diff", cwd=repo)
        self.assertEqual(diff.count("\n+ ") + diff.count("\n- "), 4, diff)
        self.assertIn("torch.float32", diff)
        self.assertTrue(ns["STATE"]["methods"]["cryozeta"]["preflight"]["detection_patched"])
        with open(os.path.join(self.work, "cryozeta_source_patch.diff"), encoding="utf-8") as f:
            self.assertIn("torch.float32", f.read())

    def test_a_gpu_without_bfloat16_matrix_products_starts_in_float32(self):
        ns, repo = self.preflight(dict(self.GOOD, bf16_matmul="failed: not implemented", bf16_conv3d="failed", bf16_autocast_conv3d="failed"))
        ns["cryozeta_preflight"](self.spec, repo, "/env/bin/python", "/env")
        self.assertEqual(ns["CZ"]["dtype"], "fp32")

    def test_no_gpu_stops_the_setup_before_any_hour_is_spent(self):
        ns, repo = self.preflight(dict(self.GOOD, cuda_available=False))
        with self.assertRaisesRegex(RuntimeError, "does not see the GPU"):
            ns["cryozeta_preflight"](self.spec, repo, "/env/bin/python", "/env")

    def test_modules_that_do_not_import_together_with_teaser_stop_the_setup(self):
        ns, repo = self.preflight(self.GOOD, rc_import=1)
        with self.assertRaisesRegex(RuntimeError, "do not import together with TEASER"):
            ns["cryozeta_preflight"](self.spec, repo, "/env/bin/python", "/env")

    def test_the_chemical_components_files_are_read_with_the_environment_that_names_their_folder(self):
        ns, repo = self.preflight(self.GOOD)
        ns["cryozeta_preflight"](self.spec, repo, "/env/bin/python", "/env")
        self.assertEqual(len(self.data_check_calls), 1)
        self.assertEqual(self.data_check_calls[0]["env"], {"ASSETS": "/a"})
        self.assertGreaterEqual(self.data_check_calls[0]["timeout"], 600)

    def test_components_files_that_the_package_cannot_read_stop_the_setup(self):
        ns, repo = self.preflight(self.GOOD, rc_data=1)
        with self.assertRaisesRegex(RuntimeError, "cannot read its chemical-components files in /a"):
            ns["cryozeta_preflight"](self.spec, repo, "/env/bin/python", "/env")

    def test_the_environment_of_the_programs_names_the_assets_folder_when_there_is_one(self):
        ns = self.ns("cz_env", env_vars=lambda py, extra=None: dict(extra or {}))
        self.assertEqual(ns["cz_env"]("/prefix", "/teaser", "/assets")["CRYOZETA_ASSETS_DIR"], "/assets")
        self.assertNotIn("CRYOZETA_ASSETS_DIR", ns["cz_env"]("/prefix", "/teaser"))

    def run_data_check(self, files=("components.v20240608.cif", "components.v20240608.cif.rdkit_mol.pkl"), atoms=3, mol=True):
        """The program text of the data check of the setup against a fake `cryozeta` package whose configuration points into a temporary folder; returns (exit code, output)."""
        code = literal(self.nb, "CZ_DATA_CHECK")
        pkg = os.path.join(self.work, "fakepkg")
        assets = os.path.join(self.work, "fakeassets")
        os.makedirs(assets, exist_ok=True)
        for n in files:
            with open(os.path.join(assets, n), "w") as f:
                f.write("x")
        for rel, text in {"cryozeta/__init__.py": "", "cryozeta/configs/__init__.py": "", "cryozeta/data/__init__.py": "",
                          "cryozeta/configs/configs_data.py": "data_configs = {'ccd_components_file': %r, 'ccd_components_rdkit_mol_file': %r}\n" % (
                              os.path.join(assets, "components.v20240608.cif"), os.path.join(assets, "components.v20240608.cif.rdkit_mol.pkl")),
                          "cryozeta/data/ccd.py": "def get_component_atom_array(code, keep_leaving_atoms=False, keep_hydrogens=False):\n    assert code == 'ALA' and keep_leaving_atoms and not keep_hydrogens\n"
                                                  "    return [1] * %d if %d else None\n\n\ndef get_component_rdkit_mol(code):\n    return object() if %r else None\n" % (atoms, atoms, mol)}.items():
            path = os.path.join(pkg, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        r = subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": pkg, "PATH": os.environ.get("PATH", "")}, capture_output=True, text=True, cwd=self.work)
        shutil.rmtree(pkg, ignore_errors=True)
        shutil.rmtree(assets, ignore_errors=True)
        return r.returncode, r.stdout + r.stderr

    def test_the_package_list_program_writes_the_python_and_the_conda_packages(self):
        out = os.path.join(self.work, "packages.txt")
        r = subprocess.run([sys.executable, "-c", literal(self.nb, "CZ_PACKAGE_LIST"), out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(text.startswith("# python packages (importlib.metadata)\n"))
        self.assertIn("# conda packages (conda-meta)\n", text)
        self.assertRegex(text, r"(?m)^[A-Za-z0-9_.-]+==[0-9][^\s]*$")

    def test_the_data_check_passes_when_the_package_reads_both_files(self):
        self.assertIn("get_component_atom_array", literal(self.nb, "CZ_DATA_CHECK"))
        rc, out = self.run_data_check()
        self.assertEqual(rc, 0, out)
        self.assertIn("the package reads its chemical-components files", out)

    def test_the_data_check_names_a_file_that_is_not_there(self):
        for missing in ("components.v20240608.cif", "components.v20240608.cif.rdkit_mol.pkl"):
            rc, out = self.run_data_check(files=[n for n in ("components.v20240608.cif", "components.v20240608.cif.rdkit_mol.pkl") if n != missing])
            self.assertNotEqual(rc, 0)
            self.assertIn("the package looks for files that are not there", out)
            self.assertIn(missing, out)

    def test_the_data_check_stops_on_a_components_file_without_alanine_or_a_missing_mol(self):
        rc, out = self.run_data_check(atoms=0)
        self.assertNotEqual(rc, 0)
        self.assertIn("no entry for ALA", out)
        rc, out = self.run_data_check(mol=False)
        self.assertNotEqual(rc, 0)
        self.assertIn("the RDKit file has no entry for ALA", out)

    def run_probe(self, available=True, failing=()):
        """Runs the GPU probe of the setup (a program text that the notebook gives to the python of the CryoZeta environment) against a fake torch."""
        import contextlib
        import types
        code = literal(self.nb, "CZ_PREFLIGHT")
        fake = types.ModuleType("torch")
        fake.__version__, fake.bfloat16, fake.version = "2.4.1", "bf16", types.SimpleNamespace(cuda="11.8")

        class Tensor:
            def __init__(self, bf=False):
                self.bf = bf

            def __matmul__(self, other):
                if "matmul" in failing:
                    raise RuntimeError("no bf16 gemm\nsecond line")
                return self

            def to(self, dtype):
                return Tensor(dtype == "bf16")

        class Conv:
            bf = False

            def __init__(self, *a, **k):
                pass

            def cuda(self):
                return self

            def to(self, dtype):
                c = Conv()
                c.bf = dtype == "bf16"
                return c

            def __call__(self, x):
                if self.bf and "conv" in failing:
                    raise RuntimeError("bf16 conv3d is not supported")
                return x

        class Autocast:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                if "autocast" in failing:
                    raise RuntimeError("autocast to bf16 is not supported")

            def __exit__(self, *a):
                return False
        fake.cuda = types.SimpleNamespace(is_available=lambda: available, synchronize=lambda: None,
                                          get_device_properties=lambda i: types.SimpleNamespace(name="Tesla T4", total_memory=15 * 2 ** 30, major=7, minor=5))
        fake.randn = lambda *a, **k: Tensor(k.get("dtype") == "bf16")
        fake.nn, fake.autocast = types.SimpleNamespace(Conv3d=Conv), Autocast
        sys.modules["torch"] = fake
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "probe", "exec"), {})
        finally:
            del sys.modules["torch"]
        return json.loads(buf.getvalue().strip().splitlines()[-1])

    def test_the_gpu_probe_reports_what_this_gpu_can_do_in_bfloat16(self):
        out = self.run_probe()
        self.assertEqual((out["gpu"], out["memory_gb"], out["capability"], out["cuda_available"]), ("Tesla T4", 15.0, "7.5", True))
        self.assertEqual((out["bf16_matmul"], out["bf16_conv3d"], out["bf16_autocast_conv3d"]), ("ok", "ok", "ok"))
        out = self.run_probe(failing=("matmul", "conv", "autocast"))
        self.assertEqual((out["bf16_matmul"], out["bf16_conv3d"], out["bf16_autocast_conv3d"]), ("failed: no bf16 gemm", "failed: bf16 conv3d is not supported", "failed: autocast to bf16 is not supported"))
        out = self.run_probe(available=False)
        self.assertEqual((out["cuda_available"], "gpu" in out, "bf16_matmul" in out), (False, False, False))

    def test_the_plan_is_one_run_of_the_runner_with_a_settings_file(self):
        import cryozeta_tools as CZT
        ns = self.ns("plan_cryozeta", "cz_prefix", CZT=CZT, CZ={"teaser": "/teaser", "assets": "/assets", "dtype": "fp32"}, RUN_ROOT=os.path.join(self.work, "runs"),
                     INPUT_ROOT=os.path.join(self.work, "input"), _env_vars_plain=lambda p: {"PATH": "x"})
        ctx = {"entry": "9T5K", "row": {"resolution_A": "2.6", "map_url": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-55586/map/emd_55586.map.gz"}, "map": "/m/emd_55586.mrc",
               "jobs": [{"sequence_submitted": "MKT", "copies_in_entry": "2"}, {"sequence_submitted": " AAA\n", "copies_in_entry": "1"}]}
        runs = ns["plan_cryozeta"](self.spec, ctx, "/repo", "/env/bin/python")
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertEqual((r["variant"], r["main"], r["cwd"], r["env"]), ("run", "cryozeta", "/repo", {"PATH": "x"}))      # the runner does not get the library path of the conda environment
        self.assertEqual((r["cmd"][0], r["cmd"][2]), (sys.executable, "run"))
        self.assertTrue(r["cmd"][1].endswith(os.path.join("scripts", "cryozeta_tools.py")))
        with open(r["cmd"][3]) as f:
            cfg = json.load(f)
        self.assertEqual((cfg["name"], cfg["env_dir"], cfg["teaser_dir"], cfg["assets"], cfg["dtype"], cfg["kernels"]), ("9T5K", "/env", "/teaser", "/assets", "fp32", "off"))
        self.assertEqual((cfg["seed"], cfg["n_sample"], cfg["n_step"], cfg["n_cycle"], cfg["num_select"]), (101, 5, 20, 10, 5))
        self.assertEqual(cfg["header_url"], "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-55586/header/emd-55586.xml")
        self.assertEqual((cfg["resolution"], cfg["map_path"], cfg["colabfold_host"]), (2.6, "/m/emd_55586.mrc", "https://api.colabfold.com"))
        self.assertEqual(cfg["chains"], [{"sequence": "MKT", "count": 2}, {"sequence": "AAA", "count": 1}])
        self.assertEqual(list(r["outputs"]), ["cryozeta", "cryozeta_s1", "cryozeta_s2", "cryozeta_s3", "cryozeta_s4"])
        self.assertTrue(r["outputs"]["cryozeta"].endswith(os.path.join("9T5K", "9T5K", "CryoZeta-Final", "9T5K_sample_0.cif")))
        self.assertEqual(r["watch"], [cfg["progress_path"]])


def TZ_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@unittest.skipUnless(ENV_PY and USALIGN, "needs MICA_TEST_ENV_PYTHON (numpy, mrcfile, biopython) and USALIGN; the AlphaFold files are not needed")
class CryoZetaEndToEndTests(unittest.TestCase):
    """The whole notebook of CryoZeta as a script against stand-ins: a local git repository (the source), stand-in programs (the three programs of CryoZeta), a local Hugging Face (the
    file list and the weights), a local ColabFold server, the EMDB headers, the maps and the deposited model from a file server, and the real US-align."""
    ENTRIES = ["9T5K", "9WQ6", "22FX", "9V4D"]
    BLOBS = {"cryozeta-detection-v0.0.1.safetensors": 4200 * 256, "cryozeta-v0.0.1.safetensors": 5000 * 256, "cryozeta-interpolate-v0.0.1.safetensors": 4400 * 256,
             "components.v20240608.cif": 4100 * 256, "components.v20240608.cif.rdkit_mol.pkl": 4300 * 256}      # 1 MB or more: recorded
    REV = "0123456789abcdef0123456789abcdef01234567"

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="czrun_")
        cls.cif = os.path.join(cls.root, "model.cif")
        subprocess.run([ENV_PY, "-c", SYNTH_MODEL, cls.cif], check=True)
        with open(cls.cif, "rb") as f:
            cif_bytes = f.read()
        mapfile = os.path.join(cls.root, "stub.map")
        subprocess.run([ENV_PY, "-c", "import mrcfile, numpy; m = mrcfile.new(%r, overwrite=True); m.set_data(numpy.random.rand(32, 32, 32).astype('float32')); "
                        "m.voxel_size = 1.0; m.close()" % mapfile], check=True)
        with open(mapfile, "rb") as f:
            map_gz = gzip.compress(f.read())
        cls.srv, cls.url = T.serve(FlakyFiles)
        cls.msa = TZ.Server(statuses=("COMPLETE",), result_for=TZ.msa_archive)
        cls.blobs = {n: bytes((i * (k + 3)) % 251 for i in range(256)) * (size // 256) for k, (n, size) in enumerate(cls.BLOBS.items())}
        cls.api = {"id": "KiharaLab/CryoZeta", "sha": cls.REV, "lastModified": "2026-09-01T00:00:00.000Z",
                   "siblings": [{"rfilename": ".gitattributes", "size": 100}] + [{"rfilename": n, "size": len(b), "lfs": {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b), "pointerSize": 134}}
                                                                                     for n, b in cls.blobs.items()]}
        routes = {"/cif/stub.cif": cif_bytes, "/hub/api/models/KiharaLab/CryoZeta": json.dumps(cls.api).encode()}
        routes.update({"/hub/resolve/%s/%s" % (cls.REV, n): b for n, b in cls.blobs.items()})
        cls.entries = os.path.join(cls.root, "entries.csv")
        with open(os.path.join(ROOT, "data", "entries_2026.csv"), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows, fields = list(reader), reader.fieldnames
        cls.rows = {r["pdb_id"]: r for r in rows}
        with open(os.path.join(ROOT, "data", "af3", "af3_jobs.csv"), newline="", encoding="utf-8") as f:
            cls.jobs = list(csv.DictReader(f))
        for r in rows:
            number = r["emdb_id"].split("-")[-1]
            r["map_url"], r["cif_url"] = "%s/emdb/EMD-%s/map/emd_%s.map.gz" % (cls.url, number, number), cls.url + "/cif/stub.cif"
            routes["/emdb/EMD-%s/map/emd_%s.map.gz" % (number, number)] = map_gz
            routes["/emdb/EMD-%s/header/emd-%s.xml" % (number, number)] = TZ.HEADER.encode()
        cls.routes = routes
        with open(cls.entries, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        cls.repo = os.path.join(cls.root, "cryozeta_stub")
        for rel, text in {"pyproject.toml": "[project]\nname = 'cryozeta'\n", "pixi.lock": "version: 6\n", "env_setup.sh": "#!/bin/sh\n", "src/cryozeta/__init__.py": ""}.items():
            path = os.path.join(cls.repo, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        T.sh("git init -q -b main . && git add -A && git -c user.name=t -c user.email=t@t commit -q -m stub", cwd=cls.repo)
        cls.commit = T.sh("git rev-parse HEAD", cwd=cls.repo)
        cls.teaser = os.path.join(cls.root, "teaser")
        os.makedirs(cls.teaser)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.msa.close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def make_env(self, scenario=None, drop=()):
        env = tempfile.mkdtemp(prefix="czenv_", dir=self.root)
        os.makedirs(os.path.join(env, "bin"))
        os.makedirs(os.path.join(env, "conda-meta"))
        os.symlink(sys.executable, os.path.join(env, "bin", "python"))
        for prog in [p for p in ("cryozeta-detection", "cryozeta-inference", "cryozeta-combine") if p not in drop]:
            path = os.path.join(env, "bin", prog)
            with open(path, "w") as f:
                f.write(STUB_CRYOZETA)
            os.chmod(path, 0o755)
        sc = {"detection": ["ok"], "cryozeta": ["ok"], "interpolate": ["ok"], "combine": ["ok"]}
        sc.update(scenario or {})
        with open(os.path.join(env, "scenario.json"), "w") as f:
            json.dump(sc, f)
        return env

    def stub_specs(self, **update):
        s = M.cryozeta_spec()
        s.update(repo="file://" + self.repo, commit=self.commit, s_per_residue=0.001, msa_poll_s=0.0, msa_tries=1, colabfold=self.msa.url)
        s["teaser"] = dict(s["teaser"], prebuilt=self.teaser)
        s["hub"] = dict(s["hub"], api=self.url + "/hub/api/models/KiharaLab/CryoZeta?blobs=true", resolve=self.url + "/hub/resolve/{rev}/{path}", min_bytes=1000)
        s.update(update)
        return {"cryozeta": s}

    def run_notebook(self, tasks=None, hours=11.0, gpus="2", scenario=None, routes=None, spec_update=None, drop=(), alone_first=(), fail=None, **extra_env):
        T.Files.routes = dict(self.routes, **(routes or {}))
        FlakyFiles.fail = dict(fail or {})
        self.msa.posts = 0
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        env_dir = self.make_env(scenario, drop)
        nb_path = os.path.join(tmp, "nb.ipynb")
        M.build(nb_path, batch="test", entries=self.ENTRIES, tasks=tasks, hours=hours, entries_path=self.entries, env_python=os.path.join(env_dir, "bin", "python"), usalign_bin=USALIGN,
                net_hosts=(), require_gpu=False, specs=self.stub_specs(**(spec_update or {})), run_methods=["cryozeta"], alone_first=alone_first)
        script = os.path.join(tmp, "nb.py")
        T.to_script(nbformat.read(nb_path, as_version=4), script)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("STUB_", "MICA_AF3"))}
        env.update(MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1", MICA_RETRY_WAIT="0.1", MICA_TEST_GPUS=gpus, MICA_POLL="0.2", MICA_RESERVE_S="0",
                   MICA_MIN_START_S="2", STUB_MODEL_CIF=self.cif, MPLBACKEND="module://matplotlib_inline.backend_inline")
        env.update(extra_env)
        r = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=900)
        work = os.path.join(tmp, "work")
        zips = sorted(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".zip")) if os.path.isdir(work) else []
        return r, zips, tmp, env_dir

    def test_a_complete_run_with_two_gpus_without_the_alphafold_dataset(self):
        r, zips, tmp, env_dir = self.run_notebook(STUB_SLOW_CRYOZETA="0.8")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out)
        files = EndToEndTests.zip_files(zips[-1])
        status = EndToEndTests.csv_rows(files, "method_status.csv")
        self.assertEqual(sorted((s["entry"], s["method"], s["status"]) for s in status), sorted((e, "cryozeta", "ok") for e in self.ENTRIES), out[-2500:])
        self.assertEqual({s["limit_s"] for s in status}, {"21600"})
        metrics = EndToEndTests.csv_rows(files, "metrics_usalign.csv")
        kinds = ("cryozeta", "cryozeta_s1", "cryozeta_s2", "cryozeta_s3", "cryozeta_s4")
        self.assertEqual(sorted((m["entry"], m["model_kind"]) for m in metrics), sorted((e, k) for e in self.ENTRIES for k in kinds))
        self.assertTrue(all(float(m["tm_score"]) > 0.99 for m in metrics), metrics)
        for e in self.ENTRIES:
            for i in range(5):
                self.assertIn("models/cryozeta/%s/%s_sample_%d.cif" % (e, e, i), files)
            self.assertIn("logs/%s_cryozeta_run.log" % e, files)
            summary = json.loads(files["method_files/cryozeta/%s/run/driver_summary.json" % e].decode())
            self.assertEqual((summary["status"], summary["input"]["chains"][0]["n_uniref_hits"]), ("ok", 2))
        for name in ("cryozeta_hub_files.json", "cryozeta_source_patch.diff"):
            self.assertIn(name, files)
        # what the programs were given: the settings of the demo, the weights of the Hub, the environment of the project, one GPU each
        calls = jsonl(os.path.join(env_dir, "calls.jsonl"))
        self.assertEqual(sorted(c["prog"] for c in calls), sorted(["cryozeta-detection"] * 4 + ["cryozeta-inference"] * 8 + ["cryozeta-combine"] * 4))
        for c in calls:
            self.assertEqual((c["env"]["CONDA_PREFIX"], c["env"]["TEASERPP_DIR"], c["env"]["USE_OPM_CHUNKED"], c["env"]["LAYERNORM_TYPE"]), (env_dir, self.teaser, "1", ""))
            self.assertEqual(c["env"]["CRYOZETA_ASSETS_DIR"], os.path.join(tmp, "big", "cryozeta_assets"))
            self.assertEqual(c["env"]["PYTHONPATH"], os.path.join(self.teaser, "build", "python"))
        inf = [c["args"] for c in calls if c["prog"] == "cryozeta-inference"]
        for a in inf:
            self.assertEqual([a[a.index(k) + 1] for k in ("--seeds", "--dtype", "--infer_setting.chunk_size", "--sample_diffusion.N_sample", "--sample_diffusion.N_step", "--model.N_cycle")],
                             ["101", "bf16", "256", "5", "20", "10"])
            self.assertEqual(a[a.index("--use_deepspeed_evo_attention") + 1], "false")
        self.assertEqual(sorted(os.path.basename(a[a.index("--load_checkpoint_path") + 1]) for a in inf),
                         sorted(["cryozeta-v0.0.1.safetensors"] * 4 + ["cryozeta-interpolate-v0.0.1.safetensors"] * 4))
        det = next(c["args"] for c in calls if c["prog"] == "cryozeta-detection")
        self.assertEqual(os.path.basename(det[det.index("--checkpoint-path") + 1]), "cryozeta-detection-v0.0.1.safetensors")
        # the input of one entry: the recommended contour level, the resolution of the table, the chains with their copies, the alignments
        with open(os.path.join(tmp, "big", "input", "9T5K", "cryozeta_input.json")) as f:
            inp = json.load(f)[0]
        self.assertEqual((inp["contour_level"], inp["resolution"]), (0.123, float(self.rows["9T5K"]["resolution_A"])))
        jobs = [j for j in self.jobs if j["pdb_id"] == "9T5K"]
        self.assertEqual([(s["proteinChain"]["sequence"], s["proteinChain"]["count"]) for s in inp["sequences"]], [(j["sequence_submitted"].strip(), int(j["copies_in_entry"])) for j in jobs])
        self.assertTrue(all(os.path.exists(os.path.join(s["proteinChain"]["msa"]["precomputed_msa_dir"], "mmseqs_other_hits.a3m")) for s in inp["sequences"]))
        self.assertEqual(self.msa.posts, sum(1 for j in self.jobs if j["pdb_id"] in self.ENTRIES))                 # one query per sequence of each entry
        # both GPUs, at the same time
        spans = [s for s in jsonl(os.path.join(env_dir, "spans.jsonl")) if s["prog"] == "cryozeta-inference"]
        self.assertEqual({s["gpu"] for s in spans}, {"0", "1"}, spans)
        self.assertTrue(any(a["t0"] < b["t1"] and b["t0"] < a["t1"] for a in spans for b in spans if a["gpu"] != b["gpu"]), "no two programs ran at the same time")
        # the weights are the ones of the Hub's file list, recorded with their hashes
        info = json.loads(files["run_info.json"].decode())
        m = info["state"]["methods"]["cryozeta"]
        self.assertEqual(m["weights_hub"]["revision"], self.REV)
        self.assertEqual({v["file"]: v["sha256"] for v in m["weights_hub"]["files"].values()}, {n: hashlib.sha256(b).hexdigest() for n, b in self.blobs.items()})
        self.assertEqual(len(m["weights"]), 5, m["weights"])                                  # three weights files and the two chemical-components files
        self.assertEqual(info["state"]["pending"], [])

    def test_a_runner_killed_by_the_system_while_the_other_gpu_runs_is_run_again_alone(self):
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"], ["9WQ6", "cryozeta"]], scenario={"detection": ["killed", "ok"]}, STUB_SLOW_CRYOZETA="2")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("KILLED while the other GPU ran; it is run again alone at the end", out)
        self.assertIn("run again alone: CryoZeta on", out)
        files = EndToEndTests.zip_files(zips[-1])
        status = {s["entry"]: s for s in EndToEndTests.csv_rows(files, "method_status.csv")}
        self.assertEqual({e: s["status"] for e, s in status.items()}, {"9T5K": "ok", "9WQ6": "ok"}, out[-2500:])
        again = [s for s in status.values() if s["attempt"] == "2"]
        self.assertEqual(len(again), 1, status)
        self.assertEqual(again[0]["other_gpu_busy"], "False")
        info = json.loads(files["run_info.json"].decode())
        first = info["state"]["methods"]["cryozeta"]["entries"][again[0]["entry"]]["first_attempt"]
        self.assertEqual(first["status"], "killed_corun")
        self.assertEqual(EndToEndTests.csv_rows(files, "pending.csv"), [])

    def test_a_runner_killed_also_when_alone_is_a_failure_of_the_method_and_not_a_crash_of_the_guard(self):
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"], ["9WQ6", "cryozeta"], ["22FX", "cryozeta"]], gpus="1", scenario={"detection": ["killed"]})
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        status = EndToEndTests.csv_rows(EndToEndTests.zip_files(zips[-1]), "method_status.csv")
        self.assertEqual([s["status"] for s in status], ["failed"] * 3, status)                   # three kills in a row: results, so the guard does not stop the method
        self.assertTrue(all("exit code -9" in s["note"] and s["attempt"] == "1" for s in status), status)
        self.assertNotIn("KILLED while the other GPU ran", out)

    def test_a_task_that_runs_first_and_alone_runs_before_the_two_workers_start(self):
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"], ["9WQ6", "cryozeta"], ["22FX", "cryozeta"]], alone_first=[["9T5K", "cryozeta"]], STUB_SLOW_CRYOZETA="0.8")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("first, alone: CryoZeta on 9T5K", out)
        status = {s["entry"]: s for s in EndToEndTests.csv_rows(EndToEndTests.zip_files(zips[-1]), "method_status.csv")}
        self.assertEqual({e: s["status"] for e, s in status.items()}, {"9T5K": "ok", "9WQ6": "ok", "22FX": "ok"}, out[-2500:])
        self.assertEqual((status["9T5K"]["attempt"], status["9T5K"]["other_gpu_busy"]), ("2", "False"))
        self.assertLess(out.index("first, alone: CryoZeta on 9T5K"), out.index("CryoZeta 9WQ6"))
        stamp = lambda s: datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
        end_of_first = stamp(status["9T5K"]["started_utc"]) + datetime.timedelta(seconds=int(status["9T5K"]["seconds"]) - 1)
        self.assertTrue(all(stamp(status[e]["started_utc"]) >= end_of_first for e in ("9WQ6", "22FX")), status)         # nothing else started before it was done

    def test_a_download_that_fails_is_asked_for_again(self):
        number = self.rows["9T5K"]["emdb_id"].split("-")[-1]
        path = "/emdb/EMD-%s/map/emd_%s.map.gz" % (number, number)
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"]], fail={path: 3})            # three refusals: the whole first call of the download helper (three attempts)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertEqual(out.count("a download failed"), 1, out[-2500:])
        status = EndToEndTests.csv_rows(EndToEndTests.zip_files(zips[-1]), "method_status.csv")
        self.assertEqual([(s["entry"], s["status"]) for s in status], [("9T5K", "ok")])

    def test_an_input_that_cannot_be_downloaded_after_the_tries_is_not_run_and_not_a_failure(self):
        number = self.rows["9T5K"]["emdb_id"].split("-")[-1]
        path = "/emdb/EMD-%s/map/emd_%s.map.gz" % (number, number)
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"]], routes={path: None})
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertEqual(out.count("a download failed"), 3, out[-2500:])                         # four tries, three waits
        status = EndToEndTests.csv_rows(EndToEndTests.zip_files(zips[-1]), "method_status.csv")
        self.assertEqual([(s["entry"], s["status"], s["failed_step"]) for s in status], [("9T5K", "not run", "inputs")])
        self.assertIn("could not download", status[0]["note"])

    def test_an_out_of_gpu_memory_error_on_every_chunk_size_is_a_result_and_does_not_trip_the_guard(self):
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"], ["9WQ6", "cryozeta"], ["22FX", "cryozeta"]], scenario={"cryozeta": ["oom_out"]}, gpus="1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = EndToEndTests.zip_files(zips[-1])
        status = EndToEndTests.csv_rows(files, "method_status.csv")
        self.assertEqual([s["status"] for s in status], ["failed"] * 3, status)
        self.assertTrue(all(s["note"].startswith("out of GPU memory") for s in status), status)
        inf = [c["args"] for c in jsonl(os.path.join(env_dir, "calls.jsonl")) if c["prog"] == "cryozeta-inference"]
        self.assertEqual(len(inf), 12)                                                        # four chunk sizes per entry, and the interpolation never starts
        self.assertEqual([a[a.index("--infer_setting.chunk_size") + 1] for a in inf[:4]], ["256", "64", "16", "4"])
        self.assertEqual(json.loads(files["method_files/cryozeta/9T5K/run/driver_summary.json"].decode())["failed_stage"], "cryozeta")
        self.assertEqual(EndToEndTests.csv_rows(files, "metrics_usalign.csv"), [])

    def test_a_server_that_gives_no_alignment_is_a_failed_input_and_two_in_a_row_stop_the_method(self):
        self.msa.first = "ERROR"
        self.addCleanup(setattr, self.msa, "first", "PENDING")
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"], ["9WQ6", "cryozeta"], ["22FX", "cryozeta"]], gpus="1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = EndToEndTests.zip_files(zips[-1])
        status = EndToEndTests.csv_rows(files, "method_status.csv")
        self.assertEqual([s["status"] for s in status], ["failed", "failed", "not run"], status)
        self.assertEqual(status[2]["failed_step"], "guard")
        summary = json.loads(files["method_files/cryozeta/9T5K/run/driver_summary.json"].decode())
        self.assertEqual(summary["failed_stage"], "input")
        self.assertIn("MsaError", summary["note"])
        self.assertFalse(os.path.exists(os.path.join(env_dir, "calls.jsonl")), "no program of CryoZeta may start without its input")

    def test_weights_that_do_not_match_the_hub_stop_the_setup_before_any_run(self):
        bad = json.loads(json.dumps(self.api))
        next(s for s in bad["siblings"] if s["rfilename"] == "cryozeta-v0.0.1.safetensors")["lfs"]["sha256"] = "0" * 64
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"], ["9WQ6", "cryozeta"]], routes={"/hub/api/models/KiharaLab/CryoZeta": json.dumps(bad).encode()})
        out = r.stdout + r.stderr
        self.assertIn("STOPPED in its setup: the file cryozeta-v0.0.1.safetensors does not have the sha256 that the Hub lists", out)
        self.assertFalse(os.path.exists(os.path.join(env_dir, "calls.jsonl")), "the programs must not start")
        status = EndToEndTests.csv_rows(EndToEndTests.zip_files(zips[-1]), "method_status.csv")
        self.assertEqual(sorted((s["entry"], s["status"]) for s in status), [("9T5K", "not run"), ("9WQ6", "not run")])

    def test_a_weights_file_that_the_hub_does_not_list_stops_the_setup(self):
        short = json.loads(json.dumps(self.api))
        short["siblings"] = [s for s in short["siblings"] if s["rfilename"] != "cryozeta-interpolate-v0.0.1.safetensors"]
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"]], routes={"/hub/api/models/KiharaLab/CryoZeta": json.dumps(short).encode()})
        self.assertIn("has no file cryozeta-interpolate-v0.0.1.safetensors", r.stdout + r.stderr)
        self.assertFalse(os.path.exists(os.path.join(env_dir, "calls.jsonl")))

    def test_a_chemical_components_file_that_the_hub_does_not_list_stops_the_setup(self):
        for name in ("components.v20240608.cif", "components.v20240608.cif.rdkit_mol.pkl"):
            short = json.loads(json.dumps(self.api))
            short["siblings"] = [s for s in short["siblings"] if s["rfilename"] != name]
            r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"]], routes={"/hub/api/models/KiharaLab/CryoZeta": json.dumps(short).encode()})
            self.assertIn("has no file " + name, r.stdout + r.stderr)
            self.assertFalse(os.path.exists(os.path.join(env_dir, "calls.jsonl")))

    def test_a_program_of_the_environment_that_is_missing_stops_the_setup(self):
        r, zips, tmp, env_dir = self.run_notebook(tasks=[["9T5K", "cryozeta"]], drop=("cryozeta-combine",))
        out = r.stdout + r.stderr
        self.assertIn("CryoZeta STOPPED in its setup: the environment lacks the programs cryozeta-combine", out)
        self.assertFalse(os.path.exists(os.path.join(env_dir, "calls.jsonl")))
        self.assertFalse(any(self.msa.posts for _ in [0]), "no alignment is asked for before the setup is complete")


if __name__ == "__main__":
    unittest.main()
