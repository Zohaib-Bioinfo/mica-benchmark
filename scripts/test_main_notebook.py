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


if __name__ == "__main__":
    unittest.main()
