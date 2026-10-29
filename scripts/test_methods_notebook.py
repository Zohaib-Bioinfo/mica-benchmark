#!/usr/bin/env python3
"""Tests for make_methods_notebook.py and the notebook it builds (the pilot of E3-CryoFold, EMProt, CryoAtom2, CryoAtom v1.0.0 and EModelX(+AF)).

Fast tests: the notebook builds, every cell compiles, the embedded files equal the repository files, the pinned environments are complete, the commands are the ones of the
READMEs, the committed notebook is current, the chain expansion of EMProt.

End-to-end tests run the whole notebook as a script against stand-ins: local git repositories that play the five programs (with the command line of the real ones and the
checks that the real ones make: the weights files, the executable bit of the shipped programs, the template folder of EModelX, one predicted chain per FASTA record of EMProt),
a file server for the maps, the deposited model and the weights archives, and the real US-align. They need

    MICA_TEST_ENV_PYTHON   a Python 3 with numpy, mrcfile and biopython (it plays the environment of every method; the stand-ins use the standard library)
    MICA_TEST_AF3_DIR      the folder with the 26 AlphaFold Server files (private; the 26DK file doubles as the deposited model and as every "built" model)
    USALIGN                the compiled US-align program
    MICA_TEST_UV=1         (optional) also builds a real Python environment with uv from a one-line pin file (needs PyPI and GitHub)

    USALIGN=... MICA_TEST_ENV_PYTHON=... MICA_TEST_AF3_DIR=... python -W error -m unittest test_methods_notebook
"""
from __future__ import annotations

import copy
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
import tarfile
import tempfile
import unittest
import zipfile

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_methods_notebook as mm  # noqa: E402
import nbkit  # noqa: E402
import test_pilot_notebook as T  # noqa: E402

COMMITTED = os.path.join(HERE, "pilot_methods.ipynb")
ENV_PY, AF3_DIR, USALIGN = T.ENV_PY, T.AF3_DIR, T.USALIGN
SIMPLE_IMPORT_TEST = "import numpy, mrcfile; print('stand-in environment: numpy', numpy.__version__)"

PRELUDE = r'''
import os, shutil, sys, time


def calls(name):
    with open(os.environ["STUB_CALLS"], "a") as f:
        f.write(name + " " + " ".join(sys.argv[1:]) + "\n")


def maybe_fail(name):
    if name in os.environ.get("STUB_FAIL", "").split(","):
        print("Error: stand-in failure of " + name)
        sys.stdout.flush()
        sys.exit(1)


def work(name):
    for i in range(int(os.environ.get("STUB_STEPS", "1"))):
        print("%s step %d ... 40%%|####      | 4/10" % (name, i + 1))
        sys.stdout.flush()
        time.sleep(float(os.environ.get("STUB_SLOW_" + name.upper(), "0")))


def need_gpu_env():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", os.environ.get("CUDA_VISIBLE_DEVICES")
    assert os.environ.get("TORCH_HOME"), "TORCH_HOME is not set"
    assert not os.environ.get("VIRTUAL_ENV") and not os.environ.get("PYTHONPATH")
    assert os.environ.get("MPLBACKEND") == "Agg", "MPLBACKEND=%r: pyplot cannot be imported without matplotlib-inline" % os.environ.get("MPLBACKEND")
'''

STUB_E3 = {
    "inference.py": PRELUDE + r'''
import argparse
p = argparse.ArgumentParser()
p.add_argument("--map_path", required=True); p.add_argument("--fasta_path", required=True); p.add_argument("--save_dir", required=True)
p.add_argument("--save_name", required=True); p.add_argument("--protocol", required=True, choices=["pre_align", "denovo", "seq_free", "seq_free_denovo"])
p.add_argument("--t", type=float, default=0.1); p.add_argument("--device", default="cuda")
a = p.parse_args()
calls("e3")
need_gpu_env()
assert a.save_dir.endswith("/"), a.save_dir                    # the real program writes save_dir + save_name + ".pdb"
assert os.path.exists(a.map_path) and a.map_path.endswith(".mrc") and os.path.exists(a.fasta_path)
for w in ("model_weight.pth", "structure_model.pth"):
    assert os.path.getsize(os.path.join("models", w)) > 1000000, w
maybe_fail("e3")
work("e3")
os.makedirs(a.save_dir, exist_ok=True)
shutil.copy(os.environ["STUB_MODEL_PDB"], a.save_dir + a.save_name + ".pdb")
print("saved", a.save_dir + a.save_name + ".pdb")
''',
}

STUB_EMPROT = {
    "emprot/__init__.py": "__version__ = '1.2-stub'\n",
    "emprot/__main__.py": PRELUDE + r'''
import argparse
p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd")
b = sub.add_parser("build")
b.add_argument("--map", required=True); b.add_argument("--seq", required=True); b.add_argument("--chain", nargs="+", required=True)
b.add_argument("--device", default="0"); b.add_argument("--output", required=True); b.add_argument("--nt", type=int, default=4)
a = p.parse_args()
assert a.cmd == "build"
calls("emprot")
need_gpu_env()
here = os.path.dirname(os.path.abspath(__file__))
records = [l for l in open(a.seq) if l.startswith(">")]
assert len(records) == len(a.chain), (len(records), len(a.chain))             # one predicted chain per record
assert len({l.strip() for l in records}) == len(records), "the FASTA records must have different names"
for c in a.chain:
    assert os.path.exists(c) and c.endswith(".cif"), c
assert os.path.getsize(os.path.join(here, "weights", "stub.pth")) > 1000000, "the weights were not unpacked into the emprot folder"
assert os.access(os.path.join(here, "bin", "tool"), os.X_OK), "the shipped program is not executable"
maybe_fail("emprot")
work("emprot")
os.makedirs(a.output, exist_ok=True)
for name in ("output.cif", "output_denovo.cif", "output_fit.cif"):
    shutil.copy(os.environ["STUB_MODEL_CIF"], os.path.join(a.output, name))
open(os.path.join(a.output, "emprot.log"), "w").write("stand-in log, %d chains\n" % len(a.chain))
''',
}


def cryoatom_main(package):
    version = "v1" if package == "CryoAtom" else "v2"
    return PRELUDE + r'''
import argparse
VERSION = "@V@"
p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd")
b = sub.add_parser("build")
b.add_argument("-v", "--volume", required=True); b.add_argument("-o", "--output", required=True)
b.add_argument("-s", "--seq", default=None); b.add_argument("-ps", "--protein_seq", default=None); b.add_argument("-d", "--device", default=None)
a = p.parse_args()
assert a.cmd == "build"
calls("cryoatom" + VERSION[1])
need_gpu_env()
here = os.path.dirname(os.path.abspath(__file__))
fasta = a.seq if VERSION == "v1" else a.protein_seq
assert fasta and os.path.exists(fasta) and os.path.exists(a.volume) and a.volume.endswith(".mrc"), (fasta, a.volume)
if VERSION == "v2":
    assert a.device == "0", a.device
else:
    assert a.device is None and a.protein_seq is None
need = ("CryNet.pth", "SimpleUnet.pth") if VERSION == "v1" else ("CryoNet.pth", "RUNet.pth")
for w in need:
    assert os.path.getsize(os.path.join(here, "checkpoint", w)) > 1000000, w
assert os.access(os.path.join(here, "utils", "getp"), os.X_OK), "getp is not executable"
if VERSION == "v2":
    assert os.path.getsize(os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", "RNA-FM_pretrained.pth")) == int(os.environ["STUB_RNAFM_BYTES"])
maybe_fail("cryoatom" + VERSION[1])
work("cryoatom" + VERSION[1])
out = a.output
os.makedirs(out, exist_ok=True)
name = os.path.basename(os.path.normpath(out))
shutil.copy(os.environ["STUB_MODEL_CIF"], os.path.join(out, name + ".cif"))
shutil.copy(os.environ["STUB_MODEL_CIF"], os.path.join(out, name + "_raw.cif"))
open(os.path.join(out, "running_time.log"), "w").write("stand-in timing\n")
'''.replace("@V@", version)


STUB_CRYOATOM = {"CryoAtom/__init__.py": '__version__ = "1.0.0-stub"\n', "CryoAtom/__main__.py": cryoatom_main("CryoAtom"),
                 "CryoAtom2/__init__.py": '__version__ = "2.1.1-stub"\n', "CryoAtom2/__main__.py": cryoatom_main("CryoAtom2")}

STUB_EMODELX = {
    "run.py": PRELUDE + r'''
import argparse
p = argparse.ArgumentParser()
p.add_argument("--protocol", required=True); p.add_argument("--EM_map", required=True); p.add_argument("--fasta", required=True)
p.add_argument("--template_dir", required=True); p.add_argument("--output_dir", required=True); p.add_argument("--run_pulchra", action="store_true")
p.add_argument("--pulchra_path", default="modules/pulchra304/src/pulchra")
a = p.parse_args()
calls("emodelx")
need_gpu_env()
assert a.protocol == "temp_flex", a.protocol
assert os.path.exists(a.EM_map) and a.EM_map.endswith(".mrc")
for w in ("best_BB_model.ckpt", "best_CA_model.ckpt", "best_AA_model.ckpt"):
    assert os.path.getsize(os.path.join("models", w)) > 1000000, w
assert os.access(a.pulchra_path, os.X_OK), "pulchra is not executable"
names = [l[1:].split("|")[0].strip() for l in open(a.fasta) if l.startswith(">")]
assert names, "no FASTA record"
for n in names:
    t = os.path.join(a.template_dir, n, "ranked_0.pdb")             # the layout the program reads for EModelX(+AF)
    atoms = [l for l in open(t) if l.startswith("ATOM")]
    assert atoms and {l[21] for l in atoms} == {"A"}, t
maybe_fail("emodelx")
pdbid = os.path.basename(a.fasta).split(".fasta")[0]            # the program names its files after the FASTA file
work("emodelx")
shutil.copy(os.environ["STUB_MODEL_PDB"], os.path.join(a.output_dir, pdbid + "_EModelX(+AF)_ca_model.pdb"))
shutil.copy(os.environ["STUB_MODEL_PDB"], os.path.join(a.output_dir, pdbid + "_EModelX(+AF)_all_atom_model.pdb"))
open(os.path.join(a.output_dir, "time_cost_%s.csv" % pdbid), "w").write("step,seconds\nall,1\n")
''',
    "modules/pulchra304/src/pulchra": "#!/bin/sh\necho pulchra stub\n",
}

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


ELF = b"\x7fELF" + b"\x02\x01\x01" + b"\x00" * 200


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mmbuild_")
        cls.path = os.path.join(cls.tmp, "nb.ipynb")
        cls.expected = mm.build(cls.path)
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
        self.assertEqual(ids, ["step1", "step2-dirs", "step2-check", "robust", "tools", "step4", "e3", "emprot", "cryoatom", "emodelx", "summary", "final"])

    def test_embedded_files_equal_the_repository_files(self):
        got = {}
        for c in self.nb.cells:
            if c.source.startswith("%%writefile"):
                first, text = c.source.split("\n", 1)
                got[first.split(None, 1)[1].strip()] = text
        self.assertEqual(sorted(got), sorted(t for _s, t in mm.FILES))
        for src, target in mm.FILES:
            with open(os.path.join(ROOT, src), encoding="utf-8") as f:
                text = f.read()
            self.assertEqual(got[target], text if text.endswith("\n") else text + "\n", target)
            self.assertEqual(self.expected[target], hashlib.sha256(got[target].encode()).hexdigest())
        for key in mm.ORDER:
            self.assertIn(mm.SPECS[key]["req"], got)

    def test_the_committed_notebook_is_current(self):
        if not os.path.exists(COMMITTED):
            self.skipTest("pilot_methods.ipynb is not built yet")
        fresh = os.path.join(self.tmp, "fresh.ipynb")
        mm.build(fresh)
        a, b = nbformat.read(fresh, as_version=4), nbformat.read(COMMITTED, as_version=4)
        self.assertEqual([c.source for c in a.cells], [c.source for c in b.cells], "run make_methods_notebook.py again")

    def test_pins_of_the_sources_are_full_hashes_and_in_the_notebook(self):
        for key, spec in mm.SPECS.items():
            self.assertRegex(spec["commit"], r"^[0-9a-f]{40}$", key)
            self.assertIn(spec["commit"], self.text)
            self.assertIn(spec["repo"], self.text)
        self.assertEqual(mm.SPECS["cryoatom1"]["commit"], "e2a31f2b2ff21fae363490ac5f4a5e945513024c")             # the tag v1.0.0
        self.assertEqual(mm.SPECS["emprot"]["commit"], "9489d52e7fdc81c85ba82e386abcfa05bac2b9d1")
        self.assertEqual(sorted(mm.ORDER), sorted(mm.SPECS))

    def test_the_pinned_environment_files_are_complete(self):
        want = {"e3": ["torch==2.6.0", "numpy==1.26", "open3d==", "pytorch-lightning==", "fair-esm==", "biopython==", "mrcfile==", "superpose3d==", "nglview==", "scikit-image==", "networkx=="],
                "emprot": ["torch==2.1.2", "numpy==1.26.2", "numba==0.59.1", "ortools==9.4.1874", "cctbx-base==2025.11", "pyhmmer==0.10.", "dm-tree==", "timm==1.0.8", "pydssp==0.9.0",
                           "mrcfile==", "einops==0.8.0", "jaxtyping==0.2.33", "beartype==0.18.5", "biopython==1.84"],
                "cryoatom2": ["torch==2.1.0", "torchvision==0.16.0", "numpy==1.21.5", "fair-esm==2.0.0", "rna-fm==0.2.2", "loguru==", "wget==", "einops==0.6.0", "scipy==1.7.3", "pyhmmer==0.7.1"],
                "cryoatom1": ["torch==1.11.0", "torchvision==0.12.0", "numpy==1.21.5", "fair-esm==2.0.0", "loguru==", "wget==", "einops==0.6.0", "scipy==1.7.3", "pyhmmer==0.7.1"],
                "emodelx": ["torch==1.8.1", "numpy==1.19.1", "open3d==0.13.0", "mrcfile==1.3.0", "superpose3d==1.1.1", "scikit-image==0.18.2", "networkx==2.6.3", "biopython==1.78"]}
        for key, needles in want.items():
            with open(os.path.join(ROOT, mm.SPECS[key]["req"]), encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            pins = [ln for ln in lines if not ln.startswith("#")]
            self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.+!\-]+", p) for p in pins), [p for p in pins if "==" not in p][:5])
            self.assertEqual(len(pins), len({p.split("==")[0].lower() for p in pins}), "%s lists a package twice" % key)
            for n in needles:
                self.assertTrue(any(p.lower().startswith(n.lower()) for p in pins), "%s: %s is missing" % (key, n))
            self.assertTrue(lines[0].startswith("#") and "uv pip compile" in "\n".join(lines[:12]), key)
            self.assertNotIn("transformers", "\n".join(pins))

    def test_the_commands_are_the_ones_of_the_readmes_with_the_gpu_named(self):
        e3, emprot, atom, emx = self.cell("e3"), self.cell("emprot"), self.cell("cryoatom"), self.cell("emodelx")
        self.assertIn('"--map_path", ctx["map"], "--fasta_path", ctx["fasta"], "--save_dir", out, "--save_name", name, "--protocol", protocol, "--t", "0.1"', e3)
        self.assertIn('(("prealign", "pre_align"), ("denovo", "denovo"))', e3)
        self.assertIn('"-m", "emprot", "build", "--map", ctx["map"], "--seq", fasta_path, "--chain"] + chains + ["--device", "0", "--output", out]', emprot)
        self.assertEqual(mm.SPECS["cryoatom1"]["cli"], ["-s", "{fasta}", "-v", "{map}", "-o", "{out}"])
        self.assertEqual(mm.SPECS["cryoatom2"]["cli"], ["-v", "{map}", "-ps", "{fasta}", "-o", "{out}", "-d", "0"])
        self.assertIn('[env_py, "-m", spec["pkg"], "build"] + args', atom)
        self.assertIn('"run.py", "--protocol=temp_flex", "--EM_map", ctx["map"], "--fasta", fasta, "--template_dir", tdir, "--output_dir", out, "--run_pulchra"', emx)
        self.assertIn('"--pulchra_path", "modules/pulchra304/src/pulchra"', emx)
        self.assertIn('"ranked_0.pdb"', emx)
        self.assertIn('"CUDA_VISIBLE_DEVICES": "0"', self.cell("tools"))

    def test_a_gpu_is_required(self):
        self.assertIn("REQUIRE_GPU = True", self.cell("step1"))

    def test_the_method_that_gets_the_predictions_is_the_one_of_the_spec(self):
        self.assertEqual({k for k, s in mm.SPECS.items() if s["uses_af3"]}, {"emprot", "emodelx"})           # decision 6 of the spec: only the methods that accept predictions

    def test_the_predictions_themselves_are_not_kept_in_the_zip(self):
        tools = self.cell("tools")
        self.assertIn('PRIVATE_KINDS = ("emprot_fit",)', tools)
        self.assertIn("if kind in PRIVATE_KINDS:", tools)
        # nothing of the folders that hold the predictions or the templates is copied by keep_results
        self.assertIn('n.endswith((".log", ".txt", ".csv", ".json", ".xlsx"))', tools)

    def exec_function(self, cell_id, name, **glob):
        src = nbkit.function_sources(self.cell(cell_id), [name])
        ns = dict(glob)
        exec(src, ns)
        return ns[name]

    def test_emprot_writes_one_record_and_one_chain_per_copy(self):
        f = self.exec_function("emprot", "emprot_inputs")
        jobs = [{"entity": "E1", "sequence_submitted": "MKV", "copies_in_entry": "4"}, {"entity": "E2", "sequence_submitted": "GGSS\n", "copies_in_entry": "1"}]
        fasta, chains = f("9OC0", jobs, ["/x/a.cif", "/x/b.cif"])
        records = fasta.strip().split("\n")
        self.assertEqual(len(records), 10)
        self.assertEqual([r for r in records if r.startswith(">")], [">9OC0_E1_1", ">9OC0_E1_2", ">9OC0_E1_3", ">9OC0_E1_4", ">9OC0_E2_1"])
        self.assertEqual([r for r in records if not r.startswith(">")], ["MKV"] * 4 + ["GGSS"])
        self.assertEqual(chains, ["/x/a.cif"] * 4 + ["/x/b.cif"])
        # the real table: 9OC0 has four chains
        with open(os.path.join(ROOT, "data", "af3", "af3_jobs.csv"), newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["pdb_id"] == "9OC0"]
        fasta, chains = f("9OC0", rows, ["/x/%d.cif" % i for i in range(len(rows))])
        self.assertEqual(len(chains), sum(int(r["copies_in_entry"]) for r in rows))
        self.assertEqual(fasta.count(">"), len(chains))

    def test_template_names_follow_the_first_token_of_the_fasta_header(self):
        emx = self.cell("emodelx")
        self.assertIn('ln[1:].split("|")[0]', emx)
        self.assertIn("MMCIFParser", emx)


@unittest.skipUnless(ENV_PY and AF3_DIR and USALIGN and any("fold_26dk_e1_model_0.cif" in f for _r, _d, f in os.walk(AF3_DIR or "")),
                     "needs MICA_TEST_ENV_PYTHON, MICA_TEST_AF3_DIR (with the 26DK file) and USALIGN")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="mmrun_")
        cls.cif = [os.path.join(r, f) for r, _d, fs in os.walk(AF3_DIR) for f in fs if f == "fold_26dk_e1_model_0.cif"][0]
        with open(cls.cif, "rb") as f:
            cif_bytes = f.read()
        cls.pdb = os.path.join(cls.root, "model.pdb")
        subprocess.run([ENV_PY, "-c", "import sys; from Bio.PDB import MMCIFParser, PDBIO; s = MMCIFParser(QUIET=True).get_structure('m', sys.argv[1]); io = PDBIO(); "
                        "io.set_structure(s); io.save(sys.argv[2])", cls.cif, cls.pdb], check=True)
        mapfile = os.path.join(cls.root, "emd_80558.map")
        subprocess.run([ENV_PY, "-c", "import mrcfile, numpy; m = mrcfile.new(%r, overwrite=True); m.set_data(numpy.random.rand(32, 32, 32).astype('float32')); "
                        "m.voxel_size = 1.0; m.close()" % mapfile], check=True)
        with open(mapfile, "rb") as f:
            map_gz = gzip.compress(f.read())
        blob = lambda n, seed: bytes((i * seed) % 251 for i in range(256)) * (n // 256)
        cls.rnafm_bytes = 3000000

        def zipped(files):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for name, data in files.items():
                    z.writestr(name, data)
            return buf.getvalue()

        def tgz(files):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as t:
                for name, data in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    t.addfile(info, io.BytesIO(data))
            return buf.getvalue()

        cls.srv, cls.url = T.serve(T.Files)
        cls.routes = {"/map/emd_80558.map.gz": map_gz, "/cif/26DK.cif": cif_bytes,
                      "/e3/models.zip": zipped({"models/model_weight.pth": blob(1500000, 3), "models/structure_model.pth": blob(1500000, 5)}),
                      "/emprot/weights.tgz": tgz({"weights/stub.pth": blob(1500000, 7)}),
                      "/cryoatom/checkpoints_v1.0.zip": zipped({"checkpoints/CryNet.pth": blob(1500000, 11), "checkpoints/SimpleUnet.pth": blob(1500000, 13)}),
                      "/cryoatom/checkpoints_v2.1.zip": zipped({"checkpoints/CryoNet.pth": blob(1500000, 17), "checkpoints/RUNet.pth": blob(1500000, 19)}),
                      "/cryoatom/RNA-FM_pretrained.pth": blob(cls.rnafm_bytes, 23)[:cls.rnafm_bytes] + b"\0" * (cls.rnafm_bytes - len(blob(cls.rnafm_bytes, 23)[:cls.rnafm_bytes])),
                      "/emodelx/models.zip": zipped({"models/best_BB_model.ckpt": blob(1500000, 29), "models/best_CA_model.ckpt": blob(1500000, 31),
                                                     "models/best_AA_model.ckpt": blob(1500000, 37)})}
        T.Files.routes = dict(cls.routes)
        assert len(cls.routes["/cryoatom/RNA-FM_pretrained.pth"]) == cls.rnafm_bytes
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
        cls.repos, cls.commits = {}, {}
        for name, files in (("e3", STUB_E3), ("emprot", STUB_EMPROT), ("cryoatom", STUB_CRYOATOM), ("emodelx", STUB_EMODELX)):
            repo = os.path.join(cls.root, name + "_stub")
            for rel, text in files.items():
                path = os.path.join(repo, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(text)
            if name == "emprot":                                         # a shipped program that git keeps without the executable bit
                os.makedirs(os.path.join(repo, "emprot", "bin"), exist_ok=True)
                with open(os.path.join(repo, "emprot", "bin", "tool"), "wb") as f:
                    f.write(ELF)
            if name == "cryoatom":
                for pkg in ("CryoAtom", "CryoAtom2"):
                    os.makedirs(os.path.join(repo, pkg, "utils"), exist_ok=True)
                    with open(os.path.join(repo, pkg, "utils", "getp"), "wb") as f:
                        f.write(ELF)
            T.sh("git init -q -b main . && git config core.fileMode true && git add -A && git -c user.name=t -c user.email=t@t commit -q -m stub", cwd=repo)
            T.sh("chmod 644 emprot/bin/tool 2>/dev/null; chmod 644 CryoAtom/utils/getp CryoAtom2/utils/getp 2>/dev/null; true", cwd=repo)
            cls.repos[name], cls.commits[name] = repo, T.sh("git rev-parse HEAD", cwd=repo)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def stub_specs(self):
        s = copy.deepcopy(mm.SPECS)
        for key, repo in (("e3", "e3"), ("emprot", "emprot"), ("cryoatom2", "cryoatom"), ("cryoatom1", "cryoatom"), ("emodelx", "emodelx")):
            s[key].update(repo="file://" + self.repos[repo], commit=self.commits[repo])
        s["e3"].update(weights_url=self.url + "/e3/models.zip", weights_min_bytes=1000)
        s["emprot"].update(weights_url=self.url + "/emprot/weights.tgz", weights_min_bytes=1000)
        s["cryoatom1"].update(weights_url=self.url + "/cryoatom/checkpoints_v1.0.zip", weights_min_bytes=1000)
        s["cryoatom2"].update(weights_url=self.url + "/cryoatom/checkpoints_v2.1.zip", weights_min_bytes=1000, rnafm_url=self.url + "/cryoatom/RNA-FM_pretrained.pth", rnafm_bytes=self.rnafm_bytes)
        s["emodelx"].update(weights_url=self.url + "/emodelx/models.zip", weights_min_bytes=1000, gdrive_id=None)
        return s

    def run_notebook(self, require_gpu=False, patch=None, specs=None, run_methods=None, env_python=ENV_PY, routes=None, af3_dir=None, build_extra=None, **extra_env):
        T.Files.routes = dict(self.routes, **(routes or {}))
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        nb_path = os.path.join(tmp, "nb.ipynb")
        mm.build(nb_path, entries_path=self.entries, env_python=env_python, usalign_bin=USALIGN, net_hosts=(), require_gpu=require_gpu, specs=specs or self.stub_specs(),
                 run_methods=run_methods, **(build_extra or {}))
        nb = nbformat.read(nb_path, as_version=4)
        for c in nb.cells:
            for old, new in (patch or {}).get(c["id"], []):
                self.assertIn(old, c.source)
                c.source = c.source.replace(old, new)
        script = os.path.join(tmp, "nb.py")
        T.to_script(nb, script)
        calls = os.path.join(tmp, "calls.txt")
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1", MICA_RETRY_WAIT="0.1", MICA_AF3_DIR=af3_dir or AF3_DIR,
                   STUB_CALLS=calls, STUB_MODEL_CIF=self.cif, STUB_MODEL_PDB=self.pdb, STUB_RNAFM_BYTES=str(self.rnafm_bytes),
                   MPLBACKEND="module://matplotlib_inline.backend_inline")      # what the Kaggle kernel sets
        for k in [k for k in env if k.startswith("STUB_") and k not in ("STUB_CALLS", "STUB_MODEL_CIF", "STUB_MODEL_PDB", "STUB_RNAFM_BYTES")]:
            env.pop(k)
        env.update(extra_env)
        r = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=1500)
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

    def test_a_complete_run_of_all_five(self):
        r, zips, tmp = self.run_notebook(STUB_STEPS="2", STUB_SLOW_E3="1.5")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out, out[-3000:])
        self.assertEqual(len(zips), 1, zips)
        self.assertNotIn("_INCOMPLETE", zips[0])
        self.assertFalse(os.path.exists(os.path.join(tmp, "work", "methods_pilot_partial.zip")))
        files = self.zip_files(zips[0])
        for needed in ("run.log", "state.json", "run_info.json", "pilot_metrics.csv", "method_status.csv", "machine.txt", "cpu_facts.txt", "scoring/26DK_e3_prealign_usalign.txt",
                       "logs/26DK_e3_prealign.log", "logs/26DK_e3_denovo.log", "logs/26DK_emprot_build.log", "logs/26DK_cryoatom2_build.log", "logs/26DK_cryoatom1_build.log",
                       "logs/26DK_emodelx_temp_flex.log", "gpu/26DK_e3_prealign.csv",
                       "models/e3/26DK/26DK_e3_prealign.pdb", "models/e3/26DK/26DK_e3_denovo.pdb", "models/emprot/26DK/output.cif", "models/emprot/26DK/output_denovo.cif",
                       "models/cryoatom2/26DK/26DK_cryoatom2.cif", "models/cryoatom2/26DK/26DK_cryoatom2_raw.cif", "models/cryoatom1/26DK/26DK_cryoatom1.cif",
                       "models/cryoatom1/26DK/26DK_cryoatom1_raw.cif", "models/emodelx/26DK/26dk_EModelX(+AF)_all_atom_model.pdb", "models/emodelx/26DK/26dk_EModelX(+AF)_ca_model.pdb",
                       "method_files/emodelx/26DK/temp_flex/time_cost_26dk.csv", "method_files/emprot/26DK/build/emprot.log", "method_files/cryoatom2/26DK/build/running_time.log",
                       "scoring/26DK_emprot_usalign.txt", "scoring/26DK_emodelx_af_ca_usalign.txt", "scoring/26DK_cryoatom1_raw_usalign.txt"):
            self.assertIn(needed, files, sorted(files))
        # what must stay out: the maps, the predictions (and the templates and inputs made from them), the fit of the prediction that EMProt writes
        self.assertFalse(any(n.endswith((".map", ".mrc", ".ccp4")) or "AF3_results" in n or "ranked_0" in n or n.endswith(("model_weight.pth", ".ckpt", ".tgz")) for n in files), sorted(files))
        self.assertNotIn("models/emprot/26DK/output_fit.cif", files)
        rows = {r["model_kind"]: r for r in self.csv_rows(files, "pilot_metrics.csv")}
        self.assertEqual(sorted(rows), ["cryoatom1", "cryoatom1_raw", "cryoatom2", "cryoatom2_raw", "e3_denovo", "e3_prealign", "emodelx_af_all_atom", "emodelx_af_ca",
                                        "emprot", "emprot_denovo", "emprot_fit"])
        for row in rows.values():
            self.assertEqual(row["status"], "ok")
            self.assertGreater(float(row["tm_score"]), 0.99)
        status = {(r["method"], r["entry"]): r for r in self.csv_rows(files, "method_status.csv")}
        self.assertEqual(sorted(status), [("cryoatom1", "26DK"), ("cryoatom2", "26DK"), ("e3", "26DK"), ("emodelx", "26DK"), ("emprot", "26DK")])
        self.assertTrue(all(r["status"] == "ok" for r in status.values()))
        info = json.loads(files["run_info.json"].decode())
        self.assertTrue(info["complete"])
        for key, spec in mm.SPECS.items():
            self.assertEqual(info["methods"][key]["status"], "ok")
            self.assertEqual(info["methods"][key]["commit"], self.stub_specs()[key]["commit"])
        self.assertGreater(sum(info["state"]["methods"]["e3"]["weights"].values()), 2000000)
        self.assertEqual(info["state"]["methods"]["e3"]["entries"]["26DK"]["runs"]["prealign"]["rc"], 0)
        # the commands that the programs received
        calls = read_lines(os.path.join(tmp, "calls.txt"))
        by = {}
        for c in calls:
            by.setdefault(c.split(" ", 1)[0], []).append(c)
        self.assertEqual(sorted(by), ["cryoatom1", "cryoatom2", "e3", "emodelx", "emprot"])
        self.assertEqual(len(by["e3"]), 2)
        self.assertRegex(by["e3"][0], r"--map_path .*/input/26DK/emd_80558\.mrc --fasta_path .*/26dk\.fasta --save_dir .*/runs/e3/26DK/prealign/ --save_name 26DK_e3_prealign --protocol pre_align --t 0\.1$")
        self.assertRegex(by["e3"][1], r"--save_dir .*/runs/e3/26DK/denovo/ --save_name 26DK_e3_denovo --protocol denovo --t 0\.1$")
        self.assertRegex(by["emprot"][0], r"build --map .*/emd_80558\.mrc --seq .*/26dk_emprot\.fasta --chain .*/26DK/AF3_results/26dk_1/fold_26dk_1_model_0\.cif --device 0 --output .*/runs/emprot/26DK$")
        self.assertRegex(by["cryoatom2"][0], r"build -v .*/emd_80558\.mrc -ps .*/26dk\.fasta -o .*/runs/cryoatom2/26DK_cryoatom2 -d 0$")
        self.assertRegex(by["cryoatom1"][0], r"build -s .*/26dk\.fasta -v .*/emd_80558\.mrc -o .*/runs/cryoatom1/26DK_cryoatom1$")
        self.assertRegex(by["emodelx"][0], r"--protocol=temp_flex --EM_map .*/emd_80558\.mrc --fasta .*/emodelx/26DK/26dk\.fasta --template_dir .*/emodelx/26DK/templates --output_dir .*/emodelx/26DK/out "
                                           r"--run_pulchra --pulchra_path modules/pulchra304/src/pulchra$")
        # the log shows what a person who watches needs
        self.assertIn("[E3-CryoFold prealign] running for", out)
        self.assertIn("last line: e3 step", out)
        for label in ("E3-CryoFold", "EMProt", "CryoAtom2", "CryoAtom v1.0.0", "EModelX(+AF)"):
            self.assertIn(label, out)
        self.assertIn("RNA-FM weights", out)
        self.assertIn("weights files of e3:", out)
        # the weights are where the programs read them
        big = os.path.join(tmp, "big")
        self.assertTrue(os.path.exists(os.path.join(big, "src_e3", "models", "model_weight.pth")))
        self.assertTrue(os.path.exists(os.path.join(big, "src_emprot", "emprot", "weights", "stub.pth")))
        self.assertEqual(os.path.getsize(os.path.join(big, "torch_home", "hub", "checkpoints", "RNA-FM_pretrained.pth")), self.rnafm_bytes)
        self.assertTrue(os.access(os.path.join(big, "src_emprot", "emprot", "bin", "tool"), os.X_OK))
        self.assertTrue(os.access(os.path.join(big, "src_cryoatom1", "CryoAtom", "utils", "getp"), os.X_OK))

    def test_a_method_whose_weights_cannot_be_downloaded_stops_alone(self):
        r, zips, tmp = self.run_notebook(routes={"/e3/models.zip": None})
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("E3-CryoFold STOPPED in its setup: the weights could not be downloaded", out)
        self.assertIn("RESULT: NOT COMPLETE (E3-CryoFold: setup)", out)
        files = self.zip_files(zips[0])
        self.assertIn("_INCOMPLETE", zips[0])
        status = {(r["method"], r["entry"]): r for r in self.csv_rows(files, "method_status.csv")}
        self.assertEqual(status[("e3", "(setup)")]["status"], "failed")
        for key in ("emprot", "cryoatom2", "cryoatom1", "emodelx"):
            self.assertEqual(status[(key, "26DK")]["status"], "ok", key)
        kinds = {r["model_kind"] for r in self.csv_rows(files, "pilot_metrics.csv")}
        self.assertNotIn("e3_prealign", kinds)
        self.assertIn("emodelx_af_ca", kinds)

    def test_a_web_page_instead_of_the_weights_is_refused_and_removed(self):
        r, zips, tmp = self.run_notebook(routes={"/e3/models.zip": b"<html>Too many users have downloaded this file recently</html>"}, run_methods=["e3", "cryoatom1"])
        out = r.stdout + r.stderr
        self.assertIn("E3-CryoFold STOPPED in its setup: the weights could not be downloaded", out)
        self.assertIn("the server sent a file of the wrong size", out)
        self.assertFalse(os.path.exists(os.path.join(tmp, "big", "downloads", "e3cryofold_models.zip")))
        status = {(r["method"], r["entry"]): r for r in self.csv_rows(self.zip_files(zips[0]), "method_status.csv")}
        self.assertEqual(status[("cryoatom1", "26DK")]["status"], "ok")

    def test_a_failing_program_stops_only_its_method_and_shows_the_last_lines(self):
        r, zips, tmp = self.run_notebook(STUB_FAIL="emprot,cryoatom2")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("EMProt build on 26DK STOPPED: exit code 1", out)
        self.assertIn("Error: stand-in failure of emprot", out)
        self.assertIn("RESULT: NOT COMPLETE", out)
        self.assertIn("EMProt: build", out)
        self.assertIn("CryoAtom2 (v2.1.0 and later fixes): build", out)
        files = self.zip_files(zips[0])
        status = {(r["method"], r["entry"]): r for r in self.csv_rows(files, "method_status.csv")}
        self.assertEqual({k: v["status"] for k, v in status.items()}, {("e3", "26DK"): "ok", ("emprot", "26DK"): "failed", ("cryoatom2", "26DK"): "failed", ("cryoatom1", "26DK"): "ok",
                                                                         ("emodelx", "26DK"): "ok"})
        self.assertIn("logs/26DK_emprot_build.log", files)
        self.assertIn("stand-in failure", files["logs/26DK_emprot_build.log"].decode())

    def test_a_run_that_takes_too_long_is_stopped_and_counted_as_failed(self):
        specs = self.stub_specs()
        specs["emodelx"]["timeout"] = 3
        r, zips, tmp = self.run_notebook(specs=specs, run_methods=["emodelx", "e3"], STUB_STEPS="30", STUB_SLOW_EMODELX="1", STUB_SLOW_E3="0")
        out = r.stdout + r.stderr
        self.assertIn("EModelX(+AF) temp_flex on 26DK STOPPED: timeout", out)
        files = self.zip_files(zips[0])
        status = {(r["method"], r["entry"]): r for r in self.csv_rows(files, "method_status.csv")}
        self.assertEqual(status[("emodelx", "26DK")]["status"], "failed")
        self.assertIn("timeout", status[("emodelx", "26DK")]["note"])
        self.assertEqual(status[("e3", "26DK")]["status"], "ok")

    def test_without_the_alphafold_files_only_the_methods_that_need_them_fail(self):
        empty = tempfile.mkdtemp(prefix="noaf3_", dir=self.root)
        r, zips, tmp = self.run_notebook(af3_dir=empty)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        files = self.zip_files(zips[0])
        status = {(r["method"], r["entry"]): r for r in self.csv_rows(files, "method_status.csv")}
        self.assertEqual(status[("emprot", "26DK")]["status"], "failed")
        self.assertEqual(status[("emodelx", "26DK")]["status"], "failed")
        self.assertEqual(status[("emprot", "26DK")]["failed_step"], "inputs")
        self.assertEqual({status[(k, "26DK")]["status"] for k in ("e3", "cryoatom2", "cryoatom1")}, {"ok"})
        self.assertIn("RESULT: NOT COMPLETE", out)

    def test_a_subset_of_the_methods_can_be_run(self):
        r, zips, tmp = self.run_notebook(run_methods=["cryoatom1"])
        out = r.stdout + r.stderr
        self.assertIn("RESULT: OK", out, out[-3000:])
        files = self.zip_files(zips[0])
        self.assertEqual({r["method"] for r in self.csv_rows(files, "pilot_metrics.csv")}, {"cryoatom1"})
        calls = read_lines(os.path.join(tmp, "calls.txt"))
        self.assertEqual([c.split(" ")[0] for c in calls], ["cryoatom1"])
        self.assertIn("left out of RUN_METHODS", out)

    def test_weights_attached_as_a_dataset_are_used_for_emodelx(self):
        extra = tempfile.mkdtemp(prefix="attached_", dir=self.root)
        for n in ("best_BB_model.ckpt", "best_CA_model.ckpt", "best_AA_model.ckpt"):
            with open(os.path.join(extra, n), "wb") as f:
                f.write(b"w" * 1500000)
        specs = self.stub_specs()
        specs["emodelx"].update(weights_url=self.url + "/emodelx/not_there.zip")
        r, zips, tmp = self.run_notebook(specs=specs, run_methods=["emodelx"], MICA_EXTRA_INPUT=extra)
        out = r.stdout + r.stderr
        self.assertIn("RESULT: OK", out, out[-3000:])
        self.assertIn("the weights files were attached as a Kaggle dataset", out)

    def test_without_a_gpu_the_notebook_says_what_to_switch_on(self):
        r, zips, tmp = self.run_notebook(require_gpu=True)
        out = r.stdout + r.stderr
        if shutil.which("nvidia-smi"):
            self.skipTest("this machine has a GPU")
        self.assertIn("this session has no GPU", out)
        self.assertIn("GPU T4 x2", out)
        self.assertIn("RESULT: STOPPED", out)
        self.assertEqual(len(zips), 1)
        self.assertEqual([], [c for c in os.listdir(tmp) if c == "calls.txt"])

    def test_a_second_run_in_the_same_folders_reuses_the_weights_and_runs_again(self):
        r, zips, tmp = self.run_notebook(run_methods=["e3", "cryoatom2"])
        self.assertIn("RESULT: OK", r.stdout, r.stdout[-2000:])
        script = os.path.join(tmp, "nb.py")
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1", MICA_RETRY_WAIT="0.1", MICA_AF3_DIR=AF3_DIR,
                   STUB_CALLS=os.path.join(tmp, "calls.txt"), STUB_MODEL_CIF=self.cif, STUB_MODEL_PDB=self.pdb, STUB_RNAFM_BYTES=str(self.rnafm_bytes), MPLBACKEND="module://matplotlib_inline.backend_inline")
        T.Files.routes = {k: v for k, v in self.routes.items() if k.startswith(("/map", "/cif"))}              # the weights are gone from the server: they must not be needed
        r2 = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=900)
        out = r2.stdout + r2.stderr
        self.assertIn("RESULT: OK", out, out[-3000:])
        self.assertIn("already downloaded", out)
        self.assertEqual(len([c for c in read_lines(os.path.join(tmp, "calls.txt")) if c.startswith("e3 ")]), 4)

    @unittest.skipUnless(os.environ.get("MICA_TEST_UV"), "set MICA_TEST_UV=1 (needs PyPI and GitHub) to build a real environment with uv")
    def test_the_environment_is_built_by_uv_from_a_pin_file_and_its_import_test_runs(self):
        pins = os.path.join(self.root, "tiny_requirements.txt")
        with open(pins, "w") as f:
            f.write("# one pure-Python wheel\ntqdm==4.67.1\n")
        specs = self.stub_specs()
        specs["e3"].update(req=pins, python="3.9.25", import_test="import sys, tqdm; print('python', sys.version.split()[0], '| tqdm', tqdm.__version__)")
        r, zips, tmp = self.run_notebook(specs=specs, run_methods=["e3"], env_python=None)
        out = r.stdout + r.stderr
        self.assertIn("RESULT: OK", out, out[-4000:])
        self.assertIn("python 3.9.25 | tqdm 4.67.1", out)
        self.assertTrue(os.path.exists(os.path.join(tmp, "big", "env_e3", "bin", "python")))


if __name__ == "__main__":
    unittest.main()
