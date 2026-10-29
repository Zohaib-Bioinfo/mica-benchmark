#!/usr/bin/env python3
"""Tests for make_survey_notebook.py and the notebook it builds (the CPU survey of what the maps hold).

Fast tests: the notebook builds, every cell compiles, no placeholder is left, the embedded files equal the repository files, the committed notebook is current.
End-to-end tests run the whole notebook as a script against a local file server (synthetic maps, models, EMDB header files and RCSB answers):

    python -W error -m unittest test_survey_notebook
"""
from __future__ import annotations

import csv
import gzip
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
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_survey_notebook as sn  # noqa: E402
import nbkit  # noqa: E402
import test_map_survey as TM  # noqa: E402
import test_pilot_notebook as T  # noqa: E402

COMMITTED = os.path.join(HERE, "survey.ipynb")

XML = """<?xml version="1.0" encoding="UTF-8"?>
<emd emdb_id="EMD-@NUM@" version="3.0.9.3">
  <sample><name>Synthetic assembly @NUM@</name>
    <supramolecule_list><complex_supramolecule supramolecule_id="1"><name>X</name></complex_supramolecule></supramolecule_list>
    <macromolecule_list><protein_or_peptide macromolecule_id="1"><name>P</name><number_of_copies>3</number_of_copies><oligomeric_state>trimer</oligomeric_state></protein_or_peptide></macromolecule_list>
  </sample>
  <structure_determination_list><structure_determination><method>singleParticle</method>
    <image_processing_list><image_processing><final_reconstruction><applied_symmetry><point_group>@PG@</point_group></applied_symmetry>
      <resolution units="A" res_type="BY AUTHOR">@RES@</resolution></final_reconstruction></image_processing></image_processing_list>
  </structure_determination></structure_determination_list>
  <map><contour_list><contour primary="true"><level>0.25</level><source>author</source></contour></contour_list></map>
</emd>
"""


class FastTests(unittest.TestCase):
    def build(self, **kw):
        tmp = tempfile.mkdtemp(prefix="sn_")
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, "nb.ipynb")
        expected = sn.build(path, **kw)
        return nbformat.read(path, as_version=4), expected

    def test_cells_compile_and_have_no_placeholders(self):
        nb, _ = self.build()
        for c in nb.cells:
            if c.cell_type == "code" and not c.source.startswith("%%writefile"):
                compile(c.source, c["id"], "exec")
            self.assertEqual([], __import__("re").findall(r"@[A-Z_0-9]+@", c.source), c["id"])

    def test_cell_order(self):
        nb, _ = self.build()
        ids = [c["id"] for c in nb.cells]
        self.assertEqual(ids[0], "intro")
        self.assertLess(ids.index("step1"), ids.index("robust"))
        self.assertLess(ids.index("robust"), ids.index("step3"))
        self.assertEqual(ids[-1], "final")

    def test_embedded_files_equal_the_repository_files(self):
        nb, expected = self.build()
        for c in nb.cells:
            if c.source.startswith("%%writefile"):
                first, text = c.source.split("\n", 1)
                target = first.split(None, 1)[1].strip()
                with open(os.path.join(ROOT, target), encoding="utf-8") as f:
                    self.assertEqual(f.read().rstrip("\n"), text.rstrip("\n"), target)
        self.assertIn("scripts/map_survey.py", expected)

    def test_the_committed_notebook_is_current(self):
        if not os.path.exists(COMMITTED):
            self.skipTest("survey.ipynb is kept in as_run/ (the notebook that was run, not rebuilt)")
        nb, _ = self.build()
        self.assertEqual(nbformat.read(COMMITTED, as_version=4).cells, nb.cells)

    def test_no_gpu_is_required_and_the_intro_promises_no_secrets_and_no_private_files(self):
        nb, _ = self.build()
        text = "\n".join(c.source for c in nb.cells)
        self.assertIn("REQUIRE_GPU = False", text)
        self.assertIn("no GPU, no Phenix, no secrets and no private files", nb.cells[0].source)
        self.assertNotIn("find_af3_dir(", [c.source for c in nb.cells if c["id"] == "step3"][0])

    def test_the_survey_columns_are_the_ones_of_the_module(self):
        nb, _ = self.build()
        self.assertIn("map_survey.SURVEY_COLUMNS", "\n".join(c.source for c in nb.cells))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="snrun_")
        TM.Base.root = cls.root
        one = ((20.0, 40.0, 20.0),)
        three = ((15.0, 20.0, 10.0), (40.0, 40.0, 10.0), (65.0, 60.0, 10.0))
        TM.write_cif(os.path.join(cls.root, "one.cif"), one)
        TM.write_cif(os.path.join(cls.root, "three.cif"), three[:1])
        maps = {}
        for name, copies in (("one", one), ("three", three)):
            pts = [p for c in copies for arr in (TM.helix(40, c)[0],) + tuple(TM.helix(40, c)[1]) for p in arr]
            TM.write_mrc(os.path.join(cls.root, name + ".map"), TM.density(TM.SurveyTests.BOX, 1.0, pts), voxel=1.0)
            with open(os.path.join(cls.root, name + ".map"), "rb") as f:
                maps[name] = gzip.compress(f.read())
        def read(n):
            with open(os.path.join(cls.root, n), "rb") as f:
                return f.read()
        xml = lambda num, pg, res: XML.replace("@NUM@", num).replace("@PG@", pg).replace("@RES@", res).encode()
        cls.srv, cls.url = T.serve(T.Files)
        T.Files.routes = {
            "/map/emd_80558.map.gz": maps["three"], "/cif/26DK.cif": read("three.cif"), "/header/emd-80558.xml": xml("80558", "C3", "3.5"),
            "/rcsb/entry/26DK": json.dumps({"rcsb_entry_info": {"assembly_count": 1}}).encode(),
            "/rcsb/assembly/26DK/1": json.dumps({"rcsb_assembly_info": {"polymer_instance_count": 3}, "pdbx_struct_assembly": {"oligomeric_count": 3},
                                                 "rcsb_struct_symmetry": [{"symbol": "C3", "kind": "Global Symmetry"}]}).encode(),
            "/map/emd_68251.map.gz": maps["one"], "/cif/22FX.cif": read("one.cif"), "/header/emd-68251.xml": xml("68251", "C1", "1.24"),
            "/map/emd_55586.map.gz": b"<html>" + b"not a map " * 30000, "/cif/9T5K.cif": read("one.cif"), "/header/emd-55586.xml": xml("55586", "C1", "2.6")}
        cls.entries = os.path.join(cls.root, "entries.csv")
        with open(os.path.join(ROOT, "data", "entries_2026.csv"), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows, fields = list(reader), reader.fieldnames
        for r in rows:
            if r["pdb_id"] in ("26DK", "22FX", "9T5K"):
                num = r["emdb_id"].split("-")[-1]
                r["map_url"], r["cif_url"] = "%s/map/emd_%s.map.gz" % (cls.url, num), "%s/cif/%s.cif" % (cls.url, r["pdb_id"])
        with open(cls.entries, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def run_notebook(self, entries):
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        nb_path = os.path.join(tmp, "nb.ipynb")
        sn.build(nb_path, entries=entries, entries_path=self.entries, net_hosts=(), rcsb_api=self.url + "/rcsb")
        nb = nbformat.read(nb_path, as_version=4)
        script = os.path.join(tmp, "nb.py")
        T.to_script(nb, script)
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1")
        r = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=900)
        work = os.path.join(tmp, "work")
        zips = sorted(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".zip")) if os.path.isdir(work) else []
        return r, zips, tmp

    @staticmethod
    def files(path):
        with zipfile.ZipFile(path) as z:
            return {n.split("/", 1)[1]: z.read(n) for n in z.namelist()}

    @staticmethod
    def rows(files, name):
        return list(csv.DictReader(io.StringIO(files[name].decode())))

    def test_a_complete_run_measures_the_copies_and_reads_the_header_and_the_assembly(self):
        r, zips, tmp = self.run_notebook(("26DK", "22FX"))
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out)
        self.assertEqual(len(zips), 1, zips)
        self.assertNotIn("_INCOMPLETE", zips[0])
        files = self.files(zips[0])
        rows = {x["entry"]: x for x in self.rows(files, "survey.csv")}
        self.assertEqual(list(rows), ["26DK", "22FX"])
        a, b = rows["26DK"], rows["22FX"]
        self.assertGreater(float(a["copies_of_model_estimate"]), 2.8)
        self.assertLess(float(a["copies_of_model_estimate"]), 3.5)
        self.assertLessEqual(float(b["copies_of_model_estimate"]), 1.15)
        self.assertEqual((a["point_group"], a["number_of_copies_header"], a["oligomeric_state_header"]), ("C3", "3", "trimer"))
        self.assertEqual((a["assembly_count"], a["assembly1_polymer_instances"], a["assembly1_oligomeric_count"], a["symmetry_rcsb"]), ("1", "3", "3", "C3 Global Symmetry"))
        self.assertEqual((a["contour_level"], a["resolution_header_A"], a["contour_source"]), ("0.25", "3.5", "EMDB header file"))
        self.assertEqual(a["sample_name"], "Synthetic assembly 80558")
        self.assertEqual((a["grid_x"], a["grid_y"], a["grid_z"], a["voxel_A"]), ("80", "80", "120", "1.0"))
        self.assertEqual(len(a["map_sha256"]), 64)
        self.assertGreater(float(a["docking_load_vs_26DK"]), 0)
        comps = self.rows(files, "survey_components.csv")
        self.assertEqual(sum(1 for c in comps if c["entry"] == "26DK"), 3)
        self.assertIn("RCSB entry", b["note"])                 # the missing RCSB answers of 22FX are only a note
        self.assertEqual(json.loads(files["state.json"])["complete"], True)
        self.assertFalse([n for n in files if n.endswith((".map", ".gz"))], "maps must not be packed")
        log = files["run.log"].decode()
        for text in ("ENTRY 26DK", "copies = volume above the contour level", "Mrefl"):
            self.assertIn(text, log)

    def test_an_entry_with_an_unreadable_map_is_recorded_and_the_next_entries_still_run(self):
        r, zips, tmp = self.run_notebook(("9T5K", "22FX"))
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("ENTRY 9T5K FAILED", out)
        self.assertIn("RESULT: NOT COMPLETE", out)
        self.assertTrue(zips[0].endswith("_INCOMPLETE.zip"), zips)
        files = self.files(zips[0])
        rows = {x["entry"]: x for x in self.rows(files, "survey.csv")}
        self.assertTrue(rows["9T5K"]["note"].startswith("FAILED: ValueError"), rows["9T5K"]["note"])
        self.assertIn("does not look like an MRC", rows["9T5K"]["note"])
        self.assertLessEqual(float(rows["22FX"]["copies_of_model_estimate"]), 1.15)
        state = json.loads(files["state.json"])
        self.assertIn("9T5K", state["stopped_at"])
        self.assertFalse(state["complete"])

    def test_an_entry_that_is_not_in_the_table_is_recorded_as_failed(self):
        r, zips, tmp = self.run_notebook(("22FX", "9ZZZ"))
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        rows = {x["entry"]: x for x in self.rows(self.files(zips[0]), "survey.csv")}
        self.assertIn("FAILED", rows["9ZZZ"]["note"])
        self.assertIn("KeyError", rows["9ZZZ"]["note"])

    def test_the_last_cell_alone_recovers_the_files_after_a_restart(self):
        r, zips, tmp = self.run_notebook(("22FX",))
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-2000:])
        nb = nbformat.read(os.path.join(tmp, "nb.ipynb"), as_version=4)
        final = [c.source for c in nb.cells if c["id"] == "final"][0]
        work = os.path.join(tmp, "work")
        for z in zips:
            os.remove(z)
        r2 = subprocess.run([sys.executable, "-c", final], cwd=tmp, env=dict(os.environ, MICA_WORK=work), capture_output=True, text=True, timeout=120)
        self.assertIn("RECOVERY: zip written", r2.stdout + r2.stderr)
        rec = [f for f in os.listdir(work) if f.endswith("_RECOVERED.zip")]
        self.assertEqual(len(rec), 1)
        self.assertIn("survey.csv", self.files(os.path.join(work, rec[0])))


if __name__ == "__main__":
    unittest.main()
