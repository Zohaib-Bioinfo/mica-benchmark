#!/usr/bin/env python3
"""Tests for make_scoring_notebook.py and the notebook it builds (scoring of the models of the other notebooks).

Fast tests: the notebook builds, every cell compiles, the embedded files equal the repository files, the cells are in order, nothing of the other notebooks' secrets is in it.
End-to-end tests run the notebook as a script against stand-ins: a Phenix download server with a fake installer (its `phenix.chain_comparison` prints the table of the real program,
computed from the two files), a file server for the deposited models (as mmCIF), a folder of result zips that looks like Kaggle's /kaggle/input, and the real US-align (USALIGN).

    USALIGN=/path/to/USalign python -W error -m unittest test_scoring_notebook
"""
from __future__ import annotations

import csv
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
import make_scoring_notebook as ms  # noqa: E402
import test_pilot_notebook as T  # noqa: E402
import test_score_models as TS  # noqa: E402

USALIGN = os.environ.get("USALIGN")
USER, PASSWORD = T.USER, T.PASSWORD
STUB_PHENIX = ('#!@PYTHON@\nimport os, sys\nprog = os.path.basename(sys.argv[0])\nif prog == "phenix.version":\n    print("Phenix version: 2.2.1-6174 (stub)")\n    sys.exit(0)\n'
               'if os.environ.get("STUB_CC_MODE") == "odd":\n    print("a table with another layout")\n    sys.exit(0)\n' + TS.STUB_PY)


def committed(name):
    return os.path.join(HERE, name + ".ipynb")


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="scoring_nb_")
        cls.path = os.path.join(cls.tmp, "scoring.ipynb")
        cls.expected = ms.build(cls.path)
        cls.nb = nbformat.read(cls.path, as_version=4)
        cls.cells = {c["id"]: c for c in cls.nb.cells}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cells_compile_and_have_no_placeholders(self):
        for c in self.nb.cells:
            text = c.source
            self.assertEqual(re.findall(r"@[A-Z_0-9]+@", text), [], c["id"])
            if c.cell_type == "code" and not text.startswith("%%writefile"):
                compile(text, c["id"], "exec")

    def test_cell_order(self):
        ids = [c["id"] for c in self.nb.cells]
        want = ["intro", "settings", "step1", "step2-dirs", "step2-check", "robust", "phenix-guard", "step3", "phenix-after", "tools", "inputs", "natives", "score", "summary", "final"]
        pos = [ids.index(w) for w in want]
        self.assertEqual(pos, sorted(pos))

    def test_embedded_files_equal_the_repository_files(self):
        for c in self.nb.cells:
            if c.source.startswith("%%writefile"):
                first, text = c.source.split("\n", 1)
                target = first.split(None, 1)[1].strip()
                with open(os.path.join(ROOT, target), encoding="utf-8") as f:
                    self.assertEqual(f.read().rstrip("\n"), text.rstrip("\n"), target)
        self.assertIn("scripts/score_models.py", self.expected)

    def test_no_gpu_no_alphafold_files_and_phenix_secrets_only(self):
        text = "\n".join(c.source for c in self.nb.cells)
        self.assertIn("REQUIRE_GPU = False", text)
        self.assertIn("**no AlphaFold file**", self.nb.cells[0].source)
        self.assertNotIn("find_af3_dir(", text)
        self.assertNotIn("AF3_results", self.cells["natives"].source)

    def test_the_intro_names_the_notebooks_whose_outputs_are_inputs(self):
        intro = self.nb.cells[0].source
        for name in ("gpu_batch1", "gpu_batch2", "gpu_batch3", "mica_gpu", "Notebook Output Files"):
            self.assertIn(name, intro)

    def test_each_scope_names_its_own_inputs_and_nothing_else(self):
        want = {"base": (["gpu_batch1", "gpu_batch2", "gpu_batch3"], ["mica_gpu"]), "mica": (["mica_gpu"], ["gpu_batch1", "gpu_batch3", "mica_gpu_nodock"]),
                "nodock": (["mica_gpu_nodock"], ["gpu_batch1", "gpu_batch3", "mica_gpu"]),
                "mica2": (["mica_gpu2_a", "mica_gpu2_b"], ["gpu_batch1", "mica_gpu", "mica_gpu_nodock", "mica_gpu2_nodock_a"]),
                "mica3": (["mica_gpu3"], ["gpu_batch1", "mica_gpu", "mica_gpu2_a", "mica_gpu2_b", "mica_gpu_nodock"]),
                "nodock2": (["mica_gpu2_nodock_a", "mica_gpu2_nodock_b"], ["gpu_batch1", "mica_gpu", "mica_gpu_nodock", "mica_gpu2_a"])}
        for scope, (named, not_named) in want.items():
            path = os.path.join(self.tmp, "scoring_%s.ipynb" % scope)
            ms.build(path, name="scoring_" + scope, scope=scope)
            nb = nbformat.read(path, as_version=4)
            intro = {c["id"]: c for c in nb.cells}["intro"].source
            self.assertIn("session `scoring_%s`" % scope, intro)
            for c in nb.cells:
                self.assertEqual(re.findall(r"@[A-Z_0-9]+@", c.source), [], c["id"])
            added = intro.split("-> add")[1].split("(the names you gave them")[0]               # the sentence that says what to add
            for n in named:
                self.assertIn("`%s`" % n, added, scope)
            for n in not_named:
                self.assertNotIn("`%s`" % n, added, scope)
            self.assertIn("**and nothing else**", added)
            code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
            self.assertIn("-> " + ", ".join(named) + "), then run again", code)
        self.assertIn("`mica_gpu`", self.cells["intro"].source)                                     # the default scope (all) names every notebook

    def test_the_required_hosts_are_checked_before_any_time_is_spent(self):
        step1 = self.cells["step1"].source
        for host in ("https://files.rcsb.org/header/26DK.cif", "https://phenix-online.org/download", "https://github.com/jianlin-cheng/MICA", "https://pypi.org/simple/uv/"):
            self.assertIn(host, step1)

    def test_the_committed_notebook_is_current(self):
        if not os.path.exists(committed("scoring")):
            self.skipTest("scoring.ipynb is not built yet")
        self.assertEqual(nbformat.read(committed("scoring"), as_version=4).cells, self.nb.cells)


@unittest.skipUnless(USALIGN, "US-align is not available (set USALIGN)")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gemmi
        cls.root = tempfile.mkdtemp(prefix="scoringrun_")
        cls.fx = TS.Fixture(cls.root)
        st = gemmi.read_structure(cls.fx.native)
        st.setup_entities()
        cif_path = os.path.join(cls.root, "native.cif")
        st.make_mmcif_document().write_file(cif_path)
        with open(cif_path, "rb") as f:
            cif_bytes = f.read()
        cls.srv, cls.url = T.serve(T.Files)
        T.Files.routes = {"/cif/9ZZZ.cif": cif_bytes}
        cls.phenix_srv, cls.phenix_url = T.serve(T.PhenixServer)
        T.PhenixServer.body = T.FAKE_INSTALLER.replace("@STUB@", STUB_PHENIX.replace("@PYTHON@", sys.executable)).encode()
        cls.entries_csv = os.path.join(cls.root, "entries.csv")
        with open(cls.entries_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["emdb_id", "pdb_id", "modeled_residues", "cif_url"])
            w.writerow(["EMD-1", "9ZZZ", "200", cls.url + "/cif/9ZZZ.cif"])
            w.writerow(["EMD-2", "9YYY", "200", cls.url + "/cif/9YYY.cif"])          # no models: never downloaded
            w.writerow(["EMD-3", "9XXX", "200", cls.url + "/cif/9XXX.cif"])          # not served: no_native

    @classmethod
    def tearDownClass(cls):
        for s in (cls.srv, cls.phenix_srv):
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def build_script(self, tmp):
        nb_path = os.path.join(tmp, "nb.ipynb")
        url = "%s/download?version=2.2.1-6174&file=%s" % (self.phenix_url, T.INSTALLER_NAME)
        ms.build(nb_path, entries_path=self.entries_csv, phenix_url=url, usalign_bin=USALIGN, net_hosts=())
        nb = nbformat.read(nb_path, as_version=4)
        for c in nb.cells:
            if c["id"] == "step3":
                c.source = c.source.replace("MIN_INSTALLER_MB = 50", "MIN_INSTALLER_MB = 0").replace("MIN_FREE_GB = 20", "MIN_FREE_GB = 0")
        script = os.path.join(tmp, "nb.py")
        T.to_script(nb, script)
        return script

    @staticmethod
    def result_zip(path, label, files):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for rel, data in files.items():
                z.writestr(label + "/" + rel, data)

    def read(self, path):
        with open(path, "rb") as f:
            return f.read()

    def inputs(self, tmp, with_mica=True, duplicate=False, missing_native=False):
        inp = os.path.join(tmp, "input")
        fx = self.fx
        status = "method,entry,status\ne3,9ZZZ,ok\nmodelangelo,9ZZZ,ok\nemprot,9ZZZ,ok\ne3,9XXX,ok\n"
        files = {"models/e3/9ZZZ/9ZZZ_e3_prealign.pdb": self.read(fx.same), "models/modelangelo/9ZZZ/9ZZZ_modelangelo.pdb": self.read(fx.fragments),
                 "models/emprot/9ZZZ/output.pdb": self.read(fx.copies3), "method_status.csv": status.encode(), "state.json": b"{}", "run.log": b"log"}
        if missing_native:
            files["models/e3/9XXX/9XXX_e3_prealign.pdb"] = self.read(fx.same)
        self.result_zip(os.path.join(inp, "gpu-batch1", "gpu_batch1_result_20261023-1416.zip"), "gpu_batch1_result_20261023-1416", files)
        if duplicate:
            self.result_zip(os.path.join(inp, "gpu-batch2", "gpu_batch2_result_20261024-1900.zip"), "gpu_batch2_result_20261024-1900",
                            {"models/e3/9ZZZ/9ZZZ_e3_prealign.pdb": self.read(fx.copy2), "method_status.csv": b"method,entry,status\ne3,9ZZZ,ok\n"})
        if with_mica:
            self.result_zip(os.path.join(inp, "mica-cpu-a", "mica_cpu_a_result_20261024-2230.zip"), "mica_cpu_a_result_20261024-2230",
                            {"models/mica/9ZZZ/EMD-1_9zzz_MICA_all_atom_model.pdb": self.read(fx.copy2), "models/mica/9ZZZ/EMD-1_9zzz_MICA_ca_model.pdb": self.read(fx.copy2),
                             "mica_status.csv": b"entry,status\n9ZZZ,ok\n"})
            os.makedirs(os.path.join(inp, "mica-cpu-a", "mica_state", "9ZZZ", "AF3_results"), exist_ok=True)       # private state next to the zip: must be ignored
            with open(os.path.join(inp, "mica-cpu-a", "mica_state", "9ZZZ", "docked.pdb"), "w", encoding="utf-8") as f:
                f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        return inp

    def run_notebook(self, mica=True, duplicate=False, input_dir=None, phenix_mode=None, secrets=True, missing_native=False):
        tmp = tempfile.mkdtemp(prefix="run_", dir=self.root)
        script = self.build_script(tmp)
        env = dict(os.environ, MICA_WORK=os.path.join(tmp, "work"), MICA_BIG=os.path.join(tmp, "big"), MICA_BEAT="1",
                   SCORING_INPUT=input_dir if input_dir is not None else self.inputs(tmp, with_mica=mica, duplicate=duplicate, missing_native=missing_native))
        env.pop("STUB_CC_MODE", None)
        if phenix_mode:
            env["STUB_CC_MODE"] = phenix_mode
        if secrets:
            env.update(PHENIX_USER=USER, PHENIX_PASSWORD=PASSWORD)
        else:
            env.pop("PHENIX_USER", None)
            env.pop("PHENIX_PASSWORD", None)
        r = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True, text=True, timeout=900)
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

    def final_zip(self, zips):
        final = [z for z in zips if "_result_" in z]
        self.assertEqual(len(final), 1, zips)
        return final[0]

    # ------------------------------------------------------------------------------------------------------------------------------
    def test_a_complete_run(self):
        r, zips, tmp = self.run_notebook()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        z = self.final_zip(zips)
        self.assertNotIn("INCOMPLETE", z)
        files = self.zip_files(z)
        rows = self.table(files, "scoring/scores_long.csv")
        got = sorted((x["entry"], x["method"], x["model_kind"], x["primary"], x["status"]) for x in rows)
        self.assertEqual(got, [("9ZZZ", "e3", "e3_prealign", "True", "ok"), ("9ZZZ", "emprot", "emprot", "True", "ok"),
                               ("9ZZZ", "mica", "mica_all_atom", "True", "ok"), ("9ZZZ", "mica", "mica_ca", "False", "ok"), ("9ZZZ", "modelangelo", "modelangelo", "True", "ok")])
        by = {(x["method"], x["model_kind"]): x for x in rows if x["entry"] == "9ZZZ"}
        self.assertEqual(by[("e3", "e3_prealign")]["cc_status"], "ok")
        self.assertGreater(float(by[("e3", "e3_prealign")]["tm_score"]), 0.9)
        self.assertEqual(float(by[("emprot", "emprot")]["coverage"]), 3.0)
        self.assertGreater(float(by[("emprot", "emprot")]["ca_quality"]), 95.0)                    # three copies: precision * coverage cancels them out
        self.assertLess(float(by[("modelangelo", "modelangelo")]["tm_score"]), 0.35)               # fragments: penalised by the chain-wise score
        self.assertGreater(float(by[("modelangelo", "modelangelo")]["tm_score_fns"]), 0.9)         # and not by the fragment-agnostic one
        self.assertGreater(float(by[("mica", "mica_all_atom")]["ca_match"]), 95.0)                 # the copy elsewhere in the map, superposed
        self.assertLess(float(by[("mica", "mica_all_atom")]["ca_match_ff"]), 5.0)
        primary = self.table(files, "scoring/scores_primary.csv")
        self.assertEqual(len(primary), 4)                                                           # e3, emprot, modelangelo, mica_all_atom (not mica_ca)
        sources = self.table(files, "scoring/sources.csv")
        self.assertEqual([s["label"] for s in sources], ["gpu_batch1_result_20261023-1416", "mica_cpu_a_result_20261024-2230"])
        self.assertTrue(all(len(s["sha256"]) == 64 for s in sources))
        self.assertIn("status/gpu_batch1_result_20261023-1416__method_status.csv", files)
        self.assertIn("status/mica_cpu_a_result_20261024-2230__mica_status.csv", files)
        self.assertEqual([n for n in files if n.endswith((".pdb", ".cif"))], [])                  # no coordinates in the zip
        self.assertTrue(any(n.startswith("scoring/raw/9ZZZ/") and n.endswith("chain_comparison_superposed.txt") for n in files))
        info = json.loads(files["run_info.json"])
        self.assertTrue(info["complete"] and info["phenix_used"])
        self.assertEqual(info["protocol_version"], "1.0")
        self.assertIn("self-test on 9ZZZ", r.stdout)
        self.assertNotIn(PASSWORD, b"".join(files.values()).decode("utf-8", "replace"))
        self.assertEqual(sorted(f for f in os.listdir(os.path.join(tmp, "work", "native")) if f.endswith(".cif")), ["9ZZZ.cif"])      # 9YYY has no models and is not downloaded

    def test_an_entry_without_a_deposited_model_is_no_native_and_the_run_is_incomplete(self):
        r, zips, _ = self.run_notebook(missing_native=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-1500:])
        self.assertIn("could not be downloaded: its models are not scored (status no_native)", r.stdout)
        self.assertIn("INCOMPLETE: some entries have no deposited model", r.stdout)
        z = [x for x in zips if "_result_" in x][0]
        self.assertIn("INCOMPLETE", os.path.basename(z))
        rows = self.table(self.zip_files(z), "scoring/scores_long.csv")
        self.assertEqual(sorted((x["entry"], x["status"]) for x in rows if x["method"] == "e3"), [("9XXX", "no_native"), ("9ZZZ", "ok")])

    def test_a_model_in_two_sources_is_taken_from_the_later_one(self):
        r, zips, _ = self.run_notebook(mica=False, duplicate=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:])
        self.assertIn("WARNING: e3 9ZZZ 9ZZZ_e3_prealign.pdb: in gpu_batch1_result_20261023-1416 and in gpu_batch2_result_20261024-1900", r.stdout)
        rows = self.table(self.zip_files(self.final_zip(zips)), "scoring/scores_long.csv")
        e3 = [x for x in rows if x["entry"] == "9ZZZ" and x["method"] == "e3"][0]
        self.assertEqual(e3["source"], "gpu_batch2_result_20261024-1900")

    def test_without_phenix_the_notebook_goes_on_and_says_it_is_incomplete(self):
        r, zips, _ = self.run_notebook(secrets=False)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn("Phenix could not be installed", r.stdout)
        self.assertIn("INCOMPLETE: the Phenix columns are missing", r.stdout)
        z = self.final_zip(zips) if any("_result_" in x for x in zips) else zips[-1]
        files = self.zip_files(z)
        rows = [x for x in self.table(files, "scoring/scores_long.csv") if x["status"] == "ok"]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(x["cc_status"] == "not run" and x["ca_match"] == "" and x["own_ca_match"] != "" and x["tm_score"] != "" for x in rows))
        self.assertIn("INCOMPLETE", os.path.basename(z))

    def test_a_phenix_that_fails_the_self_test_is_left_out_and_the_rest_is_scored(self):
        r, zips, _ = self.run_notebook(phenix_mode="odd")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn("did not pass the self-test", r.stdout)
        z = [x for x in zips if "_result_" in x][0]
        self.assertIn("INCOMPLETE", os.path.basename(z))
        rows = [x for x in self.table(self.zip_files(z), "scoring/scores_long.csv") if x["status"] == "ok"]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(x["cc_status"] == "not run" for x in rows))

    def test_no_input_stops_with_the_instruction(self):
        tmp = tempfile.mkdtemp(prefix="empty_", dir=self.root)
        r, zips, _ = self.run_notebook(input_dir=tmp)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("Add Input -> Notebook Output Files", r.stdout)
        self.assertIn("RESULT: STOPPED", r.stdout)
        self.assertTrue(any("INCOMPLETE" in z for z in zips), zips)

    def test_a_broken_model_stays_in_the_table(self):
        tmp = tempfile.mkdtemp(prefix="broken_", dir=self.root)
        inp = os.path.join(tmp, "input")
        self.result_zip(os.path.join(inp, "gb1", "gpu_batch1_result_20261023-1416.zip"), "gpu_batch1_result_20261023-1416",
                        {"models/e3/9ZZZ/9ZZZ_e3_prealign.pdb": self.read(self.fx.same), "models/emprot/9ZZZ/output.cif": b"this is not a structure\n"})
        r, zips, _ = self.run_notebook(input_dir=inp)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-1500:])
        rows = self.table(self.zip_files([z for z in zips if "_result_" in z][0]), "scoring/scores_long.csv")
        self.assertEqual(sorted((x["method"], x["status"]) for x in rows), [("e3", "ok"), ("emprot", "no_model")])
        self.assertTrue([x for x in rows if x["method"] == "emprot"][0]["note"])


if __name__ == "__main__":
    unittest.main()
