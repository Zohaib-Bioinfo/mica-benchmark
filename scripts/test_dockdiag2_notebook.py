#!/usr/bin/env python3
"""Tests for version 2 of the docking test (make_dockdiag_notebook.py --version v2, scripts/dockdiag2.ipynb): the plan (resolution ladder, then three tests at the real
resolution with a long limit), the Phaser reflection count in the results, and the order of the groups. The machinery (installer, MICA environment, program 1, scoring,
recovery) is the one of version 1 and is tested in test_dockdiag_notebook.py; here the end-to-end tests of that file are reused with the new plan.

    USALIGN=... MICA_TEST_ENV_PYTHON=... MICA_TEST_AF3_DIR=... python -W error -m unittest test_dockdiag2_notebook
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_dockdiag_notebook as dd  # noqa: E402
import nbkit  # noqa: E402
import test_dockdiag_notebook as TD  # noqa: E402
import test_pilot_notebook as T  # noqa: E402

COMMITTED = os.path.join(HERE, "dockdiag2.ipynb")

# the stand-in Phenix also writes the Phaser log of the real one, with its reflection count
TD.STUB_PHENIX = TD.STUB_PHENIX.replace('''    say("Running in %s boxes" % kv.get("nproc"))''', '''    mr = os.path.join(os.path.dirname(kv["pdb_out"]), "run_mr")
    os.makedirs(mr, exist_ok=True)
    with open(os.path.join(mr, "phaser_input.log"), "w") as f:
        f.write("   Resolution Selected:     %s 265.60\\n   Number of Reflections in Selected Resolution Range: 915139\\n" % kv.get("resolution"))
    say("Running in %s boxes" % kv.get("nproc"))''')
assert "Number of Reflections" in TD.STUB_PHENIX


class PlanTests(unittest.TestCase):
    def test_the_plan_is_consistent(self):
        plan = dd.DOCK_PLAN_V2
        names = [s["name"] for s in plan]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted({s["group"] for s in plan}), [1, 2])
        for s in plan:
            self.assertIn("cap_min", s)
            self.assertNotIn("after_failure_of", s, "version 2 runs every test")
            self.assertNotIn("needs_a_success_of", s)

    def test_no_group_asks_for_more_processes_than_the_four_cores_of_kaggle(self):
        for g in (1, 2):
            used = sum(int(s["set"].get("nproc", 2)) for s in dd.DOCK_PLAN_V2 if s["group"] == g)      # MICA's own call uses nproc=2
            self.assertLessEqual(used, 4, "group %d" % g)

    def test_group_1_is_the_resolution_ladder_and_group_2_is_the_real_resolution(self):
        g1 = [s for s in dd.DOCK_PLAN_V2 if s["group"] == 1]
        g2 = [s for s in dd.DOCK_PLAN_V2 if s["group"] == 2]
        self.assertEqual(sorted((s["set"]["resolution"], s["set"].get("quick", "False")) for s in g1), [(6.0, "False"), (6.0, "True"), (10.0, "False"), (10.0, "True")])
        self.assertTrue(all("resolution" not in s["set"] for s in g2), "group 2 searches at the resolution of the map")
        self.assertTrue(all(s["cap_min"] == 45 for s in g1) and all(s["cap_min"] == 270 for s in g2))

    def test_one_test_reproduces_mica_exactly(self):
        mica = [s for s in dd.DOCK_PLAN_V2 if s["set"] == {}]
        self.assertEqual([s["name"] for s in mica], ["F1_mica_exact"])

    def test_the_total_time_fits_the_session_of_12_hours(self):
        total = 40 + sum(max(s["cap_min"] for s in dd.DOCK_PLAN_V2 if s["group"] == g) for g in (1, 2)) + 30       # setup, the two groups, scoring
        self.assertLess(total, 12 * 60 - 120)


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dd2build_")
        cls.path = os.path.join(cls.tmp, "nb.ipynb")
        cls.expected = dd.build(cls.path, dd_version="v2")
        cls.nb = nbformat.read(cls.path, as_version=4)

    def test_the_plan_the_name_and_the_intro_are_those_of_version_2(self):
        step7 = next(c for c in self.nb.cells if c["id"] == "step7").source
        self.assertIn(repr(dd.DOCK_PLAN_V2), step7)
        self.assertNotIn("A_quick_4cpu", step7)
        intro = self.nb.cells[0].source
        self.assertIn("how long does the Phenix docking of MICA really take", intro)
        self.assertIn("dockdiag2_result", intro)
        self.assertIn('NOTEBOOK = "dockdiag2"', next(c for c in self.nb.cells if c["id"] == "robust").source)

    def test_the_mica_call_is_unchanged(self):
        step7 = next(c for c in self.nb.cells if c["id"] == "step7").source
        self.assertIn('kv = {"search_model": DOCK["search_model"], "map_file": processed, "nproc": 2, "quick": "False", "resolution": DOCK["resolution"],', step7)

    def test_the_committed_notebook_is_current(self):
        if not os.path.exists(COMMITTED):
            self.skipTest("dockdiag2.ipynb is kept in as_run/ (the notebook that was run, not rebuilt)")
        self.assertEqual(nbformat.read(COMMITTED, as_version=4).cells, self.nb.cells)

    def test_version_1_is_still_the_first_test(self):
        tmp = tempfile.mkdtemp(prefix="dd1build_")
        path = os.path.join(tmp, "nb.ipynb")
        dd.build(path)
        nb = nbformat.read(path, as_version=4)
        self.assertIn(repr(dd.DOCK_PLAN), next(c for c in nb.cells if c["id"] == "step7").source)
        self.assertIn("why is the Phenix docking of MICA so slow", nb.cells[0].source)


@unittest.skipUnless(TD.ENV_PY and TD.AF3_DIR and TD.USALIGN and any("fold_26dk_e1_model_0.cif" in f for _r, _d, f in os.walk(TD.AF3_DIR or "")),
                     "needs MICA_TEST_ENV_PYTHON, MICA_TEST_AF3_DIR (with the 26DK file) and USALIGN")
class EndToEndTestsV2(TD.EndToEndTests):
    FAST_PLAN_V2 = [dict(s, cap_min=0.5) for s in dd.DOCK_PLAN_V2]

    def build_script(self, tmp, plan=None):
        nb_path = os.path.join(tmp, "nb.ipynb")
        url = "%s/download?version=2.2.1-6174&file=%s" % (self.phenix_url, T.INSTALLER_NAME)
        dd.build(nb_path, phenix_url=url, mica_repo="file://" + self.repo, mica_commit=self.commit, entries_path=self.entries, env_python=TD.ENV_PY,
                 usalign_bin=TD.USALIGN, net_hosts=(), plan=plan, dd_version="v2")
        nb = nbformat.read(nb_path, as_version=4)
        for c in nb.cells:
            if c["id"] == "step3":
                c.source = c.source.replace("MIN_INSTALLER_MB = 50", "MIN_INSTALLER_MB = 0").replace("MIN_FREE_GB = 20", "MIN_FREE_GB = 0")
        script = os.path.join(tmp, "nb.py")
        T.to_script(nb, script)
        return script, nb

    def test_both_groups_run_in_order_and_the_results_carry_the_reflection_count(self):
        r, zips, tmp = self.run_notebook(self.FAST_PLAN_V2)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT: OK", out, out[-3000:])
        self.assertTrue(out.index("docking tests of group 1") < out.index("docking tests of group 2"))
        files = self.zip_files(zips[0])
        rows = self.rows(files)
        self.assertEqual(list(rows), [s["name"] for s in dd.DOCK_PLAN_V2])
        for name, row in rows.items():
            self.assertEqual(row["docked"], "True", name)
            self.assertEqual(row["reflections"], "915139", name)
            self.assertIn("dock/%s/run_mr/phaser_input.log" % name, files)
        calls = lambda n: files["dock/%s/calls.txt" % n].decode()
        self.assertIn("resolution=10.0", calls("L1_mica_res10"))
        self.assertIn("quick=False", calls("L1_mica_res10"))
        self.assertIn("quick=True", calls("L4_quick_res6"))
        self.assertIn("dock_with_mr=False", calls("F3_no_mr_quick"))
        self.assertIn("nproc=2", calls("F1_mica_exact"))
        self.assertIn("quick=False", calls("F1_mica_exact"))
        self.assertNotIn("resolution=10.0", calls("F1_mica_exact"))
        self.assertIn("refl.", files["run.log"].decode())


for _name in [n for n in dir(TD.EndToEndTests) if n.startswith("test_")]:        # the tests of version 1 stay in their own file
    setattr(EndToEndTestsV2, _name, None)

if __name__ == "__main__":
    unittest.main()
