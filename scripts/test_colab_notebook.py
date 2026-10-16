#!/usr/bin/env python3
"""Runs the built Colab notebook end to end in a local Jupyter kernel, against the mock servers of
test_pull_pool.py, and checks the zip it produces. Needs nbformat, nbclient and ipykernel.

    python test_colab_notebook.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import zipfile

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_colab_notebook as mk  # noqa: E402
import test_pull_pool as tp  # noqa: E402

SCRIPT = os.path.join(HERE, "pull_pool.py")
COMMITTED = os.path.join(HERE, "pull_pool_colab.ipynb")
STEP3_CMD_END = '"--out", OUT]'


def cell_text(nb):
    """All text the cells printed, in order."""
    parts = []
    for c in nb.cells:
        for o in c.get("outputs", []):
            if o.output_type == "stream":
                parts.append(o.text)
            elif o.output_type == "error":
                parts.append("ERROR %s: %s" % (o.ename, o.evalue))
    return "".join(parts)


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock = tp.Mock()
        cls.srv = tp.serve(cls.mock)
        base = "http://127.0.0.1:%d" % cls.srv.server_address[1]
        cls.mock_args = ["--search-url", base + "/search", "--cif-url", base + "/cif/{ID}.cif.gz",
                         "--cif-url", base + "/mirror/{id}.cif.gz", "--emdb-url", base + "/emdb/{EMD}",
                         "--threads", "4", "--emdb-threads", "2"]
        cls.root = tempfile.mkdtemp(prefix="nb_test_")
        cls.nb_path = os.path.join(cls.root, "pull_pool_colab.ipynb")
        mk.build(SCRIPT, cls.nb_path)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.expected, self.removed = tp.fresh(self.mock)
        self.passing = sorted(k for k, v in self.expected.items() if v is None)
        self.work = tempfile.mkdtemp(prefix="nb_work_", dir=self.root)

    def execute(self):
        nb = nbformat.read(self.nb_path, as_version=4)
        patched = 0
        for c in nb.cells:
            if c.id == "step3":
                self.assertEqual(c.source.count(STEP3_CMD_END), 1)
                c.source = c.source.replace(STEP3_CMD_END, STEP3_CMD_END + " + %r" % (self.mock_args,))
                self.assertEqual(c.source.count("time.sleep(30)"), 1)
                c.source = c.source.replace("time.sleep(30)", "time.sleep(0)")
                patched += 1
        self.assertEqual(patched, 1)
        client = NotebookClient(nb, timeout=600, kernel_name="python3",
                                resources={"metadata": {"path": self.work}})
        client.execute()
        return cell_text(nb)

    def open_zip(self, name):
        path = os.path.join(self.work, name)
        self.assertTrue(os.path.exists(path), "missing %s; have %s" % (name, os.listdir(self.work)))
        return zipfile.ZipFile(path)

    def test_success(self):
        text = self.execute()
        for s in ("Writing pull_pool.py", "canary OK", "DONE", "Step 3 finished with exit code 0",
                  "RESULT: OK. The pool is complete.", "zip written: pool_full_2026-10-12.zip",
                  "Entries left after each filter", "This is not Colab"):
            self.assertIn(s, text)
        self.assertNotIn("ERROR", text)
        z = self.open_zip("pool_full_2026-10-12.zip")
        names = set(z.namelist())
        folder = "pool_full_2026-10-12/"
        for n in ("pool_all_entries.csv", "pool_candidates.csv", "pool_flow.csv", "pool_candidate_sequences.fasta",
                  "pool_ids.txt", "pool_ids_all_cryoem.txt", "search_weeks.csv", "run_info.json", "run.log",
                  "exit_code.txt", "pull_pool.py"):
            self.assertIn(folder + n, names)
        self.assertNotIn(folder + "errors.csv", names)
        self.assertFalse(any("cache" in n for n in names))
        cands = z.read(folder + "pool_candidates.csv").decode().splitlines()
        ids = sorted(line.split(",")[0] for line in cands[1:])
        self.assertEqual(ids, self.passing)
        with open(SCRIPT, encoding="utf-8") as f:
            self.assertEqual(z.read(folder + "pull_pool.py").decode().rstrip("\n"), f.read().rstrip("\n"))
        self.assertEqual(z.read(folder + "exit_code.txt").decode().strip(), "0")

    def test_canary_failure_still_gives_a_zip_with_the_log(self):
        self.mock.cif_404.update({"5A1A", "6VXX"})
        text = self.execute()
        self.assertIn("CANARY FAILED", text)
        self.assertIn("Step 3 finished with exit code 3", text)
        self.assertIn("RESULT: STOPPED. The script could not read the real RCSB files", text)
        z = self.open_zip("pool_full_2026-10-12_exit3.zip")
        folder = "pool_full_2026-10-12_exit3/"
        self.assertIn(folder + "run.log", z.namelist())
        self.assertIn(folder + "run_info.json", z.namelist())
        self.assertNotIn(folder + "pool_candidates.csv", z.namelist())
        self.assertIn("CANARY FAILED", z.read(folder + "run.log").decode())

    def test_download_error_is_retried_and_reported(self):
        self.mock.cif_404.add("9P02")
        text = self.execute()
        self.assertIn("(attempt 2 of 3)", text)
        self.assertIn("(attempt 3 of 3)", text)
        self.assertNotIn("(attempt 4 of 3)", text)
        self.assertIn("Step 3 finished with exit code 2", text)
        self.assertIn("RESULT: INCOMPLETE", text)
        z = self.open_zip("pool_full_2026-10-12_exit2.zip")
        folder = "pool_full_2026-10-12_exit2/"
        self.assertIn("9P02", z.read(folder + "errors.csv").decode())

    def test_download_error_that_clears_up_on_retry(self):
        # fails twice, then works: the automatic retry must end with a complete pool
        calls = {"n": 0}
        original = self.mock.cif_404

        class Flaky(set):
            def __contains__(self, item):
                if item == "9P02":
                    calls["n"] += 1
                    return calls["n"] <= 2
                return False

        self.mock.cif_404 = Flaky()
        try:
            text = self.execute()
        finally:
            self.mock.cif_404 = original
        self.assertIn("(attempt 2 of 3)", text)
        self.assertIn("Step 3 finished with exit code 0", text)
        self.assertIn("RESULT: OK. The pool is complete.", text)
        z = self.open_zip("pool_full_2026-10-12.zip")
        cands = z.read("pool_full_2026-10-12/pool_candidates.csv").decode().splitlines()
        self.assertEqual(sorted(line.split(",")[0] for line in cands[1:]), self.passing)

    def test_killed_process_message(self):
        os.makedirs(os.path.join(self.work, "pool_out_full"))
        with open(os.path.join(self.work, "pool_out_full", "exit_code.txt"), "w") as f:
            f.write("-9\n")
        with open(os.path.join(self.work, "pool_out_full", "run.log"), "w") as f:
            f.write("[00:00:00] parsed 100/6116 entries\n")
        nb = nbformat.read(self.nb_path, as_version=4)
        nb.cells = [c for c in nb.cells if c.id == "step4"]
        client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": self.work}})
        client.execute()
        text = cell_text(nb)
        self.assertIn("RESULT: The script was killed by the system", text)
        self.open_zip("pool_full_2026-10-12_exit-9.zip")

    def test_search_failure(self):
        self.mock.reject_attrs.update({"exptl.method", "rcsb_entry_info.experimental_method"})
        text = self.execute()
        self.assertIn("Step 3 finished with exit code 4", text)
        self.assertIn("RESULT: STOPPED. The RCSB search did not work", text)
        self.open_zip("pool_full_2026-10-12_exit4.zip")

    def test_committed_notebook_is_current(self):
        self.assertTrue(os.path.exists(COMMITTED), "run make_colab_notebook.py")
        nb = nbformat.read(COMMITTED, as_version=4)
        nbformat.validate(nb)
        src = {c.id: c.source for c in nb.cells}
        with open(SCRIPT, encoding="utf-8") as f:
            script = f.read()
        self.assertEqual(src["step2"], "%%writefile pull_pool.py\n" + script)
        fresh_nb = nbformat.read(self.nb_path, as_version=4)
        self.assertEqual([c.source for c in nb.cells], [c.source for c in fresh_nb.cells])


if __name__ == "__main__":
    unittest.main(verbosity=2)
