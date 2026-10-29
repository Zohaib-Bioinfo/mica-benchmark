#!/usr/bin/env python3
"""Tests for ingest_results.py.

    python -W error -m unittest test_ingest_results
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ingest_results as I  # noqa: E402


def make_zip(path, files, top=None):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr((top + "/" if top else "") + name, data)


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ingest_")
        self.out = os.path.join(self.tmp, "raw")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def zip(self, name, files, top="same"):
        path = os.path.join(self.tmp, name)
        make_zip(path, files, top=name[:-4] if top == "same" else top)
        return path

    def test_text_files_are_kept_and_coordinates_never(self):
        z = self.zip("gpu_batch2_result_20261024-1334.zip", {"method_status.csv": "a,b\n1,2\n", "logs/x.log": "hello", "models/mica/26DK/m.pdb": "ATOM", "models/e3/26DK/m.cif": "data_",
                                                            "weights.pth": "x", "map.mrc": "y", "run_info.json": "{}"})
        row = I.ingest(z, self.out)
        base = os.path.join(self.out, "gpu_batch2_result_20261024-1334")
        self.assertTrue(os.path.exists(os.path.join(base, "method_status.csv")))
        self.assertTrue(os.path.exists(os.path.join(base, "logs", "x.log")))
        self.assertFalse(any(f.endswith((".pdb", ".cif", ".pth", ".mrc")) for _r, _d, fs in os.walk(self.out) for f in fs))
        self.assertEqual((row["files_kept"], row["files_skipped"], row["notebook"], row["stamp"], row["incomplete"]), (3, 4, "gpu_batch2", "20261024-1334", False))
        self.assertEqual(row["skipped_suffixes"], ".cif:1;.mrc:1;.pdb:1;.pth:1")
        self.assertEqual(len(row["zip_sha256"]), 64)

    def test_the_folder_of_the_zip_is_stripped_only_when_there_is_one(self):
        z = self.zip("mica_gpu_result_20261026-1200_INCOMPLETE.zip", {"mica_status.csv": "x\n"})
        row = I.ingest(z, self.out)
        self.assertTrue(row["incomplete"])
        self.assertTrue(os.path.exists(os.path.join(self.out, "mica_gpu_result_20261026-1200_INCOMPLETE", "mica_status.csv")))
        z2 = self.zip("mica_gpu_result_20261026-1300.zip", {"a/mica_status.csv": "x\n", "b.csv": "y\n"}, top=None)
        I.ingest(z2, self.out)
        base = os.path.join(self.out, "mica_gpu_result_20261026-1300")
        self.assertTrue(os.path.exists(os.path.join(base, "a", "mica_status.csv")))
        self.assertTrue(os.path.exists(os.path.join(base, "b.csv")))

    def test_a_member_that_leaves_the_folder_refuses_the_whole_zip(self):
        for bad in ("../evil.csv", "/abs/evil.csv", "a/../../evil.csv"):
            path = os.path.join(self.tmp, "x_result_20261025-0000.zip")
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("ok.csv", "1")
                z.writestr(bad, "2")
            with self.assertRaises(ValueError):
                I.ingest(path, self.out)
            self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil.csv")))
            self.assertFalse(os.path.exists(os.path.join(self.out, "x_result_20261025-0000", "ok.csv")))

    def test_a_zip_with_another_name_is_refused(self):
        path = self.zip("results.zip", {"a.csv": "1"})
        with self.assertRaises(ValueError):
            I.ingest(path, self.out)

    def test_the_index_and_the_order_of_the_status_tables(self):
        zs = [self.zip("gpu_batch2_result_20261024-1334.zip", {"method_status.csv": "b\n"}), self.zip("gpu_batch1_result_20261023-1416.zip", {"method_status.csv": "a\n"}),
              self.zip("mica_gpu_result_20261026-0900.zip", {"mica_status.csv": "m\n"}), self.zip("gpu_batch3_result_20261025-2300.zip", {"method_status.csv": "c\n"})]
        self.assertEqual(I.main(zs + ["--out", self.out]), 0)
        index = I.read_index(self.out)
        self.assertEqual([r["folder"] for r in index], ["gpu_batch1_result_20261023-1416", "gpu_batch2_result_20261024-1334", "gpu_batch3_result_20261025-2300", "mica_gpu_result_20261026-0900"])
        ms = [os.path.basename(os.path.dirname(p)) for p in I.status_files(self.out, "method_status.csv")]
        self.assertEqual(ms, ["gpu_batch1_result_20261023-1416", "gpu_batch2_result_20261024-1334", "gpu_batch3_result_20261025-2300"])
        self.assertEqual(len(I.status_files(self.out, "mica_status.csv")), 1)
        # reading the same zip again replaces its line, it does not add one
        I.main([zs[0], "--out", self.out])
        self.assertEqual(len(I.read_index(self.out)), 4)
        with open(os.path.join(self.out, "INDEX.csv"), newline="") as f:
            self.assertEqual(next(csv.reader(f)), I.INDEX_COLUMNS)

    def test_a_large_text_file_is_not_kept(self):
        old = I.MAX_FILE_BYTES
        I.MAX_FILE_BYTES = 10
        try:
            row = I.ingest(self.zip("a_result_20261025-0000.zip", {"small.csv": "1", "big.csv": "x" * 100}), self.out)
        finally:
            I.MAX_FILE_BYTES = old
        self.assertEqual((row["files_kept"], row["skipped_suffixes"]), (1, "(too large):1"))


if __name__ == "__main__":
    unittest.main()
