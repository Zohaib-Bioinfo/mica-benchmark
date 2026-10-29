#!/usr/bin/env python3
"""Tests for nbtools.py (progress lines, the zip, the recovery path).

    python -W error -m unittest test_nbtools
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nbtools  # noqa: E402


def write(path, text="x", binary=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if binary else "w") as f:
        f.write(text)
    return path


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nbtools_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work = os.path.join(self.tmp, "work")
        os.makedirs(self.work)


class LastLineTests(TmpCase):
    def test_last_non_empty_line_and_missing_file(self):
        p = write(os.path.join(self.tmp, "a.log"), "one\ntwo\n\n   \nthree  \n\n")
        self.assertEqual(nbtools.last_line(p), "three")
        self.assertEqual(nbtools.last_line(os.path.join(self.tmp, "nope")), "")
        self.assertEqual(nbtools.last_line(write(os.path.join(self.tmp, "empty"), "")), "")

    def test_long_line_is_cut_and_a_big_file_is_read_from_its_end(self):
        p = write(os.path.join(self.tmp, "big.log"), "first\n" + "x" * 50000 + "\nlast " + "y" * 500 + "\n")
        self.assertTrue(nbtools.last_line(p).startswith("last y"))
        self.assertEqual(len(nbtools.last_line(p)), 160)


class MachineLoadTests(unittest.TestCase):
    def test_returns_numbers_on_linux_and_never_raises(self):
        load, free, top = nbtools.machine_load()
        if os.path.exists("/proc/loadavg"):
            self.assertIsInstance(load, float)
        if os.path.exists("/proc/meminfo"):
            self.assertGreater(free, 0)
        self.assertIsInstance(top, str)

    def test_a_missing_ps_program_is_survived(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("ps")):
            self.assertEqual(nbtools.machine_load()[2], "")


class HeartbeatTests(TmpCase):
    def test_prints_lines_while_running_and_stops_quickly(self):
        lines = []
        watched = write(os.path.join(self.tmp, "phenix.log"), "starting\nbox 3 of 8\n")
        ticks = []
        hb = nbtools.Heartbeat("dock test", lines.append, watch=[watched, os.path.join(self.tmp, "missing.log")], every=0.1, on_tick=lambda: ticks.append(1))
        hb.start()
        time.sleep(0.55)
        t0 = time.time()
        hb.stop()
        self.assertLess(time.time() - t0, 2)
        self.assertGreaterEqual(len(lines), 3)
        self.assertGreaterEqual(len(ticks), 3)
        self.assertIn("[dock test] running for", lines[0])
        self.assertIn("last line: box 3 of 8", lines[0])
        self.assertIn("phenix.log: 20 bytes", lines[0])
        self.assertNotIn("missing.log", lines[0])
        self.assertFalse(hb.is_alive())

    def test_minutes_are_shown_after_the_first_minute(self):
        hb = nbtools.Heartbeat("x", print, every=1000)
        hb.t0 = time.time() - 125
        self.assertIn("running for 2 min", hb.line())

    def test_a_failing_log_function_or_tick_does_not_kill_the_thread(self):
        calls = {"n": 0}

        def bad(msg):
            calls["n"] += 1
            raise RuntimeError("log broke")

        hb = nbtools.Heartbeat("x", bad, every=0.05)
        hb.start()
        time.sleep(0.4)
        self.assertTrue(hb.is_alive())
        hb.stop()
        self.assertGreaterEqual(calls["n"], 3)

        lines = []
        hb2 = nbtools.Heartbeat("y", lines.append, every=0.05, on_tick=lambda: 1 / 0)
        hb2.start()
        time.sleep(0.4)
        hb2.stop()
        self.assertTrue(any("heartbeat problem" in ln for ln in lines))


class PackTests(TmpCase):
    def make_work(self):
        w = self.work
        write(os.path.join(w, "run.log"), "log\n")
        write(os.path.join(w, "state.json"), "{}")
        write(os.path.join(w, "logs", "a.log"), "a")
        write(os.path.join(w, "dock", "A", "phenix.log"), "phenix output")
        write(os.path.join(w, "dock", "A", "A_docked.pdb"), "ATOM")
        write(os.path.join(w, "dock", "initial_processed.mrc"), "map")
        write(os.path.join(w, "dock", "A", "superposed_map.ccp4"), "map")
        write(os.path.join(w, "emd_1.map"), "map")
        write(os.path.join(w, "AF3_results", "26dk_1", "fold.cif"), "private")
        write(os.path.join(w, "af3_unzipped", "x", "fold.cif"), "private")
        write(os.path.join(w, "native", "26DK.cif"), "public but not needed")
        write(os.path.join(w, "scripts", "nbtools.py"), "code")
        write(os.path.join(w, "data", "af3", "af3_jobs.csv"), "job")
        write(os.path.join(w, "something.zip"), "an older zip")
        write(os.path.join(w, "half.part"), "x")
        return w

    def names(self, zp):
        with zipfile.ZipFile(zp) as z:
            return sorted(z.namelist())

    def test_collect_skips_maps_private_files_and_embedded_copies(self):
        got = sorted(os.path.relpath(p, self.work) for p in nbtools.collect(self.make_work()))
        self.assertEqual(got, ["dock/A/A_docked.pdb", "dock/A/phenix.log", "logs/a.log", "run.log", "state.json"])

    def test_pack_writes_a_zip_with_a_root_folder(self):
        zp = os.path.join(self.tmp, "out.zip")
        n, left = nbtools.pack(self.make_work(), zp, "root")
        self.assertEqual(n, 5)
        self.assertEqual(left, [])
        self.assertEqual(self.names(zp), ["root/dock/A/A_docked.pdb", "root/dock/A/phenix.log", "root/logs/a.log", "root/run.log", "root/state.json"])
        self.assertFalse(os.path.exists(zp + ".tmp"))

    def test_a_file_with_the_password_and_a_big_file_are_left_out(self):
        w = self.make_work()
        write(os.path.join(w, "logs", "leak.log"), "curl -u user:s3cret-pass-987 http://x")
        write(os.path.join(w, "logs", "big.bin"), b"0" * 3000, binary=True)
        zp = os.path.join(self.tmp, "out.zip")
        n, left = nbtools.pack(w, zp, "root", secrets=["s3cret-pass-987", ""], max_file=2000)
        self.assertEqual(len(left), 2)
        self.assertTrue(any("leak.log" in x and "password" in x for x in left))
        self.assertTrue(any("big.bin" in x and "larger" in x for x in left))
        self.assertFalse(any("leak.log" in n or "big.bin" in n for n in self.names(zp)))
        with zipfile.ZipFile(zp) as z:
            self.assertFalse(any(b"s3cret-pass-987" in z.read(n) for n in z.namelist()))

    def test_the_old_zip_is_replaced_in_one_step_and_survives_a_failing_write(self):
        w = self.make_work()
        zp = os.path.join(self.tmp, "out.zip")
        nbtools.pack(w, zp, "root")
        with open(zp, "rb") as f:
            before = f.read()
        with mock.patch("zipfile.ZipFile.writestr", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                nbtools.pack(w, zp, "root")
        with open(zp, "rb") as f:
            self.assertEqual(f.read(), before, "the old zip must stay valid")
        self.assertFalse(os.path.exists(zp + ".tmp"), "a failed write must not leave its temporary file")
        with zipfile.ZipFile(zp) as z:
            self.assertIsNone(z.testzip())

    def test_extra_files_outside_the_work_folder_go_in_under_outside(self):
        w = self.make_work()
        other = write(os.path.join(self.tmp, "elsewhere", "model.pdb"), "ATOM")
        zp = os.path.join(self.tmp, "out.zip")
        nbtools.pack(w, zp, "root", extra=[other, os.path.join(self.tmp, "not_there")])
        self.assertTrue(any(n.startswith("root/outside/") and n.endswith("model.pdb") for n in self.names(zp)))

    def test_a_file_that_vanishes_during_packing_is_reported_not_fatal(self):
        w = self.make_work()
        real = nbtools.collect
        with mock.patch.object(nbtools, "collect", lambda work: real(work) + [os.path.join(work, "ghost.log")]):
            n, left = nbtools.pack(w, os.path.join(self.tmp, "o.zip"), "root", extra=[])
        self.assertEqual(n, 5)
        self.assertTrue(any("ghost.log" in x for x in left))


class FinalPackTests(TmpCase):
    def test_names_by_outcome_and_the_partial_zip_goes_away(self):
        write(os.path.join(self.work, "run.log"), "x")
        out = os.path.join(self.tmp, "out")
        os.makedirs(out)
        partial = nbtools.partial_zip_path(out, "nb")
        nbtools.pack(self.work, partial, "nb_partial")
        self.assertTrue(os.path.exists(partial))
        p1, n1, _ = nbtools.final_pack(self.work, out, "nb", True)
        self.assertRegex(os.path.basename(p1), r"^nb_result_\d{8}-\d{4}\.zip$")
        self.assertFalse(os.path.exists(partial))
        p2, _, _ = nbtools.final_pack(self.work, out, "nb", False)
        self.assertTrue(p2.endswith("_INCOMPLETE.zip"))
        p3, _, _ = nbtools.final_pack(self.work, out, "nb", None)
        self.assertTrue(p3.endswith("_RECOVERED.zip"))
        with zipfile.ZipFile(p1) as z:
            self.assertTrue(all(n.startswith(os.path.basename(p1)[:-4] + "/") for n in z.namelist()))


class RecoverTests(TmpCase):
    def test_nothing_to_pack_without_a_work_folder(self):
        said = []
        with mock.patch.object(nbtools, "WORK_CANDIDATES", ()), mock.patch("os.path.abspath", lambda p: os.path.join(self.tmp, "nowhere", p)):
            self.assertIsNone(nbtools.recover("nb", env={}, say=said.append))
        self.assertIn("no work folder", said[0])

    def test_recovers_from_disk_and_writes_run_info(self):
        write(os.path.join(self.work, "run.log"), "step 1 done\n")
        write(os.path.join(self.work, "state.json"), json.dumps({"ok": True, "entries": {"26DK": {"status": "failed"}}}))
        write(os.path.join(self.work, "dock", "A", "phenix.log"), "box 2 of 4")
        said = []
        path = nbtools.recover("nb", env={"MICA_WORK": self.work}, say=said.append)
        self.assertTrue(path.endswith("_RECOVERED.zip"))
        self.assertEqual(os.path.dirname(path), self.work)
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            info = json.loads(z.read([n for n in names if n.endswith("run_info.json")][0]))
        self.assertTrue(info["recovered"])
        self.assertTrue(info["state_json_found"])
        self.assertEqual(info["state"]["entries"]["26DK"]["status"], "failed")
        self.assertTrue(any(n.endswith("dock/A/phenix.log") for n in names))
        self.assertTrue(any("RECOVERY: zip written" in s for s in said))

    def test_recovers_without_state_json(self):
        write(os.path.join(self.work, "run.log"), "x")
        path = nbtools.recover("nb", env={"MICA_WORK": self.work}, say=lambda m: None)
        with zipfile.ZipFile(path) as z:
            info = json.loads(z.read([n for n in z.namelist() if n.endswith("run_info.json")][0]))
        self.assertFalse(info["state_json_found"])

    def test_the_password_is_removed_after_a_restart_when_it_can_be_read_again(self):
        write(os.path.join(self.work, "run.log"), "x")
        write(os.path.join(self.work, "logs", "leak.log"), "password=hunter2-hunter2")
        path = nbtools.recover("nb", env={"MICA_WORK": self.work, "PHENIX_PASSWORD": "hunter2-hunter2"}, say=lambda m: None)
        with zipfile.ZipFile(path) as z:
            self.assertFalse(any(n.endswith("leak.log") for n in z.namelist()))

    def test_known_secrets_ignores_short_values(self):
        self.assertEqual(nbtools.known_secrets({"PHENIX_PASSWORD": "abc"}), [])
        self.assertEqual(nbtools.known_secrets({"PHENIX_PASSWORD": " longpassword "}), ["longpassword"])

    def test_find_work_prefers_the_test_setting(self):
        write(os.path.join(self.work, "run.log"), "x")
        self.assertEqual(nbtools.find_work({"MICA_WORK": self.work}), self.work)

    def test_command_line(self):
        write(os.path.join(self.work, "run.log"), "x")
        out = os.path.join(self.tmp, "o")
        os.makedirs(out)
        r = subprocess.run([sys.executable, os.path.join(HERE, "nbtools.py"), "--work", self.work, "--out", out, "--notebook", "nb"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(os.listdir(out)), 1)


if __name__ == "__main__":
    unittest.main()
