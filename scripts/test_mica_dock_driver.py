#!/usr/bin/env python3
"""Tests for mica_dock_driver.py (MICA's docking program with one method replaced).

The stand-in for Phenix is a bash function `phenix.dock_in_map` in an environment file, called the way MICA calls it (`bash -c 'source <env> && phenix.dock_in_map ...'`).
Its behaviour is set with environment variables: STUB_SLEEP (seconds before it writes the docked file), STUB_RC (exit code), STUB_TAG (a word in its command line,
so that the tests can look for processes that were left behind).

    python -W error -m unittest test_mica_dock_driver
"""
from __future__ import annotations

import csv
import importlib
import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mica_dock_driver as driver  # noqa: E402

FAKE_ENV = r'''
phenix.dock_in_map() {
    local out="" model="" a
    for a in "$@"; do
        case "$a" in pdb_out=*) out="${a#pdb_out=}";; search_model=*) model="${a#search_model=}";; esac
    done
    echo "$@" >> "$(dirname "$out")/calls.txt"
    sleep "${STUB_SLEEP:-0}" ; : "${STUB_TAG:-notag}"
    if [ -n "$STUB_PARTIAL" ]; then echo partial > "$out"; fi
    if [ "${STUB_RC:-0}" != "0" ]; then echo "Sorry: stub failure"; return "$STUB_RC"; fi
    cp "$model" "$out"
    echo "Wrote placed model with CC_mask (local CC) = 0.51"
}
'''


def make_object(tmp, env_file):
    return types.SimpleNamespace(log_directory=tmp, logger=logging.getLogger("test"), phenix_command=env_file, timestamp="20260101_000000")


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="drv_")
        self.env_file = os.path.join(self.tmp, "phenix_env.sh")
        with open(self.env_file, "w") as f:
            f.write(FAKE_ENV)
        self.model = os.path.join(self.tmp, "domain_01.pdb")
        with open(self.model, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        self.map = os.path.join(self.tmp, "map.mrc")
        self.fasta = os.path.join(self.tmp, "x.fasta")
        self.out = os.path.join(self.tmp, "domain_01_docked.pdb")
        self.obj = make_object(self.tmp, self.env_file)
        self.saved = {k: os.environ.get(k) for k in ("MICA_BENCH_DOCK_ATTEMPTS", "STUB_SLEEP", "STUB_RC", "STUB_TAG", "STUB_PARTIAL")}
        for k in self.saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        subprocess.run(["pkill", "-f", "drvtag"], check=False)

    def dock(self, attempts=None):
        if attempts is not None:
            os.environ["MICA_BENCH_DOCK_ATTEMPTS"] = json.dumps(attempts)
        return driver.run_phenix_dock_in_map(self.obj, self.map, self.model, 3.5, self.out, self.fasta)

    def rows(self):
        with open(os.path.join(self.tmp, "bench_docking.csv"), newline="") as f:
            return list(csv.DictReader(f))

    def calls(self):
        with open(os.path.join(self.tmp, "calls.txt")) as f:
            return [line.split() for line in f.read().splitlines()]

    def test_without_attempts_the_call_is_the_one_of_mica(self):
        self.assertTrue(self.dock())
        nproc = min(max(int(multiprocessing.cpu_count() * 0.6), 1), 48)
        self.assertEqual(self.calls(), [["search_model=" + self.model, "map_file=" + self.map, "nproc=%d" % nproc, "quick=False", "resolution=3.5", "pdb_out=" + self.out,
                                         "dock_chains_individually=True", "sequence=" + self.fasta]])
        self.assertTrue(os.path.exists(self.out))
        r = self.rows()
        self.assertEqual((len(r), r[0]["setting"], r[0]["docked"], r[0]["rc"], r[0]["local_cc"], r[0]["timed_out"]), (1, "mica", "True", "0", "0.51", "False"))

    def test_settings_replace_or_add_parameters(self):
        self.assertTrue(self.dock([{"name": "no_mr", "set": {"dock_with_mr": "False", "quick": "True"}, "timeout_s": 60}]))
        args = dict(a.split("=", 1) for a in self.calls()[0])
        self.assertEqual((args["quick"], args["dock_with_mr"], args["dock_chains_individually"], args["resolution"]), ("True", "False", "True", "3.5"))
        self.assertEqual(sum(a.startswith("quick=") for a in self.calls()[0]), 1)
        self.assertEqual(json.loads(self.rows()[0]["settings"]), {"dock_with_mr": "False", "quick": "True"})

    def test_a_search_that_runs_out_of_time_is_killed_with_everything_below_it(self):
        os.environ.update(STUB_SLEEP="60", STUB_TAG="drvtag")
        t0 = time.time()
        self.assertFalse(self.dock([{"name": "slow", "set": {}, "timeout_s": 2}]))
        self.assertLess(time.time() - t0, 20)
        r = self.rows()[0]
        self.assertEqual((r["timed_out"], r["docked"]), ("True", "False"))
        time.sleep(0.5)
        left = subprocess.run(["pgrep", "-f", "sleep 60"], capture_output=True, text=True).stdout.split()
        self.assertEqual(left, [], "a sleep of the stand-in is still running")
        self.assertFalse(os.path.exists(self.out))

    def test_the_second_attempt_runs_after_a_timeout_unless_it_says_skip(self):
        os.environ["STUB_SLEEP"] = "60"
        attempts = [{"name": "a", "set": {}, "timeout_s": 1}, {"name": "b", "set": {"quick": "True"}, "timeout_s": 60, "skip_after_timeout": True}]
        self.assertFalse(self.dock(attempts))
        r = self.rows()
        self.assertEqual([(x["setting"], x["skipped"], x["timed_out"]) for x in r], [("a", "False", "True"), ("b", "True", "False")])
        self.assertEqual(len(self.calls()), 1)

    def test_a_failed_attempt_is_followed_by_the_next_and_a_partial_file_is_removed(self):
        os.environ.update(STUB_RC="2", STUB_PARTIAL="1")
        attempts = [{"name": "a", "set": {}, "timeout_s": 60}, {"name": "b", "set": {"quick": "True"}, "timeout_s": 60}]
        self.assertFalse(self.dock(attempts))
        self.assertEqual([(x["rc"], x["docked"]) for x in self.rows()], [("2", "False"), ("2", "False")])
        self.assertFalse(os.path.exists(self.out))
        os.environ["STUB_RC"] = "0"
        os.environ.pop("STUB_PARTIAL")
        self.assertTrue(self.dock(attempts))
        self.assertEqual(self.rows()[-1]["attempt"], "1")

    def test_the_second_attempt_docks_when_the_first_fails(self):
        # the stand-in of this test refuses quick=True
        env_file2 = os.path.join(self.tmp, "env2.sh")
        with open(env_file2, "w") as f:
            f.write(FAKE_ENV.replace('echo "$@" >> ', 'case "$*" in *quick=True*) echo "Sorry: no" ; return 4;; esac; echo "$@" >> '))
        self.obj = make_object(self.tmp, env_file2)
        attempts = [{"name": "a", "set": {"quick": "True"}, "timeout_s": 60}, {"name": "b", "set": {"quick": "False"}, "timeout_s": 60}]
        self.assertTrue(self.dock(attempts))
        self.assertEqual([(x["setting"], x["rc"], x["docked"]) for x in self.rows()], [("a", "4", "False"), ("b", "0", "True")])
        self.assertTrue(os.path.exists(self.out))

    def test_the_search_resolution_changes_the_search_and_nothing_else(self):
        attempt = {"name": "res6", "set": {"dock_with_mr": "False", "quick": "True"}, "search_resolution": 6.0, "timeout_s": 60}
        self.assertTrue(self.dock([attempt]))
        call = self.calls()[0]
        args = dict(a.split("=", 1) for a in call)
        self.assertEqual(args["resolution"], "6.0")
        self.assertNotIn("rigid_body_refinement_resolution", args)             # Phenix 2.2.1 rejects it (a True/False parameter)
        self.assertNotIn("density_radius", args)
        self.assertEqual((args["quick"], args["dock_with_mr"], args["dock_chains_individually"], args["sequence"]), ("True", "False", "True", self.fasta))
        self.assertEqual(sum(a.startswith("resolution=") for a in call), 1, "resolution must not be given twice")
        self.assertEqual(len(call), 9)
        self.assertEqual(json.loads(self.rows()[0]["settings"]), {"dock_with_mr": "False", "quick": "True", "resolution": "6.0"})

    def test_the_derived_settings(self):
        d = driver.derived_settings
        self.assertEqual(d({"search_resolution": 6.0}, 3.07), {"resolution": "6.0"})
        self.assertEqual(d({"search_resolution": 10}, "2.6"), {"resolution": "10.0"})
        self.assertEqual(d({"search_resolution": 6.0}, 6.0), {})          # a map that is not finer than the search resolution is left alone
        self.assertEqual(d({"search_resolution": 6.0}, 7.5), {})
        self.assertEqual(d({}, 3.0), {})
        self.assertEqual(d({"search_resolution": None}, 3.0), {})

    def test_a_skip_after_timeout_looks_at_every_earlier_attempt(self):
        env_file3 = os.path.join(self.tmp, "env3.sh")
        with open(env_file3, "w") as f:                                   # 6.0 A: never finishes; 10.0 A: refused; the plain call would work
            f.write(FAKE_ENV.replace('sleep "${STUB_SLEEP:-0}"', 'case "$*" in *resolution=6.0*) sleep 60;; *resolution=10.0*) echo "Sorry: no"; return 4;; esac; sleep 0'))
        self.obj = make_object(self.tmp, env_file3)
        attempts = [{"name": "a", "set": {}, "search_resolution": 6.0, "timeout_s": 1}, {"name": "b", "set": {}, "search_resolution": 10.0, "timeout_s": 60},
                    {"name": "c", "set": {}, "timeout_s": 60, "skip_after_timeout": True}]
        self.assertFalse(self.dock(attempts))
        self.assertEqual([(x["setting"], x["timed_out"], x["rc"], x["skipped"]) for x in self.rows()], [("a", "True", "-9", "False"), ("b", "False", "4", "False"), ("c", "False", "", "True")])
        attempts[2]["skip_after_timeout"] = False
        self.assertTrue(self.dock(attempts))
        self.assertEqual(self.rows()[-1]["docked"], "True")

    def test_a_kill_of_the_whole_group_takes_phenix_with_it(self):
        """The notebook stops the driver at the end of its time by killing its process group: nothing of Phenix may stay behind."""
        script = os.path.join(self.tmp, "go.py")
        with open(script, "w") as f:
            f.write("import os, sys, types, logging\nsys.path.insert(0, %r)\nimport mica_dock_driver as d\n"
                    "o = types.SimpleNamespace(log_directory=%r, logger=logging.getLogger('t'), phenix_command=%r, timestamp='x')\n"
                    "d.run_phenix_dock_in_map(o, %r, %r, 3.5, %r, %r)\n" % (HERE, self.tmp, self.env_file, self.map, self.model, self.out, self.fasta))
        env = dict(os.environ, STUB_SLEEP="61", MICA_BENCH_DOCK_ATTEMPTS=json.dumps([{"name": "a", "set": {}, "timeout_s": None}]))
        p = subprocess.Popen([sys.executable, script], env=env, start_new_session=True)
        for _ in range(50):
            if subprocess.run(["pgrep", "-f", "sleep 61"], capture_output=True).returncode == 0:
                break
            time.sleep(0.1)
        else:
            self.fail("the stand-in did not start")
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        time.sleep(0.5)
        self.assertNotEqual(subprocess.run(["pgrep", "-f", "sleep 61"], capture_output=True).returncode, 0, "the sleep of the stand-in outlived the group kill")

    def test_main_replaces_only_the_one_method_of_mica(self):
        pkg = os.path.join(self.tmp, "fake_mica")
        os.makedirs(os.path.join(pkg, "utils"))
        with open(os.path.join(pkg, "utils", "__init__.py"), "w") as f:
            f.write("")
        with open(os.path.join(pkg, "utils", "dock_in_map.py"), "w") as f:
            f.write("class PhenixDockingProcessor:\n    def run_phenix_dock_in_map(self, *a):\n        return 'original'\n    def other(self):\n        return 'kept'\n\n"
                    "def main():\n    print('ran main', PhenixDockingProcessor().run_phenix_dock_in_map is not None)\n")
        out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import mica_dock_driver as d; d.main(); "
                              "from utils import dock_in_map as u; print(u.PhenixDockingProcessor.run_phenix_dock_in_map is d.run_phenix_dock_in_map, u.PhenixDockingProcessor().other())" % HERE],
                             cwd=pkg, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("ran main True", out.stdout)
        self.assertIn("True kept", out.stdout)

    def test_an_empty_list_of_attempts_is_refused(self):
        os.environ["MICA_BENCH_DOCK_ATTEMPTS"] = "[]"
        with self.assertRaises(ValueError):
            driver.load_attempts()


if __name__ == "__main__":
    unittest.main()
