#!/usr/bin/env python3
"""Offline tests for compute_cc_mask.py: a stand-in for phenix.map_model_cc and a local file server.

    python scripts/test_compute_cc_mask.py
"""
from __future__ import annotations

import csv
import gzip
import http.server
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compute_cc_mask as cc  # noqa: E402

STUB = """#!@PYTHON@
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
args = [os.path.basename(sys.argv[0])] + sys.argv[1:]
prog, rest = args[0], args[1:]
if prog == "phenix.version":
    print("Phenix version: stub-9.9-1234")
    sys.exit(0)
if not [a for a in rest if not a.startswith("--")] or "--help" in rest:
    print("usage: stub help text of " + prog)
    sys.exit(0)
with open(os.path.join(here, "calls.log"), "a") as f:
    f.write(json.dumps({"args": args, "cwd": os.path.basename(os.getcwd()), "files": sorted(os.listdir(".")), "time": time.time()}) + "\\n")
conf = json.load(open(os.path.join(here, "behaviors.json")))
flags = conf.get("_policy", [])
if prog == "phenix.map_model_cc":
    if "old_gone" in flags:
        print("Sorry: phenix.map_model_cc does not exist any more")
        sys.exit(1)
    if "--force" not in rest:
        print("Sorry: phenix.map_model_cc and phenix.model_map_cc are replaced by phenix.map_correlations Add the keyword '--force' to use anyway")
        sys.exit(1)
    if "force_first" in flags and rest[0] != "--force":
        print("Sorry: Unknown file or keyword: --force")
        sys.exit(1)
    rest = [a for a in rest if a != "--force"]
else:
    if "new_gone" in flags:
        print("Sorry: phenix.map_correlations cannot do this")
        sys.exit(1)
    if "new_full_names" in flags and not any(a.startswith("input_files.model=") for a in rest):
        print("Sorry: Unknown command line parameter definition: resolution = 2.6")
        sys.exit(1)


def value(prefix):
    for a in rest:
        if a.startswith(prefix):
            return a.split("=", 1)[1]


files = [a for a in rest if "=" not in a]
model = value("input_files.model=") or files[0]
mapfile = value("input_files.map_in_1=") or files[1]
assert os.path.getsize(mapfile) >= 1024 and open(model).read(5) == "data_"
pid = open(model).readline().strip()[5:]
mode = conf.get(pid, "ok")
block = "Overall:\\n  CC_mask  : 0.8123\\n  CC_volume: 0.80\\n  CC_peaks : 0.70\\n  CC_box   : 0.65\\n"
if prog == "phenix.map_correlations" and "to_file" in flags:
    with open("map_correlations.log", "w") as f:
        f.write(block)
    sys.exit(0)
if mode == "ok":
    print("some header lines\\n" + block)
elif mode == "ok2":
    print("Per chain\\n  CC_mask  : 0.7000\\n" + block)
elif mode == "noparse":
    print("Finished without numbers")
elif mode == "fail":
    print("Sorry: Model and map are not aligned\\nnext line")
    sys.exit(1)
elif mode == "slow":
    time.sleep(30)
elif mode == "killed":
    os.kill(os.getpid(), 9)
elif mode == "oldphenix":
    if "ignore_symmetry_conflicts=True" in " ".join(rest):
        print("Sorry: Unknown command line parameter definition: ignore_symmetry_conflicts = True")
        sys.exit(1)
    print(block)
elif mode == "bad":
    print("  CC_mask  : 1.5000")
elif mode == "nospace":
    print("OSError: [Errno 28] No space left on device")
    sys.exit(1)
elif mode == "wait_for_prefetch":
    for _ in range(200):                      # the next entries must be downloaded while this one is "computed"
        if "emd_60003" in open(conf["_hits"]).read():
            break
        time.sleep(0.1)
    else:
        print("Sorry: the next files were not downloaded in the background")
        sys.exit(1)
    print(block)
"""


def make_mrc(extra=b""):
    head = bytearray(1024)
    head[208:212] = b"MAP "
    return bytes(head) + b"\x00" * 2048 + extra


class Server:
    cif_404, cif_500, mirror_404, map_404, map_500, map_html, map_raw, map_trunc = set(), set(), set(), set(), set(), set(), set(), set()
    hits_path = None

    @classmethod
    def reset(cls):
        for name in ("cif_404", "cif_500", "mirror_404", "map_404", "map_500", "map_html", "map_raw", "map_trunc"):
            getattr(cls, name).clear()
        cls.hits_path = None


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body=b"", ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if Server.hits_path:
            with open(Server.hits_path, "a") as f:
                f.write(self.path + "\n")
        kind, name = self.path.strip("/").split("/", 1)
        if kind == "cif":
            pid = name.split(".")[0]
            if pid in Server.cif_500:
                return self.send(500)
            if pid in Server.cif_404:
                return self.send(404)
            return self.send(200, ("data_%s\n#\n_entry.id %s\n" % (pid, pid)).encode(), "text/plain")
        if kind == "mirror":
            pid = name.split(".")[0].upper()
            if pid in Server.mirror_404:
                return self.send(404)
            return self.send(200, gzip.compress(("data_%s\n#\n_entry.id %s\n" % (pid, pid)).encode()))
        if kind == "map":
            emd = name.split(".")[0]
            if emd in Server.map_500:
                return self.send(500)
            if emd in Server.map_404:
                return self.send(404)
            if emd in Server.map_html:
                return self.send(200, b"<html>maintenance</html>" * 100, "text/html")
            if emd in Server.map_raw:
                return self.send(200, make_mrc())
            if emd in Server.map_trunc:
                data = gzip.compress(make_mrc(os.urandom(20000)))
                return self.send(200, data[: len(data) // 2])        # a download that ended in the middle of the file
            return self.send(200, gzip.compress(make_mrc()))
        self.send(404)


V1 = "map_model_cc --force"
V2 = "map_model_cc --force first"
V3 = "map_correlations"
V4 = "map_correlations full names"
V5 = "map_model_cc --force, no symmetry option"


class CcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = "http://127.0.0.1:%d" % cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls._mirror = cc.MIRROR
        cc.MIRROR = cls.base + "/mirror/%s.cif.gz"

    @classmethod
    def tearDownClass(cls):
        cc.MIRROR = cls._mirror
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Server.reset()
        self.tmp = tempfile.mkdtemp(prefix="cc_test_")
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin)
        for name in ("phenix.map_model_cc", "phenix.map_correlations", "phenix.version"):
            self.install(self.bin, name)
        self.env = os.path.join(self.tmp, "phenix_env.sh")
        with open(self.env, "w") as f:
            f.write('export PATH="%s:$PATH"\n' % self.bin)
        self.todo = os.path.join(self.tmp, "cc_todo.csv")
        self.out = os.path.join(self.tmp, "cc_mask.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def install(folder, name):
        path = os.path.join(folder, name)
        with open(path, "w") as f:
            f.write(STUB.replace("@PYTHON@", sys.executable))
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)

    def set_todo(self, ids, behaviors=None, flags=()):
        conf = dict(behaviors or {})
        conf["_policy"] = list(flags)
        if Server.hits_path:
            conf["_hits"] = Server.hits_path
        with open(os.path.join(self.bin, "behaviors.json"), "w") as f:
            json.dump(conf, f)
        with open(self.todo, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["pdb_id", "emdb_id", "stratum", "seed_rank", "resolution_A", "cif_url", "map_url", "map_size_mb"])
            for i, p in enumerate(ids, 1):
                emd = "EMD-%d" % (60000 + i)
                w.writerow([p, emd, "le3.0", i, "2.6", "%s/cif/%s.cif" % (self.base, p),
                            "%s/map/emd_%d.map.gz" % (self.base, 60000 + i), "65.5"])

    def run_cc(self, *extra, env=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cc.main(["--todo", self.todo, "--out", self.out, "--phenix-env", env or self.env] + list(extra))
        return code, out.getvalue(), err.getvalue()

    def rows(self):
        with open(self.out, newline="") as f:
            return {r["pdb_id"]: r for r in csv.DictReader(f)}

    def calls(self, pdb_id=None):
        with open(os.path.join(self.bin, "calls.log")) as f:
            calls = [json.loads(line) for line in f]
        return [c for c in calls if pdb_id is None or pdb_id + ".cif" in c["args"] or any(a.endswith("=" + pdb_id + ".cif") for a in c["args"])]

    # ------------------------------------------------------------------
    def test_results_logs_and_the_exact_command(self):
        self.set_todo(["9AAA", "9BBB", "9CCC"], {"9BBB": "ok2", "9CCC": "fail"})
        code, out, err = self.run_cc()
        self.assertEqual(code, 0, err)
        rows = self.rows()
        self.assertEqual(list(rows), ["9AAA", "9BBB", "9CCC"])
        a = rows["9AAA"]
        self.assertEqual((a["status"], a["cc_mask"], a["resolution_used"], a["variant"], a["note"]), ("ok", "0.8123", "2.6", V1, ""))
        self.assertEqual(a["phenix_version"], "Phenix version: stub-9.9-1234")
        self.assertEqual(a["emdb_id"], "EMD-60001")
        self.assertEqual((rows["9BBB"]["status"], rows["9BBB"]["cc_mask"]), ("ok", "0.7000"))
        self.assertIn("2 CC_mask lines", rows["9BBB"]["note"])
        self.assertEqual((rows["9CCC"]["status"], rows["9CCC"]["cc_mask"], rows["9CCC"]["variant"]), ("failed", "", V1))
        self.assertIn("Sorry: Model and map are not aligned", rows["9CCC"]["note"])
        call = self.calls()[0]
        self.assertEqual(call["args"], ["phenix.map_model_cc", "9AAA.cif", "emd_60001.map", "resolution=2.6",
                                        "ignore_symmetry_conflicts=True", "--force"])
        self.assertEqual(call["files"], ["9AAA.cif", "emd_60001.map"])
        self.assertEqual(call["cwd"], "9AAA")
        self.assertEqual(len(self.calls("9CCC")), 1)                  # after a variant has worked, a failure is a result of the entry
        logs = os.path.join(self.tmp, "cc_logs")
        self.assertEqual(sorted(os.listdir(logs)), ["9AAA.log", "9BBB.log", "9CCC.log"])
        with open(os.path.join(logs, "9AAA.log")) as f:
            text = f.read()
        self.assertIn("CC_mask  : 0.8123", text)
        self.assertEqual(os.listdir(os.path.join(self.tmp, "cc_work")), [])      # the big files are gone
        self.assertIn("[1/3] 9AAA", out)

    def test_a_second_run_does_nothing_and_limit_counts_only_open_entries(self):
        self.set_todo(["9AAA", "9BBB", "9CCC"])
        self.assertEqual(self.run_cc("--limit", "2")[0], 0)
        self.assertEqual(list(self.rows()), ["9AAA", "9BBB"])
        code, out, _ = self.run_cc()
        self.assertEqual(code, 0)
        self.assertEqual(list(self.rows()), ["9AAA", "9BBB", "9CCC"])
        n = len(self.calls())
        code, out, _ = self.run_cc()
        self.assertEqual((code, len(self.calls()), len(self.rows())), (0, n, 3))
        self.assertIn("0 to do now", out)

    def test_show_output_prints_the_command_and_the_cc_lines(self):
        self.set_todo(["9AAA"])
        code, out, _ = self.run_cc("--show-output")
        self.assertEqual(code, 0)
        self.assertIn("| $ phenix.map_model_cc 9AAA.cif emd_60001.map resolution=2.6 ignore_symmetry_conflicts=True --force", out)
        self.assertIn("|   CC_mask  : 0.8123", out)
        self.assertIn("| [exit code 0]", out)

    def test_the_variants_are_tried_in_order_and_the_one_that_works_is_kept(self):
        cases = [
            # flags, variant that must win, calls for the first entry (variant names in order), arguments of the winning call
            ((), V1, 1, ["phenix.map_model_cc", "{m}", "{e}", "resolution=2.6", "ignore_symmetry_conflicts=True", "--force"]),
            (("force_first",), V2, 2, ["phenix.map_model_cc", "--force", "{m}", "{e}", "resolution=2.6", "ignore_symmetry_conflicts=True"]),
            (("old_gone",), V3, 3, ["phenix.map_correlations", "{m}", "{e}", "resolution=2.6", "ignore_symmetry_conflicts=True"]),
            (("old_gone", "new_full_names"), V4, 4, ["phenix.map_correlations", "input_files.model={m}", "input_files.map_in_1={e}",
                                                      "map_model_cc.resolution=2.6", "map_model_cc.ignore_symmetry_conflicts=True"]),
            (("old_gone", "to_file"), V3, 3, ["phenix.map_correlations", "{m}", "{e}", "resolution=2.6", "ignore_symmetry_conflicts=True"]),
        ]
        for flags, winner, n_first, win_args in cases:
            with self.subTest(flags=flags):
                for f in (self.out,):
                    if os.path.exists(f):
                        os.remove(f)
                open(os.path.join(self.bin, "calls.log"), "w").close()
                self.set_todo(["9AAA", "9BBB"], flags=flags)
                code, out, err = self.run_cc()
                self.assertEqual(code, 0, err)
                rows = self.rows()
                self.assertEqual([(r["status"], r["cc_mask"], r["variant"]) for r in rows.values()],
                                 [("ok", "0.8123", winner)] * 2)
                first, second = self.calls("9AAA"), self.calls("9BBB")
                self.assertEqual(len(first), n_first)
                self.assertEqual(first[-1]["args"], [a.format(m="9AAA.cif", e="emd_60001.map") for a in win_args])
                self.assertEqual(len(second), 1)                          # the second entry goes straight to the variant that worked
                self.assertEqual(second[0]["args"], [a.format(m="9BBB.cif", e="emd_60002.map") for a in win_args])

    def test_the_variant_that_worked_is_remembered_in_the_next_run(self):
        self.set_todo(["9AAA", "9BBB", "9CCC"], flags=("force_first",))
        self.assertEqual(self.run_cc("--limit", "1")[0], 0)
        self.assertEqual(len(self.calls("9AAA")), 2)
        code, out, _ = self.run_cc()
        self.assertEqual(code, 0)
        self.assertIn("variant of the earlier results (used for every entry): " + V2, out)
        self.assertEqual([len(self.calls(p)) for p in ("9BBB", "9CCC")], [1, 1])
        self.assertEqual({r["variant"] for r in self.rows().values()}, {V2})

    def test_every_variant_failing_is_a_failed_row_that_names_all_reasons(self):
        self.set_todo(["9AAA"], flags=("old_gone", "new_gone"))
        code, out, _ = self.run_cc()
        self.assertEqual(code, 0)
        row = self.rows()["9AAA"]
        self.assertEqual((row["status"], row["cc_mask"]), ("failed", ""))
        for name in (V1, V2, V3, V4):
            self.assertIn(name + ":", row["note"])
        self.assertNotIn("no symmetry option", row["note"])               # only tried when Phenix complains about that parameter
        self.assertEqual(len(self.calls("9AAA")), 4)
        with open(os.path.join(self.tmp, "cc_logs", "9AAA.log")) as f:
            text = f.read()
        self.assertEqual(text.count("=> "), 4)
        self.assertIn("Sorry: phenix.map_correlations cannot do this", text)

    def test_phenix_that_does_not_know_the_symmetry_option_gets_the_plain_command_for_every_entry(self):
        self.set_todo(["9AAA", "9BBB"], {"9AAA": "oldphenix", "9BBB": "oldphenix"})
        code, out, _ = self.run_cc()
        self.assertEqual(code, 0)
        self.assertEqual([(r["status"], r["variant"]) for r in self.rows().values()], [("ok", V5)] * 2)
        self.assertEqual(len(self.calls("9AAA")), 5)
        self.assertEqual(self.calls("9BBB")[0]["args"], ["phenix.map_model_cc", "9BBB.cif", "emd_60002.map", "resolution=2.6", "--force"])
        self.assertEqual(len(self.calls("9BBB")), 1)

    def test_files_are_downloaded_while_phenix_works_on_the_entry_before(self):
        Server.hits_path = os.path.join(self.tmp, "hits.log")
        open(Server.hits_path, "w").close()
        self.set_todo(["9AAA", "9BBB", "9CCC"], {"9AAA": "wait_for_prefetch"})
        code, out, err = self.run_cc()
        self.assertEqual(code, 0, err)
        self.assertEqual([r["status"] for r in self.rows().values()], ["ok", "ok", "ok"], out)
        self.assertEqual(os.listdir(os.path.join(self.tmp, "cc_work")), [])

    def test_temporary_download_problems_leave_no_row(self):
        self.set_todo(["9AAA", "9BBB", "9CCC", "9DDD", "9EEE"])
        Server.map_500.add("emd_60001")
        Server.cif_500.add("9BBB")
        Server.map_html.add("emd_60003")
        Server.map_trunc.add("emd_60005")
        original = cc.time.sleep
        cc.time.sleep = lambda s: None
        try:
            code, out, _ = self.run_cc()
        finally:
            cc.time.sleep = original
        self.assertEqual(code, 2)
        self.assertEqual(list(self.rows()), ["9DDD"])
        self.assertEqual(out.count("NOT DONE"), 4)
        self.assertIn("cannot unpack", out)
        self.assertIn("done: 1 ok, 0 failed (rejected), 4 not done", out)
        Server.reset()                                                  # the server is fine again
        code, _, _ = self.run_cc()
        self.assertEqual((code, sorted(self.rows())), (0, ["9AAA", "9BBB", "9CCC", "9DDD", "9EEE"]))

    def test_a_missing_map_is_a_result_and_a_missing_model_falls_back_to_the_archive(self):
        self.set_todo(["9AAA", "9BBB", "9CCC", "9DDD"])
        Server.map_404.add("emd_60001")
        Server.cif_404.update({"9BBB", "9CCC"})
        Server.mirror_404.add("9CCC")
        code, out, _ = self.run_cc()
        self.assertEqual(code, 0)
        rows = self.rows()
        self.assertEqual(rows["9AAA"]["status"], "failed")
        self.assertIn("HTTP 404", rows["9AAA"]["note"])
        self.assertEqual(rows["9BBB"]["status"], "ok")           # served by the mirror
        self.assertEqual(rows["9CCC"]["status"], "failed")       # missing in both places
        self.assertEqual(rows["9DDD"]["status"], "ok")

    def test_an_uncompressed_map_is_accepted(self):
        self.set_todo(["9AAA"])
        Server.map_raw.add("emd_60001")
        self.assertEqual(self.run_cc()[0], 0)
        self.assertEqual(self.rows()["9AAA"]["status"], "ok")

    def test_phenix_outcomes_once_a_variant_has_worked(self):
        ids = ["9OKA", "9AAA", "9BBB", "9CCC", "9EEE"]
        self.set_todo(ids, {"9AAA": "noparse", "9BBB": "killed", "9CCC": "slow", "9EEE": "bad"})
        code, out, _ = self.run_cc("--timeout", "2")
        self.assertEqual(code, 0)
        rows = self.rows()
        self.assertEqual(rows["9OKA"]["status"], "ok")
        self.assertEqual((rows["9AAA"]["status"], rows["9AAA"]["note"]), ("failed", "no CC_mask in the output"))
        self.assertEqual(rows["9BBB"]["status"], "failed")
        self.assertIn("killed by signal 9", rows["9BBB"]["note"])
        self.assertEqual((rows["9CCC"]["status"], rows["9CCC"]["note"]), ("failed", "timeout after 2 s"))
        self.assertEqual(rows["9EEE"]["status"], "failed")
        self.assertIn("outside -1..1", rows["9EEE"]["note"])
        self.assertEqual([len(self.calls(p)) for p in ids], [1] * 5)

    def test_a_full_disk_stops_everything_and_writes_no_row_for_that_entry(self):
        self.set_todo(["9AAA", "9BBB", "9CCC"], {"9BBB": "nospace"})
        code, _, err = self.run_cc()
        self.assertEqual(code, 3)
        self.assertIn("disk full", err)
        self.assertEqual(list(self.rows()), ["9AAA"])
        self.assertEqual(os.listdir(os.path.join(self.tmp, "cc_work")), [])

    def test_phenix_that_is_not_installed_stops_cleanly(self):
        self.set_todo(["9AAA"])
        code, _, err = self.run_cc(env=os.path.join(self.tmp, "nowhere", "phenix_env.sh"))
        self.assertEqual(code, 3)
        self.assertIn("ERROR", err)
        empty_env = os.path.join(self.tmp, "empty_env.sh")
        with open(empty_env, "w") as f:
            f.write("true\n")
        code, _, err = self.run_cc(env=empty_env)
        self.assertEqual(code, 3)
        self.assertFalse(os.path.exists(self.out))

    def test_phenix_without_any_of_the_programs_stops_and_writes_no_row(self):
        only_version = os.path.join(self.tmp, "bin_version_only")
        os.makedirs(only_version)
        self.install(only_version, "phenix.version")
        env = os.path.join(self.tmp, "version_env.sh")
        with open(env, "w") as f:
            f.write('export PATH="%s:$PATH"\n' % only_version)
        self.set_todo(["9AAA"])
        code, _, err = self.run_cc(env=env)
        self.assertEqual(code, 3)
        self.assertIn("none of the Phenix programs could be started", err)
        self.assertFalse(os.path.exists(self.out))

    def test_diagnose_writes_the_help_texts(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cc.main(["--diagnose", "--out", self.out, "--phenix-env", self.env])
        self.assertEqual(code, 0, err.getvalue())
        with open(os.path.join(self.tmp, "diagnostics.txt")) as f:
            text = f.read()
        for needle in ("$ phenix.version", "Phenix version: stub-9.9-1234", "$ phenix.map_correlations --help",
                       "usage: stub help text of phenix.map_correlations", "$ phenix.map_model_cc --force"):
            self.assertIn(needle, text)
        self.assertFalse(os.path.exists(self.out))

    def test_reading_the_output(self):
        self.assertEqual(cc.parse_cc_mask("a\n  CC_mask  : 0.8123\nCC_volume: 0.7")[0], 0.8123)
        self.assertEqual(cc.parse_cc_mask("Overall CC_mask= -0.12")[0], -0.12)
        self.assertEqual(cc.parse_cc_mask("cc_mask 0.5\n")[0], 0.5)
        self.assertEqual(cc.parse_cc_mask("CC_mask: 7e-1")[0], 0.7)
        self.assertEqual(cc.parse_cc_mask("| CC_mask | 0.77 |")[0], 0.77)
        self.assertEqual(cc.parse_cc_mask("map-model (CC_mask): 0.61")[0], 0.61)
        self.assertEqual(cc.parse_cc_mask("CC_mask  CC_volume\n 0.8 0.7")[0], None)
        self.assertEqual(cc.parse_cc_mask("cc_mask = True\ncompute.cc_mask=False\nCC_mask_min 0.5")[0], None)
        self.assertEqual(cc.parse_cc_mask("")[0], None)
        value, lines = cc.parse_cc_mask("chain A\n CC_mask: 0.5\nall\n CC_mask: 0.6")
        self.assertEqual((value, len(lines)), (0.5, 2))
        self.assertEqual(cc.short_error("x\nSorry: bad thing\nmore\nlast"), "Sorry: bad thing | more")
        self.assertEqual(cc.short_error("a\nb\n"), "b")
        self.assertTrue(cc.complains_about_option("Sorry: Unknown command line parameter definition: ignore_symmetry_conflicts = True"))
        self.assertFalse(cc.complains_about_option("Sorry: Model and map are not aligned"))
        self.assertFalse(cc.complains_about_option("ignore_symmetry_conflicts = False\nresolution = None"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
