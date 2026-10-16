#!/usr/bin/env python3
"""Runs the built select_colab notebook end to end in a local Jupyter kernel against local stand-ins: a Phenix download
server with a user name and password, a fake Phenix installer, a file server for models and maps, and the RCSB search and
pdb_seqres mocks of the novelty test. Needs nbformat, nbclient and ipykernel. The novelty part needs MMseqs2
(MMSEQS environment variable or PATH); without it the test expects the notebook to report that step as failed.

    python test_select_notebook.py
"""
from __future__ import annotations

import base64
import io
import csv
import glob
import hashlib
import http.server
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile
from unittest import mock

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_select_notebook as mk  # noqa: E402
import select_entries as se  # noqa: E402
import test_compute_cc_mask as tcc  # noqa: E402
import test_novelty_check as tnv  # noqa: E402
import test_select_entries as tse  # noqa: E402

COMMITTED = os.path.join(HERE, "select_colab.ipynb")
MMSEQS = os.environ.get("MMSEQS") or shutil.which("mmseqs")
USER, PASSWORD = "pxuser-1", "s3cret-pass-123"
INSTALLER_NAME = "Phenix-2.2.1-6174-Linux-x86_64.sh"

FAKE_INSTALLER = r"""#!/bin/bash
# fake Phenix installer for the notebook test: bash installer -b -p PREFIX
while [ $# -gt 0 ]; do
  case "$1" in -p) PREFIX="$2"; shift ;; esac
  shift
done
mkdir -p "$PREFIX/bin"
for i in 1 2 3; do echo "installing part $i"; done
cat > "$PREFIX/bin/behaviors.json" <<'JSONEOF'
@BEHAVIORS@
JSONEOF
cat > "$PREFIX/bin/phenix.map_model_cc" <<'PYEOF'
#!@PYTHON@
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
prog = os.path.basename(sys.argv[0])
args = sys.argv[1:]
if prog == "phenix.version":
    if os.environ.get("STUB_NOVERSION"):
        sys.exit(1)
    print("Phenix version: 2.2.1-6174 (stub)")
    sys.exit(0)
conf = json.load(open(os.path.join(here, "behaviors.json")))
if not [a for a in args if not a.startswith("--")] or "--help" in args:
    print("usage: stub help text of " + prog)
    sys.exit(0)
if prog == "phenix.map_model_cc":                       # like Phenix 2.2.1: retired, works only with --force
    if "old_gone" in conf.get("_policy", []):
        print("Sorry: phenix.map_model_cc does not exist any more")
        sys.exit(1)
    if "--force" not in args:
        print("Sorry: phenix.map_model_cc and phenix.model_map_cc are replaced by phenix.map_correlations Add the keyword '--force' to use anyway")
        sys.exit(1)
    args = [a for a in args if a != "--force"]
def value(prefix):
    for a in args:
        if a.startswith(prefix):
            return a.split("=", 1)[1]


files = [a for a in args if "=" not in a and not a.startswith("--")]
model = value("input_files.model=") or files[0]
mapfile = value("input_files.map_in_1=") or files[1]
assert os.path.getsize(mapfile) >= 1024 and "resolution=" in " ".join(args)
pid = open(model).readline().strip()[5:]
mode = conf.get(pid, "cc=0.80")
if mode == "fail":
    print("Sorry: stub failure")
    sys.exit(1)
if mode == "leak":
    print("LEAK " + os.environ.get("PHENIX_PASSWORD", ""))
    mode = "cc=0.81"
print("Overall:\n  CC_mask  : %s\n  CC_volume: 0.5" % mode.split("=")[1])
PYEOF
cp "$PREFIX/bin/phenix.map_model_cc" "$PREFIX/bin/phenix.version"
cp "$PREFIX/bin/phenix.map_model_cc" "$PREFIX/bin/phenix.map_correlations"
chmod +x "$PREFIX/bin/"phenix.*
echo "export PATH=\"$PREFIX/bin:\$PATH\"" > "$PREFIX/phenix_env.sh"
"""


class Phenix(http.server.BaseHTTPRequestHandler):
    body = b""
    html = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        want = "Basic " + base64.b64encode(("%s:%s" % (USER, PASSWORD)).encode()).decode()
        if self.headers.get("Authorization") != want:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="phenix"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = b"<html>please log in</html>" if Phenix.html else Phenix.body
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Tar(http.server.BaseHTTPRequestHandler):
    path_to_file = ""

    def log_message(self, *a):
        pass

    def do_GET(self):
        with open(Tar.path_to_file, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(handler):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def cell_text(nb):
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
        cls.files, cls.files_url = serve(tcc.Handler)
        cls.phenix, cls.phenix_url = serve(Phenix)
        cls.rcsb, cls.rcsb_url = serve(tnv.Handler)
        cls.root = tempfile.mkdtemp(prefix="nbsel_")
        # pool: 7 entries at <= 3.0 A, 6 above; 9T04 and 9T05 are in one cluster, 9T08 and 9T09 in another
        cls.world = w = tse.World()
        cls.ids = ["9T%02d" % i for i in range(1, 14)]
        for i, pid in enumerate(cls.ids, 1):
            row = tse.candidate(pid, "EMD-%d" % (70000 + i), "le3.0" if i <= 7 else "gt3.0_le4.0", "2.50" if i <= 7 else "3.50")
            row["cif_url"] = "%s/cif/%s.cif" % (cls.files_url, pid)
            row["map_url"] = "%s/map/emd_%d.map.gz" % (cls.files_url, 70000 + i)
            w.cands.append(row)
            w.rep_of[pid] = {4: "K1", 5: "K1", 8: "K2", 9: "K2"}.get(i, pid)
        cls.pool = os.path.join(cls.root, "pool")
        tse.write_pool_files(cls.pool, w.cands)
        seqs = [tnv.Q1, tnv.Q2, tnv.Q3, tnv.R4]
        with open(os.path.join(cls.pool, "pool_candidate_sequences.fasta"), "w") as f:
            for i, p in enumerate(cls.ids):
                s = seqs[i % 4]
                f.write(">%s_E1 len=%d copies=1 longest=1\n%s\n" % (p, len(s), s))
        with open(os.path.join(cls.pool, "pool_ids_all_cryoem.txt"), "w") as f:
            f.write("\n".join(cls.ids) + "\n")
        cls.clusters = os.path.join(cls.root, "candidate_clusters.csv")
        tse.write_cluster_file(cls.clusters, w)
        for p in cls.ids:                                           # the mock RCSB lists them as released in 2026
            tnv.RELEASE[p] = "2026-04-04"
        # what the fake Phenix answers
        cls.behaviors = {"9T02": "cc=0.55", "9T03": "fail", "9T07": "cc=0.9", "9T10": "cc=0.65"}
        # a local MMseqs2 tarball
        cls.tgz = os.path.join(cls.root, "mmseqs.tar.gz")
        cls.mm_hash = ""
        if MMSEQS:
            with tarfile.open(cls.tgz, "w:gz") as t:
                t.add(MMSEQS, arcname="mmseqs/bin/mmseqs")
            with open(cls.tgz, "rb") as f:
                cls.mm_hash = hashlib.sha256(f.read()).hexdigest()
            Tar.path_to_file = cls.tgz
            cls.tar_srv, cls.tar_url = serve(Tar)

    @classmethod
    def tearDownClass(cls):
        for s in (cls.files, cls.phenix, cls.rcsb) + ((cls.tar_srv,) if MMSEQS else ()):
            s.shutdown()
            s.server_close()
        for p in cls.ids:
            tnv.RELEASE.pop(p, None)
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        tcc.Server.reset()
        tnv.Mock.max_rows, tnv.Mock.reject, tnv.Mock.hidden = None, False, set()
        Phenix.html = False
        self.set_behaviors(dict(self.behaviors))
        self.work = tempfile.mkdtemp(prefix="nbsel_work_", dir=self.root)
        self.nb_path = os.path.join(self.root, "select_colab_test.ipynb")
        url = "%s/download?version=2.2.1-6174&file=%s" % (self.phenix_url, INSTALLER_NAME)
        mk.build(self.nb_path, pool_dir=self.pool, clusters_path=self.clusters, phenix_url=url)

    def set_behaviors(self, behaviors):
        self.behaviors = behaviors
        Phenix.body = FAKE_INSTALLER.replace("@PYTHON@", sys.executable).replace("@BEHAVIORS@", json.dumps(behaviors)).encode()

    def patched(self, nb):
        def sub(cell_id, old, new):
            for c in nb.cells:
                if c.id == cell_id:
                    self.assertEqual(c.source.count(old), 1, "%s: %r" % (cell_id, old))
                    c.source = c.source.replace(old, new)
                    return
            self.fail("no cell " + cell_id)
        sub("step3", "MIN_INSTALLER_MB = 50", "MIN_INSTALLER_MB = 0")
        sub("step3", "time.sleep(10 * attempt)", "time.sleep(0)")
        sub("step4", "BATCH = %d" % mk.BATCH, "BATCH = 3")
        extra = ["--search-url", self.rcsb_url + "/search", "--seqres-url", self.rcsb_url + "/seqres.gz", "--through", "2026-10-13",
                 "--min-reference-chains", "1"]
        sub("step6", "NOVELTY_EXTRA = []", "NOVELTY_EXTRA = %r" % (extra,))
        if MMSEQS:
            sub("step6", 'MMSEQS_URL = "%s"' % mk.MMSEQS_URL, 'MMSEQS_URL = "%s/mmseqs.tar.gz"' % self.tar_url)
            sub("step6", "MMSEQS_SHA256 = %r" % (mk.MMSEQS_SHA256,), "MMSEQS_SHA256 = %r" % ({"avx2": self.mm_hash, "sse41": self.mm_hash},))

    def execute(self, user=USER, password=PASSWORD):
        nb = nbformat.read(self.nb_path, as_version=4)
        self.patched(nb)
        env = {"PHENIX_USER": user, "PHENIX_PASSWORD": password, "MICA_WORK": os.path.join(self.work, "work")}
        with mock.patch.dict(os.environ, env):
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": self.work}}).execute()
        return cell_text(nb)

    def zip_files(self):
        found = [f for f in os.listdir(os.path.join(self.work, "work")) if f.startswith("selection_result_") and f.endswith(".zip")]
        self.assertEqual(len(found), 1, found)
        z = zipfile.ZipFile(os.path.join(self.work, "work", found[0]))
        return found[0], z, {n.split("/", 1)[1]: n for n in z.namelist()}

    def expected_selection(self):
        oracle = {}
        for p in self.ids:
            mode = self.behaviors.get(p, "cc=0.80")
            oracle[p] = ("failed", None, "x") if mode == "fail" else ("ok", float(mode.split("=")[1]), "")
        _, sel, todo = se.walk(self.world.cands, self.world.rep_of, oracle)
        self.assertEqual(todo, [])
        got = tse.chosen(sel)
        return sorted(got)

    # ------------------------------------------------------------------
    def test_whole_run(self):
        text = self.execute()
        for s in ("This is not Colab", "all 7 files written and checked", "installing Phenix", "Phenix installed in", "Phenix version: 2.2.1-6174 (stub)",
                  "=== test: CC_mask for the first two entries", "The test worked", "=== round 1: apply the selection rule ===",
                  "SELECTED ENTRIES", "RESULT: OK. The entry list is final."):
            self.assertIn(s, text)
        self.assertNotIn("ERROR", text)
        self.assertNotIn("STOPPED", text)
        self.assertIn("=== round 2", text)                                    # the batch of 3 forces several rounds
        self.assertNotIn(PASSWORD, text)
        name, z, files = self.zip_files()
        self.assertNotIn("INCOMPLETE", name)
        for f in ("entries_2026.csv", "selection_log.csv", "selection_info.json", "cc_todo.csv", "cc_mask.csv", "run.log", "run_info.json",
                  "novelty/novelty_info.json" if MMSEQS else "run.log"):
            self.assertIn(f, files)
        self.assertTrue(any(n.startswith("cc_logs/9T0") for n in files))
        for n in z.namelist():
            self.assertNotIn(PASSWORD.encode(), z.read(n), n)
        entries = z.read(files["entries_2026.csv"]).decode().splitlines()
        self.assertEqual(sorted(line.split(",")[1] for line in entries[1:]), self.expected_selection())
        info = json.loads(z.read(files["run_info.json"]))
        self.assertTrue(info["final"] and info["rounds"] >= 2)
        self.assertEqual(info["phenix_installer"], INSTALLER_NAME)
        self.assertEqual(len(info["phenix_installer_sha256"]), 64)
        self.assertEqual(len(info["input_sha256"]), 7)
        self.assertTrue(10 <= info["cc_mask_rows"] <= 13 and info["cc_mask_ok"] < info["cc_mask_rows"])
        if MMSEQS:
            self.assertEqual(info["novelty"], "done")
            nov = z.read(files["novelty/novelty.csv"]).decode()
            self.assertEqual(len(nov.splitlines()) - 1, len(entries) - 1)
            self.assertIn("max_id_to_pre2026_pdb", entries[0])
        else:
            self.assertIn("NOVELTY CHECK DID NOT FINISH", text)
        # the log never shows the Phenix password, and no file with it was needed
        self.assertNotIn(PASSWORD, z.read(files["run.log"]).decode())

    def test_phenix_without_a_working_version_command_is_accepted_if_the_tool_is_there(self):
        with mock.patch.dict(os.environ, {"STUB_NOVERSION": "1"}):
            text = self.execute()
        self.assertIn("phenix.map_model_cc found at", text)
        self.assertIn("RESULT: OK", text)

    def test_a_second_run_in_the_same_session_skips_the_install(self):
        self.execute()
        text = self.execute()
        self.assertIn("Phenix is already installed", text)
        self.assertNotIn("installing Phenix", text)
        self.assertIn("RESULT: OK", text)
        self.assertIn("round 1", text)

    def test_wrong_password_stops_politely_and_still_packs_a_zip(self):
        text = self.execute(password="wrong-password")
        self.assertIn("did not accept the user name and password (HTTP 401)", text)
        self.assertIn("RESULT: STOPPED", text)
        self.assertEqual(text.count("skipped: an earlier step stopped"), 2)
        name, z, files = self.zip_files()
        self.assertIn("_INCOMPLETE", name)
        self.assertIn("run.log", files)
        self.assertNotIn("wrong-password", z.read(files["run.log"]).decode())

    def test_a_web_page_instead_of_the_installer_is_recognised(self):
        Phenix.html = True
        text = self.execute()
        self.assertIn("do not look like the installer", text)
        self.assertIn("RESULT: STOPPED", text)
        self.assertFalse(os.path.exists(os.path.join(self.work, "work", INSTALLER_NAME + ".part")))

    def test_no_cc_mask_from_the_test_entries_stops_before_the_long_run(self):
        self.set_behaviors(dict.fromkeys(self.ids, "fail"))
        text = self.execute()
        self.assertIn("Phenix gave no CC_mask for either of the two test entries", text)
        self.assertNotIn("=== round 1", text)
        name, z, files = self.zip_files()
        self.assertIn("_INCOMPLETE", name)
        self.assertTrue(any(n.startswith("cc_logs/") for n in files))
        self.assertIn("Sorry: stub failure", z.read(next(v for k, v in files.items() if k.startswith("cc_logs/"))).decode())
        self.assertIn("diagnostics.txt", files)
        self.assertIn("usage: stub help text of phenix.map_correlations", z.read(files["diagnostics.txt"]).decode())
        self.assertIn("cc_mask_FAILED_TEST.csv", files)
        self.assertNotIn("cc_mask.csv", files)              # failures of Phenix itself are not kept as results of the entries

    def test_phenix_with_only_the_replacement_program(self):
        self.set_behaviors(dict(self.behaviors, _policy=["old_gone"]))
        text = self.execute()
        self.assertIn("RESULT: OK", text)
        name, z, files = self.zip_files()
        variants = {line.split(",")[6] for line in z.read(files["cc_mask.csv"]).decode().splitlines()[1:]}
        self.assertEqual(variants, {"map_correlations"})
        self.assertIn("The test worked", text)
        self.assertIn("[map_correlations]", text)

    def test_the_rows_of_a_failed_test_do_not_become_results_in_a_second_run(self):
        self.set_behaviors(dict.fromkeys(self.ids, "fail"))
        self.execute()
        for f in os.listdir(os.path.join(self.work, "work")):
            if f.endswith(".zip"):
                os.remove(os.path.join(self.work, "work", f))
        self.set_behaviors(dict(type(self).behaviors))        # Phenix works now; the installed fake reads its behavior from a file
        found = glob.glob(os.path.join(self.work, "**", "behaviors.json"), recursive=True)
        self.assertEqual(len(found), 1)
        with open(found[0], "w") as f:
            json.dump(self.behaviors, f)
        text = self.execute()
        self.assertIn("RESULT: OK", text)
        name, z, files = self.zip_files()
        rows = {r["pdb_id"]: r for r in csv.DictReader(io.StringIO(z.read(files["cc_mask.csv"]).decode()))}
        self.assertEqual((rows["9T06"]["status"], rows["9T06"]["cc_mask"]), ("ok", "0.8000"))     # the two test entries of the failed run
        self.assertEqual((rows["9T02"]["status"], rows["9T02"]["cc_mask"]), ("ok", "0.5500"))
        self.assertEqual((rows["9T03"]["status"], rows["9T03"]["variant"]), ("failed", "map_model_cc --force"))   # a real failure of an entry stays
        entries = z.read(files["entries_2026.csv"]).decode().splitlines()
        self.assertEqual(sorted(line.split(",")[1] for line in entries[1:]), self.expected_selection())

    def test_a_log_that_contains_the_password_is_left_out_of_the_zip(self):
        self.set_behaviors(dict(self.behaviors, **{"9T01": "leak", "9T04": "leak", "9T06": "leak", "9T07": "leak"}))
        text = self.execute()
        self.assertIn("left out of the zip because they contained the password", text)
        name, z, files = self.zip_files()
        for n in z.namelist():
            self.assertNotIn(PASSWORD.encode(), z.read(n), n)
        self.assertIn("RESULT: OK", text)

    def test_committed_notebook_is_current(self):
        self.assertTrue(os.path.exists(COMMITTED), "run make_select_notebook.py")
        nb = nbformat.read(COMMITTED, as_version=4)
        nbformat.validate(nb)
        fresh = os.path.join(self.root, "fresh.ipynb")
        mk.build(fresh)
        self.assertEqual([c.source for c in nb.cells], [c.source for c in nbformat.read(fresh, as_version=4).cells])
        src = {c.id: c.source for c in nb.cells}
        for s in mk.SCRIPTS:
            with open(os.path.join(HERE, s), encoding="utf-8") as f:
                self.assertIn("%%writefile " + s + "\n" + f.read().rstrip("\n"), src[[k for k, v in src.items() if v.startswith("%%writefile " + s + "\n")][0]])
        self.assertIn(mk.PHENIX_URL, src["step3"])
        self.assertNotIn("password=", src["step3"].replace("password=password", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
