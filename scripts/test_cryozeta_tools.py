#!/usr/bin/env python3
"""Tests of scripts/cryozeta_tools.py (no GPU, no network: a local server plays the ColabFold server and the EMDB, small programs play the three programs of CryoZeta).

    python -m unittest test_cryozeta_tools
"""
from __future__ import annotations

import io
import json
import os
import re
import stat
import sys
import tarfile
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryozeta_tools as CT  # noqa: E402

SEQ = "MKTAYIAKQRQISFVKSHFSRQ"


def load(path):
    with open(path) as f:
        return json.load(f)


def text(path):
    with open(path) as f:
        return f.read()


def a3m(query, hits):
    return ">101\n%s\n" % query + "".join(">hit%d\n%s\n" % (i, h) for i, h in enumerate(hits))


def archive(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, text in files.items():
            raw = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            t.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


def msa_archive(seq):
    """The result archive of the stand-in server for one query: the query and two hits in uniref.a3m, the query and one hit in the environmental file."""
    return archive({"uniref.a3m": "\x00" + a3m(seq, [seq[:-1] + "A", seq[:-2] + "AA"]), "bfd.mgnify30.metaeuk30.smag30.a3m": a3m(seq, ["M" + seq[1:-1] + "G"])})


class Server:
    """A local HTTP server that plays the ColabFold server (a scenario of statuses) and the EMDB (a header). With `result_for` (a function of the query sequence that returns the archive) every
    query gets a ticket of its own (T1, T2, ...); without it the one ticket T1 serves one fixed archive."""

    def __init__(self, statuses=("PENDING", "RUNNING", "COMPLETE"), first="PENDING", result=None, header=None, result_for=None):
        self.statuses, self.first, self.header = list(statuses), first, header
        self.result_for, self.tickets = result_for, {}
        self.result = result if result is not None else archive({"uniref.a3m": "\x00" + a3m(SEQ, [SEQ[:-1] + "A", SEQ[:-2] + "AA"]),
                                                                  "bfd.mgnify30.metaeuk30.smag30.a3m": a3m(SEQ, ["M" + SEQ[1:-1] + "G"])})
        self.posts, self.polls, self.queries = 0, 0, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, body, ctype="application/json"):
                raw = body if isinstance(body, bytes) else body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
                query = parse_qs(body)
                outer.queries.append(query)
                outer.posts += 1
                if self.path == "/ticket/msa":
                    tid = "T1" if outer.result_for is None else "T%d" % outer.posts
                    lines = query.get("q", [""])[0].strip().split("\n")
                    outer.tickets[tid] = lines[1] if len(lines) > 1 else ""
                    self.send(200, json.dumps({"id": tid, "status": outer.first}))
                else:
                    self.send(404, "no")

            def do_GET(self):
                got = re.match(r"^/ticket/(T\d+)$", self.path)
                down = re.match(r"^/result/download/(T\d+)$", self.path)
                if got and (got.group(1) == "T1" or got.group(1) in outer.tickets):
                    outer.polls += 1
                    st = outer.statuses[min(outer.polls - 1, len(outer.statuses) - 1)]
                    self.send(200, json.dumps({"id": got.group(1), "status": st}))
                elif down and (down.group(1) == "T1" or down.group(1) in outer.tickets):
                    self.send(200, outer.result if outer.result_for is None else outer.result_for(outer.tickets[down.group(1)]), "application/gzip")
                elif self.path.startswith("/header/") and outer.header is not None:
                    self.send(200, outer.header, "text/xml")
                else:
                    self.send(404, "no")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


HEADER = """<?xml version="1.0"?><emd><map><contour_list><contour primary="false"><level>0.9</level></contour>
<contour primary="true"><level>0.123</level><source>author</source></contour></contour_list></map></emd>"""


class MsaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.logs = []
        self.sleeps = []

    def ask(self, server, seq=SEQ, **kw):
        return CT.colabfold_msa(seq, os.path.join(self.tmp, "m"), host=server.url, poll_s=0.0, log=self.logs.append, sleep=self.sleeps.append, **kw)

    def test_the_two_files_that_cryozeta_reads(self):
        s = Server()
        try:
            info = self.ask(s)
        finally:
            s.close()
        d = os.path.join(self.tmp, "m")
        other, uni = CT.read_a3m(text(os.path.join(d, "mmseqs_other_hits.a3m"))), CT.read_a3m(text(os.path.join(d, "uniref100_hits.a3m")))
        self.assertEqual(len(uni), 3)                                         # the query and the two UniRef hits
        self.assertEqual(len(other), 4)                                       # plus the environmental hit; the query only once
        self.assertEqual(other[0][1], SEQ)
        self.assertEqual((info["n_uniref_hits"], info["n_env_hits"]), (2, 1))
        self.assertEqual(s.queries[0]["q"], [">101\n%s\n" % SEQ])
        self.assertEqual(s.queries[0]["mode"], ["env"])
        self.assertTrue(os.path.exists(os.path.join(d, "msa_info.json")))

    def test_the_same_sequence_is_not_asked_for_twice(self):
        s = Server()
        try:
            self.ask(s)
            self.ask(s)
        finally:
            s.close()
        self.assertEqual(s.posts, 1)
        self.assertTrue(any("already here" in x for x in self.logs))

    def test_another_sequence_in_the_same_folder_is_asked_for(self):
        s = Server()
        try:
            self.ask(s)
            s.result = archive({"uniref.a3m": a3m("MKK", ["MKK"])})
            self.ask(s, seq="MKK")
        finally:
            s.close()
        self.assertEqual(s.posts, 2)

    def test_a_server_that_asks_to_wait_is_asked_again(self):
        s = Server(first="RATELIMIT")
        try:
            with self.assertRaises(CT.MsaError):                              # a query that is never taken: the deadline ends it
                CT.colabfold_msa(SEQ, os.path.join(self.tmp, "r"), host=s.url, poll_s=0.0, max_wait_s=0.0, tries=1, log=self.logs.append, sleep=self.sleeps.append)
        finally:
            s.close()
        self.assertIn("RATELIMIT", str(self.logs))

    def test_an_error_of_the_server_is_retried_and_then_reported(self):
        s = Server(first="ERROR")
        try:
            with self.assertRaises(CT.MsaError) as cm:
                self.ask(s, tries=2)
        finally:
            s.close()
        self.assertIn("2 attempts", str(cm.exception))
        self.assertEqual(s.posts, 2)
        self.assertEqual(self.sleeps, [30.0])                                 # one wait between the two attempts

    def test_a_result_without_the_query_is_refused(self):
        s = Server(result=archive({"uniref.a3m": a3m("MWWWW", [])}))
        try:
            with self.assertRaises(CT.MsaError) as cm:
                self.ask(s, tries=1)
        finally:
            s.close()
        self.assertIn("not the query", str(cm.exception))

    def test_a_result_without_uniref_is_refused(self):
        s = Server(result=archive({"other.a3m": "x"}))
        try:
            with self.assertRaises(CT.MsaError) as cm:
                self.ask(s, tries=1)
        finally:
            s.close()
        self.assertIn("uniref.a3m", str(cm.exception))

    def test_only_the_twenty_amino_acids_are_sent(self):
        with self.assertRaises(CT.MsaError):
            CT.colabfold_msa("MKX", os.path.join(self.tmp, "x"), host="http://127.0.0.1:1", log=print, sleep=lambda s: None)
        with self.assertRaises(CT.MsaError):
            CT.colabfold_msa("  ", os.path.join(self.tmp, "x"), host="http://127.0.0.1:1", log=print, sleep=lambda s: None)

    def test_an_unreachable_server_is_an_msa_error(self):
        with self.assertRaises(CT.MsaError):
            CT._http("http://127.0.0.1:1/x", tries=2, wait=0.0, sleep=lambda s: None)

    def test_a3m_reading_joins_wrapped_lines_and_drops_the_nul_between_queries(self):
        recs = CT.read_a3m(">a\nAC\nDE\n\x00>b\nFG\n")
        self.assertEqual(recs, [("a", "ACDE"), ("b", "FG")])


class InputTests(unittest.TestCase):
    def test_the_header_address_from_the_map_address(self):
        self.assertEqual(CT.emdb_header_url("https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-55586/map/emd_55586.map.gz"),
                         "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-55586/header/emd-55586.xml")
        with self.assertRaises(ValueError):
            CT.emdb_header_url("https://example.org/map.gz")

    def test_the_primary_contour_level_wins(self):
        self.assertEqual(CT.emdb_contour_level(HEADER), 0.123)
        self.assertEqual(CT.emdb_contour_level("<emd><contour><level>0.5</level></contour></emd>"), 0.5)
        self.assertIsNone(CT.emdb_contour_level("<emd><map/></emd>"))

    def test_the_input_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "in", "x.json")
            entry = CT.write_input_json(path, "9T5K", os.path.join(d, "m.mrc"), 2.6, 0.12, [{"sequence": SEQ, "count": 2, "msa_dir": os.path.join(d, "msa")}])
            got = load(path)
        self.assertEqual(got, [entry])
        c = got[0]["sequences"][0]["proteinChain"]
        self.assertEqual((got[0]["name"], got[0]["resolution"], got[0]["contour_level"], got[0]["modelSeeds"]), ("9T5K", 2.6, 0.12, []))
        self.assertEqual((c["sequence"], c["count"], c["msa"]["pairing_db"]), (SEQ, 2, "uniref100"))
        self.assertTrue(os.path.isabs(got[0]["map_path"]) and os.path.isabs(c["msa"]["precomputed_msa_dir"]))

    def test_prepare_input_gets_the_level_and_the_msas_and_writes_the_file(self):
        s = Server(header=HEADER)
        try:
            with tempfile.TemporaryDirectory() as d:
                cfg = {"name": "E1", "map_path": os.path.join(d, "m.mrc"), "resolution": 3.1, "header_url": s.url + "/header/emd-1.xml", "colabfold_host": s.url,
                       "chains": [{"sequence": SEQ, "count": 3}], "msa_root": os.path.join(d, "msa"), "input_json": os.path.join(d, "in.json"), "msa_poll_s": 0.0}
                lines = []
                info = CT.prepare_input(cfg, lines.append, sleep=lambda x: None)
                got = load(cfg["input_json"])
        finally:
            s.close()
        self.assertEqual(info["contour_level"], 0.123)
        self.assertEqual(info["chains"][0]["n_uniref_hits"], 2)
        self.assertEqual(got[0]["contour_level"], 0.123)
        self.assertEqual(got[0]["sequences"][0]["proteinChain"]["count"], 3)

    def test_a_header_without_a_level_stops_the_input(self):
        s = Server(header="<emd><map/></emd>")
        try:
            with tempfile.TemporaryDirectory() as d:
                cfg = {"name": "E1", "map_path": "m", "resolution": 3.0, "header_url": s.url + "/header/emd-1.xml", "chains": [], "msa_root": d, "input_json": os.path.join(d, "i.json")}
                with self.assertRaises(ValueError):
                    CT.prepare_input(cfg, print, sleep=lambda x: None)
        finally:
            s.close()


class SourceChangeTests(unittest.TestCase):
    def repo(self, d, text):
        path = os.path.join(d, CT.DETECTION_FILE)
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            f.write(text)
        return path

    SOURCE = "a = 1\nself.detection_model.to(self.device).to(torch.bfloat16)\nb = 2\n    amp_dtype = torch.bfloat16\nc = 3\n"

    def test_both_lines_become_float32_and_nothing_else_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.repo(d, self.SOURCE)
            diff = CT.patch_detection_precision(d)
            new_text = text(path)
        self.assertEqual(new_text, self.SOURCE.replace("torch.bfloat16", "torch.float32"))
        self.assertIn("-self.detection_model.to(self.device).to(torch.bfloat16)", diff)
        self.assertIn("+    amp_dtype = torch.float32".strip(), diff.replace("+amp_dtype", "+    amp_dtype"))

    def test_a_second_call_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo(d, self.SOURCE)
            CT.patch_detection_precision(d)
            self.assertEqual(CT.patch_detection_precision(d), "")

    def test_a_source_without_the_line_stops(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo(d, "nothing here\n")
            with self.assertRaises(RuntimeError):
                CT.patch_detection_precision(d)

    def test_a_line_that_occurs_twice_stops(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo(d, self.SOURCE + self.SOURCE)
            with self.assertRaises(RuntimeError):
                CT.patch_detection_precision(d)


# ---------------------------------------------------------------------------------------------------------------------------------- the three programs
STUB = r'''#!/usr/bin/env python3
import glob, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))              # the env dir: bin/<this file>
prog = os.path.basename(sys.argv[0])
args = sys.argv[1:]
scen = json.load(open(os.path.join(HERE, "scenario.json")))
cnt_dir = os.path.join(HERE, "counters")
os.makedirs(cnt_dir, exist_ok=True)


def arg(name, default=None):
    return args[args.index(name) + 1] if name in args else default


def count(label):
    p = os.path.join(cnt_dir, label)
    n = int(open(p).read()) if os.path.exists(p) else 0
    open(p, "w").write(str(n + 1))
    return n


with open(os.path.join(HERE, "calls.jsonl"), "a") as f:
    f.write(json.dumps({"prog": prog, "args": args, "env": {k: os.environ.get(k, "") for k in ("CONDA_PREFIX", "PYTHONPATH", "TEASERPP_DIR", "LAYERNORM_TYPE", "USE_OPM_CHUNKED", "CUDA_VISIBLE_DEVICES", "CRYOZETA_ASSETS_DIR")}}) + "\n")


def write(path, text="x" * 600):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


if prog == "cryozeta-detection":
    inp = json.load(open(args[1]))
    name = inp[0]["name"]
    mode = scen["detection"][min(count("detection"), len(scen["detection"]) - 1)]
    if mode == "killed":                                                        # the system's out-of-memory killer: exit code -9
        os.kill(os.getpid(), 9)
    if mode == "fail":
        print("RuntimeError: detection broke")
        sys.exit(1)
    if mode == "nofile":
        sys.exit(0)
    write(os.path.join(args[2], name, "CryoZeta-Detection", name + ".pt"))
elif prog == "cryozeta-inference":
    interp = arg("--use_interpolation") == "true"
    label = "interpolate" if interp else "cryozeta"
    stage = "CryoZeta-Interpolate" if interp else "CryoZeta"
    inp = json.load(open(arg("--input_json_path")))
    name, dump, seed = inp[0]["name"], arg("--dump_dir"), arg("--seeds")
    k = count(label)
    mode = scen[label][min(k, len(scen[label]) - 1)]
    with open(os.path.join(HERE, "inference_settings.jsonl"), "a") as f:
        f.write(json.dumps({"label": label, "call": k, "mode": mode, "chunk": arg("--infer_setting.chunk_size"), "dtype": arg("--dtype"), "kernels": [arg("--use_deepspeed_evo_attention"),
                            arg("--use_cuequivariance_attention"), arg("--use_cuequivariance_multiplicative_update")], "n_sample": arg("--sample_diffusion.N_sample"),
                            "n_step": arg("--sample_diffusion.N_step"), "n_cycle": arg("--model.N_cycle"), "ckpt": os.path.basename(arg("--load_checkpoint_path"))}) + "\n")
    print("starting " + label)
    assets = os.environ.get("CRYOZETA_ASSETS_DIR", "")
    missing = [n for n in ("components.v20240608.cif", "components.v20240608.cif.rdkit_mol.pkl") if not os.path.isfile(os.path.join(assets, n))]
    if missing:                                                                # what the real program did on 2 November 2026: it logs the error of the featurisation and ends with exit code 0
        print("Error [Errno 2] No such file or directory: '%s/%s':" % (assets or "<site-packages>/assets", missing[0]))
        print("FileNotFoundError: [Errno 2] No such file or directory")
        sys.exit(0)
    if mode == "ok":
        for i in range(int(arg("--sample_diffusion.N_sample"))):
            write(os.path.join(dump, name, stage, "seed_" + seed, "predictions", "%s_sample_%d.cif" % (name, i)))
            write(os.path.join(dump, name, stage, "seed_" + seed, "predictions", "%s_summary_confidence_sample_%d.json" % (name, i)), "{}")
        write(os.path.join(dump, name, stage, "saved_data", "scores.csv"), "a,b\n1,2\n")
    elif mode == "killed":
        os.kill(os.getpid(), 9)
    elif mode == "oom_file":                                                   # the program catches the error, writes it to a file and ends with exit code 0
        write(os.path.join(dump, name, stage, "ERR", name + ".txt"), "Traceback ...\ntorch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.1 GiB")
    elif mode == "oom_out":
        print("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB")
        sys.exit(1)
    elif mode == "fail_file":
        write(os.path.join(dump, name, stage, "ERR", name + ".txt"), "RuntimeError: no kernel image is available for execution on the device")
    elif mode == "skip":
        print("no interpolation features were found for this entry")
elif prog == "cryozeta-combine":
    inp = json.load(open(arg("--input-json-path")))
    name, dump = inp[0]["name"], arg("--dump-dir")
    mode = scen["combine"][min(count("combine"), len(scen["combine"]) - 1)]
    if mode == "fail":
        print("combine broke")
        sys.exit(1)
    for i in range(int(arg("--num-select"))):
        write(os.path.join(dump, name, "CryoZeta-Final", "%s_sample_%d.cif" % (name, i)))
'''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = os.path.join(self.tmp, "env")
        os.makedirs(os.path.join(self.env, "bin"))
        for prog in ("cryozeta-detection", "cryozeta-inference", "cryozeta-combine"):
            p = os.path.join(self.env, "bin", prog)
            with open(p, "w") as f:
                f.write(STUB)
            os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
        self.assets = os.path.join(self.tmp, "assets")
        self.put_components()
        self.server = Server(header=HEADER)
        self.addCleanup(self.server.close)
        self.lines = []

    def put_components(self):
        os.makedirs(self.assets, exist_ok=True)
        for n in ("components.v20240608.cif", "components.v20240608.cif.rdkit_mol.pkl"):
            with open(os.path.join(self.assets, n), "w") as f:
                f.write("x")

    def scenario(self, **kw):
        sc = {"detection": ["ok"], "cryozeta": ["ok"], "interpolate": ["ok"], "combine": ["ok"]}
        sc.update(kw)
        with open(os.path.join(self.env, "scenario.json"), "w") as f:
            json.dump(sc, f)

    def cfg(self, **kw):
        d = self.tmp
        c = {"name": "E1", "repo": d, "env_dir": self.env, "teaser_dir": os.path.join(d, "teaser"), "assets": os.path.join(d, "assets"),
             "weights": {"detection": "det.safetensors", "cryozeta": "cz.safetensors", "interpolate": "ip.safetensors"},
             "input_json": os.path.join(d, "in", "E1.json"), "dump_dir": os.path.join(d, "dump"), "seed": 101, "n_sample": 5, "n_step": 20, "n_cycle": 10, "num_select": 5,
             "dtype": "bf16", "kernels": "off", "summary_path": os.path.join(d, "dump", "driver_summary.json"), "progress_path": os.path.join(d, "progress.txt"),
             "map_path": os.path.join(d, "m.mrc"), "resolution": 2.5, "header_url": self.server.url + "/header/emd-1.xml", "colabfold_host": self.server.url,
             "chains": [{"sequence": SEQ, "count": 2}], "msa_root": os.path.join(d, "msa"), "msa_poll_s": 0.0}
        c.update(kw)
        return c

    def run_it(self, cfg=None, **kw):
        cfg = cfg or self.cfg(**kw)
        os.makedirs(cfg["dump_dir"], exist_ok=True)
        orig = CT._http
        CT._http = lambda *a, **k: orig(*a, **dict(k, wait=0.0, sleep=lambda s: None))         # no waiting between the attempts of the tests
        orig_sleep = CT.time.sleep
        CT.time.sleep = lambda s: None
        try:
            return CT.run_entry(cfg, say=self.lines.append)
        finally:
            CT._http = orig
            CT.time.sleep = orig_sleep

    def lines_of(self, name):
        p = os.path.join(self.env, name)
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(x) for x in f]

    def calls(self):
        return self.lines_of("calls.jsonl")

    def settings(self):
        return self.lines_of("inference_settings.jsonl")

    def test_a_complete_run(self):
        self.scenario()
        code, summary = self.run_it()
        self.assertEqual((code, summary["status"]), (0, "ok"))
        final = os.path.join(self.tmp, "dump", "E1", "CryoZeta-Final")
        self.assertEqual(sorted(os.listdir(final)), ["E1_sample_%d.cif" % i for i in range(5)])
        self.assertEqual([c["prog"] for c in self.calls()], ["cryozeta-detection", "cryozeta-inference", "cryozeta-inference", "cryozeta-combine"])
        self.assertEqual(summary["input"]["contour_level"], 0.123)
        self.assertEqual(load(os.path.join(self.tmp, "in", "E1.json"))[0]["resolution"], 2.5)
        self.assertEqual(load(self.cfg()["summary_path"])["status"], "ok")
        self.assertTrue(any("DRIVER RESULT ok" in x for x in self.lines))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "dump", "logs", "detection.log")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "dump", "summary_files", "cryozeta_scores.csv")))

    def test_every_program_is_told_where_the_assets_are(self):
        self.scenario()
        self.run_it()
        calls = self.calls()
        self.assertEqual({c["env"]["CRYOZETA_ASSETS_DIR"] for c in calls}, {self.assets})
        self.assertEqual(len(calls), 4)

    def test_missing_components_files_are_a_failure_with_a_reason_that_names_the_file(self):
        """2 November 2026: the first CryoZeta session did not download the two chemical-components files; the program logged a FileNotFoundError and ended with exit code 0 and no model."""
        self.scenario()
        os.remove(os.path.join(self.assets, "components.v20240608.cif"))
        code, summary = self.run_it()
        self.assertEqual((code, summary["status"]), (3, "failed"))
        self.assertIn("no predictions", summary["note"])
        tails = " ".join(" ".join(a["tail"]) for a in summary["attempts"])
        self.assertIn("components.v20240608.cif", tails)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "dump", "E1", "CryoZeta-Final")))

    def test_the_settings_are_the_ones_of_the_authors_demo_with_the_kernels_off(self):
        self.scenario()
        self.run_it()
        first, second = self.settings()
        for s in (first, second):
            self.assertEqual((s["n_sample"], s["n_step"], s["n_cycle"], s["dtype"], s["chunk"], s["kernels"]), ("5", "20", "10", "bf16", "256", ["false", "false", "false"]))
        self.assertEqual((first["ckpt"], second["ckpt"]), ("cz.safetensors", "ip.safetensors"))
        det, inf = self.calls()[0], self.calls()[1]
        self.assertEqual(det["args"][0], "json-run")
        self.assertIn("--seeds", inf["args"])
        self.assertEqual(inf["args"][inf["args"].index("--seeds") + 1], "101")
        self.assertEqual(inf["env"]["USE_OPM_CHUNKED"], "1")
        self.assertEqual(inf["env"]["LAYERNORM_TYPE"], "")                         # the compiled layer norm needs a CUDA compiler: off
        self.assertEqual(inf["env"]["CONDA_PREFIX"], self.env)
        self.assertEqual(inf["env"]["PYTHONPATH"], os.path.join(self.tmp, "teaser", "build", "python"))

    def test_the_demo_kernels_are_only_switched_on_when_asked(self):
        self.scenario()
        self.run_it(kernels="demo")
        self.assertEqual(self.settings()[0]["kernels"], ["true", "true", "true"])
        self.assertEqual(self.calls()[1]["env"]["LAYERNORM_TYPE"], "fast_layernorm")

    def test_an_error_that_the_program_hides_behind_exit_code_zero_is_found_and_the_memory_ladder_goes_down(self):
        self.scenario(cryozeta=["oom_file", "oom_out", "ok"], interpolate=["ok"])
        code, summary = self.run_it()
        self.assertEqual((code, summary["status"]), (0, "ok"))
        got = [(s["label"], s["chunk"]) for s in self.settings()]
        self.assertEqual(got, [("cryozeta", "256"), ("cryozeta", "64"), ("cryozeta", "16"), ("interpolate", "16")])      # the next stage keeps what worked
        self.assertEqual(summary["final_settings"]["chunk_size"], 16)
        self.assertEqual([a["oom"] for a in summary["attempts"]], [True, True, False, False])

    def test_a_ladder_that_is_used_up_is_a_failure_for_lack_of_memory(self):
        self.scenario(cryozeta=["oom_file"])
        code, summary = self.run_it()
        self.assertEqual((code, summary["status"], summary["failed_stage"]), (2, "failed", "cryozeta"))
        self.assertIn("CUDA out of memory", summary["note"])
        self.assertEqual([s["chunk"] for s in self.settings()], ["256", "64", "16", "4"])
        self.assertTrue(any("out of memory" in x.lower() for x in self.lines[-6:]))               # the notebook decides from the last lines of the program
        self.assertNotIn("cryozeta-combine", [c["prog"] for c in self.calls()])

    def test_another_failure_is_tried_once_in_float32_and_then_reported(self):
        self.scenario(cryozeta=["fail_file", "fail_file"])
        code, summary = self.run_it()
        self.assertEqual((code, summary["failed_stage"]), (3, "cryozeta"))
        self.assertEqual([(s["dtype"], s["chunk"]) for s in self.settings()], [("bf16", "256"), ("fp32", "256")])

    def test_a_failure_that_float32_cures_is_a_result(self):
        self.scenario(cryozeta=["fail_file", "ok"])
        code, summary = self.run_it()
        self.assertEqual(code, 0)
        self.assertEqual(summary["final_settings"]["dtype"], "fp32")
        self.assertEqual([s["dtype"] for s in self.settings()], ["bf16", "fp32", "fp32"])        # the interpolation stage keeps float32

    def test_an_interpolation_without_features_is_not_a_failure(self):
        self.scenario(interpolate=["skip"])
        code, summary = self.run_it()
        self.assertEqual((code, summary["status"]), (0, "ok"))
        self.assertTrue(summary["attempts"][-1]["skipped"])

    def test_a_broken_detection_ends_the_entry_at_once(self):
        self.scenario(detection=["fail"])
        code, summary = self.run_it()
        self.assertEqual((code, summary["failed_stage"]), (3, "detection"))
        self.assertEqual([c["prog"] for c in self.calls()], ["cryozeta-detection"])

    def test_a_detection_killed_by_the_system_is_exit_code_4_and_says_so(self):
        self.scenario(detection=["killed"])
        code, summary = self.run_it()
        self.assertEqual((code, summary["failed_stage"]), (4, "detection"))
        self.assertIn("exit code -9", summary["note"])
        self.assertEqual([c["prog"] for c in self.calls()], ["cryozeta-detection"])

    def test_an_inference_killed_by_the_system_is_exit_code_4_and_is_not_tried_again(self):
        self.scenario(cryozeta=["killed"])
        code, summary = self.run_it()
        self.assertEqual((code, summary["failed_stage"]), (4, "cryozeta"))
        self.assertEqual([(s["dtype"], s["chunk"]) for s in self.settings()], [("bf16", "256")])        # no float32 attempt, no smaller chunk

    def test_a_detection_that_writes_nothing_is_a_failure_although_the_exit_code_is_zero(self):
        self.scenario(detection=["nofile"])
        code, summary = self.run_it()
        self.assertEqual((code, summary["failed_stage"]), (3, "detection"))
        self.assertIn("no detection file", summary["note"])

    def test_a_broken_combine_is_a_failure(self):
        self.scenario(combine=["fail"])
        code, summary = self.run_it()
        self.assertEqual((code, summary["failed_stage"]), (3, "combine"))

    def test_a_server_that_gives_no_alignment_stops_before_any_program_runs(self):
        self.scenario()
        bad = Server(first="ERROR", header=HEADER)
        self.addCleanup(bad.close)
        cfg = self.cfg(colabfold_host=bad.url, header_url=bad.url + "/header/emd-1.xml", msa_tries=1)
        code, summary = self.run_it(cfg)
        self.assertEqual((code, summary["failed_stage"]), (3, "input"))
        self.assertIn("MsaError", summary["note"])
        self.assertEqual(self.calls(), [])

    def test_a_progress_file_follows_the_stages(self):
        self.scenario()
        self.run_it()
        self.assertEqual(text(self.cfg()["progress_path"]).strip(), "done: ok")

    def test_the_program_is_started_from_the_runner_file(self):
        self.scenario()
        cfg = self.cfg()
        os.makedirs(cfg["dump_dir"], exist_ok=True)
        path = os.path.join(self.tmp, "cfg.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        # the real entry point, with the standard library only
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cryozeta_tools.py"), "run", path], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
        self.assertIn("DRIVER RESULT ok", r.stdout)


class CommandTests(unittest.TestCase):
    CFG = {"env_dir": "/e", "assets": "/a", "weights": {"detection": "d.s", "cryozeta": "c.s", "interpolate": "i.s"}, "dump_dir": "/d", "input_json": "/i.json", "seed": 101,
           "n_sample": 5, "n_step": 20, "n_cycle": 10, "num_select": 5, "kernels": "off", "name": "X"}

    def test_the_detection_command(self):
        self.assertEqual(CT.detection_cmd(self.CFG), ["/e/bin/cryozeta-detection", "json-run", "/i.json", "/d", "--device", "cuda", "--checkpoint-path", "/a/d.s"])

    def test_the_inference_command_names_the_chunk_size_and_the_number_format(self):
        cmd = CT.inference_cmd(self.CFG, False, "bf16", 64)
        self.assertEqual(cmd[-4:], ["--dtype", "bf16", "--infer_setting.chunk_size", "64"])
        self.assertEqual(cmd[cmd.index("--use_interpolation") + 1], "false")
        self.assertEqual(cmd[cmd.index("--load_checkpoint_path") + 1], "/a/c.s")
        cmd = CT.inference_cmd(self.CFG, True, "fp32", 16)
        self.assertEqual((cmd[cmd.index("--use_interpolation") + 1], cmd[cmd.index("--load_checkpoint_path") + 1]), ("true", "/a/i.s"))

    def test_the_combine_command_selects_all_samples(self):
        self.assertEqual(CT.combine_cmd(self.CFG)[-2:], ["--num-select", "5"])

    def test_the_ladder_is_the_one_of_the_description(self):
        self.assertEqual(CT.CHUNK_LADDER, (256, 64, 16, 4))


if __name__ == "__main__":
    unittest.main()
