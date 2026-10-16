#!/usr/bin/env python3
"""Offline tests for pull_pool.py.

They use synthetic mmCIF files and a local mock of the three web services the script calls
(RCSB search, mmCIF download, EMDB API). Nothing here proves that the real services behave
the same way; the canary entries in the real run cover that. Run with:

    python scripts/test_pull_pool.py
"""
from __future__ import annotations

import csv
import gzip
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from collections import Counter

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pull_pool as pp  # noqa: E402

AA = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
      "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
      "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def protein(eid, length, copies=1, modeled=None, unk=0):
    seq = ("".join(list(AA)[(i * 7 + int(eid)) % 20] for i in range(length)))
    return {"id": str(eid), "type": "polypeptide(L)", "seq": seq, "copies": copies,
            "modeled": length if modeled is None else modeled, "unk": unk}


def rna(eid, length=20):
    return {"id": str(eid), "type": "polyribonucleotide", "seq": "ACGU" * (length // 4), "copies": 1,
            "modeled": length, "unk": 0}


def make_cif(pdb_id, release="2026-03-04", method="ELECTRON MICROSCOPY", recon="SINGLE PARTICLE",
             resolution="3.1", emdb=("EMD-90001",), entities=None, with_history=True):
    entities = entities or [protein(1, 60, copies=2, modeled=50)]
    out = ["data_%s" % pdb_id, "#", "_entry.id %s" % pdb_id, "#", "loop_", "_database_2.database_id",
           "_database_2.database_code", "PDB %s" % pdb_id, "WWPDB D_1000000001"]
    out += ["EMDB %s" % e for e in emdb]
    out += ["#", "_exptl.entry_id %s" % pdb_id, "_exptl.method '%s'" % method, "#"]
    if with_history:
        out += ["loop_", "_pdbx_audit_revision_history.ordinal", "_pdbx_audit_revision_history.data_content_type",
                "_pdbx_audit_revision_history.major_revision", "_pdbx_audit_revision_history.minor_revision",
                "_pdbx_audit_revision_history.revision_date",
                "1 'Structure model' 1 0 %s" % release, "2 'Structure model' 1 1 2027-01-01", "#"]
    if recon:
        out += ["_em_experiment.id 1", "_em_experiment.aggregation_state PARTICLE",
                "_em_experiment.reconstruction_method '%s'" % recon, "_em_experiment.entity_assembly_id 1", "#"]
    if resolution:
        out += ["_em_3d_reconstruction.id 1", "_em_3d_reconstruction.resolution %s" % resolution,
                "_em_3d_reconstruction.resolution_method 'FSC 0.143 CUT-OFF'", "#"]
    if emdb:
        out += ["loop_", "_pdbx_database_related.db_name", "_pdbx_database_related.details",
                "_pdbx_database_related.db_id", "_pdbx_database_related.content_type"]
        for i, e in enumerate(emdb):
            out.append("EMDB 'Cryo-EM map' %s '%s'" % (e, "associated EM volume" if i == 0 else "other EM volume"))
        out.append("#")
    out += ["loop_", "_entity.id", "_entity.type", "_entity.pdbx_number_of_molecules"]
    for e in entities:
        out.append("%s polymer %d" % (e["id"], e["copies"]))
    n_ent = len(entities)
    out += ["%d non-polymer 1" % (n_ent + 1), "%d water 5" % (n_ent + 2), "#"]
    out += ["loop_", "_entity_poly.entity_id", "_entity_poly.type", "_entity_poly.pdbx_seq_one_letter_code_can",
            "_entity_poly.pdbx_strand_id"]
    for e in entities:
        out.append("%s '%s'" % (e["id"], e["type"]))
        out.append(";" + "\n".join(e["seq"][i:i + 70] for i in range(0, len(e["seq"]), 70)))
        out.append(";")
        out.append("A,B")
    out.append("#")
    out += ["loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol", "_atom_site.label_atom_id",
            "_atom_site.label_alt_id", "_atom_site.label_comp_id", "_atom_site.label_asym_id",
            "_atom_site.label_entity_id", "_atom_site.label_seq_id", "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv", "_atom_site.auth_seq_id", "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id", "_atom_site.auth_atom_id", "_atom_site.pdbx_PDB_model_num"]
    atom_id, chain_i = 1, 0
    for e in entities:
        for _ in range(e["copies"]):
            asym = LETTERS[chain_i % 26] + ("" if chain_i < 26 else str(chain_i // 26))
            chain_i += 1
            for i in range(1, e["modeled"] + 1):
                if e["type"] == "polypeptide(L)":
                    comp = "UNK" if i <= e["unk"] else AA.get(e["seq"][i - 1], "ALA")
                    names = [("N", "N"), ("CA", "C")] if i == 1 else [("CA", "C")]
                else:
                    comp, names = "A", [("P", "P")]
                for aname, el in names:
                    out.append("ATOM %d %s %s . %s %s %s %d ? 1.0 1.0 %d.0 1.00 10.0 %d %s %s %s 1" % (
                        atom_id, el, aname, comp, asym, e["id"], i, atom_id % 90, i, comp, asym, aname))
                    atom_id += 1
    out.append("HETATM %d MG MG . MG %s %d . ? 8.0 8.0 8.0 1.00 10.0 301 MG A MG 1" % (atom_id, LETTERS[chain_i % 26], n_ent + 1))
    out.append("#")
    return "\n".join(out) + "\n"


class Mock:
    """In-memory stand-in for the RCSB search API, the file servers and the EMDB API."""

    def __init__(self):
        self.entries = {}
        self.reject_attrs = set()
        self.zero_attrs = set()
        self.cif_404 = set()
        self.primary_404 = set()
        self.emdb = {}
        self.cif_calls = Counter()
        self.emdb_calls = Counter()
        self.search_bodies = []
        self.reject_plain_date = False
        self.reject_paginate = False
        self.max_rows = None
        self.huge = set()

    def add(self, pdb_id, release="2026-03-04", sp=True, res_combined=3.1, types="Protein (only)", modeled=100,
            emdb_release="same", emdb_http=200, **cif_kw):
        cif_kw.setdefault("release", release)
        text = make_cif(pdb_id, **cif_kw)
        self.entries[pdb_id] = {"release": release, "text": text, "sp": sp, "res": res_combined,
                                "types": types, "modeled": modeled}
        for e in cif_kw.get("emdb", ("EMD-90001",)):
            rel = release if emdb_release == "same" else emdb_release
            self.emdb[e] = (emdb_http, {
                "emdb_id": e,
                "admin": {"current_status": {"code": {"valueOf_": "REL"}},
                          "key_dates": {"deposition": "2025-01-01", "header_release": rel, "map_release": rel,
                                        "update": "2026-09-30"}},
                "map": {"file": "x.map.gz", "format": "CCP4", "size_kbytes": 54321}})

    def search(self, body):
        self.search_bodies.append(body)
        options = body.get("request_options", {})
        if self.reject_paginate and "paginate" in options:
            return 400, {"message": "request_options.paginate is not accepted"}
        lo = hi = None
        preds = []
        for n in body["query"]["nodes"]:
            p = n["parameters"]
            attr, op, val = p["attribute"], p["operator"], p["value"]
            if attr in self.reject_attrs:
                return 400, {"message": "attribute %s is not searchable" % attr}
            if attr in self.zero_attrs:
                return 204, None
            if attr == "rcsb_accession_info.initial_release_date":
                if self.reject_plain_date and "T" not in val:
                    return 400, {"message": "date must be a full timestamp"}
                if op == "greater_or_equal":
                    lo = val[:10]
                elif op == "less":
                    hi = val[:10]
            elif attr == "exptl.method":
                if val != "ELECTRON MICROSCOPY":
                    return 204, None
            elif attr == "rcsb_entry_info.experimental_method":
                if val != "EM":
                    return 204, None
            elif attr == "em_experiment.reconstruction_method":
                preds.append(lambda e, v=val: e["sp"] and v == "SINGLE PARTICLE")
            elif attr == "rcsb_entry_info.resolution_combined":
                preds.append(lambda e, v=val: e["res"] <= v)
            elif attr == "rcsb_entry_info.selected_polymer_entity_types":
                preds.append(lambda e, v=val: e["types"] == v)
            elif attr == "rcsb_entry_info.deposited_modeled_polymer_monomer_count":
                preds.append(lambda e, v=val: e["modeled"] <= v)
            else:
                return 400, {"message": "unknown attribute %s" % attr}
        ids = sorted(i for i, e in self.entries.items()
                     if (lo is None or e["release"] >= lo) and (hi is None or e["release"] < hi)
                     and all(p(e) for p in preds))
        if not ids:
            return 204, None
        total = len(ids)
        if "paginate" in options:
            cap = options["paginate"].get("rows", 10)
            if self.max_rows:
                cap = min(cap, self.max_rows)
            ids = ids[:cap]
        return 200, {"query_id": "mock", "result_type": "entry", "total_count": total,
                     "result_set": [{"identifier": i, "score": 1.0} for i in ids]}


def serve(mock):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _send(self, code, body=b"", ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            code, js = mock.search(body)
            self._send(code, b"" if js is None else json.dumps(js).encode())

        def do_GET(self):
            path = self.path
            if path.startswith("/cif/") or path.startswith("/mirror/"):
                pid = path.split("/")[-1].split(".")[0].upper()
                mock.cif_calls[pid] += 1
                if pid in mock.cif_404 or (path.startswith("/cif/") and pid in mock.primary_404):
                    return self._send(404, b"not found", "text/plain")
                if pid not in mock.entries:
                    # canary entries live in the same dictionary under their own IDs
                    return self._send(404, b"not found", "text/plain")
                if pid in mock.huge:  # announce a 99 MB file and send nothing; the client must hang up
                    self.send_response(200)
                    self.send_header("Content-Type", "application/gzip")
                    self.send_header("Content-Length", "99000000")
                    self.end_headers()
                    return
                return self._send(200, gzip.compress(mock.entries[pid]["text"].encode()), "application/gzip")
            if path.startswith("/emdb/"):
                emd = path.split("/")[-1]
                mock.emdb_calls[emd] += 1
                code, js = mock.emdb.get(emd, (404, None))
                return self._send(code, b"{}" if js is None else json.dumps(js).encode())
            self._send(404, b"no route", "text/plain")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


FILLER_DAYS = ("2026-08-14", "2026-08-17", "2026-08-19")  # 25 ordinary entries in one search week, so the probe has a busy week


def populate(m):
    """Fill the mock. Returns (expected first_fail per non-canary entry, ids the server prefilters remove)."""
    m.add("9P01", release="2026-02-04", res_combined=2.5, resolution="2.5", emdb=("EMD-90001",),
          entities=[protein(1, 60, copies=2, modeled=50)], modeled=100)
    m.add("9P02", release="2026-06-10", res_combined=3.6, resolution="3.6", emdb=("EMD-90002",),
          entities=[protein(1, 80, modeled=70), protein(2, 40, copies=2, modeled=40)], modeled=150)
    m.add("9P03", types="Protein/NA", emdb=("EMD-90003",), entities=[protein(1, 60), rna(2)], modeled=80)
    m.add("9P04", res_combined=4.6, resolution="4.6", emdb=("EMD-90004",), modeled=60)
    m.add("9P05", release="2026-04-01", emdb=("EMD-90005",),
          entities=[protein(i, 30) for i in range(1, 5)], modeled=120)
    m.add("9P06", release="2026-05-06", emdb=("EMD-90006",),
          entities=[protein(1, 50, unk=10)], modeled=50)
    m.add("9P07", release="2026-05-06", emdb=("EMD-90007", "EMD-90107"), entities=[protein(1, 50)], modeled=50)
    m.add("9P08", release="2026-07-01", sp=False, recon="TOMOGRAPHY", emdb=("EMD-90008",), modeled=50)
    m.add("9P09", release="2026-07-01", emdb=("EMD-90009",), entities=[protein(1, 250, copies=4)], modeled=1000)
    m.add("9P10", release="2026-08-05", emdb=("EMD-90010",), emdb_release="2025-12-31", entities=[protein(1, 50)], modeled=50)
    m.add("9P11", release="2026-09-02", emdb=("EMD-90011",), emdb_http=404, entities=[protein(1, 50)], modeled=50)
    # boundaries: 4.0 A, exactly 700 residues, exactly 5% UNK
    m.add("9P12", release="2026-09-09", res_combined=4.0, resolution="4.0", emdb=("EMD-90012",),
          entities=[protein(1, 100, copies=7, unk=5)], modeled=700)
    # exactly 3 unique sequences
    m.add("9P13", release="2026-09-09", emdb=("EMD-90013",), entities=[protein(i, 30) for i in range(1, 4)], modeled=90)
    # the primary file source says 404, the mirror has it
    m.add("9P15", release="2026-09-16", emdb=("EMD-90015",), entities=[protein(1, 50)], modeled=50)
    m.primary_404.add("9P15")
    # no EMDB link at all
    m.add("9P16", release="2026-09-16", emdb=(), entities=[protein(1, 50)], modeled=50)
    expected = {
        "9P01": None, "9P02": None, "9P03": "protein_only", "9P04": "resolution", "9P05": "unique_sequences",
        "9P06": "unk_fraction", "9P07": "one_emdb_map", "9P08": "single_particle", "9P09": "modeled_residues",
        "9P10": "emdb_map_released", "9P11": None, "9P12": None, "9P13": None, "9P15": None, "9P16": "one_emdb_map",
    }
    removed = {"9P03", "9P04", "9P08", "9P09"}
    for k in range(25):
        res = round(2.0 + 0.1 * k, 1)  # 2.0 ... 4.4
        pid = "9F%02d" % k
        m.add(pid, release=FILLER_DAYS[k % 3], res_combined=res, resolution="%.1f" % res, emdb=("EMD-91%03d" % k,),
              entities=[protein(1, 40)], modeled=40)
        expected[pid] = "resolution" if res > 4.0 else None
        if res > 4.1:
            removed.add(pid)
    # canary entries (not in any search result: released before the window)
    for cid, emd in (("5A1A", "EMD-2984"), ("6VXX", "EMD-21452")):
        m.add(cid, release="2015-07-15", res_combined=2.2, resolution="2.2", emdb=(emd,),
              entities=[protein(1, 40, copies=3)], modeled=120)
    return expected, removed


def fresh(m):
    m.__init__()
    return populate(m)


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock = Mock()
        cls.srv = serve(cls.mock)
        port = cls.srv.server_address[1]
        base = "http://127.0.0.1:%d" % port
        cls.args = ["--search-url", base + "/search", "--cif-url", base + "/cif/{ID}.cif.gz",
                    "--cif-url", base + "/mirror/{id}.cif.gz", "--emdb-url", base + "/emdb/{EMD}",
                    "--threads", "4", "--emdb-threads", "2"]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.expected, self.removed = fresh(self.mock)
        self.n_all = len(self.expected)
        self.passing = sorted(k for k, v in self.expected.items() if v is None)
        self.tmp = tempfile.mkdtemp(prefix="pool_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_pool(self, *extra, out=None):
        out = out or os.path.join(self.tmp, "out")
        rc = pp.main(self.args + ["--out", out] + list(extra))
        return rc, out

    @staticmethod
    def read(out, name):
        with open(os.path.join(out, name), newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def text(out, name):
        with open(os.path.join(out, name), encoding="utf-8") as f:
            return f.read()

    @classmethod
    def info(cls, out):
        return json.loads(cls.text(out, "run_info.json"))

    def test_full_run(self):
        rc, out = self.run_pool("--no-canary", "--prefilter")
        self.assertEqual(rc, 0)
        rows = {r["pdb_id"]: r for r in self.read(out, "pool_all_entries.csv")}
        # entries the server-side prefilters remove are not downloaded in the main pass
        self.assertEqual(set(rows), set(self.expected) - self.removed)
        for pid, row in rows.items():
            self.assertEqual(row["first_fail"] or None, self.expected[pid], pid)
        cands = self.read(out, "pool_candidates.csv")
        self.assertEqual(sorted(r["pdb_id"] for r in cands), self.passing)
        by = {r["pdb_id"]: r for r in cands}
        self.assertEqual(by["9P01"]["stratum"], "le3.0")
        self.assertEqual(by["9P02"]["stratum"], "gt3.0_le4.0")
        self.assertEqual(by["9P12"]["stratum"], "gt3.0_le4.0")
        self.assertEqual(by["9P12"]["n_modeled_residues"], "700")
        self.assertEqual(by["9P12"]["n_unk"], "35")
        self.assertEqual(by["9P01"]["n_modeled_residues"], "100")
        self.assertEqual(by["9P01"]["n_chains"], "2")
        self.assertEqual(by["9P02"]["n_unique_seq"], "2")
        self.assertEqual(by["9P02"]["longest_entity"], "1")
        self.assertEqual(by["9P02"]["longest_seq_len"], "80")
        self.assertEqual(by["9P13"]["n_unique_seq"], "3")
        self.assertEqual(by["9P01"]["emdb_id"], "EMD-90001")
        self.assertTrue(by["9P01"]["map_url"].endswith("/EMD-90001/map/emd_90001.map.gz"))
        self.assertEqual(by["9P01"]["emdb_release_verified"], "yes")
        self.assertEqual(by["9P01"]["map_size_kb"], "54321")
        self.assertTrue(by["9P11"]["emdb_release_verified"].startswith("no:"))
        # flow table adds up
        flow = self.read(out, "pool_flow.csv")
        self.assertEqual(int(flow[0]["n_remaining"]), self.n_all)
        self.assertEqual(int(flow[-1]["n_remaining"]), len(self.passing))
        steps = {r["step"]: r for r in flow}
        self.assertEqual(int(steps["server_prefilter"]["n_removed_at_this_step"]), len(self.removed))
        self.assertEqual(int(steps["emdb_map_released"]["n_removed_at_this_step"]), 1)
        prev = int(flow[1]["n_remaining"])
        for r in flow[2:]:
            self.assertEqual(int(r["n_remaining"]), prev - int(r["n_removed_at_this_step"]))
            prev = int(r["n_remaining"])
        # FASTA has the unique sequences of the candidates only
        fasta = self.text(out, "pool_candidate_sequences.fasta")
        self.assertIn(">9P02_E1 len=80 copies=1 longest=1", fasta)
        self.assertIn(">9P02_E2 len=40 copies=2 longest=0", fasta)
        self.assertNotIn("9P05", fasta)
        info = self.info(out)
        self.assertEqual(info["n_candidates"], len(self.passing))
        self.assertEqual(info["n_cryoem_in_window"], self.n_all)
        self.assertEqual(sorted(info["search_prefilters_kept"]), ["protein_only", "residues", "resolution", "single_particle"])
        self.assertEqual(info["prefilter_audit"]["n_that_pass_the_file_filters"], 0)
        self.assertEqual(info["prefilter_audit"]["n_sampled"], len(self.removed))
        self.assertTrue(info["spec_values_locked"])

    def test_ids_files_and_weekly_counts(self):
        rc, out = self.run_pool("--no-canary", "--prefilter")
        ids_all = self.text(out, "pool_ids_all_cryoem.txt").split()
        ids_sel = self.text(out, "pool_ids.txt").split()
        self.assertEqual(len(ids_all), self.n_all)
        self.assertEqual(len(ids_sel), self.n_all - len(self.removed))
        self.assertTrue(set(ids_sel) < set(ids_all))
        weeks = self.read(out, "search_weeks.csv")
        self.assertEqual(weeks[0]["window_start"], "2026-01-01")
        self.assertEqual(weeks[-1]["window_end_inclusive"], "2026-10-12")
        self.assertEqual(sum(int(w["n_cryoem"]) for w in weeks), self.n_all)
        self.assertEqual(max(int(w["n_cryoem"]) for w in weeks), 25)

    def test_prefilter_fallback(self):
        self.mock.reject_attrs.add("em_experiment.reconstruction_method")
        self.mock.zero_attrs.add("rcsb_entry_info.selected_polymer_entity_types")
        rc, out = self.run_pool("--no-canary", "--prefilter")
        self.assertEqual(rc, 0)
        info = self.info(out)
        self.assertEqual(sorted(info["search_prefilters_kept"]), ["residues", "resolution"])
        dropped = {d["filter"] for d in info["search_prefilters_dropped"]}
        self.assertEqual(dropped, {"single_particle", "protein_only"})
        cands = self.read(out, "pool_candidates.csv")
        self.assertEqual(sorted(r["pdb_id"] for r in cands), self.passing)

    def test_default_downloads_every_entry(self):
        rc, out = self.run_pool("--no-canary")
        self.assertEqual(rc, 0)
        info = self.info(out)
        self.assertEqual(info["search_prefilters_kept"], [])
        self.assertNotIn("prefilter_audit", info)
        cands = self.read(out, "pool_candidates.csv")
        self.assertEqual(sorted(r["pdb_id"] for r in cands), self.passing)
        rows = {r["pdb_id"]: r for r in self.read(out, "pool_all_entries.csv")}
        self.assertEqual(set(rows), set(self.expected))
        for pid, row in rows.items():
            self.assertEqual(row["first_fail"] or None, self.expected[pid], pid)
        self.assertEqual(rows["9P03"]["first_fail"], "protein_only")
        self.assertEqual(rows["9P08"]["first_fail"], "single_particle")
        self.assertEqual(rows["9P09"]["first_fail"], "modeled_residues")
        flow = self.read(out, "pool_flow.csv")
        self.assertEqual([r["step"] for r in flow][:2], ["all_cryoem", "file_size"])
        self.assertNotIn("server_prefilter", [r["step"] for r in flow])
        self.assertEqual(int(flow[1]["n_remaining"]), self.n_all)
        self.assertEqual(int(flow[-1]["n_remaining"]), len(self.passing))
        self.assertGreater(self.info(out)["peak_memory_mb"], 0)

    def test_oversized_file_is_skipped_not_downloaded(self):
        self.mock.huge.add("9P01")  # would pass every filter
        rc, out = self.run_pool("--no-canary")
        self.assertEqual(rc, 0)
        rows = {r["pdb_id"]: r for r in self.read(out, "pool_all_entries.csv")}
        self.assertEqual(rows["9P01"]["first_fail"], "file_size")
        self.assertIn("99 MB", rows["9P01"]["note"])
        self.assertEqual(rows["9P01"]["error"], "")
        self.assertNotIn("9P01", {r["pdb_id"] for r in self.read(out, "pool_candidates.csv")})
        flow = {r["step"]: r for r in self.read(out, "pool_flow.csv")}
        self.assertEqual(int(flow["file_size"]["n_removed_at_this_step"]), 1)
        self.assertEqual(self.mock.cif_calls["9P01"], 1)  # no fall-back to the mirror
        self.assertEqual(self.info(out)["n_file_size_skipped"], 1)
        # the decision is cached, so a second run does not ask again
        self.mock.cif_calls.clear()
        rc, out = self.run_pool("--no-canary", out=out)
        self.assertEqual(rc, 0)
        self.assertEqual(self.mock.cif_calls["9P01"], 0)

    def test_resume_uses_cache(self):
        rc, out = self.run_pool("--no-canary")
        self.assertEqual(rc, 0)
        first_calls = sum(self.mock.cif_calls.values())
        self.mock.cif_calls.clear()
        rc, out = self.run_pool("--no-canary", out=out)
        self.assertEqual(rc, 0)
        info = self.info(out)
        self.assertEqual(info["n_cached"], self.n_all)
        self.assertEqual(sum(self.mock.cif_calls.values()), 0)
        self.assertGreater(first_calls, 20)

    def test_download_error_then_recovery(self):
        self.mock.cif_404.add("9P02")
        rc, out = self.run_pool("--no-canary")
        self.assertEqual(rc, 2)
        errs = self.read(out, "errors.csv")
        self.assertEqual([e["pdb_id"] for e in errs], ["9P02"])
        cands = {r["pdb_id"] for r in self.read(out, "pool_candidates.csv")}
        self.assertNotIn("9P02", cands)
        flow = {r["step"]: r for r in self.read(out, "pool_flow.csv")}
        self.assertEqual(int(flow["download_error"]["n_removed_at_this_step"]), 1)
        self.mock.cif_404.clear()
        self.mock.cif_calls.clear()
        rc, out = self.run_pool("--no-canary", out=out)
        self.assertEqual(rc, 0)
        self.assertEqual(sum(self.mock.cif_calls.values()), 1 + 0)  # only the failed entry is fetched again
        cands = {r["pdb_id"] for r in self.read(out, "pool_candidates.csv")}
        self.assertEqual(sorted(cands), self.passing)

    def test_audit_catches_a_bad_prefilter(self):
        # the server wrongly reports 9P12 as 9.9 A, so the resolution prefilter removes a good entry
        self.mock.entries["9P12"]["res"] = 9.9
        rc, out = self.run_pool("--no-canary", "--prefilter")
        info = self.info(out)
        self.assertEqual(info["prefilter_audit"]["ids_that_pass"], ["9P12"])
        log = self.text(out, "run.log")
        self.assertIn("Run again without --prefilter", log)

    def test_canary_ok_and_failure(self):
        rc, out = self.run_pool()
        self.assertEqual(rc, 0)
        info = self.info(out)
        self.assertEqual(len(info["canary"]), 2)
        # a canary file without the reconstruction method must stop the run
        self.mock.entries["5A1A"]["text"] = make_cif("5A1A", recon=None, resolution=None, emdb=(),
                                                     release="2015-07-15", entities=[protein(1, 40, copies=3)])
        rc, out2 = self.run_pool(out=os.path.join(self.tmp, "out2"))
        self.assertEqual(rc, 3)
        self.assertFalse(os.path.exists(os.path.join(out2, "pool_candidates.csv")))

    def test_search_rejected_everywhere_stops_cleanly(self):
        self.mock.reject_attrs.update({"exptl.method", "rcsb_entry_info.experimental_method"})
        rc, out = self.run_pool("--no-canary")
        self.assertEqual(rc, 4)
        self.assertIn("Search failed", self.text(out, "run.log"))

    def test_search_query_variants(self):
        # the API refuses the first form of every part of the base query
        self.mock.reject_plain_date = True
        self.mock.reject_paginate = True
        self.mock.reject_attrs.add("exptl.method")
        rc, out = self.run_pool("--no-canary")
        self.assertEqual(rc, 0)
        self.assertEqual(self.info(out)["search_config"], {"date": 1, "method": 1, "ro": 1})
        cands = self.read(out, "pool_candidates.csv")
        self.assertEqual(sorted(r["pdb_id"] for r in cands), self.passing)

    def test_truncated_search_is_split(self):
        self.mock.max_rows = 10
        rc, out = self.run_pool("--no-canary")
        self.assertEqual(rc, 0)
        self.assertEqual(self.info(out)["n_cryoem_in_window"], self.n_all)
        cands = self.read(out, "pool_candidates.csv")
        self.assertEqual(sorted(r["pdb_id"] for r in cands), self.passing)

    def test_emdb_skip(self):
        rc, out = self.run_pool("--no-canary", "--skip-emdb")
        self.assertEqual(rc, 0)
        rows = {r["pdb_id"]: r for r in self.read(out, "pool_candidates.csv")}
        self.assertIn("9P10", rows)  # not checked, so not removed
        self.assertEqual(rows["9P10"]["emdb_release_verified"], "skipped")


class ParserTests(unittest.TestCase):
    spec = dict(pp.SPEC)

    def parse(self, text, pid="9XXX"):
        return pp.parse_mmcif(text, pid, self.spec)

    def test_window_edges(self):
        r = self.parse(make_cif("9A", release="2025-12-31"))
        self.assertEqual(r["first_fail"], "window")
        r = self.parse(make_cif("9B", release="2026-01-01"))
        self.assertIsNone(r["first_fail"])
        r = self.parse(make_cif("9C", release="2026-10-12"))
        self.assertIsNone(r["first_fail"])
        r = self.parse(make_cif("9D", release="2026-10-13"))
        self.assertEqual(r["first_fail"], "window")

    def test_first_revision_date_is_the_release_date(self):
        r = self.parse(make_cif("9A", release="2026-02-04"))
        self.assertEqual(r["release_date_pdb"], "2026-02-04")  # the later 2027-01-01 revision is ignored

    def test_resolution_edges(self):
        self.assertIsNone(self.parse(make_cif("9A", resolution="4.0"))["first_fail"])
        self.assertEqual(self.parse(make_cif("9A", resolution="4.01"))["first_fail"], "resolution")
        self.assertEqual(self.parse(make_cif("9A", resolution=None))["first_fail"], "resolution")

    def test_residue_and_unk_edges(self):
        r = self.parse(make_cif("9A", entities=[protein(1, 100, copies=7)]))
        self.assertIsNone(r["first_fail"])
        r = self.parse(make_cif("9A", entities=[protein(1, 101, copies=7)]))
        self.assertEqual(r["first_fail"], "modeled_residues")
        r = self.parse(make_cif("9A", entities=[protein(1, 100, unk=5)]))
        self.assertIsNone(r["first_fail"])
        r = self.parse(make_cif("9A", entities=[protein(1, 100, unk=6)]))
        self.assertEqual(r["first_fail"], "unk_fraction")

    def test_unmodeled_residues_do_not_count(self):
        # 300 residues in the sequence, 50 modeled: counted by what is in the coordinates
        r = self.parse(make_cif("9A", entities=[protein(1, 300, modeled=50)]))
        self.assertEqual(r["n_modeled_residues"], 50)
        self.assertEqual(r["longest_seq_len"], 300)
        self.assertEqual(r["seqres_total_len"], 300)

    def test_identical_sequences_count_once(self):
        a, b = protein(1, 40), protein(2, 40)
        b["seq"] = a["seq"]
        r = self.parse(make_cif("9A", entities=[a, b]))
        self.assertEqual(r["n_unique_seq"], 1)

    def test_alt_locs_are_one_residue(self):
        text = make_cif("9A", entities=[protein(1, 10)])
        lines = text.split("\n")
        # duplicate the CA atom of residue 3 as an alternate conformer
        idx = next(i for i, ln in enumerate(lines) if ln.startswith("ATOM") and " CA " in ln and " 3 ? " in ln)
        lines.insert(idx + 1, lines[idx].replace(" . ", " B ", 1))
        r = self.parse("\n".join(lines))
        self.assertEqual(r["n_modeled_residues"], 10)

    def test_second_model_is_ignored(self):
        text = make_cif("9A", entities=[protein(1, 10)])
        lines = text.split("\n")
        extra = [ln[:-1] + "2" for ln in lines if ln.startswith("ATOM")]
        i = max(i for i, ln in enumerate(lines) if ln.startswith("HETATM"))
        r = self.parse("\n".join(lines[:i] + extra + lines[i:]))
        self.assertEqual(r["n_modeled_residues"], 10)

    def test_helical_and_mixed_methods_fail(self):
        self.assertEqual(self.parse(make_cif("9A", recon="HELICAL"))["first_fail"], "single_particle")
        self.assertEqual(self.parse(make_cif("9A", method="X-RAY DIFFRACTION"))["first_fail"], "single_particle")

    def test_emd_normalisation(self):
        self.assertEqual(pp.norm_emd("EMD-1234"), "EMD-1234")
        self.assertEqual(pp.norm_emd("emd_1234"), "EMD-1234")
        self.assertEqual(pp.norm_emd("1234"), "EMD-1234")
        self.assertEqual(pp.norm_emd(""), "")

    def test_emdb_json_parsing(self):
        js = {"admin": {"current_status": {"code": {"valueOf_": "REL"}},
                        "key_dates": {"header_release": "2026-02-04", "map_release": "2026-02-11"}},
              "map": {"size_kbytes": 99}}
        out = pp.parse_emdb_json(js)
        self.assertEqual((out["map_release"], out["status"], out["size_kb"], out["release_basis"]),
                         ("2026-02-11", "REL", 99, "map_release"))
        out = pp.parse_emdb_json({"admin": {"key_dates": {"header_release": "2026-02-04"}}})
        self.assertEqual((out["map_release"], out["release_basis"]), ("2026-02-04", "header_release"))
        out = pp.parse_emdb_json({"nothing": 1})
        self.assertIsNone(out["map_release"])

    def test_week_windows_cover_the_window_exactly(self):
        import datetime as dt
        w = pp.week_windows(dt.date(2026, 1, 1), dt.date(2026, 10, 12))
        self.assertEqual(w[0][0], dt.date(2026, 1, 1))
        self.assertEqual(w[-1][1], dt.date(2026, 10, 13))
        for (a, b), (c, d) in zip(w, w[1:]):
            self.assertEqual(b, c)
        self.assertTrue(all((b - a).days <= 7 for a, b in w))


if __name__ == "__main__":
    unittest.main(verbosity=2)
