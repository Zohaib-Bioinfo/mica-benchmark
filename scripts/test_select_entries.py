#!/usr/bin/env python3
"""Offline tests for select_entries.py (Phase 3b).

    python scripts/test_select_entries.py

The clustering plumbing is tested with a stand-in for the mmseqs program. One more test runs the real
MMseqs2 and is skipped unless a binary is on the PATH or named in the MMSEQS environment variable.
One test runs the pool script against its mock servers and feeds the pool into the selection script, so
a change of a column name or of the FASTA header in one script cannot go unnoticed in the other.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import shutil
import stat
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pull_pool as pp  # noqa: E402
import select_entries as se  # noqa: E402
import test_pull_pool as tp  # noqa: E402

DECISIONS = {"selected", "not_reached", "not_evaluated", "needs_cc", "skip_cluster_used", "skip_map_used",
             "reject_cc_mask", "reject_cc_failed"}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def candidate(pid, emdb, stratum, res):
    return {"pdb_id": pid, "emdb_id": emdb, "stratum": stratum, "release_date_pdb": "2026-03-01",
            "emdb_map_release": "2026-03-02", "map_size_kb": "65536", "resolution_A": res,
            "n_modeled_residues": "300", "n_unique_seq": "1", "unk_fraction": "0.0", "n_chains": "2",
            "longest_seq_len": "300", "cif_url": "https://files.rcsb.org/download/%s.cif" % pid,
            "map_url": "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/%s/map/x.map.gz" % emdb}


class World:
    """A synthetic pool: candidate rows plus the cluster of each one."""

    def __init__(self):
        self.cands, self.rep_of = [], {}

    def add(self, stratum="le3.0", cluster=None, emdb=None):
        n = len(self.cands) + 1
        pid = "P%03d" % n
        res = "2.50" if stratum == "le3.0" else "3.50"
        self.cands.append(candidate(pid, emdb or "EMD-%d" % (50000 + n), stratum, res))
        self.rep_of[pid] = cluster or pid
        return pid

    def ids(self, stratum):
        return [c["pdb_id"] for c in self.cands if c["stratum"] == stratum]

    def order(self, stratum):
        return se.shuffled(self.ids(stratum))

    def walk(self, cc, **kw):
        return se.walk(self.cands, self.rep_of, cc, **kw)


def ok(pids, value=0.90):
    return {p: ("ok", value, "") for p in pids}


def decisions(log):
    return {r["pdb_id"]: r["decision"] for r in log}


def chosen(selected):
    return [s["row"]["pdb_id"] for s in selected]


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = se.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


STUB = """#!@PYTHON@
import json, os, shutil, sys
args = sys.argv[1:]
here = os.path.dirname(os.path.abspath(__file__))
if args[:1] == ["version"]:
    print("stub-1.0")
    sys.exit(0)
if os.environ.get("STUB_FAIL"):
    sys.stderr.write("stub: out of memory\\n")
    sys.exit(7)
assert args[0] == "easy-cluster", args
fasta, prefix = args[1], args[2]
with open(os.path.join(here, "argv.json"), "w") as f:
    json.dump(args, f)
shutil.copy(fasta, os.path.join(here, "input.fasta"))
seqs, name = {}, None
for line in open(fasta):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:]
        seqs[name] = ""
    else:
        seqs[name] += line
n = int(os.environ.get("STUB_PREFIX", "5"))   # sequences that start alike (first n letters) share a cluster
rep, lines = {}, []
for name, s in seqs.items():
    rep.setdefault(s[:n], name)
    lines.append(rep[s[:n]] + "\\t" + name)
if os.environ.get("STUB_DROP"):
    lines = lines[:-1]
with open(prefix + "_cluster.tsv", "w") as f:
    f.write("\\n".join(lines) + "\\n")
"""


def make_stub(folder):
    path = os.path.join(folder, "mmseqs")
    with open(path, "w") as f:
        f.write(STUB.replace("@PYTHON@", sys.executable))
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def write_fasta(path, entities):
    """entities: (pdb_id, entity id, sequence, is_longest)"""
    with open(path, "w", encoding="utf-8") as f:
        for pid, eid, seq, longest in entities:
            f.write(">%s_E%s len=%d copies=1 longest=%d\n%s\n" % (pid, eid, len(seq), 1 if longest else 0, seq))


def write_pool_files(pool, cands):
    os.makedirs(pool, exist_ok=True)
    se.write_csv(os.path.join(pool, "pool_candidates.csv"), list(cands[0].keys()), cands)


def write_cluster_file(path, w):
    sizes = Counter(w.rep_of.values())
    se.write_csv(path, ["pdb_id", "cluster_id", "cluster_size", "longest_seq_len"],
                 [{"pdb_id": p, "cluster_id": c, "cluster_size": sizes[c], "longest_seq_len": 300}
                  for p, c in sorted(w.rep_of.items())])


def write_cc_file(path, cc):
    se.write_csv(path, ["pdb_id", "status", "cc_mask", "note"],
                 [{"pdb_id": p, "status": s, "cc_mask": "" if v is None else repr(v), "note": n}
                  for p, (s, v, n) in sorted(cc.items())])


# --------------------------------------------------------------------------------------
# The shuffle
# --------------------------------------------------------------------------------------
class ShuffleTests(unittest.TestCase):
    IDS = ["P%02d" % i for i in range(1, 13)]

    def test_reference_order_is_pinned(self):
        # If this fails, random.shuffle of this Python differs from the one used for the benchmark, and the
        # selection would differ too. The Python version is recorded in selection_info.json.
        self.assertEqual(se.shuffled(self.IDS, 2026),
                         ["P01", "P07", "P03", "P04", "P11", "P05", "P08", "P12", "P10", "P09", "P06", "P02"])
        self.assertEqual(se.shuffled(self.IDS, 1),
                         ["P08", "P12", "P01", "P09", "P06", "P07", "P04", "P11", "P05", "P02", "P10", "P03"])

    def test_the_seed_of_the_spec_is_the_default(self):
        self.assertEqual(se.SEED, 2026)
        self.assertEqual(se.shuffled(self.IDS), se.shuffled(self.IDS, 2026))

    def test_input_order_does_not_matter(self):
        rev = list(reversed(self.IDS))
        mixed = random.Random(5).sample(self.IDS, len(self.IDS))
        self.assertEqual(se.shuffled(rev), se.shuffled(self.IDS))
        self.assertEqual(se.shuffled(mixed), se.shuffled(self.IDS))

    def test_the_input_list_is_not_changed(self):
        ids = list(reversed(self.IDS))
        before = list(ids)
        se.shuffled(ids)
        self.assertEqual(ids, before)

    def test_another_seed_gives_another_order(self):
        self.assertNotEqual(se.shuffled(self.IDS, 1), se.shuffled(self.IDS, 2026))

    def test_spec_constants(self):
        self.assertEqual((se.QUOTA, se.CC_MIN, se.MIN_SEQ_ID), (10, 0.70, 0.30))
        self.assertEqual(se.STRATA, ("le3.0", "gt3.0_le4.0"))


# --------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------
class WalkTests(unittest.TestCase):
    def test_takes_the_first_ten_in_shuffled_order_in_each_stratum(self):
        w = World()
        for _ in range(14):
            w.add("le3.0")
        for _ in range(13):
            w.add("gt3.0_le4.0")
        log, selected, todo = w.walk(ok([c["pdb_id"] for c in w.cands]))
        self.assertEqual(todo, [])
        self.assertEqual(len(log), 27)
        for stratum in se.STRATA:
            expect = w.order(stratum)
            rows = [r for r in log if r["stratum"] == stratum]
            self.assertEqual([r["pdb_id"] for r in rows], expect)          # the log follows the walk
            self.assertEqual([r["seed_rank"] for r in rows], list(range(1, len(expect) + 1)))
            self.assertEqual([r["decision"] for r in rows], ["selected"] * 10 + ["not_reached"] * (len(expect) - 10))
            self.assertEqual([p for p in chosen(selected) if p in expect], expect[:10])

    def test_the_log_has_one_row_per_candidate_with_a_known_decision(self):
        w = World()
        for i in range(30):
            w.add("le3.0" if i % 2 else "gt3.0_le4.0", cluster="C%d" % (i % 7))
        log, _, _ = w.walk({p: ("ok", 0.5 + 0.02 * i, "") for i, p in enumerate(w.rep_of)})
        self.assertEqual(sorted(r["pdb_id"] for r in log), sorted(w.rep_of))
        self.assertTrue({r["decision"] for r in log} <= DECISIONS)

    def test_one_entry_per_cluster_and_the_first_in_the_walk_wins(self):
        w = World()
        xs = [w.add("le3.0", cluster="X") for _ in range(4)]
        ys = [w.add("le3.0", cluster="Y") for _ in range(2)]
        alone = w.add("le3.0")
        log, selected, todo = w.walk(ok(w.rep_of))
        order = w.order("le3.0")
        first_x = next(p for p in order if p in xs)
        first_y = next(p for p in order if p in ys)
        self.assertEqual(todo, [])
        self.assertEqual(sorted(chosen(selected)), sorted([first_x, first_y, alone]))
        rows = {r["pdb_id"]: r for r in log}
        for p in xs:
            if p != first_x:
                self.assertEqual(rows[p]["decision"], "skip_cluster_used")
                self.assertIn("cluster X used by %s" % first_x, rows[p]["reason"])

    def test_a_cluster_used_in_the_first_stratum_is_closed_in_the_second(self):
        w = World()
        a = w.add("le3.0", cluster="X")
        b = w.add("gt3.0_le4.0", cluster="X")
        log, selected, _ = w.walk(ok([a, b]))
        self.assertEqual(chosen(selected), [a])
        self.assertEqual(decisions(log)[b], "skip_cluster_used")

    def test_the_second_stratum_is_open_when_the_first_entry_of_the_cluster_was_rejected(self):
        w = World()
        a = w.add("le3.0", cluster="X")
        b = w.add("gt3.0_le4.0", cluster="X")
        log, selected, _ = w.walk({a: ("ok", 0.40, ""), b: ("ok", 0.90, "")})
        self.assertEqual(chosen(selected), [b])
        self.assertEqual(decisions(log)[a], "reject_cc_mask")

    def test_an_emdb_map_is_used_once_even_across_strata(self):
        w = World()
        a = w.add("le3.0", emdb="EMD-1")
        b = w.add("le3.0", emdb="EMD-1")
        c = w.add("gt3.0_le4.0", emdb="EMD-1")
        log, selected, _ = w.walk(ok([a, b, c]))
        first = next(p for p in w.order("le3.0") if p in (a, b))
        other = b if first == a else a
        self.assertEqual(chosen(selected), [first])
        rows = {r["pdb_id"]: r for r in log}
        self.assertEqual(rows[other]["decision"], "skip_map_used")
        self.assertIn("EMD-1 used by %s" % first, rows[other]["reason"])
        self.assertEqual(rows[c]["decision"], "skip_map_used")

    def test_the_cc_threshold_is_inclusive(self):
        w = World()
        a, b, c = w.add(), w.add(), w.add()
        log, selected, _ = w.walk({a: ("ok", 0.70, ""), b: ("ok", 0.6999, ""), c: ("ok", 0.7001, "")})
        self.assertEqual(sorted(chosen(selected)), sorted([a, c]))
        self.assertEqual(decisions(log)[b], "reject_cc_mask")

    def test_a_rejected_entry_does_not_use_up_its_cluster_or_its_map(self):
        for result in (("ok", 0.50, ""), ("failed", None, "phenix stopped")):
            w = World()
            a = w.add("le3.0", cluster="X", emdb="EMD-7")
            b = w.add("le3.0", cluster="X", emdb="EMD-7")
            first, second = [p for p in w.order("le3.0")]
            log, selected, _ = w.walk({first: result, second: ("ok", 0.90, "")})
            self.assertEqual(chosen(selected), [second])
            expect = "reject_cc_mask" if result[0] == "ok" else "reject_cc_failed"
            self.assertEqual(decisions(log)[first], expect)
            self.assertEqual({a, b}, {first, second})

    def test_a_failed_computation_is_a_rejection_with_its_note(self):
        w = World()
        a = w.add()
        log, selected, todo = w.walk({a: ("failed", None, "phenix.map_model_cc exited 1")})
        self.assertEqual((selected, todo), ([], []))
        self.assertEqual(log[0]["decision"], "reject_cc_failed")
        self.assertIn("phenix.map_model_cc exited 1", log[0]["reason"])
        self.assertEqual(log[0]["cc_mask"], "")

    def test_the_selected_row_records_cc_mask_with_four_decimals(self):
        w = World()
        a = w.add()
        log, selected, _ = w.walk({a: ("ok", 0.8123456, "")})
        self.assertEqual(log[0]["cc_mask"], "0.8123")
        self.assertEqual(selected[0]["cc"], 0.8123456)

    def test_fewer_candidates_than_the_quota(self):
        w = World()
        ids = [w.add("le3.0") for _ in range(4)]
        log, selected, todo = w.walk(ok(ids))
        self.assertEqual((len(selected), todo), (4, []))
        self.assertEqual(set(decisions(log).values()), {"selected"})

    def test_everything_rejected_is_a_complete_empty_result(self):
        w = World()
        ids = [w.add("le3.0") for _ in range(5)]
        log, selected, todo = w.walk(ok(ids, 0.3))
        self.assertEqual((selected, todo), ([], []))
        self.assertEqual(set(decisions(log).values()), {"reject_cc_mask"})

    def test_a_stratum_without_candidates_is_fine(self):
        w = World()
        ids = [w.add("gt3.0_le4.0") for _ in range(3)]
        log, selected, todo = w.walk(ok(ids))
        self.assertEqual(len(selected), 3)
        self.assertEqual(todo, [])

    def test_quota_and_cc_minimum_can_be_changed_for_a_dry_run(self):
        w = World()
        ids = [w.add("le3.0") for _ in range(8)]
        _, selected, _ = w.walk(ok(ids, 0.75), quota=3, cc_min=0.80)
        self.assertEqual(selected, [])
        _, selected, _ = w.walk(ok(ids, 0.75), quota=3, cc_min=0.70)
        self.assertEqual(chosen(selected), w.order("le3.0")[:3])

    def test_the_seed_changes_the_walk_and_nothing_else(self):
        w = World()
        for _ in range(25):
            w.add("le3.0")
        cc = ok(w.rep_of)
        _, a, _ = w.walk(cc, seed=2026)
        _, b, _ = w.walk(cc, seed=7)
        self.assertEqual(chosen(a), w.order("le3.0")[:10])
        self.assertEqual(chosen(b), se.shuffled(w.ids("le3.0"), 7)[:10])
        self.assertNotEqual(chosen(a), chosen(b))

    def test_the_result_does_not_depend_on_the_row_order_of_the_pool(self):
        w = World()
        for i in range(30):
            w.add("le3.0" if i % 3 else "gt3.0_le4.0", cluster="C%d" % (i % 9))
        cc = {p: ("ok", 0.55 + 0.015 * i, "") for i, p in enumerate(sorted(w.rep_of))}
        log1, sel1, _ = w.walk(cc)
        w.cands.reverse()
        log2, sel2, _ = w.walk(cc)
        self.assertEqual(log1, log2)
        self.assertEqual(chosen(sel1), chosen(sel2))


class UnknownCcTests(unittest.TestCase):
    def pool(self, n=40, stratum="le3.0"):
        w = World()
        for _ in range(n):
            w.add(stratum)
        return w

    def test_an_unknown_cc_stops_the_walk_of_its_stratum(self):
        w = self.pool(15)
        order = w.order("le3.0")
        log, selected, todo = w.walk(ok(order[:2]))
        d = decisions(log)
        self.assertEqual([d[p] for p in order[:2]], ["selected", "selected"])
        self.assertEqual(d[order[2]], "needs_cc")
        self.assertEqual({d[p] for p in order[3:]}, {"not_evaluated"})
        self.assertEqual(len(selected), 2)
        self.assertEqual([t["pdb_id"] for t in todo], order[2:])          # 13 left; 20 would be wanted

    def test_how_many_entries_are_requested(self):
        # need = 10 - known passes; request ceil(2.5 * need), at least 5, at most 25
        for known, want in ((0, 25), (3, 18), (6, 10), (8, 5), (9, 5)):
            w = self.pool(40)
            order = w.order("le3.0")
            _, _, todo = w.walk(ok(order[:known]))
            self.assertEqual([t["seed_rank"] for t in todo], list(range(known + 1, known + 1 + want)), "known=%d" % known)

    def test_batch_size_caps_the_request(self):
        w = self.pool(40)
        _, _, todo = w.walk({}, batch=12)
        self.assertEqual(len(todo), 12)

    def test_the_request_skips_entries_whose_cluster_an_earlier_one_would_use(self):
        w = self.pool(30)
        order = w.order("le3.0")
        w.rep_of[order[1]] = w.rep_of[order[0]]        # same cluster as the first: not needed if the first passes
        _, _, todo = w.walk({})
        ranks = [t["seed_rank"] for t in todo]
        self.assertEqual(ranks[:3], [1, 3, 4])
        self.assertEqual(len(ranks), 25)

    def test_the_request_skips_entries_whose_map_an_earlier_one_would_use(self):
        w = self.pool(30)
        order = w.order("le3.0")
        by_id = {c["pdb_id"]: c for c in w.cands}
        by_id[order[2]]["emdb_id"] = by_id[order[0]]["emdb_id"]
        _, _, todo = w.walk({})
        self.assertNotIn(2 + 1, [t["seed_rank"] for t in todo])
        self.assertEqual([t["seed_rank"] for t in todo][:3], [1, 2, 4])

    def test_the_request_leaves_out_entries_that_are_known_to_fail_and_counts_known_passes(self):
        w = self.pool(40)
        order = w.order("le3.0")
        cc = {order[0]: ("ok", 0.4, ""), order[2]: ("ok", 0.9, ""), order[3]: ("failed", None, "x")}
        log, selected, todo = w.walk(cc)
        d = decisions(log)
        self.assertEqual((d[order[0]], d[order[1]], d[order[2]]), ("reject_cc_mask", "needs_cc", "not_evaluated"))
        ranks = [t["seed_rank"] for t in todo]
        self.assertNotIn(1, ranks)
        self.assertNotIn(3, ranks)      # known to pass: not requested again
        self.assertNotIn(4, ranks)      # known to fail
        self.assertEqual(ranks[:2], [2, 5])
        self.assertEqual(len(ranks) + 1, 25)      # 25 assumed passes, one of them (rank 3) already known

    def test_no_request_for_an_entry_that_the_rule_skips_before_looking_at_cc(self):
        w = self.pool(30)
        order = w.order("le3.0")
        w.rep_of[order[1]] = w.rep_of[order[0]]
        log, selected, todo = w.walk(ok(order[:1]))
        d = decisions(log)
        self.assertEqual(d[order[1]], "skip_cluster_used")
        self.assertEqual(d[order[2]], "needs_cc")
        self.assertNotIn(order[1], [t["pdb_id"] for t in todo])

    def test_unknown_entries_after_the_quota_is_reached_are_not_requested(self):
        w = self.pool(15)
        order = w.order("le3.0")
        log, selected, todo = w.walk(ok(order[:10]))
        self.assertEqual((len(selected), todo), (10, []))
        self.assertEqual({decisions(log)[p] for p in order[10:]}, {"not_reached"})

    def test_both_strata_are_requested_in_the_same_round(self):
        w = World()
        for _ in range(30):
            w.add("le3.0")
            w.add("gt3.0_le4.0")
        _, _, todo = w.walk({})
        self.assertEqual(Counter(t["stratum"] for t in todo), {"le3.0": 25, "gt3.0_le4.0": 25})

    def test_the_second_stratum_can_be_open_while_the_first_is_complete(self):
        w = World()
        first = [w.add("le3.0") for _ in range(10)]
        for _ in range(12):
            w.add("gt3.0_le4.0")
        _, selected, todo = w.walk(ok(first))
        self.assertEqual(len(selected), 10)
        self.assertEqual({t["stratum"] for t in todo}, {"gt3.0_le4.0"})

    def test_the_request_has_what_the_notebook_needs(self):
        w = self.pool(10)
        _, _, todo = w.walk({})
        self.assertEqual(list(todo[0].keys()), se.TODO_COLUMNS)
        t = todo[0]
        self.assertTrue(t["cif_url"].endswith("%s.cif" % t["pdb_id"]))
        self.assertIn(t["emdb_id"], t["map_url"])
        self.assertEqual(t["map_size_mb"], 64.0)

    def test_a_missing_map_size_is_left_empty(self):
        w = self.pool(5)
        for c in w.cands:
            c["map_size_kb"] = ""
        _, _, todo = w.walk({})
        self.assertEqual({t["map_size_mb"] for t in todo}, {""})


class RoundsTests(unittest.TestCase):
    """The protocol: ask for CC_mask in rounds. The result must be what one run with all values would give."""

    def make_world(self, rnd):
        w = World()
        n1, n2 = rnd.randint(6, 45), rnd.randint(6, 45)
        n_clusters = max(3, (n1 + n2) // rnd.choice((2, 3, 4)))
        n_maps = n1 + n2 + 8
        for stratum, n in (("le3.0", n1), ("gt3.0_le4.0", n2)):
            for _ in range(n):
                emdb = "EMD-S%d" % rnd.randrange(n_maps) if rnd.random() < 0.12 else None
                w.add(stratum, cluster="C%d" % rnd.randrange(n_clusters), emdb=emdb)
        return w

    def test_rounds_give_the_same_list_as_knowing_every_value(self):
        for seed in range(60):
            rnd = random.Random(seed)
            w = self.make_world(rnd)
            truth = {}
            for p in w.rep_of:
                truth[p] = ("failed", None, "x") if rnd.random() < 0.05 else ("ok", rnd.uniform(0.50, 0.95), "")
            quota = rnd.choice((3, 5, 10))
            batch = rnd.choice((5, 10, 25))
            log0, sel0, todo0 = w.walk(truth, quota=quota, batch=batch)
            self.assertEqual(todo0, [], "seed %d" % seed)
            cc, rounds, requested = {}, 0, Counter()
            while True:
                log, sel, todo = w.walk(cc, quota=quota, batch=batch)
                if not todo:
                    break
                rounds += 1
                self.assertLess(rounds, 25, "seed %d does not converge" % seed)
                for t in todo:
                    requested[t["pdb_id"]] += 1
                    self.assertNotIn(t["pdb_id"], cc, "seed %d asked again for %s" % (seed, t["pdb_id"]))
                    cc[t["pdb_id"]] = truth[t["pdb_id"]]
                for stratum in se.STRATA:
                    self.assertLessEqual(sum(1 for t in todo if t["stratum"] == stratum), batch)
            self.assertEqual(chosen(sel), chosen(sel0), "seed %d" % seed)
            self.assertEqual(log, log0, "seed %d" % seed)
            self.assertTrue(set(cc) <= set(truth))
            self.assertLessEqual(max(requested.values() or [1]), 1)


# --------------------------------------------------------------------------------------
# Reading the inputs
# --------------------------------------------------------------------------------------
class ReadInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sel_in_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def cc_file(self, text):
        with open(self.path("cc.csv"), "w", encoding="utf-8") as f:
            f.write(text)
        return self.path("cc.csv")

    def test_cc_file_values(self):
        p = self.cc_file("pdb_id,status,cc_mask,note\n9AAA,ok,0.8123,\n9BBB,failed,,phenix error\n9CCC,OK, 0.7 ,\n")
        cc = se.read_cc(p)
        self.assertEqual(cc["9AAA"], ("ok", 0.8123, ""))
        self.assertEqual(cc["9BBB"], ("failed", None, "phenix error"))
        self.assertEqual(cc["9CCC"], ("ok", 0.7, ""))

    def test_the_last_row_of_an_entry_wins(self):
        p = self.cc_file("pdb_id,status,cc_mask,note\n9AAA,failed,,first try\n9AAA,ok,0.81,\n")
        self.assertEqual(se.read_cc(p)["9AAA"][:2], ("ok", 0.81))

    def test_no_file_means_nothing_is_known(self):
        self.assertEqual(se.read_cc(None), {})

    def test_other_statuses_are_errors_not_rejections(self):
        for status in ("error", "timeout", ""):
            p = self.cc_file("pdb_id,status,cc_mask,note\n9AAA,%s,0.9,\n" % status)
            with self.assertRaises(ValueError) as cm:
                se.read_cc(p)
            self.assertIn("9AAA", str(cm.exception))

    def test_ok_without_a_usable_number_is_an_error(self):
        for value in ("", "abc", "nan", "inf", "-inf"):
            p = self.cc_file("pdb_id,status,cc_mask,note\n9AAA,ok,%s,\n" % value)
            with self.assertRaises(ValueError):
                se.read_cc(p)

    def test_a_row_without_an_id_is_an_error(self):
        with self.assertRaises(ValueError):
            se.read_cc(self.cc_file("pdb_id,status,cc_mask,note\n,ok,0.8,\n"))

    def test_candidates_with_an_unknown_stratum_or_a_repeated_id_are_errors(self):
        a = candidate("9AAA", "EMD-1", "le3.0", "2.0")
        b = candidate("9BBB", "EMD-2", "gt4.0", "5.0")
        write_pool_files(self.path("p1"), [a, b])
        with self.assertRaises(ValueError):
            se.read_candidates(self.path("p1"))
        write_pool_files(self.path("p2"), [a, dict(a)])
        with self.assertRaises(ValueError):
            se.read_candidates(self.path("p2"))

    def test_the_longest_chain_is_the_one_flagged_in_the_fasta(self):
        fa = self.path("seqs.fasta")
        write_fasta(fa, [("9AAA", "1", "MKV", False), ("9AAA", "2", "MKVLLAGG", True), ("9BBB", "1", "GGGGAAA", True),
                         ("9CCC", "1", "WWWW", True)])
        cands = [candidate("9AAA", "EMD-1", "le3.0", "2.0"), candidate("9BBB", "EMD-2", "le3.0", "2.1")]
        self.assertEqual(se.read_longest_sequences(fa, cands), {"9AAA": "MKVLLAGG", "9BBB": "GGGGAAA"})

    def test_a_sequence_wrapped_over_several_lines_is_read_whole(self):
        fa = self.path("wrap.fasta")
        with open(fa, "w") as f:
            f.write(">9AAA_E1 len=9 copies=1 longest=1\nMKVL\nLAGG\nS\n")
        got = se.read_longest_sequences(fa, [candidate("9AAA", "EMD-1", "le3.0", "2.0")])
        self.assertEqual(got["9AAA"], "MKVLLAGGS")

    def test_a_candidate_without_a_longest_chain_is_an_error(self):
        fa = self.path("none.fasta")
        write_fasta(fa, [("9AAA", "1", "MKV", False)])
        with self.assertRaises(ValueError):
            se.read_longest_sequences(fa, [candidate("9AAA", "EMD-1", "le3.0", "2.0")])
        write_fasta(fa, [("9AAA", "1", "MKV", True), ("9AAA", "2", "MKVLL", True)])
        with self.assertRaises(ValueError):
            se.read_longest_sequences(fa, [candidate("9AAA", "EMD-1", "le3.0", "2.0")])


# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------
class ClusterStubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sel_cl_")
        self.stub_dir = os.path.join(self.tmp, "bin")
        os.makedirs(self.stub_dir)
        self.stub = make_stub(self.stub_dir)
        self.pool = os.path.join(self.tmp, "pool")
        self.out = os.path.join(self.tmp, "out")
        cands = [candidate(p, "EMD-%d" % i, "le3.0", "2.0") for i, p in enumerate(("9CCC", "9AAA", "9BBB", "9DDD"), 1)]
        write_pool_files(self.pool, cands)
        write_fasta(os.path.join(self.pool, "pool_candidate_sequences.fasta"), [
            ("9CCC", "1", "AAAAAXXXXXXXXXXX", True),
            ("9AAA", "1", "MMMMMSHORT", False), ("9AAA", "2", "AAAAAYYYYYYYYYYYYYYYY", True),
            ("9BBB", "1", "GGGGGZZZZ", True),
            ("9DDD", "1", "GGGGGWWWWWWWWWWWWW", True), ("9DDD", "2", "TTTTT", False)])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cluster(self, **env):
        with mock.patch.dict(os.environ, env):
            return run_cli("cluster", "--pool", self.pool, "--out", self.out, "--mmseqs", self.stub)

    def test_clusters_and_files(self):
        code, out, err = self.cluster()
        self.assertEqual(code, 0, err)
        rows = {r["pdb_id"]: r for r in read_rows(os.path.join(self.out, "candidate_clusters.csv"))}
        self.assertEqual(sorted(rows), ["9AAA", "9BBB", "9CCC", "9DDD"])
        # the stub clusters on the first five letters; the representative is the first in the (sorted) input
        self.assertEqual({p: rows[p]["cluster_id"] for p in rows}, {"9AAA": "9AAA", "9CCC": "9AAA", "9BBB": "9BBB", "9DDD": "9BBB"})
        self.assertEqual({p: rows[p]["cluster_size"] for p in rows}, {p: "2" for p in rows})
        self.assertEqual(rows["9AAA"]["longest_seq_len"], "21")      # the longest chain, not the first
        self.assertEqual(rows["9DDD"]["longest_seq_len"], "18")
        info = read_json(os.path.join(self.out, "mmseqs_info.json"))
        self.assertEqual((info["n_sequences"], info["n_clusters"]), (4, 2))
        self.assertEqual(info["mmseqs_version"], "stub-1.0")
        self.assertEqual((info["min_seq_id"], info["coverage"], info["cov_mode"], info["threads"]), (0.3, 0.8, 1, 1))
        self.assertEqual(info["pool_candidates_sha256"],
                         sha256_of(os.path.join(self.pool, "pool_candidates.csv")))
        self.assertIn("2 clusters", out)

    def test_mmseqs_gets_the_documented_command_and_the_longest_chains_in_id_order(self):
        self.assertEqual(self.cluster()[0], 0)
        argv = read_json(os.path.join(self.stub_dir, "argv.json"))
        self.assertEqual(argv[0], "easy-cluster")
        self.assertEqual(argv[4:], ["--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "1", "--threads", "1", "-v", "1"])
        self.assertEqual(read_text(os.path.join(self.stub_dir, "input.fasta")),
                         ">9AAA\nAAAAAYYYYYYYYYYYYYYYY\n>9BBB\nGGGGGZZZZ\n>9CCC\nAAAAAXXXXXXXXXXX\n>9DDD\nGGGGGWWWWWWWWWWWWW\n")

    def test_the_command_in_the_info_file_has_no_temporary_paths(self):
        self.cluster()
        info = read_json(os.path.join(self.out, "mmseqs_info.json"))
        self.assertTrue(all(not c.startswith(tempfile.gettempdir()) for c in info["command_template"]))

    def test_the_same_input_gives_the_same_clusters(self):
        self.cluster()
        first = read_text(os.path.join(self.out, "candidate_clusters.csv"))
        self.cluster()
        self.assertEqual(read_text(os.path.join(self.out, "candidate_clusters.csv")), first)

    def test_a_failing_mmseqs_stops_with_its_message(self):
        code, out, err = self.cluster(STUB_FAIL="1")
        self.assertEqual(code, 2)
        self.assertIn("Clustering failed", err)
        self.assertIn("out of memory", err)
        self.assertFalse(os.path.exists(os.path.join(self.out, "candidate_clusters.csv")))

    def test_a_sequence_without_a_cluster_stops_the_run(self):
        code, _, err = self.cluster(STUB_DROP="1")
        self.assertEqual(code, 2)
        self.assertIn("without a cluster", err)

    def test_a_missing_program_is_explained(self):
        code, _, err = run_cli("cluster", "--pool", self.pool, "--out", self.out, "--mmseqs", "/nonexistent/mmseqs")
        self.assertEqual(code, 2)
        self.assertIn("mmseqs not found", err)

    def test_a_candidate_missing_from_the_fasta_is_an_error(self):
        write_fasta(os.path.join(self.pool, "pool_candidate_sequences.fasta"), [("9AAA", "1", "AAAAAYYYY", True)])
        code, _, err = self.cluster()
        self.assertEqual(code, 2)
        self.assertIn("no longest-chain sequence", err)


MMSEQS = os.environ.get("MMSEQS") or shutil.which("mmseqs")


@unittest.skipUnless(MMSEQS, "no MMSEQS environment variable and no mmseqs on the PATH")
class RealMmseqsTests(unittest.TestCase):
    AA = "ACDEFGHIKLMNPQRSTVWY"

    def families(self):
        rnd = random.Random(7)

        def rand_seq(n):
            return "".join(rnd.choice(self.AA) for _ in range(n))

        def mutate(s, frac):
            s = list(s)
            for i in rnd.sample(range(len(s)), int(len(s) * frac)):
                s[i] = rnd.choice(self.AA)
            return "".join(s)

        fam = rand_seq(400)
        b1 = rand_seq(300)
        return {
            "A1": fam, "A2": mutate(fam, 0.05), "A3": mutate(fam, 0.40),
            "AFRAG": fam[100:250],            # identical to A1 over 150 residues, 38% of its length
            "AFRAGLONG": fam[0:330],          # identical to A1, 82% of its length
            "B1": b1, "B2": rand_seq(300), "B3": mutate(b1, 0.55)}

    @staticmethod
    def groups(rep_of):
        g = {}
        for k, v in rep_of.items():
            g.setdefault(v, set()).add(k)
        return sorted(sorted(x) for x in g.values())

    def test_homologues_and_fragments_share_a_cluster_and_unrelated_chains_do_not(self):
        rep_of, cmd = se.run_mmseqs(self.families(), MMSEQS)
        self.assertEqual(se.COV_MODE, 1)
        self.assertEqual(self.groups(rep_of), [["A1", "A2", "A3", "AFRAG", "AFRAGLONG"], ["B1", "B3"], ["B2"]])

    def test_the_reason_for_the_coverage_mode(self):
        # With coverage required in both directions (mode 0) a fragment stays in a cluster of its own.
        with mock.patch.object(se, "COV_MODE", 0):
            rep_of, _ = se.run_mmseqs(self.families(), MMSEQS)
        self.assertEqual(rep_of["AFRAG"], "AFRAG")

    def test_the_command_line_run_gives_the_same_clusters_twice(self):
        tmp = tempfile.mkdtemp(prefix="sel_real_")
        try:
            pool = os.path.join(tmp, "pool")
            cands, ents = [], []
            for i, (name, seq) in enumerate(sorted(self.families().items()), 1):
                pid = "9%s%s" % (name[:2].upper(), i)
                cands.append(candidate(pid, "EMD-%d" % i, "le3.0", "2.0"))
                ents.append((pid, "1", seq, True))
            write_pool_files(pool, cands)
            write_fasta(os.path.join(pool, "pool_candidate_sequences.fasta"), ents)
            outs = []
            for k in (1, 2):
                out = os.path.join(tmp, "out%d" % k)
                code, _, err = run_cli("cluster", "--pool", pool, "--out", out, "--mmseqs", MMSEQS)
                self.assertEqual(code, 0, err)
                outs.append(read_text(os.path.join(out, "candidate_clusters.csv")))
            self.assertEqual(outs[0], outs[1])
            info = read_json(os.path.join(tmp, "out1", "mmseqs_info.json"))
            self.assertNotIn("unknown", info["mmseqs_version"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------
class SelectCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sel_cli_")
        self.pool = os.path.join(self.tmp, "pool")
        self.out = os.path.join(self.tmp, "sel")
        self.entries = os.path.join(self.tmp, "entries_2026.csv")
        self.cc = os.path.join(self.out, "cc_mask.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def world(self, n_le=22, n_gt=22, clusters=None):
        w = World()
        for i in range(n_le):
            w.add("le3.0", cluster=("K%d" % (i % clusters)) if clusters else None)
        for i in range(n_gt):
            w.add("gt3.0_le4.0", cluster=("K%d" % (i % clusters)) if clusters else None)
        write_pool_files(self.pool, w.cands)
        os.makedirs(self.out, exist_ok=True)
        write_cluster_file(os.path.join(self.out, "candidate_clusters.csv"), w)
        return w

    def select(self, *extra):
        return run_cli("select", "--pool", self.pool, "--out", self.out, *extra)

    def run_rounds(self, w, truth, extra=()):
        """Plays the notebook: computes what cc_todo.csv asks for, until select is done. Returns the codes."""
        have, codes = {}, []
        for _ in range(20):
            args = ["--entries", self.entries] + list(extra) + (["--cc", self.cc] if have else [])
            code, out, err = self.select(*args)
            codes.append(code)
            if code != 3:
                self.assertEqual(code, 0, err + out)
                return codes
            for t in read_rows(os.path.join(self.out, "cc_todo.csv")):
                have[t["pdb_id"]] = truth[t["pdb_id"]]
            write_cc_file(self.cc, have)
        self.fail("no end after 20 rounds")

    def test_the_clusters_file_must_exist(self):
        w = self.world()
        os.remove(os.path.join(self.out, "candidate_clusters.csv"))
        code, _, err = self.select()
        self.assertEqual(code, 2)
        self.assertIn("run the cluster subcommand first", err)
        self.assertEqual(w.cands[0]["pdb_id"], "P001")

    def test_every_candidate_needs_a_cluster(self):
        self.world()
        rows = read_rows(os.path.join(self.out, "candidate_clusters.csv"))[1:]
        se.write_csv(os.path.join(self.out, "candidate_clusters.csv"), ["pdb_id", "cluster_id", "cluster_size", "longest_seq_len"], rows)
        code, _, err = self.select()
        self.assertEqual(code, 2)
        self.assertIn("have no cluster", err)

    def test_first_round_asks_for_cc_and_writes_no_list(self):
        self.world()
        code, out, err = self.select("--entries", self.entries)
        self.assertEqual(code, 3, err)
        self.assertIn("CC_mask is needed for 44 entries", out)
        self.assertFalse(os.path.exists(self.entries))
        todo = read_rows(os.path.join(self.out, "cc_todo.csv"))
        self.assertEqual(len(todo), 44)       # 22 candidates per stratum: fewer than the 25 asked for
        info = read_json(os.path.join(self.out, "selection_info.json"))
        self.assertFalse(info["final"])
        self.assertEqual(info["n_cc_requested"], 44)
        self.assertEqual(info["decisions"]["needs_cc"], 2)
        self.assertEqual(info["selected_per_stratum"], {"le3.0": 0, "gt3.0_le4.0": 0})

    def test_rounds_end_with_the_list(self):
        w = self.world()
        truth = {p: ("ok", 0.60 + 0.015 * (i % 20), "") for i, p in enumerate(sorted(w.rep_of))}
        codes = self.run_rounds(w, truth)
        self.assertEqual(codes[-1], 0)
        self.assertEqual(set(codes[:-1]), {3})
        rows = read_rows(self.entries)
        self.assertEqual(list(rows[0].keys()), se.ENTRY_COLUMNS)
        self.assertEqual(len(rows), 20)
        self.assertEqual(Counter(r["stratum"] for r in rows), {"le3.0": 10, "gt3.0_le4.0": 10})
        for r in rows:
            self.assertGreaterEqual(float(r["cc_mask"]), 0.70)
            self.assertRegex(r["cc_mask"], r"^\d\.\d{4}$")
        self.assertEqual(len({r["cluster_id"] for r in rows}), 20)
        self.assertEqual(len({r["emdb_id"] for r in rows}), 20)
        # the same list as one run that knows every value
        _, sel, todo = w.walk(truth)
        self.assertEqual(todo, [])
        self.assertEqual([r["pdb_id"] for r in rows], [p for s in se.STRATA for p in chosen(sel) if p in w.ids(s)])

    def test_the_list_has_the_values_of_the_pool_and_the_clusters(self):
        w = self.world(clusters=30)
        by_id = {c["pdb_id"]: c for c in w.cands}
        sizes = Counter(w.rep_of.values())
        truth = {p: ("ok", 0.85, "") for p in w.rep_of}
        self.run_rounds(w, truth)
        for r in read_rows(self.entries):
            c = by_id[r["pdb_id"]]
            self.assertEqual(r["emdb_id"], c["emdb_id"])
            self.assertEqual(r["emdb_release_date"], c["emdb_map_release"])
            self.assertEqual(r["pdb_release_date"], c["release_date_pdb"])
            self.assertEqual(r["resolution_A"], c["resolution_A"])
            self.assertEqual(r["modeled_residues"], c["n_modeled_residues"])
            self.assertEqual(r["unique_chain_seqs"], c["n_unique_seq"])
            self.assertEqual(r["cluster_id"], w.rep_of[r["pdb_id"]])
            self.assertEqual(r["cluster_size"], str(sizes[r["cluster_id"]]))
            self.assertEqual(r["map_size_mb"], "64.0")
            self.assertEqual((r["cif_url"], r["map_url"]), (c["cif_url"], c["map_url"]))
            self.assertEqual(r["max_id_to_pre2026_pdb"], "")       # filled in Phase 3c
            self.assertEqual(r["cc_mask"], "0.8500")
            self.assertEqual(r["seed_rank"], str(w.order(r["stratum"]).index(r["pdb_id"]) + 1))

    def test_info_file_of_the_final_round(self):
        w = self.world()
        truth = {p: ("ok", 0.9, "") for p in w.rep_of}
        self.run_rounds(w, truth)
        info = read_json(os.path.join(self.out, "selection_info.json"))
        self.assertTrue(info["final"])
        self.assertEqual((info["seed"], info["quota_per_stratum"], info["cc_min"]), (2026, 10, 0.7))
        self.assertEqual(info["selected_per_stratum"], {"le3.0": 10, "gt3.0_le4.0": 10})
        self.assertEqual(sum(info["decisions"].values()), 44)
        self.assertEqual(info["decisions"]["selected"], 20)
        self.assertEqual(info["decisions"]["not_reached"], 24)
        self.assertEqual(info["pool_candidates_sha256"],
                         sha256_of(os.path.join(self.pool, "pool_candidates.csv")))
        self.assertEqual(info["cc_file_sha256"], sha256_of(self.cc))
        self.assertEqual(info["first_in_walk"]["le3.0"], w.order("le3.0")[:5])
        self.assertEqual(info["python"].split(".")[0], "3")

    def test_rerun_gives_identical_files_and_row_order_of_the_pool_does_not_matter(self):
        w = self.world(clusters=25)
        truth = {p: ("ok", 0.55 + 0.01 * (i % 40), "") for i, p in enumerate(sorted(w.rep_of))}
        self.run_rounds(w, truth)
        first = {n: read_text(os.path.join(self.out, n)) for n in ("selection_log.csv", "cc_todo.csv")}
        first_entries = read_text(self.entries)
        w.cands.reverse()
        write_pool_files(self.pool, w.cands)
        code, _, err = self.select("--entries", self.entries, "--cc", self.cc)
        self.assertEqual(code, 0, err)
        for n, text in first.items():
            self.assertEqual(read_text(os.path.join(self.out, n)), text)
        self.assertEqual(read_text(self.entries), first_entries)

    def test_below_the_floor_and_below_the_target(self):
        # 5 passing in total -> below the floor
        w = self.world(n_le=3, n_gt=2)
        code, out, err = self.select("--entries", self.entries, "--cc", self._cc_all(w))
        self.assertEqual(code, 0, err)
        self.assertIn("FINAL: 5 entries", out)
        self.assertIn("WARNING: below the floor of 12", out)
        # 15 passing -> below the target, above the floor
        w = self.world(n_le=8, n_gt=7)
        code, out, err = self.select("--entries", self.entries, "--cc", self._cc_all(w))
        self.assertEqual(code, 0, err)
        self.assertIn("FINAL: 15 entries", out)
        self.assertIn("Note: below the target of 20", out)
        self.assertNotIn("WARNING", out)
        # 20 passing -> neither
        w = self.world(n_le=12, n_gt=12)
        code, out, err = self.select("--entries", self.entries, "--cc", self._cc_all(w))
        self.assertEqual(code, 0, err)
        self.assertIn("FINAL: 20 entries", out)
        self.assertNotIn("below", out)

    def _cc_all(self, w, value=0.9):
        write_cc_file(self.cc, {p: ("ok", value, "") for p in w.rep_of})
        return self.cc

    def test_a_bad_cc_file_stops_with_a_message(self):
        w = self.world(n_le=3, n_gt=2)
        with open(self.cc, "w") as f:
            f.write("pdb_id,status,cc_mask,note\n%s,timeout,,\n" % w.cands[0]["pdb_id"])
        code, _, err = self.select("--cc", self.cc)
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)
        self.assertIn("timeout", err)

    def test_a_missing_pool_file_stops_with_a_message(self):
        self.world()
        os.remove(os.path.join(self.pool, "pool_candidates.csv"))
        code, _, err = self.select()
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_the_entries_file_is_only_written_when_the_walk_is_complete(self):
        w = self.world()
        with open(self.entries, "w") as f:
            f.write("old\n")
        code, _, _ = self.select("--entries", self.entries)
        self.assertEqual(code, 3)
        self.assertEqual(read_text(self.entries), "old\n")
        code, _, _ = self.select("--entries", self.entries, "--cc", self._cc_all(w))
        self.assertEqual(code, 0)
        self.assertNotEqual(read_text(self.entries), "old\n")


# --------------------------------------------------------------------------------------
# Pool script -> selection script
# --------------------------------------------------------------------------------------
class PoolToSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock = tp.Mock()
        cls.srv = tp.serve(cls.mock)
        base = "http://127.0.0.1:%d" % cls.srv.server_address[1]
        cls.args = ["--search-url", base + "/search", "--cif-url", base + "/cif/{ID}.cif.gz",
                    "--cif-url", base + "/mirror/{id}.cif.gz", "--emdb-url", base + "/emdb/{EMD}",
                    "--threads", "4", "--emdb-threads", "2"]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        tp.fresh(self.mock)
        for k in range(24):     # entries with their own sequences, 2.0 to 3.84 A
            res = round(2.0 + 0.08 * k, 2)
            self.mock.add("9S%02d" % k, release="2026-%02d-%02d" % (1 + k % 9, 3 + k), res_combined=res,
                          resolution="%.2f" % res, emdb=("EMD-92%03d" % k,),
                          entities=[tp.protein(k + 2, 41 + k)], modeled=41 + k)
        self.tmp = tempfile.mkdtemp(prefix="sel_int_")
        self.pool = os.path.join(self.tmp, "pool")
        self.sel = os.path.join(self.tmp, "sel")
        self.stub = make_stub(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_pool_of_the_pool_script_goes_through_the_selection_script(self):
        self.assertEqual(pp.main(self.args + ["--out", self.pool, "--no-canary"]), 0)
        # every column the selection script reads is a column of the pool script
        needed = {"pdb_id", "emdb_id", "stratum", "release_date_pdb", "emdb_map_release", "map_size_kb", "resolution_A",
                  "n_modeled_residues", "n_unique_seq", "unk_fraction", "n_chains", "longest_seq_len", "cif_url", "map_url"}
        self.assertTrue(needed <= set(pp.COLUMNS), needed - set(pp.COLUMNS))
        cands = read_rows(os.path.join(self.pool, "pool_candidates.csv"))
        self.assertGreater(len(cands), 40)
        self.assertEqual({c["stratum"] for c in cands}, set(se.STRATA))

        with mock.patch.dict(os.environ, {"STUB_PREFIX": "100000"}):      # identical sequences share a cluster
            code, out, err = run_cli("cluster", "--pool", self.pool, "--out", self.sel, "--mmseqs", self.stub)
        self.assertEqual(code, 0, err)
        clusters = {r["pdb_id"]: r["cluster_id"] for r in read_rows(os.path.join(self.sel, "candidate_clusters.csv"))}
        self.assertEqual(set(clusters), {c["pdb_id"] for c in cands})
        self.assertLess(len(set(clusters.values())), len(clusters))       # the 21 mock "9F" entries share a sequence
        self.assertGreater(len(set(clusters.values())), 20)

        have, entries = {}, os.path.join(self.tmp, "entries_2026.csv")
        cc_path = os.path.join(self.sel, "cc_mask.csv")
        code = None
        for round_no in range(10):
            extra = ["--cc", cc_path] if have else []
            code, out, err = run_cli("select", "--pool", self.pool, "--out", self.sel, "--entries", entries, *extra)
            if code != 3:
                break
            for t in read_rows(os.path.join(self.sel, "cc_todo.csv")):
                self.assertTrue(t["cif_url"].startswith("http") and t["map_url"].startswith("http"))
                n = int(t["pdb_id"][-2:], 36) if t["pdb_id"][-2:].isalnum() else 0
                have[t["pdb_id"]] = ("failed", None, "no result") if n % 7 == 0 else ("ok", 0.55 + 0.04 * (n % 9), "")
            write_cc_file(cc_path, have)
        self.assertEqual(code, 0, err + out)
        rows = read_rows(entries)
        self.assertGreaterEqual(len(rows), 10)
        by_id = {c["pdb_id"]: c for c in cands}
        self.assertEqual(len({clusters[r["pdb_id"]] for r in rows}), len(rows))
        self.assertEqual(len({r["emdb_id"] for r in rows}), len(rows))
        for r in rows:
            c = by_id[r["pdb_id"]]
            self.assertEqual(r["stratum"], c["stratum"])
            self.assertGreaterEqual(float(r["cc_mask"]), 0.70)
            res = float(r["resolution_A"])
            self.assertEqual(r["stratum"], "le3.0" if res <= 3.0 else "gt3.0_le4.0")
        self.assertLessEqual(Counter(r["stratum"] for r in rows).most_common(1)[0][1], 10)
        # and it equals one run that knows every value that was asked for
        _, sel, todo = se.walk(list(by_id.values()), clusters, have)
        self.assertEqual(todo, [])
        self.assertEqual(sorted(r["pdb_id"] for r in rows), sorted(chosen(sel)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
