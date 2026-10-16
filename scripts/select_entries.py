#!/usr/bin/env python3
"""Phase 3b of the MICA 2026 benchmark: cluster the candidate pool and apply the selection rule.

The rule (docs/SPEC.md decision 5, plus the operational rules written in LOG.md):
  * One entry per MMseqs2 cluster at 30% identity. Each entry is represented by its longest chain.
  * Two strata, processed in this order with one shared list of used clusters:
    up to 3.0 A, then above 3.0 to 4.0 A.
  * In each stratum the candidates, sorted by PDB ID, are shuffled with random.Random(2026).
    The walk takes the first 10 entries that pass every filter.
  * An entry is skipped if its cluster is already used or if its EMDB map is already used.
  * An entry whose CC_mask is below 0.70 (or could not be computed) is rejected and does not use
    up its cluster. CC_mask comes from compute_cc_mask.py (Phenix map_model_cc).

Subcommands
  cluster   run MMseqs2 on the longest chain of every candidate
              -> candidate_clusters.csv, mmseqs_info.json
  select    walk the shuffled order and apply the rule
              -> selection_log.csv (every candidate, with its decision), selection_info.json,
                 cc_todo.csv (entries whose CC_mask is still needed), and, when the walk
                 finishes without a missing CC_mask, entries_2026.csv

Exit codes of select: 0 walk complete (the list is written when --entries is given); 2 input missing or
unusable; 3 CC_mask values are missing (see cc_todo.csv).

CC_mask is computed in rounds. Every round starts the walk from the beginning with all CC_mask values known so
far, so only a round that ends with exit code 0 defines the list. Earlier rounds only decide which entries to
compute next; their decisions for the second stratum are provisional, because the first stratum may still use
clusters that the second stratum has not seen yet. The result does not depend on the number of rounds.

Examples
  python select_entries.py cluster --pool data/pool --out data/selection
  python select_entries.py select  --pool data/pool --out data/selection
  python select_entries.py select  --pool data/pool --out data/selection --cc data/selection/cc_mask.csv \\
         --entries data/entries_2026.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile

SCRIPT_VERSION = "1.0"

SEED = 2026
QUOTA = 10          # entries per stratum
CC_MIN = 0.70       # decision 4
BATCH = 25          # entries per stratum whose CC_mask is requested in one round
MIN_SEQ_ID = 0.30   # decision 5
COVERAGE = 0.8      # operational choice, see LOG.md: every member must be covered 80% by its alignment
COV_MODE = 1        # to the representative, so that fragments join the cluster of a longer homologue
STRATA = ("le3.0", "gt3.0_le4.0")
CC_STATUSES = ("ok", "failed")

ENTRY_COLUMNS = [
    "emdb_id", "pdb_id", "emdb_release_date", "pdb_release_date", "resolution_A", "stratum",
    "modeled_residues", "unique_chain_seqs", "unk_fraction", "cc_mask", "cluster_id", "seed_rank",
    "max_id_to_pre2026_pdb", "cluster_size", "longest_seq_len", "n_chains", "map_size_mb", "cif_url", "map_url",
]
LOG_COLUMNS = ["stratum", "seed_rank", "pdb_id", "emdb_id", "cluster_id", "resolution_A", "modeled_residues",
               "cc_mask", "decision", "reason"]
TODO_COLUMNS = ["pdb_id", "emdb_id", "stratum", "seed_rank", "resolution_A", "cif_url", "map_url", "map_size_mb"]


# --------------------------------------------------------------------------------------
# Input and output
# --------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


def read_candidates(pool_dir):
    rows = read_csv(os.path.join(pool_dir, "pool_candidates.csv"))
    out = []
    for r in rows:
        if r.get("stratum") not in STRATA:
            raise ValueError("%s has stratum %r; expected one of %s" % (r.get("pdb_id"), r.get("stratum"), STRATA))
        out.append(r)
    ids = [r["pdb_id"] for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("pool_candidates.csv lists a PDB ID twice")
    return out


def read_longest_sequences(fasta_path, candidates):
    """The sequence of each candidate's longest entity (flag longest=1 in the FASTA header)."""
    wanted = {r["pdb_id"] for r in candidates}
    seqs, name, buf = {}, None, []

    def flush():
        if name is not None:
            head, _, rest = name.partition(" ")
            pid = head.split("_E")[0]
            if pid in wanted and "longest=1" in rest.split():
                if pid in seqs:
                    raise ValueError("two longest chains for %s" % pid)
                seqs[pid] = "".join(buf)

    with open(fasta_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                name, buf = line[1:], []
            else:
                buf.append(line.strip())
    flush()
    missing = sorted(wanted - set(seqs))
    if missing:
        raise ValueError("no longest-chain sequence for %d candidates, e.g. %s" % (len(missing), ", ".join(missing[:5])))
    return seqs


# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------
def mmseqs_version(binary):
    try:
        r = subprocess.run([binary, "version"], capture_output=True, text=True, timeout=60)
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return "unknown (%s)" % e


def run_mmseqs(seqs, binary, threads=1):
    """Cluster with MMseqs2; returns {pdb_id: representative pdb_id} and the command that was run."""
    tmp = tempfile.mkdtemp(prefix="select_mmseqs_")
    try:
        fasta = os.path.join(tmp, "longest_chains.fasta")
        with open(fasta, "w", encoding="utf-8") as f:
            for pid in sorted(seqs):  # fixed input order
                f.write(">%s\n%s\n" % (pid, seqs[pid]))
        prefix = os.path.join(tmp, "clu")
        cmd = [binary, "easy-cluster", fasta, prefix, os.path.join(tmp, "work"),
               "--min-seq-id", str(MIN_SEQ_ID), "-c", str(COVERAGE), "--cov-mode", str(COV_MODE),
               "--threads", str(threads), "-v", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("mmseqs failed (exit %d):\n%s\n%s" % (r.returncode, r.stdout[-1500:], r.stderr[-1500:]))
        rep_of = {}
        with open(prefix + "_cluster.tsv", encoding="utf-8") as f:
            for line in f:
                rep, member = line.rstrip("\n").split("\t")[:2]
                rep_of[member] = rep
        missing = set(seqs) - set(rep_of)
        if missing:
            raise RuntimeError("mmseqs left %d sequences without a cluster" % len(missing))
        return rep_of, cmd
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_cluster(args):
    candidates = read_candidates(args.pool)
    fasta = os.path.join(args.pool, "pool_candidate_sequences.fasta")
    seqs = read_longest_sequences(fasta, candidates)
    binary = args.mmseqs
    if shutil.which(binary) is None and not os.path.exists(binary):
        print("mmseqs not found (%s). Install MMseqs2 or pass --mmseqs PATH." % binary, file=sys.stderr)
        return 2
    try:
        rep_of, cmd = run_mmseqs(seqs, binary, args.threads)
    except RuntimeError as e:
        print("Clustering failed: %s" % e, file=sys.stderr)
        return 2
    sizes = {}
    for rep in rep_of.values():
        sizes[rep] = sizes.get(rep, 0) + 1
    rows = [{"pdb_id": pid, "cluster_id": rep_of[pid], "cluster_size": sizes[rep_of[pid]],
             "longest_seq_len": len(seqs[pid])} for pid in sorted(seqs)]
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "candidate_clusters.csv"), ["pdb_id", "cluster_id", "cluster_size", "longest_seq_len"], rows)
    info = {"script_version": SCRIPT_VERSION, "mmseqs_version": mmseqs_version(binary),
            "command_template": [c if not c.startswith(tempfile.gettempdir()) else "<tmp>/" + os.path.basename(c) for c in cmd],
            "min_seq_id": MIN_SEQ_ID, "coverage": COVERAGE, "cov_mode": COV_MODE, "threads": args.threads,
            "n_sequences": len(seqs), "n_clusters": len(sizes),
            "largest_clusters": sorted(sizes.values(), reverse=True)[:10],
            "pool_candidates_sha256": sha256_file(os.path.join(args.pool, "pool_candidates.csv")),
            "fasta_sha256": sha256_file(fasta), "python": platform.python_version()}
    with open(os.path.join(args.out, "mmseqs_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("%d candidates in %d clusters at %d%% identity (largest clusters: %s)"
          % (len(seqs), len(sizes), round(MIN_SEQ_ID * 100), info["largest_clusters"]))
    return 0


# --------------------------------------------------------------------------------------
# The walk
# --------------------------------------------------------------------------------------
def shuffled(ids, seed=SEED):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    return ids


def read_cc(path):
    """{pdb_id: (status, cc_mask or None, note)}; the last row of a PDB ID wins.

    status is "ok" (cc_mask is a finite number) or "failed" (Phenix ran, but no CC_mask came out).
    A transient problem (download error, timeout) is not a result and must not be in this file: the entry
    stays unknown and is computed again. Anything else in the file is an error, not a silent rejection."""
    cc = {}
    if not path:
        return cc
    for line_no, r in enumerate(read_csv(path), 2):
        pid = (r.get("pdb_id") or "").strip()
        status = (r.get("status") or "").strip().lower()
        if not pid:
            raise ValueError("%s line %d: no pdb_id" % (path, line_no))
        if status not in CC_STATUSES:
            raise ValueError("%s line %d (%s): status %r; the allowed values are %s"
                             % (path, line_no, pid, status, " and ".join(CC_STATUSES)))
        val = None
        if status == "ok":
            try:
                val = float((r.get("cc_mask") or "").strip())
            except ValueError:
                val = None
            if val is None or not math.isfinite(val):
                raise ValueError("%s line %d (%s): status ok, but cc_mask %r is not a number"
                                 % (path, line_no, pid, r.get("cc_mask")))
        cc[pid] = (status, val, r.get("note") or "")
    return cc


def walk(cands, rep_of, cc, quota=QUOTA, cc_min=CC_MIN, seed=SEED, batch=BATCH):
    """Apply the rule. Returns (log rows, selected rows, todo rows).

    A candidate whose CC_mask is unknown stops the exact walk of its stratum, because what follows depends
    on the result. The entries to evaluate next are found by a second, optimistic walk that assumes the
    unknown ones pass."""
    by_id = {r["pdb_id"]: r for r in cands}
    used_cluster, used_map = {}, {}
    log_rows, selected, todo = [], [], []
    for stratum in STRATA:
        order = shuffled([r["pdb_id"] for r in cands if r["stratum"] == stratum], seed)
        n_sel, stop_rank = 0, None
        for rank, pid in enumerate(order, 1):
            r = by_id[pid]
            row = {"stratum": stratum, "seed_rank": rank, "pdb_id": pid, "emdb_id": r["emdb_id"],
                   "cluster_id": rep_of[pid], "resolution_A": r["resolution_A"],
                   "modeled_residues": r["n_modeled_residues"], "cc_mask": "", "decision": "", "reason": ""}
            log_rows.append(row)
            if stop_rank is not None:
                row["decision"], row["reason"] = "not_evaluated", "an earlier entry still needs its CC_mask"
                continue
            if n_sel >= quota:
                row["decision"], row["reason"] = "not_reached", "%d entries already selected in this stratum" % quota
                continue
            if rep_of[pid] in used_cluster:
                row["decision"], row["reason"] = "skip_cluster_used", "cluster %s used by %s" % (rep_of[pid], used_cluster[rep_of[pid]])
                continue
            if r["emdb_id"] in used_map:
                row["decision"], row["reason"] = "skip_map_used", "%s used by %s" % (r["emdb_id"], used_map[r["emdb_id"]])
                continue
            status, val, note = cc.get(pid, ("", None, ""))
            if not status:
                row["decision"], row["reason"] = "needs_cc", "CC_mask not computed yet"
                stop_rank = rank
                continue
            if status != "ok":
                row["decision"], row["reason"] = "reject_cc_failed", "CC_mask could not be computed: %s" % note
                continue
            row["cc_mask"] = "%.4f" % val
            if val < cc_min:
                row["decision"], row["reason"] = "reject_cc_mask", "CC_mask %.4f below %.2f" % (val, cc_min)
                continue
            row["decision"] = "selected"
            n_sel += 1
            row["reason"] = "number %d of %d in the stratum" % (n_sel, quota)
            used_cluster[rep_of[pid]] = pid
            used_map[r["emdb_id"]] = pid
            selected.append({"row": r, "rank": rank, "cc": val, "cluster": rep_of[pid]})
        if stop_rank is not None:
            todo.extend(optimistic_todo(stratum, order, by_id, rep_of, cc, used_cluster, used_map,
                                        quota - n_sel, stop_rank, batch, cc_min))
    return log_rows, selected, todo


def optimistic_todo(stratum, order, by_id, rep_of, cc, used_cluster, used_map, need, start_rank, batch, cc_min):
    """Entries to evaluate next. From the first unknown entry on, assume that every unknown CC_mask passes and
    that every known one keeps its result, and collect the unknown entries until enough selections are in
    sight: two and a half times what is still needed, at least 5 and at most `batch`."""
    want = min(batch, max(5, math.ceil(2.5 * need)))
    used_c, used_m = dict(used_cluster), dict(used_map)
    out, n_assumed = [], 0
    for rank, pid in enumerate(order, 1):
        if rank < start_rank:
            continue
        if n_assumed >= want:
            break
        r = by_id[pid]
        if rep_of[pid] in used_c or r["emdb_id"] in used_m:
            continue
        known = cc.get(pid)
        if known is not None:
            status, val, _ = known
            if status != "ok" or val < cc_min:
                continue
        else:
            out.append({"pdb_id": pid, "emdb_id": r["emdb_id"], "stratum": stratum, "seed_rank": rank,
                        "resolution_A": r["resolution_A"], "cif_url": r["cif_url"], "map_url": r["map_url"],
                        "map_size_mb": mb(r.get("map_size_kb"))})
        n_assumed += 1
        used_c[rep_of[pid]] = pid
        used_m[r["emdb_id"]] = pid
    return out


def mb(kb):
    try:
        return round(float(kb) / 1024.0, 1)
    except (TypeError, ValueError):
        return ""


def entry_rows(selected, sizes):
    rows = []
    for s in selected:
        r = s["row"]
        rows.append({
            "emdb_id": r["emdb_id"], "pdb_id": r["pdb_id"], "emdb_release_date": r.get("emdb_map_release"),
            "pdb_release_date": r.get("release_date_pdb"), "resolution_A": r["resolution_A"], "stratum": r["stratum"],
            "modeled_residues": r["n_modeled_residues"], "unique_chain_seqs": r["n_unique_seq"],
            "unk_fraction": r["unk_fraction"], "cc_mask": "%.4f" % s["cc"], "cluster_id": s["cluster"],
            "seed_rank": s["rank"], "max_id_to_pre2026_pdb": "", "cluster_size": sizes.get(s["cluster"], ""),
            "longest_seq_len": r["longest_seq_len"], "n_chains": r["n_chains"], "map_size_mb": mb(r.get("map_size_kb")),
            "cif_url": r["cif_url"], "map_url": r["map_url"]})
    return rows


def cmd_select(args):
    cands = read_candidates(args.pool)
    clusters_path = args.clusters or os.path.join(args.out, "candidate_clusters.csv")
    if not os.path.exists(clusters_path):
        print("%s not found: run the cluster subcommand first." % clusters_path, file=sys.stderr)
        return 2
    cl_rows = read_csv(clusters_path)
    rep_of = {r["pdb_id"]: r["cluster_id"] for r in cl_rows}
    sizes = {}
    for rep in rep_of.values():
        sizes[rep] = sizes.get(rep, 0) + 1
    missing = [r["pdb_id"] for r in cands if r["pdb_id"] not in rep_of]
    if missing:
        print("%d candidates have no cluster (e.g. %s). Run the cluster subcommand again." % (len(missing), missing[:3]), file=sys.stderr)
        return 2
    cc = read_cc(args.cc)
    log_rows, selected, todo = walk(cands, rep_of, cc, args.quota, args.cc_min, args.seed, args.batch)
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "selection_log.csv"), LOG_COLUMNS, log_rows)
    write_csv(os.path.join(args.out, "cc_todo.csv"), TODO_COLUMNS, todo)
    final = not todo
    counts = {}
    for r in log_rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    per_stratum = {s: sum(1 for x in selected if x["row"]["stratum"] == s) for s in STRATA}
    info = {"script_version": SCRIPT_VERSION, "python": platform.python_version(), "seed": args.seed,
            "quota_per_stratum": args.quota, "cc_min": args.cc_min, "batch": args.batch, "final": final,
            "decisions": counts, "selected_per_stratum": per_stratum, "n_cc_requested": len(todo),
            "pool_candidates_sha256": sha256_file(os.path.join(args.pool, "pool_candidates.csv")),
            "clusters_sha256": sha256_file(clusters_path),
            "cc_file_sha256": sha256_file(args.cc) if args.cc else None,
            "first_in_walk": {s: [r["pdb_id"] for r in log_rows if r["stratum"] == s][:5] for s in STRATA}}
    with open(os.path.join(args.out, "selection_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("decisions: %s" % counts)
    if not final:
        print("CC_mask is needed for %d entries (cc_todo.csv). Compute it, add it to the cc_mask file, and run select again." % len(todo))
        return 3
    total = len(selected)
    if args.entries:
        write_csv(args.entries, ENTRY_COLUMNS, entry_rows(selected, sizes))
        print("wrote %s" % args.entries)
    print("FINAL: %d entries selected (%s)" % (total, ", ".join("%s: %d" % kv for kv in per_stratum.items())))
    if total < 12:
        print("WARNING: below the floor of 12 entries (decision 5); the scope-cut order in decision 12 applies.")
    elif total < 20:
        print("Note: below the target of 20 entries.")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cluster", help="run MMseqs2 on the longest chains")
    c.add_argument("--pool", required=True, help="folder with pool_candidates.csv and pool_candidate_sequences.fasta")
    c.add_argument("--out", required=True)
    c.add_argument("--mmseqs", default="mmseqs")
    c.add_argument("--threads", type=int, default=1, help="keep 1 for a reproducible clustering")
    s = sub.add_parser("select", help="apply the selection rule")
    s.add_argument("--pool", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--clusters", default=None, help="default: OUT/candidate_clusters.csv")
    s.add_argument("--cc", default=None, help="cc_mask.csv from compute_cc_mask.py")
    s.add_argument("--entries", default=None, help="where to write the final list (only if the walk is complete)")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--quota", type=int, default=QUOTA)
    s.add_argument("--cc-min", type=float, default=CC_MIN)
    s.add_argument("--batch", type=int, default=BATCH, help="entries per stratum whose CC_mask is requested in one round")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        return cmd_cluster(args) if args.cmd == "cluster" else cmd_select(args)
    except (ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
