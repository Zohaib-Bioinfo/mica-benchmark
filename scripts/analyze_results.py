#!/usr/bin/env python3
"""Statistics and tables of the benchmark (SPEC decisions 7, 8, 9 and 11; scoring protocol v1.0 of LOG.md).

Every number of the report comes from this script and the figure script, from three kinds of input that the notebooks write (nothing is typed in by hand):

  --entries  data/entries_2026.csv        the frozen list of entries
  --status   method_status.csv ...        status tables of the baseline notebooks (gpu_batch1, gpu_batch2: they have a column "method") and
                                          mica_status.csv of the MICA notebooks (mica_cpu_a/b/c, mica_gpu: no column "method"); give them oldest first
  --scores   scores_long.csv ...          from the scoring notebook (a later file replaces the rows of an earlier one with the same entry, method and model file)
  --out      results/                     the tables below

Outcome of one method on one entry (decision 11):
  model       the run finished (status ok) and its primary model file was scored
  failed      the program stopped with an error or at its time limit (status failed): a failure, it stays in every table
  unscorable  status ok, but the primary model file is missing, cannot be read or cannot be aligned: also a failure
  not_run     no result at all (killed by the system while another program ran, not started, cut by the end of a session, waiting for a GPU): neither a success
              nor a failure of the method; counted separately (completion rate "of the list" and "of the attempted")
Percentages are in percent (0 to 100: ca_match, seq_match, ca_found, ca_quality, seq_identity), TM-score and coverage are fractions, lengths are residues, RMSD is in angstrom.

Statistics (decision 8), per group (pooled, each resolution stratum) and per metric, for every method against MICA on the entries where both have a model; the difference is
always method minus MICA: paired two-sided Wilcoxon signed-rank test (zero differences dropped, exact p-value when there are no ties and at most 50 pairs, else the normal
approximation; scipy 'auto'); paired t-test; percentile bootstrap of the mean and the median difference over entries (10,000 resamples, seed 2026 combined with a fixed hash of the
comparison, so a table does not change when another row is added); Holm correction across the method pairs of the same group, metric and test (the family size is the number of
methods compared, also when one of them has no pairs); matrix-pairs rank-biserial correlation and Cohen's dz as effect sizes. Only TM-score carries the confirmatory tests of
decision 8; every other metric is descriptive and labelled so in the report.

Tables written to --out: metrics_per_entry.csv, entry_outcomes.csv, completion.csv, summary_by_method.csv, paired_vs_mica.csv, sensitivity_failures_as_zero.csv, variants.csv,
correlations.csv, runtime.csv, status_merged.csv, qa.txt, provenance.json.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026
N_BOOT = 10000
REFERENCE = "mica"
FINAL_DOCK_PROFILE = "pass3"                               # the docking set-up of the MICA results that count (LOG.md 2026-10-26); rows of pass 1 and 2 stay in `history`
METHODS = ["mica", "modelangelo", "emodelx", "emprot", "e3"]
LABELS = {"mica": "MICA", "modelangelo": "ModelAngelo", "emodelx": "EModelX(+AF)", "emprot": "EMProt", "e3": "E3-CryoFold"}
RESULT_STATUSES = ("ok", "failed")
PRIMARY_TEST_METRIC = "tm_score"
NOVELTY_MAX = 0.30                                         # decision 9: the subset at at most 30% identity to a pre-2026 PDB chain

# column, label, direction of "better" (None: no direction), used in the main tables
METRICS = [("tm_score", "TM-score", "higher"),
           ("aligned_ca_length", "Aligned C-alpha length", "higher"),
           ("seq_identity", "Sequence identity of aligned residues (%)", "higher"),
           ("ca_rmsd", "C-alpha RMSD of aligned residues (A)", "lower"),
           ("ca_match", "C-alpha match (%)", "higher"),
           ("seq_match", "Sequence match (%)", "higher"),
           ("ca_quality", "C-alpha quality score (%)", "higher"),
           ("coverage", "Residue coverage", None)]
# computed and tested as well: the sensitivity variants of the same quantities
EXTRA_METRICS = [("tm_score_fns", "TM-score, fragment-agnostic", "higher"),
                 ("ca_match_ff", "C-alpha match, model in its own frame (%)", "higher"),
                 ("ca_quality_ff", "C-alpha quality score, model in its own frame (%)", "higher"),
                 ("ca_found", "Deposited residues found (%)", "higher"),
                 ("own_ca_match", "C-alpha match, own count (%)", "higher")]
ALL_METRICS = METRICS + EXTRA_METRICS
DIRECTION = {c: d for c, _, d in ALL_METRICS}
METRIC_LABEL = {c: l for c, l, _ in ALL_METRICS}

SCORE_TEXT = {"entry", "method", "model_kind", "primary", "model_file", "source", "status", "note", "cc_status", "cc_status_ff"}
PERCENT_FROM_FRACTION = ("seq_identity", "seq_identity_fns")
SPEC_COLUMNS = ["pdb_id", "emdb_id", "method", "status", "runtime_min", "gpu", "tm_score", "aligned_ca_length", "ca_rmsd", "seq_identity", "ca_match", "seq_match",
                "ca_quality_score", "residue_coverage"]


# ---------------------------------------------------------------------------------------------------------------------------------- reading

def read_table(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def to_num(series):
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_entries(path):
    df = read_table(path)
    df["entry"] = df["pdb_id"].str.strip().str.upper()
    for c in ("resolution_A", "modeled_residues", "max_id_to_pre2026_pdb", "seed_rank", "n_chains"):
        if c in df.columns:
            df[c] = to_num(df[c])
    return df.reset_index(drop=True)


def load_status(paths):
    """The rows of all status tables in the order given (oldest first). A table with a column "method" is a baseline table; one without is a MICA table."""
    frames = []
    for k, path in enumerate(paths):
        df = read_table(path)
        if "method" not in df.columns:
            df["method"] = REFERENCE
        for c in ("failed_step", "note", "started_utc", "seconds", "gpu", "device", "p1_seconds", "dock_seconds", "p3_seconds", "attempt", "peak_gpu_mb", "min_ram_gb",
                  "other_gpu_busy", "dock_profile"):
            if c not in df.columns:
                df[c] = ""
        df["entry"] = df["entry"].str.strip().str.upper()
        df["method"] = df["method"].str.strip().str.lower()
        df["source_file"] = os.path.basename(os.path.dirname(os.path.abspath(path))) + "/" + os.path.basename(path)
        df["file_order"] = k
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["method", "entry", "status"])
    out = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    out["_order"] = np.arange(len(out))
    return out


def load_scores(paths):
    frames = []
    for path in paths:
        df = read_table(path)
        for c in df.columns:
            if c not in SCORE_TEXT:
                df[c] = to_num(df[c])
        for c in PERCENT_FROM_FRACTION:
            if c in df.columns:
                df[c] = df[c] * 100.0
        df["primary"] = df["primary"].str.strip().str.lower().isin(("true", "1", "yes"))
        df["entry"] = df["entry"].str.strip().str.upper()
        df["method"] = df["method"].str.strip().str.lower()
        df["scores_file"] = os.path.basename(os.path.dirname(os.path.abspath(path))) + "/" + os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["entry", "method", "model_kind", "primary", "status"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["entry", "method", "model_file"], keep="last").reset_index(drop=True)      # a later scoring run replaces an earlier one
    if "ca_quality" not in out.columns:
        out["ca_quality"] = np.nan
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- status of the runs

def _latest(g, column):
    """The latest non-empty value of a column of one entry (rows in the order of the input)."""
    values = [v for v in g.sort_values("_order")[column] if str(v).strip() != ""]
    return values[-1] if values else ""


def merge_status(frame):
    """One row per method and entry: the latest result (ok or failed) if there is one, else the latest row of any kind. `history` lists every status seen; `conflict` marks an entry
    that has results of different kinds (for example failed in one session and ok in a later one: both are reported, the later one counts).
    MICA: once an entry has rows of the final docking set-up (`dock_profile` = FINAL_DOCK_PROFILE) only those rows decide; a failure in pass 1 or 2 (docking that ran into the time
    limit) is part of the history but is not the outcome of the final set-up, so an entry that waits for the GPU after pass 3 is *not run*, not *failed* (LOG.md 2026-10-27)."""
    rows = []
    for (method, entry), g in frame.groupby(["method", "entry"], sort=False):
        g = g.sort_values("_order", kind="stable")
        current = g
        if method == REFERENCE and "dock_profile" in g.columns:
            final = g[g["dock_profile"] == FINAL_DOCK_PROFILE]
            if len(final):
                current = final
        results = current[current["status"].isin(RESULT_STATUSES)]
        if len(results):
            chosen = results.sort_values(["started_utc", "_order"], kind="stable").iloc[-1]
        else:
            chosen = current.iloc[-1]
        row = {"method": method, "entry": entry, "status_raw": chosen["status"], "failed_step": chosen["failed_step"], "note": chosen["note"], "is_result": chosen["status"] in RESULT_STATUSES,
               "n_rows": len(g), "n_earlier_setup_rows": len(g) - len(current), "history": "; ".join(g["status"]), "conflict": len(set(results["status"])) > 1,
               "source_file": chosen["source_file"], "started_utc": chosen["started_utc"], "gpu": chosen["gpu"] or chosen["device"], "attempt": chosen["attempt"]}
        for c in ("peak_gpu_mb", "min_ram_gb", "other_gpu_busy"):
            row[c] = chosen[c]
        if method == REFERENCE:
            stages = [_latest(current, c) for c in ("p1_seconds", "dock_seconds", "p3_seconds")]
            nums = [float(s) for s in stages if s != ""]
            row["runtime_s"] = sum(nums) if nums else (float(chosen["seconds"]) if str(chosen["seconds"]).strip() != "" else math.nan)
            row["runtime_note"] = "sum of the program 1, docking and program 3 times of the sessions" if nums else "seconds of the session"
        else:
            row["runtime_s"] = float(chosen["seconds"]) if str(chosen["seconds"]).strip() != "" and chosen["status"] in RESULT_STATUSES else math.nan
            row["runtime_note"] = "wall clock of the run (the other GPU was busy: %s)" % (chosen["other_gpu_busy"] or "unknown")
        rows.append(row)
    return pd.DataFrame(rows)


def entry_outcomes(entries, merged, scores):
    """One row per method and entry of the frozen list: outcome, reason, and the primary score row."""
    primary = scores[scores["primary"]] if len(scores) else scores
    rows = []
    for method in METHODS:
        for _, e in entries.iterrows():
            key = e["entry"]
            st = merged[(merged["method"] == method) & (merged["entry"] == key)]
            st = st.iloc[0] if len(st) else None
            sc = primary[(primary["method"] == method) & (primary["entry"] == key)] if len(primary) else primary
            sc_ok = sc[sc["status"] == "ok"] if len(sc) else sc
            row = {"entry": key, "method": method, "outcome": "not_run", "reason": "no status row", "status_raw": "", "has_score_row": bool(len(sc)), "runtime_s": math.nan,
                   "gpu": "", "history": ""}
            if st is not None:
                row.update(status_raw=st["status_raw"], runtime_s=st["runtime_s"], gpu=st["gpu"], history=st["history"])
                if st["status_raw"] == "failed":
                    row.update(outcome="failed", reason=(st["failed_step"] + ": " if st["failed_step"] else "") + st["note"])
                elif st["status_raw"] == "ok":
                    if len(sc_ok):
                        row.update(outcome="model", reason="")
                    elif len(sc):
                        row.update(outcome="unscorable", reason="primary model: %s %s" % (sc.iloc[0]["status"], sc.iloc[0].get("note", "")))
                    else:
                        row.update(outcome="unscorable", reason="status ok, but no primary model file was scored")
                else:
                    row.update(outcome="not_run", reason=(st["status_raw"] + (": " + st["note"] if st["note"] else "")))
            rows.append(row)
    out = pd.DataFrame(rows)
    return out


def metric_table(entries, outcomes, scores):
    """metrics_per_entry: one row per method and entry (failures and missing runs with empty metrics), the entry's stratum, resolution, novelty."""
    primary = scores[scores["primary"] & (scores["status"] == "ok")] if len(scores) else scores
    primary = primary.drop_duplicates(subset=["entry", "method"], keep="first")
    cols = [c for c, _, _ in ALL_METRICS] + ["pred_residues", "native_residues", "n_chains_pred", "n_chains_native", "n_chains_paired", "model_kind", "model_file"]
    cols = [c for c in cols if c in primary.columns]
    m = outcomes.merge(primary[["entry", "method"] + cols], on=["entry", "method"], how="left")
    e = entries[["entry", "emdb_id", "stratum", "resolution_A", "modeled_residues", "max_id_to_pre2026_pdb"]]
    m = m.merge(e, on="entry", how="left")
    for c, _, _ in ALL_METRICS:                                         # without scores the columns still exist (empty): every table keeps its shape
        if c not in m.columns:
            m[c] = np.nan
    order = {k: i for i, k in enumerate(METHODS)}
    pos = {k: i for i, k in enumerate(entries["entry"])}
    m["_m"], m["_e"] = m["method"].map(order), m["entry"].map(pos)
    m = m.sort_values(["_e", "_m"]).drop(columns=["_m", "_e"]).reset_index(drop=True)
    for c in cols:
        if c in m.columns and c not in ("model_kind", "model_file"):
            m.loc[m["outcome"] != "model", c] = np.nan
    return m


# ---------------------------------------------------------------------------------------------------------------------------------- statistics

def holm_adjust(p, m=None):
    """Holm's step-down adjustment; NaN p-values do not take part, `m` is the size of the family (default: the number of p-values)."""
    p = np.asarray(p, dtype=float)
    m = len(p) if m is None else m
    adj = np.full(len(p), np.nan)
    running = 0.0
    valid = [i for i in np.argsort(p, kind="stable") if not np.isnan(p[i])]
    for rank, i in enumerate(valid):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def seed_for(key, seed=SEED):
    return [seed, zlib.crc32(key.encode("utf-8"))]


def bootstrap_ci(values, statistic, key, seed=SEED, n_boot=N_BOOT, level=95.0):
    """Percentile bootstrap over entries; `statistic` is np.mean or np.median."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed_for(key, seed))
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot = statistic(v[idx], axis=1)
    lo, hi = np.percentile(boot, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


def describe(values, key, seed=SEED, n_boot=N_BOOT):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    out = {"n": len(v)}
    if not len(v):
        return dict(out, mean=math.nan, sd=math.nan, median=math.nan, q1=math.nan, q3=math.nan, min=math.nan, max=math.nan, mean_ci_low=math.nan, mean_ci_high=math.nan)
    lo, hi = bootstrap_ci(v, np.mean, key, seed, n_boot)
    return dict(out, mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else math.nan, median=float(np.median(v)), q1=float(np.percentile(v, 25)),
                q3=float(np.percentile(v, 75)), min=float(v.min()), max=float(v.max()), mean_ci_low=lo, mean_ci_high=hi)


def paired_stats(x, y, key, better=None, seed=SEED, n_boot=N_BOOT):
    """x = the method, y = MICA, on the same entries; the difference is x - y. Pairs with a missing value are dropped."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    d = x - y
    n = len(d)
    out = {"n": n, "mean_method": float(x.mean()) if n else math.nan, "mean_mica": float(y.mean()) if n else math.nan, "mean_diff": float(d.mean()) if n else math.nan,
           "median_diff": float(np.median(d)) if n else math.nan}
    out["diff_ci_low"], out["diff_ci_high"] = bootstrap_ci(d, np.mean, key + "|mean", seed, n_boot)
    out["median_diff_ci_low"], out["median_diff_ci_high"] = bootstrap_ci(d, np.median, key + "|median", seed, n_boot)
    nz = d[d != 0]
    out["n_nonzero"] = len(nz)
    if n == 0 or len(nz) == 0:
        out.update(wilcoxon_stat=0.0 if n else math.nan, wilcoxon_p=1.0 if n else math.nan, rank_biserial=0.0 if n else math.nan)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sample size too small for normal approximation")      # the p-value is returned all the same; n is in the table
            res = stats.wilcoxon(nz, alternative="two-sided")
        ranks = stats.rankdata(np.abs(nz))
        rp, rm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        out.update(wilcoxon_stat=float(res.statistic), wilcoxon_p=float(res.pvalue), rank_biserial=float((rp - rm) / (rp + rm)))
    if n >= 2 and np.ptp(d) > 1e-9 * np.abs(d).max():                       # not when the differences are all the same (to rounding): no variance, no test
        t = stats.ttest_rel(x, y)
        out.update(ttest_t=float(t.statistic), ttest_p=float(t.pvalue), cohens_dz=float(d.mean() / d.std(ddof=1)))
    else:
        out.update(ttest_t=math.nan, ttest_p=math.nan, cohens_dz=math.nan)
    out["n_method_higher"] = int((d > 0).sum())
    out["n_method_lower"] = int((d < 0).sum())
    out["n_equal"] = int((d == 0).sum())
    if better == "higher":
        out["n_method_better"], out["n_method_worse"] = out["n_method_higher"], out["n_method_lower"]
    elif better == "lower":
        out["n_method_better"], out["n_method_worse"] = out["n_method_lower"], out["n_method_higher"]
    else:
        out["n_method_better"] = out["n_method_worse"] = math.nan
    return out


def groups_of(entries):
    """pooled, one group per stratum (in the order of the list), and the novelty subset of decision 9 (sensitivity)."""
    g = {"pooled": list(entries["entry"])}
    for s in dict.fromkeys(entries["stratum"]):
        g[s] = list(entries.loc[entries["stratum"] == s, "entry"])
    g["novel_le30"] = list(entries.loc[entries["max_id_to_pre2026_pdb"] <= NOVELTY_MAX, "entry"])
    return g


def dock_mode_groups(frame):
    """MICA entries by how far the docking got (`dock_mode` of mica_status.csv, the last non-empty value of an entry): groups for the sensitivity analysis of 2026-10-26 (LOG.md).
    `mica_docked`: at least one domain docked (full or partial); `mica_nodock`: program 3 ran without docked AlphaFold input."""
    if frame is None or not len(frame) or "dock_mode" not in frame.columns:
        return {}
    f = frame[(frame["method"] == REFERENCE) & (frame["dock_mode"].astype(str) != "")]
    if not len(f):
        return {}
    mode = f.sort_values("_order").groupby("entry")["dock_mode"].last()
    out = {}
    docked = sorted(e for e, v in mode.items() if v in ("full", "partial"))
    none = sorted(e for e, v in mode.items() if v == "none")
    if docked:
        out["mica_docked"] = docked
    if none:
        out["mica_nodock"] = none
    return out


def value_table(metrics, column):
    """entry x method matrix of one column of metric_table."""
    return metrics.pivot(index="entry", columns="method", values=column)


def summary_table(metrics, groups, seed=SEED, n_boot=N_BOOT):
    rows = []
    for gname, members in groups.items():
        sub = metrics[metrics["entry"].isin(members)]
        for method in METHODS:
            s = sub[sub["method"] == method]
            counts = s["outcome"].value_counts()
            for column, label, better in ALL_METRICS:
                if column not in s.columns:
                    continue
                d = describe(s.loc[s["outcome"] == "model", column].to_numpy(dtype=float), "summary|%s|%s|%s" % (gname, method, column), seed, n_boot)
                rows.append(dict(group=gname, method=method, metric=column, metric_label=label, n_entries=len(s), n_model=int(counts.get("model", 0)), **d))
    return pd.DataFrame(rows)


def paired_table(metrics, groups, seed=SEED, n_boot=N_BOOT):
    baselines = [m for m in METHODS if m != REFERENCE]
    rows = []
    for gname, members in groups.items():
        sub = metrics[metrics["entry"].isin(members)]
        for column, label, better in ALL_METRICS:
            if column not in sub.columns:
                continue
            table = value_table(sub, column).reindex(members)
            if REFERENCE not in table.columns:
                continue
            fam = []
            for method in baselines:
                if method not in table.columns:
                    r = paired_stats([], [], "x", better, seed, n_boot)
                else:
                    r = paired_stats(table[method].to_numpy(dtype=float), table[REFERENCE].to_numpy(dtype=float), "paired|%s|%s|%s" % (gname, method, column), better, seed, n_boot)
                fam.append(dict(group=gname, metric=column, metric_label=label, better=better or "", method=method, **r))
            for test in ("wilcoxon_p", "ttest_p"):
                adj = holm_adjust([f[test] for f in fam], m=len(baselines))
                for f, a in zip(fam, adj):
                    f[test.replace("_p", "_p_holm")] = float(a)
            rows += fam
    return pd.DataFrame(rows)


def failures_as_zero(metrics, groups, seed=SEED, n_boot=N_BOOT):
    """Decision 11: failures score 0 on every metric. 'strict': failed and unscorable runs become 0, entries that were not run stay missing (not the method's failure);
    'all_missing': an entry without a model counts 0 whatever the reason."""
    baselines = [m for m in METHODS if m != REFERENCE]
    rows = []
    for policy in ("strict", "all_missing"):
        m = metrics.copy()
        for column in ("tm_score", "ca_match", "seq_match", "ca_quality", "coverage", "tm_score_fns"):
            if column not in m.columns:
                continue
            zero = m["outcome"].isin(("failed", "unscorable")) if policy == "strict" else (m["outcome"] != "model")
            m.loc[zero, column] = 0.0
            if policy == "strict":
                m.loc[m["outcome"] == "not_run", column] = np.nan
        for gname, members in groups.items():
            sub = m[m["entry"].isin(members)]
            for column in ("tm_score", "ca_quality", "tm_score_fns"):
                if column not in sub.columns:
                    continue
                table = value_table(sub, column).reindex(members)
                fam = []
                for method in [REFERENCE] + baselines:
                    d = describe(table[method].to_numpy(dtype=float), "zero|%s|%s|%s|%s" % (policy, gname, method, column), seed, n_boot) if method in table.columns else {"n": 0}
                    fam.append(dict(policy=policy, group=gname, metric=column, method=method, role="mean_over_entries", **d))
                for method in baselines:
                    if method in table.columns and REFERENCE in table.columns:
                        r = paired_stats(table[method].to_numpy(dtype=float), table[REFERENCE].to_numpy(dtype=float), "zeropair|%s|%s|%s|%s" % (policy, gname, method, column),
                                         "higher", seed, n_boot)
                        fam.append(dict(policy=policy, group=gname, metric=column, method=method, role="paired_vs_mica", **r))
                ps = [f for f in fam if f["role"] == "paired_vs_mica"]
                adj = holm_adjust([f["wilcoxon_p"] for f in ps], m=len(baselines))
                for f, a in zip(ps, adj):
                    f["wilcoxon_p_holm"] = float(a)
                rows += fam
    return pd.DataFrame(rows)


def variants_table(scores):
    """Every kind of model file that was scored (primary and secondary), as a descriptive table: the choice of the primary file was made before any result (LOG 2026-10-24)."""
    s = scores[scores["status"] == "ok"]
    rows = []
    for (method, kind), g in s.groupby(["method", "model_kind"], sort=False):
        row = {"method": method, "model_kind": kind, "primary": bool(g["primary"].iloc[0]), "n_models": len(g)}
        for column, _, _ in ALL_METRICS:
            if column in g.columns:
                row[column + "_mean"] = float(g[column].mean()) if g[column].notna().any() else math.nan
                row[column + "_median"] = float(g[column].median()) if g[column].notna().any() else math.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        order = {k: i for i, k in enumerate(METHODS)}
        out = out.assign(_m=out["method"].map(order)).sort_values(["_m", "primary"], ascending=[True, False]).drop(columns="_m")
    return out.reset_index(drop=True)


def correlations(metrics):
    """Exploratory: Spearman correlation of TM-score with resolution, size of the deposited model and identity to the older PDB, per method."""
    rows = []
    for method in METHODS:
        s = metrics[(metrics["method"] == method) & (metrics["outcome"] == "model")]
        for x, label in (("resolution_A", "resolution (A)"), ("modeled_residues", "modeled residues"), ("max_id_to_pre2026_pdb", "highest identity to a pre-2026 PDB chain")):
            for y in ("tm_score", "ca_quality"):
                ok = s[[x, y]].dropna()
                if len(ok) >= 5 and ok[x].nunique() > 1 and ok[y].nunique() > 1:
                    r = stats.spearmanr(ok[x], ok[y])
                    rows.append({"method": method, "x": x, "x_label": label, "y": y, "n": len(ok), "spearman_rho": float(r.statistic), "p": float(r.pvalue)})
                else:
                    rows.append({"method": method, "x": x, "x_label": label, "y": y, "n": len(ok), "spearman_rho": math.nan, "p": math.nan})
    return pd.DataFrame(rows)


def completion_table(outcomes, groups):
    rows = []
    for gname, members in groups.items():
        sub = outcomes[outcomes["entry"].isin(members)]
        for method in METHODS:
            s = sub[sub["method"] == method]
            c = s["outcome"].value_counts()
            n, model, failed, unscor, notrun = len(s), int(c.get("model", 0)), int(c.get("failed", 0)), int(c.get("unscorable", 0)), int(c.get("not_run", 0))
            attempted = model + failed + unscor
            rows.append({"group": gname, "method": method, "n_entries": n, "n_model": model, "n_failed": failed, "n_unscorable": unscor, "n_not_run": notrun,
                         "completion_of_list": model / n if n else math.nan, "completion_of_attempted": model / attempted if attempted else math.nan})
    return pd.DataFrame(rows)


def runtime_table(outcomes):
    rows = []
    for method in METHODS:
        s = outcomes[(outcomes["method"] == method) & outcomes["outcome"].isin(("model", "failed", "unscorable"))]
        r = s["runtime_s"].dropna() / 60.0
        rows.append({"method": method, "n_runs": len(r), "median_min": float(r.median()) if len(r) else math.nan, "min_min": float(r.min()) if len(r) else math.nan,
                     "max_min": float(r.max()) if len(r) else math.nan, "total_h": float(r.sum() / 60.0) if len(r) else math.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------------------- checks

def qa_checks(entries, frame, merged, scores, outcomes):
    lines, bad = [], 0

    def note(ok, text):
        nonlocal bad
        lines.append(("OK    " if ok else "CHECK ") + text)
        bad += 0 if ok else 1

    known = set(entries["entry"])
    note(len(known) == len(entries), "%d entries in the frozen list, all different" % len(entries))
    extra = sorted(set(frame["entry"]) - known) if len(frame) else []
    note(not extra, "status tables: every entry is in the frozen list" + ("" if not extra else " (not in the list: %s)" % ", ".join(extra)))
    odd = sorted(set(frame["method"]) - set(METHODS)) if len(frame) else []
    note(not odd, "status tables: every method is known" + ("" if not odd else " (unknown: %s)" % ", ".join(odd)))
    if len(scores):
        extra = sorted(set(scores["entry"]) - known)
        note(not extra, "scores: every entry is in the frozen list" + ("" if not extra else " (not in the list: %s)" % ", ".join(extra)))
        odd = sorted(set(scores["method"]) - set(METHODS))
        note(not odd, "scores: every method is known" + ("" if not odd else " (unknown: %s)" % ", ".join(odd)))
        dup = scores[scores["primary"] & (scores["status"] == "ok")].groupby(["entry", "method"]).size()
        dup = dup[dup > 1]
        note(len(dup) == 0, "scores: one primary model per method and entry" + ("" if not len(dup) else " (several: %s)" % ", ".join("%s/%s" % k for k in dup.index)))
    if len(merged):
        conflict = merged[merged["conflict"]]
        note(len(conflict) == 0, "status: no entry has both an ok and a failed result" + ("" if not len(conflict) else " (%s)" % ", ".join("%s/%s" % (r.method, r.entry) for r in conflict.itertuples())))
        miss = [(m, e) for m in METHODS for e in entries["entry"] if not len(merged[(merged["method"] == m) & (merged["entry"] == e)])]
        note(not miss, "status: every method has a row for every entry" + ("" if not miss else " (missing: %s)" % ", ".join("%s/%s" % x for x in miss[:12]) + (" ..." if len(miss) > 12 else "")))
    if len(scores):
        bad_state = outcomes[(outcomes["outcome"] != "model") & outcomes["has_score_row"]]
        okless = bad_state[bad_state["status_raw"] != "ok"]
        note(len(okless) == 0, "scores: no model of a run that is not ok" + ("" if not len(okless) else " (%s)" % ", ".join("%s/%s" % (r.method, r.entry) for r in okless.itertuples())))
        unsc = outcomes[outcomes["outcome"] == "unscorable"]
        note(len(unsc) == 0, "outcomes: no run that is ok but unscorable" + ("" if not len(unsc) else " (%s)" % ", ".join("%s/%s" % (r.method, r.entry) for r in unsc.itertuples())))
    for method in METHODS:
        c = outcomes[outcomes["method"] == method]["outcome"].value_counts()
        lines.append("INFO  %-12s model %2d, failed %2d, unscorable %2d, not run %2d" % (LABELS[method], c.get("model", 0), c.get("failed", 0), c.get("unscorable", 0), c.get("not_run", 0)))
    lines.append("")
    lines.append("%d point(s) to look at" % bad if bad else "all checks passed")
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------- driver

def spec_frame(metrics):
    """metrics_per_entry.csv: the columns of SPEC decision 7 / the committed header first, then the rest."""
    out = pd.DataFrame({"pdb_id": metrics["entry"], "emdb_id": metrics["emdb_id"], "method": metrics["method"], "status": metrics["outcome"],
                        "runtime_min": (metrics["runtime_s"] / 60.0).round(2), "gpu": metrics["gpu"], "tm_score": metrics.get("tm_score"),
                        "aligned_ca_length": metrics.get("aligned_ca_length"), "ca_rmsd": metrics.get("ca_rmsd"), "seq_identity": metrics.get("seq_identity"),
                        "ca_match": metrics.get("ca_match"), "seq_match": metrics.get("seq_match"), "ca_quality_score": metrics.get("ca_quality"),
                        "residue_coverage": metrics.get("coverage")})
    for c in ("stratum", "resolution_A", "reason", "model_kind", "model_file", "tm_score_fns", "ca_match_ff", "ca_quality_ff", "ca_found", "own_ca_match", "pred_residues",
              "native_residues", "n_chains_pred", "n_chains_native", "n_chains_paired", "max_id_to_pre2026_pdb"):
        if c in metrics.columns:
            out[c] = metrics[c]
    return out


def analyse(entries, frame, scores, seed=SEED, n_boot=N_BOOT, log=print):
    merged = merge_status(frame) if len(frame) else pd.DataFrame(columns=["method", "entry", "status_raw"])
    if not len(merged):
        merged = pd.DataFrame({c: [] for c in ("method", "entry", "status_raw", "failed_step", "note", "is_result", "n_rows", "n_earlier_setup_rows", "history", "conflict", "source_file", "started_utc", "gpu",
                                                  "attempt", "peak_gpu_mb", "min_ram_gb", "other_gpu_busy", "runtime_s", "runtime_note")})
    outcomes = entry_outcomes(entries, merged, scores)
    metrics = metric_table(entries, outcomes, scores)
    groups = groups_of(entries)
    groups.update(dock_mode_groups(frame))
    log("summary ...")
    summary = summary_table(metrics, groups, seed, n_boot)
    log("paired tests ...")
    paired = paired_table(metrics, groups, seed, n_boot)
    log("failures as zero ...")
    zero = failures_as_zero(metrics, groups, seed, n_boot)
    return {"merged": merged, "outcomes": outcomes, "metrics": metrics, "groups": groups, "summary": summary, "paired": paired, "zero": zero,
            "variants": variants_table(scores) if len(scores) else pd.DataFrame(), "correlations": correlations(metrics), "completion": completion_table(outcomes, groups),
            "runtime": runtime_table(outcomes), "qa": qa_checks(entries, frame, merged, scores, outcomes)}


def write_all(res, out_dir, inputs, seed=SEED, n_boot=N_BOOT):
    os.makedirs(out_dir, exist_ok=True)

    def save(df, name):
        df.to_csv(os.path.join(out_dir, name), index=False, float_format="%.6g")

    save(spec_frame(res["metrics"]), "metrics_per_entry.csv")
    save(res["outcomes"].drop(columns=["has_score_row"]), "entry_outcomes.csv")
    save(res["completion"], "completion.csv")
    save(res["summary"], "summary_by_method.csv")
    save(res["paired"], "paired_vs_mica.csv")
    save(res["zero"], "sensitivity_failures_as_zero.csv")
    save(res["variants"], "variants.csv")
    save(res["correlations"], "correlations.csv")
    save(res["runtime"], "runtime.csv")
    save(res["merged"], "status_merged.csv")
    with open(os.path.join(out_dir, "qa.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(res["qa"]) + "\n")
    import scipy
    prov = {"script": "scripts/analyze_results.py", "seed": seed, "bootstrap_resamples": n_boot, "novelty_max_identity": NOVELTY_MAX, "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "inputs": {k: {os.path.basename(p): sha256(p) for p in v} for k, v in inputs.items()}}
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entries", default=os.path.join(HERE, "..", "data", "entries_2026.csv"))
    p.add_argument("--status", nargs="+", required=True, help="method_status.csv and mica_status.csv files, oldest first")
    p.add_argument("--scores", nargs="+", required=True, help="scores_long.csv of the scoring notebook")
    p.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--resamples", type=int, default=N_BOOT)
    args = p.parse_args(argv)
    entries = load_entries(args.entries)
    res = analyse(entries, load_status(args.status), load_scores(args.scores), args.seed, args.resamples)
    write_all(res, args.out, {"entries": [args.entries], "status": args.status, "scores": args.scores}, args.seed, args.resamples)
    print("\n".join(res["qa"]))
    print("tables written to %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
