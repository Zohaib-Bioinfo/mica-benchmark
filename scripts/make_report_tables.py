#!/usr/bin/env python3
"""The LaTeX tables and the number macros of the report, written from the files of the repository (nothing is typed in by hand):

    python3 scripts/make_report_tables.py --results results --raw results/raw --data data --out report/generated

Reads the tables of `analyze_results.py` (`results/`), the unpacked result zips (`results/raw/`, for the docking tables and the pinned versions) and the frozen inputs (`data/`).
Writes, in `--out`: `tab_*.tex` (a `tabular` each; the captions are in the sections), `numbers.tex` (one `\\newcommand` per number that the text quotes) and `numbers.json`
(the same, for the tests and for the provenance). A number that cannot be computed is written as `n/a`; the build of the report stops when a section uses a macro that is not defined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

METHODS = ["mica", "modelangelo", "emodelx", "emprot", "e3", "cryoatom1", "cryozeta"]
NAMES = {"mica": "MICA", "modelangelo": "ModelAngelo", "emodelx": "EModelX(+AF)", "emprot": "EMProt", "e3": "E3-CryoFold", "cryoatom1": "CryoAtom", "cryozeta": "CryoZeta"}
METHOD_KEYS = {"mica": "Mica", "modelangelo": "Angelo", "emodelx": "Emodelx", "emprot": "Emprot", "e3": "Ethree", "cryoatom1": "Cryoatom", "cryozeta": "Cryozeta"}
METRIC_KEYS = {"tm_score": "Tm", "ca_match": "Cam", "seq_match": "Sqm", "coverage": "Cov", "ca_quality": "Caq", "seq_identity": "Sid", "ca_rmsd": "Rmsd", "aligned_ca_length": "Ali"}
METRIC_LABEL = {"tm_score": "TM-score", "ca_match": "C$\\alpha$ match (\\%)", "seq_match": "Sequence match (\\%)", "coverage": "Coverage", "ca_quality": "C$\\alpha$ quality (\\%)",
                "seq_identity": "Aligned sequence identity (\\%)", "ca_rmsd": "RMSD (\\AA)", "aligned_ca_length": "Aligned length"}
METRIC_DIGITS = {"tm_score": 2, "ca_match": 1, "seq_match": 1, "coverage": 2, "ca_quality": 1, "seq_identity": 1, "ca_rmsd": 2, "aligned_ca_length": 0}
GROUPS = {"pooled": "", "le3.0": "Low", "gt3.0_le4.0": "High", "novel_le30": "Novel", "mica_docked": "Docked", "mica_nodock": "NoDock"}
GROUP_LABEL = {"pooled": "All entries", "le3.0": "Resolution $\\leq$ 3.0 \\AA", "gt3.0_le4.0": "Resolution $>$ 3.0 to 4.0 \\AA", "novel_le30": "Novel ($\\leq$ 30\\% identity)",
               "mica_docked": "Entries where MICA docked at least one domain", "mica_nodock": "Entries where MICA docked no domain"}
MAIN_GROUPS = ["pooled", "le3.0", "gt3.0_le4.0"]
MODE_GROUPS = ["pooled", "mica_docked", "mica_nodock"]


# ---------------------------------------------------------------------------------------------------------------------------------- formatting
def esc(text):
    text = str(text)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("$", "\\$"), ("~", "\\textasciitilde{}")):
        text = text.replace(a, b)
    return text


def ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and not (isinstance(x, str) and x.strip() == "")


def num(x, nd=2):
    if not ok(x):
        return "--"
    x = float(x)
    return ("%.*f" % (nd, x)).replace("-", "$-$") if x < 0 else "%.*f" % (nd, x)


def zero_free(x, nd=2):
    """A number for a median or a mean: a value that rounds to zero is printed without a sign (a confidence bound keeps its sign through `num`)."""
    t = num(x, nd)
    return ("%.*f" % (nd, 0.0)) if t.replace("$-$", "") == "%.*f" % (nd, 0.0) else t


def plain(x, nd=2):
    """A number for a macro (text mode, ASCII minus replaced by a LaTeX minus)."""
    return "n/a" if not ok(x) else num(x, nd)


def pval(p):
    if not ok(p):
        return "--"
    p = float(p)
    return "$<$0.001" if p < 0.001 else "%.3f" % p


def pct(x, nd=0):
    return "--" if not ok(x) else "%.*f" % (nd, 100 * float(x))


def ci(lo, hi, nd=2):
    return "[%s, %s]" % (num(lo, nd), num(hi, nd)) if ok(lo) and ok(hi) else ""


def booktabs(colspec, header, rows, extra_header=None):
    lines = ["\\begin{tabular}{%s}" % colspec, "\\toprule"]
    if extra_header:
        lines.append(extra_header)
    lines += [" & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MIDRULE":
            lines.append("\\midrule")
        elif isinstance(r, str):
            lines.append(r)
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write(out, name, text):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        f.write(text)
    return name


def read(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else pd.DataFrame()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Macros:
    """\\newcommand lines for the numbers of the text. Names are letters only (TeX)."""

    def __init__(self):
        self.items = {}

    def add(self, name, value):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError("a macro name has letters only: %r" % name)
        self.items[name] = str(value)

    def tex(self):
        return "% generated by scripts/make_report_tables.py; do not edit\n" + "".join("\\newcommand{\\%s}{%s}\n" % (k, v) for k, v in sorted(self.items.items()))


# ---------------------------------------------------------------------------------------------------------------------------------- the tables of the analysis
def completion_table(comp):
    d = comp[comp["group"] == "pooled"].set_index("method")
    rows = []
    for m in METHODS:
        if m not in d.index:
            continue
        r = d.loc[m]
        rows.append([NAMES[m], "%d" % r["n_entries"], "%d" % r["n_model"], "%d" % r["n_failed"], "%d" % r["n_unscorable"], "%d" % r["n_not_run"], pct(r["completion_of_list"]), pct(r["completion_of_attempted"])])
    return booktabs("lrrrrrrr", ["Method", "Entries", "Model scored", "Failed", "No usable model", "Not run", "Completion, of 20 (\\%)", "of attempted (\\%)"], rows)


def summary_table(summ, names=None):
    names = names or NAMES
    d = summ[summ["group"] == "pooled"]
    rows = []
    for m in METHODS:
        s = d[d["method"] == m].set_index("metric")
        if s.empty or "tm_score" not in s.index:
            continue
        t = s.loc["tm_score"]
        cells = [names[m], "%d" % t["n"], "%s (%s--%s)" % (num(t["median"]), num(t["q1"]), num(t["q3"])), "%s %s" % (num(t["mean"]), ci(t["mean_ci_low"], t["mean_ci_high"]))]
        for key in ("ca_match", "coverage", "ca_quality"):
            cells.append(num(s.loc[key]["median"], METRIC_DIGITS[key]) if key in s.index else "--")
        rows.append(cells)
    return booktabs("lrccccc", ["Method", "$n$", "TM-score, median (IQR)", "TM-score, mean [95\\% CI]", "C$\\alpha$ match", "Coverage", "C$\\alpha$ quality"], rows)


def summary_strata_table(summ, groups=None):
    rows = []
    groups = groups or MAIN_GROUPS[1:]
    for g in groups:
        rows.append("\\multicolumn{5}{l}{\\emph{%s}} \\\\" % GROUP_LABEL[g])
        for m in METHODS:
            s = summ[(summ["group"] == g) & (summ["method"] == m) & (summ["metric"] == "tm_score")]
            if s.empty:
                continue
            t = s.iloc[0]
            rows.append([NAMES[m], "%d" % t["n_entries"], "%d" % t["n"], "%s (%s--%s)" % (num(t["median"]), num(t["q1"]), num(t["q3"])), "%s %s" % (num(t["mean"]), ci(t["mean_ci_low"], t["mean_ci_high"]))])
        rows.append("\\midrule") if g != groups[-1] else None
    rows = [r for r in rows if r is not None]
    return booktabs("lrrcc", ["Method", "Entries", "Models", "Median (IQR)", "Mean [95\\% CI]"], rows)


def summary_all_table(summ):
    d = summ[summ["group"] == "pooled"]
    rows = []
    for key in METRIC_KEYS:
        cells = [METRIC_LABEL[key]]
        for m in METHODS:
            s = d[(d["method"] == m) & (d["metric"] == key)]
            if s.empty or not ok(s.iloc[0]["median"]):
                cells.append("--")
                continue
            t, nd = s.iloc[0], METRIC_DIGITS[key]
            cells.append("%s (%s--%s)" % (num(t["median"], nd), num(t["q1"], nd), num(t["q3"], nd)))
        rows.append(cells)
    return booktabs("l" + "c" * len(METHODS), ["Metric"] + [NAMES[m] for m in METHODS], rows)


def paired_table(paired, metric="tm_score", groups=None):
    rows = []
    groups = groups or MAIN_GROUPS
    for g in groups:
        rows.append("\\multicolumn{8}{l}{\\emph{%s}} \\\\" % GROUP_LABEL[g])
        for m in METHODS[1:]:
            s = paired[(paired["group"] == g) & (paired["metric"] == metric) & (paired["method"] == m)]
            if s.empty:
                continue
            r = s.iloc[0]
            rows.append([NAMES[m], "%d" % r["n"], "%s %s" % (num(r["mean_diff"]), ci(r["diff_ci_low"], r["diff_ci_high"])), zero_free(r["median_diff"]), pval(r["wilcoxon_p"]), pval(r["wilcoxon_p_holm"]),
                         pval(r["ttest_p_holm"]), "%d / %d / %d" % (r["n_method_higher"], r["n_method_lower"], r["n_equal"])])
        if g != groups[-1]:
            rows.append("\\midrule")
    return booktabs("lrccrrrc", ["Method", "Pairs", "Mean difference [95\\% CI]", "Median diff.", "Wilcoxon $p$", "Holm $p$", "$t$-test Holm $p$", "higher / lower / equal"], rows)


def zero_table(zero):
    rows = []
    for policy, label in (("strict", "Failed and unusable runs scored 0 (runs not made are left out)"), ("all_missing", "Every missing result scored 0")):
        rows.append("\\multicolumn{6}{l}{\\emph{%s}} \\\\" % label)
        means = zero[(zero["policy"] == policy) & (zero["group"] == "pooled") & (zero["metric"] == "tm_score") & (zero["role"] == "mean_over_entries")].set_index("method")
        pairs = zero[(zero["policy"] == policy) & (zero["group"] == "pooled") & (zero["metric"] == "tm_score") & (zero["role"] == "paired_vs_mica")].set_index("method")
        for m in METHODS:
            if m not in means.index:
                continue
            r = means.loc[m]
            if m == "mica":
                rows.append([NAMES[m], "%d" % r["n"], "%s %s" % (num(r["mean"]), ci(r["mean_ci_low"], r["mean_ci_high"])), "--", "--", "--"])
            elif m in pairs.index:
                p = pairs.loc[m]
                rows.append([NAMES[m], "%d" % r["n"], "%s %s" % (num(r["mean"]), ci(r["mean_ci_low"], r["mean_ci_high"])), num(p["mean_diff"]), pval(p["wilcoxon_p"]), pval(p["wilcoxon_p_holm"])])
        if policy != "all_missing":
            rows.append("\\midrule")
    return booktabs("lrcrrr", ["Method", "$n$", "Mean TM-score [95\\% CI]", "Mean diff. to MICA", "Wilcoxon $p$", "Holm $p$"], rows)


def runtime_table(rt, outcomes):
    rows = []
    gpu = {}
    if not outcomes.empty and "gpu" in outcomes.columns:
        for m, g in outcomes.groupby("method")["gpu"]:
            vals = sorted({str(v) for v in g if ok(v) and str(v) != ""})
            gpu[m] = ", ".join(vals) if vals else "--"
    for m in METHODS:
        s = rt[rt["method"] == m]
        if s.empty:
            continue
        r = s.iloc[0]
        rows.append([NAMES[m], "%d" % r["n_runs"], num(r["median_min"], 1), "%s--%s" % (num(r["min_min"], 1), num(r["max_min"], 1)), num(r["total_h"], 1), esc(gpu.get(m, "--"))])
    return booktabs("lrrrrl", ["Method", "Runs", "Median (min)", "Range (min)", "Total (h)", "Device"], rows)


def per_entry_table(metrics, outcomes, entries):
    tm = {(r["pdb_id"], r["method"]): r["tm_score"] for _, r in metrics.iterrows()} if not metrics.empty else {}
    out = {(r["entry"], r["method"]): r["outcome"] for _, r in outcomes.iterrows()} if not outcomes.empty else {}
    mark = {"failed": "F", "unscorable": "U", "not_run": "N"}
    rows = []
    for stratum in ("le3.0", "gt3.0_le4.0"):
        sub = entries[entries["stratum"] == stratum]
        rows.append("\\multicolumn{%d}{l}{\\emph{%s}} \\\\" % (2 + len(METHODS), GROUP_LABEL[stratum]))
        for _, e in sub.iterrows():
            cells = [esc(e["pdb_id"]), num(e["resolution_A"], 2)]
            for m in METHODS:
                o = out.get((e["pdb_id"], m), "")
                v = tm.get((e["pdb_id"], m), float("nan"))
                cells.append(num(v, 2) if o == "model" and ok(v) else mark.get(o, "--"))
            rows.append(cells)
        if stratum == "le3.0":
            rows.append("\\midrule")
    return booktabs("lr" + "c" * len(METHODS), ["Entry", "Res. (\\AA)"] + [NAMES[m] for m in METHODS], rows)


def failures_table(outcomes):
    rows = []
    if not outcomes.empty:
        d = outcomes[outcomes["outcome"] != "model"].copy()
        d["m"] = d["method"].map({k: i for i, k in enumerate(METHODS)})
        label = {"failed": "failed", "unscorable": "no usable model", "not_run": "not run"}
        for _, r in d.sort_values(["entry", "m"]).iterrows():
            rows.append([esc(r["entry"]), NAMES.get(r["method"], r["method"]), label.get(r["outcome"], r["outcome"]), esc(str(r["reason"])[:160] if ok(r["reason"]) else "")])
    return booktabs("@{}lllp{9.4cm}@{}", ["Entry", "Method", "Outcome", "Reason (from the status table)"], rows) if rows else "\\emph{No failed, unusable or missing run.}\n"


def entries_table(entries, af3):
    plddt = {}
    if not af3.empty and "pdb_id" in af3.columns:
        for pid, g in af3.groupby("pdb_id"):
            plddt[pid] = float(np.average(g["mean_plddt_ca"], weights=g["residues"])) if "residues" in g.columns else float(g["mean_plddt_ca"].mean())
    rows = []
    for stratum in ("le3.0", "gt3.0_le4.0"):
        sub = entries[entries["stratum"] == stratum]
        rows.append("\\multicolumn{9}{l}{\\emph{%s}} \\\\" % GROUP_LABEL[stratum])
        for _, e in sub.iterrows():
            rows.append([esc(e["pdb_id"]), esc(str(e["emdb_id"]).replace("EMD-", "")), num(e["resolution_A"], 2), "%d" % e["modeled_residues"], "%d / %d" % (e["unique_chain_seqs"], e["n_chains"]),
                         num(e["cc_mask"], 2), num(e["max_id_to_pre2026_pdb"], 2), num(plddt.get(e["pdb_id"], float("nan")), 0), num(e.get("map_size_mb", float("nan")), 0)])
        if stratum == "le3.0":
            rows.append("\\midrule")
    return booktabs("llrrcrrrr", ["PDB", "EMDB", "Res. (\\AA)", "Residues", "Unique / all chains", "CC\\_mask", "Max. identity to older PDB", "AF3 pLDDT", "Map (MB)"], rows)


# ---------------------------------------------------------------------------------------------------------------------------------- the unpacked result zips
def raw_folders(raw):
    index = read(os.path.join(raw, "INDEX.csv"), dtype=str)
    if index.empty:
        return []
    return [(r["notebook"], r["stamp"], os.path.join(raw, r["folder"])) for _, r in index.sort_values(["stamp", "notebook"]).iterrows()]


def docking_passes(raw):
    """One row per (pass, entry, domain): was it docked, with which setting, and the local CC that Phenix reported. Pass 1: notebooks mica_cpu_*; pass 2: mica_dock2_*; pass 3: mica_dock3_*."""
    rows = []
    for notebook, stamp, folder in raw_folders(raw):
        p = 1 if notebook.startswith("mica_cpu_") else 2 if notebook.startswith("mica_dock2_") else 3 if notebook.startswith("mica_dock3_") else 0
        table = read(os.path.join(folder, "docking_domains.csv"), dtype=str)
        if not p or table.empty:
            continue
        for (entry, model), g in table.groupby(["entry", "model"], sort=False):
            g = g[g["skipped"] != "True"]
            hit = g[g["docked"] == "True"]
            cc = pd.to_numeric(hit["local_cc"], errors="coerce")
            rows.append({"pass": p, "entry": entry, "model": model, "docked": len(hit) > 0, "setting": hit.iloc[0]["setting"] if len(hit) else "", "local_cc": float(cc.iloc[0]) if len(cc) and ok(cc.iloc[0]) else float("nan"),
                         "seconds": float(pd.to_numeric(g["seconds"], errors="coerce").sum()), "timed_out": bool((g["timed_out"] == "True").any())})
    return pd.DataFrame(rows)


def docking_table(dom, entries):
    rows = []
    for _, e in entries.iterrows():
        cells = [esc(e["pdb_id"])]
        n = None
        for p in (1, 2, 3):
            d = dom[(dom["pass"] == p) & (dom["entry"] == e["pdb_id"])] if not dom.empty else pd.DataFrame()
            if d.empty:
                cells += ["--", "--"]
                continue
            n = len(d) if n is None else n
            cells += ["%d / %d" % (d["docked"].sum(), len(d)), num(d["local_cc"].median(), 2)]
        d3 = dom[(dom["pass"] == 3) & (dom["entry"] == e["pdb_id"])] if not dom.empty else pd.DataFrame()
        cells.append(dock_mode(int(d3["docked"].sum()), len(d3)) if len(d3) else "--")
        cells.insert(1, "%d" % n if n is not None else "--")
        rows.append(cells)
    if not dom.empty:
        tot = []
        for p in (1, 2, 3):
            d = dom[dom["pass"] == p]
            tot += ["%d / %d" % (d["docked"].sum(), len(d)), num(d["local_cc"].median(), 2)] if len(d) else ["--", "--"]
        rows += ["\\midrule", ["All", "%d" % dom[dom["pass"] == max(dom["pass"])].shape[0]] + tot + [""]]
    return booktabs("lrcccccc" + "l", ["Entry", "Domains", "Pass 1 docked", "CC", "Pass 2 docked", "CC", "Pass 3 docked", "CC", "Mode"], rows)


def dock_mode(docked, domains):
    return "none" if docked == 0 else ("full" if docked >= domains else "partial")


PIN_KEYS = {"MICA": "Mica", "ModelAngelo": "Angelo", "EModelX(+AF)": "Emodelx", "EMProt": "Emprot", "E3-CryoFold": "Ethree", "CryoAtom": "Cryoatom", "CryoZeta": "Cryozeta", "US-align": "Usalign"}


def collect_pins(raw):
    """What the runs report as pinned (run_info.json of every result zip, oldest first): name -> value. Two different values of one name are joined with '; ' (the first is the oldest)."""
    seen = {}
    for notebook, stamp, folder in raw_folders(raw):
        path = os.path.join(folder, "run_info.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        items = {}
        for key, name in (("mica_commit", "MICA"), ("phenix_installer", "Phenix"), ("usalign_commit", "US-align")):
            if info.get(key):
                items[name] = str(info[key])
        for key, spec in (info.get("methods") or {}).items():
            if isinstance(spec, dict) and spec.get("commit"):
                items[NAMES.get(key, key)] = str(spec["commit"])
        for name, value in items.items():
            values = seen.setdefault(name, [])
            if value not in values:
                values.append(value)
    return {name: "; ".join(values) for name, values in seen.items()}


def pins_table(raw):
    pins = collect_pins(raw) if raw else {}
    rows = []
    for name in ["MICA", "ModelAngelo", "EModelX(+AF)", "EMProt", "E3-CryoFold", "CryoAtom", "CryoZeta", "Phenix", "US-align"]:
        if name in pins:
            v = pins[name]
            rows.append([esc(name), "\\texttt{%s}" % esc(v[:40] + ("..." if len(v) > 40 else ""))])
    return booktabs("ll", ["Component", "Commit or file (pinned)"], rows) if rows else "\\emph{No run information was found in the raw folder.}\n"


def commit_macros(m, raw):
    """\\CommitMica (first 10 characters), \\CommitMicaFull, \\CommitMicaPins (number of different values seen; 1 = every zip used the same), ..., \\PhenixInstaller, \\PhenixVersion."""
    pins = collect_pins(raw) if raw else {}
    for name, key in PIN_KEYS.items():
        if name in pins:
            values = pins[name].split("; ")
            m.add("Commit" + key, values[0][:10])
            m.add("Commit" + key + "Full", values[0])
            m.add("Commit" + key + "Pins", len(values))
    if "Phenix" in pins:
        values = pins["Phenix"].split("; ")
        found = re.search(r"Phenix-([0-9][0-9.]*-[0-9]+)", values[0])
        m.add("PhenixInstaller", values[0])
        m.add("PhenixVersion", found.group(1) if found else values[0])
        m.add("PhenixPins", len(values))


def methods_screen_table(csv_path):
    t = read(csv_path, dtype=str).fillna("")
    if t.empty:
        return "\\emph{The methods table was not found.}\n"
    lines = ["\\begin{longtable}{@{}p{3.0cm}p{0.9cm}p{1.5cm}p{9.0cm}@{}}", "\\toprule", "Method & Year & Included & Reason \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "Method & Year & Included & Reason \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    label = {"baseline": "baseline", "yes": "passed", "reserve": "reserve", "no": "no"}
    for _, r in t.iterrows():
        year = re.match(r"\s*(\d{4})", str(r["paper_year"]))
        lines.append("%s & %s & %s & {\\footnotesize %s} \\\\" % (esc(r["method"]), year.group(1) if year else "--", label.get(r["included"], esc(r["included"])), esc(r["reason"])))
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def methods_macros(m, csv_path):
    t = read(csv_path, dtype=str).fillna("")
    if t.empty:
        return
    m.add("NScreened", int((t["included"] != "baseline").sum()))
    for key, tag in (("yes", "Passed"), ("reserve", "Reserve"), ("no", "Excluded")):
        m.add("NScreen" + tag, int((t["included"] == key).sum()))


def thousands(x):
    return "{:,}".format(int(x)).replace(",", "\\,")


def pool_flow_table(data):
    flow = read(os.path.join(data, "pool", "pool_flow.csv"))
    if flow.empty:
        return "\\emph{The pool flow was not found.}\n"
    rows = []
    for _, r in flow.iterrows():
        removed = r["n_removed_at_this_step"]
        text = esc(r["description"]).replace("4.0 A", "4.0 \\AA{}")
        rows.append([text, thousands(r["n_remaining"]), thousands(removed) if ok(removed) else "--"])
    return booktabs("lrr", ["Step", "Entries left", "Removed"], rows)


def program3_text(st):
    """The status of program 3: the last one, and the sequence of results when the entry was run more than once (failed, then ok)."""
    results = st.get("_results", [])
    if len(results) > 1 and len(set(results)) > 1:
        return " $\\to$ ".join(results)
    return esc(st.get("status", "--") or "--")


def mica_modes_table(raw, entries, metrics, outcomes):
    """One row per entry: how far the docking got (pass 3), the status of MICA's program 3 and its TM-score."""
    status = {}
    for notebook, stamp, folder in raw_folders(raw) if raw else []:
        if not (notebook.startswith("mica_dock3_") or (notebook.startswith("mica_gpu") and "nodock" not in notebook)):
            continue
        t = read(os.path.join(folder, "mica_status.csv"), dtype=str).fillna("")
        for _, r in t.iterrows():
            old = status.get(r["entry"], {})
            results = old.get("_results", []) + ([r["status"]] if notebook.startswith("mica_gpu") and r["status"] in ("ok", "failed") else [])
            status[r["entry"]] = dict(old, **{k: v for k, v in r.items() if v != ""})
            status[r["entry"]]["_results"] = results
    tm = {r["pdb_id"]: r["tm_score"] for _, r in metrics.iterrows() if r.get("method") == "mica"} if not metrics.empty else {}
    out = {r["entry"]: r["outcome"] for _, r in outcomes.iterrows() if r["method"] == "mica"} if not outcomes.empty else {}
    rows = []
    for _, e in entries.iterrows():
        st = status.get(e["pdb_id"], {})
        o = out.get(e["pdb_id"], "")
        v = tm.get(e["pdb_id"], float("nan"))
        rows.append([esc(e["pdb_id"]), esc(st.get("domains", "--") or "--"), esc(st.get("domains_docked", "--") or "--"), esc(st.get("dock_mode", "--") or "--"), program3_text(st),
                     num(v, 2) if o == "model" and ok(v) else {"failed": "F", "unscorable": "U", "not_run": "N"}.get(o, "--")])
    return booktabs("lrrlll", ["Entry", "Domains", "Docked", "Mode", "Program 3", "TM-score"], rows)


def dock_effect(main_metrics, nd_metrics, dom, entries):
    """MICA with and without docked AlphaFold input on the entries that have a model in both runs (`mica_gpu*` and `mica_gpu*_nodock`).
    The difference is docked minus no docked input. Returns the table and the numbers for the text."""
    import analyze_results as A

    def tm(df):
        if df.empty or "method" not in df.columns:
            return {}
        return {r["pdb_id"]: float(r["tm_score"]) for _, r in df.iterrows() if r["method"] == "mica" and ok(r["tm_score"])}
    a, b = tm(main_metrics), tm(nd_metrics)
    modes = {}
    if not dom.empty and (dom["pass"] == 3).any():
        modes = {e: dock_mode(int(g["docked"].sum()), len(g)) for e, g in dom[dom["pass"] == 3].groupby("entry")}
    rows, xs, ys = [], [], []
    for _, e in entries.iterrows():
        k = e["pdb_id"]
        if k in a and k in b:
            xs.append(a[k])
            ys.append(b[k])
            rows.append([esc(k), num(e["resolution_A"], 2), modes.get(k, "--"), num(a[k], 3), num(b[k], 3), ("%+.3f" % (a[k] - b[k])).replace("-0.000", "0.000").replace("+0.000", "0.000")])
    macros = {name: "n/a" for name in ("N", "Mean", "Lo", "Hi", "Med", "P", "Higher", "Lower", "Equal", "MeanDocked", "MeanNoDock", "Dz", "RankBis")}
    if not rows:
        return "\\emph{No entry has a result both with and without docked input.}\n", macros
    st = A.paired_stats(xs, ys, "dock_effect|tm_score", better="higher")
    macros.update(N=str(st["n"]), Mean=plain(st["mean_diff"], 3), Lo=plain(st["diff_ci_low"], 3), Hi=plain(st["diff_ci_high"], 3), Med=zero_free(st["median_diff"], 3) if ok(st["median_diff"]) else "n/a", P=pval(st["wilcoxon_p"]),
                  Higher=str(st["n_method_higher"]), Lower=str(st["n_method_lower"]), Equal=str(st["n_equal"]), MeanDocked=plain(st["mean_method"], 3), MeanNoDock=plain(st["mean_mica"], 3),
                  Dz=plain(st["cohens_dz"], 2), RankBis=plain(st["rank_biserial"], 2))
    rows += ["\\midrule",
             "\\multicolumn{6}{l}{Mean difference %s %s; median %s; Wilcoxon $p$ = %s; higher / lower / equal: %d / %d / %d} \\\\" % (
                 num(st["mean_diff"], 3), ci(st["diff_ci_low"], st["diff_ci_high"], 3), zero_free(st["median_diff"], 3), pval(st["wilcoxon_p"]), st["n_method_higher"], st["n_method_lower"], st["n_equal"])]
    return booktabs("lrlccr", ["Entry", "Res. (\\AA)", "Docking", "TM docked", "TM no docked input", "Difference"], rows), macros


# ---------------------------------------------------------------------------------------------------------------------------------- the numbers of the text
def dataset_macros(m, data):
    flow = read(os.path.join(data, "pool", "pool_flow.csv"))
    names = {"all_cryoem": "PoolAll", "single_particle": "PoolSingleParticle", "resolution": "PoolResolution", "protein_only": "PoolProteinOnly", "modeled_residues": "PoolResidues",
             "unique_sequences": "PoolSequences", "unk_fraction": "PoolUnk", "one_emdb_map": "PoolOneMap", "emdb_map_released": "PoolFinal"}
    for _, r in flow.iterrows():
        if r["step"] in names:
            m.add(names[r["step"]], "{:,}".format(int(r["n_remaining"])).replace(",", "\\,"))
    path = os.path.join(data, "selection", "selection_info.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            sel = json.load(f)
        d = sel.get("decisions", {})
        m.add("SelSeed", sel.get("seed", ""))
        m.add("SelCcMin", plain(sel.get("cc_min"), 2))
        m.add("SelSelected", d.get("selected", "n/a"))
        m.add("SelCcRejected", d.get("reject_cc_mask", "n/a"))
        m.add("SelCcFailed", d.get("reject_cc_failed", "n/a"))
        m.add("SelClusterUsed", d.get("skip_cluster_used", "n/a"))
        m.add("SelNotReached", d.get("not_reached", "n/a"))
        for k, v in (sel.get("selected_per_stratum") or {}).items():
            m.add({"le3.0": "SelLow", "gt3.0_le4.0": "SelHigh"}.get(k, "SelOther"), v)
    path = os.path.join(data, "selection", "mmseqs_info.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            mm = json.load(f)
        m.add("SelClusters", mm.get("n_clusters", "n/a"))
        m.add("SelSequences", mm.get("n_sequences", "n/a"))
    af3 = read(os.path.join(data, "af3", "af3_results_check.csv"))
    if not af3.empty and "mean_plddt_ca" in af3.columns:                              # the AlphaFold 3 predictions: one per unique chain sequence
        m.add("AfChains", len(af3))
        m.add("AfPlddtMin", plain(af3["mean_plddt_ca"].min(), 1))
        m.add("AfPlddtMed", plain(af3["mean_plddt_ca"].median(), 1))
        m.add("AfPlddtMax", plain(af3["mean_plddt_ca"].max(), 1))
        if "frac_ca_plddt_lt50" in af3.columns:
            worst = af3.sort_values(["frac_ca_plddt_lt50", "pdb_id"], ascending=[False, True]).head(3)
            m.add("AfLowList", listing(["%s chain %s (%s\\,\\%%)" % (r["pdb_id"], r["entity"], pct(r["frac_ca_plddt_lt50"])) for _, r in worst.iterrows()]))
    entries = read(os.path.join(data, "entries_2026.csv"))
    if not entries.empty:
        m.add("NEntries", len(entries))
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
        m.add("NMethods", len(METHODS))                                  # the methods of the analysis, MICA included
        m.add("NMethodsWord", words[len(METHODS)])
        m.add("NMethodsWordCap", words[len(METHODS)].capitalize())
        m.add("NFamily", len(METHODS) - 1)                               # the method pairs of the Holm correction
        m.add("NFamilyWord", words[len(METHODS) - 1])
        m.add("ResMin", plain(entries["resolution_A"].min(), 2))
        m.add("ResMax", plain(entries["resolution_A"].max(), 2))
        m.add("ResiduesMin", int(entries["modeled_residues"].min()))
        m.add("ResiduesMax", int(entries["modeled_residues"].max()))
        m.add("ResiduesMedian", plain(entries["modeled_residues"].median(), 0))
        m.add("NMultiChain", int((entries["n_chains"] > 1).sum()))
        m.add("NSingleChain", int((entries["n_chains"] == 1).sum()))
        ident = entries["max_id_to_pre2026_pdb"]
        m.add("NIdentHigh", int((ident >= 0.95).sum()))
        m.add("NIdentLow", int((ident <= 0.30).sum()))
        m.add("NIdentMid", int(((ident > 0.30) & (ident < 0.95)).sum()))


def result_macros(m, res, prefix=""):
    """`prefix` ("Nd" for the analysis of the sessions without docked AlphaFold input) goes in front of every name."""
    comp, summ, paired = res["completion"], res["summary"], res["paired"]
    if not comp.empty:
        for _, r in comp[comp["group"] == "pooled"].iterrows():
            k = METHOD_KEYS.get(r["method"])
            if k:
                m.add(prefix + "CompList" + k, pct(r["completion_of_list"]))
                m.add(prefix + "CompAtt" + k, pct(r["completion_of_attempted"]))
                for col, tag in (("n_model", "NModel"), ("n_failed", "NFailed"), ("n_unscorable", "NUnusable"), ("n_not_run", "NNotRun")):
                    m.add(prefix + tag + k, int(r[col]))
                m.add(prefix + "NRan" + k, int(r["n_model"] + r["n_failed"] + r["n_unscorable"]))      # the entries on which the method was run (not the ones that were not run)
    if not summ.empty:
        for _, r in summ.iterrows():
            mk, k, g = METRIC_KEYS.get(r["metric"]), METHOD_KEYS.get(r["method"]), GROUPS.get(r["group"])
            if mk is None or k is None or g is None:
                continue
            nd = METRIC_DIGITS[r["metric"]]
            for tag, col in (("Med", "median"), ("Mean", "mean"), ("Qone", "q1"), ("Qthree", "q3"), ("Lo", "mean_ci_low"), ("Hi", "mean_ci_high")):
                m.add(prefix + mk + tag + k + g, plain(r[col], nd))
            m.add(prefix + mk + "N" + k + g, int(r["n"]))
    if not paired.empty:
        for _, r in paired.iterrows():
            mk, k, g = METRIC_KEYS.get(r["metric"]), METHOD_KEYS.get(r["method"]), GROUPS.get(r["group"])
            if mk is None or k is None or g is None:
                continue
            nd = METRIC_DIGITS[r["metric"]]
            m.add(prefix + "Diff" + mk + k + g, plain(r["mean_diff"], max(nd, 2)))
            m.add(prefix + "DiffLo" + mk + k + g, plain(r["diff_ci_low"], max(nd, 2)))
            m.add(prefix + "DiffHi" + mk + k + g, plain(r["diff_ci_high"], max(nd, 2)))
            m.add(prefix + "MedDiff" + mk + k + g, zero_free(r["median_diff"], max(nd, 2)) if ok(r["median_diff"]) else "n/a")
            m.add(prefix + "Pw" + mk + k + g, pval(r["wilcoxon_p"]))
            m.add(prefix + "PwHolm" + mk + k + g, pval(r["wilcoxon_p_holm"]))
            m.add(prefix + "PtHolm" + mk + k + g, pval(r["ttest_p_holm"]))
            m.add(prefix + "NPairs" + mk + k + g, int(r["n"]))
            m.add(prefix + "Higher" + mk + k + g, int(r["n_method_higher"]))
            m.add(prefix + "Lower" + mk + k + g, int(r["n_method_lower"]))
            m.add(prefix + "RankBis" + mk + k + g, plain(r["rank_biserial"], 2))


def docking_macros(m, dom):
    if dom.empty:
        return
    for p, tag in ((1, "One"), (2, "Two"), (3, "Three")):
        d = dom[dom["pass"] == p]
        if not len(d):
            continue
        m.add("DockDomains" + tag, len(d))
        m.add("DockDocked" + tag, int(d["docked"].sum()))
        m.add("DockTimedOut" + tag, int(((~d["docked"]) & d["timed_out"]).sum()))
        m.add("DockEntriesNone" + tag, int((d.groupby("entry")["docked"].sum() == 0).sum()))
        m.add("DockEntriesFull" + tag, int((d.groupby("entry")["docked"].agg(lambda x: x.all())).sum()))
        m.add("DockEntries" + tag, int(d["entry"].nunique()))
        m.add("DockCcMed" + tag, plain(d[d["docked"]]["local_cc"].median(), 2) if d["docked"].any() else "n/a")
        m.add("DockMinutesMed" + tag, plain(d["seconds"].median() / 60.0, 0))
    if (dom["pass"] == 3).any():
        d3 = dom[dom["pass"] == 3]
        modes = d3.groupby("entry")["docked"].agg(lambda x: dock_mode(int(x.sum()), len(x)))
        for mode, tag in (("full", "Full"), ("partial", "Partial"), ("none", "None")):
            m.add("DockMode" + tag, int((modes == mode).sum()))
        m.add("DockRungOne", int(((d3["docked"]) & (d3["setting"] == "res6_quick")).sum()))
        m.add("DockRungTwo", int(((d3["docked"]) & (d3["setting"] == "res10_quick")).sum()))
    if (dom["pass"] == 1).any() and (dom["pass"] == 3).any():
        a = dom[(dom["pass"] == 1) & dom["docked"]].set_index(["entry", "model"])["local_cc"]
        b = dom[(dom["pass"] == 3) & dom["docked"]].set_index(["entry", "model"])["local_cc"]
        both = a.index.intersection(b.index)
        m.add("DockBoth", len(both))
        diff = (b.loc[both] - a.loc[both]).dropna() if len(both) else pd.Series(dtype=float)
        m.add("DockCcDiffMed", plain(diff.median(), 2) if len(diff) else "n/a")


def listing(items):
    """'a', 'a and b', 'a, b and c'."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else (", ".join(items[:-1]) + " and " + items[-1] if items else "none")


OUTPUT_KINDS = {"mica_all_atom": "MICA, all-atom model (scored)", "mica_ca": "MICA, C$\\alpha$ model after chain building", "mica_init_ca": "MICA, initial C$\\alpha$ model of the network",
                "modelangelo": "ModelAngelo, sequence-assigned model (scored)", "modelangelo_raw": "ModelAngelo, model without sequence assignment",
                "emodelx_af_all_atom": "EModelX(+AF), all-atom model (scored)", "emodelx_af_ca": "EModelX(+AF), C$\\alpha$ model", "emprot": "EMProt, build with the AF3 models (scored)",
                "emprot_denovo": "EMProt, de novo model", "e3_prealign": "E3-CryoFold, \\texttt{pre\\_align} model (scored)",
                "cryoatom1": "CryoAtom v1.0.0, pruned model (scored)", "cryoatom1_raw": "CryoAtom v1.0.0, model before pruning", "cryozeta": "CryoZeta, top-ranked sample 0 (scored)", "cryozeta_s1": "CryoZeta, sample 1 (ranked second)", "cryozeta_s2": "CryoZeta, sample 2 (ranked third)",
                "cryozeta_s3": "CryoZeta, sample 3 (ranked fourth)", "cryozeta_s4": "CryoZeta, sample 4 (ranked fifth)"}


def variants_table(var):
    """The kinds of output of every method that were scored (the primary kind is marked), by median and mean TM-score and by the median of the other measures."""
    if var.empty:
        return "\\emph{No variants were scored.}\n"
    rows = []
    for _, r in var.iterrows():
        rows.append([OUTPUT_KINDS.get(r["model_kind"], esc(r["model_kind"])), str(int(r["n_models"])), num(r["tm_score_median"], 2), num(r["tm_score_mean"], 2), num(r["tm_score_fns_median"], 2),
                     num(r["ca_match_median"], 1), num(r["coverage_median"], 2), num(r["ca_quality_median"], 1)])
    return booktabs("@{}p{5.2cm}r*{6}{>{\\raggedleft\\arraybackslash}p{1.45cm}}@{}", ["Output", "$n$", "TM-score, median", "TM-score, mean", "Fragment-agnostic TM, median", "C$\\alpha$ match, median", "Coverage, median", "C$\\alpha$ quality, median"], rows)


def fns_table(summ, paired):
    """TM-score by the fragment-agnostic alignment (one chain per structure, `-mm 5`): median (IQR) and the paired comparison with MICA."""
    if summ.empty or paired.empty:
        return "\\emph{Not available.}\n"
    s = summ[(summ["group"] == "pooled") & (summ["metric"] == "tm_score_fns")].set_index("method")
    p = paired[(paired["group"] == "pooled") & (paired["metric"] == "tm_score_fns")].set_index("method")
    rows = []
    for k in METHODS:
        if k not in s.index:
            continue
        r = s.loc[k]
        if k == "mica" or k not in p.index:
            rows.append([NAMES[k], str(int(r["n"])), "%s (%s--%s)" % (num(r["median"], 2), num(r["q1"], 2), num(r["q3"], 2)), "--", "--", "--"])
        else:
            q = p.loc[k]
            rows.append([NAMES[k], str(int(r["n"])), "%s (%s--%s)" % (num(r["median"], 2), num(r["q1"], 2), num(r["q3"], 2)), "%s %s" % (num(q["mean_diff"], 2), ci(q["diff_ci_low"], q["diff_ci_high"], 2)),
                         pval(q["wilcoxon_p"]), pval(q["wilcoxon_p_holm"])])
    return booktabs("lrcccc", ["Method", "$n$", "TM-score, fragment-agnostic, median (IQR)", "Mean difference to MICA [95\\% CI]", "Wilcoxon $p$", "Holm $p$"], rows)


def corr_table(cor):
    """Spearman rank correlation of the TM-score with the resolution, the number of modelled residues and the identity to older PDB entries (descriptive)."""
    if cor.empty:
        return "\\emph{Not available.}\n"
    rows = []
    for k in METHODS:
        c = cor[(cor["method"] == k) & (cor["y"] == "tm_score")].set_index("x")
        cells = ["%s (%s)" % (num(c.loc[x, "spearman_rho"], 2), pval(c.loc[x, "p"])) if x in c.index else "--" for x in ("resolution_A", "modeled_residues", "max_id_to_pre2026_pdb")]
        rows.append([NAMES[k], str(int(c["n"].iloc[0])) if len(c) else "--"] + cells)
    return booktabs("lrccc", ["Method", "$n$", "Resolution", "Modelled residues", "Identity to older PDB"], rows)


def extra_macros(m, res, dom=None):
    """Numbers that the text quotes beyond the tables: minimum and maximum of every measure, the fragment-agnostic TM-score, the kinds of MICA's output, the rank correlations,
    the run times, the counts above or below a TM-score, the reruns of MICA and MICA by docking mode."""
    summ, paired, metrics, runtime = res["summary"], res["paired"], res["metrics"], res["runtime"]
    out = res.get("outcomes", pd.DataFrame())
    if not out.empty and "outcome" in out.columns:                                  # why the runs without a model ended: the system killed them, a limit, or the program stopped
        f = out[out["outcome"].isin(("failed", "unscorable"))].copy()                  # a run that was not run (no time left in the session) is not a failure
        reason = f["reason"].fillna("").astype(str) if "reason" in f.columns else pd.Series([""] * len(f), index=f.index)
        kill = reason.str.contains("exit code -9|killed by the system", regex=True)
        limit = reason.str.contains("timeout", case=False) & ~kill
        crash = ~kill & ~limit
        m.add("FailTotal", len(f))
        m.add("FailKill", int(kill.sum()))
        m.add("FailTimeout", int(limit.sum()))
        m.add("FailCrash", int(crash.sum()))
        for method, key in METHOD_KEYS.items():
            sel = f["method"] == method
            m.add("FailKill" + key, int((kill & sel).sum()))
            m.add("FailTimeout" + key, int((limit & sel).sum()))
            m.add("FailCrash" + key, int((crash & sel).sum()))
            m.add("FailCrashList" + key, listing(sorted(f.loc[crash & sel, "entry"])))
            m.add("FailKillList" + key, listing(sorted(f.loc[kill & sel, "entry"])))
            m.add("FailTimeoutList" + key, listing(sorted(f.loc[limit & sel, "entry"])))
        for method, key in METHOD_KEYS.items():                                  # which entries have a model, failed, or were not run
            for outcome, tag in (("model", "ModelList"), ("failed", "FailedList"), ("not_run", "NotRunList")):
                m.add(tag + key, listing(sorted(out.loc[(out["method"] == method) & (out["outcome"] == outcome), "entry"])))
    if not summ.empty:
        for _, r in summ[summ["group"] == "pooled"].iterrows():
            mk, k = METRIC_KEYS.get(r["metric"]), METHOD_KEYS.get(r["method"])
            if mk and k:
                m.add(mk + "Min" + k, plain(r["min"], METRIC_DIGITS[r["metric"]]))
                m.add(mk + "Max" + k, plain(r["max"], METRIC_DIGITS[r["metric"]]))
            if r["metric"] == "tm_score_fns" and k:
                m.add("FnsMean" + k, plain(r["mean"], 2))
                m.add("FnsMed" + k, plain(r["median"], 2))
    if not paired.empty:
        for _, r in paired[(paired["group"] == "pooled") & (paired["metric"] == "tm_score_fns")].iterrows():
            k = METHOD_KEYS.get(r["method"])
            if k:
                m.add("DiffFns" + k, plain(r["mean_diff"], 2))
                m.add("DiffLoFns" + k, plain(r["diff_ci_low"], 2))
                m.add("DiffHiFns" + k, plain(r["diff_ci_high"], 2))
                m.add("PwFns" + k, pval(r["wilcoxon_p"]))
                m.add("PwHolmFns" + k, pval(r["wilcoxon_p_holm"]))
                m.add("NPairsFns" + k, int(r["n"]))
                m.add("HigherFns" + k, int(r["n_method_higher"]))
                m.add("LowerFns" + k, int(r["n_method_lower"]))
    var = res.get("variants", pd.DataFrame())
    if not var.empty:
        for kind, tag in (("mica_init_ca", "Init"), ("mica_ca", "Ca"), ("mica_all_atom", "All")):
            r = var[(var["method"] == "mica") & (var["model_kind"] == kind)]
            if len(r):
                r = r.iloc[0]
                m.add("MicaVar" + tag + "TmMean", plain(r["tm_score_mean"], 2))
                m.add("MicaVar" + tag + "TmMed", plain(r["tm_score_median"], 2))
                m.add("MicaVar" + tag + "CovMed", plain(r["coverage_median"], 2))
                m.add("MicaVar" + tag + "N", int(r["n_models"]))
        for kind, name in (("modelangelo_raw", "AngeloRaw"), ("emprot_denovo", "EmprotDenovo"), ("cryoatom1_raw", "CryoatomRaw")):
            r = var[var["model_kind"] == kind]
            if len(r):
                m.add("Var" + name + "TmMed", plain(r.iloc[0]["tm_score_median"], 2))
                m.add("Var" + name + "TmMean", plain(r.iloc[0]["tm_score_mean"], 2))
    cor = res.get("correlations", pd.DataFrame())
    if not cor.empty:
        xs, ys = {"resolution_A": "Res", "modeled_residues": "Size", "max_id_to_pre2026_pdb": "Ident"}, {"tm_score": "Tm", "ca_quality": "Caq"}
        for _, r in cor.iterrows():
            k = METHOD_KEYS.get(r["method"])
            if k and r["x"] in xs and r["y"] in ys:
                tag = xs[r["x"]] + ys[r["y"]] + k
                m.add("Rho" + tag, plain(r["spearman_rho"], 2))
                m.add("RhoP" + tag, pval(r["p"]))
                m.add("RhoN" + tag, int(r["n"]))
    if not cor.empty:
        ident = cor[(cor["x"] == "max_id_to_pre2026_pdb") & (cor["y"] == "tm_score")]["p"].dropna()
        if len(ident):
            m.add("RhoIdentMinP", pval(float(ident.min())))
    if not metrics.empty and "n_chains_pred" in metrics.columns:
        got = metrics[metrics["status"] == "model"]
        for method, k in METHOD_KEYS.items():
            v = got[got["method"] == method]["n_chains_pred"].dropna()
            if len(v):
                m.add("ChainsMed" + k, plain(float(v.median()), 0))
                m.add("ChainsMax" + k, plain(float(v.max()), 0))
        if "n_chains_native" in got.columns and got["n_chains_native"].notna().any():
            m.add("ChainsNativeMax", plain(float(got["n_chains_native"].dropna().max()), 0))
    if not metrics.empty and "residue_coverage" in metrics.columns:                 # the second highest coverage: how far one entry stands out
        got = metrics[metrics["status"] == "model"]
        for method, k in METHOD_KEYS.items():
            v = sorted(got[got["method"] == method]["residue_coverage"].dropna(), reverse=True)
            if len(v) >= 2:
                m.add("CovSecond" + k, plain(float(v[1]), 2))
    if not runtime.empty:
        for _, r in runtime.iterrows():
            k = METHOD_KEYS.get(r["method"])
            if k:
                m.add("RtMed" + k, plain(r["median_min"], 0))
                m.add("RtMin" + k, plain(r["min_min"], 0))
                m.add("RtMax" + k, plain(r["max_min"], 0))
                m.add("RtTot" + k, plain(r["total_h"], 1))
                m.add("RtN" + k, int(r["n_runs"]))
    if not metrics.empty and "runtime_min" in metrics.columns:                      # models that took more than the 3 h that the baselines had (a longer limit was given to CryoAtom and CryoZeta)
        over = metrics[(metrics["status"] == "model") & (metrics["runtime_min"] > 180)]
        for method, k in METHOD_KEYS.items():
            g = over[over["method"] == method].sort_values("pdb_id")
            m.add("NOverThreeH" + k, len(g))
            m.add("OverThreeHList" + k, listing(["%s (%s\\,min)" % (r["pdb_id"], plain(r["runtime_min"], 0)) for _, r in g.iterrows()]) if len(g) else "none")
    if not metrics.empty and "tm_score" in metrics.columns:
        got = metrics[metrics["status"] == "model"]
        for method, k in METHOD_KEYS.items():
            v = got[got["method"] == method]["tm_score"].dropna()
            m.add("NTmGeNinetyFive" + k, int((v >= 0.95).sum()))
            m.add("NTmGeNinety" + k, int((v >= 0.90).sum()))
            m.add("NTmLtHalf" + k, int((v < 0.50).sum()))
            m.add("NTmLtNinety" + k, int((v < 0.90).sum()))
    zero = res.get("zero", pd.DataFrame())
    if not zero.empty:
        z = zero[(zero["policy"] == "strict") & (zero["group"] == "pooled") & (zero["metric"] == "tm_score")]
        for _, r in z.iterrows():
            k = METHOD_KEYS.get(r["method"])
            if not k:
                continue
            if r["role"] == "mean_over_entries":
                m.add("ZeroMean" + k, plain(r["mean"], 2))
                m.add("ZeroLo" + k, plain(r["mean_ci_low"], 2))
                m.add("ZeroHi" + k, plain(r["mean_ci_high"], 2))
                m.add("ZeroN" + k, int(r["n"]))
            elif r["role"] == "paired_vs_mica":
                m.add("ZeroDiff" + k, plain(r["mean_diff"], 2))
                m.add("ZeroDiffLo" + k, plain(r["diff_ci_low"], 2))
                m.add("ZeroDiffHi" + k, plain(r["diff_ci_high"], 2))
                m.add("ZeroPw" + k, pval(r["wilcoxon_p"]))
                m.add("ZeroPwHolm" + k, pval(r["wilcoxon_p_holm"]))
    merged = res.get("merged", pd.DataFrame())
    if not merged.empty and "conflict" in merged.columns:
        r = merged[(merged["method"] == "mica") & (merged["conflict"].astype(str) == "True")]
        order = list(res["entries"]["pdb_id"]) if "entries" in res and not res["entries"].empty else list(r["entry"])
        reruns = [e for e in order if e in set(r["entry"])]
        m.add("NRerunMica", len(reruns))
        m.add("RerunMicaList", listing(reruns))
        if not metrics.empty and "tm_score" in metrics.columns:
            got = metrics[(metrics["method"] == "mica") & (metrics["status"] == "model")]
            first = got[~got["pdb_id"].isin(reruns)]["tm_score"].dropna()           # what MICA would have if the reruns counted as failures
            n_all = len(res["entries"]) if "entries" in res and len(res["entries"]) else len(order)
            m.add("FirstModels", len(first))
            m.add("FirstTmMean", plain(float(first.mean()), 2) if len(first) else "n/a")
            m.add("FirstZeroMean", plain(float(first.sum()) / n_all, 2) if n_all else "n/a")
    if dom is not None and not dom.empty and (dom["pass"] == 3).any() and not metrics.empty:
        d3 = dom[dom["pass"] == 3]
        modes = {e: dock_mode(int(g["docked"].sum()), len(g)) for e, g in d3.groupby("entry")}
        tm = {r["pdb_id"]: float(r["tm_score"]) for _, r in metrics.iterrows() if r["method"] == "mica" and r["status"] == "model" and ok(r["tm_score"])}
        for mode, tag in (("full", "Full"), ("partial", "Partial"), ("none", "None")):
            es = [e for e, md in modes.items() if md == mode]
            vals = [tm[e] for e in es if e in tm]
            m.add("Mode" + tag + "N", len(es))
            m.add("Mode" + tag + "Models", len(vals))
            m.add("Mode" + tag + "Failed", len(es) - len(vals))
            m.add("Mode" + tag + "TmMean", plain(float(np.mean(vals)), 2) if vals else "n/a")
            m.add("Mode" + tag + "TmMin", plain(min(vals), 2) if vals else "n/a")
            m.add("Mode" + tag + "TmMax", plain(max(vals), 2) if vals else "n/a")
        lowest = sorted(tm, key=lambda e: tm[e])[:3]
        m.add("ModeLowThreePartial", sum(1 for e in lowest if modes.get(e) == "partial"))
        m.add("ModeLowThreeList", listing(lowest))


STATUS_ORDER = ["ok", "docked", "failed", "deferred", "interrupted", "killed_corun", "not run"]


def session_rows(raw):
    """One row per unpacked result zip: the notebook, its start (UTC), its length in minutes, whether it used a GPU, whether it ended complete, and what it produced
    (the status counts of its status tables, or the number of models that it scored)."""
    rows = []
    for notebook, stamp, folder in raw_folders(raw) if raw and os.path.isdir(raw) else []:
        info = {}
        path = os.path.join(folder, "run_info.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
        counts = {}
        for name in ("mica_status.csv", "method_status.csv"):
            t = read(os.path.join(folder, name), dtype=str)
            if len(t) and "status" in t.columns:
                for v, n in t["status"].fillna("").value_counts().items():
                    counts[v] = counts.get(v, 0) + int(n)
        scored = read(os.path.join(folder, "scoring", "scores_long.csv"), dtype=str)
        try:
            minutes = float(info.get("elapsed_minutes", ""))
        except (TypeError, ValueError):
            minutes = math.nan
        rows.append({"notebook": notebook, "stamp": stamp, "started": str(info.get("started_utc", ""))[:16].replace("T", " "), "minutes": minutes, "gpu": bool(info.get("gpu")),
                     "complete": info.get("complete", True) is not False, "problem": str(info.get("problem") or ""), "counts": counts, "scored": len(scored)})
    return rows


def sessions_table(raw):
    rows = session_rows(raw)
    if not rows:
        return "\\emph{No result zip has been unpacked.}\n"
    out = []
    for r in rows:
        parts = ["%s %d" % (k, r["counts"][k]) for k in STATUS_ORDER if r["counts"].get(k)] + ["%s %d" % (k, v) for k, v in sorted(r["counts"].items()) if k not in STATUS_ORDER and k]
        what = ", ".join(parts) if parts else ("%d models scored" % r["scored"] if r["scored"] else "--")
        out.append(["\\texttt{%s}" % esc(r["notebook"]), r["started"] or "--", plain(r["minutes"], 0) if ok(r["minutes"]) else "--", "yes" if r["gpu"] else "no", esc(what) + ("" if r["complete"] else (" (stopped by a check of the notebook)" if r["problem"] else " (cut by the time budget)"))])
    return booktabs("llrcl", ["Notebook", "Started (UTC)", "Minutes", "GPU", "Entries by status, or models scored"], out)


def session_macros(m, raw):
    rows = session_rows(raw)
    if not rows:
        return
    m.add("NSessions", len(rows))
    m.add("GpuSessionHours", plain(sum(r["minutes"] for r in rows if r["gpu"] and ok(r["minutes"])) / 60.0, 1))
    m.add("CpuSessionHours", plain(sum(r["minutes"] for r in rows if not r["gpu"] and ok(r["minutes"])) / 60.0, 1))


def environment_macros(m, raw):
    for notebook, stamp, folder in raw_folders(raw):
        path = os.path.join(folder, "run_info.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        if info.get("cpu_cores") and "CpuCores" not in m.items:
            m.add("CpuCores", info["cpu_cores"])
            m.add("RamGb", info.get("ram_gb", ""))
        if info.get("gpu") and "GpuName" not in m.items:
            m.add("GpuName", esc(str(info["gpu"]).splitlines()[0].split(",")[0]))
            m.add("GpuMemory", esc(str(info["gpu"]).splitlines()[0].split(",")[1].strip()) if "," in str(info["gpu"]) else "")


def build(results, raw, data, out):
    """Writes every table and the macros. Returns the list of files."""
    res = {"completion": read(os.path.join(results, "completion.csv")), "summary": read(os.path.join(results, "summary_by_method.csv")),
           "paired": read(os.path.join(results, "paired_vs_mica.csv")), "zero": read(os.path.join(results, "sensitivity_failures_as_zero.csv")),
           "runtime": read(os.path.join(results, "runtime.csv")), "outcomes": read(os.path.join(results, "entry_outcomes.csv")),
           "metrics": read(os.path.join(results, "metrics_per_entry.csv")),
           "variants": read(os.path.join(results, "variants.csv")), "correlations": read(os.path.join(results, "correlations.csv")), "merged": read(os.path.join(results, "status_merged.csv"))}
    entries = read(os.path.join(data, "entries_2026.csv"))
    res["entries"] = entries
    af3 = read(os.path.join(data, "af3", "af3_results_check.csv"))
    dom = docking_passes(raw) if raw and os.path.isdir(raw) else pd.DataFrame()
    nd_dir = os.path.join(results, "nodock")                           # the same analysis for the sessions without docked AlphaFold input (LOG.md 2026-10-27)
    nd = {"completion": read(os.path.join(nd_dir, "completion.csv")), "summary": read(os.path.join(nd_dir, "summary_by_method.csv")), "paired": read(os.path.join(nd_dir, "paired_vs_mica.csv")),
          "metrics": read(os.path.join(nd_dir, "metrics_per_entry.csv"))}
    no_nd = "\\emph{The analysis without docked AlphaFold input has not been run.}\n"
    effect_tex, effect_macros = dock_effect(res["metrics"], nd["metrics"], dom, entries)
    files = []
    tables = {"tab_completion.tex": lambda: completion_table(res["completion"]), "tab_summary.tex": lambda: summary_table(res["summary"]),
              "tab_summary_strata.tex": lambda: summary_strata_table(res["summary"]), "tab_summary_all.tex": lambda: summary_all_table(res["summary"]),
              "tab_paired.tex": lambda: paired_table(res["paired"]), "tab_zero.tex": lambda: zero_table(res["zero"]),
              "tab_runtime.tex": lambda: runtime_table(res["runtime"], res["outcomes"]), "tab_per_entry.tex": lambda: per_entry_table(res["metrics"], res["outcomes"], entries),
              "tab_failures.tex": lambda: failures_table(res["outcomes"]), "tab_entries.tex": lambda: entries_table(entries, af3),
              "tab_docking.tex": lambda: docking_table(dom, entries), "tab_pins.tex": lambda: pins_table(raw) if raw else "",
              "tab_summary_modes.tex": lambda: summary_strata_table(res["summary"], MODE_GROUPS[1:]), "tab_paired_modes.tex": lambda: paired_table(res["paired"], "tm_score", MODE_GROUPS),
              "tab_methods_screen.tex": lambda: methods_screen_table(os.path.join(HERE, "..", "methods", "methods_table.csv")), "tab_pool_flow.tex": lambda: pool_flow_table(data),
              "tab_mica_modes.tex": lambda: mica_modes_table(raw, entries, res["metrics"], res["outcomes"]),
              "tab_summary_nodock.tex": lambda: summary_table(nd["summary"], dict(NAMES, mica="MICA, no docked input")) if not nd["summary"].empty else no_nd,
              "tab_paired_nodock.tex": lambda: paired_table(nd["paired"]) if not nd["paired"].empty else no_nd, "tab_dock_effect.tex": lambda: effect_tex,
              "tab_variants.tex": lambda: variants_table(res["variants"]), "tab_fns.tex": lambda: fns_table(res["summary"], res["paired"]),
              "tab_sessions.tex": lambda: sessions_table(raw), "tab_corr.tex": lambda: corr_table(res["correlations"])}
    for name, fn in tables.items():
        files.append(write(out, name, fn()))
    m = Macros()
    dataset_macros(m, data)
    result_macros(m, res)
    if not nd["summary"].empty:
        result_macros(m, nd, "Nd")
    for k, v in effect_macros.items():
        m.add("DockEffect" + k, v)
    docking_macros(m, dom)
    extra_macros(m, res, dom)
    session_macros(m, raw)
    commit_macros(m, raw)
    methods_macros(m, os.path.join(HERE, "..", "methods", "methods_table.csv"))
    if raw:
        environment_macros(m, raw)
    files.append(write(out, "numbers.tex", m.tex()))
    files.append(write(out, "numbers.json", json.dumps(m.items, indent=1, sort_keys=True) + "\n"))
    return files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--raw", default="results/raw")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="report/generated")
    args = p.parse_args(argv)
    files = build(args.results, args.raw, args.data, args.out)
    print("wrote %d files to %s" % (len(files), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
