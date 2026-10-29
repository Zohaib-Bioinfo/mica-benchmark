#!/usr/bin/env python3
"""Figures of the report, drawn from the tables of analyze_results.py (nothing is typed in by hand):

    python3 scripts/make_figures.py --results results --out report/figures

Writes one PDF (vector, for the report) and one PNG per figure. Every figure has its table twin in the report or in its appendix (the CSV that it is drawn from is named in the
caption). Colours: the validated categorical palette of the dataviz reference, in a fixed order that follows the method (MICA is always blue), plus a marker shape per method,
so that the figures also read in grey print; outcomes use the status colours. Text is always in ink colours, never in the colour of a series.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Patch, Rectangle                    # noqa: E402

import analyze_results as A                                        # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
COLORS = {"mica": "#2a78d6", "modelangelo": "#eb6834", "emodelx": "#1baf7a", "emprot": "#eda100", "e3": "#e87ba4"}
MARKERS = {"mica": "o", "modelangelo": "s", "emodelx": "^", "emprot": "D", "e3": "v"}
SHORT = {"mica": "MICA", "modelangelo": "ModelAngelo", "emodelx": "EModelX(+AF)", "emprot": "EMProt", "e3": "E3-CryoFold"}
BLUES = ["#e8f1fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]     # sequential, light to dark
STATUS = {"model": "#0ca30c", "failed": "#d03b3b", "unscorable": "#ec835a", "not_run": "#c3c2b7"}
STATUS_LABEL = {"model": "model scored", "failed": "failed", "unscorable": "model missing or unusable", "not_run": "not run (no result)"}
STRATUM_LABEL = {"pooled": "all entries", "le3.0": "resolution ≤ 3.0 Å", "gt3.0_le4.0": "resolution > 3.0 to 4.0 Å"}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 8.5,
                         "axes.titlecolor": INK, "axes.titleweight": "regular", "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
                         "legend.fontsize": 8, "lines.solid_capstyle": "round"})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return name


def method_legend(fig, methods, y=1.0, ncol=None):
    handles = [Line2D([0], [0], marker=MARKERS[m], color="none", markerfacecolor=COLORS[m], markeredgecolor=SURFACE, markersize=7, label=SHORT[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(methods), handletextpad=0.3, columnspacing=1.4)


def tidy(ax, grid="y"):
    ax.tick_params(length=2.5, width=0.8)
    if grid:
        ax.grid(axis=grid, zorder=0)
        ax.set_axisbelow(True)


def read(results, name):
    path = os.path.join(results, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def value(row, column):
    v = row.get(column)
    return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else math.nan


# ------------------------------------------------------------------------------------------------------------------------------ figure 1: every entry, every method

def fig_heatmap(results, out):
    m = read(results, "metrics_per_entry.csv")
    if m.empty:
        return None
    entries = list(dict.fromkeys(m["pdb_id"]))
    methods = [x for x in A.METHODS if x in set(m["method"])]
    info = m.drop_duplicates("pdb_id").set_index("pdb_id")
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    n, k = len(entries), len(methods)
    fig, ax = plt.subplots(figsize=(1.55 + 0.92 * k, 0.9 + 0.225 * n))
    ax.set_xlim(0, k)
    ax.set_ylim(n, 0)
    strata = list(info.loc[entries, "stratum"])
    gaps = [i for i in range(1, n) if strata[i] != strata[i - 1]]
    for i, e in enumerate(entries):
        for j, meth in enumerate(methods):
            row = m[(m["pdb_id"] == e) & (m["method"] == meth)]
            if row.empty:
                continue
            row = row.iloc[0]
            st, tm = row["status"], value(row, "tm_score")
            x0, y0 = j + 0.04, i + 0.06
            if st == "model" and not math.isnan(tm):
                ax.add_patch(Rectangle((x0, y0), 0.92, 0.88, facecolor=cmap(min(max(tm, 0), 1)), edgecolor="none"))
                ax.text(j + 0.5, i + 0.52, "%.2f" % tm, ha="center", va="center", fontsize=6.8, color="#ffffff" if tm > 0.55 else INK)
            else:
                ax.add_patch(Rectangle((x0, y0), 0.92, 0.88, facecolor="#f3f2ee", edgecolor=AXIS, linewidth=0.5, hatch="////" if st in ("failed", "unscorable") else None))
                ax.text(j + 0.5, i + 0.52, {"failed": "failed", "unscorable": "no model", "not_run": "not run"}.get(st, st), ha="center", va="center", fontsize=6.2, color=INK2,
                        bbox=dict(facecolor="#f3f2ee", edgecolor="none", pad=0.6) if st in ("failed", "unscorable") else None)
    for g in gaps:
        ax.axhline(g, color=INK2, linewidth=0.9)
    ax.set_xticks([j + 0.5 for j in range(k)])
    ax.set_xticklabels([SHORT[x] for x in methods], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(["%s  %.1f Å" % (e, info.loc[e, "resolution_A"]) for e in entries], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.025, pad=0.02, aspect=40)
    cb.set_label("TM-score (normalised by the deposited model)", fontsize=7.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=7)
    return save(fig, out, "fig1_tm_score_per_entry")


# ------------------------------------------------------------------------------------------------------------------------------ figure 2: paired differences with intervals

def fig_forest(results, out, metric="tm_score", name="fig2_paired_differences", xlabel="Difference in TM-score (method − MICA)"):
    p = read(results, "paired_vs_mica.csv")
    if p.empty:
        return None
    p = p[p["metric"] == metric]
    groups = [g for g in ("pooled", "le3.0", "gt3.0_le4.0") if g in set(p["group"])]
    methods = [x for x in A.METHODS if x != A.REFERENCE and x in set(p["method"])]
    if not groups or not methods:
        return None
    lim = 0.1
    ok = p[p["group"].isin(groups) & p["method"].isin(methods)].dropna(subset=["diff_ci_low", "diff_ci_high"])
    if len(ok):
        lim = max(0.1, math.ceil(max(abs(ok["diff_ci_low"]).max(), abs(ok["diff_ci_high"]).max()) * 1.15 / 0.05) * 0.05)
    fig, axes = plt.subplots(1, len(groups), figsize=(2.55 * len(groups) + 0.9, 0.55 + 0.52 * len(methods)), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        ax.axvline(0, color=INK2, linewidth=0.9, zorder=1)
        for i, meth in enumerate(methods):
            r = p[(p["group"] == g) & (p["method"] == meth)]
            if r.empty:
                continue
            r = r.iloc[0]
            if r["n"] == 0 or math.isnan(r["mean_diff"]):
                ax.text(lim * 0.02, i, "no pairs", va="center", ha="left", fontsize=7, color=MUTED)
                continue
            if not math.isnan(r["diff_ci_low"]):
                ax.plot([r["diff_ci_low"], r["diff_ci_high"]], [i, i], color=COLORS[meth], linewidth=2.0, solid_capstyle="round", zorder=2)
            ax.scatter([r["mean_diff"]], [i], s=46, marker=MARKERS[meth], color=COLORS[meth], edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.text(lim * 0.98, i - 0.33, "n = %d" % r["n"], ha="right", va="center", fontsize=6.5, color=MUTED)
        ax.set_xlim(-lim, lim)
        ticks = np.linspace(-lim, lim, 5)                                          # five ticks, so that the labels of neighbouring panels do not touch
        ax.set_xticks(ticks)
        ax.set_xticklabels([("%g" % round(v, 2)).replace("-", "\u2212") for v in ticks])
        ax.set_ylim(len(methods) - 0.5, -0.65)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([SHORT[x] for x in methods])
        ax.set_title(STRATUM_LABEL.get(g, g), loc="left")
        tidy(ax, "x")
        if ax is axes[len(axes) // 2]:                                              # one label for the whole row, so that long labels do not run into each other
            ax.set_xlabel(xlabel, fontsize=7.5)
    fig.tight_layout()
    return save(fig, out, name)


# ------------------------------------------------------------------------------------------------------------------------------ figure 3: pair by pair

def fig_pairs(results, out, column="tm_score", name="fig3_tm_score_pairs", label="TM-score"):
    m = read(results, "metrics_per_entry.csv")
    if m.empty or column not in m.columns:
        return None
    w = m[m["status"] == "model"].pivot(index="pdb_id", columns="method", values=column)
    if A.REFERENCE not in w.columns:
        return None
    methods = [x for x in A.METHODS if x != A.REFERENCE and x in w.columns]
    fig, axes = plt.subplots(1, len(methods), figsize=(2.15 * len(methods) + 0.4, 2.55), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, meth in zip(axes, methods):
        d = w[[A.REFERENCE, meth]].dropna()
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.9, zorder=1)
        ax.scatter(d[A.REFERENCE], d[meth], s=30, marker=MARKERS[meth], color=COLORS[meth], edgecolor=SURFACE, linewidth=0.9, zorder=3)
        far = (d[meth] - d[A.REFERENCE]).abs().sort_values(ascending=False).head(3)
        offsets = [(5, -2), (5, -10), (-4, 6)]
        for k, e in enumerate([e for e in far.index if far[e] >= 0.3]):
            x, y = d.loc[e, A.REFERENCE], d.loc[e, meth]
            ax.annotate(e, (x, y), xytext=offsets[k], textcoords="offset points", fontsize=6, color=INK2, ha="left" if offsets[k][0] > 0 else "right")
        above, below = int((d[meth] > d[A.REFERENCE]).sum()), int((d[meth] < d[A.REFERENCE]).sum())
        ax.set_title("%s (n = %d)\n%d above, %d below the line" % (SHORT[meth], len(d), above, below), loc="left", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("MICA %s" % label, fontsize=7.5)
        tidy(ax, "both")
    axes[0].set_ylabel("method %s" % label, fontsize=7.5)
    fig.tight_layout()
    return save(fig, out, name)


# ------------------------------------------------------------------------------------------------------------------------------ figure 4: distributions

def strip_panels(m, metrics, out, name, ncols=2, width=6.9, row_h=1.9):
    methods = [x for x in A.METHODS if x in set(m["method"])]
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, row_h * nrows + 0.5), squeeze=False)
    rng = np.random.default_rng(7)
    for ax, (column, label, better) in zip(axes.flat, metrics):
        for j, meth in enumerate(methods):
            v = m[(m["method"] == meth) & (m["status"] == "model")][column].dropna().to_numpy(dtype=float)
            if not len(v):
                continue
            x = j + rng.uniform(-0.2, 0.2, size=len(v))
            ax.scatter(x, v, s=17, marker=MARKERS[meth], color=COLORS[meth], edgecolor=SURFACE, linewidth=0.6, alpha=0.95, zorder=3)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([j, j], [q1, q3], color=INK2, linewidth=1.2, zorder=4, solid_capstyle="butt")
            ax.plot([j - 0.3, j + 0.3], [med, med], color=INK, linewidth=2.0, zorder=5, solid_capstyle="butt")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([SHORT[x].replace("EModelX(+AF)", "EModelX\n(+AF)").replace("E3-CryoFold", "E3-\nCryoFold").replace("ModelAngelo", "Model-\nAngelo") for x in methods], fontsize=6.8)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        title = label + ("" if better is None else (" – higher is better" if better == "higher" else " – lower is better"))
        ax.set_title(title, loc="left", fontsize=7.8)
        tidy(ax, "y")
        if column in ("tm_score", "tm_score_fns"):
            ax.set_ylim(-0.02, 1.02)
        elif column in ("ca_match", "seq_match", "ca_quality", "seq_identity", "ca_found"):
            hi = max(100.0, float(np.nanmax(m[column].to_numpy(dtype=float))) if m[column].notna().any() else 100.0)
            ax.set_ylim(-2, hi * 1.03)
        elif column == "coverage":
            ax.axhline(1.0, color=AXIS, linewidth=0.9, zorder=1)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    method_legend(fig, methods, y=1.02)
    fig.text(0.0, -0.01, "Dot: one entry. Black bar: median. Grey bar: interquartile range. Only entries with a scored model are drawn.", fontsize=6.5, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, out, name)


def fig_distributions(results, out):
    m = read(results, "metrics_per_entry.csv")
    if m.empty:
        return []
    m = m.rename(columns={"ca_quality_score": "ca_quality", "residue_coverage": "coverage"})
    main = [x for x in A.METRICS if x[0] in ("tm_score", "ca_match", "seq_match", "ca_quality")]
    allm = [x for x in A.METRICS]
    return [strip_panels(m, main, out, "fig4_distributions", ncols=2, row_h=2.0), strip_panels(m, allm, out, "figA1_all_metrics", ncols=2, row_h=1.75)]


# ------------------------------------------------------------------------------------------------------------------------------ figure 5: outcomes

def fig_outcomes(results, out):
    c = read(results, "completion.csv")
    if c.empty:
        return None
    c = c[c["group"] == "pooled"].set_index("method")
    methods = [x for x in A.METHODS if x in c.index]
    n = int(c["n_entries"].max())
    fig, ax = plt.subplots(figsize=(6.4, 0.5 + 0.42 * len(methods) + 0.35))
    for i, meth in enumerate(methods):
        left = 0
        for key, col in (("model", "n_model"), ("failed", "n_failed"), ("unscorable", "n_unscorable"), ("not_run", "n_not_run")):
            k = int(c.loc[meth, col])
            if k:
                ax.barh(i, k - 0.08, left=left, height=0.58, color=STATUS[key], edgecolor="none")
                ax.text(left + k / 2 - 0.04, i, str(k), ha="center", va="center", fontsize=7.5, color="#ffffff" if key in ("model", "failed") else INK)
            left += k
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([SHORT[x] for x in methods])
    ax.set_ylim(len(methods) - 0.4, -0.6)
    ax.set_xlim(0, n)
    ax.set_xticks(range(0, n + 1, 5))
    ax.set_xlabel("entries (of %d)" % n, fontsize=7.5)
    tidy(ax, None)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=STATUS[k], label=STATUS_LABEL[k]) for k in ("model", "failed", "unscorable", "not_run")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1.2), ncol=4, handlelength=1.0, columnspacing=1.2, fontsize=7)
    fig.tight_layout()
    return save(fig, out, "fig5_outcomes")


# ------------------------------------------------------------------------------------------------------------------------------ figure 6: runtime

def fig_runtime(results, out):
    o = read(results, "entry_outcomes.csv")
    if o.empty:
        return None
    o = o[o["outcome"].isin(["model", "failed", "unscorable"]) & o["runtime_s"].notna()]
    methods = [x for x in A.METHODS if x in set(o["method"])]
    if not methods:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    rng = np.random.default_rng(3)
    for j, meth in enumerate(methods):
        v = o[o["method"] == meth]["runtime_s"].to_numpy(dtype=float) / 60.0
        v = v[v > 0]
        if not len(v):
            continue
        ax.scatter(j + rng.uniform(-0.2, 0.2, size=len(v)), v, s=17, marker=MARKERS[meth], color=COLORS[meth], edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=INK, linewidth=2.0, zorder=5, solid_capstyle="butt")
        ax.text(j + 0.34, np.median(v), "%.0f min" % np.median(v), va="center", ha="left", fontsize=6.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([SHORT[x] for x in methods], fontsize=7.5)
    ax.set_xlim(-0.6, len(methods) - 0.1)
    ax.set_ylabel("run time per entry (minutes, log scale)", fontsize=7.5)
    tidy(ax, "y")
    fig.tight_layout()
    return save(fig, out, "fig6_runtime")


# ------------------------------------------------------------------------------------------------------------------------------ appendix: fragmentation and resolution

def fig_fragmentation(results, out):
    m = read(results, "metrics_per_entry.csv")
    if m.empty or "tm_score_fns" not in m.columns:
        return None
    m = m[m["status"] == "model"]
    methods = [x for x in A.METHODS if x in set(m["method"])]
    fig, axes = plt.subplots(1, len(methods), figsize=(1.9 * len(methods) + 0.5, 2.3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, meth in zip(axes, methods):
        d = m[m["method"] == meth][["tm_score", "tm_score_fns"]].dropna()
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.9, zorder=1)
        ax.scatter(d["tm_score"], d["tm_score_fns"], s=22, marker=MARKERS[meth], color=COLORS[meth], edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.set_title(SHORT[meth], loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("TM-score, chain by chain", fontsize=7)
        tidy(ax, "both")
    axes[0].set_ylabel("TM-score, fragment-agnostic", fontsize=7)
    fig.tight_layout()
    return save(fig, out, "figA2_fragmentation")


def fig_resolution(results, out):
    m = read(results, "metrics_per_entry.csv")
    if m.empty:
        return None
    m = m[m["status"] == "model"]
    methods = [x for x in A.METHODS if x in set(m["method"])]
    fig, axes = plt.subplots(1, len(methods), figsize=(1.9 * len(methods) + 0.5, 2.3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, meth in zip(axes, methods):
        d = m[m["method"] == meth][["resolution_A", "tm_score"]].dropna()
        ax.scatter(d["resolution_A"], d["tm_score"], s=22, marker=MARKERS[meth], color=COLORS[meth], edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.set_title(SHORT[meth], loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("resolution (Å)", fontsize=7)
        tidy(ax, "y")
    axes[0].set_ylabel("TM-score", fontsize=7)
    fig.tight_layout()
    return save(fig, out, "figA3_resolution")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--results", default=os.path.join(here, "..", "results"))
    p.add_argument("--out", default=os.path.join(here, "..", "report", "figures"))
    args = p.parse_args(argv)
    setup()
    made = []
    for f in (fig_heatmap, fig_forest, fig_pairs, fig_distributions, fig_outcomes, fig_runtime, fig_fragmentation, fig_resolution):
        r = f(args.results, args.out)
        made += [x for x in (r if isinstance(r, list) else [r]) if x]
    made.append(fig_forest(args.results, args.out, metric="tm_score_fns", name="figA4_paired_differences_fragment_agnostic", xlabel="Difference in fragment-agnostic TM-score (method − MICA)"))
    made = [x for x in made if x]
    print("figures: " + ", ".join(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
