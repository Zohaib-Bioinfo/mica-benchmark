#!/usr/bin/env python3
"""Shared parts of the generators of the Phase 4 notebooks that came after the first MICA pilot (make_dockdiag_notebook.py,
make_modelangelo_notebook.py): the cell that adds the progress lines and the zip tools, the last cell, and the helpers that build and check a notebook.

The first pilot generator (make_pilot_notebook.py) is reused for the cells that are the same (machine check, Phenix, MICA environment,
the helpers of the input preparation); a few functions are cut out of its cells with `function_sources`, so there is one copy of the code that ran on Kaggle.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_repo_file(rel, override=None):
    """Text of a file of the repository (with a final newline); a .py file must compile."""
    with open(override or os.path.join(ROOT, rel), encoding="utf-8") as f:
        text = f.read()
    text = text if text.endswith("\n") else text + "\n"
    if rel.endswith(".py"):
        compile(text, rel, "exec")
    return text


def fill(text, mapping):
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text


def function_sources(source, names):
    """The source text of the top-level functions `names` of `source` (a filled cell), in the order given. A missing name is an error."""
    tree = ast.parse(source)
    found = {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = [n for n in names if n not in found]
    if missing:
        raise ValueError("function(s) not found in the cell: %s" % ", ".join(missing))
    return "\n\n\n".join(found[n] for n in names) + "\n"


def cut_before(source, marker):
    """The part of a cell before the line that starts with `marker`; the marker must occur exactly once."""
    if source.count(marker) != 1:
        raise ValueError("%r occurs %d times in the cell, once expected" % (marker, source.count(marker)))
    return source[:source.index(marker)].rstrip("\n") + "\n"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("%r occurs %d times in the cell, once expected" % (old, source.count(old)))
    return source.replace(old, new)


def file_cells(files, entries_path=None):
    """(list of %%writefile cells, dict path -> sha256) for [(path in the repo, path in the notebook), ...]."""
    cells, expected = [], {}
    for i, (src, target) in enumerate(files, 1):
        text = read_repo_file(src, entries_path if (entries_path and src == "data/entries_2026.csv") else None)
        expected[target] = sha256_text(text)
        cells.append(("file-%d" % i, new_code_cell("%%writefile " + target + "\n" + text)))
    return cells, expected


def assemble(out_path, nb_name, cells):
    """Builds, checks and writes a notebook from [(cell id, cell), ...]: every code cell compiles (a %%writefile cell was compiled when it was read),
    no @PLACEHOLDER@ is left, the notebook validates."""
    nb = new_notebook()
    nb.metadata = {"colab": {"name": nb_name, "provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    seen = set()
    for cid, cell in cells:
        if cid in seen:
            raise ValueError("cell id %s is used twice" % cid)
        seen.add(cid)
        cell["id"] = cid
        nb.cells.append(cell)
    for cell in nb.cells:
        if cell.cell_type == "code" and not cell.source.startswith("%%writefile"):
            compile(cell.source, "cell " + cell["id"], "exec")
        left = re.findall(r"@[A-Z_0-9]+@", cell.source)
        if left:
            raise ValueError("placeholder %s is left in cell %s" % (left[0], cell["id"]))
    nbformat.validate(nb)
    nbformat.write(nb, out_path)
    return nb


md = new_markdown_cell
code = new_code_cell

ROBUST_MD = """\
## Step 2b. Progress lines, a zip that is refreshed, and the last cell that works after a restart

This cell adds three things that the first pilot lacked: a progress line every few minutes while a program is silent, the file `state.json`, and a zip
(`@NOTEBOOK@_partial.zip` in the output files) that is written again after every stage and every ten minutes, so a stop in the middle never leaves nothing."""

ROBUST = r'''
sys.path.insert(0, os.path.join(WORK, "scripts"))
import nbtools

# Kaggle's kernel sets MPLBACKEND=module://matplotlib_inline.backend_inline. Every program started from this notebook inherits it, and `import matplotlib.pyplot`
# then fails in any environment without matplotlib-inline (ModelAngelo's, in the first pilot: the run stopped after 5 minutes). Agg needs nothing.
os.environ["MPLBACKEND"] = "Agg"

NOTEBOOK = "@NOTEBOOK@"
STATE_FILE = os.path.join(WORK, "state.json")
PARTIAL_ZIP = nbtools.partial_zip_path(OUT_DIR, NOTEBOOK)
PACK_LOCK = threading.Lock()
LAST_PARTIAL = {"t": 0.0}


def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(STATE, f, indent=1, default=str)
    except Exception as ex:
        log("could not save state.json: %s" % ex)


def pack_now(min_gap=0):
    """Writes the partial zip again (same name every time). min_gap: do nothing if the last one is younger than this many seconds."""
    with PACK_LOCK:
        if time.time() - LAST_PARTIAL["t"] < min_gap:
            return
        LAST_PARTIAL["t"] = time.time()
        try:
            save_state()
            nbtools.pack(WORK, PARTIAL_ZIP, NOTEBOOK + "_partial", secrets=SECRETS)
        except Exception as ex:
            log("the partial zip could not be written: %s" % ex)


BEAT = float(os.environ.get("MICA_BEAT", "120"))          # seconds between progress lines (the tests make it short)


def run_watched(cmd, label, watch=(), beat=None, zip_every=600, **kw):
    """run() with a progress line every `beat` seconds (time, load, free RAM, busiest processes, last line of the files in `watch`)."""
    hb = nbtools.Heartbeat(label, log, watch=watch, every=beat or BEAT, on_tick=lambda: pack_now(zip_every))
    hb.start()
    try:
        return run(cmd, **kw)
    finally:
        hb.stop()


def finish(complete, extra_info=None):
    """Writes run_info.json and the final zip; returns the path of the zip."""
    info = {"notebook": NOTEBOOK, "started_utc": STATE["started"], "finished_utc": nbtools.utc_now().isoformat(), "elapsed_minutes": round((time.time() - STATE["t0"]) / 60),
            "problem": STATE["problem"], "complete": complete, "platform": PLATFORM, "python": platform.python_version(), "os": platform.platform(), "gpu": STATE["gpu"],
            "cpu_cores": os.cpu_count(), "ram_gb": meminfo_gb(), "state": STATE}
    info.update(extra_info or {})
    with open(os.path.join(WORK, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    save_state()
    path, n, left_out = nbtools.final_pack(WORK, OUT_DIR, NOTEBOOK, complete, secrets=SECRETS)
    log("zip written: %s (%d files, %d MB)" % (path, n, os.path.getsize(path) // 1000000))
    if left_out:
        log("left out of the zip: %s" % "; ".join(left_out))
    return path


def cpu_facts():
    """How many processors the program sees and how many it may really use (a container can show more cores than its limit allows)."""
    import multiprocessing
    facts = {"os.cpu_count": os.cpu_count(), "multiprocessing.cpu_count": multiprocessing.cpu_count(),
             "usable (sched_getaffinity)": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "nproc": sh("nproc 2>&1"),
             "cgroup v2 cpu.max": sh("cat /sys/fs/cgroup/cpu.max 2>&1"), "cgroup v1 quota": sh("cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>&1"),
             "MICA's nproc formula min(max(int(0.6 * cpu_count), 1), 48)": min(max(int(0.6 * multiprocessing.cpu_count()), 1), 48)}
    with open(os.path.join(WORK, "cpu_facts.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join("%s: %s" % kv for kv in facts.items()) + "\n")
    log("processors: " + " | ".join("%s %s" % (k.split(" ")[0], v) for k, v in facts.items()))
    STATE["cpu_facts"] = facts


save_state()
cpu_facts()
log("progress lines, state.json and the partial zip (%s) are ready" % PARTIAL_ZIP)
'''

FINAL_MD = """\
## Last step. Pack the results

Run this cell last, also after a problem. It works after a restart of the kernel too: it then packs the files that are on disk."""

FINAL = r'''
import os, sys
try:
    finish
except NameError:                                  # fresh kernel: nothing of the earlier cells is left, only the files on disk
    work = None
    for cand in (os.environ.get("MICA_WORK"), "/kaggle/working/pilot", "/content/pilot", os.path.abspath("pilot")):
        if cand and os.path.isfile(os.path.join(cand, "scripts", "nbtools.py")):
            work = cand
            break
    if work is None:
        print("RECOVERY: the work folder was not found, so there is nothing to pack. Run the notebook from the top.")
    else:
        sys.path.insert(0, os.path.join(work, "scripts"))
        import nbtools
        nbtools.recover("@NOTEBOOK@")
else:
    zip_path = finish(bool(STATE["ok"] and STATE.get("complete")), @EXTRA_INFO@)
    if STATE["ok"] and STATE.get("complete"):
        log("\nRESULT: OK. Download %s (Kaggle: the output files of this version) and send it to me." % zip_path)
    elif STATE["ok"]:
        log("\nRESULT: NOT COMPLETE (%s). Send me %s; it shows where it stopped." % (STATE.get("stopped_at") or "a step did not finish", zip_path))
    else:
        log("\nRESULT: STOPPED (%s). Send me %s." % (STATE["problem"], zip_path))
    try:
        from google.colab import files
        files.download(zip_path)
    except Exception:
        log("(Kaggle: open the version, then Output, then download the zip. Colab: the file list on the left.)")
'''
