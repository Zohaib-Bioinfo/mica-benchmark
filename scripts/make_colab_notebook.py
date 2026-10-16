#!/usr/bin/env python3
"""Build scripts/pull_pool_colab.ipynb from scripts/pull_pool.py.

The notebook carries a verbatim copy of pull_pool.py, so the one notebook file is all that has
to be uploaded to Colab (or Kaggle). Rebuild it after every change to pull_pool.py:

    python make_colab_notebook.py

The cell sources live in this file, not in the .ipynb, so the notebook is a build product.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
FREEZE = "2026-10-12"
ZIP_NAME = "pool_full_%s.zip" % FREEZE

INTRO = """\
# MICA benchmark, Phase 3a: pull the 2026 candidate pool

This notebook downloads every cryo-EM entry released from 2026-01-01 to %(freeze)s from RCSB (about 6,000 files,
no shortcuts), applies the file-based filters of `docs/SPEC.md` (single particle, 4.0 A or better, protein only,
at most 700 modeled residues, at most 3 unique chain sequences, at most 5%% UNK) and checks the EMDB release date
of each map. It then packs the results into one zip file and downloads it.

**What you do**

1. Click **Runtime -> Run all**. If Colab warns that the notebook was not authored by Google, click **Run anyway**.
2. Keep this tab open. Do not edit any code.
3. Within about 2 minutes, step 3 prints `canary OK`. If it prints `CANARY FAILED` instead, let it finish:
   step 4 still packs the log into a zip. Send me that zip. If any step shows a red error, copy all of its text and send it to me.
4. After 10 to 40 minutes, step 4 prints `RESULT: OK` and the zip `%(zip)s` downloads by itself.
   If no download starts: click the folder icon on the left, find the zip, click the three dots, then **Download**.
5. Attach the zip to the chat.

The run is repeatable. Entries are chosen by release date, so running it on any day gives the same pool.
""" % {"freeze": FREEZE, "zip": ZIP_NAME}

STEP1_MD = "## Step 1 of 4: set up (about 20 seconds)"
STEP1 = """\
import platform, shutil, sys

print("python", sys.version.split()[0], "|", platform.platform())
%pip install -q "gemmi>=0.6" requests
import gemmi, requests

print("gemmi", gemmi.__version__, "| requests", requests.__version__)
print("free disk: %.1f GB" % (shutil.disk_usage(".").free / 1e9))
"""

STEP2_MD = "## Step 2 of 4: write the script (a few seconds; it prints `Writing pull_pool.py`)"

STEP3_MD = ("## Step 3 of 4: run the pull (10 to 40 minutes)\n"
            "Expect `canary OK` within about 2 minutes. This step can be run again; it continues from its cache "
            "and retries only what failed.")
STEP3 = '''\
import hashlib, os, subprocess, sys, time

OUT = "pool_out_full"
FREEZE = "%(freeze)s"

os.makedirs(OUT, exist_ok=True)
with open("pull_pool.py", "rb") as f:
    print("script sha256:", hashlib.sha256(f.read()).hexdigest())

cmd = [sys.executable, "-u", "pull_pool.py", "--freeze", FREEZE, "--out", OUT]
procs = []


def run_once():
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    procs.append(p)
    for line in p.stdout:
        print(line, end="")
    return p.wait()


t0 = time.time()
rc = None
try:
    for attempt in (1, 2, 3):
        if attempt > 1:
            print("\\nSome entries could not be downloaded. Waiting 30 s, then retrying only those (attempt %%d of 3)." %% attempt)
            time.sleep(30)
        rc = run_once()
        if rc != 2:  # 2 means: finished, but some downloads failed
            break
except KeyboardInterrupt:
    for p in procs:
        if p.poll() is None:
            p.terminate()
            p.wait()
    rc = -2
    print("\\nStopped by you. Run this step again to continue where it stopped.")
with open(os.path.join(OUT, "exit_code.txt"), "w") as f:
    f.write("%%d\\n" %% rc)
print("\\nStep 3 finished with exit code %%d after %%.1f minutes (0 means complete)." %% (rc, (time.time() - t0) / 60))
''' % {"freeze": FREEZE}

STEP4_MD = "## Step 4 of 4: pack the results and download the zip"
STEP4 = '''\
import csv, json, os, zipfile

OUT = "pool_out_full"
FREEZE = "%(freeze)s"
code_path = os.path.join(OUT, "exit_code.txt")
if not os.path.exists(code_path):
    raise SystemExit("Step 3 has not run yet. Run step 3 first, then this step.")
with open(code_path) as f:
    rc = int(f.read().strip())

verdict = {
    0: "OK. The pool is complete.",
    2: "INCOMPLETE. Some entries could not be downloaded. Run step 3 again (it retries only those), then run this step again.",
    3: "STOPPED. The script could not read the real RCSB files (the canary check failed). The cause is on my side. Send me the zip.",
    4: "STOPPED. The RCSB search did not work. Check that this notebook has Internet access. If it does, send me the zip.",
    -2: "STOPPED by you. Run step 3 again, then this step.",
}
if rc < 0 and rc != -2:
    verdict[rc] = ("The script was killed by the system (probably out of memory). Run step 3 again: it continues from "
                   "its cache. If it happens twice, send me the zip.")
print("RESULT: " + verdict.get(rc, "UNEXPECTED exit code %%d. Send me the zip." %% rc))

names = ["pool_all_entries.csv", "pool_candidates.csv", "pool_flow.csv", "pool_candidate_sequences.fasta",
         "pool_ids.txt", "pool_ids_all_cryoem.txt", "search_weeks.csv", "run_info.json", "run.log",
         "errors.csv", "exit_code.txt"]
zip_name = "pool_full_%%s.zip" %% FREEZE if rc == 0 else "pool_full_%%s_exit%%d.zip" %% (FREEZE, rc)
folder = os.path.splitext(zip_name)[0]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    for name in names:
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            z.write(path, arcname=folder + "/" + name)
    if os.path.exists("pull_pool.py"):
        z.write("pull_pool.py", arcname=folder + "/pull_pool.py")
print("zip written: %%s (%%.1f MB)" %% (zip_name, os.path.getsize(zip_name) / 1e6))

flow_path = os.path.join(OUT, "pool_flow.csv")
if os.path.exists(flow_path):
    print("\\nEntries left after each filter:")
    with open(flow_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            print("  %%-18s left %%6s   removed %%6s" %% (r["step"], r["n_remaining"], r["n_removed_at_this_step"]))
info_path = os.path.join(OUT, "run_info.json")
if os.path.exists(info_path):
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    if "n_candidates" in info:
        print("\\ncandidates for the next step: %%s  (per stratum: %%s)" %% (info["n_candidates"], info.get("candidates_per_stratum")))
    print("peak memory: %%s MB | files skipped for size: %%s" %% (info.get("peak_memory_mb", "n/a"), info.get("n_file_size_skipped", "n/a")))
log_path = os.path.join(OUT, "run.log")
if os.path.exists(log_path):
    with open(log_path, encoding="utf-8") as f:
        flagged = [x.rstrip() for x in f if "WARNING" in x or "FAILED" in x]
    if flagged:
        print("\\nLines to look at:")
        for x in flagged[-10:]:
            print("  " + x)

print("\\nIf the download does not start by itself: click the folder icon on the left, find %%s, "
      "click the three dots, then Download." %% zip_name)
try:
    from google.colab import files
    files.download(zip_name)
except ImportError:
    print("This is not Colab, so there is no automatic download. The zip is at:", os.path.abspath(zip_name))
''' % {"freeze": FREEZE}


def build(script_path, out_path):
    with open(script_path, encoding="utf-8") as f:
        script = f.read()
    compile(script, script_path, "exec")  # fail here, not in Colab, if the script has a syntax error
    if not script.endswith("\n"):
        script += "\n"
    nb = new_notebook()
    nb.metadata = {
        "colab": {"name": "pull_pool_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    cells = [
        ("intro", new_markdown_cell(INTRO)),
        ("step1-title", new_markdown_cell(STEP1_MD)),
        ("step1", new_code_cell(STEP1)),
        ("step2-title", new_markdown_cell(STEP2_MD)),
        ("step2", new_code_cell("%%writefile pull_pool.py\n" + script)),
        ("step3-title", new_markdown_cell(STEP3_MD)),
        ("step3", new_code_cell(STEP3)),
        ("step4-title", new_markdown_cell(STEP4_MD)),
        ("step4", new_code_cell(STEP4)),
    ]
    for cid, cell in cells:
        cell["id"] = cid
        nb.cells.append(cell)
    nbformat.validate(nb)
    nbformat.write(nb, out_path)
    return hashlib.sha256(script.encode("utf-8")).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--script", default=os.path.join(HERE, "pull_pool.py"))
    p.add_argument("--out", default=os.path.join(HERE, "pull_pool_colab.ipynb"))
    args = p.parse_args(argv)
    digest = build(args.script, args.out)
    print("wrote %s (embedded script sha256 %s)" % (args.out, digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
