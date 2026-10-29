#!/usr/bin/env python3
"""Build scripts/dockdiag.ipynb: a CPU-only test of the Phenix docking step of MICA on the pilot entry (default 26DK).

Why it exists. In the first pilot, MICA's program 2 (`utils/dock_in_map.py`, which calls `phenix.dock_in_map ... quick=False nproc=2 ...`) did not finish
the first domain of 26DK (113 residues, a 160^3 map) in three hours, and its Phenix log was lost. A benchmark that needs hours per domain cannot run on free
machines for 20 entries, so this notebook finds out (1) what MICA's call really does, with the Phenix log kept and a progress line every two minutes,
(2) which faster setting gives the same placement, and (3) whether the scoring commands (US-align, phenix.chain_comparison, phenix.map_model_cc) read real
output correctly.

What it does. Steps 1 to 5 are the cells of the first pilot (machine check, Phenix, MICA's Python 3.8 environment, MICA at its pinned commit, US-align);
the MICA weights are not needed and not downloaded. Step 6 prepares the entry and runs MICA's program 1 (domain split, on the CPU). Step 7 docks the first
domain into the same processed map that MICA builds, with the settings of `DOCK_PLAN` (each with a time limit, the Phenix log saved), step 8 scores every docked model.

    python make_dockdiag_notebook.py            # writes scripts/dockdiag.ipynb

Run it on Kaggle with the accelerator off, as "Save & Run All (Commit)"; about 2 hours.

Version 2 (`--version v2`, scripts/dockdiag2.ipynb, written on 2026-10-22). The first run (v1) showed that no docking test finished in 20 minutes on the 265 A cube of 26DK, not
even Phenix's own quick search: Phaser gets 915,139 reflections (P1, 3.5 A) and after the "VRMS value" line the Phenix log stays silent while every process runs at 100% of
a core. Version 2 asks how long it really takes and what makes it faster: group 1 searches at a lower resolution (10 A and 6 A, quick or not, 45 minutes each, four tests at once on
one core each), group 2 runs MICA's own call and two Phenix alternatives at the real resolution with a limit of 4.5 hours (about 6 hours in all).
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_pilot_notebook as P  # noqa: E402
import nbkit  # noqa: E402

NOTEBOOK = "dockdiag"
NB_NAME = "dockdiag.ipynb"
FILES = (("scripts/nbtools.py", "scripts/nbtools.py"),
         ("scripts/mica_inputs.py", "scripts/mica_inputs.py"),
         ("scripts/evaluate_models.py", "scripts/evaluate_models.py"),
         ("data/af3/af3_jobs.csv", "data/af3/af3_jobs.csv"),
         ("data/af3/af3_files_sha256.txt", "data/af3/af3_files_sha256.txt"),
         ("data/entries_2026.csv", "data/entries_2026.csv"),
         ("env/mica_requirements.txt", "env/mica_requirements.txt"))
NET_HOSTS = (("PyPI", "https://pypi.org/simple/uv/", True),
             ("GitHub", "https://github.com/jianlin-cheng/MICA", True),
             ("EMDB API", "https://www.ebi.ac.uk/emdb/api/entry/80558", False),
             ("EBI FTP", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-80558/header/emd-80558.xml", False),
             ("RCSB", "https://files.rcsb.org/header/26DK.cif", False),
             ("Phenix", "https://phenix-online.org/download", False))

# The docking tests. group: tests of one group run at the same time (each with its own `nproc`), the groups one after the other.
# set: parameters of phenix.dock_in_map that replace MICA's (MICA's call: nproc=2 quick=False resolution=<map resolution> dock_chains_individually=True).
# cap_min: a test that takes longer is stopped (and counts as "did not finish").
# after_failure_of: the test runs only if that one gave no docked model; needs_a_success_of: only if one of those gave a docked model.
DOCK_PLAN = [
    {"name": "A_quick_4cpu", "group": 1, "set": {"quick": "True", "nproc": 4}, "cap_min": 20,
     "note": "Phenix default search (quick=True), 4 processes (all the processors of the machine)"},
    {"name": "A2_quick_1cpu", "group": 2, "set": {"quick": "True", "nproc": 1}, "cap_min": 20, "after_failure_of": "A_quick_4cpu",
     "note": "only if the first test gave no model: quick=True with a single process"},
    {"name": "B_mica_exact", "group": 3, "set": {}, "cap_min": 90, "needs_a_success_of": ["A_quick_4cpu", "A2_quick_1cpu"],
     "note": "MICA's own call: quick=False, nproc=2, the resolution of the map"},
    {"name": "C_quick_false_res6", "group": 3, "set": {"resolution": 6.0}, "cap_min": 90, "needs_a_success_of": ["A_quick_4cpu", "A2_quick_1cpu"],
     "note": "MICA's call, but searching at 6 A (the Phenix documentation: the search is quicker at lower resolution)"},
]

DOCK_PLAN_V2 = [
    {"name": "L1_mica_res10", "group": 1, "set": {"resolution": 10.0, "nproc": 1}, "cap_min": 45,
     "note": "MICA's call (quick=False), searching at 10 A (about 39,000 reflections), one process"},
    {"name": "L2_mica_res6", "group": 1, "set": {"resolution": 6.0, "nproc": 1}, "cap_min": 45,
     "note": "MICA's call (quick=False), searching at 6 A (about 180,000 reflections), one process"},
    {"name": "L3_quick_res10", "group": 1, "set": {"resolution": 10.0, "nproc": 1, "quick": "True"}, "cap_min": 45,
     "note": "Phenix default search (quick=True) at 10 A, one process"},
    {"name": "L4_quick_res6", "group": 1, "set": {"resolution": 6.0, "nproc": 1, "quick": "True"}, "cap_min": 45,
     "note": "Phenix default search (quick=True) at 6 A, one process"},
    {"name": "F1_mica_exact", "group": 2, "set": {}, "cap_min": 270,
     "note": "MICA's own call: quick=False, nproc=2, the resolution of the map (915,139 reflections)"},
    {"name": "F2_quick_1cpu", "group": 2, "set": {"quick": "True", "nproc": 1}, "cap_min": 270,
     "note": "Phenix default search (quick=True), one process, the resolution of the map"},
    {"name": "F3_no_mr_quick", "group": 2, "set": {"quick": "True", "nproc": 1, "dock_with_mr": "False"}, "cap_min": 270,
     "note": "Phenix's other engine (dock_with_mr=False: Fourier convolution searches in boxes), quick=True, one process"},
]
PLANS = {"v1": DOCK_PLAN, "v2": DOCK_PLAN_V2}
VERSIONS = {"v1": {"notebook": "dockdiag", "nb_name": "dockdiag.ipynb"}, "v2": {"notebook": "dockdiag2", "nb_name": "dockdiag2.ipynb"}}

INTRO_V2 = """\
# MICA benchmark: how long does the Phenix docking of MICA really take? (CPU test 2 on entry @ENTRY_LIST@)

The first docking test (2026-10-19) did not finish in 20 minutes with any setting: Phaser gets 915,139 reflections from the 265 A cube of this map and the Phenix log
stays silent while every process runs at full speed. This notebook gives the docking much more time, and finds out what makes it faster.

1. checks the machine, installs Phenix from your download link (user name and password from Secrets, never saved in a file),
2. builds MICA's Python 3.8 environment and fetches MICA (no weights needed here),
3. prepares the entry (your AlphaFold Server prediction, the EMDB map, the deposited model) and runs MICA's program 1 (domain split),
4. **group 1 (up to 45 minutes)**: four docking tests of the first domain at the same time, searching at 10 A and 6 A (fewer reflections), with MICA's search and with Phenix's quick search,
5. **group 2 (up to 4.5 hours)**: three tests at the real resolution at the same time: MICA's own call, Phenix's quick search, and Phenix's other engine,
6. scores every docked model (US-align, `phenix.chain_comparison`, `phenix.map_model_cc`) and packs one zip.

**What you do (Kaggle, background mode)**: accelerator **None**, Internet **On**, the two Phenix secrets attached, the private AlphaFold dataset added as input, then
*Save Version -> Save & Run All (Commit)*. It runs on Kaggle's servers; you can close the tab and your Wi-Fi does not matter. About 6 hours in all (the setup takes about 40 minutes).
When the version shows *Succeeded*, open it and download the zip `dockdiag2_result_....zip` from its Output. Send it to me.
**Do not stop or cancel the run**: a stopped run keeps nothing. (If group 1 shows a docked model after 10 minutes in the log, that is good news, not a reason to stop.)

**Phenix** is licensed to you personally. Do not share the installer, the password, or this notebook with the secrets attached.
"""

INTRO = """\
# MICA benchmark: why is the Phenix docking of MICA so slow? (CPU test on entry @ENTRY_LIST@)

In the first pilot the docking step of MICA did not finish in three hours. This notebook repeats only that step, on the CPU, with the
Phenix log kept and a progress line every two minutes, and tries faster settings next to MICA's own. It then scores every docked model.

1. checks the machine, installs Phenix from your download link (user name and password from Secrets, never saved in a file),
2. builds MICA's Python 3.8 environment and fetches MICA (no weights needed here),
3. prepares the entry (your AlphaFold Server prediction, the EMDB map, the deposited model) and runs MICA's program 1 (domain split),
4. docks the first domain with several settings, one test after the other (the plan is in step 7), each with a time limit,
5. scores every docked model (US-align, `phenix.chain_comparison`, `phenix.map_model_cc`) and packs one zip.

**What you do (Kaggle, background mode)**: accelerator **None**, Internet **On**, the two Phenix secrets attached, the private AlphaFold dataset added as input, then
*Save Version -> Save & Run All (Commit)*. It runs on Kaggle's servers; you can close the tab and your Wi-Fi does not matter. About 2 hours.
When the version shows *Succeeded*, open it and download the zip `dockdiag_result_....zip` from its Output. Send it to me.
**Do not stop or cancel the run** unless it is stuck for more than 4 hours: a stopped run keeps nothing.

**Phenix** is licensed to you personally. Do not share the installer, the password, or this notebook with the secrets attached.
"""

STEP4_MD = """\
## Step 4. Python 3.8 environment for MICA

MICA's domain split (program 1) runs in MICA's own Python 3.8 environment (PyTorch 1.8.1). It is built with `uv` from PyPI wheels with the exact pins
of `env/mica_requirements.txt` (171 packages), as in the first pilot. No GPU is needed in this notebook, so the GPU test is skipped."""

STEP5_MD = """\
## Step 5. MICA code and US-align

MICA is fetched at its pinned commit (a hash, so it cannot change under us); the domain-splitting weights of Merizo are part of that repository.
The big MICA weights (Zenodo) are not downloaded: program 1 and the docking do not use them. US-align is compiled from its pinned commit."""

STEP3B_MD = """\
## Step 3b. What does this Phenix say about `dock_in_map`?

Saves the parameter list and the help text of `phenix.dock_in_map` of the installed version (file `phenix_dock_in_map_help.txt`), so the settings of the tests
below can be compared with what this Phenix version really has."""
STEP3B = r'''
def phenix_dock_help():
    out = []
    for flag in ("--show_defaults", "--help"):
        try:
            r = subprocess.run(["bash", "-c", 'source "$0" && phenix.dock_in_map "$1" 2>&1 | head -400', STATE["phenix_env"], flag],
                               capture_output=True, text=True, errors="replace", timeout=600)
            out.append("==== phenix.dock_in_map %s (exit code %s)\n%s" % (flag, r.returncode, r.stdout + r.stderr))
        except subprocess.TimeoutExpired:
            out.append("==== phenix.dock_in_map %s: no answer within 10 minutes\n" % flag)
    text = "\n".join(out)
    with open("phenix_dock_in_map_help.txt", "w", encoding="utf-8") as f:
        f.write(text)
    for key in ("quick", "nproc", "target_boxes", "run_in_boxes", "resolution", "dock_with_mr", "verbose"):
        log("  dock_in_map parameter %-13s %s" % (key, "listed" if re.search(r"^\s*%s\b" % key, text, re.M) else "not seen in the help text"))


if STATE["ok"]:
    phenix_dock_help()
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

STEP6_MD = """\
## Step 6. Prepare the entry and run MICA's program 1 (domain split)

The same preparation as in the first pilot (AlphaFold Server files with the hashes checked, map from EMDB, deposited model from RCSB, contour level and
resolution from MICA's own `emdb_extractor.py`), then MICA's program 1 on the CPU. Its output, the domain files, are the search models of the docking."""
STEP6_NEW = r'''

DOCK = {}                      # what the next steps need: the files and numbers of the entry


def program_1(entry, folder):
    fasta, af3 = os.path.join(folder, entry.lower() + ".fasta"), os.path.join(folder, "AF3_results")
    logf = os.path.join(WORK, "logs", "%s_1_process_af3.log" % entry)
    cmd = [ENV_PY, "utils/process_AF3_results.py", "-f", fasta, "-a", af3, "-d", "cpu"]
    log("\n=== MICA program 1: %s" % " ".join(cmd[1:]))
    t0 = time.time()
    rc, last, timed_out = run_watched(cmd, "MICA program 1", watch=[logf], env=MICA_ENV, cwd=MICA_DIR, log_file=logf, timeout=TIMEOUTS["1_process_af3"], every=1)
    log("program 1 finished: exit code %s, %d s" % (rc, time.time() - t0))
    domains = sorted(glob.glob(os.path.join(folder, "AF3_domains", "*.pdb")))
    return rc, timed_out, domains, last


def prepare_and_split():
    entry = ENTRIES[0]
    res = {"entry": entry, "status": "failed", "failed_step": "", "note": "", "steps": {}, "models": []}
    STATE["entries"][entry] = res
    row = {r["pdb_id"]: r for r in read_csv("data/entries_2026.csv")}.get(entry)
    if row is None:
        fail("%s is not in data/entries_2026.csv" % entry)
        return
    log("\n" + "=" * 72 + "\nENTRY %s (EMDB %s, %s A, %s modeled residues)\n" % (entry, row["emdb_id"], row["resolution_A"], row["modeled_residues"]) + "=" * 72)
    try:
        folder, map_path, native, contour, resolution = prepare_entry(entry, row, res)
    except Exception as ex:
        res.update(failed_step="inputs", note=str(ex)[:500])
        fail("the inputs of %s could not be prepared: %s" % (entry, ex))
        return
    rc, timed_out, domains, last = program_1(entry, folder)
    if rc != 0 or timed_out or not domains:
        res.update(failed_step="1_process_af3", note=("timeout" if timed_out else "exit code %s" % rc) + ("" if domains else "; no domain file was written"))
        log("the last lines of program 1:\n" + "\n".join(last[-15:]))
        fail("MICA's program 1 gave no domain file (%s). Send me the zip." % res["note"])
        return
    keep = os.path.join(WORK, "dock_inputs")
    os.makedirs(keep, exist_ok=True)
    for p in glob.glob(os.path.join(folder, "AF3_domains", "*")):
        shutil.copy(p, keep)
    sizes = {os.path.basename(d): sum(1 for ln in open(d, errors="replace") if ln.startswith("ATOM") and ln[12:16].strip() == "CA") for d in domains}
    log("domain files of program 1 (C-alpha atoms each): %s" % sizes)
    DOCK.update(entry=entry, row=row, folder=folder, map=map_path, native=native, contour=contour, resolution=resolution,
                fasta=os.path.join(folder, entry.lower() + ".fasta"), search_model=domains[0], domains=domains)
    STATE["dock_inputs"] = {"entry": entry, "contour": contour, "resolution": resolution, "search_model": os.path.basename(domains[0]), "domain_calpha_counts": sizes,
                            "map_sha256": res.get("map_sha256")}
    log("the first domain, %s, is the search model of the docking tests (MICA docks the domains in this order)" % os.path.basename(domains[0]))


if STATE["ok"] and STATE["usalign"]:
    try:
        prepare_and_split()
    except Exception:
        import traceback
        fail("unexpected error while preparing the entry: " + traceback.format_exc()[-500:])
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

STEP7_MD = """\
## Step 7. The docking tests

The map is processed exactly as MICA does it (voxels below the contour level set to 0), the search model is the first domain of program 1, and
`phenix.dock_in_map` is called with MICA's arguments except for the setting that a test changes. Each test has a time limit; tests of one group run at the
same time. The Phenix log of each test is kept (`dock/<test>/phenix.log`), and a progress line is printed every two minutes with the last line of that log."""
STEP7 = r'''
DOCK_PLAN = @PLAN@
DOCK_DIR = os.path.join(WORK, "dock")
RESULTS = {}
PROCESS_MAP = r"""
import sys, mrcfile, numpy as np
src, dst, level = sys.argv[1], sys.argv[2], float(sys.argv[3])
with mrcfile.open(src, mode="r") as mrc:
    data, voxel_size, origin = mrc.data.copy(), mrc.voxel_size, mrc.header.origin
clipped = np.where(data < level, 0, data)
with mrcfile.new(dst, overwrite=True) as mrc:
    mrc.set_data(clipped.astype(np.float32))
    mrc.voxel_size = voxel_size
    mrc.header.origin = origin
print("processed map: %d of %d voxels are above zero; grid %s; voxel size %.3f A" % (int((clipped > 0).sum()), clipped.size, data.shape, float(voxel_size.x)))
"""


def make_processed_map():
    os.makedirs(DOCK_DIR, exist_ok=True)
    processed = os.path.join(DOCK_DIR, "initial_processed.mrc")
    rc, last, _ = run([ENV_PY, "-c", PROCESS_MAP, DOCK["map"], processed, str(DOCK["contour"])], env=MICA_ENV)
    if rc != 0 or not os.path.exists(processed):
        raise RuntimeError("the processed map could not be written: " + " | ".join(last[-3:]))
    return processed


def dock_kv(spec, out_dir, processed):
    kv = {"search_model": DOCK["search_model"], "map_file": processed, "nproc": 2, "quick": "False", "resolution": DOCK["resolution"],
          "pdb_out": os.path.join(out_dir, spec["name"] + "_docked.pdb"), "dock_chains_individually": "True", "sequence": DOCK["fasta"]}
    kv.update(spec["set"])
    return kv


def dock_command(kv):
    return ["bash", "-c", 'source "$0" && exec "$@"', STATE["phenix_env"], "phenix.dock_in_map"] + ["%s=%s" % (k, v) for k, v in kv.items()]


def run_variant(spec, processed):
    name = spec["name"]
    out_dir = os.path.join(DOCK_DIR, name)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    kv = dock_kv(spec, out_dir, processed)
    logf = os.path.join(out_dir, "phenix.log")
    rec = {"name": name, "note": spec.get("note", ""), "cap_min": spec["cap_min"], "started_utc": nbtools.utc_now().isoformat(), "seconds": None, "rc": None,
           "timed_out": False, "docked": False, "skipped": False, "log_lines": 0,
           "parameters": " ".join("%s=%s" % (k, v) for k, v in kv.items() if k not in ("search_model", "map_file", "pdb_out", "sequence"))}
    t0 = time.time()
    for attempt, extra in enumerate(({"verbose": "True"}, {})):
        if os.path.exists(logf):
            os.remove(logf)
        cmd = dock_command(dict(kv, **extra))
        log("\n=== docking test %s%s\n    %s" % (name, " (with verbose=True, for more lines in the Phenix log)" if extra else "", " ".join(cmd[4:])))
        t1 = time.time()
        rc, last, timed_out = run_watched(cmd, "dock " + name, watch=[logf, os.path.join(out_dir, "run_mr", "phaser_input.log")], log_file=logf, timeout=spec["cap_min"] * 60,
                                          quiet=True, cwd=out_dir, every=1)
        text = open(logf, errors="replace").read() if os.path.exists(logf) else ""
        if attempt == 0 and rc != 0 and not timed_out and time.time() - t1 < 90 and "verbose" in text.lower():
            log("    Phenix did not accept verbose=True (%s); the test is started again without it" % ((last or [""])[-1][:120]))
            continue
        break
    rec.update(seconds=round(time.time() - t0), rc=rc, timed_out=timed_out, log=logf, log_lines=text.count("\n"))
    mr_log = os.path.join(out_dir, "run_mr", "phaser_input.log")
    m = re.search(r"Number of Reflections in Selected Resolution Range:\s*(\d+)", open(mr_log, errors="replace").read()) if os.path.exists(mr_log) else None
    rec["reflections"] = int(m.group(1)) if m else ""
    docked = kv["pdb_out"]
    rec["docked"] = bool(not timed_out and os.path.isfile(docked) and os.path.getsize(docked) > 100)
    rec["pdb"] = docked if rec["docked"] else ""
    how = "STOPPED at its time limit of %g min" % spec["cap_min"] if timed_out else "exit code %s" % rc
    log("docking test %s finished after %d s: %s; docked model %s. Last lines of the Phenix log:\n    %s" % (
        name, rec["seconds"], how, "written" if rec["docked"] else "NOT written", "\n    ".join(text.splitlines()[-12:])))
    return rec


def run_group(specs, processed):
    def work(spec):
        try:
            RESULTS[spec["name"]] = run_variant(spec, processed)
        except Exception:
            import traceback
            RESULTS[spec["name"]] = {"name": spec["name"], "error": traceback.format_exc()[-500:], "docked": False, "skipped": False}
            log("docking test %s stopped by an error:\n%s" % (spec["name"], traceback.format_exc()))
    threads = [threading.Thread(target=work, args=(s,), daemon=True) for s in specs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_plan():
    processed = make_processed_map()
    for g in sorted({s["group"] for s in DOCK_PLAN}):
        specs = []
        for s in DOCK_PLAN:
            if s["group"] != g:
                continue
            earlier = s.get("after_failure_of")
            needs = s.get("needs_a_success_of")
            if earlier and RESULTS.get(earlier, {}).get("docked"):
                log("\ndocking test %s is skipped: %s already gave a docked model" % (s["name"], earlier))
                RESULTS[s["name"]] = {"name": s["name"], "skipped": True, "docked": False, "note": s.get("note", "")}
            elif needs and not any(RESULTS.get(n, {}).get("docked") for n in needs):
                log("\ndocking test %s is skipped: none of %s gave a docked model, so a longer test would not help" % (s["name"], ", ".join(needs)))
                RESULTS[s["name"]] = {"name": s["name"], "skipped": True, "docked": False, "note": s.get("note", "")}
            else:
                specs.append(s)
        if specs:
            log("\n--- docking tests of group %s (run at the same time): %s" % (g, ", ".join(s["name"] for s in specs)))
            run_group(specs, processed)
        STATE["dock_results"] = RESULTS
        pack_now()


if STATE["ok"] and DOCK:
    try:
        run_plan()
    except Exception:
        import traceback
        fail("unexpected error in the docking tests: " + traceback.format_exc()[-500:])
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

STEP8_MD = """\
## Step 8. Score every docked model

Each docked model is compared with the deposited model by US-align and by `phenix.chain_comparison`, and its fit to the map is measured by
`phenix.map_model_cc` (CC_mask; the deposited model gets the same call as the reference). The Calpha positions of the tests are compared with each other without
superposition: the tests start from the same search model, so equal placements give a distance near 0. The raw output of every call is kept in `scoring/`."""
STEP8_FUNCS = r'''
CC_RE = re.compile(r"\bCC_mask\b[\s:=|,()\[\]]*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.I)
RESULT_COLUMNS = ["name", "parameters", "cap_min", "seconds", "reflections", "rc", "timed_out", "docked", "skipped", "tm_score", "ca_rmsd", "aligned_ca_length", "pred_residues", "seq_identity",
                  "cc_mask", "chain_comparison", "ca_distance_to_reference", "reference", "note"]


def cc_mask(model, map_path, resolution, out_txt):
    """phenix.map_model_cc with the call that worked in the selection run (docs/LOG.md, 2026-10-14). Returns (CC_mask or None, status)."""
    if os.path.exists(out_txt):
        os.remove(out_txt)
    cmd = ["bash", "-c", 'source "$0" && phenix.map_model_cc "$1" "$2" resolution="$3" ignore_symmetry_conflicts=True --force', STATE["phenix_env"], model, map_path, str(resolution)]
    rc, last, timed_out = run(cmd, log_file=out_txt, timeout=900, quiet=True)
    text = open(out_txt, errors="replace").read() if os.path.exists(out_txt) else ""
    m = CC_RE.search(text)
    log("phenix.map_model_cc (%s): %s" % (os.path.basename(model), ("CC_mask " + m.group(1)) if m else "no CC_mask; last lines: " + " | ".join(last[-3:])))
    return (float(m.group(1)) if m else None), ("timeout" if timed_out else "ok" if rc == 0 else "failed (exit code %s)" % rc)


def ca_positions(path):
    pos = {}
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA":
                try:
                    pos[(line[21], line[22:27].strip())] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    pass
    return pos


def ca_distance(a, b):
    """Root mean square distance of the Calpha atoms that both files have (same chain, same residue number), without any superposition; (value, number of atoms)."""
    pa, pb = ca_positions(a), ca_positions(b)
    keys = [k for k in pa if k in pb]
    if not keys:
        return None, 0
    total = sum(sum((x - y) ** 2 for x, y in zip(pa[k], pb[k])) for k in keys)
    return round((total / len(keys)) ** 0.5, 3), len(keys)


def score_docking():
    row, native, folder = DOCK["row"], DOCK["native"], DOCK["folder"]
    os.makedirs(os.path.join(WORK, "scoring"), exist_ok=True)
    rows, reference = [], None
    ref_cc, ref_status = cc_mask(native, DOCK["map"], DOCK["resolution"], os.path.join(WORK, "scoring", "native_map_model_cc.txt"))
    STATE["native_cc_mask"] = {"measured_here": ref_cc, "status": ref_status, "in_entries_table": row.get("cc_mask")}
    for spec in DOCK_PLAN:
        r = RESULTS.get(spec["name"], {"name": spec["name"], "skipped": True, "docked": False})
        out = dict.fromkeys(RESULT_COLUMNS, "")
        out.update({"name": spec["name"], "parameters": r.get("parameters", ""), "cap_min": spec["cap_min"], "seconds": r.get("seconds", ""), "rc": r.get("rc", ""),
                    "timed_out": r.get("timed_out", ""), "docked": r.get("docked", False), "skipped": r.get("skipped", False), "note": spec.get("note", ""),
                    "reflections": r.get("reflections", "")})
        if r.get("docked"):
            pdb = r["pdb"]
            ev = evaluate_models.evaluate(STATE["usalign"], pdb, native, int(row["modeled_residues"]))
            full = subprocess.run([STATE["usalign"], pdb, native, "-mol", "protein"], capture_output=True, text=True, errors="replace")
            with open(os.path.join(WORK, "scoring", "%s_usalign.txt" % spec["name"]), "w", encoding="utf-8") as f:
                f.write("# USalign %s %s -mol protein\n%s%s" % (pdb, native, full.stdout, full.stderr))
            out.update({k: ev.get(k, "") for k in ("tm_score", "ca_rmsd", "aligned_ca_length", "pred_residues", "seq_identity")})
            out["chain_comparison"] = chain_comparison(pdb, native, os.path.join(WORK, "scoring", "%s_chain_comparison.txt" % spec["name"]))
            out["cc_mask"], _ = cc_mask(pdb, DOCK["map"], DOCK["resolution"], os.path.join(WORK, "scoring", "%s_map_model_cc.txt" % spec["name"]))
            if reference is None:
                reference = (spec["name"], pdb)
            out["reference"] = reference[0]
            out["ca_distance_to_reference"] = "%s (%d atoms)" % ca_distance(pdb, reference[1])
        rows.append(out)
    write_csv("dock_results.csv", RESULT_COLUMNS, rows)
    log("\n%-20s %7s %9s %9s %6s %7s %8s %8s  %s" % ("docking test", "docked", "seconds", "refl.", "TM", "CC_mask", "RMSD", "Ca dist", "settings"))
    for o in rows:
        log("%-20s %7s %9s %9s %6s %7s %8s %8s  %s" % (o["name"], "skipped" if o["skipped"] else ("yes" if o["docked"] else ("TIMEOUT" if o["timed_out"] else "no")), o["seconds"], o["reflections"], o["tm_score"],
                                                 o["cc_mask"], o["ca_rmsd"], str(o.get("ca_distance_to_reference", "")).split(" ")[0], o["parameters"]))
    log("deposited model CC_mask measured here: %s (entries table: %s)" % (ref_cc, row.get("cc_mask")))


if STATE["ok"] and DOCK:
    try:
        score_docking()
        STATE["complete"] = True
    except Exception:
        import traceback
        STATE["stopped_at"] = "scoring"
        log("scoring stopped by an error:\n%s" % traceback.format_exc())
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

EXTRA_INFO = ('{"mica_commit": globals().get("MICA_COMMIT", ""), "phenix_installer": os.path.basename(globals().get("INSTALLER", "")), "phenix_installer_sha256": STATE["installer_sha256"], '
              '"phenix_version_output": STATE["phenix_version"], "usalign_commit": globals().get("USALIGN_COMMIT", ""), "usalign_version": STATE["usalign_version"], '
              '"input_sha256": globals().get("EXPECTED", {}), "dock_plan": globals().get("DOCK_PLAN", []), "dock_results": globals().get("RESULTS", {})}')


def step6_code():
    """The preparation functions of the first pilot (read, download, find the AlphaFold files, contour level, prepare_entry), then the new part."""
    pre = nbkit.cut_before(P.STEP6, "def mica_programs(")
    return pre + STEP6_NEW


def step8_code(filled_step6):
    chain = nbkit.function_sources(filled_step6, ["chain_comparison"])
    return chain + "\n\n" + STEP8_FUNCS


def build(out_path, entries=("26DK",), phenix_url=P.PHENIX_URL, mica_repo=P.MICA_REPO, mica_commit=P.MICA_COMMIT, usalign_repo=P.USALIGN_REPO,
          usalign_commit=P.USALIGN_COMMIT, entries_path=None, env_python=None, usalign_bin=None, net_hosts=NET_HOSTS, plan=None, require_gpu=False, dd_version="v1"):
    """Writes the notebook. dd_version "v1" is the first test (dockdiag.ipynb), "v2" the long one (dockdiag2.ipynb). The keyword arguments other than `entries`, `plan` and
    `dd_version` exist for the tests (local servers, a ready environment)."""
    notebook, nb_name = VERSIONS[dd_version]["notebook"], VERSIONS[dd_version]["nb_name"]
    import re
    m = re.search(r"version=([^&]+)&file=([^&]+)$", phenix_url)
    if not m:
        raise ValueError("the Phenix link must end in ?version=...&file=...: %s" % phenix_url)
    version, installer = m.group(1), m.group(2)
    file_cells, expected = nbkit.file_cells(FILES, entries_path)
    mapping = {"@PHENIX_URL@": phenix_url, "@PHENIX_VERSION@": version, "@INSTALLER@": installer, "@MICA_REPO@": mica_repo, "@MICA_COMMIT@": mica_commit,
               "@WEIGHTS_URL@": P.WEIGHTS_URL, "@ZENODO_API@": P.ZENODO_API, "@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit,
               "@UV_VERSION@": P.UV_VERSION, "@PY_VERSION@": P.PY_VERSION, "@ENTRIES@": repr(list(entries)), "@ENTRY_LIST@": ", ".join(entries),
               "@REQUIRE_GPU@": repr(bool(require_gpu)), "@ENV_PYTHON@": repr(env_python), "@USALIGN_BIN@": repr(usalign_bin), "@NET_HOSTS@": repr(tuple(net_hosts)),
               "@EXPECTED@": repr(expected), "@NOTEBOOK@": notebook, "@PLAN@": repr(plan if plan is not None else PLANS[dd_version]), "@EXTRA_INFO@": EXTRA_INFO}
    fl = lambda text: nbkit.fill(text, mapping).strip("\n") + "\n"
    step5 = nbkit.replace_once(P.STEP5, "fetch_mica() and fetch_weights() and import_mica()", "fetch_mica() and import_mica()")
    step6_filled = nbkit.fill(step6_code(), mapping)
    cells = [("intro", nbkit.md(nbkit.fill(INTRO_V2 if dd_version == "v2" else INTRO, mapping))),
             ("step1-title", nbkit.md(P.STEP1_MD)), ("step1", nbkit.code(fl(P.STEP1))),
             ("step2-title", nbkit.md(P.STEP2_MD)), ("step2-dirs", nbkit.code(P.STEP2_DIRS.strip("\n") + "\n"))]
    cells += file_cells
    cells += [("step2-check", nbkit.code(fl(P.STEP2_CHECK))),
              ("robust-title", nbkit.md(nbkit.fill(nbkit.ROBUST_MD, mapping))), ("robust", nbkit.code(fl(nbkit.ROBUST))),
              ("step3-title", nbkit.md(P.STEP3_MD)), ("step3", nbkit.code(fl(P.STEP3))),
              ("step3b-title", nbkit.md(STEP3B_MD)), ("step3b", nbkit.code(fl(STEP3B))),
              ("step4-title", nbkit.md(STEP4_MD)), ("step4", nbkit.code(fl(P.STEP4))),
              ("step5-title", nbkit.md(STEP5_MD)), ("step5", nbkit.code(fl(step5))),
              ("step6-title", nbkit.md(STEP6_MD)), ("step6", nbkit.code(step6_filled.strip("\n") + "\n")),
              ("step7-title", nbkit.md(STEP7_MD)), ("step7", nbkit.code(fl(STEP7))),
              ("step8-title", nbkit.md(STEP8_MD)), ("step8", nbkit.code(nbkit.fill(step8_code(nbkit.fill(P.STEP6, mapping)), mapping).strip("\n") + "\n")),
              ("final-title", nbkit.md(nbkit.FINAL_MD)), ("final", nbkit.code(fl(nbkit.FINAL)))]
    nbkit.assemble(out_path, nb_name, cells)
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", choices=sorted(PLANS), default="v1")
    p.add_argument("--out")
    p.add_argument("--entries", nargs="+", default=["26DK"])
    p.add_argument("--phenix-url", default=P.PHENIX_URL)
    args = p.parse_args(argv)
    args.out = args.out or os.path.join(HERE, VERSIONS[args.version]["nb_name"])
    expected = build(args.out, entries=args.entries, phenix_url=args.phenix_url, dd_version=args.version)
    print("wrote %s with %d embedded files" % (args.out, len(expected)))
    for path, digest in expected.items():
        print("  %s  %s" % (digest[:16], path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
