#!/usr/bin/env python3
"""Build the notebooks that run MICA itself on the benchmark entries (scripts/mica_cpu_a.ipynb, ..., scripts/mica_gpu.ipynb).

MICA's pipeline (MICA_pipeline.sh) has three programs: 1. process the AlphaFold predictions (domain split), 2. dock the domains into the map with Phenix, 3. `run.py`: the neural
network (the only part that needs a GPU), then model building, Pulchra and Phenix refinement (all CPU). Two facts of the pilots decide how they are run here:

* the docking of program 2 (Phenix `dock_in_map`, MICA's call) does not finish on a free machine for a big map, but takes minutes with Phenix's other engine
  (LOG.md, 2026-10-24): the notebooks run MICA's own docking program through `mica_dock_driver.py`, which changes only that call and adds a time limit;
* MICA's network cuts the resampled map (1 A voxels) into windows of 64^3 with a step of 48 and predicts every window: 216 windows for the 266 A cube of 26DK. On a CPU a
  window takes about 25 to 40 seconds, on a T4 a fraction of a second. So the long CPU work (docking, model building) runs in CPU sessions, which have no weekly quota, and
  the network runs on a CPU only for small maps; the other entries are "deferred" and run in a short GPU session with the docked state of the CPU sessions.

Notebooks (all built by this file):

    python make_mica_notebook.py --kind cpu --shards 3          # mica_cpu_a/b/c.ipynb: stages 1, 2 and (small maps) 3; the entries are shared round-robin
    python make_mica_notebook.py --kind gpu --done <mica_status.csv> ...    # mica_gpu.ipynb: stage 3 for the entries that no session finished

Each notebook works through its entries in the order of the seeded selection, `--lanes` entries at a time, with a time budget (`HOURS`) after which it starts nothing new;
what is not finished is written to `pending.csv`. The files of an entry that later stages need (domains, docked models) are private (they contain the AlphaFold predictions of the
user): they are kept in the notebook's output folder `mica_state/`, never in the zip.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_main_notebook as MN  # noqa: E402  (the order of the entries, the list of tasks)
import make_pilot_notebook as P  # noqa: E402
import nbkit  # noqa: E402

ROOT = nbkit.ROOT
FILES = (("scripts/nbtools.py", "scripts/nbtools.py"),
         ("scripts/mica_inputs.py", "scripts/mica_inputs.py"),
         ("scripts/evaluate_models.py", "scripts/evaluate_models.py"),
         ("scripts/mica_dock_driver.py", "scripts/mica_dock_driver.py"),
         ("data/af3/af3_jobs.csv", "data/af3/af3_jobs.csv"),
         ("data/af3/af3_files_sha256.txt", "data/af3/af3_files_sha256.txt"),
         ("data/entries_2026.csv", "data/entries_2026.csv"),
         ("env/mica_requirements.txt", "env/mica_requirements.txt"))
NET_HOSTS = (("PyPI", "https://pypi.org/simple/uv/", True),
             ("GitHub", "https://github.com/jianlin-cheng/MICA", True),
             ("Zenodo", "https://zenodo.org/api/records/15756654", True),
             ("EBI FTP", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-80558/header/emd-80558.xml", True),
             ("RCSB", "https://files.rcsb.org/header/26DK.cif", True),
             ("EMDB API", "https://www.ebi.ac.uk/emdb/api/entry/80558", False),
             ("Phenix", "https://phenix-online.org/download", False))

# The docking attempts of mica_dock_driver.py. PASS 1 (LOG.md, 2026-10-23; run on 2026-10-24: 14 of 59 domains docked, 43 ran into the 45 minutes): Phenix's Fourier-convolution engine
# with its default quick search; the second attempt, only if the first ended without a solution (not if it ran out of time), is MICA's own setting quick=False.
# timeout_s: per attempt and domain.
DOCK_ATTEMPTS_PASS1 = [{"name": "no_mr_quick", "set": {"dock_with_mr": "False", "quick": "True"}, "timeout_s": 2700},
                       {"name": "no_mr_thorough", "set": {"dock_with_mr": "False", "quick": "False"}, "timeout_s": 3600, "skip_after_timeout": True}]
# PASS 2 (LOG.md, 2026-10-25; run on 2026-10-25): the same plus a search at 6 A / 10 A that also set `rigid_body_refinement_resolution` and `density_radius`. Phenix 2.2.1 rejected the first of them
# (a True/False parameter) within 2 seconds in all 59 attempts of both rungs; only the third rung, the plain call of pass 1, ran: 12 of 59 domains docked. Kept as the record.
DOCK_ATTEMPTS_PASS2 = [{"name": "res6_quick", "set": {"dock_with_mr": "False", "quick": "True"}, "search_resolution": 6.0, "timeout_s": 2700},
                       {"name": "res10_quick", "set": {"dock_with_mr": "False", "quick": "True"}, "search_resolution": 10.0, "timeout_s": 1800},
                       {"name": "plain_quick", "set": {"dock_with_mr": "False", "quick": "True"}, "timeout_s": 2700, "skip_after_timeout": True}]
# PASS 3 (LOG.md, 2026-10-26): the same engine and quick search, but the whole docking (search and rigid-body refinement) runs at 6 A, set through `resolution`, the one parameter that MICA's own
# call passes (`search_resolution` of mica_dock_driver.py); then at 10 A after a failure or a timeout. No third rung: the plain call of pass 1 ran twice (14 and 12 of 59 domains) and is not repeated.
# A domain that no rung docks is skipped, as in MICA; an entry without any docked domain goes on to program 3 in MICA's documented mode without docked AlphaFold input (`dock_mode` none).
DOCK_PROFILE = "pass3"
DOCK_ATTEMPTS = [{"name": "res6_quick", "set": {"dock_with_mr": "False", "quick": "True"}, "search_resolution": 6.0, "timeout_s": 2400},
                 {"name": "res10_quick", "set": {"dock_with_mr": "False", "quick": "True"}, "search_resolution": 10.0, "timeout_s": 1500}]
DOCK_STAGE_CAP_S = 8 * 3600          # all the domains of one entry
# The four CPU sessions of pass 2 and pass 3 (programs 1 and 2 only): the 20 entries, the heaviest first, balanced by the number of domains that MICA's program 1 made (15, 15, 15, 14).
PASS3_SHARDS = {"a": ["9WME", "13BD", "9T5K", "26DK", "9TKA"], "b": ["9OC0", "9VI4", "9ZLH", "22FX", "9WQ6"],
                "c": ["9WFL", "9WTX", "9RWS", "9JKK", "9OTA"], "d": ["9VBD", "9VUH", "9Z24", "9KQM", "9V4D"]}
PASS2_SHARDS = PASS3_SHARDS          # pass 2 used the same partition
# PASS 4 (LOG.md, 2026-10-27): the 12 entries whose program 3 did not run in the GPU sessions of 2026-10-27 (CUDA out of memory in MICA's batch strategy, see BATCH_PATCH), in two GPU sessions,
# balanced by the number of windows (2185 and 1935: the network predicts about 0.32 windows per second on a T4). Each is run twice: with the docked input (mica_gpu2_a/b) and without (mica_gpu2_nodock_a/b).
PASS4_SHARDS = {"a": ["9WQ6", "9V4D", "9T5K", "9OC0", "9VI4"], "b": ["22FX", "9TKA", "9VBD", "9WTX", "26DK", "9Z24", "13BD"]}
PASS4_HOURS = {"main": 3.5, "nodock": 3.0}
BATCH_THRESHOLD_NEW = 1000000
PASS5_ENTRIES = ["9OC0", "9VI4"]                           # cut by the end of the time budget of mica_gpu2_a, not results (LOG.md 2026-10-28); 26DK was killed for lack of RAM: a failure
PASS5_HOURS = 4.2                                          # the GPU quota left on 2026-10-28 was 5 h 14 m: hours left minus 1
P3_CAP_CPU_S = 3 * 3600              # program 3 on a CPU
P3_CAP_GPU_S = 2 * 3600              # program 3 on a GPU
CPU_MAX_WINDOWS = 120                # more windows than this are not predicted on a CPU (about 25 to 40 s per window)
KINDS = {"cpu": {"stages": (1, 2, 3), "lanes": 2, "hours": 10.5, "device": "auto", "require_gpu": False},
         "gpu": {"stages": (3,), "lanes": 3, "hours": 9.0, "device": "cuda", "require_gpu": True}}


def shard_entries(order, index, count):
    """Entry i goes to shard i mod count: every shard gets early (first-priority) entries and late ones."""
    return [e for i, e in enumerate(order) if i % count == index]


def pending_entries(entries, status_csvs, redo=()):
    """The entries that no earlier session finished: `ok` and `failed` are results, everything else (deferred, docked, interrupted, not run) is still to do.
    `redo`: entries to run again although a session finished them (a failure that turned out to be a problem of the notebook and not of MICA; LOG.md says so)."""
    done = set()
    for path in status_csvs:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["status"] in ("ok", "failed"):
                    done.add(r["entry"])
    done -= set(redo)
    return [e for e in entries if e not in done]


def dock_text(attempts):
    """The docking attempts in words (for the notebook's intro): 'res6_quick: search at 6 A, at most 45 minutes; ...'."""
    parts = []
    for a in attempts:
        s = a.get("search_resolution")
        parts.append("%s: %s, at most %s" % (a.get("name"), "search at %g A" % s if s else "search at the resolution of the map",
                                             "%d minutes" % round(a["timeout_s"] / 60.0) if a.get("timeout_s") else "no time limit"))
    return "; ".join(parts)


def gpu_order(entries, geometry_csv=os.path.join(ROOT, "data", "mica_map_geometry.csv")):
    """The GPU session runs the entries with the fewest windows first (the most entries are finished if the budget runs out); equal windows keep the seeded order."""
    if not os.path.exists(geometry_csv):
        return list(entries)
    with open(geometry_csv, newline="", encoding="utf-8") as f:
        windows = {r["entry"]: int(r["windows"]) for r in csv.DictReader(f)}
    return sorted(entries, key=lambda e: (windows.get(e, 10 ** 9), list(entries).index(e)))


# ---------------------------------------------------------------------------------------------------------- text of the cells
INTRO_CPU = """\
# MICA benchmark: MICA on the CPU, session `@NOTEBOOK@` (entries @ENTRY_LIST@)

MICA's own pipeline for the entries listed above, in the order of the seeded selection, @LANES@ entries at a time (each docking uses MICA's own number of processes):

1. **program 1**: MICA splits your AlphaFold Server predictions into domains,
2. **program 2**: MICA docks every domain into the map with Phenix. Phenix's default engine of MICA's call does not finish on a free machine (LOG.md, 2026-10-24, 2026-10-25), so
   `mica_dock_driver.py` runs MICA's own docking program with Phenix's other engine (`dock_with_mr=False`), a docking at a coarser resolution and a time limit per domain
   (@DOCK_TEXT@); everything else is MICA's. A domain that cannot be docked is skipped, as in MICA. If **no** domain of an entry can be docked, program 3 runs without docked
   AlphaFold input, the mode that MICA's README describes ("skip this step if you don't want to use AF3 at input level"); the table says so (`dock_mode`),
3. **program 3**: MICA's neural network, model building, Pulchra and Phenix refinement. @PROGRAM3_TEXT@

The notebook plans its work to end before `HOURS` are used; what it did not finish is written to `pending.csv`. Every entry that fails is a result (the reason is in the table);
an entry that the end of the session cuts is not a result: it is run again.

**What you do (Kaggle, background mode)**: accelerator **None**, Internet **On**, the two Phenix secrets (`PHENIX_USER`, `PHENIX_PASSWORD`) attached, the private AlphaFold dataset added as input,
then *Save Version -> Save & Run All (Commit)*. It runs on Kaggle's servers; you can close the tab and your Wi-Fi does not matter. At most @HOURS_TEXT@ hours (@TIME_TEXT@).
When the version shows *Succeeded*, open it and download the zip `@NOTEBOOK@_result_....zip` from its Output and send it to me.
**Do not stop or cancel the run**: a stopped run keeps nothing. **Keep the notebook private**: its output folder `mica_state` holds files made from your AlphaFold predictions.

**Phenix** is licensed to you personally. Do not share the installer, the password, or this notebook with the secrets attached.
"""

INTRO_GPU = """\
# MICA benchmark: MICA's network and model building on the GPU, session `@NOTEBOOK@` (entries @ENTRY_LIST@)

Program 3 of MICA (neural network on the GPU, model building, Pulchra, Phenix refinement) for the entries that the CPU sessions have docked but not finished. The docked state of an entry
(`mica_state/<entry>/`, private) comes from the output of the CPU notebook that docked it: **add the output of each of those notebooks as an input of this notebook**
(Add Input -> Notebook Output). @STATE_TEXT@ An entry without a docked state is marked *not run*. The table `mica_status.csv` has a column `dock_mode` for every entry (`full`: every domain docked, `partial`, `none`: program 3 ran without docked AlphaFold input). Up to @LANES@ entries are prepared at the same time, but program 3 itself runs for one entry at a time (it rewrites a folder of MICA's that is shared); the
notebook plans its work to end before `HOURS` of GPU time are used and writes what it did not finish to `pending.csv`. The smallest maps come first, so that the most entries are finished if the budget runs out.

**What you do (Kaggle, background mode)**: accelerator **GPU T4 x2** (not P100), Internet **On**, the two Phenix secrets attached, the private AlphaFold dataset added as input,
the outputs of the CPU notebooks added as inputs, `HOURS` in the first cell (the GPU hours you allow, at most your quota left minus 1), then *Save Version -> Save & Run All (Commit)*.
When the version shows *Succeeded*, download the zip `@NOTEBOOK@_result_....zip` from its Output and send it to me. **Do not stop or cancel the run**. **Keep the notebook private.**

**Phenix** is licensed to you personally. Do not share the installer, the password, or this notebook with the secrets attached.
"""

SETTINGS_MD = "## Settings (the only number you may have to change)"
SETTINGS = '''\
# HOURS = how many hours this notebook may run. A Kaggle session lasts at most 12 hours (CPU) and the notebook needs about 30 minutes for the setup and 15 for the zip, so 10.5 is the
# most that makes sense. GPU notebook: the hours of GPU time you allow (Kaggle gives about 30 per week; use your hours left minus 1). No new entry is started when the budget is nearly used up.
HOURS = @HOURS@
'''

STEP3_MD_NOTE = P.STEP3_MD
PRODUCTION_MD = """\
## Step 6. MICA's programs for the entries, one after the other

For each entry: the input folder is built (your AlphaFold Server prediction, the map from EMDB, the deposited model from RCSB, the contour level and resolution from MICA's own
`emdb_extractor.py`), then the programs of the stage list run one after the other. A program that runs longer than its limit is stopped and counted as failed (docking: @DOCK_TEXT@;
all the domains of one entry together @DOCK_CAP_H@ hours; program 3: 3 hours on a CPU, 2 hours on a GPU). A program that the end of the time budget cuts is *interrupted*: not a result.
The progress line every two minutes shows the last line of the program's own log. The files that later stages need are copied to `mica_state/<entry>/` (private)."""

PRODUCTION = r'''
STAGES = @STAGES@                                   # 1 = domain split, 2 = docking, 3 = MICA's network and model building
LANES = @LANES@                                     # entries that are processed at the same time
P3_DEVICE = "@P3_DEVICE@"                           # "auto": cuda if the session has a GPU, else cpu
CPU_MAX_WINDOWS = @CPU_MAX_WINDOWS@                 # a map with more windows than this is not predicted on a CPU
DOCK_ATTEMPTS = @DOCK_ATTEMPTS@
DOCK_PROFILE = "@DOCK_PROFILE@"                    # written to dock_profile.json in the entry folder when a session docks it
STATE_PROFILE = @STATE_PROFILE@                    # a session that does not dock (program 3 only) restores only the state that was docked with this profile (None: any)
CAPS = {"1_process_af3": 1800, "2_dock": @DOCK_STAGE_CAP_S@, "3_build_cpu": @P3_CAP_CPU_S@, "3_build_cuda": @P3_CAP_GPU_S@}     # seconds
REFINE_WAIT_S = float(os.environ.get("MICA_REFINE_WAIT", "1800"))
ROWS = {r["pdb_id"]: r for r in read_csv("data/entries_2026.csv")}
# private: files made from the AlphaFold predictions. The zip is made from the work folder only; on Kaggle this folder is next to it (/kaggle/working/mica_state), so it is kept as
# output of the notebook (the GPU notebook reads it from there) but is never in the zip. In the tests the work folder and the output folder are the same: the state goes beside it.
STATE_DIR = os.path.join(OUT_DIR, "mica_state") if os.path.abspath(WORK) != os.path.abspath(OUT_DIR) else os.path.abspath(WORK) + "_mica_state"
DRIVER = os.path.join(WORK, "scripts", "mica_dock_driver.py")
HOURS_SETTING = float(globals().get("HOURS", @HOURS@))
SESSION_S = min(HOURS_SETTING, 11.5) * 3600
RESERVE_S = float(os.environ.get("MICA_RESERVE_S", "900"))          # the end of the budget is kept for the summary and the zip
MIN_START_S = float(os.environ.get("MICA_MIN_START_S", "1800"))     # no new entry is started when less than this is left
HARD_STOP = STATE["t0"] + SESSION_S - RESERVE_S                    # a program that still runs at this time is stopped ("interrupted", not a failure)
EARLY_FAIL_S = float(os.environ.get("MICA_EARLY_FAIL", "1200"))      # two entries in a row that fail in program 1 or 3 sooner than this: the installation is probably at fault
QUEUE = list(ENTRIES)
QUEUE_LOCK, INPUT_LOCK = threading.Lock(), threading.RLock()
P3_LOCK = threading.Lock()                                          # program 3 of MICA deletes and rewrites output/results/predicted_grids (utils/predict.py): one at a time
P3 = {"fails": 0, "off": False}
EARLY = {"fails": 0, "stop": False, "reason": ""}
os.makedirs(os.path.join(WORK, "docking_logs"), exist_ok=True)
os.makedirs(os.path.join(WORK, "models", "mica"), exist_ok=True)
MICA_METRIC_COLUMNS = ["method", "entry", "model_kind", "model_file", "status", "note", "tm_score", "tm_score_pred_norm", "ca_rmsd", "aligned_ca_length", "seq_identity",
                       "pred_residues", "native_residues", "coverage", "chain_comparison"]


def gpu_count():
    if not shutil.which("nvidia-smi"):
        return 0
    out = sh("nvidia-smi --query-gpu=index --format=csv,noheader", timeout=30).splitlines()
    return len([ln for ln in out if ln.strip().isdigit()])


N_GPUS = gpu_count()
DEVICE = "cuda" if (P3_DEVICE == "cuda" or (P3_DEVICE == "auto" and N_GPUS > 0)) else "cpu"
GPU_OF = lambda lane: (lane % N_GPUS) if (DEVICE == "cuda" and N_GPUS) else None
if 3 in STAGES and DEVICE == "cuda" and not N_GPUS and STATE["ok"]:
    fail("this notebook runs program 3 on a GPU but the session has none. Kaggle: Session options -> Accelerator -> GPU T4 x2, then Run all again.")
log("stages %s | %d entries at a time | program 3 on the %s | %d GPU(s) | budget %.1f hours (until %.1f hours after the start of the notebook)" % (
    STAGES, LANES, DEVICE.upper(), N_GPUS, HOURS_SETTING, (HARD_STOP - STATE["t0"]) / 3600))


class GpuMonitor(Monitor):
    """The Monitor of step 1 for the GPU of one lane (nvidia-smi -i <n>): the base class would always read the first GPU."""

    def __init__(self, path, gpu, every=5):
        super().__init__(path, every)
        self.gpu = gpu

    def sample(self):
        gpu_mb, util = "", ""
        if shutil.which("nvidia-smi"):
            out = sh("nvidia-smi -i %d --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits" % self.gpu, timeout=20).splitlines()
            try:
                a, b = [x.strip() for x in out[0].split(",")]
                gpu_mb, util = int(a), int(b)
                self.peak_gpu_mb, self.peak_util = max(self.peak_gpu_mb, gpu_mb), max(self.peak_util, util)
            except (ValueError, IndexError):
                gpu_mb, util = "", ""
        ram = ""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        ram = round(int(line.split()[1]) / 1048576, 2)
        except OSError:
            pass
        if ram != "":
            self.min_ram_gb = ram if self.min_ram_gb is None else min(self.min_ram_gb, ram)
        self.samples += 1
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("%s,%s,%s,%s\n" % (datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"), gpu_mb, util, ram))


def save_state():
    """state.json from a snapshot: the lanes change STATE while it is written, and json.dump refuses a dictionary that changes size."""
    for _ in range(5):
        try:
            text = json.dumps(STATE, indent=1, default=str)
        except RuntimeError:
            time.sleep(0.2)
            continue
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as ex:
            log("could not save state.json: %s" % ex)
        return
    log("could not save state.json: the state kept changing")


def utc_iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def queue_line():
    counts = {}
    for e in STATE["entries"].values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return "queue: %d waiting; %s; %.1f of %.1f hours used" % (len(QUEUE), ", ".join("%d %s" % (n, s) for s, n in sorted(counts.items())) or "nothing finished yet",
                                                              (time.time() - STATE["t0"]) / 3600, SESSION_S / 3600)


GEOMETRY_CODE = """
import json, math, sys, mrcfile
res = float(sys.argv[2])
with mrcfile.open(sys.argv[1], permissive=True) as m:
    shape = [int(n) for n in m.data.shape]
    voxel = [float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)]
resampled = [int(round(n * v)) for n, v in zip(shape, voxel)]          # MICA: scipy zoom to 1 A voxels (utils/preprocessing.py), the zoom factors in the order x, y, z
windows = 1
for n in resampled:
    windows *= int(math.ceil(n / 48.0))                                  # utils/create_grids.py: grid_size 48, padding 8 (windows of 64^3)
box = [n * v for n, v in zip(shape[::-1], voxel)]
print(json.dumps({"grid": shape, "voxel_A": voxel, "box_A": [round(b, 1) for b in box], "resampled_grid": resampled, "windows": windows,
                  "reflections_M": round(2.0944 / res ** 3 * box[0] * box[1] * box[2] / 1e6, 3)}))
"""


def map_geometry(map_path, resolution):
    """Grid, box, number of windows of MICA's network and the number of reflections Phenix's Phaser would see (2.0944 / d^3 * V, Friedel pairs counted once; 26DK: 0.915 M)."""
    try:
        rc, last, _ = run([ENV_PY, "-c", GEOMETRY_CODE, map_path, str(resolution)], env=MICA_ENV, quiet=True)
        return json.loads(last[-1]) if rc == 0 and last else {}
    except Exception as ex:
        log("the geometry of the map could not be read: %s" % ex)
        return {}


# ---- the state that later stages (and later sessions) need
SNAPSHOT_SKIP_DIRS = {"AF3_results", "grids", "AF3_encodings", "temp_maps", "__pycache__"}
SNAPSHOT_SKIP_SUFFIXES = (".map", ".mrc", ".ccp4", ".gz", ".part", ".tmp")


def find_state_roots():
    roots = [STATE_DIR] + [p for p in os.environ.get("MICA_STATE_DIRS", "").split(os.pathsep) if p]
    base = "/kaggle/input"
    if os.path.isdir(base):
        for dirpath, dirs, _ in os.walk(base):
            if os.path.basename(dirpath) == "mica_state":
                roots.append(dirpath)
                dirs[:] = []
            elif dirpath[len(base):].count(os.sep) >= 6:
                dirs[:] = []
    return [r for r in dict.fromkeys(roots) if os.path.isdir(r)]


def state_profile(folder):
    try:
        with open(os.path.join(folder, "dock_profile.json"), encoding="utf-8") as f:
            return json.load(f).get("profile", "")
    except (OSError, ValueError):
        return ""


def restore_state(entry, folder, res):
    """Copies the saved state of this entry (an earlier session's private files) into the entry folder; the one with the most finished programs wins.
    With STATE_PROFILE only the state that was docked with that profile counts (the state of pass 1 has none and is left where it is)."""
    best = None
    for root in find_state_roots():
        src = os.path.join(root, entry)
        if os.path.isdir(src):
            if STATE_PROFILE and state_profile(src) != STATE_PROFILE:
                log("%s: the state in %s was not docked with the profile %s (it has %r) and is not used" % (entry, src, STATE_PROFILE, state_profile(src)))
                continue
            key = (sum(os.path.exists(os.path.join(src, ".done_" + n)) for n in ("1_process_af3", "2_dock", "3_build")), os.path.getmtime(src))
            if best is None or key > best[0]:
                best = (key, src)
    if best is None:
        return None
    shutil.copytree(best[1], folder, dirs_exist_ok=True)
    res["restored_from"] = best[1]
    return best[1]


def snapshot_state(entry, folder):
    dst = os.path.join(STATE_DIR, entry)
    os.makedirs(dst, exist_ok=True)
    for dirpath, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in SNAPSHOT_SKIP_DIRS]
        target = os.path.join(dst, os.path.relpath(dirpath, folder))
        os.makedirs(target, exist_ok=True)
        for n in names:
            if n.endswith(SNAPSHOT_SKIP_SUFFIXES):
                continue
            src = os.path.join(dirpath, n)
            out = os.path.join(target, n)
            if not os.path.exists(out) or os.path.getsize(out) != os.path.getsize(src) or os.path.getmtime(out) < os.path.getmtime(src):
                shutil.copy2(src, out)


def copy_docking_logs(entry, folder):
    src = os.path.join(folder, "docking_logs")
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(WORK, "docking_logs", entry), dirs_exist_ok=True)


def docking_summary(folder):
    domains = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "AF3_domains", "*.pdb")))
    rows = []
    table = os.path.join(folder, "docking_logs", "bench_docking.csv")
    if os.path.exists(table):
        rows = read_csv(table)
    docked = sorted({r["model"] for r in rows if r["docked"] == "True"})
    return {"domains": len(domains), "docked": len(docked), "seconds": sum(int(r["seconds"] or 0) for r in rows), "rows": rows}


def docking_state(folder):
    """(domains made by program 1, domains docked) read from the files of the entry folder, so that a session that does not dock (program 3 only) knows the mode."""
    return (len(glob.glob(os.path.join(folder, "AF3_domains", "*.pdb"))), len(glob.glob(os.path.join(folder, "AF3_docked_models", "*_docked.pdb"))))


def dock_mode_of(domains, docked):
    return "none" if docked == 0 else ("full" if docked >= domains else "partial")


def stage_made(entry, folder, name):
    if name == "1_process_af3":
        return glob.glob(os.path.join(folder, "AF3_domains", "*.pdb"))
    if name == "2_dock":
        return glob.glob(os.path.join(folder, "%s_af3_docked.pdb" % entry))
    return glob.glob(os.path.join(MICA_DIR, "output", "*_%s_MICA_all_atom_model.pdb" % entry.lower()))


def run_stage(entry, name, cmd, folder, res, cap, env, gpu=None, watch=(), tag="", lock=None):
    """One program of MICA for one entry. Returns "ok", "failed" or "interrupted" (cut by the end of the time budget); the details go to res["steps"][name].
    `lock`: held while the program runs (program 3: one at a time, see P3_LOCK)."""
    marker = os.path.join(folder, ".done_" + name)
    if os.path.exists(marker):
        log("%s: %s finished in an earlier session, not repeated" % (tag, name))
        res["steps"][name] = {"seconds": 0, "skipped": "finished in an earlier session"}
        return "ok"
    if lock is None:
        return run_stage_now(entry, name, cmd, folder, res, cap, env, gpu, watch, tag)
    t_wait = time.time()
    with lock:
        if time.time() - t_wait > 60:
            log("%s: waited %d s for the program 3 of another entry" % (tag, time.time() - t_wait))
        return run_stage_now(entry, name, cmd, folder, res, cap, env, gpu, watch, tag)


def run_stage_now(entry, name, cmd, folder, res, cap, env, gpu, watch, tag):
    marker = os.path.join(folder, ".done_" + name)
    limit = min(cap, HARD_STOP - time.time())
    if limit < 120:
        res.update(failed_step=name, note="the time budget ended before %s could start" % name)
        return "interrupted"
    logf = os.path.join(WORK, "logs", "%s_%s.log" % (entry, name))
    mon = GpuMonitor(os.path.join(WORK, "gpu", "%s_%s.csv" % (entry, name)), gpu if gpu is not None else 0)
    mon.start()
    t0 = time.time()
    log("\n=== %s: MICA program %s: %s" % (tag, name, " ".join(str(c) for c in cmd[1:])[:260]))
    rc, last, timed_out = run_watched(cmd, "%s %s" % (tag, name), watch=[logf] + list(watch), env=env, cwd=MICA_DIR, log_file=logf, timeout=limit, every=1, quiet=True)
    info = mon.stop()
    info.update(rc=rc, seconds=round(time.time() - t0), timed_out=timed_out, limit_s=round(limit), started_utc=utc_iso(t0))
    res["steps"][name] = info
    made = stage_made(entry, folder, name)
    log("%s: %s finished: exit code %s, %d s%s, lowest free RAM %s GB; output %s" % (tag, name, rc, info["seconds"], " (STOPPED at its limit)" if timed_out else "",
                                                                                   info["min_ram_available_gb"], "written" if made else "NOT written"))
    if timed_out and time.time() >= HARD_STOP - 30:
        res.update(failed_step=name, note="cut by the end of the time budget during %s" % name)
        return "interrupted"
    if name == "2_dock" and rc == 0 and not timed_out and not made:
        # No domain could be docked within the limits. MICA's README: "Skip this step if you don't want to use AF3 at input level": program 3 then runs without the docked AlphaFold
        # input (its network reads zeros for those channels) and still uses the AlphaFold models when it builds the chains. The entry goes on in that mode (LOG.md, 2026-10-26).
        res["dock_mode"] = "none"
        res["note"] = "no domain could be docked: program 3 runs without docked AlphaFold input (MICA's documented mode)"
        log("%s: %s" % (tag, res["note"]))
        with open(marker, "w") as f:
            f.write("no domain docked\n")
        return "ok"
    if rc != 0 or timed_out or not made:
        res.update(failed_step=name, note=("timeout after %d s" % limit if timed_out else "exit code %s" % rc) + ("" if made else "; the output of %s was not written" % name))
        log("%s STOPPED in %s: %s. The last lines of the program:\n%s" % (tag, name, res["note"], "\n".join(last[-15:])))
        return "failed"
    with open(marker, "w") as f:
        f.write("done\n")
    return "ok"


def refine_running(entry):
    try:
        return subprocess.run(["pgrep", "-f", "real_space_refine.*_%s_MICA_all_atom_model" % entry.lower()], capture_output=True).returncode == 0
    except OSError:
        return False


def wait_for_refine(entry, tag):
    """MICA starts phenix.real_space_refine in the background (modules/phenix.sh ends with `&`) and run.py ends before it is finished: wait for that process
    instead of starting a second one."""
    pattern = os.path.join(MICA_DIR, "output", "*_%s_MICA_all_atom_model_real_space_refined*.pdb" % entry.lower())
    t0 = time.time()
    while not glob.glob(pattern) and refine_running(entry) and time.time() - t0 < REFINE_WAIT_S and time.time() < HARD_STOP:
        time.sleep(15)
    if time.time() - t0 > 5:
        log("%s: waited %d s for MICA's own Phenix refinement; refined model %s" % (tag, time.time() - t0, "found" if glob.glob(pattern) else "not found"))


def keep_models(entry):
    target = os.path.join(WORK, "models", "mica", entry)
    os.makedirs(target, exist_ok=True)
    kept = []
    for path in sorted(glob.glob(os.path.join(MICA_DIR, "output", "*_%s_MICA*" % entry.lower()))):
        if os.path.isfile(path) and os.path.getsize(path) <= MAX_FILE_BYTES:
            shutil.copy2(path, target)
            kept.append(os.path.basename(path))
    return kept


MAX_FILE_BYTES = 60 * 1000 * 1000


def stage_3(entry, tag, lane, gpu, row, folder, map_path, native, resolution, res):
    """Program 3 and what follows it. Returns the verdict of program 3."""
    device = DEVICE
    windows = res.get("windows")
    if P3["off"]:
        res.update(status="deferred", failed_step="", note="program 3 was switched off in this session after two failures in a row (see the log); the docked state is kept")
        log("%s: DEFERRED. %s" % (tag, res["note"]))
        return "deferred"
    if device == "cpu" and windows and windows > CPU_MAX_WINDOWS:
        res.update(status="deferred", failed_step="", note="program 3 needs a GPU session: %d windows of 64^3 would take about %.1f hours on this CPU (limit %d windows)" % (
            windows, windows * 30 / 3600.0, CPU_MAX_WINDOWS))
        log("%s: DEFERRED. %s" % (tag, res["note"]))
        return "deferred"
    fasta, af3 = os.path.join(folder, entry.lower() + ".fasta"), os.path.join(folder, "AF3_results")
    cmd = [ENV_PY, "run.py", "-m", map_path, "-f", fasta, "-a", af3, "--run_pulchra", "--pulchra_path=modules/pulchra304/src/pulchra", "--run_phenix",
           "--phenix_act=" + STATE["phenix_env"], "--resolution=" + str(resolution), "--device=" + device]
    env = dict(MICA_ENV)
    if device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        threads = str(os.cpu_count() or 2)                     # one program 3 at a time (P3_LOCK): it may use every core
        env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    verdict = run_stage(entry, "3_build", cmd, folder, res, CAPS["3_build_" + device], env, gpu=gpu, tag=tag, lock=P3_LOCK)
    step = res["steps"].setdefault("3_build", {})
    step["device"] = device
    with QUEUE_LOCK:
        if verdict == "ok":
            P3["fails"] = 0
        elif verdict == "failed" and not step.get("timed_out"):
            P3["fails"] += 1
            if P3["fails"] >= 2 and not P3["off"]:
                P3["off"] = True
                log("\nGUARD: program 3 failed on two entries in a row. Program 3 is switched off for the rest of this session (the docking goes on and the docked state is kept). "
                    "Send me the zip before anything else is run: if the cause is the notebook and not MICA, program 3 is run again and the LOG says so.")
    if verdict == "ok":
        wait_for_refine(entry, tag)
        diagnose_refine(entry, map_path, resolution)
    if verdict in ("ok", "failed"):
        res["models_kept"] = keep_models(entry)
        try:
            score_entry(entry, row, folder, native, res)
        except Exception as ex:
            log("%s: scoring stopped: %s" % (tag, ex))
            res["note"] = (res["note"] + " scoring: %s" % ex)[:500]
    return verdict


def run_entry_stages(entry, tag, lane, gpu, row, folder, map_path, native, contour, resolution, res):
    if not os.path.exists(os.path.join(folder, ".done_1_process_af3")):
        restored = restore_state(entry, folder, res)
        if restored:
            log("%s: the state of an earlier session was restored from %s" % (tag, restored))
    res.update(map_geometry(map_path, resolution))
    log("%s: map grid %s, voxel %s A, box %s A -> %s windows of 64^3 for MICA's network; about %s million reflections for Phaser at %s A" % (
        tag, res.get("grid"), res.get("voxel_A"), res.get("box_A"), res.get("windows"), res.get("reflections_M"), resolution))
    if 3 in STAGES and 1 not in STAGES and not os.path.exists(os.path.join(folder, ".done_2_dock")):
        res.update(status="not run", failed_step="state", note="no docked state of this entry was found (add the output of the notebook that docked it as an input)")
        log("%s: NOT RUN. %s" % (tag, res["note"]))
        return
    fasta, af3 = os.path.join(folder, entry.lower() + ".fasta"), os.path.join(folder, "AF3_results")
    verdict = "ok"
    if 2 in STAGES:
        with open(os.path.join(folder, "dock_profile.json"), "w", encoding="utf-8") as f:
            json.dump({"profile": DOCK_PROFILE, "attempts": DOCK_ATTEMPTS}, f)
    res["dock_profile"] = state_profile(folder)
    if 1 in STAGES:
        verdict = run_stage(entry, "1_process_af3", [ENV_PY, "utils/process_AF3_results.py", "-f", fasta, "-a", af3, "-d", "cpu"], folder, res, CAPS["1_process_af3"], dict(MICA_ENV), tag=tag)
        pack_now(60)
    if verdict == "ok" and 2 in STAGES:
        env = dict(MICA_ENV, MICA_BENCH_DOCK_ATTEMPTS=json.dumps(DOCK_ATTEMPTS))
        verdict = run_stage(entry, "2_dock", [ENV_PY, DRIVER, "-m", map_path, "-c", str(contour), "-r", str(resolution), "-f", fasta, "-a", af3, "--phenix_act", STATE["phenix_env"]],
                            folder, res, CAPS["2_dock"], env, tag=tag, watch=[os.path.join(folder, "docking_logs", "bench_docking.csv")])
        d = docking_summary(folder)
        res["docking"] = {k: v for k, v in d.items() if k != "rows"}
        log("%s: docking: %d of %d domains docked, %d s in all" % (tag, d["docked"], d["domains"], d["seconds"]))
        res["dock_mode"] = dock_mode_of(d["domains"], d["docked"])
        snapshot_state(entry, folder)
        pack_now(60)
    if verdict == "ok" and not res.get("dock_mode") and os.path.exists(os.path.join(folder, ".done_2_dock")):
        n_dom, n_hit = docking_state(folder)                    # a session that does not dock: the mode of the restored state
        res["docking"] = {"domains": n_dom, "docked": n_hit, "seconds": ""}
        res["dock_mode"] = dock_mode_of(n_dom, n_hit)
        log("%s: restored docking state: %d of %d domains docked (%s)" % (tag, n_hit, n_dom, res["dock_mode"]))
    if verdict == "ok" and 3 in STAGES:
        verdict = stage_3(entry, tag, lane, gpu, row, folder, map_path, native, resolution, res)
        pack_now(60)
    if verdict == "deferred":
        return
    if verdict == "ok":
        res.update(status="ok" if 3 in STAGES else "docked", failed_step="", note=res.get("note", "") if 3 in STAGES else "stages %s finished; program 3 is for another session" % (list(STAGES),))
    else:
        res["status"] = "failed" if verdict == "failed" else "interrupted"


def process_entry(entry, lane):
    gpu = GPU_OF(lane)
    tag = "lane %d %s" % (lane, entry)
    res = {"entry": entry, "status": "running", "failed_step": "", "note": "", "steps": {}, "models": [], "lane": lane, "stages": list(STAGES), "device": DEVICE if 3 in STAGES else ""}
    STATE["entries"][entry] = res
    t_entry = time.time()
    res["started_utc"] = utc_iso(t_entry)
    row = ROWS[entry]
    log("\n" + "=" * 72 + "\n%s (EMDB %s, %s A, %s modeled residues)\n%s\n" % (tag, row["emdb_id"], row["resolution_A"], row["modeled_residues"], queue_line()) + "=" * 72)
    try:
        with INPUT_LOCK:
            folder, map_path, native, contour, resolution = prepare_entry(entry, row, res)
    except Exception as ex:
        res.update(status="not run", failed_step="inputs", note=str(ex)[:500], seconds=round(time.time() - t_entry))
        log("%s: NOT RUN, the inputs could not be prepared: %s" % (tag, ex))
        return res
    try:
        run_entry_stages(entry, tag, lane, gpu, row, folder, map_path, native, contour, resolution, res)
    finally:
        try:
            snapshot_state(entry, folder)
            copy_docking_logs(entry, folder)
        except Exception as ex:
            log("%s: the state could not be saved: %s" % (tag, ex))
        res["seconds"] = round(time.time() - t_entry)
    return res


def lane_main(lane):
    while True:
        with QUEUE_LOCK:
            if EARLY["stop"] or not QUEUE or HARD_STOP - time.time() < MIN_START_S:
                return
            entry = QUEUE.pop(0)
        try:
            res = process_entry(entry, lane)
        except Exception:
            import traceback
            res = STATE["entries"].setdefault(entry, {"entry": entry, "steps": {}, "models": []})
            res.update(status="failed", failed_step="unexpected error", note=traceback.format_exc()[-600:])
            log("\n%s STOPPED by an unexpected error:\n%s" % (entry, traceback.format_exc()))
        quick = res["status"] in ("failed", "not run") and res.get("failed_step") in ("1_process_af3", "inputs") and res.get("seconds", 1e9) < EARLY_FAIL_S
        with QUEUE_LOCK:
            EARLY["fails"] = EARLY["fails"] + 1 if quick else (0 if res["status"] in ("ok", "docked", "deferred") else EARLY["fails"])
            if EARLY["fails"] >= 2 and not EARLY["stop"]:
                EARLY.update(stop=True, reason="two entries in a row failed in %s within %d minutes: that is probably a problem of the installation or the inputs, not a result" % (res["failed_step"], EARLY_FAIL_S // 60))
                log("\nGUARD: %s. No new entry is started; the entries that wait are 'not run'." % EARLY["reason"])
        pack_now(60)
        save_state()


def run_all():
    threads = [threading.Thread(target=lane_main, args=(i,), daemon=True) for i in range(LANES)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    why = EARLY["reason"] or "the time budget (HOURS) was used up"
    for entry in QUEUE:
        STATE["entries"][entry] = {"entry": entry, "status": "not run", "failed_step": "time" if not EARLY["stop"] else "guard", "note": why, "steps": {}, "models": []}
    QUEUE.clear()


def check_af3_present():
    """Stops at once, before any hour is spent, if the private AlphaFold files of the entries are not attached or are not the checked ones (the hashes of the repository list)."""
    manifest = {}
    with open("data/af3/af3_files_sha256.txt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                h, name = line.split()
                manifest[name] = h
    for entry in ENTRIES:
        jobs = mica_inputs.read_jobs("data/af3/af3_jobs.csv", entry)
        models = find_af3_dir(jobs)
        for r in jobs:
            name = "fold_%s_model_0.cif" % r["job_name"].lower()
            got = sha256_file(mica_inputs.find_model(models, r["job_name"]))
            if got != manifest.get(name):
                raise RuntimeError("the file %s is not the one that was checked (sha256 %s)" % (name, got[:16]))
    log("the private AlphaFold files of %d entries are here and match the checked list" % len(ENTRIES))


if STATE["ok"] and STATE["usalign"] and STATE["phenix_env"]:
    try:
        check_af3_present()
    except Exception as ex:
        fail("the private AlphaFold dataset is missing or is not the checked one (%s). Kaggle: right-hand panel, Add Input, choose your AlphaFold dataset, then run again. "
             "No time was spent." % str(ex)[:300])
    if STATE["ok"]:
        try:
            run_all()
        except Exception:
            import traceback
            fail("unexpected error in the schedule: " + traceback.format_exc()[-500:])
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

SUMMARY_MD = """\
## Step 7. The tables

`mica_status.csv` (one line per entry: status, the step where it stopped, seconds per program, domains docked, windows), `pending.csv` (what a later session still has to do),
`docking_domains.csv` (one line per docking attempt of every domain), `metrics_usalign.csv` (a first look: US-align of every model against the deposited model; the final numbers
come from the scoring notebook)."""
SUMMARY = r'''
STATUS_COLUMNS = ["entry", "status", "failed_step", "note", "lane", "device", "windows", "box_A", "reflections_M", "domains", "domains_docked", "dock_seconds", "p1_seconds", "p3_seconds",
                  "seconds", "contour", "resolution", "parameters_from", "map_sha256", "restored_from", "started_utc", "dock_profile", "dock_mode"]


def status_rows():
    rows = []
    for entry in ENTRIES:
        e = STATE["entries"].get(entry, {"entry": entry, "status": "not run", "failed_step": "time", "note": "not started"})
        d, st = e.get("docking", {}), e.get("steps", {})
        rows.append({"entry": entry, "status": e.get("status"), "failed_step": e.get("failed_step", ""), "note": (e.get("note") or "")[:300], "lane": e.get("lane", ""),
                     "device": e.get("device", ""), "windows": e.get("windows", ""), "box_A": "x".join(str(b) for b in e.get("box_A", [])), "reflections_M": e.get("reflections_M", ""),
                     "domains": d.get("domains", ""), "domains_docked": d.get("docked", ""), "dock_seconds": st.get("2_dock", {}).get("seconds", ""),
                     "p1_seconds": st.get("1_process_af3", {}).get("seconds", ""), "p3_seconds": st.get("3_build", {}).get("seconds", ""), "seconds": e.get("seconds", ""),
                     "contour": e.get("contour", ""), "resolution": e.get("resolution", ""), "parameters_from": e.get("parameters_from", ""), "map_sha256": e.get("map_sha256", ""),
                     "restored_from": e.get("restored_from", ""), "started_utc": e.get("started_utc", ""), "dock_profile": e.get("dock_profile", ""), "dock_mode": e.get("dock_mode", "")})
    return rows


def docking_rows():
    rows = []
    for entry in ENTRIES:
        path = os.path.join(WORK, "docking_logs", entry, "bench_docking.csv")
        if os.path.exists(path):
            rows += [dict(r, entry=entry) for r in read_csv(path)]
    return rows


def write_tables():
    rows = status_rows()
    write_csv("mica_status.csv", STATUS_COLUMNS, rows)
    write_csv("pending.csv", ["entry", "status", "failed_step", "note"], [r for r in rows if r["status"] not in ("ok", "failed")])
    write_csv("docking_domains.csv", ["entry", "model", "attempt", "setting", "settings", "timeout_s", "seconds", "rc", "timed_out", "skipped", "docked", "local_cc", "log"], docking_rows())
    metrics = [dict(m, method="mica") for e in STATE["entries"].values() for m in e.get("models", [])]
    write_csv("metrics_usalign.csv", MICA_METRIC_COLUMNS, metrics)
    return rows


def summary_table(rows):
    log("\n%-5s %-12s %-14s %7s %9s %9s %9s %9s  %s" % ("entry", "status", "stopped in", "windows", "domains", "dock s", "program 3 s", "total s", "note"))
    for r in rows:
        log("%-5s %-12s %-14s %7s %9s %9s %9s %9s  %s" % (r["entry"], r["status"], r["failed_step"], r["windows"], "%s/%s" % (r["domains_docked"], r["domains"]) if r["domains"] != "" else "",
                                                          r["dock_seconds"], r["p3_seconds"], r["seconds"], (r["note"] or "")[:90]))
    log("\n%-5s %-14s %7s %8s %7s %9s %8s  %s" % ("entry", "model", "TM", "aligned", "RMSD", "coverage", "seq id", "chain_comparison"))
    for e in STATE["entries"].values():
        for m in e.get("models", []):
            log("%-5s %-14s %7s %8s %7s %9s %8s  %s" % (m["entry"], m["model_kind"], m["tm_score"], m["aligned_ca_length"], m["ca_rmsd"], m["coverage"], m["seq_identity"], m["chain_comparison"]))


if STATE["ok"] and STATE["usalign"] and STATE["phenix_env"]:
    rows = write_tables()
    summary_table(rows)
    states = [r["status"] for r in rows]
    STATE["complete"] = all(s in ("ok", "failed", "deferred", "docked") for s in states)
    waiting = [r["entry"] for r in rows if r["status"] not in ("ok", "failed")]
    if waiting:
        log("\nTO DO IN A LATER SESSION (pending.csv): %s" % ", ".join(waiting))
    if any(s in ("deferred", "docked") for s in states):
        log("The entries marked deferred/docked need program 3 on a GPU: their state is in the output folder mica_state of this notebook (private). Add this notebook's output as an input of the GPU notebook.")
else:
    log("skipped: an earlier step stopped")
'''

EXTRA_INFO = ('{"mica_commit": globals().get("MICA_COMMIT", ""), "phenix_installer": os.path.basename(globals().get("INSTALLER", "")), "phenix_installer_sha256": STATE["installer_sha256"], '
              '"phenix_version_output": STATE["phenix_version"], "usalign_commit": globals().get("USALIGN_COMMIT", ""), "usalign_version": STATE["usalign_version"], '
              '"input_sha256": globals().get("EXPECTED", {}), "weights": STATE["weights"], "stages": list(globals().get("STAGES", ())), "lanes": globals().get("LANES"), '
              '"program_3_device": globals().get("DEVICE"), "hours_setting": globals().get("HOURS_SETTING"), "dock_attempts": globals().get("DOCK_ATTEMPTS"), '
              '"entries_in_this_notebook": globals().get("ENTRIES", [])}')


RUN_ALL_CALL = "    if STATE[\"ok\"]:\n        try:\n            run_all()\n"
BATCH_PATCH = '''

# PASS 4 (LOG.md, 2026-10-27): MICA's program 3 predicts a map of more than `batch_threshold` = 200 windows in batches of 8 windows (utils/predict.py, "OPTIMIZED BATCH PROCESSING"; its estimate of the
# memory per window is far too small), which needs more than the 15 GB of a T4: "CUDA out of memory" for every map of 216 windows or more (9WQ6, 22FX, 9V4D, in both GPU sessions of 2026-10-27).
# The threshold is raised so that every map is predicted one window at a time ("SINGLE SAMPLE PROCESSING"), which is what MICA itself does for 200 windows or fewer. The network is in
# evaluation mode, so the output of a window does not depend on the others; nothing else of MICA is changed.
BATCH_THRESHOLD = @BATCH_THRESHOLD@


def patch_batch_threshold():
    path = os.path.join(MICA_DIR, "utils", "predict.py")
    old = "self.batch_threshold = 200"
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.count(old) != 1:
        raise RuntimeError("utils/predict.py of MICA does not contain %r exactly once: the batch threshold cannot be raised" % old)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, "self.batch_threshold = %d" % BATCH_THRESHOLD))
    log("MICA's utils/predict.py: batch_threshold raised from 200 to %d (every map is predicted one window at a time)" % BATCH_THRESHOLD)
'''
NODOCK_HOURS = 5.0
NODOCK_ANCHOR = '    log("the private AlphaFold files of %d entries are here and match the checked list" % len(ENTRIES))\n'
NODOCK_PATCH = '''

# SENSITIVITY SESSION (LOG.md, 2026-10-27): MICA's program 3 runs WITHOUT docked AlphaFold input for every entry (MICA's README: "Skip this step if you don't want to use AF3 at input
# level"). The docked models of the restored state are deleted from this session's entry folder (the saved state that was attached as an input is never changed). Everything else is
# the same as in the main GPU session.
FORCE_NO_DOCK = True
_stage_3_with_docking = stage_3


def remove_docked_input(entry, folder):
    removed = []
    for path in glob.glob(os.path.join(folder, "AF3_docked_models", "*_docked.pdb")) + glob.glob(os.path.join(folder, "%s_af3_docked.pdb" % entry)):
        os.remove(path)
        removed.append(os.path.basename(path))
    return removed


def stage_3(entry, tag, lane, gpu, row, folder, map_path, native, resolution, res):
    before = res.get("dock_mode", "") or "unknown"
    removed = remove_docked_input(entry, folder)
    n_dom, n_hit = docking_state(folder)
    res["docking"] = {"domains": n_dom, "docked": n_hit, "seconds": ""}
    res["dock_mode"] = "none"
    res["note"] = "sensitivity session: the docked AlphaFold models of the saved state (mode %s, %d files) were removed; program 3 ran without docked AlphaFold input" % (before, len(removed))
    log("%s: %s" % (tag, res["note"]))
    return _stage_3_with_docking(entry, tag, lane, gpu, row, folder, map_path, native, resolution, res)
'''
NODOCK_STATE_TEXT = (" **This is the sensitivity session: before program 3 it deletes the docked AlphaFold models of each restored state (the attached state itself is not changed), so that "
                     "MICA runs without docked AlphaFold input for every entry. Start it only in addition to the main GPU session, with the same inputs.**")


def build(out_path, kind="cpu", name=None, entries=None, hours=None, lanes=None, stages=None, device=None, require_gpu=None, entries_path=None, phenix_url=P.PHENIX_URL,
          mica_repo=P.MICA_REPO, mica_commit=P.MICA_COMMIT, usalign_repo=P.USALIGN_REPO, usalign_commit=P.USALIGN_COMMIT, env_python=None, usalign_bin=None, net_hosts=NET_HOSTS,
          weights_url=P.WEIGHTS_URL, zenodo_api=P.ZENODO_API, dock_attempts=None, cpu_max_windows=CPU_MAX_WINDOWS, dock_profile=DOCK_PROFILE, state_profile="default", nodock=False, batch_threshold=None):
    """Writes the notebook. `kind` gives the defaults (cpu: stages 1, 2, 3, two entries at a time; gpu: stage 3, three entries at a time, GPU required). The keyword arguments from
    `entries_path` on exist for the tests (local servers, a ready environment)."""
    import re
    k = KINDS[kind]
    name = name or ("mica_gpu_nodock" if nodock else "mica_" + kind)
    if nodock and kind != "gpu":
        raise ValueError("the sensitivity session is a GPU session (program 3 only)")
    stages = tuple(stages or k["stages"])
    lanes = lanes or k["lanes"]
    hours = hours or (NODOCK_HOURS if nodock else k["hours"])
    device = device or k["device"]
    require_gpu = k["require_gpu"] if require_gpu is None else require_gpu
    if entries is None:
        entries = MN.seeded_order(MN.read_entries(entries_path))
    entries = list(entries)
    if not entries:
        raise ValueError("no entries")
    m = re.search(r"version=([^&]+)&file=([^&]+)$", phenix_url)
    if not m:
        raise ValueError("the Phenix link must end in ?version=...&file=...: %s" % phenix_url)
    version, installer = m.group(1), m.group(2)
    attempts = DOCK_ATTEMPTS if dock_attempts is None else dock_attempts
    if state_profile == "default":
        state_profile = dock_profile if (kind == "gpu" or (3 in stages and 2 not in stages)) else None
    if 3 in stages:
        program3 = ("The network needs a GPU for big maps: an entry whose map has more than %d windows (a few minutes on a GPU, hours on this CPU) is marked *deferred* and is "
                    "finished in a short GPU session." % cpu_max_windows)
    else:
        program3 = ("It is **not run in this session**: the GPU notebook runs it for every entry, from the docked state that this session leaves in `mica_state/` (status *docked*).")
    file_cells, expected = nbkit.file_cells(FILES, entries_path)
    extra_info = EXTRA_INFO
    if nodock:
        extra_info = extra_info.replace('"stages": list(', '"force_no_dock": True, "stages": list(')
    if batch_threshold:
        extra_info = extra_info.replace('"stages": list(', '"batch_threshold": globals().get("BATCH_THRESHOLD"), "stages": list(')
    mapping = {"@PHENIX_URL@": phenix_url, "@PHENIX_VERSION@": version, "@INSTALLER@": installer, "@MICA_REPO@": mica_repo, "@MICA_COMMIT@": mica_commit,
               "@WEIGHTS_URL@": weights_url, "@ZENODO_API@": zenodo_api, "@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit,
               "@UV_VERSION@": P.UV_VERSION, "@PY_VERSION@": P.PY_VERSION, "@ENTRIES@": repr(entries), "@ENTRY_LIST@": ", ".join(entries),
               "@REQUIRE_GPU@": repr(bool(require_gpu)), "@ENV_PYTHON@": repr(env_python), "@USALIGN_BIN@": repr(usalign_bin), "@NET_HOSTS@": repr(tuple(net_hosts)),
               "@EXPECTED@": repr(expected), "@NOTEBOOK@": name, "@EXTRA_INFO@": extra_info, "@STAGES@": repr(stages), "@LANES@": str(lanes), "@P3_DEVICE@": device,
               "@CPU_MAX_WINDOWS@": str(cpu_max_windows), "@DOCK_ATTEMPTS@": repr(attempts), "@DOCK_STAGE_CAP_S@": str(DOCK_STAGE_CAP_S), "@P3_CAP_CPU_S@": str(P3_CAP_CPU_S),
               "@P3_CAP_GPU_S@": str(P3_CAP_GPU_S), "@HOURS@": repr(float(hours)), "@HOURS_TEXT@": "%g" % hours,
               "@DOCK_TEXT@": dock_text(attempts), "@PROGRAM3_TEXT@": program3, "@DOCK_CAP_H@": "%g" % (DOCK_STAGE_CAP_S / 3600.0), "@DOCK_PROFILE@": dock_profile,
               "@STATE_PROFILE@": repr(state_profile), "@TIME_TEXT@": "expect 3 to 9 hours" if (2 in stages and 3 not in stages) else "about 10 hours",
               "@STATE_TEXT@": ("Only state that was docked with the profile `%s` is used (`dock_profile.json` in the entry folder); the output of a session with another docking set-up "
                                "is ignored even if you add it by mistake." % state_profile) if state_profile else ""}
    if nodock:
        mapping["@STATE_TEXT@"] += NODOCK_STATE_TEXT
    fl = lambda text: nbkit.fill(text, mapping).strip("\n") + "\n"
    step6_filled = nbkit.fill(P.STEP6, mapping)
    step6_pre = nbkit.cut_before(step6_filled, "def mica_programs(")
    helpers = nbkit.function_sources(step6_filled, ["classify", "chain_comparison", "score_entry", "diagnose_refine"])
    production = nbkit.fill(PRODUCTION, mapping)
    if nodock:
        production = nbkit.replace_once(production, NODOCK_ANCHOR, NODOCK_ANCHOR + NODOCK_PATCH)
    if batch_threshold:
        production = nbkit.replace_once(production, NODOCK_ANCHOR, NODOCK_ANCHOR + BATCH_PATCH.replace("@BATCH_THRESHOLD@", str(int(batch_threshold))))
        production = nbkit.replace_once(production, RUN_ALL_CALL, RUN_ALL_CALL.replace("            run_all()\n", "            patch_batch_threshold()\n            run_all()\n"))
    # the pilot's step 5 fetches the weights, which program 3 needs; a notebook without program 3 does not download them
    step5 = P.STEP5 if 3 in stages else nbkit.replace_once(P.STEP5, "fetch_mica() and fetch_weights() and import_mica()", "fetch_mica() and import_mica()")
    intro = INTRO_GPU if kind == "gpu" else INTRO_CPU
    cells = [("intro", nbkit.md(nbkit.fill(intro, mapping))), ("settings-title", nbkit.md(SETTINGS_MD)), ("settings", nbkit.code(fl(SETTINGS))),
             ("step1-title", nbkit.md(P.STEP1_MD)), ("step1", nbkit.code(fl(P.STEP1))),
             ("step2-title", nbkit.md(P.STEP2_MD)), ("step2-dirs", nbkit.code(P.STEP2_DIRS.strip("\n") + "\n"))]
    cells += file_cells
    cells += [("step2-check", nbkit.code(fl(P.STEP2_CHECK))),
              ("robust-title", nbkit.md(nbkit.fill(nbkit.ROBUST_MD, mapping))), ("robust", nbkit.code(fl(nbkit.ROBUST))),
              ("step3-title", nbkit.md(P.STEP3_MD)), ("step3", nbkit.code(fl(P.STEP3))),
              ("step4-title", nbkit.md(P.STEP4_MD)), ("step4", nbkit.code(fl(P.STEP4))),
              ("step5-title", nbkit.md(P.STEP5_MD)), ("step5", nbkit.code(fl(step5))),
              ("step6-title", nbkit.md(nbkit.fill(PRODUCTION_MD, mapping))), ("step6", nbkit.code(step6_pre.strip("\n") + "\n\n\n" + helpers.strip("\n") + "\n\n\n" + production.strip("\n") + "\n")),
              ("step7-title", nbkit.md(SUMMARY_MD)), ("step7", nbkit.code(fl(SUMMARY))),
              ("final-title", nbkit.md(nbkit.FINAL_MD)), ("final", nbkit.code(fl(nbkit.FINAL)))]
    nbkit.assemble(out_path, name + ".ipynb", cells)
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=sorted(KINDS), default="cpu")
    p.add_argument("--shards", type=int, default=1, help="cpu: build this many notebooks mica_cpu_a, mica_cpu_b, ... (entry i goes to shard i mod n)")
    p.add_argument("--name", help="name of the notebook (default mica_cpu_a ... or mica_gpu)")
    p.add_argument("--entries", nargs="+", help="entries in the order to be run (default: all, in the order of the seeded selection)")
    p.add_argument("--done", nargs="*", default=[], help="mica_status.csv files of earlier sessions: entries with status ok or failed are left out")
    p.add_argument("--redo", nargs="*", default=[], help="entries to run again although an earlier session finished them")
    p.add_argument("--hours", type=float)
    p.add_argument("--lanes", type=int)
    p.add_argument("--stages", type=int, nargs="+")
    p.add_argument("--outdir", default=HERE)
    p.add_argument("--phenix-url", default=P.PHENIX_URL)
    p.add_argument("--pass4", action="store_true", help="build the four GPU notebooks of pass 4 (mica_gpu2_a, mica_gpu2_b, mica_gpu2_nodock_a, mica_gpu2_nodock_b: program 3 for the 12 entries of PASS4_SHARDS, one window at a time)")
    p.add_argument("--pass5", action="store_true", help="build mica_gpu3 (GPU session: program 3 for the entries of PASS5_ENTRIES that the time budget cut in pass 4, one window at a time, %g hours)" % PASS5_HOURS)
    p.add_argument("--nodock", action="store_true", help="build the sensitivity notebook mica_gpu_nodock (GPU session: program 3 without docked AlphaFold input, %g hours)" % NODOCK_HOURS)
    p.add_argument("--pass3", action="store_true", help="build the four CPU notebooks of pass 3 (mica_dock3_a ... d: programs 1 and 2 only, the entries of PASS3_SHARDS)")
    args = p.parse_args(argv)
    if args.pass3:
        for letter, entries in PASS3_SHARDS.items():
            name = "mica_dock3_" + letter
            out = os.path.join(args.outdir, name + ".ipynb")
            build(out, kind="cpu", name=name, entries=entries, hours=args.hours, lanes=args.lanes, stages=(1, 2), phenix_url=args.phenix_url)
            print("wrote %s: %d entries (%s)" % (out, len(entries), ", ".join(entries)))
        return 0
    if args.pass5:
        name = "mica_gpu3"
        out = os.path.join(args.outdir, name + ".ipynb")
        build(out, kind="gpu", name=name, entries=gpu_order(PASS5_ENTRIES), hours=args.hours or PASS5_HOURS, phenix_url=args.phenix_url, batch_threshold=BATCH_THRESHOLD_NEW)
        print("wrote %s: %d entries (%s)" % (out, len(PASS5_ENTRIES), ", ".join(gpu_order(PASS5_ENTRIES))))
        return 0
    if args.pass4:
        for variant, nodock in (("", False), ("nodock_", True)):
            for letter, entries in PASS4_SHARDS.items():
                name = "mica_gpu2_%s%s" % (variant, letter)
                out = os.path.join(args.outdir, name + ".ipynb")
                build(out, kind="gpu", name=name, entries=gpu_order(entries), hours=args.hours or PASS4_HOURS["nodock" if nodock else "main"], phenix_url=args.phenix_url, nodock=nodock,
                      batch_threshold=BATCH_THRESHOLD_NEW)
                print("wrote %s: %d entries (%s)" % (out, len(entries), ", ".join(gpu_order(entries))))
        return 0
    if args.nodock:
        entries = args.entries or gpu_order(MN.seeded_order(MN.read_entries()))
        name = args.name or "mica_gpu_nodock"
        out = os.path.join(args.outdir, name + ".ipynb")
        build(out, kind="gpu", name=name, entries=entries, hours=args.hours, lanes=args.lanes, stages=args.stages, phenix_url=args.phenix_url, nodock=True)
        print("wrote %s: %d entries (%s)" % (out, len(entries), ", ".join(entries)))
        return 0
    order = args.entries or MN.seeded_order(MN.read_entries())
    if args.kind == "gpu" and not args.entries:
        order = gpu_order(order)
    if args.done:
        order = pending_entries(order, args.done, args.redo)
    if not order:
        print("nothing is left to do")
        return 0
    jobs = []
    if args.kind == "cpu" and args.shards > 1:
        for i in range(args.shards):
            sub = shard_entries(order, i, args.shards)
            if sub:
                jobs.append(("%s_%s" % (args.name or "mica_cpu", "abcdefghij"[i]), sub))
    else:
        jobs.append((args.name or ("mica_" + args.kind), order))
    for name, entries in jobs:
        out = os.path.join(args.outdir, name + ".ipynb")
        expected = build(out, kind=args.kind, name=name, entries=entries, hours=args.hours, lanes=args.lanes, stages=args.stages, phenix_url=args.phenix_url)
        print("wrote %s: %d entries (%s), %d embedded files" % (out, len(entries), ", ".join(entries), len(expected)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
