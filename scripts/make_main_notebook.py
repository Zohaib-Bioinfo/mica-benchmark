#!/usr/bin/env python3
"""Build scripts/gpu_<batch>.ipynb: the production runs of the four GPU methods of the benchmark (ModelAngelo, E3-CryoFold, EMProt, EModelX(+AF)) on the entries of
data/entries_2026.csv, in one Kaggle session.

What it adds to the pilot notebooks (make_methods_notebook.py, make_modelangelo_notebook.py), whose cells it reuses so that the code that ran on Kaggle is not copied twice:

* many entries: the work is a list of (entry, method) tasks in the order of the seeded selection (the two strata interleaved, so that any prefix of the list is a fair
  sample and the entries to drop first, from the bottom, are the ones of decision 12 of the spec);
* two GPUs (Kaggle "T4 x2") run two programs at the same time, one on each GPU, so a session of N hours does 2N GPU hours of work; the two methods that are known to
  allocate N x N arrays in main memory (E3-CryoFold and EModelX) never run at the same time; a program that the system kills while the other GPU also runs is run
  again, alone, at the end (it may have been the neighbour's memory), and both outcomes are recorded;
* a time budget (the `HOURS` cell): no run is started that cannot finish before the budget ends, a run that is cut by the end of the session is "interrupted" (not a
  failure) and goes back into the list of the next session; a run limit that is the same for every method (`RUN_LIMIT`);
* a method that crashes (not an out-of-memory kill) within an hour on two entries in a row is left alone for the rest of the session (that is probably an installation
  problem, not a result: the two runs stay in the tables, the next session looks at the logs first);
* `method_status.csv` (one line per task), `pending.csv` (what the next session must do) and, per run, the start and end time, the GPU and whether the other GPU was busy.

    python make_main_notebook.py --batch batch1                          # all entries, all four methods
    python make_main_notebook.py --batch batch2 --done <result folder>/method_status.csv ...   # what is left after earlier sessions
    python make_main_notebook.py --batch cryoatom_a --methods cryoatom1 --hours 11             # CryoAtom v1.0.0 alone (see below)

Run it on Kaggle with the GPU "T4 x2" and the private AlphaFold dataset attached, as "Save & Run All (Commit)".

CryoAtom v1.0.0 (added on 2026-10-31 at Prof. Cheng's request) has a notebook of its own, with the same scheduler, scoring and zips: it needs no AlphaFold dataset, a run may take 10 hours
instead of 3 (its pilot did not finish in 2), the time of a run grows with the size of the entry (the budget check uses the residues of the entry and the seconds per residue of the runs that
finished in the session) and the ESM-2 weights are fetched once, before the two GPUs start. The four other methods are not changed by any of this.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_methods_notebook as MM  # noqa: E402
import make_modelangelo_notebook as MA  # noqa: E402
import make_pilot_notebook as P  # noqa: E402
import nbkit  # noqa: E402

RUN_LIMIT_S = 3 * 3600                          # per run, the same for every method (LOG.md, 2026-10-22)
CRYOATOM_LIMIT_S = 10 * 3600                    # per run of CryoAtom v1.0.0 (LOG.md, 2026-10-31): its pilot did not finish in 2 hours
CRYOATOM_S_PER_RESIDUE = 51                     # the pilot on 26DK: about 2.7 hours for 191 deposited residues on a T4 (LOG.md, 2026-10-22); used until a run of the session has finished
ORDER = ["modelangelo", "e3", "emprot", "emodelx"]
ORDER_ALL = ORDER + ["cryoatom1"]
EXPLOSIVE = ("e3", "emodelx")

MODELANGELO_SPEC = {
    "key": "modelangelo", "label": "ModelAngelo", "repo": MA.MA_REPO, "commit": MA.MA_COMMIT, "python": MA.MA_PY_VERSION, "req": "env/modelangelo_requirements.txt",
    "timeout": RUN_LIMIT_S, "uses_af3": False, "bundle": MA.MA_BUNDLE, "weights_url": None, "weights_min_bytes": 0, "gdrive_id": None, "import_test": MA.IMPORT_TEST,
}


def make_specs():
    specs = {k: copy.deepcopy(MM.SPECS[k]) for k in ("e3", "emprot", "emodelx")}
    specs["modelangelo"] = copy.deepcopy(MODELANGELO_SPEC)
    for s in specs.values():
        s["timeout"] = RUN_LIMIT_S
    return {k: specs[k] for k in ORDER}


SPECS = make_specs()


def cryoatom_spec():
    """CryoAtom v1.0.0, the code of the paper: the spec of the methods pilot with this notebook's run limit, its time model (seconds per residue) and the ESM-2 weights that the program
    fetches by torch.hub at its first run (here they are fetched once, before the two GPUs start)."""
    s = copy.deepcopy(MM.SPECS["cryoatom1"])
    s.update(timeout=CRYOATOM_LIMIT_S, s_per_residue=CRYOATOM_S_PER_RESIDUE, oom_is_result=True,
             esm={"call": "esm2_t33_650M_UR50D", "file": "esm2_t33_650M_UR50D.pt", "min_bytes": 500000000})     # the file is about 2.5 GB; the call itself loads it, so a short file fails there
    return s


SPECS_ALL = dict(SPECS, cryoatom1=cryoatom_spec())
ENV_FILES = [SPECS[k]["req"] for k in ORDER]
BASE_FILES = (("scripts/nbtools.py", "scripts/nbtools.py"), ("scripts/mica_inputs.py", "scripts/mica_inputs.py"), ("scripts/evaluate_models.py", "scripts/evaluate_models.py"),
              ("data/af3/af3_jobs.csv", "data/af3/af3_jobs.csv"), ("data/af3/af3_files_sha256.txt", "data/af3/af3_files_sha256.txt"),
              ("data/entries_2026.csv", "data/entries_2026.csv"))
FILES = BASE_FILES + tuple((f, f) for f in ENV_FILES)
NET_HOSTS = (("PyPI", "https://pypi.org/simple/uv/", True),
             ("GitHub", "https://github.com/3dem/model-angelo", True),
             ("GitHub files", "https://github.com/A4Bio/E3-CryoFold/releases/download/checkpoint/models.zip", False),
             ("Zenodo (ModelAngelo)", "https://zenodo.org/api/records/7942241", False),
             ("Meta ESM weights", "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt", False),
             ("HUST (EMProt)", "http://huanglab.phys.hust.edu.cn/EMProt/pretrained_weights/weights.tgz", False),
             ("Google Drive", "https://drive.google.com/uc?id=13BKzEBfL0uubYgcJJTGSZ-PkO9hXAB9X", False),
             ("EBI FTP", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-80558/header/emd-80558.xml", True),
             ("RCSB", "https://files.rcsb.org/header/26DK.cif", True))
NET_HOSTS_CRYOATOM = (("PyPI", "https://pypi.org/simple/uv/", True),
                      ("GitHub", "https://github.com/YangLab-SDU/CryoAtom", True),
                      ("SDU (CryoAtom weights)", "https://yanglab.qd.sdu.edu.cn/CryoAtom/download/checkpoints_v1.0.zip", False),
                      ("Meta ESM weights", "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt", False),
                      ("EBI FTP", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-80558/header/emd-80558.xml", True),
                      ("RCSB", "https://files.rcsb.org/header/26DK.cif", True))


# ---------------------------------------------------------------------------------------------------------- the list of tasks
def seeded_order(rows):
    """The entries of data/entries_2026.csv in the order in which they are run: each stratum in the order of the seeded shuffle (seed_rank), the two strata interleaved
    (the stratum of the better resolution first). Dropping entries "from the bottom of the seeded order" (decision 12) then means dropping from the end of this list."""
    strata = {}
    for r in rows:
        strata.setdefault(r["stratum"], []).append(r)
    names = sorted(strata, key=lambda s: (0 if s.startswith("le") else 1, s))
    for s in names:
        strata[s].sort(key=lambda r: int(r["seed_rank"]))
    order = []
    for i in range(max(len(v) for v in strata.values())):
        for s in names:
            if i < len(strata[s]):
                order.append(strata[s][i]["pdb_id"])
    return order


def read_entries(entries_path=None):
    with open(entries_path or os.path.join(ROOT, "data", "entries_2026.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def all_tasks(entries, methods=ORDER):
    """Entry by entry, the four methods of an entry one after the other: the first entries are complete as early as possible."""
    return [[e, m] for e in entries for m in methods]


def pending_tasks(tasks, status_csvs, redo=()):
    """The tasks that no earlier session finished: a task is finished when its status is `ok` or `failed` (a failure is a result), unless it is in `redo` ("ENTRY:method"; for a
    failure that turned out to be a problem of the notebook and not of the method)."""
    done = set()
    for path in status_csvs:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["status"] in ("ok", "failed"):
                    done.add((r["entry"], r["method"]))
    again = {tuple(x.split(":")) for x in redo}
    return [t for t in tasks if tuple(t) not in done or tuple(t) in again]


# ---------------------------------------------------------------------------------------------------------- text of the cells
INTRO = """\
# MICA benchmark: GPU session `@BATCH@` (ModelAngelo, E3-CryoFold, EMProt, EModelX(+AF))

This notebook builds atomic models from the cryo-EM maps of the benchmark with four methods, one after the other for each entry, and scores every model with US-align.
ModelAngelo and E3-CryoFold get the map and the sequence only; EMProt and EModelX(+AF) get the same AlphaFold Server predictions as MICA (private dataset, decision 6 of
the spec). Each method has its own Python environment (exact pins, `uv`), its own source at a pinned commit and its own weights.

@LANES_TEXT@
* The list of tasks (entry, method) is @N_TASKS@ long. The entries come in the order of the seeded selection (the two resolution classes interleaved). When the time budget
  (`HOURS` in the first cell) is nearly used up, no new run is started; what is not done is written to `pending.csv` and is run in the next session.
* A run that crashes is a result and stays in the tables (with the reason); a run that the end of the session cuts is **not** a result: it is run again.
* Every run has the same time limit, @LIMIT_TEXT@ hours, for every method.

**What you do (Kaggle, background mode)**: accelerator **GPU T4 x2** (not P100), Internet **On**, the private AlphaFold dataset added as input (no secrets), `HOURS` in the
first cell (the hours of GPU time you allow), then *Save Version -> Save & Run All (Commit)*. It runs on Kaggle's servers; you can close the tab and your Wi-Fi does not matter.
When the version shows *Succeeded*, open it and download the zip `@NOTEBOOK@_result_....zip` from its Output. Send it to me. **Do not stop or cancel the run**: a cancelled run keeps nothing.
"""

SETTINGS_MD = "## Settings (the only number you may have to change)"
SETTINGS = '''\
# HOURS = how many hours of Kaggle GPU time this run may use. Kaggle gives about 30 GPU hours per week for free. If you have less than 9 hours left this week,
# type (hours left minus 1) here. The notebook plans its work to end before HOURS are used and never goes above 11.
HOURS = @HOURS@
# GUARD_MINUTES: a method that crashes (not an out-of-memory kill) within this many minutes on two entries in a row is paused for the rest of the session, because that is probably a
# problem of the installation and not a result. Leave it as it is unless I tell you to change it (0 switches the guard off).
GUARD_MINUTES = @GUARD_MINUTES@
'''

TOOLS_MD = """\
## Step 3. Tools shared by the methods, the schedule and the limits

`uv` and the environment builder, the downloader, the archive helpers, the program runner (time limit, GPU and RAM log of the GPU it runs on, a progress line every two
minutes), the scoring, and the scheduler: a list of tasks, one worker per GPU, the rules that keep the two memory-hungry methods apart and that stop the work in time.
Nothing is installed or downloaded here yet."""

TOOLS_TAIL = r'''
def run_program(label, cmd, cwd, env, logname, gpu, watch=(), timeout=7200):
    """Runs a method's program with a time limit, a progress line every two minutes and a log of the memory of its GPU and of the free RAM;
    returns (exit code, last lines, timed_out, info). The output of the program goes to its log file (in the zip), not to the notebook."""
    logf = os.path.join(WORK, "logs", logname)
    mon = GpuMonitor(os.path.join(WORK, "gpu", logname.replace(".log", ".csv")), gpu)
    mon.start()
    t0 = time.time()
    rc, last, timed_out = run_watched(cmd, label, watch=[logf] + list(watch), env=env, cwd=cwd, log_file=logf, timeout=timeout, every=1, quiet=True)
    info = mon.stop()
    info.update(rc=rc, seconds=round(time.time() - t0), timed_out=timed_out, started=t0, ended=time.time(), gpu=gpu)
    return rc, last, timed_out, info


class GpuMonitor(Monitor):
    """The Monitor of step 1, for the GPU of one lane (nvidia-smi -i <n>): the base class would always read the first GPU."""

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
    """state.json from a snapshot: two workers change STATE while it is written, and json.dump refuses a dictionary that changes size."""
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


INPUTS = {}


@PREPARE_FUNCTIONS@


METRIC_COLUMNS = ["method", "entry", "model_kind", "model_file", "status", "note", "tm_score", "tm_score_pred_norm", "ca_rmsd", "aligned_ca_length", "seq_identity",
                  "pred_residues", "native_residues", "coverage"]
ROWS = []
SETUPS, PLANS, READY = {}, {}, {}


# ---- the time budget, the limits and the rules of the schedule
RUN_LIMIT = @RUN_LIMIT@                                           # seconds per run, the same for every method unless the spec of a method has its own `timeout` (LOG.md)
HOURS_SETTING = float(globals().get("HOURS", 9.0))
SESSION_S = min(HOURS_SETTING, 11.0) * 3600
RESERVE_S = float(os.environ.get("MICA_RESERVE_S", "600"))        # the end of the budget is kept for the summary and the zip
MIN_START_S = float(os.environ.get("MICA_MIN_START_S", "900"))    # a run is not started if less than this (or 1.25 times the usual time of its method) is left
POLL_S = float(os.environ.get("MICA_POLL", "10"))
GUARD_MINUTES = float(globals().get("GUARD_MINUTES", 60))
QUICK_CRASH_S = float(os.environ.get("MICA_QUICK_CRASH", GUARD_MINUTES * 60))  # a crash (not an out-of-memory kill) sooner than this is probably a problem of the installation
T0 = STATE["t0"]
HARD_STOP = T0 + SESSION_S - RESERVE_S                            # a program that still runs at this time is stopped ("interrupted", not a failure)
EXPLOSIVE = @EXPLOSIVE@                                           # the pilot: AutoEM allocates np.zeros([6, N, N]) and the process is killed (out of memory); never two of them at once
TASKS = @TASKS@                                                   # [entry, method]: the order in which the tasks are taken
SCHED = threading.RLock()
INPUT_LOCK = threading.RLock()
PENDING = [tuple(t) for t in TASKS if t[1] in RUN_METHODS]
BUSY = {}                                                         # gpu -> what runs on it now
INTERVALS = []                                                    # (gpu, start, end) of the programs that finished
DURATIONS = {}                                                    # method -> seconds of its successful runs
SIZES = @SIZES@                                                   # entry -> residues of the deposited model (data/entries_2026.csv)
RATES = {}                                                        # method -> seconds per residue of its successful runs (only for a method whose spec has `s_per_residue`)
CRASH_STREAK = {}
RETRY_ALONE = []
LANES = []
MAX_LANES = @MAX_LANES@                                             # how many GPUs run work at the same time (1: one program at a time, for the entries that ran out of RAM next to another program)


def utc_iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def need_s(key, entry=None):
    """Seconds that must be left in the budget to start a run of this method (on this entry): 1.25 times its usual time, at least MIN_START_S. The usual time of a method is the mean of its
    successful runs in this session. For a method whose time grows with the size of the entry (`s_per_residue` in its spec) it is the residues of the entry times the largest seconds per
    residue seen in this session, or the rate of the pilot until a run has finished."""
    rate = SPECS[key].get("s_per_residue")
    if rate and entry in SIZES:
        seen = RATES.get(key) or []
        return max(MIN_START_S, 1.25 * (max(seen) if seen else rate) * SIZES[entry])
    d = DURATIONS.get(key) or []
    return max(MIN_START_S, 1.25 * sum(d) / len(d)) if d else MIN_START_S


def overlapped(gpu, t0, t1):
    """True if a program of another GPU ran at some time between t0 and t1."""
    with SCHED:
        spans = [(a, b) for g, a, b in INTERVALS if g != gpu] + [(b["program_start"], t1) for g, b in BUSY.items() if g != gpu and b.get("program_start")]
    return any(a < t1 and b > t0 for a, b in spans)


def mark(entry, key, status, note, failed_step=""):
    m = STATE["methods"][key]
    e = m["entries"].setdefault(entry, {"entry": entry, "method": key, "runs": {}, "models": []})
    e.update(status=status, failed_step=failed_step, note=note)


def disable_method(key, reason, status="disabled", failed_step=""):
    """No more tasks of this method in this session; its waiting tasks are 'not run' (they are not results: the next session runs them)."""
    with SCHED:
        m = STATE["methods"][key]
        m.update(status=status, failed_step=failed_step, note=reason)
        for t in [t for t in PENDING if t[1] == key]:
            PENDING.remove(t)
            mark(t[0], key, "not run", reason, failed_step)


def take_task(gpu):
    """The next task for this GPU: (entry, method), "wait" (something waits for the other GPU) or None (nothing more can start on this GPU)."""
    with SCHED:
        now = time.time()
        blocked = False
        for t in list(PENDING):
            entry, key = t
            if STATE["methods"][key]["status"] != "ready":
                continue
            if HARD_STOP - now < need_s(key, entry):
                continue                                                  # too late for this method (or this entry); a shorter one may still fit
            if key in EXPLOSIVE and any(b["key"] in EXPLOSIVE for g, b in BUSY.items() if g != gpu):
                blocked = True
                continue
            PENDING.remove(t)
            BUSY[gpu] = {"entry": entry, "key": key, "start": now, "program_start": None}
            return t
        return "wait" if blocked else None


def queue_line():
    done = sum(1 for m in STATE["methods"].values() for e in m["entries"].values() if e.get("status") == "ok")
    failed = sum(1 for m in STATE["methods"].values() for e in m["entries"].values() if e.get("status") == "failed")
    return "queue: %d waiting, %d ok, %d failed; %.1f of %.1f hours used" % (len(PENDING), done, failed, (time.time() - T0) / 3600, SESSION_S / 3600)


def run_task(gpu, entry, key, attempt=1):
    """One method on one entry on one GPU. Returns "ok", "failed", "quick_crash", "interrupted", "killed_corun", "inputs" or "time"."""
    spec, m = SPECS[key], STATE["methods"][key]
    env_py, repo = READY[key]
    first = m["entries"].get(entry) if attempt > 1 else None
    e = m["entries"][entry] = {"entry": entry, "method": key, "status": "not run", "failed_step": "", "note": "", "runs": {}, "models": [], "gpu": gpu, "attempt": attempt}
    if first:
        e["first_attempt"] = {"status": first.get("status"), "note": first.get("note"), "gpu": first.get("gpu"), "runs": first.get("runs")}
    tag = "GPU%d %s %s" % (gpu, spec["label"], entry)
    t_task = time.time()
    e["started_utc"] = utc_iso(t_task)
    log("\n" + "-" * 72 + "\n%s (attempt %d)\n%s\n" % (tag, attempt, queue_line()) + "-" * 72)
    try:
        with INPUT_LOCK:
            ctx = prepare_inputs(entry)
            runs = PLANS[key](spec, ctx, repo, env_py)
    except Exception as ex:
        e.update(status="not run", failed_step="inputs", note=str(ex)[:600])
        log("%s: NOT RUN, the inputs could not be prepared: %s" % (tag, ex))
        return "inputs"
    verdict = "ok"
    run_limit = spec.get("timeout") or RUN_LIMIT                          # the limit of one run of this method (the same for every method unless its spec says otherwise)
    for r in runs:
        limit = min(run_limit, HARD_STOP - time.time())
        if limit < min(60, MIN_START_S):
            e.update(status="not run", failed_step="", note="no time left in the session")
            return "time"
        cut = limit < run_limit - 1
        env = dict(r["env"], CUDA_VISIBLE_DEVICES=str(gpu))
        shutil.rmtree(r["out_dir"], ignore_errors=True)
        os.makedirs(r["out_dir"], exist_ok=True)
        log("\n=== %s, %s: %s" % (tag, r["variant"], " ".join(str(c) for c in r["cmd"][1:])[:600]))
        with SCHED:
            BUSY[gpu]["program_start"] = time.time()
        rc, last, timed_out, info = run_program(tag, r["cmd"], r["cwd"], env, "%s_%s_%s.log" % (entry, key, r["variant"]), gpu, watch=r.get("watch", ()), timeout=limit)
        with SCHED:
            INTERVALS.append((gpu, info["started"], info["ended"]))
            BUSY[gpu]["program_start"] = None
        info.update(limit_s=round(limit), other_gpu_busy=overlapped(gpu, info["started"], info["ended"]), started_utc=utc_iso(info["started"]), ended_utc=utc_iso(info["ended"]))
        e["runs"][r["variant"]] = info
        log("%s %s finished: exit code %s, %d s, GPU memory peak %s MB, lowest free RAM %s GB, other GPU busy: %s" % (
            tag, r["variant"], rc, info["seconds"], info["peak_gpu_mb"], info["min_ram_available_gb"], info["other_gpu_busy"]))
        produced = {k: p for k, p in r["outputs"].items() if os.path.exists(p) and os.path.getsize(p) > 500}
        main_ok = r["main"] in produced
        if timed_out and cut:
            keep_results(key, entry, r["variant"], r["out_dir"], {})
            e.update(status="interrupted", failed_step=r["variant"], note="stopped by the end of the session after %d s; run again in the next session" % info["seconds"])
            log("%s: INTERRUPTED by the end of the session (not a failure)" % tag)
            pack_now()
            return "interrupted"
        if rc == -9 and not timed_out and info["other_gpu_busy"] and attempt == 1:
            keep_results(key, entry, r["variant"], r["out_dir"], {})
            e.update(status="killed_corun", failed_step=r["variant"], note="killed by the system (exit code -9, probably out of memory) while the other GPU's program ran; run again alone")
            log("%s: KILLED while the other GPU ran; it is run again alone at the end" % tag)
            RETRY_ALONE.append((entry, key))
            pack_now()
            return "killed_corun"
        keep_results(key, entry, r["variant"], r["out_dir"], produced)
        if rc != 0 or timed_out or not main_ok:
            gpu_oom = bool(spec.get("oom_is_result")) and any("out of memory" in line.lower() for line in last)    # CryoAtom: 15 GB of GPU memory may be too little for an entry; that is a result
            crashed = (rc not in (0, -9) or (rc == 0 and not main_ok)) and not timed_out and not gpu_oom          # an out-of-memory kill (-9) is a result, a crash is a suspect
            verdict = "quick_crash" if (crashed and info["ended"] - info["started"] < QUICK_CRASH_S) else "failed"
            if timed_out:
                note = "timeout: the run limit of %d s" % limit
            elif rc == -9:
                note = "killed by the system (exit code -9, probably out of memory%s)" % ("; also when it ran alone" if attempt > 1 else "")
            elif gpu_oom:
                note = "out of GPU memory (exit code %s; the error message of the program)" % rc
            else:
                note = "exit code %s" % rc
            if not main_ok:
                note += "; the model file %s was not written" % os.path.basename(r["outputs"][r["main"]])
            e.update(status="failed", failed_step=r["variant"], note=note)
            log("%s: FAILED: %s. The last lines of the program:\n%s" % (tag, note, "\n".join(last[-20:])))
        log("%s wrote: %s" % (spec["label"], ", ".join("%s (%d bytes)" % (os.path.relpath(os.path.join(d, n), r["out_dir"]), os.path.getsize(os.path.join(d, n)))
                                                         for d, _s, ns in os.walk(r["out_dir"]) for n in sorted(ns))[:700]))
        for kind, path in produced.items():
            try:
                row = score_model(key, entry, ctx["row"], ctx["native"], kind, path)
                ROWS.append(row)
                e["models"].append(row)
            except Exception as ex:
                e.setdefault("score_errors", []).append("%s: %s" % (kind, str(ex)[:200]))
                log("scoring of %s stopped: %s (the model is in the zip; it is scored again later)" % (kind, ex))
        pack_now()
    e["seconds"] = round(time.time() - t_task)
    if verdict == "ok":
        e["status"] = "ok"
        spent = sum(v["seconds"] for v in e["runs"].values())
        DURATIONS.setdefault(key, []).append(spent)
        if spec.get("s_per_residue") and entry in SIZES:
            RATES.setdefault(key, []).append(spent / SIZES[entry])
    return verdict


def after_task(entry, key, result):
    """The rules that look at the outcome: two crashes within three minutes in a row stop the method for this session."""
    with SCHED:
        if result == "quick_crash":
            CRASH_STREAK[key] = CRASH_STREAK.get(key, 0) + 1
        elif result in ("ok", "failed"):
            CRASH_STREAK[key] = 0
        streak = CRASH_STREAK.get(key, 0)
    if streak >= 2 and STATE["methods"][key]["status"] == "ready":
        reason = ("two runs in a row ended in a crash within %d minutes: probably a problem of the installation or of the inputs, not a result; "
                  "the other entries of this method are left for the next session" % (QUICK_CRASH_S // 60))
        log("\n%s: %s" % (SPECS[key]["label"], reason))
        disable_method(key, reason, status="disabled", failed_step="guard")


def lane_main(gpu):
    """The worker of one GPU: takes the next task until nothing more can start."""
    while True:
        t = take_task(gpu)
        if t is None:
            log("GPU %d: nothing more to start in this session" % gpu)
            return
        if t == "wait":
            time.sleep(POLL_S)
            continue
        entry, key = t
        try:
            result = run_task(gpu, entry, key)
        except Exception:
            import traceback
            result = "quick_crash"
            m = STATE["methods"][key]
            m["entries"].setdefault(entry, {"entry": entry, "method": key, "runs": {}, "models": []}).update(
                status="not run", failed_step="notebook error", note=traceback.format_exc()[-600:])
            log("\nGPU %d: %s on %s STOPPED by an error of the notebook (not a result of the method):\n%s" % (gpu, SPECS[key]["label"], entry, traceback.format_exc()))
        finally:
            with SCHED:
                BUSY.pop(gpu, None)
        after_task(entry, key, result)


def detect_gpus():
    forced = os.environ.get("MICA_TEST_GPUS")                         # only the tests set this
    if forced:
        return list(range(int(forced)))
    ids = []
    if shutil.which("nvidia-smi"):
        for line in sh("nvidia-smi --query-gpu=index --format=csv,noheader", timeout=30).splitlines():
            if line.strip().isdigit():
                ids.append(int(line.strip()))
    return ids or [0]


def usable_lanes():
    """The GPUs that run work: every GPU that a small PyTorch computation can use (tested with the environment of the first method that is ready)."""
    gpus = detect_gpus()
    if len(gpus) < 2 or ENV_PY_GIVEN or not REQUIRE_GPU:
        return gpus
    ready = [READY[k][0] for k in RUN_METHODS if k in READY]
    if not ready:
        return gpus[:1]
    good = []
    for g in gpus:
        rc, last, _ = run([ready[0], "-c", GPU_TEST], env=env_vars(ready[0], {"CUDA_VISIBLE_DEVICES": str(g)}), quiet=True)
        log("GPU %d: %s" % (g, "a PyTorch computation works" if rc == 0 else "does NOT work (exit code %s): %s; it is not used" % (rc, " ".join(last[-2:])[:200])))
        if rc == 0:
            good.append(g)
    return good or gpus[:1]


def check_af3_files():
    """Stops the session at once, before an hour is spent, if the private AlphaFold dataset is not attached or is not the checked one (the hashes of the list of the repository)."""
    entries = sorted({t[0] for t in PENDING if SPECS[t[1]]["uses_af3"]})
    if not entries:
        return
    manifest = {}
    with open("data/af3/af3_files_sha256.txt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                h, name = line.split()
                manifest[name] = h
    models = ""
    for entry in entries:
        jobs = mica_inputs.read_jobs("data/af3/af3_jobs.csv", entry)
        models = find_af3_dir(jobs)
        for r in jobs:
            name = "fold_%s_model_0.cif" % r["job_name"].lower()
            got = sha256_file(mica_inputs.find_model(models, r["job_name"]))
            if got != manifest.get(name):
                raise RuntimeError("the file %s is not the one that was checked (sha256 %s, expected %s)" % (name, got[:16], str(manifest.get(name))[:16]))
    log("AlphaFold Server files of %d entries: found in %s, every hash matches the checked list" % (len(entries), models))


def setup_phase():
    """Source, environment, weights of every method, one after the other; a method whose setup fails is left out (its tasks are 'not run'), the others go on."""
    for key in RUN_METHODS:
        spec = SPECS[key]
        m = STATE["methods"][key] = {"key": key, "label": spec["label"], "commit": spec["commit"], "status": "setup", "failed_step": "", "note": "", "setup_seconds": None, "entries": {}}
        log("\n" + "#" * 78 + "\n# setup of %s (commit %s)\n" % (spec["label"], spec["commit"][:10]) + "#" * 78)
        t0 = time.time()
        try:
            READY[key] = setup_method(spec)
            m.update(status="ready", setup_seconds=round(time.time() - t0))
            log("%s is ready (%d s)" % (spec["label"], m["setup_seconds"]))
        except Exception as ex:
            import traceback
            m["setup_seconds"] = round(time.time() - t0)
            log("\n%s STOPPED in its setup: %s\n%s" % (spec["label"], ex, traceback.format_exc()[-1200:]))
            disable_method(key, "the setup failed: %s" % str(ex)[:400], status="setup failed", failed_step="setup")
        pack_now()


log("tools ready; methods: %s; %d tasks; budget %.1f hours (the programs are stopped %.0f minutes before the end)" % (", ".join(RUN_METHODS), len(PENDING), SESSION_S / 3600, RESERVE_S / 60))
'''

MODELANGELO_MD = """\
## Step 5. ModelAngelo

`model_angelo build -v <map> -pf <fasta> -o <folder> --device 0`, the command of the README with the GPU named (`CUDA_VISIBLE_DEVICES` selects the GPU of the worker). The weights are
the default bundle of `model_angelo build` (`model_angelo setup_weights --bundle-name nucleotides`: Zenodo, MD5 checked by ModelAngelo itself; the ESM-1b language model comes from Meta).
The model is `<folder name>.cif`; `<folder name>_raw.cif` is the model before pruning. Both are scored."""
MODELANGELO = r'''
def setup_modelangelo(spec, repo, env_py):
    missing = [p for p in ("model_angelo/__main__.py", "model_angelo/model_definitions/" + spec["bundle"]) if not os.path.exists(os.path.join(repo, p))]
    if missing:
        raise RuntimeError("the ModelAngelo source at commit %s lacks %s" % (spec["commit"], ", ".join(missing)))
    run_env = env_vars(env_py)
    rc, last, _ = run([env_py, "-m", "model_angelo", "--version"], env=run_env, cwd=repo, quiet=True)
    STATE["methods"]["modelangelo"]["version"] = (last or [""])[-1]
    log("version printed by the program: %s" % STATE["methods"]["modelangelo"]["version"])
    code = "import model_angelo.apps.build, model_angelo.utils.setup_weights; print('ModelAngelo modules import')"
    rc, last, _ = run([env_py, "-c", code], env=run_env, cwd=repo)
    if rc != 0:
        raise RuntimeError("ModelAngelo's own Python modules cannot be imported (see the lines above)")
    bundle = os.path.join(TORCH_HOME, "hub", "checkpoints", "model_angelo_v1.0", spec["bundle"])
    cfg = os.path.join(bundle, "config.json")
    if not os.path.exists(cfg):
        logf = os.path.join(WORK, "logs", "modelangelo_setup_weights.log")
        log("downloading the weights: model_angelo setup_weights --bundle-name %s" % spec["bundle"])
        rc, last, timed_out = run_watched([env_py, "-m", "model_angelo", "setup_weights", "--bundle-name", spec["bundle"]], "ModelAngelo weights", watch=[logf], env=run_env, cwd=repo,
                                          log_file=logf, timeout=3600, every=1)
        text = open(logf, errors="replace").read() if os.path.exists(logf) else ""
        if rc != 0 or timed_out or "Successful!" not in text:
            log("\n".join(last[-15:]))
            raise RuntimeError("the ModelAngelo weights could not be downloaded or did not pass ModelAngelo's own MD5 check (%s)" % ("timeout" if timed_out else "exit code %s" % rc))
    if not os.path.exists(cfg):
        raise RuntimeError("the weights bundle %s was not installed where expected (%s)" % (spec["bundle"], bundle))
    shutil.copy(cfg, os.path.join(WORK, "modelangelo_config.json"))
    record_files("modelangelo", [os.path.join(TORCH_HOME, "hub", "checkpoints")])


def plan_modelangelo(spec, ctx, repo, env_py):
    entry = ctx["entry"]
    name = "%s_modelangelo" % entry                                           # ModelAngelo names its models after the output folder
    out = os.path.join(RUN_ROOT, "modelangelo", entry, name)
    cmd = [env_py, "-m", "model_angelo", "build", "-v", ctx["map"], "-pf", ctx["fasta"], "-o", out, "--device", "0"]
    outputs = {"modelangelo": os.path.join(out, name + ".cif"), "modelangelo_raw": os.path.join(out, name + "_raw.cif")}
    return [{"variant": "build", "cmd": cmd, "cwd": repo, "env": env_vars(env_py), "out_dir": out, "outputs": outputs, "main": "modelangelo",
             "watch": [os.path.join(out, "model_angelo.log")]}]


SETUPS["modelangelo"], PLANS["modelangelo"] = setup_modelangelo, plan_modelangelo
'''

E3_MD = """\
## Step 6. E3-CryoFold

`inference.py --map_path <map> --fasta_path <fasta> --save_dir <folder>/ --save_name <name> --protocol pre_align --t 0.1`, the first example of the README (map and sequence; `pre_align`
is the default protocol of the program). The `denovo` protocol of the pilot is not part of the benchmark. The weights are the `models.zip` of the repository's `checkpoint` release (1.5 GB)."""
EMPROT_MD = MM.EMPROT_MD.replace("## Step 6. EMProt", "## Step 7. EMProt")
EMODELX_MD = MM.EMODELX_MD

RUN_MD = """\
## Step 9. The runs

First the setup of the four methods (one after the other), then one worker per GPU takes tasks from the list until the time budget says stop. A progress line of every running
program appears every two minutes; the program's own output is in `logs/` of the zip. Everything is packed into a zip after every run."""
RUN = r'''
def run_schedule():
    global LANES
    setup_phase()
    LANES = usable_lanes()[:MAX_LANES]
    log("\nGPUs used: %s; %s" % (LANES, queue_line()))
    workers = [threading.Thread(target=lane_main, args=(g,), name="gpu%d" % g) for g in LANES]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    for entry, key in list(RETRY_ALONE):                      # killed while the other GPU ran: once more, alone
        if STATE["methods"][key]["status"] == "ready" and HARD_STOP - time.time() >= need_s(key, entry):
            log("\nrun again alone: %s on %s" % (SPECS[key]["label"], entry))
            with SCHED:
                BUSY[LANES[0]] = {"entry": entry, "key": key, "start": time.time(), "program_start": None}
            try:
                run_task(LANES[0], entry, key, attempt=2)
            finally:
                with SCHED:
                    BUSY.pop(LANES[0], None)
        else:
            log("\n%s on %s cannot be run again alone now (no time left or the method was stopped): it stays 'killed_corun' and is run again in the next session" % (SPECS[key]["label"], entry))
    for entry, key in list(PENDING):
        PENDING.remove((entry, key))
        mark(entry, key, "not run", "not started: not enough time left in the session")


if STATE["ok"] and STATE["usalign"]:
    try:
        check_af3_files()
    except Exception as ex:
        fail("the private AlphaFold dataset is missing or is not the checked one (%s). Kaggle: right-hand panel, Add Input, choose your AlphaFold dataset, then run again. "
             "No GPU time was spent." % str(ex)[:300])
    if STATE["ok"]:
        try:
            run_schedule()
        except BaseException as ex:
            import traceback
            fail("the schedule stopped with an error of the notebook: %s" % str(ex)[:300])
            log(traceback.format_exc())
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

SUMMARY_MD = """\
## Step 10. Summary

`metrics_usalign.csv` (every model that was scored, first pass: the final scores are computed later from the models in the zip), `method_status.csv` (one line per task),
`pending.csv` (the tasks for the next session), and a table of the whole run."""
SUMMARY = r'''
def status_rows():
    rows = []
    for key in RUN_METHODS:
        m = STATE["methods"].get(key)
        if not m:
            continue
        for entry, e in m["entries"].items():
            runs = list(e.get("runs", {}).values())
            rows.append(dict(method=key, entry=entry, status=e.get("status", ""), failed_step=e.get("failed_step", ""), note=e.get("note", ""), attempt=e.get("attempt", ""),
                             gpu=e.get("gpu", ""), started_utc=e.get("started_utc", ""), seconds=sum(r["seconds"] for r in runs) if runs else "",
                             peak_gpu_mb=max([r["peak_gpu_mb"] for r in runs] or [""]) if runs else "", min_ram_gb=min([r["min_ram_available_gb"] for r in runs if r["min_ram_available_gb"] is not None] or [""]) if runs else "",
                             other_gpu_busy=any(r["other_gpu_busy"] for r in runs) if runs else "", limit_s=runs[0]["limit_s"] if runs else ""))
    return rows


def summary_table(rows):
    keys = [k for k in RUN_METHODS if k in STATE["methods"]]
    entries = [t[0] for t in TASKS]
    entries = list(dict.fromkeys(entries))
    by = {(r["entry"], r["method"]): r for r in rows}
    best = {}
    for r in ROWS:
        if r["model_kind"] == SPECS[r["method"]]["main_kind"]:
            best[(r["entry"], r["method"])] = r["tm_score"]
    log("\n%-6s " % "entry" + " ".join("%-24s" % SPECS[k]["label"][:24] for k in keys))
    for entry in entries:
        cells = []
        for k in keys:
            r = by.get((entry, k))
            if not r:
                cells.append("%-24s" % "-")
            else:
                mins = ("%dmin" % round(r["seconds"] / 60)) if r["seconds"] != "" else ""
                cells.append("%-24s" % ("%s %s %s" % (r["status"], mins, ("TM " + str(best[(entry, k)])) if (entry, k) in best else "")).strip()[:24])
        log("%-6s " % entry + " ".join(cells))
    log("")
    for k in keys:
        m = STATE["methods"][k]
        log("%-14s %-13s setup %s s%s" % (SPECS[k]["label"], m["status"].upper(), m["setup_seconds"], ("; " + m["note"][:160]) if m["status"] != "ready" else ""))


for _k, _s in SPECS.items():
    _s["main_kind"] = {"modelangelo": "modelangelo", "e3": "e3_prealign", "emprot": "emprot", "emodelx": "emodelx_af_all_atom", "cryoatom1": "cryoatom1"}[_k]
if STATE["ok"]:
    rows = status_rows()
    write_csv("metrics_usalign.csv", METRIC_COLUMNS, ROWS)
    write_csv("method_status.csv", ["method", "entry", "status", "failed_step", "note", "attempt", "gpu", "started_utc", "seconds", "peak_gpu_mb", "min_ram_gb", "other_gpu_busy", "limit_s"], rows)
    left = [r for r in rows if r["status"] not in ("ok", "failed")]
    write_csv("pending.csv", ["entry", "method", "status", "note"], [dict(entry=r["entry"], method=r["method"], status=r["status"], note=r["note"]) for r in left])
    summary_table(rows)
    log("\n%d tasks are finished (ok or failed), %d are left for the next session (pending.csv)" % (len(rows) - len(left), len(left)))
    STATE["complete"] = any(r["status"] in ("ok", "failed") for r in rows)
    if not STATE["complete"]:
        STATE["stopped_at"] = "no run finished (see the setup lines of each method)"
    STATE["pending"] = [[r["entry"], r["method"], r["status"]] for r in left]
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

# ---------------------------------------------------------------------------------------------------------- the notebook of CryoAtom v1.0.0 (alone)
INTRO_CRYOATOM = """\
# MICA benchmark: GPU session `@BATCH@` (CryoAtom v1.0.0)

This notebook builds atomic models from the cryo-EM maps of the benchmark with **CryoAtom v1.0.0** (the code of the Nature Structural & Molecular Biology paper, commit `e2a31f2b`), one entry
after the other, and scores every model with US-align, the same scoring as for the other methods. CryoAtom gets the map and the sequence only, like ModelAngelo and E3-CryoFold, so
**no AlphaFold dataset is needed**. The method was added to the benchmark after Prof. Cheng's request of 31 October 2026; the other methods are not run here.

@LANES_TEXT@
* The list has @N_TASKS@ entries, in the order of the seeded selection (the two resolution classes interleaved). The time of a run grows with the size of the entry (the pilot on 26DK needed about
  2.7 hours on a T4). Before an entry starts, the notebook checks that its estimated time (1.25 x the residues of the entry x the largest seconds per residue seen in this session, or the
  rate of the pilot until the first run has finished) fits in what is left of the budget (`HOURS` in the first cell). When the budget is nearly used up, no new run starts; what is not done is
  written to `pending.csv` and is run in the next session.
* A run that crashes is a result and stays in the tables (with the reason); a run that the end of the session cuts is **not** a result: it is run again.
* Every run has a time limit of @LIMIT_TEXT@ hours (the four other methods of the benchmark have 3 hours; the pilot of CryoAtom did not finish in 2).

**What you do (Kaggle, background mode)**: accelerator **GPU T4 x2** (not P100), Internet **On**, no input dataset and no secret, `HOURS` in the first cell (the hours of GPU time you allow), then
*Save Version -> Save & Run All (Commit)*. It runs on Kaggle's servers; you can close the tab and your Wi-Fi does not matter. When the version shows *Succeeded*, open it and download the zip
`@NOTEBOOK@_result_....zip` from its Output. Send it to me. **Do not stop or cancel the run**: a cancelled run keeps nothing.
"""
SETTINGS_CRYOATOM = SETTINGS.replace("If you have less than 9 hours left this week,", "If you have less than 12 hours left this week,")
assert SETTINGS_CRYOATOM != SETTINGS

CRYOATOM_MD = """\
## Step 5. CryoAtom v1.0.0

`python -m CryoAtom build -s <fasta> -v <map> -o <folder>` from the source folder at the pinned commit: the README's example of v1.0.0, the code of the paper (the same program as the `cryoatom` command
that `setup.py` installs). The weights are the zip of the authors' server (certificate check off, as in their own `wget --no-check-certificate`). The ESM-2 language model (about 2.5 GB, from Meta) is
fetched once here: the program would otherwise fetch it at its first run, on both GPUs at the same time. **One line of the source is changed** (file `CryoAtom/utils/aa_probs_to_hmm.py`): the
program writes the profile of every chain to the single file `/tmp/CryoAtom_temp.hmm`, so two programs at the same time would overwrite and delete each other's file; the process number is added
to the name. Nothing else is changed; `cryoatom_source_patch.diff` in the zip shows the change. The model is `<folder name>.cif`; `<folder name>_raw.cif` is the model before pruning. Both are
scored; the first one is the model of the benchmark."""
CRYOATOM_REGISTER_OLD = 'for _k in ("cryoatom2", "cryoatom1"):\n    SETUPS[_k], PLANS[_k] = setup_cryoatom, plan_cryoatom\n'
CRYOATOM_SESSION = r'''

def patch_cryoatom_tempfile(spec, repo):
    """CryoAtom v1.0.0 writes the profile of every chain to ONE file, /tmp/CryoAtom_temp.hmm, reads it and deletes it. Two programs at the same time (one per GPU) would read and delete each other's
    file: a missing file stops a run, and the profile of the other program would align the wrong sequence without any message. The only change to the source of the method: the process number
    is part of the file name. The line must exist exactly once (otherwise the session stops: the source is not the pinned commit); `git diff` of the change goes into the zip."""
    path = os.path.join(repo, "CryoAtom", "utils", "aa_probs_to_hmm.py")
    old = 'tmp_path = os.path.join(base_dir, f"CryoAtom_temp.hmm")'
    new = 'tmp_path = os.path.join(base_dir, f"CryoAtom_temp_{os.getpid()}.hmm")'
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.count(old) == 1:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new))
        log("changed one line of %s: the temporary profile file is named after the process, so that two programs at the same time do not share it" % os.path.relpath(path, repo))
    elif text.count(new) != 1:
        raise RuntimeError("the line of CryoAtom that names its temporary profile file was not found in %s: the source is not the pinned commit, and two programs at the same time could share the file" % path)
    with open(os.path.join(WORK, "cryoatom_source_patch.diff"), "w", encoding="utf-8") as f:
        f.write(sh("git -C %s diff" % repo, timeout=60) + "\n")
    STATE["methods"][spec["key"]]["source_patch"] = {"file": os.path.relpath(path, repo), "line_before": old, "line_after": new, "file_sha256": sha256_file(path)}


def setup_cryoatom_session(spec, repo, env_py):
    """The setup of the methods pilot (source, weights), the one-line change of the temporary file, then the ESM-2 weights: the program fetches them by torch.hub at its first run, and two programs
    on two GPUs would both do it."""
    setup_cryoatom(spec, repo, env_py)
    patch_cryoatom_tempfile(spec, repo)
    esm = spec.get("esm")
    if not esm:
        return
    ckpt = os.path.join(TORCH_HOME, "hub", "checkpoints")
    path = os.path.join(ckpt, esm["file"])
    if not (os.path.exists(path) and os.path.getsize(path) >= esm["min_bytes"]):
        logf = os.path.join(WORK, "logs", "cryoatom_esm2_weights.log")
        code = "import esm; esm.pretrained.%s(); print('the ESM-2 weights are in place')" % esm["call"]
        for attempt in (1, 2):
            log("downloading the ESM-2 weights: esm.pretrained.%s() (attempt %d)" % (esm["call"], attempt))
            rc, last, timed_out = run_watched([env_py, "-c", code], "ESM-2 weights", watch=[logf], env=env_vars(env_py), cwd=repo, log_file=logf, timeout=1800, every=1)
            if rc == 0 and not timed_out and os.path.exists(path) and os.path.getsize(path) >= esm["min_bytes"]:
                break
        else:
            log("\n".join(last[-12:]))
            raise RuntimeError("the ESM-2 weights could not be downloaded (%s)" % ("timeout" if timed_out else "exit code %s" % rc))
    record_files(spec["key"], [ckpt], "esm_weights")


SETUPS["cryoatom1"], PLANS["cryoatom1"] = setup_cryoatom_session, plan_cryoatom
'''
RUN_MD_CRYOATOM = """\
## Step 6. The runs

First the setup of CryoAtom (source, environment, weights, ESM-2), then one worker per GPU takes entries from the list until the time budget says stop. A progress line of every running
program appears every two minutes (the last line of CryoAtom's own `running_time.log`); the program's output is in `logs/` of the zip. Everything is packed into a zip after every run."""


EXTRA_INFO = ('{"methods": {k: {"label": m["label"], "commit": m["commit"], "status": m["status"], "setup_seconds": m["setup_seconds"], "version": m.get("version", "")} for k, m in STATE["methods"].items()}, '
              '"specs": globals().get("SPECS", {}), "usalign_commit": globals().get("USALIGN_COMMIT", ""), "usalign_version": STATE["usalign_version"], "input_sha256": globals().get("EXPECTED", {}), '
              '"tasks": globals().get("TASKS"), "hours_setting": globals().get("HOURS_SETTING"), "run_limit_s": globals().get("RUN_LIMIT"), "lanes": globals().get("LANES")}')


LANES_TEXT = {1: "* The notebook runs **one program at a time** (on the first GPU of the session). In the first session some programs were killed for lack of RAM while the program of the other GPU ran; "
                 "the entries of this session run alone, so a kill now is a result and not a neighbour's fault.",
              2: "* The Kaggle session has **two GPUs**. The notebook uses both: two programs run at the same time, one on each GPU, so the work of about 2 x @HOURS_TEXT@ hours fits in the session."}


# ---------------------------------------------------------------------------------------------------------- the notebook
def build(out_path, batch="batch1", entries=None, tasks=None, hours=9.0, guard_minutes=60, max_lanes=2, usalign_repo=P.USALIGN_REPO, usalign_commit=P.USALIGN_COMMIT, entries_path=None, env_python=None,
          usalign_bin=None, net_hosts=NET_HOSTS, require_gpu=True, import_tests=None, specs=None, run_methods=None):
    """Writes the notebook. `tasks` ([entry, method] pairs) overrides `entries`; the keyword arguments after `hours` exist for the tests."""
    run_methods = list(run_methods or ORDER)
    cryo = "cryoatom1" in run_methods                                     # CryoAtom v1.0.0 has a notebook of its own
    if cryo and run_methods != ["cryoatom1"]:
        raise ValueError("CryoAtom v1.0.0 is built into its own notebook: run_methods must be ['cryoatom1'] alone")
    specs = copy.deepcopy(specs or (SPECS_ALL if cryo else SPECS))
    if cryo:
        specs = {"cryoatom1": specs["cryoatom1"]}
        if net_hosts is NET_HOSTS:
            net_hosts = NET_HOSTS_CRYOATOM
    rows = read_entries(entries_path)
    entries = list(entries or seeded_order(rows))
    tasks = [list(t) for t in (tasks if tasks is not None else all_tasks(entries, ["cryoatom1"] if cryo else ORDER))]
    sizes = {r["pdb_id"]: int(r["modeled_residues"]) for r in rows}
    known = {r["pdb_id"] for r in rows}
    for e, m in tasks:
        if e not in known or m not in specs:
            raise ValueError("task %s %s: unknown entry or method" % (e, m))
    wanted = {m for _e, m in tasks}
    run_methods = [m for m in run_methods if m in wanted]                 # only the methods that have a task are set up (a session that finishes one method's leftovers must not install the others)
    notebook = "gpu_" + batch
    files = FILES if not cryo else BASE_FILES + ((specs["cryoatom1"]["req"], specs["cryoatom1"]["req"]),)
    file_cells, expected = nbkit.file_cells(files, entries_path)
    step5 = nbkit.fill(P.STEP5, {"@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit, "@MICA_REPO@": "x", "@MICA_COMMIT@": "x", "@WEIGHTS_URL@": "x", "@ZENODO_API@": "x", "@USALIGN_BIN@": "None"})
    git_and_usalign = nbkit.function_sources(step5, ["git_checkout", "build_usalign"]).rstrip("\n")
    step6 = nbkit.fill(P.STEP6, {"@ENTRIES@": "[]"})
    read_write_download = nbkit.function_sources(step6, ["read_csv", "write_csv", "download"]).rstrip("\n")
    find_af3 = nbkit.function_sources(step6, ["find_af3_dir"]).rstrip("\n")
    mapping = {"@UV_VERSION@": P.UV_VERSION, "@ENV_PYTHON@": repr(env_python), "@IMPORT_TESTS@": repr(import_tests), "@SPECS@": repr(specs), "@RUN_METHODS@": repr(run_methods),
               "@ENTRIES@": repr(entries), "@ENTRY_LIST@": ", ".join(entries), "@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit,
               "@USALIGN_BIN@": repr(usalign_bin), "@REQUIRE_GPU@": repr(bool(require_gpu)), "@NET_HOSTS@": repr(tuple(net_hosts)), "@EXPECTED@": repr(expected),
               "@NOTEBOOK@": notebook, "@EXTRA_INFO@": EXTRA_INFO, "@BATCH@": batch, "@HOURS@": repr(float(hours)), "@GUARD_MINUTES@": repr(int(guard_minutes)), "@HOURS_TEXT@": ("%g" % hours),
               "@MAX_LANES@": str(int(max_lanes)), "@LANES_TEXT@": LANES_TEXT[1 if int(max_lanes) <= 1 else 2].replace("@HOURS_TEXT@", "%g" % hours), "@N_TASKS@": str(len(tasks)), "@LIMIT_TEXT@": ("%g" % ((specs["cryoatom1"]["timeout"] if cryo else RUN_LIMIT_S) / 3600)), "@SIZES@": repr(sizes), "@RUN_LIMIT@": repr(RUN_LIMIT_S), "@EXPLOSIVE@": repr(EXPLOSIVE), "@TASKS@": repr(tasks)}
    fl = lambda text: nbkit.fill(text, mapping).strip("\n") + "\n"
    # the cells of the methods notebook that are reused: the head of its tools cell (up to run_program), five of its functions, and the cells of the three methods without the lines that ran them
    head = nbkit.cut_before(MM.TOOLS, "def run_program").replace("@READ_WRITE_DOWNLOAD@", read_write_download).replace("@GIT_AND_USALIGN@", git_and_usalign).replace("@FIND_AF3@", find_af3)
    full_old = nbkit.fill(MM.TOOLS, {}).replace("@READ_WRITE_DOWNLOAD@", read_write_download).replace("@GIT_AND_USALIGN@", git_and_usalign).replace("@FIND_AF3@", find_af3)
    prepare = nbkit.function_sources(fl(full_old), ["prepare_inputs", "af3_inputs", "score_model", "keep_results", "setup_method"]).rstrip("\n")
    tools = head.rstrip("\n") + "\n\n\n" + TOOLS_TAIL.replace("@PREPARE_FUNCTIONS@", prepare)
    marker = 'if STATE["ok"] and STATE["usalign"]:'
    e3 = nbkit.replace_once(nbkit.cut_before(MM.E3, marker), '(("prealign", "pre_align"), ("denovo", "denovo"))', '(("prealign", "pre_align"),)')
    emprot, emodelx = nbkit.cut_before(MM.EMPROT, marker), nbkit.cut_before(MM.EMODELX, marker)
    if cryo:
        cryoatom = nbkit.replace_once(nbkit.cut_before(MM.CRYOATOM, marker), CRYOATOM_REGISTER_OLD, "").rstrip("\n") + "\n" + CRYOATOM_SESSION
        method_cells = [("cryoatom-title", nbkit.md(CRYOATOM_MD)), ("cryoatom", nbkit.code(fl(cryoatom)))]
    else:
        method_cells = [("modelangelo-title", nbkit.md(MODELANGELO_MD)), ("modelangelo", nbkit.code(fl(MODELANGELO))),
                        ("e3-title", nbkit.md(E3_MD)), ("e3", nbkit.code(fl(e3))),
                        ("emprot-title", nbkit.md(EMPROT_MD)), ("emprot", nbkit.code(fl(emprot))),
                        ("emodelx-title", nbkit.md(EMODELX_MD)), ("emodelx", nbkit.code(fl(emodelx)))]
    cells = [("intro", nbkit.md(nbkit.fill(INTRO_CRYOATOM if cryo else INTRO, mapping))),
             ("settings-title", nbkit.md(SETTINGS_MD)), ("settings", nbkit.code(fl(SETTINGS_CRYOATOM if cryo else SETTINGS))),
             ("step1-title", nbkit.md(P.STEP1_MD)), ("step1", nbkit.code(fl(P.STEP1))),
             ("step2-title", nbkit.md(P.STEP2_MD)), ("step2-dirs", nbkit.code(P.STEP2_DIRS.strip("\n") + "\n"))]
    cells += file_cells
    cells += [("step2-check", nbkit.code(fl(P.STEP2_CHECK))),
              ("robust-title", nbkit.md(nbkit.fill(nbkit.ROBUST_MD, mapping))), ("robust", nbkit.code(fl(nbkit.ROBUST))),
              ("tools-title", nbkit.md(TOOLS_MD)), ("tools", nbkit.code(fl(tools))),
              ("step4-title", nbkit.md(MM.STEP4_MD)), ("step4", nbkit.code(fl(MM.STEP4))),
              ] + method_cells + [
              ("run-title", nbkit.md(RUN_MD_CRYOATOM if cryo else RUN_MD)), ("run", nbkit.code(fl(RUN))),
              ("summary-title", nbkit.md(SUMMARY_MD.replace("Step 10", "Step 7") if cryo else SUMMARY_MD)), ("summary", nbkit.code(fl(SUMMARY))),
              ("final-title", nbkit.md(nbkit.FINAL_MD)), ("final", nbkit.code(fl(nbkit.FINAL)))]
    nbkit.assemble(out_path, "gpu_%s.ipynb" % batch, cells)
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", default="batch1", help="the name of the batch (the notebook is scripts/gpu_<batch>.ipynb and its zip gpu_<batch>_result_....zip)")
    p.add_argument("--out", default=None)
    p.add_argument("--entries", nargs="+", default=None, help="the entries (default: all, in the seeded order)")
    p.add_argument("--methods", nargs="+", default=ORDER, choices=ORDER_ALL, help="the four methods of the first batches, or cryoatom1 alone (its own notebook)")
    p.add_argument("--done", nargs="*", default=[], help="method_status.csv files of earlier sessions: their finished tasks (ok or failed) are not run again")
    p.add_argument("--redo", nargs="*", default=[], help="ENTRY:method pairs to run again although an earlier session finished them")
    p.add_argument("--hours", type=float, default=None, help="the budget of the session in hours (default 9; 11 for CryoAtom)")
    p.add_argument("--lanes", type=int, default=2, help="GPUs used at the same time (1 for entries that were killed for lack of RAM while another program ran)")
    p.add_argument("--guard-minutes", type=int, default=60, help="a crash sooner than this on two entries in a row pauses the method for the session (0: no guard)")
    args = p.parse_args(argv)
    rows = read_entries()
    entries = args.entries or seeded_order(rows)
    tasks = pending_tasks(all_tasks(entries, args.methods), args.done, args.redo)
    out = args.out or os.path.join(HERE, "gpu_%s.ipynb" % args.batch)
    hours = args.hours if args.hours is not None else (11.0 if "cryoatom1" in args.methods else 9.0)
    expected = build(out, batch=args.batch, entries=entries, tasks=tasks, hours=hours, guard_minutes=args.guard_minutes, run_methods=args.methods, max_lanes=args.lanes)
    print("wrote %s: %d tasks (%d entries), %d embedded files" % (out, len(tasks), len({t[0] for t in tasks}), len(expected)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
