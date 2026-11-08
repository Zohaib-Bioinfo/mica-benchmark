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
    python make_main_notebook.py --batch cryozeta_a --methods cryozeta --hours 11              # CryoZeta alone (see below)

Run it on Kaggle with the GPU "T4 x2" and the private AlphaFold dataset attached, as "Save & Run All (Commit)".

CryoAtom v1.0.0 (added on 2026-10-31 at Prof. Cheng's request) has a notebook of its own, with the same scheduler, scoring and zips: it needs no AlphaFold dataset, a run may take 10 hours
instead of 3 (its pilot did not finish in 2), the time of a run grows with the size of the entry (the budget check uses the residues of the entry and the seconds per residue of the runs that
finished in the session) and the ESM-2 weights are fetched once, before the two GPUs start. The four other methods are not changed by any of this.

CryoZeta (Kihara lab; added on 2026-10-31 at Prof. Cheng's request) has a notebook of its own as well (`--methods cryozeta`): its project is a pixi project (the environment is installed from the
lock file of the repository), TEASER++ is built at a pinned commit, the weights come from Hugging Face and the alignments of the sequences from the public ColabFold server. The runs go through
`scripts/cryozeta_tools.py`, one program of this repository that runs the four programs of the authors' demo for one entry and reports what happened. The authors ask for a GPU with 32 GB;
on a T4 (15 GB) an entry may run out of memory, and that is a result of the method on this hardware.
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
CRYOZETA_LIMIT_S = 6 * 3600                     # per run of CryoZeta (LOG.md, 2026-11-02): no run on a T4 has been timed; twice the limit of the first four methods
CRYOZETA_S_PER_RESIDUE = 15                     # a first guess for the budget check (no run on a T4 exists); the measured rate of the session replaces it as soon as one run has finished
ORDER = ["modelangelo", "e3", "emprot", "emodelx"]
SOLO = ("cryoatom1", "cryozeta")                # the methods that have a notebook of their own
ORDER_ALL = ORDER + list(SOLO)
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


def cryozeta_spec():
    """CryoZeta (Kihara lab) at the pinned commit: the pixi project of the repository (a pinned release of pixi, the lock file of the repository), TEASER++ at a pinned commit (built with pinned
    cmake and pybind11 from PyPI), the three weights files and the two chemical-components files of the Hugging Face repository (size and sha256 are checked against what the Hub lists) and the public
    ColabFold server for the alignments.
    The settings of a run are those of the authors' inference_demo.sh (5 samples, 20 diffusion steps, 10 cycles, seed 101, the 5 samples are kept)."""
    return {
        "key": "cryozeta", "label": "CryoZeta", "repo": "https://github.com/kiharalab/CryoZeta", "commit": "c544c6bbd0366951d8b4e0bc4c0da547836f7b8f", "python": "3.12", "req": None,
        "timeout": CRYOZETA_LIMIT_S, "s_per_residue": CRYOZETA_S_PER_RESIDUE, "oom_is_result": True, "killed_rc": 4, "uses_af3": False, "bundle": None, "weights_url": None, "weights_min_bytes": 0, "gdrive_id": None,
        "import_test": ("import importlib.util, torch, safetensors; assert importlib.util.find_spec('cryozeta') is not None; "
                        "print('the CryoZeta environment imports: torch ' + torch.__version__)"),
        "pixi": {"version": "0.41.1", "url": "https://github.com/prefix-dev/pixi/releases/download/v0.41.1/pixi-x86_64-unknown-linux-musl", "min_bytes": 30000000,
                 "sha256": "f4e98919053c099c6839602c3334e1b7802d4db445a2ffa9282ff6abac6c0418", "env": "cu11"},
        "teaser": {"repo": "https://github.com/MIT-SPARK/TEASER-plusplus", "commit": "52a9c52ee7d4c838c5e8a75458c33178be5bfb70", "cmake": "3.31.6", "pybind11": "3.0.1", "prebuilt": None},
        "hub": {"repo": "KiharaLab/CryoZeta", "api": "https://huggingface.co/api/models/KiharaLab/CryoZeta?blobs=true", "resolve": "https://huggingface.co/KiharaLab/CryoZeta/resolve/{rev}/{path}",
                "min_bytes": 1000000, "files": {"detection": "cryozeta-detection-v0.0.1.safetensors", "cryozeta": "cryozeta-v0.0.1.safetensors", "interpolate": "cryozeta-interpolate-v0.0.1.safetensors"},
                "data_files": {"ccd": "components.v20240608.cif", "ccd_rdkit": "components.v20240608.cif.rdkit_mol.pkl"}},      # read by cryozeta-inference when it builds the features of a sequence
        "colabfold": "https://api.colabfold.com", "msa_poll_s": 8.0, "msa_tries": 3,
        "run": {"seed": 101, "n_sample": 5, "n_step": 20, "n_cycle": 10, "num_select": 5, "dtype": "bf16", "kernels": "off"},
    }


SPECS_ALL = dict(SPECS, cryoatom1=cryoatom_spec(), cryozeta=cryozeta_spec())
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
NET_HOSTS_CRYOZETA = (("PyPI", "https://pypi.org/simple/uv/", True),
                      ("conda-forge", "https://conda.anaconda.org/conda-forge/noarch/current_repodata.json", True),
                      ("GitHub", "https://github.com/kiharalab/CryoZeta", True),
                      ("Hugging Face (CryoZeta weights)", "https://huggingface.co/api/models/KiharaLab/CryoZeta", True),
                      ("ColabFold server", "https://api.colabfold.com/", False),
                      ("EBI FTP", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-80558/header/emd-80558.xml", True),
                      ("RCSB", "https://files.rcsb.org/header/26DK.cif", True))
SOLO_NET_HOSTS = {"cryoatom1": NET_HOSTS_CRYOATOM, "cryozeta": NET_HOSTS_CRYOZETA}


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


def measured_rates(status_csvs, key, sizes):
    """What earlier sessions measured: for the finished runs (status `ok`) of `key` the number, the lowest and the highest seconds per residue of the deposited model, and the mean (all
    seconds over all residues); None if there is no such run."""
    rates, spent, residues = [], 0.0, 0
    for path in status_csvs:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["method"] == key and r["status"] == "ok" and r.get("seconds") and r["entry"] in sizes:
                    s, n = float(r["seconds"]), sizes[r["entry"]]
                    rates.append(s / n)
                    spent += s
                    residues += n
    return {"n": len(rates), "lo": min(rates), "hi": max(rates), "mean": spent / residues} if rates else None


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
INPUT_TRIES = 4                                                   # a download that fails (a network error) is asked for again after INPUT_WAIT_S, then after twice and three times that (9ZLH, 2026-11-03)
INPUT_WAIT_S = 12 * RETRY_WAIT                                    # 120 seconds; the tests make RETRY_WAIT short
T0 = STATE["t0"]
HARD_STOP = T0 + SESSION_S - RESERVE_S                            # a program that still runs at this time is stopped ("interrupted", not a failure)
EXPLOSIVE = @EXPLOSIVE@                                           # the pilot: AutoEM allocates np.zeros([6, N, N]) and the process is killed (out of memory); never two of them at once
TASKS = @TASKS@                                                   # [entry, method]: the order in which the tasks are taken
ALONE_FIRST = @ALONE_FIRST@                                       # [entry, method]: tasks that run first and alone (they were killed for lack of RAM while the program of the other GPU ran)
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
    for k_try in range(1, INPUT_TRIES + 1):
        try:
            with INPUT_LOCK:
                ctx = prepare_inputs(entry)
                runs = PLANS[key](spec, ctx, repo, env_py)
            break
        except Exception as ex:
            wait = INPUT_WAIT_S * k_try
            if "could not download" in str(ex) and k_try < INPUT_TRIES and HARD_STOP - time.time() > wait + 60:       # a network error: wait and ask again
                log("%s: a download failed (%s); the next try in %d s" % (tag, str(ex)[:200], wait))
                time.sleep(wait)
                continue
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
        if spec.get("killed_rc") is not None and rc == spec["killed_rc"]:         # the runner of CryoZeta ends with this code when the system killed one of its programs (exit code -9): the rules below look at the kill
            rc = -9
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
    for entry, key in list(ALONE_FIRST):                      # killed for lack of RAM next to another program in an earlier session: first, alone, so that a kill now is a result of the method
        if (entry, key) in PENDING and STATE["methods"][key]["status"] == "ready":
            PENDING.remove((entry, key))
            log("\nfirst, alone: %s on %s" % (SPECS[key]["label"], entry))
            with SCHED:
                BUSY[LANES[0]] = {"entry": entry, "key": key, "start": time.time(), "program_start": None}
            try:
                result = run_task(LANES[0], entry, key, attempt=2)
            finally:
                with SCHED:
                    BUSY.pop(LANES[0], None)
            after_task(entry, key, result)
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
    _s["main_kind"] = {"modelangelo": "modelangelo", "e3": "e3_prealign", "emprot": "emprot", "emodelx": "emodelx_af_all_atom", "cryoatom1": "cryoatom1", "cryozeta": "cryozeta"}[_k]
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


# ---------------------------------------------------------------------------------------------------------- the notebook of CryoZeta (alone)
INTRO_CRYOZETA = """\
# MICA benchmark: GPU session `@BATCH@` (CryoZeta)

This notebook builds atomic models from the cryo-EM maps of the benchmark with **CryoZeta** (Kihara lab; code at commit `c544c6bb`, weights `v0.0.1` from Hugging Face), one entry
after the other, and scores every model with US-align, the same scoring as for the other methods. CryoZeta gets the map, the sequences and the multiple sequence alignments of the sequences
(searched on the public ColabFold server), **no AlphaFold dataset is needed**. The method was added to the benchmark after Prof. Cheng's request of 31 October 2026; the other methods are not run here.

**Hardware.** The authors ask for a GPU with 32 GB of memory; a Kaggle T4 has 15 GB. The notebook keeps the settings of the authors' demo and lowers only the attention chunk size when a run
runs out of GPU memory (four steps, the size changes the memory and the time, not the algorithm). An entry that still does not fit is a result of the method on this hardware (`failed`, "CUDA out of
memory") and stays in the tables. A larger GPU is needed for those entries.

@LANES_TEXT@
* The list has @N_TASKS@ entries, @ORDER_TEXT@. @RATE_SENTENCE@ Before an entry starts, the notebook checks that 1.25 times its
  estimated time fits in what is left of the budget (`HOURS` in the first cell). When the budget is nearly used up, no new run starts; what is not done is written to `pending.csv`
  and is run in the next session.
* A run that crashes is a result and stays in the tables (with the reason); a run that the end of the session cuts is **not** a result: it is run again.
* Every run has a time limit of @LIMIT_TEXT@ hours.

**What you do (Kaggle, background mode)**: accelerator **GPU T4 x2** (not P100), Internet **On**, no input dataset and no secret, `HOURS` in the first cell (the hours of GPU time you allow), then
*Save Version -> Save & Run All (Commit)*. It runs on Kaggle's servers; you can close the tab and your Wi-Fi does not matter. The setup (environment, weights, TEASER++) takes 20 to 40 minutes
before the first run. When the version shows *Succeeded*, open it and download the zip `@NOTEBOOK@_result_....zip` from its Output. Send it to me. **Do not stop or cancel the run**: a cancelled run keeps nothing.
"""
SETTINGS_CRYOZETA = SETTINGS.replace("If you have less than 9 hours left this week,", "If you have less than 12 hours left this week,")
assert SETTINGS_CRYOZETA != SETTINGS

CRYOZETA_MD = """\
## Step 5. CryoZeta

The four programs of the authors' `inference_demo.sh`, in the same order and with the same settings (5 samples, 20 diffusion steps, 10 cycles, seed 101): `cryozeta-detection json-run`,
`cryozeta-inference`, `cryozeta-inference --use_interpolation true` and `cryozeta-combine --num-select 5`. The model of the benchmark is `<entry>_sample_0.cif` of `CryoZeta-Final`, the sample that the program
ranks first by its own confidence; the four other samples are kept and scored too. A small program of the repository (`scripts/cryozeta_tools.py`, standard library only) runs the four steps for
one entry because `cryozeta-inference` catches the error of a sample, writes it to a file and ends with exit code 0: it looks at the files and not at the exit code.

**What is installed.** The environment `cu11` of the project (Python 3.12, PyTorch 2.4 with CUDA 11.8, the one the project selects for a GPU of the Turing generation), exactly as the lock file of the
repository says (`pixi install -e cu11 --frozen`, pixi 0.41.1). TEASER++ is built at a pinned commit with pinned `cmake` and `pybind11` from PyPI (the project builds its master). The weights are the three
files `cryozeta*-v0.0.1.safetensors` of the Hugging Face repository `KiharaLab/CryoZeta` at the revision that the Hub reports, and so are the two chemical-components files that `cryozeta-inference` reads when it
builds the features of a sequence (`components.v20240608.cif`, 428 MB, and its RDKit cache, 126 MB; the notebook sets `CRYOZETA_ASSETS_DIR` to their folder); the sha256 of every file is compared with the one of the Hub.
The setup reads the components file with the package's own function before any run starts.
**Input.** The map, the resolution of the entries table, the recommended contour level of the EMDB header, the sequences and numbers of copies of the deposited model, and for every chain the alignment from the public
ColabFold server (`api.colabfold.com`, UniRef30 and the environmental databases; the README of CryoZeta names the ColabFold search scripts as the way to get MSAs). The server's alignments carry no species,
so no chains are paired across sequences; that is a limit of this input and it is reported.
**Settings that depend on this machine.** The compiled kernels of the demo script (DeepSpeed and cuEquivariance attention, the fast layer norm) need a CUDA compiler and an Ampere GPU: they are off (the
program has a plain implementation of each). The attention chunk size starts at the default (256) and goes down to 64, 16 and 4 only after an out-of-memory error. The number format is the default (bfloat16);
if a stage fails for another reason it is tried once in float32. If the GPU test of the setup shows that this GPU cannot run bfloat16 in the detection network, two lines of the detection code are changed to
float32 (`cryozeta_source_patch.diff` in the zip shows what was changed; it is empty if nothing was)."""
CRYOZETA = r'''
import cryozeta_tools as CZT

CZ = {}                                   # what the setup found: the folders of TEASER++ and of the weights, the number format, the result of the GPU test
CZ_PROGRAMS = ("cryozeta-detection", "cryozeta-inference", "cryozeta-combine")
CZ_PREFLIGHT = r"""
import json, torch
out = {"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    out.update(gpu=p.name, memory_gb=round(p.total_memory / 2**30, 2), capability="%d.%d" % (p.major, p.minor))

    def attempt(name, fn):
        try:
            fn()
            torch.cuda.synchronize()
            out[name] = "ok"
        except Exception as ex:
            out[name] = "failed: " + (str(ex).splitlines() or [type(ex).__name__])[0][:160]

    attempt("bf16_matmul", lambda: torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) @ torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
    x = torch.randn(1, 4, 16, 16, 16, device="cuda")
    conv = torch.nn.Conv3d(4, 8, 3, padding=1).cuda()
    conv16 = torch.nn.Conv3d(4, 8, 3, padding=1).cuda().to(torch.bfloat16)
    attempt("bf16_conv3d", lambda: conv16(x.to(torch.bfloat16)))

    def amp():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            conv(x)

    attempt("bf16_autocast_conv3d", amp)
print(json.dumps(out))
"""
CZ_PACKAGE_LIST = r"""
import importlib.metadata as m, os, sys
meta = os.path.join(sys.prefix, "conda-meta")
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("# python packages (importlib.metadata)\n")
    f.write("\n".join(sorted("%s==%s" % (d.metadata["Name"] or "?", d.version) for d in m.distributions())) + "\n")
    f.write("# conda packages (conda-meta)\n")
    f.write(("\n".join(sorted(n[:-5] for n in os.listdir(meta) if n.endswith(".json"))) + "\n") if os.path.isdir(meta) else "")
"""
CZ_DATA_CHECK = r"""
import os, time
from cryozeta.configs.configs_data import data_configs
files = [data_configs["ccd_components_file"], data_configs["ccd_components_rdkit_mol_file"]]
missing = [f for f in files if not os.path.isfile(f)]
assert not missing, "the package looks for files that are not there: " + ", ".join(missing)
from cryozeta.data import ccd
t0 = time.time()
atoms = ccd.get_component_atom_array("ALA", keep_leaving_atoms=True, keep_hydrogens=False)
t1 = time.time()
mol = ccd.get_component_rdkit_mol("ALA")
t2 = time.time()
assert atoms is not None and len(atoms) > 0, "the components file has no entry for ALA"
assert mol is not None, "the RDKit file has no entry for ALA"
print("the package reads its chemical-components files: components file %.0f s, RDKit file %.0f s" % (t1 - t0, t2 - t1))
"""
_env_vars_plain = env_vars


def env_vars(env_py, extra=None):
    """The environment of a program of a method (the function of the tools), and for a conda prefix (the pixi environment of CryoZeta) what its activation script sets: CONDA_PREFIX and its lib folder
    on LD_LIBRARY_PATH."""
    e = _env_vars_plain(env_py, extra)
    prefix = os.path.dirname(os.path.dirname(env_py))
    if os.path.isdir(os.path.join(prefix, "conda-meta")):
        e.setdefault("CONDA_PREFIX", prefix)
        old = os.environ.get("LD_LIBRARY_PATH", "")
        e.setdefault("LD_LIBRARY_PATH", os.path.join(prefix, "lib") + (os.pathsep + old if old else ""))
    return e


def cz_prefix(env_py):
    return os.path.dirname(os.path.dirname(env_py))


def cz_env(prefix, teaser=None, assets=None):
    """What `env_setup.sh` of the project sets when pixi activates the environment (the activation script itself needs pixi), for the programs that the notebook starts; `CRYOZETA_ASSETS_DIR` is the
    folder of the weights and of the chemical-components files (without it the package looks in `assets/` next to its own files, and an installed package has none)."""
    e = env_vars(os.path.join(prefix, "bin", "python"), {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "USE_OPM_CHUNKED": "1", "CUTLASS_PATH": prefix})
    if assets:
        e["CRYOZETA_ASSETS_DIR"] = assets
    if teaser:
        e["TEASERPP_DIR"] = teaser
        e["PYTHONPATH"] = os.path.join(teaser, "build", "python")
    return e


def pixi_vars():
    return {"PIXI_HOME": os.path.join(BIG, "pixi_home"), "PIXI_CACHE_DIR": os.path.join(BIG, "pixi_cache"), "RATTLER_CACHE_DIR": os.path.join(BIG, "rattler_cache"),
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or os.path.join(BIG, "uv_cache"), "PIXI_NO_PROGRESS": "true", "PIXI_COLOR": "never", "NO_COLOR": "1"}


def install_pixi(spec):
    """The pinned release of pixi, the program that installs the environment of CryoZeta from the lock file of the project."""
    p = spec["pixi"]
    exe = os.path.join(BIG, "pixi", "pixi-%s" % p["version"])
    if not (os.path.exists(exe) and os.access(exe, os.X_OK)):
        if not fetch(p["url"], exe, p["min_bytes"], "pixi %s" % p["version"]):
            raise RuntimeError("pixi could not be downloaded from %s" % p["url"])
        if p.get("sha256") and sha256_file(exe) != p["sha256"]:
            got = sha256_file(exe)
            os.remove(exe)
            raise RuntimeError("the downloaded pixi does not have the sha256 of release %s (%s, expected %s)" % (p["version"], got[:16], p["sha256"][:16]))
        os.chmod(exe, 0o755)
    rc, last, _ = run([exe, "--version"], env=pixi_vars(), quiet=True)
    if rc != 0 or p["version"] not in " ".join(last):
        raise RuntimeError("the downloaded pixi does not report version %s (%s)" % (p["version"], " ".join(last[-2:])[:200]))
    log("pixi: " + last[-1])
    return exe


def make_env_cryozeta(spec, repo):
    """The conda environment of CryoZeta, installed exactly as the lock file of the repository says; returns the path of its python."""
    name = spec["pixi"]["env"]
    env_py = os.path.join(repo, ".pixi", "envs", name, "bin", "python")
    test = (IMPORT_TESTS or {}).get(spec["key"]) or spec["import_test"]
    if os.path.exists(env_py):
        rc, last, _ = run([env_py, "-c", test], env=env_vars(env_py), cwd=repo, quiet=True)
        if rc == 0:
            log("the environment of %s exists already: %s" % (spec["label"], last[-1]))
            return env_py
    for f in ("pixi.lock", "pyproject.toml", "env_setup.sh"):
        if not os.path.exists(os.path.join(repo, f)):
            raise RuntimeError("the CryoZeta source at commit %s lacks %s" % (spec["commit"], f))
    pixi = install_pixi(spec)
    logf = os.path.join(WORK, "logs", "cryozeta_pixi_install.log")
    log("installing the environment of %s with pixi, from the lock file of the repository (10 to 25 minutes) ..." % spec["label"])
    rc, last, timed_out = run_watched([pixi, "install", "-e", name, "--frozen"], "pixi install", watch=[logf], env=pixi_vars(), cwd=repo, log_file=logf, timeout=3600, every=1, quiet=True)
    if rc != 0 or timed_out or not os.path.exists(env_py):
        log("\n".join(last[-25:]))
        raise RuntimeError("pixi could not install the environment %s (%s; see logs/cryozeta_pixi_install.log in the zip)" % (name, "timeout" if timed_out else "exit code %s" % rc))
    rc, last, _ = run([env_py, "-c", CZ_PACKAGE_LIST, os.path.join(WORK, "cryozeta_env_packages.txt")], env=env_vars(env_py), cwd=repo, quiet=True)      # the versions of what was installed
    if rc != 0:
        log("the list of the installed packages could not be written (exit code %s; not needed for the runs)" % rc)
    rc, last, _ = run([env_py, "-c", test], env=env_vars(env_py), cwd=repo)
    if rc != 0:
        raise RuntimeError("the packages of CryoZeta were installed but its own modules do not import (see the lines above)")
    log(last[-1])
    return env_py


_make_env_plain = make_env


def make_env(spec, repo):
    """The environment of a method: the one of the tools (uv, pinned wheels) for every method but CryoZeta, whose project is a pixi project."""
    if spec["key"] != "cryozeta":
        return _make_env_plain(spec, repo)
    if ENV_PY_GIVEN:
        return ENV_PY_GIVEN
    return make_env_cryozeta(spec, repo)


def build_teaser(spec, prefix):
    """TEASER++ (the registration library that CryoZeta imports as teaserpp_python) at a pinned commit, built with the headers of the environment (Eigen, Boost, Python), the compiler of this
    machine and pinned cmake and pybind11 (PyPI; the lock file has neither); returns the folder. The project's own task builds the master branch with the cmake of the system."""
    t = spec["teaser"]
    if t.get("prebuilt"):                                                     # the tests
        return t["prebuilt"]
    root = os.path.join(BIG, "teaser_plusplus")
    if not git_checkout(t["repo"], t["commit"], root):
        raise RuntimeError("TEASER++ could not be fetched at commit %s (see the lines above)" % t["commit"])
    build = os.path.join(root, "build")
    pkg = os.path.join(build, "python", "teaserpp_python")
    if not glob.glob(os.path.join(pkg, "_teaserpp*.so")):
        tools = os.path.join(BIG, "teaser_tools")
        cmake = os.path.join(tools, "cmake", "data", "bin", "cmake")
        pyb = os.path.join(tools, "pybind11", "share", "cmake", "pybind11")
        if not (os.path.exists(cmake) and os.path.isdir(pyb)):
            rc, last, _ = run([sys.executable, "-m", "pip", "install", "-q", "--target", tools, "cmake==" + t["cmake"], "pybind11==" + t["pybind11"]], quiet=True)
            if rc != 0 or not os.path.exists(cmake):
                log("\n".join(last[-12:]))
                raise RuntimeError("pip could not install cmake %s and pybind11 %s for the TEASER++ build" % (t["cmake"], t["pybind11"]))
        shutil.rmtree(build, ignore_errors=True)
        logf = os.path.join(WORK, "logs", "cryozeta_teaser_build.log")
        py = os.path.join(prefix, "bin", "python")
        steps = (("configure", [cmake, "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_PYTHON_BINDINGS=ON", "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF", "-DBUILD_TESTS=OFF",
                                "-DBUILD_DOC=OFF", "-DCMAKE_PREFIX_PATH=" + prefix, "-Dpybind11_DIR=" + pyb, "-DPYTHON_EXECUTABLE=" + py]),
                 ("build", [cmake, "--build", build, "-j", str(max(1, os.cpu_count() or 1))]))
        for what, cmd in steps:
            log("TEASER++: %s ..." % what)
            rc, last, timed_out = run_watched(cmd, "TEASER++ " + what, watch=[logf], env={"CC": None, "CXX": None, "PYTHONPATH": None}, log_file=logf, timeout=3000, every=1, quiet=True)
            if rc != 0 or timed_out:
                log("\n".join(last[-25:]))
                raise RuntimeError("TEASER++ could not be built (%s: %s; see logs/cryozeta_teaser_build.log in the zip)" % (what, "timeout" if timed_out else "exit code %s" % rc))
        if not glob.glob(os.path.join(pkg, "_teaserpp*.so")):
            raise RuntimeError("the TEASER++ build did not make the Python module (nothing like %s)" % os.path.join(pkg, "_teaserpp*.so"))
    init = os.path.join(root, "python", "teaserpp_python", "__init__.py")
    if os.path.exists(init) and not os.path.exists(os.path.join(pkg, "__init__.py")):
        shutil.copyfile(init, os.path.join(pkg, "__init__.py"))               # as env_setup.sh does: the build makes the compiled module only
    return root


def fetch_cryozeta_weights(spec):
    """The three weights files and the two chemical-components files from the Hugging Face repository, at the revision that the Hub reports, with the size and the sha256 that the Hub lists for them;
    returns (folder, facts, revision)."""
    import urllib.request
    hub = spec["hub"]
    try:
        with urllib.request.urlopen(urllib.request.Request(hub["api"], headers={"User-Agent": "mica-benchmark-cryozeta"}), timeout=60) as r:
            meta = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as ex:
        raise RuntimeError("the file list of %s could not be read from %s: %s" % (hub["repo"], hub["api"], str(ex)[:200]))
    rev = meta.get("sha") or "main"
    sib = {s["rfilename"]: s for s in meta.get("siblings", []) if s.get("rfilename")}
    with open(os.path.join(WORK, "cryozeta_hub_files.json"), "w", encoding="utf-8") as f:
        json.dump({"repo": hub["repo"], "revision": rev, "last_modified": meta.get("lastModified", ""),
                   "files": [{"path": k, "size": v.get("size"), "lfs_sha256": (v.get("lfs") or {}).get("sha256", "")} for k, v in sorted(sib.items())]}, f, indent=1)
    assets = os.path.join(BIG, "cryozeta_assets")
    facts = {}
    for role, fname in list(hub["files"].items()) + list(hub.get("data_files", {}).items()):
        paths = [k for k in sib if k == fname] or [k for k in sib if os.path.basename(k) == fname]
        if not paths:
            raise RuntimeError("the Hugging Face repository %s has no file %s at revision %s (it has %s)" % (hub["repo"], fname, rev[:10], sorted(sib)[:12]))
        meta_f = sib[paths[0]]
        lfs = meta_f.get("lfs") or {}
        size = meta_f.get("size") or lfs.get("size")
        dest = os.path.join(assets, fname)
        if not fetch(hub["resolve"].format(rev=rev, path=paths[0]), dest, hub["min_bytes"], "CryoZeta file " + fname, expect_size=size):
            raise RuntimeError("the file %s could not be downloaded from %s" % (fname, hub["repo"]))
        digest = sha256_file(dest)
        if lfs.get("sha256") and digest != lfs["sha256"]:
            raise RuntimeError("the file %s does not have the sha256 that the Hub lists (%s, expected %s)" % (fname, digest[:16], lfs["sha256"][:16]))
        facts[role] = {"file": fname, "hub_path": paths[0], "bytes": os.path.getsize(dest), "sha256": digest, "hub_sha256": lfs.get("sha256", "")}
        log("%s: %d MB, sha256 %s%s" % (fname, facts[role]["bytes"] // 1000000, digest[:16], " (the Hub's value)" if lfs.get("sha256") else " (the Hub lists none)"))
    return assets, facts, rev


def cryozeta_preflight(spec, repo, env_py, prefix):
    """Before any hour is spent: the modules of the three programs import together with TEASER++, and (on a GPU) bfloat16 works. A GPU that cannot run bfloat16 in the detection network gets
    the two-line change of the detection code (float32) and a float32 start; nothing is changed on a GPU that can. The package must also find and read its chemical-components files (the run of
    2 November 2026 stopped on this: `cryozeta-inference` logged the missing file and ended with exit code 0)."""
    env = cz_env(prefix, CZ["teaser"], CZ["assets"])
    code = "import cryozeta.runner.inference_detection, cryozeta.runner.inference, cryozeta.runner.combine, teaserpp_python; print('the modules of the three programs and TEASER++ import')"
    if not ENV_PY_GIVEN:
        rc, last, _ = run([env_py, "-c", code], env=env, cwd=repo)
        if rc != 0:
            raise RuntimeError("the modules of the CryoZeta programs do not import together with TEASER++ (see the lines above)")
        rc, last, timed_out = run([env_py, "-c", CZ_DATA_CHECK], env=env, cwd=repo, timeout=1200)
        if rc != 0 or timed_out:
            raise RuntimeError("the CryoZeta package cannot read its chemical-components files in %s (%s; see the lines above)" % (CZ["assets"], "timeout" if timed_out else "exit code %s" % rc))
    pf, patched = {}, ""
    if REQUIRE_GPU and not ENV_PY_GIVEN:
        rc, last, _ = run([env_py, "-c", CZ_PREFLIGHT], env=env, cwd=repo, quiet=True)
        try:
            pf = json.loads(last[-1])
        except (ValueError, IndexError):
            raise RuntimeError("the GPU test of the CryoZeta environment gave no result (exit code %s): %s" % (rc, " ".join(last[-3:])[:300]))
        log("GPU test of the CryoZeta environment: %s" % json.dumps(pf))
        if not pf.get("cuda_available"):
            raise RuntimeError("PyTorch in the CryoZeta environment does not see the GPU")
        if pf.get("bf16_conv3d") != "ok" or pf.get("bf16_autocast_conv3d") != "ok":
            patched = CZT.patch_detection_precision(repo)
            log("this GPU cannot run the detection network in bfloat16 (%s): two lines of the detection code are changed to float32:\n%s" % (pf.get("bf16_conv3d"), patched))
    with open(os.path.join(WORK, "cryozeta_source_patch.diff"), "w", encoding="utf-8") as f:
        f.write(sh("git -C %s diff" % repo, timeout=60) + "\n")
    CZ["dtype"] = spec["run"]["dtype"] if (not pf or pf.get("bf16_matmul") == "ok") else "fp32"
    CZ["preflight"] = pf
    STATE["methods"][spec["key"]]["preflight"] = {"gpu_test": pf, "detection_patched": bool(patched), "dtype_start": CZ["dtype"]}
    if CZ["dtype"] != spec["run"]["dtype"]:
        log("bfloat16 matrix products do not work on this GPU: the runs start in float32")


def setup_cryozeta(spec, repo, env_py):
    """TEASER++, the weights and the checks on this GPU; the environment is the one that make_env made."""
    prefix = cz_prefix(env_py)
    missing = [p for p in CZ_PROGRAMS if not os.path.exists(os.path.join(prefix, "bin", p))]
    if missing:
        raise RuntimeError("the environment lacks the programs %s (the project's console scripts)" % ", ".join(missing))
    m = STATE["methods"][spec["key"]]
    CZ["teaser"] = build_teaser(spec, prefix)
    CZ["assets"], files, rev = fetch_cryozeta_weights(spec)
    m["weights_hub"] = {"repo": spec["hub"]["repo"], "revision": rev, "files": files}
    cryozeta_preflight(spec, repo, env_py, prefix)
    m["version"] = "commit %s, pixi %s, environment %s, weights revision %s" % (spec["commit"][:10], spec["pixi"]["version"], spec["pixi"]["env"], rev[:10])
    m["teaser"] = {"commit": spec["teaser"]["commit"], "folder": CZ["teaser"]}
    record_files(spec["key"], [CZ["assets"]])


def plan_cryozeta(spec, ctx, repo, env_py):
    """One run per entry: the runner of the repository (scripts/cryozeta_tools.py) with a settings file; it fetches the contour level and the alignments, runs the four programs of the
    authors' demo and writes the summary of what happened."""
    entry, row = ctx["entry"], ctx["row"]
    out = os.path.join(RUN_ROOT, "cryozeta", entry)
    inp = os.path.join(INPUT_ROOT, entry)
    run_s = spec["run"]
    cfg = {"name": entry, "repo": repo, "env_dir": cz_prefix(env_py), "teaser_dir": CZ["teaser"], "assets": CZ["assets"], "weights": spec["hub"]["files"],
           "map_path": ctx["map"], "resolution": float(row["resolution_A"]), "header_url": CZT.emdb_header_url(row["map_url"]), "colabfold_host": spec["colabfold"],
           "chains": [{"sequence": j["sequence_submitted"].strip(), "count": int(j["copies_in_entry"])} for j in ctx["jobs"]], "msa_root": os.path.join(inp, "cryozeta_msa"), "msa_poll_s": float(spec["msa_poll_s"]), "msa_tries": int(spec["msa_tries"]),
           "input_json": os.path.join(inp, "cryozeta_input.json"), "dump_dir": out, "seed": run_s["seed"], "n_sample": run_s["n_sample"], "n_step": run_s["n_step"], "n_cycle": run_s["n_cycle"],
           "num_select": run_s["num_select"], "dtype": CZ.get("dtype", run_s["dtype"]), "kernels": run_s["kernels"], "chunk_sizes": list(CZT.CHUNK_LADDER),
           "summary_path": os.path.join(out, "driver_summary.json"), "progress_path": os.path.join(out, "progress.txt")}
    os.makedirs(inp, exist_ok=True)
    cfg_path = os.path.join(inp, "cryozeta_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    final = os.path.join(out, entry, "CryoZeta-Final")
    outputs = {("cryozeta" if i == 0 else "cryozeta_s%d" % i): os.path.join(final, "%s_sample_%d.cif" % (entry, i)) for i in range(int(run_s["num_select"]))}
    cmd = [sys.executable, os.path.join(WORK, "scripts", "cryozeta_tools.py"), "run", cfg_path]
    # the runner itself is plain Python of this machine: it gets the environment of any method (not the library path of the conda environment, which its own programs get from the runner)
    return [{"variant": "run", "cmd": cmd, "cwd": repo, "env": _env_vars_plain(env_py), "out_dir": out, "outputs": outputs, "main": "cryozeta", "watch": [cfg["progress_path"]]}]


SETUPS["cryozeta"], PLANS["cryozeta"] = setup_cryozeta, plan_cryozeta
'''
RUN_MD_CRYOZETA = """\
## Step 6. The runs

First the setup of CryoZeta (source, environment, TEASER++, weights, checks on this GPU), then one worker per GPU takes entries from the list until the time budget says stop. A progress line of every running
program appears every two minutes (the stage that the runner is in and its last line); the output of the programs is in `logs/` of the zip. Everything is packed into a zip after every run."""


EXTRA_INFO = ('{"methods": {k: {"label": m["label"], "commit": m["commit"], "status": m["status"], "setup_seconds": m["setup_seconds"], "version": m.get("version", "")} for k, m in STATE["methods"].items()}, '
              '"specs": globals().get("SPECS", {}), "usalign_commit": globals().get("USALIGN_COMMIT", ""), "usalign_version": STATE["usalign_version"], "input_sha256": globals().get("EXPECTED", {}), '
              '"tasks": globals().get("TASKS"), "hours_setting": globals().get("HOURS_SETTING"), "run_limit_s": globals().get("RUN_LIMIT"), "lanes": globals().get("LANES")}')


LANES_TEXT = {1: "* The notebook runs **one program at a time** (on the first GPU of the session). In the first session some programs were killed for lack of RAM while the program of the other GPU ran; "
                 "the entries of this session run alone, so a kill now is a result and not a neighbour's fault.",
              2: "* The Kaggle session has **two GPUs**. The notebook uses both: two programs run at the same time, one on each GPU, so the work of about 2 x @HOURS_TEXT@ hours fits in the session."}


# ---------------------------------------------------------------------------------------------------------- the notebook
def build(out_path, batch="batch1", entries=None, tasks=None, hours=9.0, guard_minutes=60, max_lanes=2, usalign_repo=P.USALIGN_REPO, usalign_commit=P.USALIGN_COMMIT, entries_path=None, env_python=None,
          usalign_bin=None, net_hosts=NET_HOSTS, require_gpu=True, import_tests=None, specs=None, run_methods=None, alone_first=(), rate=None, order_text=None, rate_sentence=None):
    """Writes the notebook. `tasks` ([entry, method] pairs) overrides `entries`; `alone_first` ([entry, method] pairs) run first and alone; `rate` (seconds per residue) replaces the first
    guess of CryoZeta's time; `order_text` and `rate_sentence` are the sentences of the intro about the order and the rate; the keyword arguments after `hours` exist for the tests."""
    run_methods = list(run_methods or ORDER)
    solo = next((k for k in SOLO if k in run_methods), None)              # CryoAtom v1.0.0 and CryoZeta have a notebook of their own
    if solo and run_methods != [solo]:
        raise ValueError("%s is built into its own notebook: run_methods must be ['%s'] alone" % (SPECS_ALL[solo]["label"], solo))
    specs = copy.deepcopy(specs or (SPECS_ALL if solo else SPECS))
    if solo:
        specs = {solo: specs[solo]}
        if rate:
            specs[solo]["s_per_residue"] = float(rate)
        if net_hosts is NET_HOSTS:
            net_hosts = SOLO_NET_HOSTS[solo]
    rows = read_entries(entries_path)
    entries = list(entries or seeded_order(rows))
    tasks = [list(t) for t in (tasks if tasks is not None else all_tasks(entries, [solo] if solo else ORDER))]
    sizes = {r["pdb_id"]: int(r["modeled_residues"]) for r in rows}
    known = {r["pdb_id"] for r in rows}
    for e, m in tasks:
        if e not in known or m not in specs:
            raise ValueError("task %s %s: unknown entry or method" % (e, m))
    wanted = {m for _e, m in tasks}
    run_methods = [m for m in run_methods if m in wanted]                 # only the methods that have a task are set up (a session that finishes one method's leftovers must not install the others)
    notebook = "gpu_" + batch
    solo_files = (("scripts/cryozeta_tools.py", "scripts/cryozeta_tools.py"),) if solo == "cryozeta" else (((specs[solo]["req"], specs[solo]["req"]),) if solo else ())
    files = FILES if not solo else BASE_FILES + solo_files
    file_cells, expected = nbkit.file_cells(files, entries_path)
    step5 = nbkit.fill(P.STEP5, {"@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit, "@MICA_REPO@": "x", "@MICA_COMMIT@": "x", "@WEIGHTS_URL@": "x", "@ZENODO_API@": "x", "@USALIGN_BIN@": "None"})
    git_and_usalign = nbkit.function_sources(step5, ["git_checkout", "build_usalign"]).rstrip("\n")
    step6 = nbkit.fill(P.STEP6, {"@ENTRIES@": "[]"})
    read_write_download = nbkit.function_sources(step6, ["read_csv", "write_csv", "download"]).rstrip("\n")
    read_write_download = nbkit.replace_once(read_write_download, "time.sleep(5 * attempt)", "time.sleep(0.5 * RETRY_WAIT * attempt)")      # 5, 10, 15 seconds as before (RETRY_WAIT is 10); the tests make it short
    find_af3 = nbkit.function_sources(step6, ["find_af3_dir"]).rstrip("\n")
    mapping = {"@UV_VERSION@": P.UV_VERSION, "@ENV_PYTHON@": repr(env_python), "@IMPORT_TESTS@": repr(import_tests), "@SPECS@": repr(specs), "@RUN_METHODS@": repr(run_methods),
               "@ENTRIES@": repr(entries), "@ENTRY_LIST@": ", ".join(entries), "@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit,
               "@USALIGN_BIN@": repr(usalign_bin), "@REQUIRE_GPU@": repr(bool(require_gpu)), "@NET_HOSTS@": repr(tuple(net_hosts)), "@EXPECTED@": repr(expected),
               "@NOTEBOOK@": notebook, "@EXTRA_INFO@": EXTRA_INFO, "@BATCH@": batch, "@HOURS@": repr(float(hours)), "@GUARD_MINUTES@": repr(int(guard_minutes)), "@HOURS_TEXT@": ("%g" % hours),
               "@MAX_LANES@": str(int(max_lanes)), "@LANES_TEXT@": LANES_TEXT[1 if int(max_lanes) <= 1 else 2].replace("@HOURS_TEXT@", "%g" % hours), "@N_TASKS@": str(len(tasks)), "@LIMIT_TEXT@": ("%g" % ((specs[solo]["timeout"] if solo else RUN_LIMIT_S) / 3600)), "@RATE_TEXT@": ("%g" % (specs[solo].get("s_per_residue") or 0 if solo else 0)), "@SIZES@": repr(sizes), "@RUN_LIMIT@": repr(RUN_LIMIT_S), "@EXPLOSIVE@": repr(EXPLOSIVE), "@TASKS@": repr(tasks),
               "@ALONE_FIRST@": repr([list(t) for t in alone_first]),
               "@ORDER_TEXT@": order_text or "in the order of the seeded selection (the two resolution classes interleaved)",
               "@RATE_SENTENCE@": rate_sentence or ("No run of CryoZeta has been timed on a T4 yet: the first estimate is %g seconds per residue of the deposited model; as soon as one run has finished, "
                                                    "the measured rate replaces it." % (specs[solo].get("s_per_residue") or 0 if solo else 0))}
    fl = lambda text: nbkit.fill(text, mapping).strip("\n") + "\n"
    # the cells of the methods notebook that are reused: the head of its tools cell (up to run_program), five of its functions, and the cells of the three methods without the lines that ran them
    head = nbkit.cut_before(MM.TOOLS, "def run_program").replace("@READ_WRITE_DOWNLOAD@", read_write_download).replace("@GIT_AND_USALIGN@", git_and_usalign).replace("@FIND_AF3@", find_af3)
    full_old = nbkit.fill(MM.TOOLS, {}).replace("@READ_WRITE_DOWNLOAD@", read_write_download).replace("@GIT_AND_USALIGN@", git_and_usalign).replace("@FIND_AF3@", find_af3)
    prepare = nbkit.function_sources(fl(full_old), ["prepare_inputs", "af3_inputs", "score_model", "keep_results", "setup_method"]).rstrip("\n")
    tools = head.rstrip("\n") + "\n\n\n" + TOOLS_TAIL.replace("@PREPARE_FUNCTIONS@", prepare)
    marker = 'if STATE["ok"] and STATE["usalign"]:'
    e3 = nbkit.replace_once(nbkit.cut_before(MM.E3, marker), '(("prealign", "pre_align"), ("denovo", "denovo"))', '(("prealign", "pre_align"),)')
    emprot, emodelx = nbkit.cut_before(MM.EMPROT, marker), nbkit.cut_before(MM.EMODELX, marker)
    if solo == "cryoatom1":
        cryoatom = nbkit.replace_once(nbkit.cut_before(MM.CRYOATOM, marker), CRYOATOM_REGISTER_OLD, "").rstrip("\n") + "\n" + CRYOATOM_SESSION
        method_cells = [("cryoatom-title", nbkit.md(CRYOATOM_MD)), ("cryoatom", nbkit.code(fl(cryoatom)))]
    elif solo == "cryozeta":
        method_cells = [("cryozeta-title", nbkit.md(CRYOZETA_MD)), ("cryozeta", nbkit.code(fl(CRYOZETA)))]
    else:
        method_cells = [("modelangelo-title", nbkit.md(MODELANGELO_MD)), ("modelangelo", nbkit.code(fl(MODELANGELO))),
                        ("e3-title", nbkit.md(E3_MD)), ("e3", nbkit.code(fl(e3))),
                        ("emprot-title", nbkit.md(EMPROT_MD)), ("emprot", nbkit.code(fl(emprot))),
                        ("emodelx-title", nbkit.md(EMODELX_MD)), ("emodelx", nbkit.code(fl(emodelx)))]
    cells = [("intro", nbkit.md(nbkit.fill({"cryoatom1": INTRO_CRYOATOM, "cryozeta": INTRO_CRYOZETA}.get(solo, INTRO), mapping))),
             ("settings-title", nbkit.md(SETTINGS_MD)), ("settings", nbkit.code(fl({"cryoatom1": SETTINGS_CRYOATOM, "cryozeta": SETTINGS_CRYOZETA}.get(solo, SETTINGS)))),
             ("step1-title", nbkit.md(P.STEP1_MD)), ("step1", nbkit.code(fl(P.STEP1))),
             ("step2-title", nbkit.md(P.STEP2_MD)), ("step2-dirs", nbkit.code(P.STEP2_DIRS.strip("\n") + "\n"))]
    cells += file_cells
    cells += [("step2-check", nbkit.code(fl(P.STEP2_CHECK))),
              ("robust-title", nbkit.md(nbkit.fill(nbkit.ROBUST_MD, mapping))), ("robust", nbkit.code(fl(nbkit.ROBUST))),
              ("tools-title", nbkit.md(TOOLS_MD)), ("tools", nbkit.code(fl(tools))),
              ("step4-title", nbkit.md(MM.STEP4_MD)), ("step4", nbkit.code(fl(MM.STEP4))),
              ] + method_cells + [
              ("run-title", nbkit.md({"cryoatom1": RUN_MD_CRYOATOM, "cryozeta": RUN_MD_CRYOZETA}.get(solo, RUN_MD))), ("run", nbkit.code(fl(RUN))),
              ("summary-title", nbkit.md(SUMMARY_MD.replace("Step 10", "Step 7") if solo else SUMMARY_MD)), ("summary", nbkit.code(fl(SUMMARY))),
              ("final-title", nbkit.md(nbkit.FINAL_MD)), ("final", nbkit.code(fl(nbkit.FINAL)))]
    nbkit.assemble(out_path, "gpu_%s.ipynb" % batch, cells)
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", default="batch1", help="the name of the batch (the notebook is scripts/gpu_<batch>.ipynb and its zip gpu_<batch>_result_....zip)")
    p.add_argument("--out", default=None)
    p.add_argument("--entries", nargs="+", default=None, help="the entries (default: all, in the seeded order)")
    p.add_argument("--methods", nargs="+", default=ORDER, choices=ORDER_ALL, help="the four methods of the first batches, or cryoatom1 or cryozeta alone (their own notebooks)")
    p.add_argument("--done", nargs="*", default=[], help="method_status.csv files of earlier sessions: their finished tasks (ok or failed) are not run again")
    p.add_argument("--redo", nargs="*", default=[], help="ENTRY:method pairs to run again although an earlier session finished them")
    p.add_argument("--hours", type=float, default=None, help="the budget of the session in hours (default 9; 11 for CryoAtom and CryoZeta)")
    p.add_argument("--lanes", type=int, default=2, help="GPUs used at the same time (1 for entries that were killed for lack of RAM while another program ran)")
    p.add_argument("--guard-minutes", type=int, default=60, help="a crash sooner than this on two entries in a row pauses the method for the session (0: no guard)")
    p.add_argument("--alone-first", nargs="*", default=[], help="ENTRY:method pairs that run first and alone (killed for lack of RAM next to another program in an earlier session)")
    p.add_argument("--rate", type=float, default=None, help="CryoZeta: seconds per residue for the first estimate of the time (default: what the --done files measured, else 15)")
    p.add_argument("--order-text", default=None, help="the sentence of the intro about the order of the entries")
    args = p.parse_args(argv)
    rows = read_entries()
    entries = args.entries or seeded_order(rows)
    tasks = pending_tasks(all_tasks(entries, args.methods), args.done, args.redo)
    out = args.out or os.path.join(HERE, "gpu_%s.ipynb" % args.batch)
    hours = args.hours if args.hours is not None else (11.0 if any(k in args.methods for k in SOLO) else 9.0)
    rate, rate_sentence = args.rate, None
    if "cryozeta" in args.methods:
        seen = measured_rates(args.done, "cryozeta", {r["pdb_id"]: int(r["modeled_residues"]) for r in rows})
        if seen:
            rate = rate or round(seen["mean"])
            rate_sentence = ("The earlier session%s timed %d runs of CryoZeta on this GPU: %.0f to %.0f seconds per residue of the deposited model (%.0f for all together). The first estimate is "
                             "%g; as soon as a run of this session has finished, the largest rate measured in this session replaces it." % ("s" if len(args.done) > 1 else "", seen["n"], seen["lo"], seen["hi"], seen["mean"], rate))
    order_text = args.order_text or ("in an order that was written down before the session (LOG.md)" if args.entries else None)
    alone_first = [tuple(x.split(":")) for x in args.alone_first]
    expected = build(out, batch=args.batch, entries=entries, tasks=tasks, hours=hours, guard_minutes=args.guard_minutes, run_methods=args.methods, max_lanes=args.lanes,
                     alone_first=alone_first, rate=rate, order_text=order_text, rate_sentence=rate_sentence)
    print("wrote %s: %d tasks (%d entries), %d embedded files" % (out, len(tasks), len({t[0] for t in tasks}), len(expected)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
