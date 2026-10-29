#!/usr/bin/env python3
"""Build scripts/pilot_methods.ipynb: the Phase 4 pilot of the four methods that came after the MICA and ModelAngelo pilots, on the pilot entry (default 26DK):
EModelX(+AF), CryoAtom (the code of the paper, v1.0.0, and the current CryoAtom2), EMProt and E3-CryoFold.

One notebook, one Kaggle run: each method has its own Python environment (built from exact pins in env/*_requirements.txt with `uv`, from PyPI wheels, not conda),
its own checkout at a pinned commit and its own weights; the methods run one after the other, a method that fails does not stop the next one, and every model that a
method writes is scored with US-align. EModelX(+AF) and EMProt take the same AlphaFold Server predictions as MICA (decision 6 of the spec); CryoAtom and
E3-CryoFold take the map and the sequence only. No Phenix and no secret are needed, but the private AlphaFold dataset must be attached.

    python make_methods_notebook.py            # writes scripts/pilot_methods.ipynb

Run it on Kaggle with the GPU on, as "Save & Run All (Commit)".
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_pilot_notebook as P  # noqa: E402
import nbkit  # noqa: E402

NOTEBOOK = "methods_pilot"
NB_NAME = "pilot_methods.ipynb"

# What differs between the methods is data (this table) and the code of the three functions of each method (setup, plan) in its cell.
# The commits are hashes, so they cannot change under us; the Python versions are the ones that uv installs.
SPECS = {
    "e3": {
        "key": "e3", "label": "E3-CryoFold", "repo": "https://github.com/A4Bio/E3-CryoFold", "commit": "241cef48da83b96519e62ab02cccd9db96d8e68b",   # main, 2025-09-01
        "python": "3.9.25", "req": "env/e3cryofold_requirements.txt", "timeout": 3600, "uses_af3": False,
        "weights_url": "https://github.com/A4Bio/E3-CryoFold/releases/download/checkpoint/models.zip", "weights_min_bytes": 50000000, "gdrive_id": None,
        "import_test": ("import sys, torch, pytorch_lightning, numpy, scipy, Bio, mrcfile, open3d, skimage, sklearn, superpose3d, esm, networkx, nglview, matplotlib, PIL, requests; "
                        "from cryo_module.AutoEM import run_pulchra; from E3CryoFold2 import E3CryoFold2; "
                        "print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| numpy', numpy.__version__, '| open3d', open3d.__version__)"),
    },
    "emprot": {
        "key": "emprot", "label": "EMProt", "repo": "https://github.com/huang-laboratory/EMProt", "commit": "9489d52e7fdc81c85ba82e386abcfa05bac2b9d1",
        "python": "3.10.20", "req": "env/emprot_requirements.txt", "timeout": 7200, "uses_af3": True,
        "weights_url": "http://huanglab.phys.hust.edu.cn/EMProt/pretrained_weights/weights.tgz", "weights_min_bytes": 1000000, "gdrive_id": None,
        "import_test": ("import sys, torch, numpy, emprot, emprot.run, emprot.eval, emprot.clean, emprot.fit, emprot.refine; "
                        "print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| numpy', numpy.__version__, '| emprot', emprot.__version__)"),
    },
    "cryoatom2": {
        "key": "cryoatom2", "label": "CryoAtom2 (v2.1.0 and later fixes)", "repo": "https://github.com/YangLab-SDU/CryoAtom", "commit": "856e250df7b784b854b892f1b619d32d51188cef",   # master, 2026-06-17
        "python": "3.9.25", "req": "env/cryoatom2_requirements.txt", "timeout": 7200, "uses_af3": False, "pkg": "CryoAtom2", "ckpt_files": ["CryoNet.pth", "RUNet.pth"],
        "cli": ["-v", "{map}", "-ps", "{fasta}", "-o", "{out}", "-d", "0"],
        "weights_url": "https://yanglab.qd.sdu.edu.cn/CryoAtom/download/checkpoints_v2.1.zip", "weights_min_bytes": 100000000, "gdrive_id": None,
        "rnafm_url": "https://yanglab.qd.sdu.edu.cn/CryoAtom/download/RNA-FM_pretrained.pth", "rnafm_bytes": 1194424423,
        "import_test": ("import sys, torch, numpy, esm, fm, CryoAtom2, CryoAtom2.build; "
                        "print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| numpy', numpy.__version__, '| CryoAtom2', CryoAtom2.__version__)"),
    },
    "cryoatom1": {
        "key": "cryoatom1", "label": "CryoAtom v1.0.0 (code of the paper)", "repo": "https://github.com/YangLab-SDU/CryoAtom", "commit": "e2a31f2b2ff21fae363490ac5f4a5e945513024c",   # tag v1.0.0
        "python": "3.8.20", "req": "env/cryoatom1_requirements.txt", "timeout": 7200, "uses_af3": False, "pkg": "CryoAtom", "ckpt_files": ["CryNet.pth", "SimpleUnet.pth"],
        "cli": ["-s", "{fasta}", "-v", "{map}", "-o", "{out}"],
        "weights_url": "https://yanglab.qd.sdu.edu.cn/CryoAtom/download/checkpoints_v1.0.zip", "weights_min_bytes": 50000000, "gdrive_id": None,
        "import_test": ("import sys, torch, numpy, esm, CryoAtom, CryoAtom.build; "
                        "print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| numpy', numpy.__version__)"),
    },
    "emodelx": {
        "key": "emodelx", "label": "EModelX(+AF)", "repo": "https://github.com/biomed-AI/EModelX", "commit": "53713cf820d10d78b1ca6669ef197850cf907843",   # main, 2025-02-18
        "python": "3.8.20", "req": "env/emodelx_requirements.txt", "timeout": 7200, "uses_af3": True, "ckpt_files": ["best_BB_model.ckpt", "best_CA_model.ckpt", "best_AA_model.ckpt"],
        "weights_url": None, "weights_min_bytes": 50000000, "gdrive_id": "13BKzEBfL0uubYgcJJTGSZ-PkO9hXAB9X",
        "import_test": ("import sys, torch, numpy, mrcfile, networkx, open3d, pandas, requests, skimage, superpose3d, Bio; "
                        "from modules.AutoEM import Solver; from unet3d.emmodel import ResUNet3D4EM; "
                        "print('python', sys.version.split()[0], '| torch', torch.__version__, '| open3d', open3d.__version__, '| numpy', numpy.__version__)"),
    },
}
ORDER = ["e3", "emprot", "cryoatom2", "cryoatom1", "emodelx"]          # quick and likely first; EModelX last because its weights come from Google Drive
ENV_FILES = [SPECS[k]["req"] for k in ORDER]
FILES = tuple([("scripts/nbtools.py", "scripts/nbtools.py"), ("scripts/mica_inputs.py", "scripts/mica_inputs.py"), ("scripts/evaluate_models.py", "scripts/evaluate_models.py"),
               ("data/af3/af3_jobs.csv", "data/af3/af3_jobs.csv"), ("data/af3/af3_files_sha256.txt", "data/af3/af3_files_sha256.txt"),
               ("data/entries_2026.csv", "data/entries_2026.csv")] + [(f, f) for f in ENV_FILES])
NET_HOSTS = (("PyPI", "https://pypi.org/simple/uv/", True),
             ("GitHub", "https://github.com/A4Bio/E3-CryoFold", True),
             ("GitHub files", "https://github.com/A4Bio/E3-CryoFold/releases/download/checkpoint/models.zip", False),
             ("Meta ESM weights", "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt", False),
             ("SDU (CryoAtom)", "https://yanglab.qd.sdu.edu.cn/CryoAtom/download/checkpoints_v2.1.zip", False),
             ("HUST (EMProt)", "http://huanglab.phys.hust.edu.cn/EMProt/pretrained_weights/weights.tgz", False),
             ("Google Drive", "https://drive.google.com/uc?id=13BKzEBfL0uubYgcJJTGSZ-PkO9hXAB9X", False),
             ("EBI FTP", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-80558/header/emd-80558.xml", False),
             ("RCSB", "https://files.rcsb.org/header/26DK.cif", False))

INTRO = """\
# MICA benchmark: pilot of EModelX(+AF), CryoAtom, EMProt and E3-CryoFold on entry @ENTRY_LIST@

The four methods that were not piloted yet, in one run. Each method gets its own Python environment (exact pins in `env/*_requirements.txt`, installed with `uv` from PyPI wheels),
its own source at a pinned commit, and its own weights. The methods run one after the other; **a method that fails does not stop the next one**. Every model a method writes is
scored with US-align. EModelX(+AF) and EMProt get the same AlphaFold Server predictions as MICA (private dataset); CryoAtom and E3-CryoFold get the map and the sequence only.
CryoAtom is run twice: the code of the paper (v1.0.0) and the current CryoAtom2; E3-CryoFold twice: the default `pre_align` protocol and `denovo`.

**What you do (Kaggle, background mode)**: accelerator **GPU T4 x2** (not P100), Internet **On**, the private AlphaFold dataset added as input (no secrets), then
*Save Version -> Save & Run All (Commit)*. It runs on Kaggle's servers; you can close the tab and your Wi-Fi does not matter. About 2 to 5 hours. When the version shows
*Succeeded*, open it and download the zip `methods_pilot_result_....zip` from its Output. Send it to me. **Do not stop or cancel the run** unless it is stuck for more than 6 hours.
"""

TOOLS_MD = """\
## Step 3. Tools shared by the methods

`uv` and the environment builder, the downloader (curl, with resume), the archive helpers, the program runner (time limit, GPU and RAM log, a progress line every two
minutes) and the scoring. Nothing is installed or downloaded here yet."""
TOOLS = r'''
sys.path.insert(0, os.path.join(WORK, "scripts"))
import evaluate_models, mica_inputs

UV_VERSION = "@UV_VERSION@"
ENV_PY_GIVEN = @ENV_PYTHON@                       # None: build the environments; a path is only used by the tests
IMPORT_TESTS = @IMPORT_TESTS@                     # None: the import test of each method (a dict is only used by the tests)
SPECS = @SPECS@
RUN_METHODS = @RUN_METHODS@                       # the order in which the methods run; delete a name to leave that method out
ENTRIES = @ENTRIES@
USALIGN_REPO = "@USALIGN_REPO@"
USALIGN_COMMIT = "@USALIGN_COMMIT@"
USALIGN_GIVEN = @USALIGN_BIN@                     # None: compile it; a path is only used by the tests
TORCH_HOME = os.path.join(BIG, "torch_home")      # the ESM weights and other downloads of torch.hub go here
RETRY_WAIT = float(os.environ.get("MICA_RETRY_WAIT", "10"))          # seconds; the tests make it short
PRIVATE_KINDS = ("emprot_fit",)                  # models that are the AlphaFold Server predictions placed into the map: scored, but not copied into the zip
INPUT_ROOT = os.path.join(BIG, "input")
RUN_ROOT = os.path.join(BIG, "runs")
os.makedirs(TORCH_HOME, exist_ok=True)
os.makedirs(os.path.join(WORK, "logs"), exist_ok=True)
STATE.setdefault("methods", {})
UV = {}
GPU_TEST = r"""
import torch
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(3)
p = torch.cuda.get_device_properties(0)
print("gpu: %s | %.1f GB | capability %d.%d" % (p.name, p.total_memory / 2**30, p.major, p.minor))
x = torch.randn(1, 4, 24, 24, 24, device="cuda")
y = torch.nn.Conv3d(4, 8, 3, padding=1).cuda()(x)
a = torch.randn(512, 512, device="cuda")
z = float((a @ a).sum()) + float(y.sum())
torch.cuda.synchronize()
print("3D convolution and matrix product on the GPU work (torch %s, cuda %s)" % (torch.__version__, torch.version.cuda))
"""


@READ_WRITE_DOWNLOAD@


@GIT_AND_USALIGN@


@FIND_AF3@


def ensure_uv():
    if UV:
        return True
    rc, last, _ = run([sys.executable, "-m", "pip", "install", "-q", "uv==" + UV_VERSION])
    if rc != 0:
        return False
    UV.update(cmd=[sys.executable, "-m", "uv"],
              env={"UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or os.path.join(BIG, "uv_cache"),
                   "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR") or os.path.join(BIG, "uv_python"), "UV_PYTHON_PREFERENCE": "only-managed"})
    log("uv: " + sh("%s -m uv --version" % sys.executable))
    return True


def env_vars(env_py, extra=None):
    """The environment variables for a program of a method: its own python first on the PATH, torch.hub folder, the first GPU only."""
    e = {"PATH": os.path.dirname(env_py) + os.pathsep + os.environ.get("PATH", ""), "VIRTUAL_ENV": None, "PYTHONPATH": None, "PYTHONHOME": None,
         "TORCH_HOME": TORCH_HOME, "CUDA_VISIBLE_DEVICES": "0", "MPLBACKEND": "Agg"}
    e.update(extra or {})
    return e


def pinned(req_file):
    with open(req_file, encoding="utf-8") as f:
        return [ln.strip() for ln in f if "==" in ln and not ln.startswith("#")]


def make_env(spec, repo):
    """Builds the Python environment of a method from its pins (uv, PyPI wheels); returns the path of its python, or raises."""
    key, test = spec["key"], (IMPORT_TESTS or {}).get(spec["key"]) or spec["import_test"]
    if ENV_PY_GIVEN:
        return ENV_PY_GIVEN
    env_dir = os.path.join(BIG, "env_" + key)
    env_py = os.path.join(env_dir, "bin", "python")
    if os.path.exists(env_py):
        rc, last, _ = run([env_py, "-c", test], env=env_vars(env_py), cwd=repo, quiet=True)
        if rc == 0:
            log("the environment of %s exists already: %s" % (spec["label"], last[-1]))
            return env_py
        shutil.rmtree(env_dir, ignore_errors=True)
    if not ensure_uv():
        raise RuntimeError("pip could not install uv (see the lines above)")
    uv = UV["cmd"]
    for what, cmd in (("Python " + spec["python"], uv + ["python", "install", spec["python"]]),
                      ("the virtual environment", uv + ["venv", "--python", spec["python"], env_dir]),
                      ("the %d packages of %s (a few minutes)" % (len(pinned(spec["req"])), spec["req"]), uv + ["pip", "install", "--python", env_py, "--no-deps", "-r", spec["req"]])):
        log("installing " + what + " ...")
        rc, last, _ = run_watched(cmd, "uv: " + what[:30], env=UV["env"], every=20)
        if rc != 0:
            raise RuntimeError("uv could not install %s (exit code %d, the last lines are above)" % (what, rc))
    rc, last, _ = run([env_py, "-c", test], env=env_vars(env_py), cwd=repo)
    if rc != 0 and any("libGL" in ln or "libgomp" in ln for ln in last):
        log("a system library is missing; installing it with apt ...")
        run(["bash", "-c", "apt-get update -qq && apt-get install -y -qq libgl1 libgomp1"], every=20)
        rc, last, _ = run([env_py, "-c", test], env=env_vars(env_py), cwd=repo)
    if rc != 0:
        raise RuntimeError("the packages of %s were installed but the method's own modules do not import (see the lines above)" % spec["label"])
    log(last[-1])
    return env_py


def gpu_test(spec, env_py):
    rc, last, _ = run([env_py, "-c", GPU_TEST], env=env_vars(env_py))
    if rc == 3:
        raise RuntimeError("PyTorch in the environment of %s does not see the GPU" % spec["label"])
    if rc != 0:
        raise RuntimeError("the GPU of this session cannot run the PyTorch of %s (see the lines above)" % spec["label"])


def fetch(url, dest, min_bytes, label, insecure=False, expect_size=None, max_seconds=5400):
    """Downloads with curl (resumes a partial file, 3 attempts, gives up on a connection slower than 20 kB/s for 5 minutes); returns True when the file has at least min_bytes
    (and expect_size bytes if given)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def size():
        return os.path.getsize(dest) if os.path.exists(dest) else 0

    def good():
        return size() >= min_bytes and (expect_size is None or size() == expect_size)

    if good():
        log("%s: already downloaded (%d MB)" % (label, size() // 1000000))
        return True
    cmd = ["curl", "-sS", "-L", "--fail", "--connect-timeout", "60", "--speed-limit", "20000", "--speed-time", "300", "--max-time", str(max_seconds), "-C", "-", "-o", dest, url]
    if insecure:
        cmd.insert(1, "-k")                                  # the host of the CryoAtom weights has a certificate that wget of the authors also ignores (--no-check-certificate)
    log("downloading %s from %s" % (label, url))
    hb = nbtools.Heartbeat(label, log, every=BEAT, on_tick=lambda: log("   %s: %d MB so far" % (label, size() // 1000000)))
    hb.start()
    t0 = time.time()
    try:
        for attempt in (1, 2, 3):
            rc, last, timed_out = run(cmd, timeout=max_seconds + 300, quiet=True)
            if good():
                break
            if rc == 0 and os.path.exists(dest):
                os.remove(dest)                                  # a complete download of the wrong size (a web page instead of the file): do not resume it
            if rc == 0:
                log("download attempt %d of 3: the server sent a file of the wrong size" % attempt)
                time.sleep(RETRY_WAIT * attempt)
                continue
            log("download attempt %d of 3 failed: curl exit code %s %s" % (attempt, rc, " ".join(last[-2:])[:300]))
            time.sleep(RETRY_WAIT * attempt)
    finally:
        hb.stop()
    log("%s: %d MB in %d s" % (label, size() // 1000000, time.time() - t0))
    return good()


def fetch_gdrive(file_id, dest, min_bytes, label):
    """Downloads a file of Google Drive with gdown (pip installs it into this notebook's own Python)."""
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "gdown"], quiet=True)
    for attempt in (1, 2):
        rc, last, timed_out = run_watched([sys.executable, "-m", "gdown", "https://drive.google.com/uc?id=" + file_id, "-O", dest], label, timeout=3 * 3600, every=1)
        if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
            return True
        log("Google Drive download attempt %d of 2 failed (exit code %s): %s" % (attempt, rc, " ".join(last[-3:])[:400]))
        time.sleep(3 * RETRY_WAIT)
    return False


def unpack(archive, dest):
    os.makedirs(dest, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:                                    # a Python without the filter argument
                t.extractall(dest)
            except tarfile.FilterError:                          # a link or a path that the safe filter refuses; the archive is the authors' own weights file
                t.extractall(dest, filter="tar")
    else:
        with open(archive, "rb") as f:
            head = f.read(200)
        raise RuntimeError("%s is not a zip or tar archive (it starts with %r); Google Drive or the server may have sent a web page instead of the file" % (os.path.basename(archive), head[:80]))


def find_files(root, names):
    """{name: path} of the first file called `name` under root, for each name that exists."""
    found = {}
    for dirpath, _dirs, files in os.walk(root):
        for n in files:
            if n in names and n not in found:
                found[n] = os.path.join(dirpath, n)
    return found


def input_file(name):
    """A file that the user attached as a Kaggle dataset (a way to give a weights file that cannot be downloaded)."""
    for base in [b for b in (os.environ.get("MICA_EXTRA_INPUT"), "/kaggle/input") if b and os.path.isdir(b)]:
        hit = find_files(base, {name}).get(name)
        if hit:
            return hit
    return None


def place_files(archive, names, dest_dir, key):
    """Unpacks `archive` and copies the files `names` (found anywhere in it) into dest_dir."""
    tmp = os.path.join(BIG, "unpack_" + key)
    shutil.rmtree(tmp, ignore_errors=True)
    unpack(archive, tmp)
    found = find_files(tmp, set(names))
    missing = [n for n in names if n not in found]
    if missing:
        listing = sorted(os.path.relpath(os.path.join(d, f), tmp) for d, _s, fs in os.walk(tmp) for f in fs)
        raise RuntimeError("the archive %s lacks %s; it contains %s" % (os.path.basename(archive), ", ".join(missing), listing[:12]))
    os.makedirs(dest_dir, exist_ok=True)
    for n in names:
        shutil.copyfile(found[n], os.path.join(dest_dir, n))
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        os.remove(archive)                                       # the weights are in place; the archive would only use disk space
    except OSError:
        pass


def record_files(key, folders, label="weights"):
    """Logs and remembers the big files (1 MB or more) under `folders`; returns their total size in MB."""
    files = {}
    for folder in folders:
        for dirpath, _dirs, names in os.walk(folder):
            for n in names:
                p = os.path.join(dirpath, n)
                if os.path.getsize(p) >= 1000000:
                    files[os.path.relpath(p, BIG)] = os.path.getsize(p)
    STATE["methods"][key][label] = files
    mb = sum(files.values()) // 1000000
    log("%s files of %s: %d files, %d MB %s" % (label, key, len(files), mb, {k: v // 1000000 for k, v in list(files.items())[:8]}))
    return mb


def chmod_programs(folder, exclude_suffix=(".py", ".md", ".txt", ".json", ".cif", ".pdb", ".so", ".h", ".c", ".cpp", ".pth", ".pt", ".ckpt")):
    """Makes the programs that a repository ships (git may have kept them without the executable bit) executable."""
    n = 0
    for dirpath, _dirs, names in os.walk(folder):
        for f in names:
            p = os.path.join(dirpath, f)
            if not f.lower().endswith(exclude_suffix) and os.path.isfile(p) and not os.access(p, os.X_OK):
                with open(p, "rb") as fh:
                    if fh.read(4) == b"\x7fELF":
                        os.chmod(p, 0o755)
                        n += 1
    return n


def run_program(label, cmd, cwd, env, logname, watch=(), timeout=7200):
    """Runs a method's program with a time limit, a progress line every two minutes and a GPU/RAM log; returns (exit code, last lines, timed_out, info)."""
    logf = os.path.join(WORK, "logs", logname)
    mon = Monitor(os.path.join(WORK, "gpu", logname.replace(".log", ".csv")))
    mon.start()
    t0 = time.time()
    rc, last, timed_out = run_watched(cmd, label, watch=[logf] + list(watch), env=env, cwd=cwd, log_file=logf, timeout=timeout, every=1)
    info = mon.stop()
    info.update(rc=rc, seconds=round(time.time() - t0), timed_out=timed_out)
    return rc, last, timed_out, info


INPUTS = {}


def prepare_inputs(entry):
    """Map (unzipped), deposited model (for scoring) and the FASTA file of the entry; the same for every method."""
    if entry in INPUTS:
        return INPUTS[entry]
    row = {r["pdb_id"]: r for r in read_csv("data/entries_2026.csv")}.get(entry)
    if row is None:
        raise RuntimeError("%s is not in data/entries_2026.csv" % entry)
    folder = os.path.join(INPUT_ROOT, entry)
    os.makedirs(folder, exist_ok=True)
    number = row["emdb_id"].split("-")[-1]
    map_path = os.path.join(folder, "emd_%s.mrc" % number)
    if not os.path.exists(map_path):
        gz = download(row["map_url"], os.path.join(BIG, "downloads", os.path.basename(row["map_url"])), 100000)
        try:
            with gzip.open(gz, "rb") as src, open(map_path, "wb") as out:
                shutil.copyfileobj(src, out)
        except OSError:
            shutil.copyfile(gz, map_path)                           # not gzipped after all
    native = download(row["cif_url"], os.path.join(WORK, "native", entry + ".cif"), 1000)
    jobs = mica_inputs.read_jobs("data/af3/af3_jobs.csv", entry)
    fasta = os.path.join(folder, entry.lower() + ".fasta")
    with open(fasta, "w", encoding="utf-8") as f:
        f.write(mica_inputs.mica_fasta(jobs))
    info = {"entry": entry, "row": row, "folder": folder, "map": map_path, "map_sha256": sha256_file(map_path), "native": native, "jobs": jobs, "fasta": fasta}
    INPUTS[entry] = info
    log("%s: map %s (%d MB, sha256 %s), %d sequence(s): %s" % (entry, os.path.basename(map_path), os.path.getsize(map_path) // 1000000, info["map_sha256"][:16], len(jobs),
                                                              [{"job": r["job_name"], "length": int(r["length"]), "copies": int(r["copies_in_entry"])} for r in jobs]))
    return info


def af3_inputs(entry):
    """The AlphaFold Server prediction of every sequence of the entry (hashes checked against the list of the repository), in the order of the FASTA records."""
    info = prepare_inputs(entry)
    if "af3" not in info:
        jobs = info["jobs"]
        models = find_af3_dir(jobs)
        manifest = {}
        for line in open("data/af3/af3_files_sha256.txt", encoding="utf-8"):
            if line.strip():
                h, name = line.split()
                manifest[name] = h
        for r in jobs:
            name = "fold_%s_model_0.cif" % r["job_name"].lower()
            got = sha256_file(mica_inputs.find_model(models, r["job_name"]))
            if got != manifest.get(name):
                raise RuntimeError("the file %s is not the one that was checked (sha256 %s, expected %s)" % (name, got[:16], str(manifest.get(name))[:16]))
        built = mica_inputs.build("data/af3/af3_jobs.csv", models, entry, INPUT_ROOT)
        info["af3"] = [os.path.join(built["folder"], f["target"]) for f in built["files"]]
        log("AlphaFold Server files of %s: found in %s, hashes match the checked list" % (entry, models))
    return info["af3"]


METRIC_COLUMNS = ["method", "entry", "model_kind", "model_file", "status", "note", "tm_score", "tm_score_pred_norm", "ca_rmsd", "aligned_ca_length", "seq_identity",
                  "pred_residues", "native_residues", "coverage"]
ROWS = []


def score_model(method, entry, row, native, kind, path):
    ev = evaluate_models.evaluate(STATE["usalign"], path, native, int(row["modeled_residues"]))
    os.makedirs(os.path.join(WORK, "scoring"), exist_ok=True)
    full = subprocess.run([STATE["usalign"], path, native, "-mol", "protein"], capture_output=True, text=True, errors="replace")
    with open(os.path.join(WORK, "scoring", "%s_%s_usalign.txt" % (entry, kind)), "w", encoding="utf-8") as f:
        f.write("# USalign %s %s -mol protein\n%s%s" % (path, native, full.stdout, full.stderr))
    return dict(ev, method=method, entry=entry, model_kind=kind, model_file=os.path.basename(path))


def keep_results(key, entry, variant, out_dir, produced):
    """Copies the models and the small text files of a run into the work folder (so they go into the zip); the intermediate files stay out."""
    for kind, path in produced.items():
        if kind in PRIVATE_KINDS:
            continue
        dst = os.path.join(WORK, "models", key, entry)
        os.makedirs(dst, exist_ok=True)
        shutil.copyfile(path, os.path.join(dst, os.path.basename(path)))
    dst = os.path.join(WORK, "method_files", key, entry, variant)
    for dirpath, _dirs, names in os.walk(out_dir):
        if os.path.relpath(dirpath, out_dir).count(os.sep) > 1:
            continue
        for n in names:
            p = os.path.join(dirpath, n)
            if n.endswith((".log", ".txt", ".csv", ".json", ".xlsx")) and os.path.getsize(p) <= 2000000:
                target = os.path.join(dst, os.path.relpath(p, out_dir))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(p, target)


def setup_method(spec):
    """Source at the pinned commit, the Python environment, a GPU test, then the method's own setup (weights); returns (python, source folder)."""
    key = spec["key"]
    repo = os.path.join(BIG, "src_" + key)
    log("source: %s at commit %s" % (spec["repo"], spec["commit"]))
    if not git_checkout(spec["repo"], spec["commit"], repo):
        raise RuntimeError("the source could not be fetched at commit %s (see the lines above)" % spec["commit"])
    env_py = make_env(spec, repo)
    if REQUIRE_GPU and not ENV_PY_GIVEN:
        gpu_test(spec, env_py)
    SETUPS[key](spec, repo, env_py)
    return env_py, repo


def run_entry(spec, m, env_py, repo, entry):
    key = spec["key"]
    e = m["entries"][entry] = {"entry": entry, "status": "failed", "failed_step": "", "note": "", "runs": {}, "models": []}
    t0 = time.time()
    log("\n" + "-" * 72 + "\n%s on entry %s\n" % (spec["label"], entry) + "-" * 72)
    try:
        ctx = prepare_inputs(entry)
        runs = PLANS[key](spec, ctx, repo, env_py)
    except Exception as ex:
        e.update(failed_step="inputs", note=str(ex)[:600])
        log("%s on %s STOPPED while preparing the inputs: %s" % (spec["label"], entry, ex))
        return
    ok = True
    for r in runs:
        log("\n=== %s, %s: %s" % (spec["label"], r["variant"], " ".join(str(c) for c in r["cmd"][1:])[:700]))
        shutil.rmtree(r["out_dir"], ignore_errors=True)
        os.makedirs(r["out_dir"], exist_ok=True)
        rc, last, timed_out, info = run_program("%s %s" % (spec["label"], r["variant"]), r["cmd"], r["cwd"], r["env"], "%s_%s_%s.log" % (entry, key, r["variant"]),
                                                watch=r.get("watch", ()), timeout=spec["timeout"])
        e["runs"][r["variant"]] = info
        log("%s %s finished: exit code %s, %d s, GPU memory peak %s MB, lowest free RAM %s GB" % (spec["label"], r["variant"], rc, info["seconds"], info["peak_gpu_mb"], info["min_ram_available_gb"]))
        produced = {k: p for k, p in r["outputs"].items() if os.path.exists(p) and os.path.getsize(p) > 500}
        keep_results(key, entry, r["variant"], r["out_dir"], produced)
        if rc != 0 or timed_out or r["main"] not in produced:
            ok = False
            e.update(failed_step=r["variant"], note=("timeout" if timed_out else "exit code %s" % rc) + ("" if r["main"] in produced else "; the model file %s was not written" % os.path.basename(r["outputs"][r["main"]])))
            log("%s %s on %s STOPPED: %s. The last lines of the program:\n%s" % (spec["label"], r["variant"], entry, e["note"], "\n".join(last[-25:])))
        log("%s wrote: %s" % (spec["label"], ", ".join("%s (%d bytes)" % (os.path.relpath(os.path.join(d, n), r["out_dir"]), os.path.getsize(os.path.join(d, n)))
                                                          for d, _s, ns in os.walk(r["out_dir"]) for n in sorted(ns))[:900]))
        for kind, path in produced.items():
            try:
                row = score_model(key, entry, ctx["row"], ctx["native"], kind, path)
                ROWS.append(row)
                e["models"].append(row)
            except Exception as ex:
                ok = False
                e.update(failed_step="scoring", note=("scoring of %s: %s" % (kind, ex))[:500])
                log("scoring of %s stopped: %s" % (kind, ex))
        pack_now()
    e["seconds"] = round(time.time() - t0)
    if ok:
        e["status"] = "ok"


def execute_method(key):
    spec = SPECS[key]
    m = STATE["methods"][key] = {"key": key, "label": spec["label"], "commit": spec["commit"], "status": "failed", "failed_step": "", "note": "", "setup_seconds": None, "entries": {}}
    if key not in RUN_METHODS:
        m.update(status="not run", note="left out of RUN_METHODS")
        log("\n%s: left out of RUN_METHODS" % spec["label"])
        return
    log("\n" + "#" * 78 + "\n# %s  (commit %s)\n" % (spec["label"], spec["commit"][:10]) + "#" * 78)
    t0 = time.time()
    try:
        env_py, repo = setup_method(spec)
    except Exception as ex:
        import traceback
        m.update(failed_step="setup", note=str(ex)[:800], setup_seconds=round(time.time() - t0))
        log("\n%s STOPPED in its setup: %s\n%s" % (spec["label"], ex, traceback.format_exc()[-1200:]))
        pack_now()
        return
    m["setup_seconds"] = round(time.time() - t0)
    log("%s is ready (%d s)" % (spec["label"], m["setup_seconds"]))
    pack_now()
    for entry in ENTRIES:
        try:
            run_entry(spec, m, env_py, repo, entry)
        except Exception:
            import traceback
            m["entries"].setdefault(entry, {"entry": entry, "runs": {}, "models": []}).update(status="failed", failed_step="unexpected error", note=traceback.format_exc()[-600:])
            log("\n%s on %s STOPPED by an unexpected error:\n%s" % (spec["label"], entry, traceback.format_exc()))
    bad = [e for e in m["entries"].values() if e["status"] != "ok"]
    m["status"] = "ok" if m["entries"] and not bad else "failed"
    if bad:
        m.update(failed_step=bad[0]["failed_step"], note=bad[0]["note"])
    pack_now()


SETUPS, PLANS = {}, {}
log("tools ready; the methods of this run: %s; entries: %s" % (", ".join(RUN_METHODS), ", ".join(ENTRIES)))
'''

STEP4_MD = """\
## Step 4. US-align

US-align is compiled from its pinned commit (the program that scores every model)."""
STEP4 = r'''
if STATE["ok"]:
    build_usalign()
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

E3_MD = """\
## Step 5. E3-CryoFold

`inference.py --map_path <map> --fasta_path <fasta> --save_dir <folder>/ --save_name <name> --protocol pre_align --t 0.1`, the first example of the README (map and sequence; the
`pre_align` protocol is the default of the program), and the same with `--protocol denovo` (the second example). The weights are the `models.zip` of the repository's `checkpoint` release (1.5 GB); the
file holds the Chroma weights too, so the only other download is the small ESM-2 (35M) model that the `denovo` protocol loads through `fair-esm`."""
E3 = r'''
def setup_e3(spec, repo, env_py):
    models = os.path.join(repo, "models")
    need = ["model_weight.pth", "structure_model.pth"]
    if not all(os.path.exists(os.path.join(models, n)) and os.path.getsize(os.path.join(models, n)) > 1000000 for n in need):
        archive = os.path.join(BIG, "downloads", "e3cryofold_models.zip")
        if not fetch(spec["weights_url"], archive, spec["weights_min_bytes"], "E3-CryoFold weights"):
            raise RuntimeError("the weights could not be downloaded from %s" % spec["weights_url"])
        place_files(archive, need, models, "e3")
    record_files("e3", [models])


def plan_e3(spec, ctx, repo, env_py):
    entry, runs = ctx["entry"], []
    for variant, protocol in (("prealign", "pre_align"), ("denovo", "denovo")):
        out = os.path.join(RUN_ROOT, "e3", entry, variant) + os.sep           # the program writes save_dir + save_name + ".pdb"
        name = "%s_e3_%s" % (entry, variant)
        cmd = [env_py, "inference.py", "--map_path", ctx["map"], "--fasta_path", ctx["fasta"], "--save_dir", out, "--save_name", name, "--protocol", protocol, "--t", "0.1"]
        runs.append({"variant": variant, "cmd": cmd, "cwd": repo, "env": env_vars(env_py), "out_dir": out, "outputs": {"e3_" + variant: out + name + ".pdb"}, "main": "e3_" + variant})
    return runs


SETUPS["e3"], PLANS["e3"] = setup_e3, plan_e3
if STATE["ok"] and STATE["usalign"]:
    execute_method("e3")
else:
    log("skipped: an earlier step stopped")
'''

EMPROT_MD = """\
## Step 6. EMProt

`emprot build --map <map> --seq <fasta> --chain <one predicted structure per chain> --device 0 --output <folder>`, the third example of the README (map, sequences and predicted models).
The predicted models are the AlphaFold Server predictions that MICA gets; the FASTA file has one record per chain (a sequence that occurs twice in the entry is written twice, with the
same prediction twice). The weights are `weights.tgz` of the authors' server, unpacked in the `emprot` folder as the README says. EMProt's own `cctbx` is the PyPI package."""
EMPROT = r'''
def emprot_inputs(entry, jobs, af3_files):
    """One FASTA record and one predicted chain per chain of the entry, in the order of the entities."""
    records, chains = [], []
    for job, path in zip(jobs, af3_files):
        for copy in range(int(job["copies_in_entry"])):
            records.append(">%s_%s_%d\n%s\n" % (entry, job["entity"], copy + 1, job["sequence_submitted"].strip()))
            chains.append(path)
    return "".join(records), chains


def setup_emprot(spec, repo, env_py):
    chmod_programs(os.path.join(repo, "emprot", "bin"))
    weights_dir = os.path.join(repo, "emprot")
    before = {os.path.join(d, f) for d, _s, fs in os.walk(weights_dir) for f in fs if os.path.getsize(os.path.join(d, f)) >= 1000000}
    marker = os.path.join(BIG, "downloads", "emprot_weights_unpacked.txt")
    if not os.path.exists(marker):
        archive = os.path.join(BIG, "downloads", "emprot_weights.tgz")
        if not fetch(spec["weights_url"], archive, spec["weights_min_bytes"], "EMProt weights"):
            raise RuntimeError("the weights could not be downloaded from %s" % spec["weights_url"])
        unpack(archive, weights_dir)
        after = {os.path.join(d, f) for d, _s, fs in os.walk(weights_dir) for f in fs if os.path.getsize(os.path.join(d, f)) >= 1000000}
        if not (after - before):
            raise RuntimeError("weights.tgz did not add any big file to %s" % weights_dir)
        if not os.path.isdir(os.path.join(weights_dir, "weights")):
            raise RuntimeError("weights.tgz did not create the folder emprot/weights that the program reads (it made %s)" % sorted(os.listdir(weights_dir))[:12])
        with open(marker, "w") as f:
            f.write("ok\n")
        os.remove(archive)
    chmod_programs(os.path.join(repo, "emprot", "bin"))
    record_files("emprot", [os.path.join(repo, "emprot")])


def plan_emprot(spec, ctx, repo, env_py):
    entry = ctx["entry"]
    af3 = af3_inputs(entry)
    out = os.path.join(RUN_ROOT, "emprot", entry)
    os.makedirs(out + "_in", exist_ok=True)
    fasta, chains = emprot_inputs(entry, ctx["jobs"], af3)
    fasta_path = os.path.join(out + "_in", entry.lower() + "_emprot.fasta")
    with open(fasta_path, "w", encoding="utf-8") as f:
        f.write(fasta)
    cmd = [env_py, "-m", "emprot", "build", "--map", ctx["map"], "--seq", fasta_path, "--chain"] + chains + ["--device", "0", "--output", out]
    outputs = {"emprot": os.path.join(out, "output.cif"), "emprot_denovo": os.path.join(out, "output_denovo.cif"), "emprot_fit": os.path.join(out, "output_fit.cif")}
    return [{"variant": "build", "cmd": cmd, "cwd": repo, "env": env_vars(env_py), "out_dir": out, "outputs": outputs, "main": "emprot"}]


SETUPS["emprot"], PLANS["emprot"] = setup_emprot, plan_emprot
if STATE["ok"] and STATE["usalign"]:
    execute_method("emprot")
else:
    log("skipped: an earlier step stopped")
'''

CRYOATOM_MD = """\
## Step 7. CryoAtom (two versions)

Run from the source folder with `python -m <package> build ...` (the same program as the `cryoatom` command that `setup.py` installs). **v1.0.0** is the code of the Nature Structural &
Molecular Biology paper (`-s <fasta> -v <map> -o <folder>`, the README's example); **CryoAtom2** is the current repository (`-v <map> -ps <fasta> -o <folder> -d 0`, the README's example). The weights
are the zip files of the authors' server (certificate check off, as in the authors' `wget --no-check-certificate`); CryoAtom2 also needs the RNA-FM weights (placed where `install.sh` puts them);
the ESM-1b weights are downloaded by the program at its first run. The model is `<folder name>.cif`; `<folder name>_raw.cif` is the model before pruning. Both are scored."""
CRYOATOM = r'''
def setup_cryoatom(spec, repo, env_py):
    ckpt = os.path.join(repo, spec["pkg"], "checkpoint")
    need = spec["ckpt_files"]
    if not all(os.path.exists(os.path.join(ckpt, n)) and os.path.getsize(os.path.join(ckpt, n)) > 1000000 for n in need):
        archive = os.path.join(BIG, "downloads", os.path.basename(spec["weights_url"]))
        if not fetch(spec["weights_url"], archive, spec["weights_min_bytes"], "%s weights" % spec["label"], insecure=True):
            raise RuntimeError("the weights could not be downloaded from %s" % spec["weights_url"])
        place_files(archive, need, ckpt, spec["key"])
    n = chmod_programs(os.path.join(repo, spec["pkg"], "utils"))
    log("made %d program(s) of %s executable" % (n, spec["pkg"]))
    if spec.get("rnafm_url"):
        rna = os.path.join(TORCH_HOME, "hub", "checkpoints", "RNA-FM_pretrained.pth")
        if not fetch(spec["rnafm_url"], rna, spec["rnafm_bytes"], "RNA-FM weights", insecure=True, expect_size=spec["rnafm_bytes"]):
            log("WARNING: the RNA-FM weights could not be downloaded; a protein-only run may not need them, the run will show")
    record_files(spec["key"], [ckpt])


def plan_cryoatom(spec, ctx, repo, env_py):
    entry = ctx["entry"]
    name = "%s_%s" % (entry, spec["key"])                                     # the model is named after the output folder
    out = os.path.join(RUN_ROOT, spec["key"], name)
    args = [a.format(map=ctx["map"], fasta=ctx["fasta"], out=out) for a in spec["cli"]]
    cmd = [env_py, "-m", spec["pkg"], "build"] + args
    outputs = {spec["key"]: os.path.join(out, name + ".cif"), spec["key"] + "_raw": os.path.join(out, name + "_raw.cif")}
    return [{"variant": "build", "cmd": cmd, "cwd": repo, "env": env_vars(env_py), "out_dir": out, "outputs": outputs, "main": spec["key"], "watch": [os.path.join(out, "running_time.log")]}]


for _k in ("cryoatom2", "cryoatom1"):
    SETUPS[_k], PLANS[_k] = setup_cryoatom, plan_cryoatom
if STATE["ok"] and STATE["usalign"]:
    execute_method("cryoatom2")
    execute_method("cryoatom1")
else:
    log("skipped: an earlier step stopped")
'''

EMODELX_MD = """\
## Step 8. EModelX(+AF)

`run.py --protocol=temp_flex --EM_map <map> --fasta <fasta> --template_dir <folder> --output_dir <folder> --run_pulchra --pulchra_path modules/pulchra304/src/pulchra`, the README's EModelX(+AF)
example, with the AlphaFold Server predictions as templates: `<template_dir>/<FASTA name>/ranked_0.pdb` for every FASTA record (the prediction converted from mmCIF to PDB, chain A) and no
download from AlphaFold DB. The three weights files come from the authors' Google Drive link (`gdown`); if Drive refuses (quota), attach the files `best_BB_model.ckpt`, `best_CA_model.ckpt` and
`best_AA_model.ckpt` as a Kaggle dataset and run again. The models are `<fasta name>_EModelX(+AF)_ca_model.pdb` (C-alpha trace) and `_all_atom_model.pdb` (after Pulchra); both are scored."""
EMODELX = r'''
TEMPLATE_CONVERT = r"""
import sys
from Bio.PDB import MMCIFParser, PDBIO
src, dst = sys.argv[1], sys.argv[2]
structure = MMCIFParser(QUIET=True).get_structure("af3", src)
chains = list(structure[0])
assert len(chains) == 1, "%d chains in %s" % (len(chains), src)
chains[0].id = "A"
io = PDBIO()
io.set_structure(structure)
io.save(dst)
print("wrote", dst, sum(1 for r in chains[0] if "CA" in r), "C-alpha atoms")
"""


def setup_emodelx(spec, repo, env_py):
    models = os.path.join(repo, "models")
    need = spec["ckpt_files"]
    if not all(os.path.exists(os.path.join(models, n)) and os.path.getsize(os.path.join(models, n)) > 1000000 for n in need):
        given = {n: input_file(n) for n in need}
        os.makedirs(models, exist_ok=True)
        if all(given.values()):
            log("the weights files were attached as a Kaggle dataset: %s" % given)
            for n, p in given.items():
                shutil.copyfile(p, os.path.join(models, n))
        else:
            archive = os.path.join(BIG, "downloads", "emodelx_models_download")
            if spec["weights_url"]:
                ok = fetch(spec["weights_url"], archive, spec["weights_min_bytes"], "EModelX weights")
            else:
                ok = fetch_gdrive(spec["gdrive_id"], archive, spec["weights_min_bytes"], "EModelX weights (Google Drive)")
            if not ok:
                raise RuntimeError("the weights could not be downloaded (Google Drive may have refused: too many downloads). Attach best_BB_model.ckpt, best_CA_model.ckpt and "
                                   "best_AA_model.ckpt as a Kaggle dataset (Add Input) and run again")
            place_files(archive, need, models, "emodelx")
    pulchra = os.path.join(repo, "modules", "pulchra304", "src", "pulchra")
    if not os.path.exists(pulchra):
        raise RuntimeError("Pulchra is missing at %s" % pulchra)
    os.chmod(pulchra, 0o755)
    r = subprocess.run([pulchra], capture_output=True, text=True, errors="replace", timeout=60)
    log("pulchra starts (exit code %s): %s" % (r.returncode, (r.stdout + r.stderr).strip().splitlines()[:1]))
    record_files("emodelx", [models])


def plan_emodelx(spec, ctx, repo, env_py):
    entry = ctx["entry"]
    af3 = af3_inputs(entry)
    base = os.path.join(RUN_ROOT, "emodelx", entry)
    out = os.path.join(base, "out")
    tdir = os.path.join(base, "templates")
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(out)
    fasta = os.path.join(base, entry.lower() + ".fasta")                      # the program names its files after the FASTA file
    shutil.copyfile(ctx["fasta"], fasta)
    names = [ln[1:].split("|")[0] for ln in open(fasta, encoding="utf-8") if ln.startswith(">")]
    if len(names) != len(af3):
        raise RuntimeError("%d FASTA records but %d predictions" % (len(names), len(af3)))
    for name, src in zip(names, af3):
        dst = os.path.join(tdir, name, "ranked_0.pdb")
        os.makedirs(os.path.dirname(dst))
        rc, last, _ = run([env_py, "-c", TEMPLATE_CONVERT, src, dst], env=env_vars(env_py), quiet=True)
        if rc != 0 or not os.path.exists(dst):
            raise RuntimeError("the prediction %s could not be converted to PDB: %s" % (os.path.basename(src), " ".join(last[-3:])))
        log("template %s: %s" % (name, last[-1]))
    cmd = [env_py, "run.py", "--protocol=temp_flex", "--EM_map", ctx["map"], "--fasta", fasta, "--template_dir", tdir, "--output_dir", out, "--run_pulchra",
           "--pulchra_path", "modules/pulchra304/src/pulchra"]
    low = entry.lower()
    outputs = {"emodelx_af_all_atom": os.path.join(out, "%s_EModelX(+AF)_all_atom_model.pdb" % low), "emodelx_af_ca": os.path.join(out, "%s_EModelX(+AF)_ca_model.pdb" % low)}
    return [{"variant": "temp_flex", "cmd": cmd, "cwd": repo, "env": env_vars(env_py), "out_dir": out, "outputs": outputs, "main": "emodelx_af_all_atom"}]


SETUPS["emodelx"], PLANS["emodelx"] = setup_emodelx, plan_emodelx
if STATE["ok"] and STATE["usalign"]:
    execute_method("emodelx")
else:
    log("skipped: an earlier step stopped")
'''

SUMMARY_MD = """\
## Step 9. Summary

One table of every model that was scored (`pilot_metrics.csv`), one line per method and entry (`method_status.csv`), and the versions that were run."""
SUMMARY = r'''
def summary_table():
    log("\n%-10s %-5s %-20s %-8s %7s %8s %7s %9s %8s" % ("method", "entry", "model", "status", "TM", "aligned", "RMSD", "coverage", "seq id"))
    for r in ROWS:
        log("%-10s %-5s %-20s %-8s %7s %8s %7s %9s %8s" % (r["method"], r["entry"], r["model_kind"], r["status"], r["tm_score"], r["aligned_ca_length"], r["ca_rmsd"], r["coverage"], r["seq_identity"]))
    log("")
    for key in ORDER_ALL:
        m = STATE["methods"].get(key)
        if not m:
            continue
        log("%-36s %-8s setup %s s%s" % (m["label"], m["status"].upper(), m["setup_seconds"], ("; stopped in %s: %s" % (m["failed_step"], m["note"][:160])) if m["status"] == "failed" else ""))
        for e in m["entries"].values():
            log("    %-5s %-8s runs (s): %s%s" % (e["entry"], e["status"], {k: v["seconds"] for k, v in e["runs"].items()},
                                                 ("; stopped in %s: %s" % (e["failed_step"], e["note"][:120])) if e["status"] != "ok" else ""))


ORDER_ALL = @ORDER_ALL@
if STATE["ok"]:
    write_csv("pilot_metrics.csv", METRIC_COLUMNS, ROWS)
    write_csv("method_status.csv", ["method", "entry", "status", "failed_step", "note", "seconds"],
              [dict(method=k, entry=e["entry"], status=e["status"], failed_step=e["failed_step"], note=e["note"], seconds=e.get("seconds", "")) for k, m in STATE["methods"].items() for e in m["entries"].values()] +
              [dict(method=k, entry="(setup)", status="failed", failed_step=m["failed_step"], note=m["note"], seconds=m["setup_seconds"]) for k, m in STATE["methods"].items() if m["failed_step"] == "setup"])
    summary_table()
    ran = [m for m in STATE["methods"].values() if m["status"] != "not run"]
    STATE["complete"] = bool(ran) and all(m["status"] == "ok" for m in ran)
    if not STATE["complete"]:
        STATE["stopped_at"] = ", ".join("%s: %s" % (m["label"], m["failed_step"] or "?") for m in ran if m["status"] != "ok")
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

EXTRA_INFO = ('{"methods": {k: {"label": m["label"], "commit": m["commit"], "status": m["status"], "setup_seconds": m["setup_seconds"]} for k, m in STATE["methods"].items()}, '
              '"specs": globals().get("SPECS", {}), "usalign_commit": globals().get("USALIGN_COMMIT", ""), "usalign_version": STATE["usalign_version"], "input_sha256": globals().get("EXPECTED", {})}')


def build(out_path, entries=("26DK",), usalign_repo=P.USALIGN_REPO, usalign_commit=P.USALIGN_COMMIT, entries_path=None, env_python=None, usalign_bin=None, net_hosts=NET_HOSTS,
          require_gpu=True, import_tests=None, specs=None, run_methods=None):
    """Writes the notebook. The keyword arguments other than `entries` exist for the tests (local repositories and servers, a ready environment, no GPU)."""
    specs = copy.deepcopy(specs or SPECS)
    run_methods = list(run_methods or ORDER)
    file_cells, expected = nbkit.file_cells(FILES, entries_path)
    step5_for_functions = nbkit.fill(P.STEP5, {"@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit, "@MICA_REPO@": "x", "@MICA_COMMIT@": "x",
                                               "@WEIGHTS_URL@": "x", "@ZENODO_API@": "x", "@USALIGN_BIN@": "None"})
    git_and_usalign = nbkit.function_sources(step5_for_functions, ["git_checkout", "build_usalign"]).rstrip("\n")
    step6_for_functions = nbkit.fill(P.STEP6, {"@ENTRIES@": "[]"})
    read_write_download = nbkit.function_sources(step6_for_functions, ["read_csv", "write_csv", "download"]).rstrip("\n")
    find_af3 = nbkit.function_sources(step6_for_functions, ["find_af3_dir"]).rstrip("\n")
    mapping = {"@UV_VERSION@": P.UV_VERSION, "@ENV_PYTHON@": repr(env_python), "@IMPORT_TESTS@": repr(import_tests), "@SPECS@": repr(specs), "@RUN_METHODS@": repr(run_methods),
               "@ENTRIES@": repr(list(entries)), "@ENTRY_LIST@": ", ".join(entries), "@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit,
               "@USALIGN_BIN@": repr(usalign_bin), "@REQUIRE_GPU@": repr(bool(require_gpu)), "@NET_HOSTS@": repr(tuple(net_hosts)), "@EXPECTED@": repr(expected),
               "@NOTEBOOK@": NOTEBOOK, "@EXTRA_INFO@": EXTRA_INFO, "@ORDER_ALL@": repr(list(ORDER))}
    fl = lambda text: nbkit.fill(text, mapping).strip("\n") + "\n"
    tools = TOOLS.replace("@READ_WRITE_DOWNLOAD@", read_write_download).replace("@GIT_AND_USALIGN@", git_and_usalign).replace("@FIND_AF3@", find_af3)
    cells = [("intro", nbkit.md(nbkit.fill(INTRO, mapping))),
             ("step1-title", nbkit.md(P.STEP1_MD)), ("step1", nbkit.code(fl(P.STEP1))),
             ("step2-title", nbkit.md(P.STEP2_MD)), ("step2-dirs", nbkit.code(P.STEP2_DIRS.strip("\n") + "\n"))]
    cells += file_cells
    cells += [("step2-check", nbkit.code(fl(P.STEP2_CHECK))),
              ("robust-title", nbkit.md(nbkit.fill(nbkit.ROBUST_MD, mapping))), ("robust", nbkit.code(fl(nbkit.ROBUST))),
              ("tools-title", nbkit.md(TOOLS_MD)), ("tools", nbkit.code(fl(tools))),
              ("step4-title", nbkit.md(STEP4_MD)), ("step4", nbkit.code(fl(STEP4))),
              ("e3-title", nbkit.md(E3_MD)), ("e3", nbkit.code(fl(E3))),
              ("emprot-title", nbkit.md(EMPROT_MD)), ("emprot", nbkit.code(fl(EMPROT))),
              ("cryoatom-title", nbkit.md(CRYOATOM_MD)), ("cryoatom", nbkit.code(fl(CRYOATOM))),
              ("emodelx-title", nbkit.md(EMODELX_MD)), ("emodelx", nbkit.code(fl(EMODELX))),
              ("summary-title", nbkit.md(SUMMARY_MD)), ("summary", nbkit.code(fl(SUMMARY))),
              ("final-title", nbkit.md(nbkit.FINAL_MD)), ("final", nbkit.code(fl(nbkit.FINAL)))]
    nbkit.assemble(out_path, NB_NAME, cells)
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, NB_NAME))
    p.add_argument("--entries", nargs="+", default=["26DK"])
    args = p.parse_args(argv)
    expected = build(args.out, entries=args.entries)
    print("wrote %s with %d embedded files" % (args.out, len(expected)))
    for path, digest in expected.items():
        print("  %s  %s" % (digest[:16], path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
