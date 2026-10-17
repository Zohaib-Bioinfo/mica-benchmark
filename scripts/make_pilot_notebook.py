#!/usr/bin/env python3
"""Build scripts/pilot_mica.ipynb: the Phase 4 pilot of MICA on one entry of the frozen list (default 26DK, the smallest).

One notebook for Kaggle (first choice) or Colab (backup). It checks the machine, installs Phenix from the user's own download link
(user name and password come from Kaggle or Colab Secrets, or are typed in; they are never written to a file), builds MICA's Python 3.8
environment from PyPI wheels with exact pins, fetches MICA at its pinned commit and its weights from Zenodo, prepares the input folder
of the entry (AlphaFold Server predictions of the user, map from EMDB), runs MICA's three programs exactly as MICA_pipeline.sh does,
scores every model it finds with US-align (pinned commit) and Phenix chain_comparison, and packs one zip with the logs and numbers.

    python make_pilot_notebook.py                      # writes scripts/pilot_mica.ipynb
    python make_pilot_notebook.py --entries 26DK 9OC0  # the same notebook for other entries

The cell sources live in this file; the .ipynb is a build product. The pilot exists to find, on one small entry, every command and
version that the full runs need (docs/SPEC.md, section 4): the notebook records them and the raw output of every tool.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_select_notebook import PHENIX_URL  # noqa: E402  (one place for the Phenix link)

MICA_REPO = "https://github.com/jianlin-cheng/MICA"
MICA_COMMIT = "726d1bb14be04887a43b7f5aefcff0d843761d01"
WEIGHTS_URL = "https://zenodo.org/records/15756654/files/trained_models.tar.gz?download=1"
ZENODO_API = "https://zenodo.org/api/records/15756654"
USALIGN_REPO = "https://github.com/pylelab/USalign"
USALIGN_COMMIT = "746ad8a2a4dea9f317075cc482df0ecb5440bdea"
UV_VERSION = "0.13.1"
PY_VERSION = "3.8.20"
NET_HOSTS = (("PyPI", "https://pypi.org/simple/uv/", True),
             ("GitHub", "https://github.com/jianlin-cheng/MICA", True),
             ("Zenodo", "https://zenodo.org/api/records/15756654", True),
             ("EMDB API", "https://www.ebi.ac.uk/emdb/api/entry/80558", False),
             ("EBI FTP", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-80558/header/emd-80558.xml", False),
             ("RCSB", "https://files.rcsb.org/header/26DK.cif", False),
             ("Phenix", "https://phenix-online.org/download", False))
# (path in the repo, path in the notebook)
FILES = (("scripts/mica_inputs.py", "scripts/mica_inputs.py"),
         ("scripts/evaluate_models.py", "scripts/evaluate_models.py"),
         ("data/af3/af3_jobs.csv", "data/af3/af3_jobs.csv"),
         ("data/af3/af3_files_sha256.txt", "data/af3/af3_files_sha256.txt"),
         ("data/entries_2026.csv", "data/entries_2026.csv"),
         ("env/mica_requirements.txt", "env/mica_requirements.txt"))

INTRO = """\
# MICA benchmark, Phase 4 pilot: MICA on one small entry

This notebook runs MICA on entry **@ENTRY_LIST@** of the frozen list and scores the result. It is the pilot that fixes every command and
version for the full runs, so it saves the raw output of every tool.

1. checks the machine (GPU, disk, internet),
2. installs Phenix from your download link (user name and password from Secrets, or typed in; never saved in a file),
3. builds MICA's Python 3.8 environment from exact pins and tests the GPU,
4. fetches MICA (pinned commit) and its weights (Zenodo),
5. prepares the entry: your AlphaFold Server prediction, the EMDB map, the deposited model,
6. runs MICA's three programs (the same commands as `MICA_pipeline.sh`),
7. scores every model with US-align and Phenix `chain_comparison`, and packs one zip.

**What you do** (the chat message has the click-by-click steps for Kaggle and Colab)

- Before **Run all**: GPU on, Internet on, the two Phenix secrets (`PHENIX_USER`, `PHENIX_PASSWORD`) attached, and the AlphaFold Server zip
  added as an input (Kaggle) or ready to upload (Colab).
- Run all, keep the tab open. About 1 to 2 hours. Do not edit any code.
- At the end the zip `pilot_result_....zip` is in the output files (Colab downloads it by itself). Send it to me.

**If something goes wrong.** A step that cannot continue prints `STOPPED` and a plain explanation; later steps are skipped and the last cell
still packs a zip with the log. Send me that zip. If a cell shows a red error instead, copy all of its text and send it to me.

**Phenix** is licensed to you personally. Do not share the installer, the password, or this session. When you are done, stop the session.
"""

STEP1_MD = "## Step 1. Look at the machine"
STEP1 = r'''
import csv, datetime, getpass, glob, gzip, hashlib, json, os, platform, re, shutil, signal, subprocess, sys, tarfile, threading, time, zipfile
from collections import deque

IN_KAGGLE = os.path.isdir("/kaggle/working") and not os.environ.get("MICA_WORK")      # MICA_WORK and MICA_BIG are only used by the tests
IN_COLAB = (not IN_KAGGLE) and os.path.isdir("/content") and not os.environ.get("MICA_WORK")
PLATFORM = "kaggle" if IN_KAGGLE else ("colab" if IN_COLAB else "other")
if os.environ.get("MICA_WORK"):
    WORK = OUT_DIR = os.environ["MICA_WORK"]
elif IN_KAGGLE:
    WORK, OUT_DIR = "/kaggle/working/pilot", "/kaggle/working"
elif IN_COLAB:
    WORK, OUT_DIR = "/content/pilot", "/content"
else:
    WORK = OUT_DIR = os.path.abspath("pilot")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
LOG = os.path.join(WORK, "run.log")
REQUIRE_GPU = @REQUIRE_GPU@
NET_HOSTS = @NET_HOSTS@
STATE = {"ok": True, "problem": "", "t0": time.time(), "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "phenix_env": None, "phenix_version": "", "installer_sha256": "", "usalign": "", "usalign_version": "", "weights": {}, "env": {},
         "entries": {}, "gpu": ""}
SECRETS = []          # kept in memory only, to make sure they never end up in the zip


def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def fail(msg):
    STATE["ok"] = False
    STATE["problem"] = msg
    log("\n" + "!" * 72 + "\nSTOPPED: " + msg + "\nThe steps below are skipped. The last step still packs a zip with the log: send it to me.\n" + "!" * 72)


def run(cmd, env=None, every=1, cwd=None, log_file=None, timeout=None, quiet=False):
    """Runs a command, prints and logs its output as it comes. Returns (exit code, last lines, timed_out).
    env: values are set, None removes the variable. A command that runs longer than `timeout` seconds is killed with its child processes.
    Progress bars (lines with %|) are printed only now and then; quiet=True keeps the output out of the notebook (it still goes to log_file)."""
    e = dict(os.environ)
    for k, v in (env or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    e["PYTHONUNBUFFERED"] = "1"
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=e, cwd=cwd, errors="replace",
                             start_new_session=True)
    except OSError as ex:
        log("could not start %s: %s" % (cmd[0] if isinstance(cmd, (list, tuple)) else cmd, ex))
        return 127, [str(ex)], False
    timed = {"out": False}

    def killer():
        timed["out"] = True
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass

    timer = threading.Timer(timeout, killer) if timeout else None
    if timer:
        timer.start()
    lf = open(log_file, "a", encoding="utf-8") if log_file else None
    last, n, bars = deque(maxlen=40), 0, 0
    try:
        for line in p.stdout:
            line = line.rstrip("\n")
            if len(line) > 600:
                line = line[:600] + " ..."
            last.append(line)
            n += 1
            if lf:
                lf.write(line + "\n")
            if "%|" in line:
                bars += 1
                show = bars % 100 == 1
            else:
                show = every == 1 or n <= 5 or n % every == 0
            if show and not quiet:
                log(line)
        p.wait()
    finally:
        if timer:
            timer.cancel()
        if lf:
            lf.close()
        p.stdout.close()
    return p.returncode, list(last), timed["out"]


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, errors="replace")
        return (r.stdout + r.stderr).strip()
    except Exception as ex:
        return "ERROR: %s" % ex


def secret(name, prompt, hidden=True):
    """Colab Secrets, Kaggle Secrets, an environment variable, or a prompt, in this order. Returns "" if nothing can be found."""
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v.strip()
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret(name)
        if v:
            return v.strip()
    except Exception:
        pass
    if os.environ.get(name):
        return os.environ[name].strip()
    try:
        return (getpass.getpass(prompt) if hidden else input(prompt)).strip()
    except Exception:
        return ""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def meminfo_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1048576, 1)
    except OSError:
        pass
    return None


def pick_big_dir():
    """The folder for everything large (Phenix, Python environment, MICA): the disk with the most free space."""
    if os.environ.get("MICA_BIG"):
        return os.environ["MICA_BIG"]
    best = None
    for base in ("/kaggle/temp", "/tmp", "/content", WORK):
        if os.path.isdir(base) and os.access(base, os.W_OK):
            free = shutil.disk_usage(base).free
            if best is None or free > best[1] + 5e9:
                best = (base, free)
    return os.path.join(best[0], "mica_pilot")


BIG = pick_big_dir()
os.makedirs(BIG, exist_ok=True)


class Monitor(threading.Thread):
    """Samples GPU memory, GPU load and free RAM every few seconds into a csv file; summary() gives the peaks."""

    def __init__(self, path, every=5):
        super().__init__(daemon=True)
        self.path, self.every, self.stop_flag = path, every, threading.Event()
        self.peak_gpu_mb, self.peak_util, self.min_ram_gb, self.samples = 0, 0, None, 0

    def sample(self):
        gpu_mb, util = "", ""
        if shutil.which("nvidia-smi"):
            out = sh("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits", timeout=20).splitlines()
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

    def run(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("time_utc,gpu_mem_used_mb,gpu_util_pct,ram_available_gb\n")
        while not self.stop_flag.is_set():
            try:
                self.sample()
            except Exception:
                pass
            self.stop_flag.wait(self.every)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=30)
        return {"peak_gpu_mb": self.peak_gpu_mb, "peak_gpu_util_pct": self.peak_util, "min_ram_available_gb": self.min_ram_gb, "samples": self.samples}


def check_internet():
    import urllib.request
    bad_required = []
    for name, url, required in NET_HOSTS:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mica-benchmark-pilot"}), timeout=20) as r:
                r.read(512)
                log("  reachable: %-9s %s" % (name, r.status))
        except Exception as ex:
            log("  NOT reachable: %-9s %s" % (name, str(ex)[:100]))
            if required:
                bad_required.append(name)
    return bad_required


free_big, free_work = round(shutil.disk_usage(BIG).free / 1e9), round(shutil.disk_usage(WORK).free / 1e9)
cpu_model = sh("lscpu | grep -m1 'Model name'").split(":")[-1].strip()
log("platform %s | python %s | %s | %s CPU cores (%s) | %s GB RAM" % (PLATFORM, platform.python_version(), platform.platform(), os.cpu_count(), cpu_model, meminfo_gb()))
log("large files go to %s (%d GB free); results to %s (%d GB free)" % (BIG, free_big, WORK, free_work))
with open("machine.txt", "w", encoding="utf-8") as f:
    f.write(sh("nvidia-smi") + "\n\n" + sh("lscpu") + "\n\n" + sh("free -h") + "\n\n" + sh("df -h") + "\n\n" + sh("g++ --version | head -1; git --version; uname -a; cat /etc/os-release | head -3") + "\n")
STATE["gpu"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader") if shutil.which("nvidia-smi") else ""
if STATE["gpu"] and not STATE["gpu"].startswith("ERROR"):
    log("GPU: " + STATE["gpu"].replace("\n", " | "))
elif REQUIRE_GPU:
    fail("this session has no GPU. Kaggle: open Session options on the right, set Accelerator to GPU T4 x2 (or GPU P100), then Run all again. "
         "Colab: Runtime -> Change runtime type -> T4 GPU, then Run all again.")
if STATE["ok"]:
    log("checking the internet:")
    missing = check_internet() if NET_HOSTS else []
    if "PyPI" in missing or "GitHub" in missing:
        fail("the internet is off. Kaggle: open Session options on the right and switch Internet on (the account must be phone-verified), then Run all again.")
    elif missing:
        log("WARNING: %s cannot be reached from here; the step that needs it will tell you." % ", ".join(missing))
'''

STEP2_MD = """\
## Step 2. Write the scripts and the frozen data files

The files below are exact copies of the files in the repository (scripts, the AlphaFold Server job table and the hashes of the 26 predictions,
the entry list, the exact list of Python packages). The last cell of this step checks their hashes."""

STEP2_DIRS = '''
import os
for d in ("scripts", "data/af3", "env", "logs", "gpu"):
    os.makedirs(d, exist_ok=True)
print("folders ready")
'''

STEP2_CHECK = r'''
EXPECTED = @EXPECTED@
bad = [p for p, h in EXPECTED.items() if not os.path.exists(p) or sha256_file(p) != h]
if bad:
    fail("these files were not written as expected: %s. Use Run all again; if it happens again, send me this message." % ", ".join(bad))
else:
    log("all %d files written and checked (hashes match)" % len(EXPECTED))
'''

STEP3_MD = """\
## Step 3. Install Phenix

Needed once per session (about 15 to 25 minutes). If the user name and password are not found in the Secrets, two boxes ask for them.
If Phenix is already installed in this session, this step is skipped."""
STEP3 = r'''
PHENIX_URL = "@PHENIX_URL@"
PHENIX_VERSION = "@PHENIX_VERSION@"
INSTALLER = os.path.join(BIG, "@INSTALLER@")
PREFIX = os.path.join(BIG, "phenix-" + PHENIX_VERSION)
MIN_INSTALLER_MB = 50
MIN_FREE_GB = 20


def find_phenix_env():
    for pattern in ("phenix_env.sh", "*/phenix_env.sh", "*/*/phenix_env.sh"):
        hits = sorted(glob.glob(os.path.join(PREFIX, pattern)))
        if hits:
            return hits[0]
    return None


def phenix_works(env_script):
    r = subprocess.run(["bash", "-c", 'source "$0" && phenix.version', env_script], capture_output=True, text=True, errors="replace")
    lines = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    text = " | ".join(lines[:3])[:200]
    if r.returncode == 0:
        return True, text
    r2 = subprocess.run(["bash", "-c", 'source "$0" && command -v phenix.real_space_refine', env_script], capture_output=True, text=True, errors="replace")
    if r2.returncode == 0:
        return True, "phenix.real_space_refine found at %s (phenix.version said: %s)" % (r2.stdout.strip(), text or "nothing")
    return False, text or "phenix.version did not run"


def download_installer():
    import requests
    if os.path.exists(INSTALLER) and os.path.getsize(INSTALLER) >= MIN_INSTALLER_MB * 1e6:
        log("%s is already here (%d MB)" % (INSTALLER, os.path.getsize(INSTALLER) // 1000000))
        return True
    user = secret("PHENIX_USER", "Phenix user name (from the email), then Enter: ", hidden=False)
    password = secret("PHENIX_PASSWORD", "Phenix password (from the email), then Enter: ")
    if not user or not password:
        fail("no Phenix user name or password found. Kaggle: Add-ons -> Secrets, add PHENIX_USER and PHENIX_PASSWORD and tick both for this notebook. "
             "Colab: the key icon on the left, add the same two secrets and switch on notebook access. Then Run all again.")
        return False
    if len(password) >= 6:
        SECRETS.append(password)
    last = ""
    for attempt in range(1, 4):
        try:
            with requests.get(PHENIX_URL, auth=(user, password), stream=True, timeout=(30, 120)) as r:
                if r.status_code == 401:
                    fail("the Phenix server did not accept the user name and password (HTTP 401). The password changes on the 1st of every "
                         "month, so use the one from the newest email, and type both exactly as they appear. Then use Run all again.")
                    return False
                if r.status_code != 200:
                    last = "HTTP %d" % r.status_code
                    raise IOError(last)
                total = int(r.headers.get("Content-Length") or 0)
                done, shown = 0, -1
                with open(INSTALLER + ".part", "wb") as f:
                    for block in r.iter_content(1 << 22):
                        f.write(block)
                        done += len(block)
                        mark = done * 10 // total if total else done // 250000000
                        if mark != shown:
                            shown = mark
                            log("  downloaded %d MB%s" % (done // 1000000, " of %d MB" % (total // 1000000) if total else ""))
            if total and done != total:
                last = "the file is incomplete (%d of %d bytes)" % (done, total)
                raise IOError(last)
            with open(INSTALLER + ".part", "rb") as f:
                head = f.read(2)
            if head != b"#!" or done < MIN_INSTALLER_MB * 1e6:
                fail("the Phenix server sent %d bytes that do not look like the installer (it is probably a web page, not the file). Check the "
                     "user name and password, then use Run all again." % done)
                os.remove(INSTALLER + ".part")
                return False
            os.replace(INSTALLER + ".part", INSTALLER)
            return True
        except (requests.RequestException, IOError) as ex:
            last = last or str(ex)[:150]
            log("download attempt %d of 3 failed: %s" % (attempt, last))
            time.sleep(10 * attempt)
            last = ""
    fail("the Phenix installer could not be downloaded after 3 attempts. Use Run all again in a few minutes.")
    return False


def install_phenix():
    env = find_phenix_env()
    if env:
        ok, text = phenix_works(env)
        if ok:
            log("Phenix is already installed: %s (%s)" % (text, env))
            STATE.update(phenix_env=env, phenix_version=text)
            return
    free = shutil.disk_usage(BIG).free / 1e9
    if free < MIN_FREE_GB:
        fail("only %d GB of disk are free in %s; Phenix needs about %d GB or more while it installs. Kaggle: use a new session (Session options -> restart); "
             "Colab: Runtime -> Disconnect and delete runtime. Then Run all again." % (free, BIG, MIN_FREE_GB))
        return
    if not download_installer():
        return
    STATE["installer_sha256"] = sha256_file(INSTALLER)
    log("installer sha256: %s" % STATE["installer_sha256"])
    tmp = os.path.join(BIG, "tmp_phenix_install")
    os.makedirs(tmp, exist_ok=True)
    shutil.rmtree(PREFIX, ignore_errors=True)
    log("installing Phenix (15 to 25 minutes; a line is printed now and then) ...")
    t0 = time.time()
    rc, last, _ = run(["bash", INSTALLER, "-b", "-p", PREFIX], env={"TMPDIR": tmp}, every=60)
    shutil.rmtree(tmp, ignore_errors=True)
    if rc != 0:
        log("\n".join(last[-25:]))
        fail("the Phenix installer stopped with exit code %d (the last lines are above). If they say 'No space left', start a fresh session and "
             "run again; otherwise send me the zip." % rc)
        return
    env = find_phenix_env()
    ok, text = phenix_works(env) if env else (False, "phenix_env.sh was not found under %s" % PREFIX)
    if not ok:
        fail("Phenix was installed but does not start: %s" % text)
        return
    STATE.update(phenix_env=env, phenix_version=text)
    log("Phenix installed in %d minutes: %s" % ((time.time() - t0) // 60, text))
    os.remove(INSTALLER)


def phenix_probe():
    """Which Phenix programs exist, and what chain_comparison says about itself (its exact usage is part of the protocol)."""
    script = ('source "$0" && for t in phenix.dock_in_map phenix.real_space_refine phenix.chain_comparison phenix.version; do '
              'printf "%s: " $t; command -v $t || echo MISSING; done; echo; phenix.version 2>&1 | head -5; echo; echo "==== phenix.chain_comparison --help"; '
              'phenix.chain_comparison --help 2>&1 | head -120')
    try:
        r = subprocess.run(["bash", "-c", script, STATE["phenix_env"]], capture_output=True, text=True, errors="replace", timeout=600)
        text = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        text = "the probe of the Phenix programs did not finish within 10 minutes\n"
    with open("phenix_help.txt", "w", encoding="utf-8") as f:
        f.write(text)
    log("\n".join(text.splitlines()[:8]))
    missing = [t for t in ("phenix.dock_in_map", "phenix.real_space_refine", "phenix.chain_comparison") if (t + ": MISSING") in text]
    if missing:
        log("WARNING: this Phenix has no %s (see phenix_help.txt). MICA needs phenix.dock_in_map and phenix.real_space_refine; the scoring needs phenix.chain_comparison." % ", ".join(missing))


if STATE["ok"]:
    install_phenix()
    if STATE["ok"]:
        phenix_probe()
else:
    log("skipped: an earlier step stopped")
'''

STEP4_MD = """\
## Step 4. Python 3.8 environment for MICA

MICA needs Python 3.8 with PyTorch 1.8.1. The notebook builds it with `uv` from PyPI wheels with exact pins (`env/mica_requirements.txt`, 171 packages),
then tests the imports and runs a small 3D convolution on the GPU, so a GPU that this old PyTorch cannot drive is found here and not after an hour."""
STEP4 = r'''
UV_VERSION = "@UV_VERSION@"
PY_VERSION = "@PY_VERSION@"
ENV_PY_GIVEN = @ENV_PYTHON@                       # None: build the environment; a path is only used by the tests
ENV_DIR = os.path.join(BIG, "mica_env")
ENV_PY = ENV_PY_GIVEN or os.path.join(ENV_DIR, "bin", "python")
ENV_BIN = os.path.dirname(ENV_PY)
MICA_ENV = {"PATH": ENV_BIN + os.pathsep + os.environ.get("PATH", ""), "VIRTUAL_ENV": None, "PYTHONPATH": None, "PYTHONHOME": None}
IMPORT_TEST = ("import sys, torch, numpy, scipy, pandas, Bio, mrcfile, open3d, skimage, networkx, einops, natsort, atom3d, freesasa, psutil, superpose3d, tqdm, "
               "wandb, rotary_embedding_torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, "
               "'| numpy', numpy.__version__, '| open3d', open3d.__version__)")
GPU_TEST = r"""
import torch
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(3)
p = torch.cuda.get_device_properties(0)
print("gpu: %s | %.1f GB | capability %d.%d | this PyTorch is compiled for %s" % (p.name, p.total_memory / 2**30, p.major, p.minor, torch.cuda.get_arch_list()))
x = torch.randn(1, 4, 24, 24, 24, device="cuda")
y = torch.nn.Conv3d(4, 8, 3, padding=1).cuda()(x)
a = torch.randn(512, 512, device="cuda")
z = float((a @ a).sum()) + float(y.sum())
torch.cuda.synchronize()
print("3D convolution and matrix product on the GPU work")
"""


def env_imports():
    return run([ENV_PY, "-c", IMPORT_TEST], env=MICA_ENV, quiet=True)


def make_env():
    rc, last, _ = env_imports() if os.path.exists(ENV_PY) else (1, [], False)
    if rc == 0:
        log("the Python 3.8 environment already exists: " + last[-1])
        return True
    tools = {"UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or os.path.join(BIG, "uv_cache"),
             "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR") or os.path.join(BIG, "uv_python"), "UV_PYTHON_PREFERENCE": "only-managed"}
    rc, last, _ = run([sys.executable, "-m", "pip", "install", "-q", "uv==" + UV_VERSION])
    if rc != 0:
        fail("pip could not install uv (see the lines above). Use Run all again in a few minutes.")
        return False
    uv = [sys.executable, "-m", "uv"]
    log("uv: " + sh("%s -m uv --version" % sys.executable))
    for what, cmd in (("Python " + PY_VERSION, uv + ["python", "install", PY_VERSION]),
                      ("the virtual environment", uv + ["venv", "--python", PY_VERSION, ENV_DIR]),
                      ("the 171 packages (a few minutes; PyTorch alone is about 800 MB)", uv + ["pip", "install", "--python", ENV_PY, "--no-deps", "-r", "env/mica_requirements.txt"])):
        log("installing " + what + " ...")
        rc, last, _ = run(cmd, env=tools, every=20)
        if rc != 0:
            fail("uv could not install %s (exit code %d, the last lines are above). Use Run all again in a few minutes; if it happens again, send me the zip." % (what, rc))
            return False
    rc, last, _ = env_imports()
    if rc != 0 and any("libGL" in ln or "libGLU" in ln or "libgomp" in ln for ln in last):
        log("a system library is missing; installing it with apt ...")
        run(["bash", "-c", "apt-get update -qq && apt-get install -y -qq libgl1 libglu1-mesa libgomp1"], every=20)
        rc, last, _ = env_imports()
    if rc != 0:
        log("\n".join(last))
        fail("the Python packages were installed but do not import (see the lines above). Send me the zip.")
        return False
    log(last[-1])
    return True


def gpu_test():
    rc, last, _ = run([ENV_PY, "-c", GPU_TEST], env=MICA_ENV)
    if rc == 3:
        fail("PyTorch in the MICA environment does not see the GPU. Switch the GPU on (Kaggle: Session options -> Accelerator; Colab: Runtime -> "
             "Change runtime type) and Run all again.")
    elif rc != 0:
        fail("this GPU cannot run PyTorch 1.8.1 (see the lines above). Choose another GPU: Kaggle GPU T4 x2 or GPU P100, Colab T4. Then Run all again.")
    return rc == 0


def freeze_check():
    """The installed packages, compared with the pinned list (names are compared in the normalised form that pip uses)."""
    code = "import importlib.metadata as m; print(chr(10).join(sorted('%s==%s' % (d.metadata['Name'], d.version) for d in m.distributions())))"
    if os.path.exists("mica_env_freeze.txt"):
        os.remove("mica_env_freeze.txt")
    run([ENV_PY, "-c", code], env=MICA_ENV, log_file="mica_env_freeze.txt", quiet=True)
    norm = lambda line: re.sub(r"[-_.]+", "-", line.strip().lower())
    got = {norm(ln) for ln in open("mica_env_freeze.txt", encoding="utf-8") if "==" in ln}
    want = {norm(ln) for ln in open("env/mica_requirements.txt", encoding="utf-8") if "==" in ln and not ln.startswith("#")}
    STATE["env"] = {"python_version": PY_VERSION, "uv_version": UV_VERSION, "packages": len(got), "differences_to_pinned_list": sorted(got ^ want)[:20]}
    log("installed packages: %d; differences to env/mica_requirements.txt: %d %s" % (len(got), len(got ^ want), sorted(got ^ want)[:6] or ""))


if STATE["ok"]:
    if make_env():
        if REQUIRE_GPU:
            gpu_test()
        if ENV_PY_GIVEN is None:
            freeze_check()
else:
    log("skipped: an earlier step stopped")
'''

STEP5_MD = """\
## Step 5. MICA code, MICA weights, US-align

MICA is fetched at its pinned commit (a hash, so it cannot change under us). The weights come from Zenodo (record 15756654); their MD5 is compared
with the one that Zenodo lists. US-align is compiled from its pinned commit."""
STEP5 = r'''
MICA_REPO = "@MICA_REPO@"
MICA_COMMIT = "@MICA_COMMIT@"
WEIGHTS_URL = "@WEIGHTS_URL@"
ZENODO_API = "@ZENODO_API@"
USALIGN_REPO = "@USALIGN_REPO@"
USALIGN_COMMIT = "@USALIGN_COMMIT@"
USALIGN_GIVEN = @USALIGN_BIN@                      # None: compile it; a path is only used by the tests
MICA_DIR = os.path.join(BIG, "MICA")


def git_checkout(repo, commit, folder):
    if os.path.isdir(os.path.join(folder, ".git")) and sh("git -C %s rev-parse HEAD" % folder) == commit:
        return True
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    for cmd in (["git", "init", "-q", folder], ["git", "-C", folder, "remote", "add", "origin", repo],
                ["git", "-C", folder, "fetch", "-q", "--depth", "1", "origin", commit], ["git", "-C", folder, "checkout", "-q", "FETCH_HEAD"]):
        rc, last, _ = run(cmd)
        if rc != 0:
            log("\n".join(last))
            return False
    return sh("git -C %s rev-parse HEAD" % folder) == commit


def fetch_mica():
    if not git_checkout(MICA_REPO, MICA_COMMIT, MICA_DIR):
        fail("MICA could not be fetched at commit %s (see the lines above). Use Run all again in a few minutes." % MICA_COMMIT)
        return False
    log("MICA at commit %s" % MICA_COMMIT)
    pulchra = os.path.join(MICA_DIR, "modules", "pulchra304", "src", "pulchra")
    if not os.path.exists(pulchra):
        fail("MICA's pulchra program is missing at %s" % pulchra)
        return False
    os.chmod(pulchra, 0o755)
    return True


def zenodo_md5(key):
    import requests
    try:
        r = requests.get(ZENODO_API, timeout=30)
        r.raise_for_status()
        for f in r.json().get("files", []):
            if f.get("key") == key:
                return str(f.get("checksum", "")).split(":", 1)[-1] or None
    except Exception as ex:
        log("could not read the checksum from Zenodo: %s" % str(ex)[:150])
    return None


def fetch_weights():
    import requests
    pth = os.path.join(MICA_DIR, "trained_models", "MICA_best_model.pth")
    if os.path.exists(pth) and os.path.getsize(pth) > 1000:
        log("the weights are already here: %s" % pth)
    else:
        tgz = os.path.join(BIG, "trained_models.tar.gz")
        last = ""
        for attempt in range(1, 4):
            try:
                with requests.get(WEIGHTS_URL, stream=True, timeout=(30, 120)) as r:
                    r.raise_for_status()
                    done, shown = 0, -1
                    with open(tgz, "wb") as f:
                        for block in r.iter_content(1 << 22):
                            f.write(block)
                            done += len(block)
                            if done // 200000000 != shown:
                                shown = done // 200000000
                                log("  downloaded %d MB" % (done // 1000000))
                log("downloaded the weights: %d MB" % (done // 1000000))
                break
            except (requests.RequestException, IOError) as ex:
                last = str(ex)[:150]
                log("download attempt %d of 3 failed: %s" % (attempt, last))
                time.sleep(10 * attempt)
        else:
            fail("the MICA weights could not be downloaded from Zenodo after 3 attempts (%s). Use Run all again in a few minutes." % last)
            return False
        md5 = md5_file(tgz)
        want = zenodo_md5("trained_models.tar.gz")
        STATE["weights"] = {"url": WEIGHTS_URL, "size_bytes": os.path.getsize(tgz), "md5": md5, "md5_listed_by_zenodo": want or "not available"}
        if want and want != md5:
            fail("the downloaded weights have MD5 %s but Zenodo lists %s. Use Run all again; if it happens again, send me the zip." % (md5, want))
            return False
        log("weights archive: md5 %s (%s)" % (md5, "equals the one Zenodo lists" if want else "Zenodo gave no checksum to compare"))
        try:
            with tarfile.open(tgz) as t:
                try:
                    t.extractall(MICA_DIR, filter="data")
                except TypeError:
                    t.extractall(MICA_DIR)
        except (tarfile.TarError, OSError, EOFError) as ex:
            os.remove(tgz)
            fail("the weights archive cannot be unpacked (%s). Use Run all again; if it happens again, send me the zip." % str(ex)[:150])
            return False
        os.remove(tgz)
    if not os.path.exists(pth):
        found = glob.glob(os.path.join(MICA_DIR, "**", "*.pth"), recursive=True)
        fail("the weights archive did not give trained_models/MICA_best_model.pth (found: %s)" % (found or "no .pth file"))
        return False
    STATE["weights"].update(file=pth, file_size_bytes=os.path.getsize(pth), file_sha256=sha256_file(pth))
    log("weights file: %s (%d MB, sha256 %s)" % (pth, os.path.getsize(pth) // 1000000, STATE["weights"]["file_sha256"][:16]))
    return True


def import_mica():
    code = "import utils.modeler, utils.predict, utils.dock_in_map, utils.process_AF3_results, utils.create_grids, utils.preprocessing; print('MICA modules import')"
    rc, last, _ = run([ENV_PY, "-c", code], env=MICA_ENV, cwd=MICA_DIR)
    if rc != 0:
        fail("MICA's own Python modules cannot be imported (see the lines above). Send me the zip.")
    return rc == 0


def build_usalign():
    if USALIGN_GIVEN:
        STATE["usalign"] = USALIGN_GIVEN
    else:
        src = os.path.join(BIG, "USalign_src")
        exe = os.path.join(src, "USalign")
        if not os.path.exists(exe):
            if not git_checkout(USALIGN_REPO, USALIGN_COMMIT, src):
                fail("US-align could not be fetched at commit %s" % USALIGN_COMMIT)
                return False
            rc, last, _ = run(["g++", "-O3", "-ffast-math", "-o", exe, os.path.join(src, "USalign.cpp"), "-lm"], cwd=src, every=10)
            if rc != 0:
                fail("US-align could not be compiled (see the lines above). Send me the zip.")
                return False
        STATE["usalign"] = exe
    r = subprocess.run([STATE["usalign"], "-h"], capture_output=True, text=True, errors="replace")
    head = [ln for ln in (r.stdout + r.stderr).splitlines() if "Version" in ln]
    STATE["usalign_version"] = head[0].strip() if head else "unknown"
    log("US-align: %s (%s)" % (STATE["usalign"], STATE["usalign_version"]))
    return True


if STATE["ok"]:
    if fetch_mica() and fetch_weights() and import_mica():
        build_usalign()
else:
    log("skipped: an earlier step stopped")
'''

STEP6_MD = """\
## Step 6. Run MICA on the entry and score the models

For each entry: the input folder is built, then MICA's three programs run one after the other with the same arguments as `MICA_pipeline.sh`
(1. process the AlphaFold predictions, 2. dock them into the map with Phenix, 3. neural network, model building, Pulchra, Phenix refinement).
Time, GPU memory and free RAM are recorded for every program. Then every model that MICA wrote is scored against the deposited model with US-align
and Phenix `chain_comparison`; the raw output of both is saved."""
STEP6 = r'''
sys.path.insert(0, os.path.join(WORK, "scripts"))
import evaluate_models, mica_inputs

ENTRIES = @ENTRIES@
TIMEOUTS = {"1_process_af3": 1800, "2_dock": 3 * 3600, "3_build": 3 * 3600}     # seconds; a program that takes longer is stopped and counted as failed
INPUT_ROOT = os.path.join(MICA_DIR, "input")
METRIC_COLUMNS = ["entry", "model_kind", "model_file", "status", "note", "tm_score", "tm_score_pred_norm", "ca_rmsd", "aligned_ca_length", "seq_identity",
                  "pred_residues", "native_residues", "coverage", "chain_comparison"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def download(url, dest, min_bytes=1, tries=3):
    import requests
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last = ""
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                done = 0
                with open(dest + ".part", "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
                        done += len(block)
            if done < min_bytes:
                raise IOError("only %d bytes arrived" % done)
            os.replace(dest + ".part", dest)
            return dest
        except (requests.RequestException, IOError) as ex:
            last = str(ex)[:200]
            log("download attempt %d of %d failed (%s): %s" % (attempt, tries, url, last))
            time.sleep(5 * attempt)
    raise RuntimeError("could not download %s: %s" % (url, last))


def find_af3_dir(rows):
    """The folder that holds the AlphaFold Server files of this entry. Looks in the Kaggle inputs, unpacks a zip if needed, and on Colab asks for the zip."""
    bases = [b for b in (os.environ.get("MICA_AF3_DIR"), "/kaggle/input", os.path.join(WORK, "af3_upload"), WORK) if b and os.path.isdir(b)]

    def complete(root):
        try:
            for r in rows:
                mica_inputs.find_model(root, r["job_name"])
            return True
        except ValueError:
            return False

    def search():
        for b in bases:
            if complete(b):
                return b
        for b in bases:
            for z in sorted(glob.glob(os.path.join(b, "**", "*.zip"), recursive=True)):
                name = os.path.basename(z).lower()
                if os.path.dirname(os.path.abspath(z)) == os.path.abspath(OUT_DIR) or not ("alpha" in name or "af3" in name or "fold" in name):
                    continue
                target = os.path.join(WORK, "af3_unzipped", os.path.splitext(os.path.basename(z))[0])
                if not os.path.isdir(target):
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(target)
                if complete(target):
                    return target
        return None

    found = search()
    if not found and IN_COLAB:
        log("The AlphaFold Server files were not found. A file chooser appears now: choose AlphaFold_ServerJOBS.zip (the zip with the 26 fold_*_model_0.cif files).")
        try:
            os.makedirs(os.path.join(WORK, "af3_upload"), exist_ok=True)
            os.chdir(os.path.join(WORK, "af3_upload"))
            from google.colab import files
            files.upload()
        finally:
            os.chdir(WORK)
        bases.append(os.path.join(WORK, "af3_upload"))
        found = search()
    if not found:
        raise RuntimeError("the AlphaFold Server files of this entry (%s) were not found. Kaggle: Add Input -> Upload -> New Dataset, upload "
                           "AlphaFold_ServerJOBS.zip, keep it private, and Run all again. Colab: run the cell again and choose the zip when the file chooser appears."
                           % ", ".join("fold_%s_model_0.cif" % r["job_name"].lower() for r in rows))
    return found


def contour_from_header_xml(text):
    """(contour level, resolution) from an EMDB header file: the primary contour and the final reconstruction resolution. None where missing."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    local = lambda el: el.tag.rsplit("}", 1)[-1]
    level, resolution = None, None
    contours = [el for el in root.iter() if local(el) == "contour"]
    contours.sort(key=lambda el: el.get("primary", "false").lower() != "true")
    for el in contours:
        for ch in el:
            if local(ch) == "level" and ch.text and level is None:
                level = float(ch.text)
    for el in root.iter():
        if local(el) == "final_reconstruction":
            for ch in el:
                if local(ch) == "resolution" and ch.text and resolution is None:
                    resolution = float(ch.text)
    return level, resolution


def map_parameters(row):
    """Contour level and resolution of the map, the way MICA's README does it (utils/emdb_extractor.py), with the EMDB header file as the second way."""
    number = row["emdb_id"].split("-")[-1]
    rc, last, _ = run([ENV_PY, "utils/emdb_extractor.py", "--emdb_id", number], cwd=MICA_DIR, env=MICA_ENV, quiet=True)
    text = "\n".join(last)
    m1, m2 = re.search(r"Contour Level:\s*([-+0-9.eE]+)", text), re.search(r"Resolution:\s*([-+0-9.eE]+)", text)
    if rc == 0 and m1 and m2:
        return float(m1.group(1)), float(m2.group(1)), "emdb_extractor.py"
    log("emdb_extractor.py gave no numbers (%s); trying the EMDB header file" % text.strip()[-150:])
    url = re.sub(r"/map/[^/]*$", "/header/emd-%s.xml" % number, row["map_url"])
    path = download(url, os.path.join(BIG, "downloads", "emd-%s.xml" % number), 200)
    with open(path, encoding="utf-8") as f:
        level, resolution = contour_from_header_xml(f.read())
    if level is None:
        raise RuntimeError("no contour level for %s in the EMDB header file %s" % (row["emdb_id"], url))
    return level, resolution if resolution is not None else float(row["resolution_A"]), "emdb header file"


def prepare_entry(entry, row, res):
    jobs = mica_inputs.read_jobs("data/af3/af3_jobs.csv", entry)
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
    log("AlphaFold Server files of %s: found in %s, hashes match the checked list" % (entry, models))
    info = mica_inputs.build("data/af3/af3_jobs.csv", models, entry, INPUT_ROOT)
    folder = info["folder"]
    number = row["emdb_id"].split("-")[-1]
    map_path = os.path.join(folder, "emd_%s.map" % number)
    if not os.path.exists(map_path):
        gz = download(row["map_url"], os.path.join(BIG, "downloads", os.path.basename(row["map_url"])), 100000)
        try:
            with gzip.open(gz, "rb") as src, open(map_path, "wb") as out:
                shutil.copyfileobj(src, out)
        except OSError:
            shutil.copyfile(gz, map_path)                           # not gzipped after all
    res["map_sha256"] = sha256_file(map_path)
    run([ENV_PY, "-c", "import sys, mrcfile; m = mrcfile.open(sys.argv[1], permissive=True); d = m.data; "
         "print('map grid', d.shape, 'voxel size', [round(float(v), 4) for v in (m.voxel_size.x, m.voxel_size.y, m.voxel_size.z)], "
         "'min %.4g max %.4g mean %.4g std %.4g' % (d.min(), d.max(), d.mean(), d.std()))", map_path], env=MICA_ENV)
    native = download(row["cif_url"], os.path.join(WORK, "native", entry + ".cif"), 1000)
    contour, resolution, how = map_parameters(row)
    res.update(contour=contour, resolution=resolution, parameters_from=how, resolution_in_table=row["resolution_A"])
    log("%s: contour level %s and resolution %s from %s (resolution in data/entries_2026.csv: %s)" % (entry, contour, resolution, how, row["resolution_A"]))
    return folder, map_path, native, contour, resolution


def mica_programs(entry, folder, map_path, contour, resolution):
    fasta, af3, act = os.path.join(folder, entry.lower() + ".fasta"), os.path.join(folder, "AF3_results"), STATE["phenix_env"]
    return [("1_process_af3", [ENV_PY, "utils/process_AF3_results.py", "-f", fasta, "-a", af3]),
            ("2_dock", [ENV_PY, "utils/dock_in_map.py", "-m", map_path, "-c", str(contour), "-r", str(resolution), "-f", fasta, "-a", af3, "--phenix_act", act]),
            ("3_build", [ENV_PY, "run.py", "-m", map_path, "-f", fasta, "-a", af3, "--run_pulchra", "--pulchra_path=modules/pulchra304/src/pulchra",
                         "--run_phenix", "--phenix_act=" + act, "--resolution=" + str(resolution), "--device=cuda"])]


def stage_output(entry, folder, name):
    """The file that shows that a program of MICA worked: (path or glob, what it is)."""
    low = entry.lower()
    if name == "1_process_af3":
        return os.path.join(folder, "AF3_structures", "%s_1" % entry, "ranked_0.pdb"), "the processed AlphaFold structure"
    if name == "2_dock":
        return os.path.join(folder, "%s_af3_docked.pdb" % entry), "the docked AlphaFold model"
    return os.path.join(MICA_DIR, "output", "*_%s_MICA_all_atom_model.pdb" % low), "the all-atom model"


def run_mica(entry, folder, map_path, contour, resolution, res):
    for name, cmd in mica_programs(entry, folder, map_path, contour, resolution):
        marker = os.path.join(folder, ".done_" + name)
        if os.path.exists(marker):
            log("\n=== %s: finished earlier in this session, not repeated ===" % name)
            continue
        log("\n=== MICA program %s: %s" % (name, " ".join(cmd[1:])))
        mon = Monitor(os.path.join(WORK, "gpu", "%s_%s.csv" % (entry, name)))
        mon.start()
        t0 = time.time()
        rc, last, timed_out = run(cmd, env=MICA_ENV, cwd=MICA_DIR, log_file=os.path.join(WORK, "logs", "%s_%s.log" % (entry, name)), timeout=TIMEOUTS[name], every=1)
        info = mon.stop()
        info.update(rc=rc, seconds=round(time.time() - t0), timed_out=timed_out)
        res["steps"][name] = info
        log("%s finished: exit code %s, %d s, GPU memory peak %s MB, lowest free RAM %s GB" % (name, rc, info["seconds"], info["peak_gpu_mb"], info["min_ram_available_gb"]))
        path, what = stage_output(entry, folder, name)
        made = glob.glob(path)
        if rc != 0 or timed_out or not made:
            res.update(failed_step=name, note=("timeout" if timed_out else "exit code %s" % rc) + ("" if made else "; %s was not written" % what))
            log("ENTRY %s STOPPED in %s: %s. The last lines of the program:\n%s" % (entry, name, res["note"], "\n".join(last[-15:])))
            if name == "3_build":
                diagnose_refine(entry, map_path, resolution)
            return False
        with open(marker, "w") as f:
            f.write("done\n")
        if name == "3_build":
            diagnose_refine(entry, map_path, resolution)
    return True


def diagnose_refine(entry, map_path, resolution):
    """MICA runs phenix.real_space_refine with its output thrown away; if the refined model is missing, the same command is run again with the output kept."""
    models = glob.glob(os.path.join(MICA_DIR, "output", "*_%s_MICA_all_atom_model.pdb" % entry.lower()))
    if not models or glob.glob(os.path.join(MICA_DIR, "output", "*_%s_MICA_all_atom_model_real_space_refined*.pdb" % entry.lower())):
        return
    log("the all-atom model exists but the refined model does not: running phenix.real_space_refine again, with its output kept")
    out = os.path.join(MICA_DIR, "output")
    cmd = ["bash", "modules/phenix.sh", STATE["phenix_env"], out, "phenix.real_space_refine %s %s %s resolution=%s" % (
        os.path.abspath(models[0]), os.path.abspath(map_path), os.path.abspath(os.path.join(MICA_DIR, "modules", "phenix.eff")), resolution)]
    rc, last, _ = run(cmd, cwd=MICA_DIR, log_file=os.path.join(WORK, "logs", "%s_refine_diagnosis.log" % entry), timeout=3600, every=5)
    log("phenix.real_space_refine exit code %s" % rc)


def classify(path):
    n = os.path.basename(path)
    if "real_space_refined" in n:
        return "mica_refined"
    if n.endswith("_all_atom_model.pdb"):
        return "mica_all_atom"
    if "(init)" in n:
        return "mica_init_ca"
    if n.endswith("_ca_model.pdb"):
        return "mica_ca"
    return "mica_other"


def chain_comparison(query, target, out_txt):
    """phenix.chain_comparison target.pdb query.pdb (Phenix documentation: the target comes first). The whole output is kept.
    If Phenix cannot read the deposited model as mmCIF, it is converted to a PDB file (Biopython) and the call is repeated."""
    def call(tgt):
        with open(out_txt, "a", encoding="utf-8") as f:
            f.write("# phenix.chain_comparison %s %s\n" % (tgt, query))
        cmd = ["bash", "-c", 'source "$0" && phenix.chain_comparison "$1" "$2"', STATE["phenix_env"], tgt, query]
        return run(cmd, log_file=out_txt, timeout=900, quiet=True)
    if os.path.exists(out_txt):
        os.remove(out_txt)
    rc, last, timed_out = call(target)
    status = "timeout" if timed_out else ("ok" if rc == 0 else "failed (exit code %s)" % rc)
    if rc != 0 and not timed_out and target.endswith(".cif"):
        as_pdb = target[:-4] + "_as.pdb"
        rc2, _, _ = run([ENV_PY, "-c", "import sys; from Bio.PDB import MMCIFParser, PDBIO; s = MMCIFParser(QUIET=True).get_structure('n', sys.argv[1]); "
                         "io = PDBIO(); io.set_structure(s); io.save(sys.argv[2])", target, as_pdb], env=MICA_ENV, quiet=True)
        if rc2 == 0:
            rc, last, timed_out = call(as_pdb)
            status = "timeout" if timed_out else ("ok (deposited model as pdb)" if rc == 0 else "failed (exit code %s)" % rc)
    log("phenix.chain_comparison (%s) says, last lines:\n%s" % (os.path.basename(query), "\n".join(last[-14:])))
    return status


def score_entry(entry, row, folder, native, res):
    models = sorted(glob.glob(os.path.join(MICA_DIR, "output", "*_%s_MICA*.pdb" % entry.lower())))
    kinds = [(classify(p), p) for p in models if not p.endswith(".part")]
    docked = os.path.join(folder, "%s_af3_docked.pdb" % entry)
    if os.path.exists(docked):
        kinds.append(("af3_docked", docked))
    if row["n_chains"] == "1" and row["unique_chain_seqs"] == "1":                  # control: the AlphaFold Server prediction alone
        kinds.append(("af3_model_0", os.path.join(folder, "AF3_results", "%s_1" % entry.lower(), "fold_%s_1_model_0.cif" % entry.lower())))
    rows = []
    os.makedirs(os.path.join(WORK, "scoring"), exist_ok=True)
    for kind, path in kinds:
        ev = evaluate_models.evaluate(STATE["usalign"], path, native, int(row["modeled_residues"]))
        full = subprocess.run([STATE["usalign"], path, native, "-mol", "protein"], capture_output=True, text=True, errors="replace")
        with open(os.path.join(WORK, "scoring", "%s_%s_usalign.txt" % (entry, kind)), "w", encoding="utf-8") as f:
            f.write("# USalign %s %s -mol protein\n%s%s" % (path, native, full.stdout, full.stderr))
        cc = chain_comparison(path, native, os.path.join(WORK, "scoring", "%s_%s_chain_comparison.txt" % (entry, kind))) if STATE["phenix_env"] else "no Phenix"
        rows.append(dict(ev, entry=entry, model_kind=kind, model_file=os.path.basename(path), chain_comparison=cc))
    res["models"] = rows
    return rows


def process_entry(entry):
    res = {"entry": entry, "status": "failed", "failed_step": "", "note": "", "steps": {}, "models": []}
    STATE["entries"][entry] = res
    t0 = time.time()
    row = {r["pdb_id"]: r for r in read_csv("data/entries_2026.csv")}.get(entry)
    if row is None:
        res.update(failed_step="inputs", note="%s is not in data/entries_2026.csv" % entry)
        return res
    log("\n" + "=" * 72 + "\nENTRY %s (EMDB %s, %s A, %s modeled residues)\n" % (entry, row["emdb_id"], row["resolution_A"], row["modeled_residues"]) + "=" * 72)
    try:
        folder, map_path, native, contour, resolution = prepare_entry(entry, row, res)
    except Exception as ex:
        res.update(failed_step="inputs", note=str(ex)[:500])
        log("\nENTRY %s STOPPED while preparing the inputs: %s" % (entry, ex))
        return res
    ok = run_mica(entry, folder, map_path, contour, resolution, res)
    try:
        score_entry(entry, row, folder, native, res)
    except Exception as ex:
        res.update(failed_step=res["failed_step"] or "scoring", note=(res["note"] + " scoring: %s" % ex)[:500])
        log("scoring stopped: %s" % ex)
        ok = False
    if ok:
        res["status"] = "ok"
    res["seconds"] = round(time.time() - t0)
    return res


def summary_table():
    log("\n%-5s %-14s %-8s %7s %8s %7s %9s %8s  %s" % ("entry", "model", "status", "TM", "aligned", "RMSD", "coverage", "seq id", "chain_comparison"))
    for e in STATE["entries"].values():
        for m in e["models"]:
            log("%-5s %-14s %-8s %7s %8s %7s %9s %8s  %s" % (m["entry"], m["model_kind"], m["status"], m["tm_score"], m["aligned_ca_length"], m["ca_rmsd"],
                                                            m["coverage"], m["seq_identity"], m["chain_comparison"]))
        log("%-5s ENTRY %s%s; seconds per program: %s" % (e["entry"], e["status"].upper(), (" (stopped in %s: %s)" % (e["failed_step"], e["note"][:120])) if e["failed_step"] else "",
                                                          {k: v["seconds"] for k, v in e["steps"].items()}))


if STATE["ok"] and STATE["usalign"]:
    for entry in ENTRIES:
        try:
            process_entry(entry)
        except Exception:
            import traceback
            STATE["entries"].setdefault(entry, {"entry": entry, "steps": {}, "models": []}).update(status="failed", failed_step="unexpected error",
                                                                                               note=traceback.format_exc()[-600:])
            log("\nENTRY %s STOPPED by an unexpected error:\n%s" % (entry, traceback.format_exc()))
    write_csv("pilot_metrics.csv", METRIC_COLUMNS, [m for e in STATE["entries"].values() for m in e["models"]])
    write_csv("entry_status.csv", ["entry", "status", "failed_step", "note", "seconds", "contour", "resolution", "parameters_from", "map_sha256"],
              list(STATE["entries"].values()))
    summary_table()
else:
    log("skipped: an earlier step stopped")
'''

STEP7_MD = """\
## Step 7. Pack the results

Always run this last cell, also after a problem or after stopping early. It packs the logs, the numbers, the raw tool output and MICA's models into one zip
(not the maps, the weights or the AlphaFold predictions). The Phenix password is never written to the zip; the cell checks every file for it first."""
STEP7 = r'''
MAX_FILE = 60 * 1000 * 1000


def collect():
    files = []
    for name in ("run.log", "machine.txt", "phenix_help.txt", "mica_env_freeze.txt", "pilot_metrics.csv", "entry_status.csv"):
        files.append(os.path.join(WORK, name))
    for sub in ("logs", "gpu", "scoring"):
        for dirpath, _, names in os.walk(os.path.join(WORK, sub)):
            files += [os.path.join(dirpath, n) for n in sorted(names)]
    mica = globals().get("MICA_DIR")
    if mica:
        for dirpath, _, names in os.walk(os.path.join(mica, "docking_logs")):
            files += [os.path.join(dirpath, n) for n in sorted(names)]
        for entry in STATE["entries"]:
            low = entry.lower()
            for dirpath, _, names in os.walk(os.path.join(mica, "output")):
                files += [os.path.join(dirpath, n) for n in sorted(names) if ("_%s_" % low) in n or ("_%s." % low) in n]
            folder = os.path.join(mica, "input", entry)
            for dirpath, dirs, names in os.walk(folder):
                dirs[:] = [d for d in dirs if d not in ("AF3_results", "grids", "AF3_encodings", "temp_maps")]
                for n in sorted(names):
                    if not n.endswith((".map", ".mrc")):
                        files.append(os.path.join(dirpath, n))
    return [p for p in dict.fromkeys(files) if os.path.isfile(p)]


def pack():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")
    all_ok = STATE["ok"] and bool(STATE["entries"]) and all(e["status"] == "ok" for e in STATE["entries"].values())
    name = "pilot_result_%s%s" % (stamp, "" if all_ok else "_INCOMPLETE")
    info = {"notebook": "pilot_mica", "started_utc": STATE["started"], "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_minutes": round((time.time() - STATE["t0"]) / 60), "problem": STATE["problem"], "platform": PLATFORM, "python": platform.python_version(),
            "os": platform.platform(), "gpu": STATE["gpu"], "cpu_cores": os.cpu_count(), "ram_gb": meminfo_gb(), "phenix_installer": os.path.basename(globals().get("INSTALLER", "")),
            "phenix_installer_sha256": STATE["installer_sha256"], "phenix_version_output": STATE["phenix_version"], "env": STATE["env"],
            "mica_commit": globals().get("MICA_COMMIT", ""), "weights": STATE["weights"], "usalign_commit": globals().get("USALIGN_COMMIT", ""),
            "usalign_version": STATE["usalign_version"], "input_sha256": globals().get("EXPECTED", {}),
            "entries": {k: {kk: vv for kk, vv in v.items() if kk != "models"} for k, v in STATE["entries"].items()}}
    with open("run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    wanted = [os.path.join(WORK, "run_info.json")] + collect()
    zip_path = os.path.join(OUT_DIR, name + ".zip")
    left_out = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in wanted:
            if os.path.getsize(path) > MAX_FILE:
                left_out.append(path + " (larger than 60 MB)")
                continue
            with open(path, "rb") as f:
                data = f.read()
            if any(s and s.encode() in data for s in SECRETS):
                left_out.append(path + " (contained the password)")
                continue
            z.writestr(name + "/" + (os.path.relpath(path, WORK) if path.startswith(WORK + os.sep) else path.lstrip(os.sep)), data)
        n = len(z.namelist())
    log("zip written: %s (%d files, %d MB)" % (zip_path, n, os.path.getsize(zip_path) // 1000000))
    if left_out:
        log("left out of the zip: %s" % "; ".join(left_out))
    return zip_path, all_ok


zip_path, all_ok = pack()
if all_ok:
    log("\nRESULT: OK. Send me %s." % zip_path)
elif STATE["ok"]:
    log("\nRESULT: AT LEAST ONE ENTRY DID NOT FINISH. Send me %s; it shows where it stopped." % zip_path)
else:
    log("\nRESULT: STOPPED (%s). Send me %s." % (STATE["problem"], zip_path))
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    where = "Kaggle: in the right panel open Data -> Output, click the three dots next to the zip, then Download." if IN_KAGGLE else "Download it from the file list."
    log("(%s)" % where)
'''


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text if text.endswith("\n") else text + "\n"


def build(out_path, entries=("26DK",), phenix_url=PHENIX_URL, mica_repo=MICA_REPO, mica_commit=MICA_COMMIT, weights_url=WEIGHTS_URL,
          zenodo_api=ZENODO_API, usalign_repo=USALIGN_REPO, usalign_commit=USALIGN_COMMIT, entries_path=None, require_gpu=True,
          env_python=None, usalign_bin=None, net_hosts=NET_HOSTS, nb_name="pilot_mica.ipynb"):
    """Writes the notebook. The keyword arguments other than `entries` exist for the tests (local servers, a ready environment, no GPU)."""
    m = re.search(r"version=([^&]+)&file=([^&]+)$", phenix_url)
    if not m:
        raise ValueError("the Phenix link must end in ?version=...&file=...: %s" % phenix_url)
    version, installer = m.group(1), m.group(2)
    files = []
    for src, target in FILES:
        path = entries_path if (entries_path and src == "data/entries_2026.csv") else os.path.join(ROOT, src)
        text = read(path)
        if target.endswith(".py"):
            compile(text, target, "exec")
        files.append((target, text))
    expected = {p: sha256_text(t) for p, t in files}

    def fill(text):
        for key, value in (("@PHENIX_URL@", phenix_url), ("@PHENIX_VERSION@", version), ("@INSTALLER@", installer), ("@MICA_REPO@", mica_repo),
                           ("@MICA_COMMIT@", mica_commit), ("@WEIGHTS_URL@", weights_url), ("@ZENODO_API@", zenodo_api), ("@USALIGN_REPO@", usalign_repo),
                           ("@USALIGN_COMMIT@", usalign_commit), ("@UV_VERSION@", UV_VERSION), ("@PY_VERSION@", PY_VERSION),
                           ("@ENTRIES@", repr(list(entries))), ("@ENTRY_LIST@", ", ".join(entries)), ("@REQUIRE_GPU@", repr(bool(require_gpu))),
                           ("@ENV_PYTHON@", repr(env_python)), ("@USALIGN_BIN@", repr(usalign_bin)), ("@NET_HOSTS@", repr(tuple(net_hosts))),
                           ("@EXPECTED@", repr(expected))):
            text = text.replace(key, value)
        return text

    nb = new_notebook()
    nb.metadata = {"colab": {"name": nb_name, "provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    cells = [("intro", new_markdown_cell(fill(INTRO))), ("step1-title", new_markdown_cell(STEP1_MD)), ("step1", new_code_cell(fill(STEP1).strip("\n") + "\n")),
             ("step2-title", new_markdown_cell(STEP2_MD)), ("step2-dirs", new_code_cell(STEP2_DIRS.strip("\n") + "\n"))]
    for i, (path, text) in enumerate(files, 1):
        cells.append(("file-%d" % i, new_code_cell("%%writefile " + path + "\n" + text)))
    cells += [("step2-check", new_code_cell(fill(STEP2_CHECK).strip("\n") + "\n")),
              ("step3-title", new_markdown_cell(STEP3_MD)), ("step3", new_code_cell(fill(STEP3).strip("\n") + "\n")),
              ("step4-title", new_markdown_cell(STEP4_MD)), ("step4", new_code_cell(fill(STEP4).strip("\n") + "\n")),
              ("step5-title", new_markdown_cell(STEP5_MD)), ("step5", new_code_cell(fill(STEP5).strip("\n") + "\n")),
              ("step6-title", new_markdown_cell(STEP6_MD)), ("step6", new_code_cell(fill(STEP6).strip("\n") + "\n")),
              ("step7-title", new_markdown_cell(STEP7_MD)), ("step7", new_code_cell(fill(STEP7).strip("\n") + "\n"))]
    for cid, cell in cells:
        cell["id"] = cid
        nb.cells.append(cell)
    for cell in nb.cells:
        if cell.cell_type == "code":
            if not cell.source.startswith("%%writefile"):                # the scripts were compiled above; the others are data
                compile(cell.source, "cell " + cell["id"], "exec")      # a syntax error shows up here, not on Kaggle
        left = re.findall(r"@[A-Z_]+@", cell.source)
        if left:
            raise ValueError("placeholder %s is left in cell %s" % (left[0], cell["id"]))
    nbformat.validate(nb)
    nbformat.write(nb, out_path)
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, "pilot_mica.ipynb"))
    p.add_argument("--entries", nargs="+", default=["26DK"])
    p.add_argument("--phenix-url", default=PHENIX_URL)
    args = p.parse_args(argv)
    expected = build(args.out, entries=args.entries, phenix_url=args.phenix_url)
    print("wrote %s with %d embedded files" % (args.out, len(expected)))
    for path, digest in expected.items():
        print("  %s  %s" % (digest[:16], path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
