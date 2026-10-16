#!/usr/bin/env python3
"""Build scripts/select_colab.ipynb: Phase 3b and 3c in one Colab run.

The notebook carries verbatim copies of select_entries.py, compute_cc_mask.py and novelty_check.py and of the frozen
pool files, so the one notebook file is all that has to be uploaded to Colab. It installs Phenix from the user's own
download link (credentials are typed in at run time and never written to a file), computes CC_mask for the entries
that the selection rule asks for, repeats until the list is final, runs the novelty check and packs one zip.

    python make_select_notebook.py            # uses data/pool, data/selection and the Phenix link below

The cell sources live in this file, not in the .ipynb, so the notebook is a build product.
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
PHENIX_URL = "https://phenix-online.org/download/phenix/release/send_octet_stream.cgi?version=2.2.1-6174&file=Phenix-2.2.1-6174-Linux-x86_64.sh"
MMSEQS_VERSION = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
MMSEQS_SHA256 = {"avx2": "bd9b0234da5949ad528d5b5f9ea4cda9c1e23dce14b46c0791d4d919a76e61ce",
                 "sse41": "0ece6a6af8f5d198bccabc98cb5fddfc9212ce04bb332e486bfdcc468c0d7b08"}
MMSEQS_URL = "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-@ARCH@.tar.gz"
BATCH = 16
SCRIPTS = ("select_entries.py", "compute_cc_mask.py", "novelty_check.py")
DATA = (("data/pool/pool_candidates.csv", "pool", "pool_candidates.csv"),
        ("data/pool/pool_candidate_sequences.fasta", "pool", "pool_candidate_sequences.fasta"),
        ("data/pool/pool_ids_all_cryoem.txt", "pool", "pool_ids_all_cryoem.txt"),
        ("data/selection/candidate_clusters.csv", "clusters", "candidate_clusters.csv"))

INTRO = """\
# MICA benchmark, Phase 3b and 3c: choose the test entries

This notebook does the rest of the entry selection in one run:

1. installs Phenix from your download link (you type the user name and password when asked),
2. computes the map-model correlation (CC_mask, `phenix.map_model_cc`) for the entries that the selection rule asks for,
3. applies the selection rule of `docs/SPEC.md` (decisions 4 and 5) and repeats until the list of entries is final,
4. checks how close each chosen protein is to proteins in the PDB before 2026 (decision 9, MMseqs2),
5. packs everything into one zip file and downloads it.

**What you do**

1. Click **Runtime -> Run all**. If Colab warns that the notebook was not authored by Google, click **Run anyway**.
2. In step 3 two boxes appear. Paste the Phenix **user name** from the Phenix email, press Enter, then the **password**
   (the second box hides what you type). The password is not saved in the notebook or in the zip.
3. Keep this tab open and the computer awake. It takes about 1.5 to 3 hours. Do not edit any code.
   If Colab disconnects, Colab erases its files: run all again from the top.
4. At the end the zip `selection_result_....zip` downloads by itself. Send it to me.
   If no download starts: click the folder icon on the left, open the folder `work`, click the three dots next to the zip, then **Download**.

**If something goes wrong.** A step that cannot continue prints `STOPPED` and a plain explanation; the later steps are skipped
and the last step still makes a zip with the log. Send me that zip. If a step shows a red error instead, copy all of its text and send it to me.
To stop early, press the stop button, then run only the last cell: it packs what exists.

**Phenix.** Phenix is licensed to you personally. Do not share the installer, the password or this Colab session. When you are done,
use **Runtime -> Disconnect and delete runtime** to remove the Phenix copy from Colab.
"""

STEP1_MD = "## Step 1. Look at the machine"
STEP1 = r'''
import datetime, getpass, glob, hashlib, json, os, platform, re, shutil, subprocess, sys, time, zipfile
from collections import deque

IN_COLAB = os.path.isdir("/content") and not os.environ.get("MICA_WORK")      # MICA_WORK is only used by the tests
WORK = os.environ.get("MICA_WORK") or ("/content/work" if IN_COLAB else os.path.abspath("work"))
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
LOG = os.path.join(WORK, "run.log")
STATE = {"ok": True, "problem": "", "final": False, "rounds": 0, "t0": time.time(), "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "phenix_env": None, "phenix_version": "", "installer_sha256": "", "mmseqs_version": "", "novelty": "not run"}
SECRETS = []          # kept in memory only, to make sure they never end up in the zip


def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def fail(msg):
    STATE["ok"] = False
    STATE["problem"] = msg
    log("\n" + "!" * 72 + "\nSTOPPED: " + msg + "\nThe steps below are skipped. The last step still packs a zip with the log: send it to me.\n" + "!" * 72)


def run(cmd, env=None, every=1, cwd=None):
    """Runs a command and prints its output as it comes. Returns (exit code, last lines). every=N prints one line in N."""
    e = dict(os.environ)
    e.update(env or {})
    e["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=e, cwd=cwd, errors="replace")
    last, n = deque(maxlen=40), 0
    for line in p.stdout:
        line = line.rstrip("\n")
        last.append(line)
        n += 1
        if every == 1 or n <= 5 or n % every == 0:
            log(line)
    p.wait()
    return p.returncode, list(last)


def secret(name, prompt, hidden=True):
    """Colab Secrets, Kaggle Secrets, an environment variable, or a prompt, in this order."""
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
    return (getpass.getpass(prompt) if hidden else input(prompt)).strip()


def sha256_file(path):
    h = hashlib.sha256()
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


free_gb = round(shutil.disk_usage(WORK).free / 1e9)
log("python %s | %s | %s CPU cores | %s GB RAM | %d GB free disk" % (platform.python_version(), platform.platform(), os.cpu_count(), meminfo_gb(), free_gb))
log("This is Colab." if IN_COLAB else "This is not Colab (a test run).")
if free_gb < 45:
    log("WARNING: only %d GB of disk are free; Phenix needs about 40 GB while it installs. If step 3 fails with 'No space left', open Runtime -> "
        "Disconnect and delete runtime, then run all again." % free_gb)
'''

STEP2_MD = """\
## Step 2. Write the scripts and the frozen pool files

The scripts and the data below are exact copies of the files in the repository. The last cell of this step checks their hashes."""

STEP2_DIRS = '''
import os
for d in ("data/pool", "data/selection", "sel_out"):
    os.makedirs(d, exist_ok=True)
print("folders ready")
'''

STEP2_CHECK = r'''
EXPECTED = @EXPECTED@
bad = [p for p, h in EXPECTED.items() if not os.path.exists(p) or sha256_file(p) != h]
if bad:
    fail("these files were not written as expected: %s. Use Runtime -> Run all again; if it happens again, send me this message." % ", ".join(bad))
else:
    log("all %d files written and checked (hashes match)" % len(EXPECTED))
'''

STEP3_MD = """\
## Step 3. Install Phenix

Needed once per session (about 15 to 25 minutes). Two boxes ask for the user name and the password from the Phenix email.
If Phenix is already installed in this session, this step is skipped."""
STEP3 = r'''
PHENIX_URL = "@PHENIX_URL@"
PHENIX_VERSION = "@PHENIX_VERSION@"
INSTALLER = "@INSTALLER@"
PREFIX = "/content/phenix-" + PHENIX_VERSION if IN_COLAB else os.path.join(WORK, "phenix-" + PHENIX_VERSION)
MIN_INSTALLER_MB = 50


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
    r2 = subprocess.run(["bash", "-c", 'source "$0" && command -v phenix.map_model_cc', env_script], capture_output=True, text=True, errors="replace")
    if r2.returncode == 0:
        return True, "phenix.map_model_cc found at %s (phenix.version said: %s)" % (r2.stdout.strip(), text or "nothing")
    return False, text or "phenix.version did not run"


def download_installer():
    import requests
    if os.path.exists(INSTALLER) and os.path.getsize(INSTALLER) >= MIN_INSTALLER_MB * 1e6:
        log("%s is already here (%d MB)" % (INSTALLER, os.path.getsize(INSTALLER) // 1000000))
        return True
    user = secret("PHENIX_USER", "Phenix user name (from the email), then Enter: ", hidden=False)
    password = secret("PHENIX_PASSWORD", "Phenix password (from the email), then Enter: ")
    if len(password) >= 6:
        SECRETS.append(password)
    last = ""
    for attempt in range(1, 4):
        try:
            with requests.get(PHENIX_URL, auth=(user, password), stream=True, timeout=(30, 120)) as r:
                if r.status_code == 401:
                    fail("the Phenix server did not accept the user name and password (HTTP 401). The password changes on the 1st of every "
                         "month, so use the one from the newest email, and type both exactly as they appear. Then use Runtime -> Run all again.")
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
                     "user name and password. As a last resort, download the file in your browser from the same link and upload it into the "
                     "folder 'work' with the folder icon on the left, then use Runtime -> Run all again." % done)
                os.remove(INSTALLER + ".part")
                return False
            os.replace(INSTALLER + ".part", INSTALLER)
            return True
        except (requests.RequestException, IOError) as e:
            last = last or str(e)[:150]
            log("download attempt %d of 3 failed: %s" % (attempt, last))
            time.sleep(10 * attempt)
            last = ""
    fail("the Phenix installer could not be downloaded after 3 attempts. Use Runtime -> Run all again in a few minutes.")
    return False


def install_phenix():
    env = find_phenix_env()
    if env:
        ok, text = phenix_works(env)
        if ok:
            log("Phenix is already installed: %s (%s)" % (text, env))
            STATE.update(phenix_env=env, phenix_version=text)
            return
    if not download_installer():
        return
    STATE["installer_sha256"] = sha256_file(INSTALLER)
    log("installer sha256: %s" % STATE["installer_sha256"])
    tmp = os.path.join(os.path.dirname(PREFIX), "tmp_phenix_install")
    os.makedirs(tmp, exist_ok=True)
    shutil.rmtree(PREFIX, ignore_errors=True)
    log("installing Phenix (15 to 25 minutes; a line is printed now and then) ...")
    t0 = time.time()
    rc, last = run(["bash", INSTALLER, "-b", "-p", PREFIX], env={"TMPDIR": tmp}, every=60)
    shutil.rmtree(tmp, ignore_errors=True)
    if rc != 0:
        log("\n".join(last[-25:]))
        fail("the Phenix installer stopped with exit code %d (the last lines are above). If they say 'No space left', disconnect and delete the "
             "runtime and start again; otherwise send me the zip." % rc)
        return
    env = find_phenix_env()
    ok, text = phenix_works(env) if env else (False, "phenix_env.sh was not found under %s" % PREFIX)
    if not ok:
        fail("Phenix was installed but does not start: %s" % text)
        return
    STATE.update(phenix_env=env, phenix_version=text)
    log("Phenix installed in %d minutes: %s" % ((time.time() - t0) // 60, text))
    os.remove(INSTALLER)


if STATE["ok"]:
    install_phenix()
else:
    log("skipped: an earlier step stopped")
'''

STEP4_MD = """\
## Step 4. Test Phenix on two entries

Before the long run, Phenix computes CC_mask for the first two entries and shows what it printed, so a problem shows up in minutes, not hours.
Phenix 2.2.1 has retired `phenix.map_model_cc`, so the script tries the ways of calling Phenix one after the other and keeps the first one that works
for all later entries. If none gives a number, the run stops here and the zip then also holds Phenix's own help texts."""
STEP4 = r'''
SEL = "sel_out"
BATCH = @BATCH@
MAX_ROUNDS = 12


def cc_rows():
    import csv
    path = os.path.join(SEL, "cc_mask.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_todo():
    import csv
    with open(os.path.join(SEL, "cc_todo.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_round():
    cmd = [sys.executable, "-u", "select_entries.py", "select", "--pool", "data/pool", "--out", SEL,
           "--clusters", "data/selection/candidate_clusters.csv", "--entries", SEL + "/entries_2026.csv", "--batch", str(BATCH)]
    if os.path.exists(os.path.join(SEL, "cc_mask.csv")):
        cmd += ["--cc", SEL + "/cc_mask.csv"]
    return run(cmd)[0]


def compute(limit=0, show=False):
    cmd = [sys.executable, "-u", "compute_cc_mask.py", "--todo", SEL + "/cc_todo.csv", "--out", SEL + "/cc_mask.csv",
           "--phenix-env", STATE["phenix_env"], "--logs", SEL + "/cc_logs", "--work", "cc_work"]
    if limit:
        cmd += ["--limit", str(limit)]
    if show:
        cmd.append("--show-output")
    return run(cmd)[0]


def diagnose():
    run([sys.executable, "-u", "compute_cc_mask.py", "--diagnose", "--out", SEL + "/cc_mask.csv", "--phenix-env", STATE["phenix_env"], "--work", "cc_work"])


def drop_rows(ids):
    """Takes rows out of cc_mask.csv and keeps them in cc_mask_FAILED_TEST.csv: a failure of Phenix itself is not a result of an entry."""
    import csv
    path = os.path.join(SEL, "cc_mask.csv")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    keep = [r for r in rows if r["pdb_id"] not in ids]
    with open(os.path.join(SEL, "cc_mask_FAILED_TEST.csv"), "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        if f.tell() == 0:
            w.writeheader()
        w.writerows([r for r in rows if r["pdb_id"] in ids])
    if not keep:
        os.remove(path)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(keep)


def test_two_entries():
    log("=== first round of the selection rule ===")
    rc = select_round()
    if rc == 0:
        log("the list is already final")
        return
    if rc != 3:
        fail("select_entries.py stopped with exit code %d (see the lines above)" % rc)
        return
    done_before = {r["pdb_id"] for r in cc_rows()}
    log("\n=== test: CC_mask for the first two entries, with Phenix's own output ===")
    rc = compute(limit=2, show=True)
    if rc == 3:
        fail("Phenix or the disk stopped the test (see the lines above)")
        return
    new = [r for r in cc_rows() if r["pdb_id"] not in done_before]
    if not new:
        fail("the two test entries could not be downloaded (see the lines above). Use Runtime -> Run all again in a few minutes.")
        return
    if not any(r["status"] == "ok" for r in new):
        drop_rows({r["pdb_id"] for r in new if not r["note"].startswith("download:")})
        diagnose()
        fail("Phenix gave no CC_mask for either of the two test entries (reasons: %s). Send me the zip: its folder cc_logs has the full Phenix output, "
             "and diagnostics.txt has Phenix's help texts." % "; ".join(dict.fromkeys(r["note"][:160] for r in new)))
        return
    log("\nThe test worked: %s" % ", ".join("%s CC_mask %s [%s]" % (r["pdb_id"], r["cc_mask"] or r["status"], r["variant"]) for r in new))


if STATE["ok"]:
    test_two_entries()
else:
    log("skipped: an earlier step stopped")
'''

STEP5_MD = """\
## Step 5. Compute CC_mask and select, round by round

Each round applies the selection rule to the results so far, asks for the CC_mask of the next entries, and computes them.
It repeats until the rule needs nothing more. The result does not depend on how many rounds it takes. This is the long step."""
STEP5 = r'''
def selection_loop():
    for round_no in range(1, MAX_ROUNDS + 1):
        log("\n=== round %d: apply the selection rule ===" % round_no)
        rc = select_round()
        STATE["rounds"] = round_no
        if rc == 0:
            STATE["final"] = True
            return
        if rc != 3:
            fail("select_entries.py stopped with exit code %d (see the lines above)" % rc)
            return
        have = {r["pdb_id"] for r in cc_rows()}
        todo = [r for r in read_todo() if r["pdb_id"] not in have]
        log("%d entries need a CC_mask now" % len(todo))
        before = len(have)
        rc = compute()
        if rc == 3:
            fail("Phenix or the disk stopped the CC_mask computation (see the lines above)")
            return
        log("elapsed since the start: %d minutes" % ((time.time() - STATE["t0"]) // 60))
        if len(cc_rows()) == before:
            fail("no CC_mask could be computed in this round (download problems?). Use Runtime -> Run all again in a few minutes: it continues where it stopped.")
            return
    fail("the selection did not finish within %d rounds" % MAX_ROUNDS)


def show_result():
    import csv
    path = os.path.join(SEL, "entries_2026.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log("\nSELECTED ENTRIES (%d)" % len(rows))
    log("%-5s %-10s %-12s %5s %9s %7s %9s %7s" % ("PDB", "EMDB", "stratum", "res A", "residues", "CC_mask", "clus.size", "map MB"))
    for r in rows:
        log("%-5s %-10s %-12s %5s %9s %7s %9s %7s" % (r["pdb_id"], r["emdb_id"], r["stratum"], r["resolution_A"], r["modeled_residues"], r["cc_mask"],
                                                     r["cluster_size"], r["map_size_mb"]))
    rows_cc = cc_rows()
    high = sum(1 for r in rows_cc if r["status"] == "ok" and float(r["cc_mask"]) >= 0.70)
    low = sum(1 for r in rows_cc if r["status"] == "ok" and float(r["cc_mask"]) < 0.70)
    log("\nCC_mask was computed for %d entries: %d at 0.70 or more, %d below 0.70, %d without a number (Phenix failed)"
        % (len(rows_cc), high, low, len(rows_cc) - high - low))


if STATE["ok"]:
    selection_loop()
    if STATE["final"]:
        show_result()
else:
    log("skipped: an earlier step stopped")
'''

STEP6_MD = """\
## Step 6. Novelty check

For each chosen entry, how similar is its protein to proteins in the PDB before 2026? (MMseqs2, decision 9.) This step needs a few minutes.
If it fails, the selection above is still valid and the zip is still made."""
STEP6 = r'''
MMSEQS_VERSION = "@MMSEQS_VERSION@"
MMSEQS_URL = "@MMSEQS_URL@"
MMSEQS_SHA256 = @MMSEQS_SHA256@
NOVELTY_EXTRA = []


def get_mmseqs():
    import tarfile, urllib.request
    arch = "avx2"
    try:
        with open("/proc/cpuinfo") as f:
            arch = "avx2" if " avx2" in f.read() else "sse41"
    except OSError:
        pass
    exe = os.path.abspath("mmseqs/bin/mmseqs")
    if not os.path.exists(exe):
        url, tgz = MMSEQS_URL.replace("@ARCH@", arch), "mmseqs.tar.gz"
        urllib.request.urlretrieve(url, tgz)
        if sha256_file(tgz) != MMSEQS_SHA256[arch]:
            raise RuntimeError("the MMseqs2 download has an unexpected hash (%s)" % sha256_file(tgz))
        with tarfile.open(tgz) as t:
            t.extractall(".", filter="data") if sys.version_info >= (3, 12) else t.extractall(".")
        os.remove(tgz)
    r = subprocess.run([exe, "version"], capture_output=True, text=True)
    version = (r.stdout or r.stderr).strip()
    if version != MMSEQS_VERSION:
        raise RuntimeError("MMseqs2 reports version %s, expected %s" % (version, MMSEQS_VERSION))
    STATE["mmseqs_version"] = version
    return exe


def novelty():
    exe = get_mmseqs()
    cmd = [sys.executable, "-u", "novelty_check.py", "--entries", SEL + "/entries_2026.csv", "--fasta", "data/pool/pool_candidate_sequences.fasta",
           "--cryoem-ids", "data/pool/pool_ids_all_cryoem.txt", "--out", SEL + "/novelty", "--mmseqs", exe,
           "--threads", str(max(1, min(4, os.cpu_count() or 1)))] + NOVELTY_EXTRA
    rc, last = run(cmd)
    if rc != 0:
        raise RuntimeError("novelty_check.py stopped with exit code %d (see the lines above)" % rc)


if STATE["ok"] and STATE["final"]:
    try:
        novelty()
        STATE["novelty"] = "done"
    except Exception as e:
        STATE["novelty"] = "failed: %s" % e
        log("\nNOVELTY CHECK DID NOT FINISH: %s\nThe selection is not affected. Send me the zip anyway; I will run this step another way." % e)
else:
    log("skipped: the selection is not final")
'''

STEP7_MD = """\
## Step 7. Pack the results

Always run this last cell, also after a problem or after stopping early. It packs what exists into one zip and downloads it.
The Phenix password is never written to the zip; the cell checks every file for it before packing."""
STEP7 = r'''
def pack():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")
    name = "selection_result_%s%s" % (stamp, "" if STATE["final"] and STATE["ok"] else "_INCOMPLETE")
    info = {"notebook": "select_colab", "started_utc": STATE["started"], "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_minutes": round((time.time() - STATE["t0"]) / 60), "final": STATE["final"], "rounds": STATE["rounds"], "problem": STATE["problem"],
            "python": platform.python_version(), "platform": platform.platform(), "in_colab": IN_COLAB, "phenix_installer": INSTALLER,
            "phenix_installer_sha256": STATE["installer_sha256"], "phenix_version_output": STATE["phenix_version"],
            "mmseqs_version": STATE["mmseqs_version"], "novelty": STATE["novelty"], "batch": BATCH, "input_sha256": EXPECTED,
            "cc_mask_rows": len(cc_rows()), "cc_mask_ok": sum(1 for r in cc_rows() if r["status"] == "ok")}
    with open("run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    wanted = ["run.log", "run_info.json"]
    for f in ("entries_2026.csv", "selection_log.csv", "selection_info.json", "cc_todo.csv", "cc_mask.csv", "cc_mask_FAILED_TEST.csv", "diagnostics.txt"):
        wanted.append(os.path.join(SEL, f))
    for root in (os.path.join(SEL, "cc_logs"), os.path.join(SEL, "novelty")):
        for dirpath, _, files in os.walk(root):
            wanted += [os.path.join(dirpath, f) for f in sorted(files)]
    zip_path = name + ".zip"
    left_out = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in wanted:
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                data = f.read()
            if any(s and s.encode() in data for s in SECRETS):
                left_out.append(path)
                continue
            arc = os.path.relpath(path, SEL) if path.startswith(SEL + os.sep) else path
            z.write(path, name + "/" + arc)
        n = len(z.namelist())
    log("zip written: %s (%d files)" % (zip_path, n))
    if left_out:
        log("left out of the zip because they contained the password: %s" % ", ".join(left_out))
    return zip_path


zip_path = pack()
if STATE["ok"] and STATE["final"]:
    log("\nRESULT: OK. The entry list is final. Send me %s." % zip_path)
elif STATE["ok"]:
    log("\nRESULT: NOT FINISHED. Send me %s anyway; it shows how far the run got." % zip_path)
else:
    log("\nRESULT: STOPPED (%s). Send me %s." % (STATE["problem"], zip_path))
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    log("(This is not Colab, or the download did not start. Download %s from the folder 'work' on the left.)" % zip_path)
'''


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text if text.endswith("\n") else text + "\n"


def build(out_path, pool_dir=None, clusters_path=None, script_dir=HERE, phenix_url=PHENIX_URL):
    pool_dir = pool_dir or os.path.join(ROOT, "data", "pool")
    clusters_path = clusters_path or os.path.join(ROOT, "data", "selection", "candidate_clusters.csv")
    m = re.search(r"version=([^&]+)&file=([^&]+)$", phenix_url)
    if not m:
        raise ValueError("the Phenix link must end in ?version=...&file=...: %s" % phenix_url)
    version, installer = m.group(1), m.group(2)
    files = []                                                        # (path in the notebook, text)
    for s in SCRIPTS:
        text = read(os.path.join(script_dir, s))
        compile(text, s, "exec")                                      # a syntax error shows up here, not in Colab
        files.append((s, text))
    for target, kind, name in DATA:
        src = clusters_path if kind == "clusters" else os.path.join(pool_dir, name)
        files.append((target, read(src)))
    expected = {p: sha256_text(t) for p, t in files}

    def fill(text):
        return (text.replace("@PHENIX_URL@", phenix_url).replace("@PHENIX_VERSION@", version).replace("@INSTALLER@", installer)
                .replace("@BATCH@", str(BATCH)).replace("@MMSEQS_VERSION@", MMSEQS_VERSION).replace("@MMSEQS_URL@", MMSEQS_URL)
                .replace("@MMSEQS_SHA256@", repr(MMSEQS_SHA256)).replace("@EXPECTED@", repr(expected)))

    nb = new_notebook()
    nb.metadata = {"colab": {"name": "select_colab.ipynb", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    cells = [("intro", new_markdown_cell(INTRO)), ("step1-title", new_markdown_cell(STEP1_MD)), ("step1", new_code_cell(fill(STEP1).strip("\n") + "\n")),
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
    nbformat.validate(nb)
    nbformat.write(nb, out_path)
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, "select_colab.ipynb"))
    p.add_argument("--phenix-url", default=PHENIX_URL)
    args = p.parse_args(argv)
    expected = build(args.out, phenix_url=args.phenix_url)
    print("wrote %s with %d embedded files" % (args.out, len(expected)))
    for path, digest in expected.items():
        print("  %s  %s" % (digest[:16], path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
