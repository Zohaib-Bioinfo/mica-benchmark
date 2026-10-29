#!/usr/bin/env python3
"""Tools that the Phase 4 notebooks share. They were added after the first MICA pilot, which showed three weaknesses:

1. a program that is silent for an hour (Phenix docking) looked like a frozen notebook: `Heartbeat` prints one line every few minutes
   (elapsed time, load, free RAM, the busiest processes, the last line of the program's own log);
2. the zip with the results was only written by the very last cell, so a stopped session left nothing: `pack` writes the zip in
   one step and the notebooks call it after every stage (a fixed name, replaced each time);
3. the last cell used variables of earlier cells and failed after the kernel was restarted: `recover` needs nothing from the notebook,
   it finds the work folder on disk and packs whatever is there.

Everything the zip should hold is copied into the work folder by the notebook, so that these functions only walk that folder.
Not packed: maps and other large data, files over 60 MB, the AlphaFold Server predictions of the user (they must stay private), and any file
that contains one of the secrets (the Phenix password).

    python scripts/nbtools.py --work <work folder> --out <folder for the zip> --notebook <name>      # the recovery path, by hand
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time
import zipfile

MAX_FILE = 60 * 1000 * 1000
SKIP_DIRS = {"AF3_results", "af3_upload", "af3_unzipped", "grids", "AF3_encodings", "temp_maps", "native", "downloads", "scripts", "data", "env", "__pycache__"}
SKIP_SUFFIXES = (".map", ".mrc", ".ccp4", ".gz", ".part", ".tmp", ".zip")
WORK_CANDIDATES = ("/kaggle/working/pilot", "/content/pilot")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def last_line(path, max_chars=160):
    """The last non-empty line of a text file, or "" if there is none or the file cannot be read."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    return lines[-1][:max_chars] if lines else ""


def machine_load():
    """(load average over 1 minute, free RAM in GB, text about the three busiest processes), each None or "" where it cannot be read."""
    load = free = None
    try:
        with open("/proc/loadavg") as f:
            load = float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    free = round(int(line.split()[1]) / 1048576, 1)
    except (OSError, ValueError, IndexError):
        pass
    top = ""
    try:
        r = subprocess.run(["ps", "-eo", "pcpu,etimes,comm", "--sort=-pcpu"], capture_output=True, text=True, timeout=10, errors="replace")
        rows = [ln.split(None, 2) for ln in r.stdout.splitlines()[1:4] if ln.strip()]
        top = ", ".join("%s %s%% (%ss)" % (c[2][:18], c[0], c[1]) for c in rows if len(c) == 3)
    except (OSError, subprocess.SubprocessError):
        pass
    return load, free, top


class Heartbeat(threading.Thread):
    """While a long program runs in the main thread, prints one line every `every` seconds through `log`. `watch` is a list of files whose last
    line is shown (the program's own log); `on_tick` is called after each line (the notebooks use it to refresh the partial zip)."""

    def __init__(self, label, log, watch=(), every=120, on_tick=None):
        super().__init__(daemon=True)
        self.label, self.log, self.watch, self.every, self.on_tick = label, log, list(watch), every, on_tick
        self.stop_flag, self.t0, self.ticks = threading.Event(), time.time(), 0

    def line(self):
        load, free, top = machine_load()
        minutes = (time.time() - self.t0) / 60
        parts = ["[%s] running for %.0f min" % (self.label, minutes) if minutes >= 1 else "[%s] running for %.0f s" % (self.label, time.time() - self.t0)]
        if load is not None:
            parts.append("load %.1f" % load)
        if free is not None:
            parts.append("free RAM %.1f GB" % free)
        if top:
            parts.append("busiest: " + top)
        for path in self.watch:
            if os.path.exists(path):
                age = time.time() - os.path.getmtime(path)
                parts.append("%s: %d bytes, changed %s ago, last line: %s" % (os.path.basename(path), os.path.getsize(path),
                                                                           "%.0f s" % age if age < 120 else "%.0f min" % (age / 60), last_line(path) or "(empty)"))
        return " | ".join(parts)

    def run(self):
        while not self.stop_flag.wait(self.every):
            try:
                self.log(self.line())
                self.ticks += 1
                if self.on_tick:
                    self.on_tick()
            except Exception as ex:                       # a failing heartbeat must never stop the program it watches
                try:
                    self.log("[%s] heartbeat problem: %s" % (self.label, str(ex)[:150]))
                except Exception:
                    pass

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=30)


def collect(work, skip_dirs=SKIP_DIRS, skip_suffixes=SKIP_SUFFIXES):
    """All files below `work` that may go into the zip, in a fixed order."""
    files = []
    for dirpath, dirs, names in os.walk(work):
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
        for n in sorted(names):
            if not n.endswith(skip_suffixes):
                files.append(os.path.join(dirpath, n))
    return files


def pack(work, zip_path, root_name, secrets=(), max_file=MAX_FILE, extra=()):
    """Writes `zip_path` in one step (a temporary file that replaces the old zip at the end, so a stop in the middle never leaves a broken zip).
    Returns (number of files, list of what was left out and why). `extra` are files outside `work` that go in as well."""
    wanted = collect(work) + [p for p in extra if os.path.isfile(p)]
    left_out, tmp = [], zip_path + ".tmp"
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for path in dict.fromkeys(wanted):
                try:
                    if os.path.getsize(path) > max_file:
                        left_out.append("%s (larger than %d MB)" % (os.path.relpath(path, work), max_file // 1000000))
                        continue
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError as ex:
                    left_out.append("%s (%s)" % (os.path.relpath(path, work), str(ex)[:60]))
                    continue
                if any(s and s.encode() in data for s in secrets):
                    left_out.append("%s (contained the password)" % os.path.relpath(path, work))
                    continue
                inside = os.path.relpath(path, work) if path.startswith(work + os.sep) else os.path.join("outside", path.lstrip(os.sep))
                z.writestr(root_name + "/" + inside, data)
            n = len(z.namelist())
        os.replace(tmp, zip_path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return n, left_out


def partial_zip_path(out_dir, notebook):
    return os.path.join(out_dir, "%s_partial.zip" % notebook)


def final_pack(work, out_dir, notebook, complete, secrets=(), extra=(), note=""):
    """The zip that is sent to me. complete: True (everything ran), False (something stopped) or None (unknown: recovered after a restart).
    The partial zip is removed once this one exists."""
    stamp = utc_now().strftime("%Y%m%d-%H%M")
    suffix = "" if complete else ("_INCOMPLETE" if complete is False else "_RECOVERED")
    name = "%s_result_%s%s" % (notebook, stamp, suffix)
    path = os.path.join(out_dir, name + ".zip")
    n, left_out = pack(work, path, name, secrets=secrets, extra=extra)
    try:
        os.remove(partial_zip_path(out_dir, notebook))
    except OSError:
        pass
    return path, n, left_out


def find_work(env=None):
    """The work folder of a notebook that ran in this session: the one with a run.log."""
    env = os.environ if env is None else env
    for cand in (env.get("MICA_WORK"),) + WORK_CANDIDATES + (os.path.abspath("pilot"),):
        if cand and os.path.isfile(os.path.join(cand, "run.log")):
            return cand
    return None


def known_secrets(env=None):
    """The Phenix password, if it can be read again (environment, Kaggle or Colab secrets), to check the files for it after a restart."""
    env = os.environ if env is None else env
    found = []
    if env.get("PHENIX_PASSWORD"):
        found.append(env["PHENIX_PASSWORD"].strip())
    for getter in ("kaggle", "colab"):
        try:
            if getter == "kaggle":
                from kaggle_secrets import UserSecretsClient
                v = UserSecretsClient().get_secret("PHENIX_PASSWORD")
            else:
                from google.colab import userdata
                v = userdata.get("PHENIX_PASSWORD")
            if v:
                found.append(v.strip())
        except Exception:
            pass
    return [s for s in dict.fromkeys(found) if len(s) >= 6]


def read_state(work):
    try:
        with open(os.path.join(work, "state.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def recover(notebook, out_dir=None, env=None, say=print):
    """The last step when nothing from the notebook is left in memory (kernel restarted, cell run alone, ...). Returns the zip path or None."""
    env = os.environ if env is None else env
    work = find_work(env)
    if not work:
        say("RECOVERY: no work folder with a run.log was found (looked in %s). Nothing to pack." % ", ".join(WORK_CANDIDATES))
        return None
    out_dir = out_dir or env.get("MICA_WORK") or os.path.dirname(work)
    state = read_state(work)
    info = {"notebook": notebook, "recovered": True, "recovered_utc": utc_now().isoformat(), "state_json_found": bool(state), "state": state}
    with open(os.path.join(work, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    path, n, left_out = final_pack(work, out_dir, notebook, complete=None, secrets=known_secrets(env))
    say("RECOVERY: zip written from the files that are on disk: %s (%d files, %d MB)" % (path, n, os.path.getsize(path) // 1000000))
    if left_out:
        say("left out of the zip: %s" % "; ".join(left_out))
    say("If a notebook cell stopped with an error, send me this zip and the text of the error.")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--notebook", required=True)
    args = p.parse_args(argv)
    path, n, left_out = final_pack(args.work, args.out, args.notebook, complete=None)
    print("%s (%d files)" % (path, n))
    for x in left_out:
        print("left out:", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
