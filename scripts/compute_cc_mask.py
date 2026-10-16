#!/usr/bin/env python3
"""Phase 3b: map-model correlation (CC_mask) of the deposited model against the deposited map, with Phenix.

Decision 4 of docs/SPEC.md: an entry needs CC_mask >= 0.70 from phenix.map_model_cc, with identical settings for
every entry. The command is

    phenix.map_model_cc <model.cif> <map.map> resolution=<resolution_A of the entry> ignore_symmetry_conflicts=True --force

Phenix 2.2.1 has retired that program: without --force it stops with "phenix.map_model_cc and phenix.model_map_cc are
replaced by phenix.map_correlations". So the script knows a few ways to call Phenix (VARIANTS below). For the first
entries it tries them one after the other until one gives a CC_mask; from then on that one variant is used for every entry
(it is written in the column "variant" of cc_mask.csv, and a later run reads it from there). For every entry listed in
cc_todo.csv the script downloads the model (RCSB) and the map (EMDB, .map.gz), runs Phenix, reads CC_mask from its
output, deletes the big files and appends one row to cc_mask.csv. The files of the next entries are downloaded while
Phenix works on the current one. Entries that already have a row are skipped, so the script can be stopped and
started again at any time.

A row has status "ok" (Phenix finished and printed a CC_mask) or "failed" (Phenix ran but gave no usable CC_mask,
timed out or was killed, or the file does not exist on the server, HTTP 404). select_entries.py counts "failed" as a
rejection. A temporary problem (download error, server busy) is not a result: no row is written and the entry is tried
again next time.

Exit codes: 0 every entry of the todo list has a row; 2 some entries have none (temporary problems);
3 Phenix cannot be run at all or the disk is full (nothing is written for the entry that was being processed).

    python compute_cc_mask.py --todo cc_todo.csv --out cc_mask.csv --phenix-env /path/to/phenix_env.sh
    python compute_cc_mask.py --diagnose --out cc_mask.csv --phenix-env /path/to/phenix_env.sh    (writes diagnostics.txt)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zlib

import requests

SCRIPT_VERSION = "1.1"
TIMEOUT_S = 1800
AHEAD = 2                      # entries whose files are downloaded while Phenix works on the current one
COLUMNS = ["pdb_id", "emdb_id", "status", "cc_mask", "resolution_used", "seconds", "variant", "phenix_version", "note"]
CC_RE = re.compile(r"\bCC_mask\b[\s:=|,()\[\]]*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.I)
UA = "mica-benchmark-2026 (academic benchmark; contact via GitHub Zohaib-Bioinfo)"
MIRROR = "https://files.wwpdb.org/pub/pdb/data/structures/all/mmCIF/%s.cif.gz"
STOP = threading.Event()       # set when the script ends early: running downloads give up

# (name, command with {model} {map} {res} filled in, only tried after Phenix complained about ignore_symmetry_conflicts)
VARIANTS = (
    ("map_model_cc --force",
     ["phenix.map_model_cc", "{model}", "{map}", "resolution={res}", "ignore_symmetry_conflicts=True", "--force"], False),
    ("map_model_cc --force first",
     ["phenix.map_model_cc", "--force", "{model}", "{map}", "resolution={res}", "ignore_symmetry_conflicts=True"], False),
    ("map_correlations",
     ["phenix.map_correlations", "{model}", "{map}", "resolution={res}", "ignore_symmetry_conflicts=True"], False),
    ("map_correlations full names",
     ["phenix.map_correlations", "input_files.model={model}", "input_files.map_in_1={map}", "map_model_cc.resolution={res}",
      "map_model_cc.ignore_symmetry_conflicts=True"], False),
    ("map_model_cc --force, no symmetry option",
     ["phenix.map_model_cc", "{model}", "{map}", "resolution={res}", "--force"], True),
    ("map_correlations, no symmetry option",
     ["phenix.map_correlations", "{model}", "{map}", "resolution={res}"], True),
)
TEXT_SUFFIXES = (".log", ".txt", ".out", ".json", ".csv", ".eff")


class Transient(Exception):
    """A problem that says nothing about the entry: try again later."""


class Permanent(Exception):
    """The file does not exist on the server."""


class Infrastructure(Exception):
    """Phenix cannot run or the disk is full: stop everything."""


# --------------------------------------------------------------------------------------
# Reading the output of Phenix
# --------------------------------------------------------------------------------------
def parse_cc_mask(text):
    """The first CC_mask value in the output of Phenix, and every line that carries one."""
    values, lines = [], []
    for line in text.splitlines():
        m = CC_RE.search(line)
        if m:
            values.append(float(m.group(1)))
            lines.append(line.strip())
    return (values[0] if values else None), lines


def short_error(text):
    """One line that says what went wrong."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("Sorry") or "Traceback" in ln:
            idx = lines.index(ln)
            return " | ".join(lines[idx: idx + 2])[:300]
    return (lines[-1] if lines else "no output")[:300]


def complains_about_option(text):
    """True if Phenix says it does not know the parameter ignore_symmetry_conflicts."""
    words = ("unknown", "unrecognized", "not recognized", "ambiguous", "invalid", "no such", "unexpected", "sorry")
    return any("ignore_symmetry_conflicts" in ln.lower() and any(w in ln.lower() for w in words) for ln in text.splitlines())


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------
def download(url, dest, retries=3, timeout=180):
    """Downloads url to dest. Raises Permanent for HTTP 404 and Transient for everything else."""
    last = "unknown error"
    for attempt in range(1, retries + 1):
        part = dest + ".part"
        try:
            with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": UA}) as r:
                if r.status_code == 404:
                    raise Permanent("HTTP 404 for %s" % url)
                if r.status_code != 200:
                    raise IOError("HTTP %d" % r.status_code)
                with open(part, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        if STOP.is_set():
                            raise Transient("stopped")
                        f.write(block)
            os.replace(part, dest)
            return
        except (Permanent, Transient):
            raise
        except OSError as e:
            if "No space left" in str(e):
                raise Infrastructure("disk full while downloading %s" % url)
            last = str(e)[:200]
        except requests.RequestException as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:150])
        finally:
            if os.path.exists(part):
                os.remove(part)
        if attempt < retries:
            time.sleep(2 * attempt)
    raise Transient("download of %s failed after %d attempts: %s" % (url, retries, last))


def gunzip_if_needed(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic != b"\x1f\x8b":
        return
    tmp = path + ".raw"
    try:
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    except (OSError, EOFError, zlib.error) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if "No space left" in str(e):
            raise Infrastructure("disk full while unpacking %s" % path)
        raise Transient("cannot unpack %s: %s" % (path, e))
    os.replace(tmp, path)


def fetch_model(row, workdir):
    dest = os.path.join(workdir, "%s.cif" % row["pdb_id"])
    try:
        download(row["cif_url"], dest)
    except Permanent:
        mirror = MIRROR % row["pdb_id"].lower()
        download(mirror, dest + ".gz")                      # the same file from the wwPDB archive
        os.replace(dest + ".gz", dest)
    gunzip_if_needed(dest)
    with open(dest, "rb") as f:
        head = f.read(200)
    if not head.lstrip().startswith(b"data_"):
        raise Transient("%s is not an mmCIF file (starts with %r)" % (row["cif_url"], head[:40]))
    return dest


def fetch_map(row, workdir):
    dest = os.path.join(workdir, "%s.map" % row["emdb_id"].replace("EMD-", "emd_"))
    download(row["map_url"], dest)
    gunzip_if_needed(dest)
    with open(dest, "rb") as f:
        head = f.read(1024)
    if len(head) < 1024 or head[208:212] != b"MAP ":
        raise Transient("%s is not an MRC/CCP4 map (header %r)" % (row["map_url"], head[:16]))
    return dest


def fetch_files(row, workdir):
    os.makedirs(workdir, exist_ok=True)
    return fetch_model(row, workdir), fetch_map(row, workdir)


# --------------------------------------------------------------------------------------
# Phenix
# --------------------------------------------------------------------------------------
def bash_phenix(env_script, args, timeout, cwd):
    """Runs a Phenix command with its environment loaded. Returns (return code, output) or (None, output) on timeout."""
    cmd = ["bash", "-c", 'source "$0" && exec "$@"', env_script] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, errors="replace")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")
        out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        return None, out
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def phenix_version(env_script, workdir):
    rc, out = bash_phenix(env_script, ["phenix.version"], 180, workdir)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    hit = [ln for ln in lines if re.search(r"\d+\.\d+", ln)]
    return (hit[0] if hit else (lines[0] if lines else "unknown"))[:80], rc


def new_text_files(workdir, before, limit=2000000):
    """Names and text of the small text files that Phenix wrote into workdir (some versions print nothing)."""
    names, chunks = [], []
    for name in sorted(set(os.listdir(workdir)) - before):
        path = os.path.join(workdir, name)
        if os.path.isfile(path) and name.lower().endswith(TEXT_SUFFIXES) and os.path.getsize(path) <= limit:
            with open(path, "r", errors="replace") as f:
                chunks.append(f.read())
            names.append(name)
    return names, "\n".join(chunks)


def run_phenix(env_script, model, mapfile, resolution, workdir, timeout, locked=None):
    """Returns (status, cc, variant, note, log text).

    Without `locked` the variants are tried one after the other until one gives a CC_mask. `locked` is the name of the
    variant that has already worked for an earlier entry: it is then the only one used, so that every entry gets the same
    settings and a failure is a result of the entry, not of the way Phenix was called."""
    log, reasons, complained, not_found = [], [], False, 0
    variants = [v for v in VARIANTS if v[0] == locked] if locked else list(VARIANTS)
    tried = []
    for name, template, after_complaint in variants:
        if after_complaint and not (complained or locked):
            continue
        tried.append(name)
        args = [a.format(model=os.path.basename(model), map=os.path.basename(mapfile), res=resolution) for a in template]
        before = set(os.listdir(workdir))
        rc, out = bash_phenix(env_script, args, timeout, workdir)
        entry = "$ %s\n%s\n[exit code %s]\n" % (" ".join(args), out[-200000:], "timeout" if rc is None else rc)
        if "No space left on device" in out:
            raise Infrastructure("disk full while running Phenix")
        if rc is None:
            log.append(entry + "=> %s: timeout after %d s\n" % (name, timeout))
            return "failed", None, name, "timeout after %d s" % timeout, "".join(log)     # another variant would take as long again
        if rc < 0 or rc == 137:
            note = "killed by signal %d (probably out of memory)" % (-rc if rc < 0 else 9)
            log.append(entry + "=> %s: %s\n" % (name, note))
            return "failed", None, name, note, "".join(log)
        value, lines = parse_cc_mask(out)
        if value is None and rc == 0:
            names, text = new_text_files(workdir, before)
            value, lines = parse_cc_mask(text)
            if names:
                entry += "[files written by Phenix: %s]\n" % ", ".join(names)
        if rc == 0 and value is not None and -1.0 <= value <= 1.0:
            note = "" if len(lines) == 1 else "%d CC_mask lines in the output, the first one is used" % len(lines)
            log.append(entry + "=> %s: CC_mask %s\n" % (name, value))
            return "ok", value, name, note, "".join(log)
        if rc == 127:
            not_found += 1
            reason = "program not found: %s" % short_error(out)
        elif rc != 0:
            reason = "phenix exited with %d: %s%s" % (rc, short_error(out), "" if value is None else " (its output does have CC_mask %s)" % value)
            if complains_about_option(out):
                complained = True
        elif value is None:
            reason = "no CC_mask in the output"
        else:
            reason = "CC_mask %s is outside -1..1" % value
        log.append(entry + "=> %s: %s\n" % (name, reason))
        reasons.append(reason)
    if tried and not_found == len(tried):
        raise Infrastructure("none of the Phenix programs could be started (%s): %s" % (", ".join(tried), reasons[-1]))
    last = tried[-1] if tried else (locked or VARIANTS[0][0])
    if len(reasons) == 1:
        note = reasons[0]
    else:
        note = "; ".join("%s: %s" % (n, r[:110]) for n, r in zip(tried, reasons))[:600]
    return "failed", None, last, note, "".join(log)


def diagnose(env_script, out_path, workdir):
    """Writes what Phenix says about its own programs, for the case that no variant works."""
    os.makedirs(workdir, exist_ok=True)
    commands = [["phenix.version"], ["phenix.map_correlations", "--help"], ["phenix.map_correlations"],
                ["phenix.map_model_cc", "--force", "--help"], ["phenix.map_model_cc", "--force"],
                ["bash", "-c", 'ls "$PHENIX/phenix_bin" "$PHENIX/bin" 2>/dev/null | grep -i -E "map_|_cc|validation_cryo|mtriage" | sort -u']]
    with open(out_path, "w", encoding="utf-8") as f:
        for args in commands:
            rc, out = bash_phenix(env_script, args, 180, workdir)
            f.write("$ %s\n[exit code %s]\n%s\n\n" % (" ".join(args), "timeout" if rc is None else rc, "\n".join(out.splitlines()[:300])[:30000]))
    print("diagnostics written to %s" % out_path, flush=True)
    return 0


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def locked_variant(rows):
    """The variant that gave the first CC_mask of an earlier run, if there was one."""
    known = {v[0] for v in VARIANTS}
    for r in rows:
        if r.get("status") == "ok" and r.get("variant") in known:
            return r["variant"]
    return None


def process(row, args, workdir, version, files, locked):
    t0 = time.time()
    try:
        model, mapfile = files()
        waited = time.time() - t0
        status, cc, variant, note, log = run_phenix(args.phenix_env, model, mapfile, row["resolution_A"], workdir, args.timeout, locked)
    except Permanent as e:
        waited = time.time() - t0
        status, cc, variant, note, log = "failed", None, locked or "", "download: %s" % e, "%s\n" % e
    seconds = round(time.time() - t0)
    return {"pdb_id": row["pdb_id"], "emdb_id": row["emdb_id"], "status": status,
            "cc_mask": "" if cc is None else "%.4f" % cc, "resolution_used": row["resolution_A"], "seconds": seconds,
            "variant": variant, "phenix_version": version, "note": note}, log, round(waited)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--todo", default=None, help="cc_todo.csv from select_entries.py")
    p.add_argument("--out", required=True, help="cc_mask.csv (rows are appended)")
    p.add_argument("--phenix-env", required=True, help="path of phenix_env.sh")
    p.add_argument("--logs", default=None, help="folder for one log per entry (default: cc_logs next to --out)")
    p.add_argument("--work", default=None, help="scratch folder (default: cc_work next to --out)")
    p.add_argument("--limit", type=int, default=0, help="only the first N entries that still need a result")
    p.add_argument("--timeout", type=int, default=TIMEOUT_S, help="seconds allowed for one Phenix run")
    p.add_argument("--show-output", action="store_true", help="print the commands, exit codes and CC lines of the Phenix output")
    p.add_argument("--diagnose", action="store_true", help="write diagnostics.txt next to --out (Phenix help texts) and stop")
    args = p.parse_args(argv)
    base = os.path.dirname(os.path.abspath(args.out))
    logs = args.logs or os.path.join(base, "cc_logs")
    work = args.work or os.path.join(base, "cc_work")
    if args.diagnose:
        if not os.path.exists(args.phenix_env):
            print("ERROR: %s does not exist" % args.phenix_env, file=sys.stderr)
            return 3
        return diagnose(args.phenix_env, os.path.join(base, "diagnostics.txt"), os.path.join(work, "diagnose"))
    if not args.todo:
        p.error("--todo is required")
    os.makedirs(logs, exist_ok=True)
    todo = read_csv(args.todo)
    have = {r["pdb_id"] for r in read_csv(args.out)}
    pending = [r for r in todo if r["pdb_id"] not in have]
    if args.limit:
        pending = pending[: args.limit]
    print("compute_cc_mask %s: %d entries in the list, %d already done, %d to do now" % (SCRIPT_VERSION, len(todo), len(have & {r["pdb_id"] for r in todo}), len(pending)), flush=True)
    if not pending:
        return 0
    os.makedirs(work, exist_ok=True)
    version, rc = phenix_version(args.phenix_env, work)
    if rc == 127 or not os.path.exists(args.phenix_env):
        print("ERROR: Phenix does not start (exit code %s). Is %s the right file?" % (rc, args.phenix_env), file=sys.stderr)
        return 3
    print("Phenix: %s" % version, flush=True)
    locked = locked_variant(read_csv(args.out))
    if locked:
        print("variant of the earlier results (used for every entry): %s" % locked, flush=True)
    n_ok = n_failed = n_skipped = 0
    STOP.clear()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, AHEAD))
    futures = {}
    try:
        for i, row in enumerate(pending, 1):
            for nxt in pending[i - 1: i + AHEAD]:           # this entry and the next ones: their files are downloaded in the background
                if nxt["pdb_id"] not in futures:
                    futures[nxt["pdb_id"]] = pool.submit(fetch_files, nxt, os.path.join(work, nxt["pdb_id"]))
            files = futures.pop(row["pdb_id"]).result
            entry_dir = os.path.join(work, row["pdb_id"])
            try:
                result, log, waited = process(row, args, entry_dir, version, files, locked)
            except Transient as e:
                n_skipped += 1
                print("[%d/%d] %s  NOT DONE (temporary problem, will be tried again): %s" % (i, len(pending), row["pdb_id"], e), flush=True)
                continue
            except Infrastructure as e:
                print("ERROR: %s" % e, file=sys.stderr)
                return 3
            finally:
                shutil.rmtree(entry_dir, ignore_errors=True)
            with open(os.path.join(logs, "%s.log" % row["pdb_id"]), "w", encoding="utf-8") as f:
                f.write("entry %s (%s), stratum %s, resolution %s A\nscript compute_cc_mask %s, %s\nresult: %s\n\n%s"
                        % (row["pdb_id"], row["emdb_id"], row.get("stratum", ""), row["resolution_A"], SCRIPT_VERSION, version,
                           {k: result[k] for k in ("status", "cc_mask", "variant", "note")}, log))
            append_row(args.out, result)
            if result["status"] == "ok":
                n_ok += 1
                locked = locked or result["variant"]
            else:
                n_failed += 1
            print("[%d/%d] %s  %s  %s A  map %s MB: %s  (%d s, of which %d s waiting for the files)%s" % (
                i, len(pending), row["pdb_id"], row.get("stratum", ""), row["resolution_A"], row.get("map_size_mb", "?"),
                ("CC_mask %s [%s]" % (result["cc_mask"], result["variant"])) if result["status"] == "ok" else "FAILED: %s" % result["note"],
                result["seconds"], waited, "" if result["status"] != "ok" or not result["note"] else "  [" + result["note"] + "]"), flush=True)
            if args.show_output:
                for ln in log.splitlines():
                    if "CC_mask" in ln or ln.startswith("$") or ln.startswith("[exit") or ln.startswith("=>"):
                        print("    | " + ln[:200], flush=True)
    finally:
        STOP.set()                                              # running downloads give up, waiting ones are cancelled
        pool.shutdown(wait=True, cancel_futures=True)
        for pdb_id in futures:                                  # files of entries that were downloaded ahead but not used
            shutil.rmtree(os.path.join(work, pdb_id), ignore_errors=True)
    print("done: %d ok, %d failed (rejected), %d not done" % (n_ok, n_failed, n_skipped), flush=True)
    have = {r["pdb_id"] for r in read_csv(args.out)}
    return 0 if all(r["pdb_id"] in have for r in pending) else 2


if __name__ == "__main__":
    sys.exit(main())
