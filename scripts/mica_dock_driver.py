#!/usr/bin/env python3
"""MICA's docking program (utils/dock_in_map.py) with ONE method replaced, for the benchmark on free machines.

Why. MICA's program 2 calls `phenix.dock_in_map ... nproc=<60% of the cores> quick=False ...` once per domain, with no time limit. On the 160^3 map of the pilot entry
26DK that call did not finish in 66 minutes (killed for lack of RAM), the quick search did not finish in 4.5 hours, and a search at lower resolution did not finish in 45
minutes; Phenix's other engine (`dock_with_mr=False`, Fourier convolution searches in boxes) docked the first domain in 12 minutes (LOG.md, 2026-10-24).
So the call has to be changed to make the benchmark possible at all, and a time limit has to be added, because a free session lasts at most 12 hours.

What it does. It imports MICA's own `utils/dock_in_map.py` (not modified), replaces `PhenixDockingProcessor.run_phenix_dock_in_map` by the function below and runs MICA's
`main()`. Everything else is MICA's: the order of the domains, the masking of the map after each docked domain, the combination of the docked domains, the
`nproc` formula, `dock_chains_individually=True`, the sequence file. A domain that cannot be docked is skipped, as in MICA.

The function tries a list of attempts (environment variable MICA_BENCH_DOCK_ATTEMPTS, a JSON list). Each attempt has a name, `set` (Phenix parameters that replace or
add to MICA's) and `timeout_s` (the attempt and all its processes are killed after that many seconds; null = no limit, as in MICA). The first attempt that gives exit
code 0 and a docked model ends the search. `skip_after_timeout: true` skips an attempt if an earlier attempt for the same domain ran into its time limit (a slower search
would not do better). Without the variable there is one attempt with MICA's own call and no limit. Every attempt is written as a row of `bench_docking.csv` in MICA's
docking log folder.

`search_resolution` (Angstrom; LOG.md, 2026-10-26): the whole docking, the search AND the rigid-body refinement of the placed model, runs at max(resolution of the map,
search_resolution). It is set through the one parameter that MICA's own call already passes (`resolution`), so that Phenix cannot reject it. (Pass 2, 2026-10-25, also set
`rigid_body_refinement_resolution` and `density_radius`; Phenix 2.2.1 rejected the first ("One True or False value expected") within 2 seconds in all 59 attempts of the first two
rungs.) A map whose resolution is not finer than `search_resolution` is left as it is.

    cd <MICA folder> && python scripts/mica_dock_driver.py -m <map> -c <contour> -r <resolution> -f <fasta> -a <AF3_results> --phenix_act <phenix_env.sh>
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import time

DEFAULT_ATTEMPTS = [{"name": "mica", "set": {}, "timeout_s": None}]
LOCAL_CC = re.compile(r"Wrote placed model with CC_mask \(local CC\)\s*=\s*(-?\d+(?:\.\d+)?)")
COLUMNS = ["model", "attempt", "setting", "settings", "timeout_s", "seconds", "rc", "timed_out", "skipped", "docked", "local_cc", "log"]


def load_attempts(env=None):
    text = (os.environ if env is None else env).get("MICA_BENCH_DOCK_ATTEMPTS", "")
    attempts = json.loads(text) if text.strip() else DEFAULT_ATTEMPTS
    if not attempts:
        raise ValueError("MICA_BENCH_DOCK_ATTEMPTS is an empty list")
    return attempts


def phenix_args(search_model, density_map, nproc, resolution, pdb_out, fasta_file, settings):
    """The arguments of MICA's call (utils/dock_in_map.py, run_phenix_dock_in_map), in MICA's order; `settings` replace or add parameters."""
    args = ['phenix.dock_in_map', f'search_model={search_model}', f'map_file={density_map}', f'nproc={nproc}', 'quick=False', f'resolution={resolution}',
            f'pdb_out={pdb_out}', 'dock_chains_individually=True', f'sequence={fasta_file}']
    for key, value in (settings or {}).items():
        args = [a for a in args if not a.startswith(key + "=")] + ["%s=%s" % (key, value)]
    return args


def derived_settings(attempt, resolution):
    """The Phenix parameter that follows from `search_resolution` of an attempt (see the top of this file): `resolution`, which MICA's own call passes already."""
    search = attempt.get("search_resolution")
    if not search or float(search) <= float(resolution):
        return {}
    return {"resolution": "%.1f" % float(search)}


def descendants(pid):
    """Every process below `pid` (children, grandchildren, ...), read from /proc."""
    children = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % name) as f:
                text = f.read()
            children.setdefault(int(text.rsplit(")", 1)[1].split()[1]), []).append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    found, todo = [], [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            found.append(child)
            todo.append(child)
    return found


def kill_tree(proc):
    """Kills the shell that runs Phenix and every process below it. They stay in the process group of this program on purpose: when the notebook stops this program at the
    end of its time (a kill of the whole group), Phenix goes with it instead of running on in the background."""
    for _ in range(3):
        pids = descendants(proc.pid)
        for pid in [proc.pid] + pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if not pids:
            break
        time.sleep(0.2)
    proc.wait()


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def run_phenix_dock_in_map(self, density_map, atomic_model, resolution, save_file, fasta_file):
    """Replaces PhenixDockingProcessor.run_phenix_dock_in_map: same arguments, same return value (True if the domain was docked)."""
    model_name = os.path.basename(atomic_model)
    cpu_cores = multiprocessing.cpu_count()
    optimal_nproc = min(max(int(cpu_cores * 0.6), 1), 48)
    attempts = load_attempts()
    table = os.path.join(self.log_directory, "bench_docking.csv")
    earlier_timed_out = False
    for number, attempt in enumerate(attempts, 1):
        setting = attempt.get("name", "attempt%d" % number)
        settings = dict(attempt.get("set") or {})
        settings.update(derived_settings(attempt, resolution))
        row = {"model": model_name, "attempt": number, "setting": setting, "settings": json.dumps(settings, sort_keys=True),
               "timeout_s": attempt.get("timeout_s") or "", "seconds": 0, "rc": "", "timed_out": False, "skipped": False, "docked": False, "local_cc": "", "log": ""}
        if earlier_timed_out and attempt.get("skip_after_timeout"):
            self.logger.info(f"BENCH ATTEMPT {number} ({setting}) for {model_name} skipped: an earlier attempt ran into its time limit")
            row["skipped"] = True
            append_row(table, row)
            continue
        args = phenix_args(atomic_model, density_map, optimal_nproc, resolution, save_file, fasta_file, settings)
        full_command = f"bash -c 'source {self.phenix_command} && {' '.join(args)}'"
        self.logger.info(f"PHENIX COMMAND for {model_name} (benchmark attempt {number} of {len(attempts)}, {setting}): {full_command}")
        individual_log = os.path.join(self.log_directory, f'phenix_{model_name}_{self.timestamp}' + ('' if number == 1 else f'_attempt{number}') + '.log')
        row["log"] = os.path.basename(individual_log)
        t0, timed_out = time.time(), False
        try:
            with open(individual_log, 'w') as log_file:
                proc = subprocess.Popen(full_command, shell=True, stdout=log_file, stderr=subprocess.STDOUT, text=True)
                try:
                    proc.wait(timeout=attempt.get("timeout_s") or None)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    kill_tree(proc)
                except BaseException:
                    kill_tree(proc)
                    raise
        except OSError as ex:
            self.logger.error(f"PHENIX ERROR: could not run the docking command: {ex}")
            row.update(seconds=round(time.time() - t0), rc="error")
            append_row(table, row)
            return False
        rc = proc.returncode
        with open(individual_log, errors="replace") as f:
            text = f.read()
        found = LOCAL_CC.findall(text)
        row.update(seconds=round(time.time() - t0), rc=rc, timed_out=timed_out, local_cc=found[-1] if found else "")
        docked = rc == 0 and not timed_out and os.path.exists(save_file)
        row["docked"] = docked
        append_row(table, row)
        if docked:
            self.logger.info(f"PHENIX SUCCESS: Docking completed for {model_name} (attempt {number}, {setting}, {row['seconds']} s, local CC {row['local_cc'] or 'not reported'})")
            return True
        earlier_timed_out = earlier_timed_out or timed_out
        if timed_out:
            self.logger.error(f"PHENIX TIMEOUT: Docking for {model_name} took longer than {attempt.get('timeout_s')} s (attempt {number}, {setting})")
        else:
            self.logger.error(f"PHENIX FAILED: {model_name} - return code {rc} (attempt {number}, {setting})")
            lines = [line for line in text.split('\n') if 'sorry' in line.lower() or 'error' in line.lower()]
            if lines:
                self.logger.error(f"PHENIX ERROR: {lines[0].strip()}")
        if os.path.exists(save_file):
            os.remove(save_file)                      # MICA does the same after a failed docking
    return False


def main():
    sys.path.insert(0, os.getcwd())                   # run from the MICA folder: `utils` is MICA's package
    from utils import dock_in_map as upstream
    upstream.PhenixDockingProcessor.run_phenix_dock_in_map = run_phenix_dock_in_map
    upstream.main()


if __name__ == "__main__":
    main()
