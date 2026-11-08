#!/usr/bin/env python3
"""Builds scripts/scoring.ipynb: scores every model of the result zips of the other notebooks, with US-align and Phenix chain_comparison, in one way for every method
(protocol v1.0 of LOG.md, 2026-10-24; the code is scripts/score_models.py, which the notebook carries inside).

    python scripts/make_scoring_notebook.py [--outdir scripts] [--name scoring]

A CPU session of about one hour: Internet on, the two Phenix secrets, and the outputs of the other notebooks as inputs (no AlphaFold dataset: the scoring needs no AlphaFold file).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_pilot_notebook as P  # noqa: E402
import nbkit  # noqa: E402

FILES = (("scripts/nbtools.py", "scripts/nbtools.py"),
         ("scripts/score_models.py", "scripts/score_models.py"),
         ("data/entries_2026.csv", "data/entries_2026.csv"))
NET_HOSTS = (("PyPI", "https://pypi.org/simple/uv/", True),
             ("GitHub", "https://github.com/jianlin-cheng/MICA", True),
             ("RCSB", "https://files.rcsb.org/header/26DK.cif", True),
             ("Phenix", "https://phenix-online.org/download", True))

INTRO = """\
# MICA benchmark: scoring session `@NOTEBOOK@`

Scores every model that the other notebooks built (@SCORED_TEXT@)
against the deposited models from RCSB, in one way for every method (the scoring protocol of LOG.md, 2026-10-24):

* US-align `-mm 1` (multi-chain; TM-score normalised by the deposited model, aligned length, RMSD, sequence identity),
* US-align `-mm 5` on one chain per structure (the fragment-agnostic TM-score),
* Phenix `chain_comparison` on the model superposed onto the deposited one and in its own frame (Calpha match, sequence match, with the coverage the Calpha quality score),
* the same counts from a nearest-neighbour search (a check, and the fallback if Phenix cannot be used).

The notebook needs **no AlphaFold file**. It writes `scores_long.csv` (one line per model), the status tables of the notebooks it read, the raw output of every program and a zip. The zip holds no coordinates.

**What you do (Kaggle, background mode)**: accelerator **None**, Internet **On**, the two Phenix secrets (`PHENIX_USER`, `PHENIX_PASSWORD`) attached, and **the outputs of the other notebooks as inputs**:
*Add Input -> Notebook Output Files* (the tab with your own notebooks) -> add @INPUT_TEXT@ (the names you gave them; only the
versions that *Succeeded*; not the CPU notebooks). Then *Save Version -> Save & Run All (Commit)*. About one hour. When the version shows *Succeeded*, download `@NOTEBOOK@_result_....zip` from its Output and send it to me.
**Do not stop or cancel the run**: a stopped run keeps nothing.

**Phenix** is licensed to you personally. Do not share the installer, the password, or this notebook with the secrets attached.
"""

SETTINGS_MD = "## Settings"
SETTINGS = '''\
# WORKERS = models scored at the same time (a Kaggle CPU session has 4 cores).
WORKERS = 4
'''

PHENIX_AFTER = r'''
# Phenix is needed for the columns of chain_comparison only. If it could not be installed, the notebook goes on without it (US-align and the own counts are complete)
# and says so at the end; the zip is then marked INCOMPLETE.
if STATE.get("ok_before_phenix") and not STATE["ok"]:
    STATE["phenix_problem"] = STATE["problem"]
    STATE["ok"], STATE["problem"], STATE["phenix_env"] = True, "", None
    log("\n" + "!" * 72 + "\nPhenix could not be installed (%s).\nThe notebook goes on without it: the Phenix columns stay empty; US-align and the own counts are computed.\n" % STATE["phenix_problem"][:300] + "!" * 72)
'''

TOOLS_MD = """\
## Step 4. Python packages and US-align

`gemmi` (reads the models), `scipy` and `numpy` come from PyPI; US-align is compiled from the pinned commit (the same one as in every other notebook)."""
TOOLS = r'''
USALIGN_GIVEN = @USALIGN_BIN@                      # None: compile it; a path is only used by the tests
USALIGN_REPO = "@USALIGN_REPO@"
USALIGN_COMMIT = "@USALIGN_COMMIT@"
@TOOL_FUNCTIONS@


def install_packages():
    missing = []
    for module, package in (("numpy", "numpy"), ("scipy", "scipy"), ("gemmi", "gemmi")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        rc, last, _ = run([sys.executable, "-m", "pip", "install", "-q"] + missing, every=10)
        if rc != 0:
            fail("pip could not install %s (see the lines above). Send me the zip." % ", ".join(missing))
            return False
    import gemmi
    log("gemmi %s, numpy %s, scipy %s" % (gemmi.__version__, __import__("numpy").__version__, __import__("scipy").__version__))
    STATE["gemmi"] = gemmi.__version__
    return True


if STATE["ok"]:
    if install_packages():
        build_usalign()
else:
    log("skipped: an earlier step stopped")
'''

INPUTS_MD = """\
## Step 5. The models of the other notebooks

Looks in `/kaggle/input` for the result zips (`*_result_*.zip`) and for result folders (a folder with `models/<method>/<ENTRY>/`), oldest first, takes the models and the status tables out of them
(never the private AlphaFold-derived state), and lists what it found. A model that is in two sources is taken from the later one and the notebook says so."""
INPUTS = r'''
import glob, zipfile
sys.path.insert(0, os.path.join(WORK, "scripts"))
import score_models

INPUT_ROOTS = [r for r in (os.environ.get("SCORING_INPUT"), "/kaggle/input") if r and os.path.isdir(r)]
INPUTS_DIR = os.path.join(WORK, "downloads", "inputs")
STATUS_DIR = os.path.join(WORK, "status")
WANTED_FILES = ("method_status.csv", "mica_status.csv", "pending.csv", "run_info.json", "metrics_usalign.csv", "docking_domains.csv")
PRUNE = {"mica_state", "AF3_results", "af3_upload", "grids", "temp_maps", "models", "__pycache__"}
MODELS = []
SOURCES = []


def stamp_of(path):
    m = re.search(r"(\d{8}-\d{4})", os.path.basename(path.rstrip(os.sep)))
    return m.group(1) if m else ""


def find_sources(max_depth=6):
    zips, folders = [], []
    for root in INPUT_ROOTS:
        base = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - base >= max_depth:
                dirnames[:] = []
            if os.path.isdir(os.path.join(dirpath, "models")) and glob.glob(os.path.join(dirpath, "models", "*", "*")):
                folders.append(dirpath)
            dirnames[:] = [d for d in dirnames if d not in PRUNE]
            for n in filenames:
                if n.endswith(".zip") and "_result_" in n and not n.endswith("_partial.zip"):
                    zips.append(os.path.join(dirpath, n))
    return sorted(zips, key=lambda p: (stamp_of(p), p)), sorted(folders, key=lambda p: (stamp_of(p), p))


def extract_zip(path):
    label = os.path.splitext(os.path.basename(path))[0]
    dest = os.path.join(INPUTS_DIR, label)
    shutil.rmtree(dest, ignore_errors=True)
    n_models = 0
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            parts = name.split("/")
            rel = name if parts[0] == "models" else "/".join(parts[1:])
            if rel.startswith("models/") or rel in WANTED_FILES:
                target = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(name) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                n_models += rel.startswith("models/")
    return dest, n_models


def gather():
    zips, folders = find_sources()
    log("input folders: %s" % (", ".join(INPUT_ROOTS) or "none"))
    log("result zips found: %d, result folders found: %d" % (len(zips), len(folders)))
    roots = []
    for z in zips:
        try:
            dest, n = extract_zip(z)
        except (zipfile.BadZipFile, OSError) as ex:
            log("  %s: cannot be read (%s)" % (z, ex))
            continue
        with open(z, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        log("  zip    %s: %d model files" % (os.path.relpath(z, "/") if z.startswith("/") else z, n))
        roots.append(dest)
        SOURCES.append({"label": os.path.basename(dest), "kind": "zip", "path": z, "bytes": os.path.getsize(z), "sha256": digest, "model_files": n})
    for d in folders:
        if any(os.path.abspath(d).startswith(os.path.abspath(r) + os.sep) or os.path.abspath(d) == os.path.abspath(r) for r in roots):
            continue
        n = len(glob.glob(os.path.join(d, "models", "*", "*", "*")))
        log("  folder %s: %d model files" % (d, n))
        roots.append(d)
        SOURCES.append({"label": os.path.basename(d.rstrip(os.sep)), "kind": "folder", "path": d, "bytes": "", "sha256": "", "model_files": n})
    return roots


def copy_status(roots):
    os.makedirs(STATUS_DIR, exist_ok=True)
    for root in roots:
        label = os.path.basename(root.rstrip(os.sep))
        for name in WANTED_FILES:
            src = os.path.join(root, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(STATUS_DIR, "%s__%s" % (label, name)))


def table_of_models():
    counts = {}
    for m in MODELS:
        counts.setdefault((m["method"], m["model_kind"]), set()).add(m["entry"])
    log("\n%-14s %-22s %7s" % ("method", "kind", "entries"))
    for (method, kind), entries in sorted(counts.items()):
        log("%-14s %-22s %7d" % (method, kind, len(entries)))


if STATE["ok"]:
    roots = gather()
    if not roots:
        fail("no result zip and no result folder was found in the input folders. Add the outputs of the other notebooks as inputs of this notebook "
             "(Add Input -> Notebook Output Files -> @INPUT_PLAIN@), then run again.")
    else:
        MODELS, problems = score_models.collect_models(roots)
        for w in problems:
            log("WARNING: " + w)
        if not MODELS:
            fail("the inputs hold no model (no models/<method>/<ENTRY>/ folder with .pdb or .cif files)")
        else:
            copy_status(roots)
            table_of_models()
            STATE["sources"] = SOURCES
            log("\n%d model files of %d entries; %d of them are the primary file of their method" % (len(MODELS), len({m["entry"] for m in MODELS}), sum(m["primary"] for m in MODELS)))
            pack_now()
else:
    log("skipped: an earlier step stopped")
'''

NATIVES_MD = """\
## Step 6. The deposited models, and a test of the scoring

Downloads the deposited model of every entry that has models (the `cif_url` of `data/entries_2026.csv`), keeps its amino-acid residues (C-alpha trace, first model, first alternative location) and
checks the residue count against the entries table. Then the deposited model of the first entry is scored against itself: US-align must give TM-score 1 and every residue aligned, Phenix must
find every residue. If Phenix does not pass this test, the notebook goes on without it (the own counts are still written) and says so in the first line of the result."""
NATIVES = r'''
import urllib.request

NATIVE_DIR = os.path.join(WORK, "native")
SCORING_DIR = os.path.join(WORK, "scoring")
NATIVES = {}


def download_to(url, dest, tries=4, timeout=120):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(dest + ".part", dest)
            return True
        except Exception as ex:
            log("  download of %s failed (try %d of %d): %s" % (url, i + 1, tries, ex))
            time.sleep(5 * (i + 1))
    return False


def get_natives():
    rows = {r["pdb_id"].upper(): r for r in score_models.read_entries(os.path.join(WORK, "data", "entries_2026.csv"))}
    os.makedirs(NATIVE_DIR, exist_ok=True)
    wanted = sorted({m["entry"] for m in MODELS})
    have = []
    for entry in wanted:
        if entry not in rows:
            log("  %s is not in data/entries_2026.csv: its models are not scored" % entry)
            continue
        dest = os.path.join(NATIVE_DIR, entry + ".cif")
        if os.path.exists(dest) or download_to(rows[entry]["cif_url"], dest):
            have.append(rows[entry])
        else:
            log("  the deposited model of %s could not be downloaded: its models are not scored (status no_native)" % entry)
    got = score_models.prepare_natives(have, NATIVE_DIR, os.path.join(NATIVE_DIR, "ca")) if have else {}
    out = []
    for entry, (trace, path, note) in sorted(got.items()):
        out.append({"entry": entry, "residues": trace.n_residues, "chains": trace.n_chains, "note": note, "sha256": sha256_file(os.path.join(NATIVE_DIR, entry + ".cif"))})
        if note:
            log("  %s: %s" % (entry, note))
    write_csv(os.path.join(WORK, "scoring", "natives.csv"), ["entry", "residues", "chains", "note", "sha256"], out)
    return got


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def selftest(natives):
    """The first deposited model against itself."""
    entry = sorted(natives)[0]
    trace, path, _ = natives[entry]
    r = score_models.score_model(STATE["usalign"], STATE["phenix_env"], trace, path, os.path.join(SCORING_DIR, "selftest"), native_file=path)
    problems = []
    if r["status"] != "ok" or not (isinstance(r["tm_score"], float) and r["tm_score"] > 0.999) or r["aligned_ca_length"] != trace.n_residues:
        problems.append("US-align: %s" % {k: r[k] for k in ("status", "tm_score", "aligned_ca_length", "note")})
    phenix_bad = None
    if STATE["phenix_env"]:
        if r["cc_status"] != "ok" or r["cc_status_ff"] != "ok" or r["ca_match"] != 100.0 or r["ca_found"] != 100.0 or r["seq_match"] != 100.0:
            phenix_bad = "Phenix: %s" % {k: r[k] for k in ("cc_status", "cc_status_ff", "ca_match", "ca_found", "seq_match", "note")}
    log("self-test on %s (%d residues): TM-score %s, fNS %s, own match %s, Phenix match %s found %s sequence %s" % (
        entry, trace.n_residues, r["tm_score"], r["tm_score_fns"], r["own_ca_match"], r["ca_match"], r["ca_found"], r["seq_match"]))
    for name in score_models.COORDINATE_FILES:
        try:
            os.remove(os.path.join(SCORING_DIR, "selftest", name))
        except OSError:
            pass
    return problems, phenix_bad


if STATE["ok"] and MODELS:
    NATIVES = get_natives()
    if not NATIVES:
        fail("no deposited model could be read: nothing can be scored")
    else:
        problems, phenix_bad = selftest(NATIVES)
        if problems:
            fail("the self-test of the scoring failed: " + " | ".join(problems))
        STATE["phenix_used"] = bool(STATE["phenix_env"])
        if phenix_bad:
            log("\n" + "!" * 72 + "\nPhenix chain_comparison did not pass the self-test, so the Phenix columns are NOT computed.\n" + phenix_bad + "\n" + "!" * 72)
            STATE["phenix_env_unused"] = STATE["phenix_env"]
            STATE["phenix_used"] = False
        elif not STATE["phenix_env"]:
            log("\n" + "!" * 72 + "\nPhenix is not available: the Phenix columns are NOT computed (the own counts are).\n" + "!" * 72)
else:
    log("skipped: an earlier step stopped")
'''

SCORE_MD = """\
## Step 7. Score every model

`WORKERS` models at a time: US-align `-mm 1` (the original file), US-align `-mm 5` (one chain per structure), Phenix chain_comparison twice, the own counts. One line per model; the raw outputs of the programs
are kept (the coordinate files that the scoring writes are removed after each model). A model that cannot be read is `no_model`, a model whose alignment fails or runs out of time (15 minutes) is
`no_alignment`; both stay in the table."""
SCORE = r'''
SCORE_ROWS = []


def progress(rows):
    if len(rows) % 10 == 0:
        score_models.write_scores(rows, os.path.join(SCORING_DIR, "scores_long.csv"))
        pack_now(600)


if STATE["ok"] and MODELS and NATIVES:
    t_score = time.time()
    SCORE_ROWS = score_models.score_all(MODELS, NATIVES, STATE["usalign"], STATE["phenix_env"] if STATE.get("phenix_used") else None, SCORING_DIR, workers=WORKERS, log=log, progress=progress)
    log("\n%d models scored in %d minutes" % (len(SCORE_ROWS), round((time.time() - t_score) / 60)))
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

SUMMARY_MD = """\
## Step 8. The tables

`scoring/scores_long.csv` (every model; the columns are explained in `scripts/score_models.py`), `scoring/scores_primary.csv` (the primary file of each method for each entry), `scoring/sources.csv`
(the result zips that were read, with their hashes), `status/` (the status tables of the notebooks that were read), `scoring/natives.csv` (the deposited models)."""
SUMMARY = r'''
def summary_tables(rows):
    cols = score_models.SCORE_COLUMNS
    write_csv(os.path.join(SCORING_DIR, "scores_long.csv"), cols, rows)
    write_csv(os.path.join(SCORING_DIR, "scores_primary.csv"), cols, [r for r in rows if str(r["primary"]) == "True"])
    write_csv(os.path.join(SCORING_DIR, "sources.csv"), ["label", "kind", "path", "bytes", "sha256", "model_files"], SOURCES)
    prim = [r for r in rows if str(r["primary"]) == "True"]
    methods = sorted({r["method"] for r in prim})
    log("\nprimary models: TM-score (-mm 1) / fragment-agnostic TM-score / Calpha match / coverage")
    log("%-5s " % "entry" + " ".join("%-34s" % m for m in methods))
    for entry in sorted({r["entry"] for r in prim}):
        cells = []
        for m in methods:
            sel = [r for r in prim if r["entry"] == entry and r["method"] == m]
            if not sel:
                cells.append("%-34s" % "-")
            else:
                r = sel[0]
                cells.append("%-34s" % ("%s / %s / %s / %s" % (r["tm_score"], r["tm_score_fns"], r["ca_match"], r["coverage"]) if r["status"] == "ok" else r["status"]))
        log("%-5s " % entry + " ".join(cells))
    states = {}
    for r in rows:
        states[r["status"]] = states.get(r["status"], 0) + 1
    log("\nstatus of the %d models: %s" % (len(rows), ", ".join("%s %d" % kv for kv in sorted(states.items()))))
    return states


if STATE["ok"] and SCORE_ROWS:
    states = summary_tables(SCORE_ROWS)
    STATE["complete"] = bool(STATE.get("phenix_used")) and all(r["status"] != "no_native" for r in SCORE_ROWS)
    if not STATE.get("phenix_used"):
        log("\nINCOMPLETE: the Phenix columns are missing (see the box above). The US-align columns and the own counts are complete.")
    if any(r["status"] == "no_native" for r in SCORE_ROWS):
        log("\nINCOMPLETE: some entries have no deposited model (status no_native).")
else:
    log("skipped: an earlier step stopped")
'''

EXTRA_INFO = ('{"protocol_version": globals().get("score_models").PROTOCOL_VERSION if "score_models" in globals() else "", "usalign_commit": globals().get("USALIGN_COMMIT", ""), '
              '"usalign_version": STATE["usalign_version"], "gemmi": STATE.get("gemmi", ""), "phenix_installer": os.path.basename(globals().get("INSTALLER", "")), '
              '"phenix_installer_sha256": STATE["installer_sha256"], "phenix_version_output": STATE["phenix_version"], "phenix_used": bool(STATE.get("phenix_used")), '
              '"sources": STATE.get("sources", []), "models": len(globals().get("MODELS", [])), "workers": globals().get("WORKERS"), "input_sha256": globals().get("EXPECTED", {})}')


SCOPES = {
    "all": dict(scored="ModelAngelo, E3-CryoFold, EMProt, EModelX(+AF) from `gpu_batch1`, `gpu_batch2` and `gpu_batch3`; MICA from `mica_gpu`",
                inputs="`gpu_batch1`, `gpu_batch2`, `gpu_batch3` and `mica_gpu`", plain="gpu_batch1, gpu_batch2, gpu_batch3, mica_gpu"),
    "base": dict(scored="ModelAngelo, E3-CryoFold, EMProt, EModelX(+AF) from `gpu_batch1`, `gpu_batch2` and `gpu_batch3`; MICA is scored in another session",
                 inputs="`gpu_batch1`, `gpu_batch2` and `gpu_batch3` **and nothing else**", plain="gpu_batch1, gpu_batch2, gpu_batch3"),
    "mica": dict(scored="MICA from `mica_gpu`; the other methods are scored in another session",
                 inputs="`mica_gpu` **and nothing else**", plain="mica_gpu"),
    "nodock": dict(scored="MICA without docked AlphaFold input from `mica_gpu_nodock` (the sensitivity session); the other methods are scored in another session",
                   inputs="`mica_gpu_nodock` **and nothing else**", plain="mica_gpu_nodock"),
    "mica2": dict(scored="MICA from `mica_gpu2_a` and `mica_gpu2_b` (the second GPU round, LOG.md 2026-10-27); the other methods are scored in another session",
                  inputs="`mica_gpu2_a` and `mica_gpu2_b` **and nothing else**", plain="mica_gpu2_a, mica_gpu2_b"),
    "mica3": dict(scored="MICA from `mica_gpu3` (the third GPU round, LOG.md 2026-10-28); the other methods are scored in another session",
                  inputs="`mica_gpu3` **and nothing else**", plain="mica_gpu3"),
    "nodock2": dict(scored="MICA without docked AlphaFold input from `mica_gpu2_nodock_a` and `mica_gpu2_nodock_b` (the sensitivity session, second GPU round); the other methods are scored in another session",
                    inputs="`mica_gpu2_nodock_a` and `mica_gpu2_nodock_b` **and nothing else**", plain="mica_gpu2_nodock_a, mica_gpu2_nodock_b"),
    "cryoatom": dict(scored="CryoAtom v1.0.0 from `gpu_cryoatom_a` (added to the benchmark after Prof. Cheng's request of 2026-10-31); the other methods are scored in another session",
                     inputs="`gpu_cryoatom_a` **and nothing else**", plain="gpu_cryoatom_a"),
    "cryozeta_b": dict(scored="CryoZeta from `gpu_cryozeta_b` (the first full session; added to the benchmark after Prof. Cheng's request of 2026-10-31); the other methods are scored in another session",
                       inputs="`gpu_cryozeta_b` **and nothing else**", plain="gpu_cryozeta_b"),
    "cryozeta_c": dict(scored="CryoZeta from `gpu_cryozeta_c` (the second full session; added to the benchmark after Prof. Cheng's request of 2026-10-31); the other methods are scored in another session",
                       inputs="`gpu_cryozeta_c` **and nothing else**", plain="gpu_cryozeta_c"),
    "cryozeta_d": dict(scored="CryoZeta from `gpu_cryozeta_d` (the third full session; added to the benchmark after Prof. Cheng's request of 2026-10-31); the other methods are scored in another session",
                       inputs="`gpu_cryozeta_d` **and nothing else**", plain="gpu_cryozeta_d"),
}


def build(out_path, name="scoring", scope="all", entries_path=None, phenix_url=P.PHENIX_URL, usalign_repo=P.USALIGN_REPO, usalign_commit=P.USALIGN_COMMIT, usalign_bin=None, net_hosts=NET_HOSTS):
    """Writes the notebook. The keyword arguments from `entries_path` on exist for the tests (local servers, a compiled US-align)."""
    m = re.search(r"version=([^&]+)&file=([^&]+)$", phenix_url)
    if not m:
        raise ValueError("the Phenix link must end in ?version=...&file=...: %s" % phenix_url)
    version, installer = m.group(1), m.group(2)
    text = SCOPES[scope]
    file_cells, expected = nbkit.file_cells(FILES, entries_path)
    mapping = {"@PHENIX_URL@": phenix_url, "@PHENIX_VERSION@": version, "@INSTALLER@": installer, "@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit,
               "@USALIGN_BIN@": repr(usalign_bin), "@NET_HOSTS@": repr(tuple(net_hosts)), "@REQUIRE_GPU@": "False", "@EXPECTED@": repr(expected), "@NOTEBOOK@": name,
               "@EXTRA_INFO@": EXTRA_INFO, "@MICA_REPO@": P.MICA_REPO, "@MICA_COMMIT@": P.MICA_COMMIT, "@WEIGHTS_URL@": P.WEIGHTS_URL, "@ZENODO_API@": P.ZENODO_API,
               "@SCORED_TEXT@": text["scored"], "@INPUT_TEXT@": text["inputs"], "@INPUT_PLAIN@": text["plain"]}
    fl = lambda text: nbkit.fill(text, mapping).strip("\n") + "\n"
    step5_filled = nbkit.fill(P.STEP5, mapping)
    tool_functions = nbkit.function_sources(step5_filled, ["git_checkout", "build_usalign"])
    tools = nbkit.fill(TOOLS.replace("@TOOL_FUNCTIONS@", tool_functions.strip("\n") + "\n"), mapping)
    cells = [("intro", nbkit.md(nbkit.fill(INTRO, mapping))), ("settings-title", nbkit.md(SETTINGS_MD)), ("settings", nbkit.code(fl(SETTINGS))),
             ("step1-title", nbkit.md(P.STEP1_MD)), ("step1", nbkit.code(fl(P.STEP1))),
             ("step2-title", nbkit.md(P.STEP2_MD)), ("step2-dirs", nbkit.code(P.STEP2_DIRS.strip("\n") + "\n"))]
    cells += file_cells
    cells += [("step2-check", nbkit.code(fl(P.STEP2_CHECK))),
              ("robust-title", nbkit.md(nbkit.fill(nbkit.ROBUST_MD, mapping))), ("robust", nbkit.code(fl(nbkit.ROBUST))),
              ("step3-title", nbkit.md(P.STEP3_MD)), ("phenix-guard", nbkit.code("STATE['ok_before_phenix'] = bool(STATE['ok'])\n")), ("step3", nbkit.code(fl(P.STEP3))),
              ("phenix-after", nbkit.code(PHENIX_AFTER.strip("\n") + "\n")),
              ("tools-title", nbkit.md(TOOLS_MD)), ("tools", nbkit.code(tools.strip("\n") + "\n")),
              ("inputs-title", nbkit.md(INPUTS_MD)), ("inputs", nbkit.code(fl(INPUTS))),
              ("natives-title", nbkit.md(NATIVES_MD)), ("natives", nbkit.code(fl(NATIVES))),
              ("score-title", nbkit.md(SCORE_MD)), ("score", nbkit.code(fl(SCORE))),
              ("summary-title", nbkit.md(SUMMARY_MD)), ("summary", nbkit.code(fl(SUMMARY))),
              ("final-title", nbkit.md(nbkit.FINAL_MD)), ("final", nbkit.code(fl(nbkit.FINAL)))]
    nbkit.assemble(out_path, name + ".ipynb", cells)
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scope", choices=sorted(SCOPES), default="all", help="which notebooks' models this session scores (the text of the notebook says so): all, base (the four other methods), mica, nodock")
    p.add_argument("--name", help="default: scoring (scope all) or scoring_<scope>")
    p.add_argument("--outdir", default=HERE)
    p.add_argument("--phenix-url", default=P.PHENIX_URL)
    args = p.parse_args(argv)
    name = args.name or ("scoring" if args.scope == "all" else "scoring_" + args.scope)
    out = os.path.join(args.outdir, name + ".ipynb")
    expected = build(out, name=name, scope=args.scope, phenix_url=args.phenix_url)
    print("wrote %s (%d embedded files)" % (out, len(expected)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
