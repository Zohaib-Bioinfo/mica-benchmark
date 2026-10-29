#!/usr/bin/env python3
"""Build scripts/pilot_modelangelo.ipynb: the Phase 4 pilot of ModelAngelo (v1.0.18) on the pilot entry (default 26DK).

ModelAngelo needs only the map and the protein sequence (no AlphaFold prediction, no Phenix), so this notebook has no secrets and no private input.
It checks the machine (a GPU is required), builds ModelAngelo's own environment (Python 3.11, PyTorch 2.9.1, exact pins in env/modelangelo_requirements.txt, the recipe of
ModelAngelo's install_script.sh), fetches ModelAngelo at its pinned commit and its weights (`model_angelo setup_weights --bundle-name nucleotides`, the default bundle
of `model_angelo build`; ModelAngelo itself checks the MD5 of the bundle, the ESM-1b language model is downloaded from Meta), prepares the map and the FASTA of
the entry, runs `model_angelo build` with a progress line every two minutes, and scores the models with US-align. The Phenix scoring (chain_comparison) is a later
notebook: this pilot only has to show that the method runs on a free GPU, how long it takes, what it writes, and that the US-align scoring reads its files.

    python make_modelangelo_notebook.py            # writes scripts/pilot_modelangelo.ipynb

Run it on Kaggle with the GPU on, as "Save & Run All (Commit)".
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_pilot_notebook as P  # noqa: E402
import nbkit  # noqa: E402

NOTEBOOK = "modelangelo_pilot"
NB_NAME = "pilot_modelangelo.ipynb"
MA_REPO = "https://github.com/3dem/model-angelo"
MA_COMMIT = "994945bdfa6e5368e0d62349a47792f4864eebc3"       # tag v1.0.18, 2026-06-15
MA_PY_VERSION = "3.11.14"
MA_BUNDLE = "nucleotides"
FILES = (("scripts/nbtools.py", "scripts/nbtools.py"),
         ("scripts/mica_inputs.py", "scripts/mica_inputs.py"),
         ("scripts/evaluate_models.py", "scripts/evaluate_models.py"),
         ("data/af3/af3_jobs.csv", "data/af3/af3_jobs.csv"),
         ("data/entries_2026.csv", "data/entries_2026.csv"),
         ("env/modelangelo_requirements.txt", "env/modelangelo_requirements.txt"))
NET_HOSTS = (("PyPI", "https://pypi.org/simple/uv/", True),
             ("GitHub", "https://github.com/3dem/model-angelo", True),
             ("Zenodo", "https://zenodo.org/api/records/7942241", True),
             ("Meta ESM weights", "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt", True),
             ("EBI FTP", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-80558/header/emd-80558.xml", False),
             ("RCSB", "https://files.rcsb.org/header/26DK.cif", False))
IMPORT_TEST = ("import sys, torch, numpy, scipy, pandas, Bio, mrcfile, einops, esm, pyhmmer, loguru, psutil, matplotlib, matplotlib.pyplot; "
               "print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| numpy', numpy.__version__, "
               "'| pandas', pandas.__version__, '| biopython', Bio.__version__, '| pyhmmer', pyhmmer.__version__)")

INTRO = """\
# MICA benchmark: pilot of ModelAngelo on entry @ENTRY_LIST@

ModelAngelo (v1.0.18) builds an atomic model from the map and the protein sequence alone. This notebook runs it once, on one small entry, to find out how long it
takes on a free GPU, what it writes, and whether the scoring reads its files. No Phenix, no secrets and no private files are needed.

1. checks the machine (a GPU is needed) and the internet,
2. builds ModelAngelo's Python 3.11 environment from exact pins and tests the GPU,
3. fetches ModelAngelo (pinned commit) and its weights (the MD5 is checked by ModelAngelo itself),
4. downloads the map and the deposited model, writes the FASTA file of the entry,
5. runs `model_angelo build` (a progress line every two minutes), then scores the models with US-align and packs one zip.

**What you do (Kaggle, background mode)**: accelerator **GPU T4 x2** (not P100), Internet **On**, then *Save Version -> Save & Run All (Commit)*. It runs on Kaggle's
servers; you can close the tab and your Wi-Fi does not matter. About 40 minutes to 1.5 hours. When the version shows *Succeeded*, open it and download the zip
`modelangelo_pilot_result_....zip` from its Output. Send it to me. **Do not stop or cancel the run** unless it is stuck for more than 3 hours.
"""

STEP3_MD = """\
## Step 3. Python 3.11 environment for ModelAngelo

ModelAngelo's own install script uses Python 3.11 and `torch==2.9.1`. The notebook builds the environment with `uv` from PyPI wheels with the exact pins of
`env/modelangelo_requirements.txt` (47 packages, the versions that pip would pick in the week of the ModelAngelo release), tests the imports, and runs a small
computation on the GPU, so a GPU that this PyTorch cannot drive is found here and not later."""
STEP3 = r'''
UV_VERSION = "@UV_VERSION@"
PY_VERSION = "@MA_PY_VERSION@"
ENV_PY_GIVEN = @ENV_PYTHON@                       # None: build the environment; a path is only used by the tests
ENV_DIR = os.path.join(BIG, "ma_env")
ENV_PY = ENV_PY_GIVEN or os.path.join(ENV_DIR, "bin", "python")
ENV_BIN = os.path.dirname(ENV_PY)
MA_ENV = {"PATH": ENV_BIN + os.pathsep + os.environ.get("PATH", ""), "VIRTUAL_ENV": None, "PYTHONPATH": None, "PYTHONHOME": None, "MPLBACKEND": "Agg"}
REQ_FILE = "env/modelangelo_requirements.txt"
IMPORT_TEST = @IMPORT_TEST@
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


def pinned(path=REQ_FILE):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if "==" in ln and not ln.startswith("#")]


def env_imports():
    return run([ENV_PY, "-c", IMPORT_TEST], env=MA_ENV, quiet=True)


def make_env():
    rc, last, _ = env_imports() if os.path.exists(ENV_PY) else (1, [], False)
    if rc == 0:
        log("the Python environment already exists: " + last[-1])
        return True
    tools = {"UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or os.path.join(BIG, "uv_cache"),
             "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR") or os.path.join(BIG, "uv_python"), "UV_PYTHON_PREFERENCE": "only-managed"}
    rc, last, _ = run([sys.executable, "-m", "pip", "install", "-q", "uv==" + UV_VERSION])
    if rc != 0:
        fail("pip could not install uv (see the lines above). Run the notebook again in a few minutes.")
        return False
    uv = [sys.executable, "-m", "uv"]
    log("uv: " + sh("%s -m uv --version" % sys.executable))
    for what, cmd in (("Python " + PY_VERSION, uv + ["python", "install", PY_VERSION]),
                      ("the virtual environment", uv + ["venv", "--python", PY_VERSION, ENV_DIR]),
                      ("the %d packages (a few minutes; PyTorch with its CUDA libraries is about 4 GB)" % len(pinned()),
                       uv + ["pip", "install", "--python", ENV_PY, "--no-deps", "-r", REQ_FILE])):
        log("installing " + what + " ...")
        rc, last, _ = run_watched(cmd, "uv: " + what[:30], env=tools, every=20)
        if rc != 0:
            fail("uv could not install %s (exit code %d, the last lines are above). Run the notebook again in a few minutes; if it happens again, send me the zip." % (what, rc))
            return False
    rc, last, _ = env_imports()
    if rc != 0 and any("libGL" in ln or "libgomp" in ln for ln in last):
        log("a system library is missing; installing it with apt ...")
        run(["bash", "-c", "apt-get update -qq && apt-get install -y -qq libgl1 libgomp1"], every=20)
        rc, last, _ = env_imports()
    if rc != 0:
        log("\n".join(last))
        fail("the Python packages were installed but do not import (see the lines above). Send me the zip.")
        return False
    log(last[-1])
    return True


def gpu_test():
    rc, last, _ = run([ENV_PY, "-c", GPU_TEST], env=MA_ENV)
    if rc == 3:
        fail("PyTorch in the ModelAngelo environment does not see the GPU. Switch the GPU on (Kaggle: Session options -> Accelerator, GPU T4 x2) and run again.")
    elif rc != 0:
        fail("this GPU cannot run PyTorch 2.9.1 with CUDA 12.8 (see the lines above). Choose another GPU: Kaggle GPU T4 x2 (the P100 is too old for this PyTorch).")
    return rc == 0


def freeze_check():
    code = "import importlib.metadata as m; print(chr(10).join(sorted('%s==%s' % (d.metadata['Name'], d.version) for d in m.distributions())))"
    if os.path.exists("ma_env_freeze.txt"):
        os.remove("ma_env_freeze.txt")
    run([ENV_PY, "-c", code], env=MA_ENV, log_file="ma_env_freeze.txt", quiet=True)
    norm = lambda line: re.sub(r"[-_.]+", "-", line.strip().lower())
    got = {norm(ln) for ln in open("ma_env_freeze.txt", encoding="utf-8") if "==" in ln}
    want = {norm(ln) for ln in pinned()}
    differences = sorted(got ^ want)
    STATE["env"] = {"python_version": PY_VERSION, "uv_version": UV_VERSION, "packages": len(got), "differences_to_pinned_list": differences[:20]}
    log("installed packages: %d; differences to %s: %d %s" % (len(got), REQ_FILE, len(differences), differences[:6] or ""))


if STATE["ok"]:
    if make_env():
        if REQUIRE_GPU:
            gpu_test()
        if ENV_PY_GIVEN is None:
            freeze_check()
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

STEP4_MD = """\
## Step 4. ModelAngelo code, weights, US-align

ModelAngelo is fetched at its pinned commit and run from that source folder (`python -m model_angelo`, the same program as the `model_angelo` command).
`model_angelo setup_weights --bundle-name nucleotides` downloads the bundle that `model_angelo build` uses by default (Zenodo, MD5 checked by ModelAngelo) and the
ESM-1b language model (Meta); `TORCH_HOME` points to a folder of this session. US-align is compiled from its pinned commit."""
STEP4 = r'''
MA_REPO = "@MA_REPO@"
MA_COMMIT = "@MA_COMMIT@"
MA_BUNDLE = "@MA_BUNDLE@"
USALIGN_REPO = "@USALIGN_REPO@"
USALIGN_COMMIT = "@USALIGN_COMMIT@"
USALIGN_GIVEN = @USALIGN_BIN@                      # None: compile it; a path is only used by the tests
MA_DIR = os.path.join(BIG, "ModelAngelo")
TORCH_HOME = os.path.join(BIG, "torch_home")
MA_RUN_ENV = dict(MA_ENV, TORCH_HOME=TORCH_HOME)


@GIT_AND_USALIGN@


def fetch_modelangelo():
    if not git_checkout(MA_REPO, MA_COMMIT, MA_DIR):
        fail("ModelAngelo could not be fetched at commit %s (see the lines above). Run the notebook again in a few minutes." % MA_COMMIT)
        return False
    missing = [p for p in ("model_angelo/__main__.py", "model_angelo/model_definitions/" + MA_BUNDLE) if not os.path.exists(os.path.join(MA_DIR, p))]
    if missing:
        fail("the ModelAngelo source at commit %s lacks %s" % (MA_COMMIT, ", ".join(missing)))
        return False
    log("ModelAngelo at commit %s" % MA_COMMIT)
    rc, last, _ = run([ENV_PY, "-m", "model_angelo", "--version"], env=MA_RUN_ENV, cwd=MA_DIR, quiet=True)
    STATE["modelangelo_version"] = (last or [""])[-1]
    log("version printed by the program: %s" % STATE["modelangelo_version"])
    code = "import model_angelo.apps.build, model_angelo.utils.setup_weights; print('ModelAngelo modules import')"
    rc, last, _ = run([ENV_PY, "-c", code], env=MA_RUN_ENV, cwd=MA_DIR)
    if rc != 0:
        fail("ModelAngelo's own Python modules cannot be imported (see the lines above). Send me the zip.")
        return False
    return True


def bundle_dir():
    return os.path.join(TORCH_HOME, "hub", "checkpoints", "model_angelo_v1.0", MA_BUNDLE)


def fetch_weights():
    logf = os.path.join(WORK, "logs", "setup_weights.log")
    cmd = [ENV_PY, "-m", "model_angelo", "setup_weights", "--bundle-name", MA_BUNDLE]
    log("downloading the weights: %s" % " ".join(cmd[1:]))
    t0 = time.time()
    rc, last, timed_out = run_watched(cmd, "weights download", watch=[logf], env=MA_RUN_ENV, cwd=MA_DIR, log_file=logf, timeout=3600, every=1)
    text = open(logf, errors="replace").read() if os.path.exists(logf) else ""
    if rc != 0 or timed_out or "Successful!" not in text:
        log("\n".join(last[-15:]))
        fail("the ModelAngelo weights could not be downloaded or did not pass ModelAngelo's own MD5 check (%s). Run the notebook again in a few minutes; if it happens again, send me the zip."
             % ("timeout" if timed_out else "exit code %s" % rc))
        return False
    files = {}
    for dirpath, _dirs, names in os.walk(TORCH_HOME):
        for n in names:
            p = os.path.join(dirpath, n)
            if n.endswith((".torch", ".pt", ".json", ".txt", ".zip")) and os.path.getsize(p) > 0:
                files[os.path.relpath(p, TORCH_HOME)] = os.path.getsize(p)
    cfg = os.path.join(bundle_dir(), "config.json")
    if not os.path.exists(cfg):
        fail("the weights bundle %s was not installed where expected (%s)" % (MA_BUNDLE, bundle_dir()))
        return False
    shutil.copy(cfg, os.path.join(WORK, "modelangelo_config.json"))
    STATE["weights"] = {"bundle": MA_BUNDLE, "seconds": round(time.time() - t0), "md5_checked_by_modelangelo": "Checksum of downloaded file matches expected checksum" in text,
                        "files_bytes": files, "bundle_dir": bundle_dir()}
    log("weights installed in %d s; files: %s" % (STATE["weights"]["seconds"], {k: v for k, v in files.items() if v > 1000000}))
    return True


if STATE["ok"]:
    if fetch_modelangelo() and fetch_weights():
        build_usalign()
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

STEP5_MD = """\
## Step 5. The entry: map, deposited model, FASTA file

The map comes from EMDB (unzipped to a `.mrc` file), the deposited model from RCSB (only used for scoring), and the FASTA file from the job table of the benchmark
(`data/af3/af3_jobs.csv`: one record per different chain sequence, with the number of copies in the header, the format of the RCSB FASTA files that ModelAngelo reads).
Every method of the benchmark gets this same sequence information."""
STEP5 = r'''
sys.path.insert(0, os.path.join(WORK, "scripts"))
import evaluate_models, mica_inputs

ENTRIES = @ENTRIES@
INPUT_DIR = os.path.join(BIG, "input")
METRIC_COLUMNS = ["entry", "model_kind", "model_file", "status", "note", "tm_score", "tm_score_pred_norm", "ca_rmsd", "aligned_ca_length", "seq_identity",
                  "pred_residues", "native_residues", "coverage"]


@READ_WRITE_DOWNLOAD@


def prepare_entry(entry, row, res):
    folder = os.path.join(INPUT_DIR, entry)
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
    res["map_sha256"] = sha256_file(map_path)
    run([ENV_PY, "-c", "import sys, mrcfile; m = mrcfile.open(sys.argv[1], permissive=True); d = m.data; "
         "print('map grid', d.shape, 'voxel size', [round(float(v), 4) for v in (m.voxel_size.x, m.voxel_size.y, m.voxel_size.z)], "
         "'min %.4g max %.4g mean %.4g std %.4g' % (d.min(), d.max(), d.mean(), d.std()))", map_path], env=MA_ENV)
    native = download(row["cif_url"], os.path.join(WORK, "native", entry + ".cif"), 1000)
    jobs = mica_inputs.read_jobs("data/af3/af3_jobs.csv", entry)
    fasta = os.path.join(folder, entry.lower() + ".fasta")
    with open(fasta, "w", encoding="utf-8") as f:
        f.write(mica_inputs.mica_fasta(jobs))
    os.makedirs(os.path.join(WORK, "inputs"), exist_ok=True)
    shutil.copy(fasta, os.path.join(WORK, "inputs", os.path.basename(fasta)))
    res["sequences"] = [{"job": r["job_name"], "length": int(r["length"]), "copies": int(r["copies_in_entry"])} for r in jobs]
    log("%s: map %s (sha256 %s), FASTA with %d sequence(s) %s" % (entry, os.path.basename(map_path), res["map_sha256"][:16], len(jobs), res["sequences"]))
    return map_path, fasta, native
'''

STEP6_MD = """\
## Step 6. Run ModelAngelo

`model_angelo build -v <map> -pf <fasta> -o <folder> --device 0`, the command of ModelAngelo's README with the GPU named explicitly. A progress line is printed every two
minutes with the last line of ModelAngelo's own log; GPU memory and free RAM are sampled every 5 seconds into `gpu/`. The time limit is 3 hours; a stopped run is a
result and stays in the tables."""
STEP6 = r'''
OUT_ROOT = os.path.join(WORK, "ma_out")
TIMEOUT = 3 * 3600                  # seconds; a run that takes longer is stopped and counted as failed
ENTRY_RESULTS = {}


def run_modelangelo(entry, map_path, fasta, res):
    out_dir = os.path.join(OUT_ROOT, "%s_modelangelo" % entry)     # ModelAngelo names its result after this folder
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    cmd = [ENV_PY, "-m", "model_angelo", "build", "-v", map_path, "-pf", fasta, "-o", out_dir, "--device", "0"]
    log("\n=== ModelAngelo: %s" % " ".join(cmd[1:]))
    logf = os.path.join(WORK, "logs", "%s_modelangelo.log" % entry)
    mon = Monitor(os.path.join(WORK, "gpu", "%s_modelangelo.csv" % entry))
    mon.start()
    t0 = time.time()
    rc, last, timed_out = run_watched(cmd, "ModelAngelo", watch=[logf, os.path.join(out_dir, "model_angelo.log")], env=MA_RUN_ENV, cwd=MA_DIR, log_file=logf, timeout=TIMEOUT, every=1)
    info = mon.stop()
    info.update(rc=rc, seconds=round(time.time() - t0), timed_out=timed_out)
    res["steps"]["modelangelo"] = info
    log("ModelAngelo finished: exit code %s, %d s, GPU memory peak %s MB, lowest free RAM %s GB" % (rc, info["seconds"], info["peak_gpu_mb"], info["min_ram_available_gb"]))
    final, raw = os.path.join(out_dir, "%s_modelangelo.cif" % entry), os.path.join(out_dir, "%s_modelangelo_raw.cif" % entry)
    made = os.path.exists(final) and os.path.getsize(final) > 500
    if rc != 0 or timed_out or not made:
        res.update(failed_step="modelangelo", note=("timeout" if timed_out else "exit code %s" % rc) + ("" if made else "; the model file %s was not written" % os.path.basename(final)))
        log("ENTRY %s STOPPED in ModelAngelo: %s. The last lines of the program:\n%s" % (entry, res["note"], "\n".join(last[-20:])))
        return None
    log("ModelAngelo wrote: %s" % ", ".join("%s (%d bytes)" % (os.path.relpath(os.path.join(d, n), out_dir), os.path.getsize(os.path.join(d, n))) for d, _s, ns in os.walk(out_dir) for n in sorted(ns)))
    return {"modelangelo": final, "modelangelo_raw": raw}


def score_entry(entry, row, native, models, res):
    rows = []
    os.makedirs(os.path.join(WORK, "scoring"), exist_ok=True)
    for kind, path in models.items():
        if not os.path.exists(path):
            log("%s: %s was not written, not scored" % (entry, os.path.basename(path)))
            continue
        ev = evaluate_models.evaluate(STATE["usalign"], path, native, int(row["modeled_residues"]))
        full = subprocess.run([STATE["usalign"], path, native, "-mol", "protein"], capture_output=True, text=True, errors="replace")
        with open(os.path.join(WORK, "scoring", "%s_%s_usalign.txt" % (entry, kind)), "w", encoding="utf-8") as f:
            f.write("# USalign %s %s -mol protein\n%s%s" % (path, native, full.stdout, full.stderr))
        rows.append(dict(ev, entry=entry, model_kind=kind, model_file=os.path.basename(path)))
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
        map_path, fasta, native = prepare_entry(entry, row, res)
    except Exception as ex:
        res.update(failed_step="inputs", note=str(ex)[:500])
        log("\nENTRY %s STOPPED while preparing the inputs: %s" % (entry, ex))
        return res
    pack_now()
    models = run_modelangelo(entry, map_path, fasta, res)
    ok = models is not None
    pack_now()
    if ok:
        try:
            score_entry(entry, row, native, models, res)
        except Exception as ex:
            res.update(failed_step="scoring", note=("scoring: %s" % ex)[:500])
            log("scoring stopped: %s" % ex)
            ok = False
    if ok:
        res["status"] = "ok"
    res["seconds"] = round(time.time() - t0)
    return res


def summary_table():
    log("\n%-5s %-16s %-8s %7s %8s %7s %9s %8s" % ("entry", "model", "status", "TM", "aligned", "RMSD", "coverage", "seq id"))
    for e in STATE["entries"].values():
        for m in e["models"]:
            log("%-5s %-16s %-8s %7s %8s %7s %9s %8s" % (m["entry"], m["model_kind"], m["status"], m["tm_score"], m["aligned_ca_length"], m["ca_rmsd"], m["coverage"], m["seq_identity"]))
        log("%-5s ENTRY %s%s; seconds: %s" % (e["entry"], e["status"].upper(), (" (stopped in %s: %s)" % (e["failed_step"], e["note"][:120])) if e["failed_step"] else "",
                                             {k: v["seconds"] for k, v in e["steps"].items()}))


if STATE["ok"] and STATE["usalign"]:
    for entry in ENTRIES:
        try:
            process_entry(entry)
        except Exception:
            import traceback
            STATE["entries"].setdefault(entry, {"entry": entry, "steps": {}, "models": []}).update(status="failed", failed_step="unexpected error", note=traceback.format_exc()[-600:])
            log("\nENTRY %s STOPPED by an unexpected error:\n%s" % (entry, traceback.format_exc()))
    write_csv("pilot_metrics.csv", METRIC_COLUMNS, [m for e in STATE["entries"].values() for m in e["models"]])
    write_csv("entry_status.csv", ["entry", "status", "failed_step", "note", "seconds", "map_sha256"], list(STATE["entries"].values()))
    summary_table()
    STATE["complete"] = bool(STATE["entries"]) and all(e["status"] == "ok" for e in STATE["entries"].values())
    if not STATE["complete"]:
        STATE["stopped_at"] = ", ".join("%s: %s" % (e["entry"], e["failed_step"]) for e in STATE["entries"].values() if e["status"] != "ok")
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

EXTRA_INFO = ('{"modelangelo_commit": globals().get("MA_COMMIT", ""), "modelangelo_version": STATE.get("modelangelo_version", ""), "bundle": globals().get("MA_BUNDLE", ""), '
              '"weights": STATE.get("weights", {}), "env": STATE["env"], "usalign_commit": globals().get("USALIGN_COMMIT", ""), "usalign_version": STATE["usalign_version"], '
              '"input_sha256": globals().get("EXPECTED", {})}')


def build(out_path, entries=("26DK",), ma_repo=MA_REPO, ma_commit=MA_COMMIT, usalign_repo=P.USALIGN_REPO, usalign_commit=P.USALIGN_COMMIT, entries_path=None,
          env_python=None, usalign_bin=None, net_hosts=NET_HOSTS, require_gpu=True, import_test=IMPORT_TEST):
    """Writes the notebook. The keyword arguments other than `entries` exist for the tests (a local repository, a ready environment, no GPU)."""
    file_cells, expected = nbkit.file_cells(FILES, entries_path)
    step5_filled_for_functions = nbkit.fill(P.STEP5, {"@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit, "@MICA_REPO@": "x", "@MICA_COMMIT@": "x",
                                                      "@WEIGHTS_URL@": "x", "@ZENODO_API@": "x", "@USALIGN_BIN@": "None"})
    git_and_usalign = nbkit.function_sources(step5_filled_for_functions, ["git_checkout", "build_usalign"]).rstrip("\n")
    step6_filled_for_functions = nbkit.fill(P.STEP6, {"@ENTRIES@": "[]"})
    read_write_download = nbkit.function_sources(step6_filled_for_functions, ["read_csv", "write_csv", "download"]).rstrip("\n")
    mapping = {"@UV_VERSION@": P.UV_VERSION, "@MA_PY_VERSION@": MA_PY_VERSION, "@ENV_PYTHON@": repr(env_python), "@IMPORT_TEST@": repr(import_test),
               "@MA_REPO@": ma_repo, "@MA_COMMIT@": ma_commit, "@MA_BUNDLE@": MA_BUNDLE, "@USALIGN_REPO@": usalign_repo, "@USALIGN_COMMIT@": usalign_commit,
               "@USALIGN_BIN@": repr(usalign_bin), "@ENTRIES@": repr(list(entries)), "@ENTRY_LIST@": ", ".join(entries), "@REQUIRE_GPU@": repr(bool(require_gpu)),
               "@NET_HOSTS@": repr(tuple(net_hosts)), "@EXPECTED@": repr(expected), "@NOTEBOOK@": NOTEBOOK, "@EXTRA_INFO@": EXTRA_INFO}
    fl = lambda text: nbkit.fill(text, mapping).strip("\n") + "\n"
    step4 = STEP4.replace("@GIT_AND_USALIGN@", git_and_usalign)
    step5 = STEP5.replace("@READ_WRITE_DOWNLOAD@", read_write_download)
    cells = [("intro", nbkit.md(nbkit.fill(INTRO, mapping))),
             ("step1-title", nbkit.md(P.STEP1_MD)), ("step1", nbkit.code(fl(P.STEP1))),
             ("step2-title", nbkit.md(P.STEP2_MD)), ("step2-dirs", nbkit.code(P.STEP2_DIRS.strip("\n") + "\n"))]
    cells += file_cells
    cells += [("step2-check", nbkit.code(fl(P.STEP2_CHECK))),
              ("robust-title", nbkit.md(nbkit.fill(nbkit.ROBUST_MD, mapping))), ("robust", nbkit.code(fl(nbkit.ROBUST))),
              ("step3-title", nbkit.md(STEP3_MD)), ("step3", nbkit.code(fl(STEP3))),
              ("step4-title", nbkit.md(STEP4_MD)), ("step4", nbkit.code(fl(step4))),
              ("step5-title", nbkit.md(STEP5_MD)), ("step5", nbkit.code(fl(step5))),
              ("step6-title", nbkit.md(STEP6_MD)), ("step6", nbkit.code(fl(STEP6))),
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
