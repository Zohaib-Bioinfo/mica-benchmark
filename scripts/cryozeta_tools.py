#!/usr/bin/env python3
"""CryoZeta helpers of the MICA benchmark (added 2026-11-01 at the request of Prof. Cheng): the parts of the CryoZeta notebook that can be tested without a GPU.

Standard library only. The notebook runs this file with its own Python; CryoZeta itself runs in its own pixi environment (CUDA 11.8 on a T4).

    colabfold_msa()           the MSA of one sequence from the public ColabFold server, written as the two files CryoZeta reads
    emdb_contour_level()      the recommended contour level of a map, from the EMDB header
    write_input_json()        the input file of CryoZeta for one entry
    patch_detection_precision()   the only change of the CryoZeta source that is ever made, and only on a GPU that cannot do bfloat16 (see the function)
    run_entry()               one entry: detection -> CryoZeta -> CryoZeta-Interpolate -> combine, with the fall-backs of the memory ladder

    prepare_input()           the contour level, the MSA of every chain and the input file of one entry (stage 0 of the runner)

    python cryozeta_tools.py run <config.json>        # what the notebook calls for every entry

Why a runner of its own: `cryozeta-inference` catches the error of a sample, writes it to a file and ends with exit code 0, so the exit code says nothing. The runner looks at the files.
The settings of the authors' `inference_demo.sh` (5 samples, 20 diffusion steps, 10 cycles, seed 101) are kept; only the memory-saving knobs may change, in this order:
the attention chunk size (256, 64, 16, 4; it changes the memory and the time, not the algorithm), and the number format (bfloat16, then float32) if a run fails for another reason.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

COLABFOLD_HOST = "https://api.colabfold.com"
USER_AGENT = "mica-benchmark-cryozeta/1.0"
AMINO = frozenset("ACDEFGHIKLMNPQRSTVWY")
CHUNK_LADDER = (256, 64, 16, 4)                      # `--infer_setting.chunk_size`; 256 is the default of CryoZeta


class MsaError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------------------------ MSA
def read_a3m(text):
    """[(header, sequence)] of an a3m text. The NUL characters that the server puts between the results of several queries are removed; wrapped sequences are joined."""
    records, header, parts = [], None, []
    for line in text.replace("\x00", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header, parts = line[1:], []
        elif header is not None:
            parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def write_a3m(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(">%s\n%s\n" % (header, seq))


def _http(url, data=None, timeout=60.0, tries=3, wait=5.0, sleep=time.sleep):
    body = urllib.parse.urlencode(data).encode("ascii") if data is not None else None
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as ex:
            last = ex
            if attempt < tries:
                sleep(wait * attempt)
    raise MsaError("%s: %s" % (url, last))


def _json(raw):
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        raise MsaError("the MSA server answered with something that is not JSON: %r" % raw[:200])


def _colabfold_once(sequence, host, poll_s, max_wait_s, log, sleep):
    t0 = time.time()
    deadline = t0 + max_wait_s
    while True:                                       # the server may ask us to wait (RATELIMIT) or not know the query yet (UNKNOWN): ask again
        out = _json(_http(host + "/ticket/msa", {"q": ">101\n%s\n" % sequence, "mode": "env"}))
        status = out.get("status")
        if status not in ("UNKNOWN", "RATELIMIT"):
            break
        if time.time() > deadline:
            raise MsaError("the server did not take the query in time (status %s)" % status)
        sleep(poll_s)
    if status in ("ERROR", "MAINTENANCE"):
        raise MsaError("the server answered %s" % status)
    ticket = out.get("id")
    if not ticket:
        raise MsaError("the server gave no ticket: %s" % str(out)[:200])
    while status in ("PENDING", "RUNNING", "UNKNOWN"):
        if time.time() > deadline:
            raise MsaError("the job %s did not finish in %d s (status %s)" % (ticket, max_wait_s, status))
        sleep(poll_s)
        out = _json(_http("%s/ticket/%s" % (host, ticket)))
        status = out.get("status")
    if status != "COMPLETE":
        raise MsaError("the job %s ended with status %s" % (ticket, status))
    blob = _http("%s/result/download/%s" % (host, ticket), timeout=300.0)
    try:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
    except tarfile.TarError as ex:
        raise MsaError("the result of job %s is not an archive: %s" % (ticket, ex))
    texts = {}
    for member in tf.getmembers():                     # read two members into memory; nothing is extracted to disk
        base = os.path.basename(member.name)
        if member.isfile() and (base == "uniref.a3m" or base.startswith("bfd.mgnify")):
            texts[base] = tf.extractfile(member).read().decode("utf-8", "replace")
    if "uniref.a3m" not in texts:
        raise MsaError("the result of job %s has no uniref.a3m (it has: %s)" % (ticket, ", ".join(sorted(m.name for m in tf.getmembers()))[:300]))
    uni = read_a3m(texts["uniref.a3m"])
    env_name = next((k for k in texts if k.startswith("bfd.mgnify")), None)
    env = read_a3m(texts[env_name]) if env_name else []
    for label, recs in (("uniref.a3m", uni), (env_name, env)):
        if recs and recs[0][1].replace("-", "").upper() != sequence:
            raise MsaError("the first record of %s is not the query" % label)
    return {"ticket": ticket, "uniref": uni, "env": env, "seconds": round(time.time() - t0, 1)}


def colabfold_msa(sequence, out_dir, host=COLABFOLD_HOST, poll_s=8.0, max_wait_s=2400.0, tries=3, log=print, sleep=time.sleep):
    """The MSA of one protein sequence from the ColabFold server (UniRef30 and the environmental databases, the server's `env` mode), as the two files that CryoZeta reads:
    `mmseqs_other_hits.a3m` (the query, the UniRef hits, the environmental hits) and `uniref100_hits.a3m` (the query and the UniRef hits; CryoZeta reads it only to pair the chains
    of a complex of different sequences, and the server's headers carry no species, so no pairs are found; the README of CryoZeta says that it ships no MSA tool and names the
    ColabFold search script). The result is kept in `out_dir`; the same sequence is not asked for twice. Returns the info that is also written to msa_info.json."""
    sequence = "".join(sequence.split()).upper()
    bad = sorted(set(sequence) - AMINO)
    if not sequence or bad:
        raise MsaError("the MSA server takes only the 20 standard amino acids; this sequence has %s" % (bad or "no residues"))
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    other, uni_path, info_path = (os.path.join(out_dir, n) for n in ("mmseqs_other_hits.a3m", "uniref100_hits.a3m", "msa_info.json"))
    if all(os.path.exists(p) for p in (other, uni_path, info_path)):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        if info.get("sequence_sha256") == digest:
            log("MSA of %d residues: already here (%d UniRef and %d environmental hits)" % (len(sequence), info["n_uniref_hits"], info["n_env_hits"]))
            return info
    os.makedirs(out_dir, exist_ok=True)
    last = None
    for attempt in range(1, tries + 1):
        try:
            got = _colabfold_once(sequence, host, poll_s, max_wait_s, log, sleep)
            break
        except MsaError as ex:
            last = ex
            log("MSA server, attempt %d of %d: %s" % (attempt, tries, ex))
            if attempt < tries:
                sleep(30.0 * attempt)
    else:
        raise MsaError("the MSA server gave no alignment after %d attempts: %s" % (tries, last))
    uni, env = got["uniref"], got["env"]
    write_a3m(uni, uni_path)
    write_a3m([uni[0]] + uni[1:] + env[1:], other)
    info = {"host": host, "ticket": got["ticket"], "mode": "env", "sequence_length": len(sequence), "sequence_sha256": digest, "n_uniref_hits": len(uni) - 1, "n_env_hits": max(len(env) - 1, 0),
            "server_seconds": got["seconds"], "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    log("MSA of %d residues from the ColabFold server: %d UniRef and %d environmental hits (%.0f s)" % (len(sequence), info["n_uniref_hits"], info["n_env_hits"], got["seconds"]))
    return info


# ------------------------------------------------------------------------------------------------------------------------------ the input of CryoZeta
def emdb_header_url(map_url):
    """https://.../EMD-55586/map/emd_55586.map.gz -> https://.../EMD-55586/header/emd-55586.xml"""
    m = re.match(r"^(.*/EMD-(\d+))/map/emd_\2\.map(?:\.gz)?$", map_url)
    if not m:
        raise ValueError("not the address of an EMDB map: %s" % map_url)
    return "%s/header/emd-%s.xml" % (m.group(1), m.group(2))


def emdb_contour_level(xml_text):
    """The recommended contour level of the map from the EMDB header (the primary contour, else the first); None if the header has none."""
    root = ET.fromstring(xml_text)
    found = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "contour":
            continue
        for child in el:
            if child.tag.split("}")[-1] == "level" and child.text and child.text.strip():
                found.append((el.get("primary", "").lower() == "true", float(child.text)))
    if not found:
        return None
    return next((lev for primary, lev in found if primary), found[0][1])


def write_input_json(path, name, map_path, resolution, contour_level, chains):
    """The input of CryoZeta for one entry. chains: [{"sequence": ..., "count": copies, "msa_dir": folder with the two a3m files}], all of them proteins."""
    entry = {"name": name, "modelSeeds": [], "map_path": os.path.abspath(map_path), "resolution": float(resolution), "contour_level": float(contour_level),
             "sequences": [{"proteinChain": {"sequence": c["sequence"], "count": int(c["count"]),
                                             "msa": {"precomputed_msa_dir": os.path.abspath(c["msa_dir"]), "pairing_db": "uniref100"}}} for c in chains]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([entry], f, indent=2)
    return entry


def prepare_input(cfg, say, sleep=time.sleep):
    """Stage 0 of one entry: the recommended contour level (EMDB header, unless cfg has `contour_level`), the MSA of every chain (ColabFold server) and the input file of CryoZeta.
    cfg: name, map_path, resolution, header_url (or contour_level), chains [{"sequence", "count"}], msa_root, input_json, optional colabfold_host. Returns the facts for the summary."""
    info = {"chains": [], "contour_level": None, "contour_level_source": ""}
    level = cfg.get("contour_level")
    if level is None:
        raw = _http(cfg["header_url"], timeout=60.0, sleep=sleep)
        level = emdb_contour_level(raw.decode("utf-8", "replace"))
        if level is None:
            raise ValueError("the EMDB header %s has no recommended contour level" % cfg["header_url"])
        info["contour_level_source"] = cfg["header_url"]
    else:
        info["contour_level_source"] = "given"
    info["contour_level"] = float(level)
    say("contour level of %s: %s (%s)" % (cfg["name"], level, info["contour_level_source"]))
    chains = []
    for i, c in enumerate(cfg["chains"], 1):
        folder = os.path.join(cfg["msa_root"], "chain%d" % i)
        msa = colabfold_msa(c["sequence"], folder, host=cfg.get("colabfold_host") or COLABFOLD_HOST, poll_s=float(cfg.get("msa_poll_s", 8.0)), tries=int(cfg.get("msa_tries", 3)), log=say, sleep=sleep)
        chains.append({"sequence": c["sequence"], "count": c["count"], "msa_dir": folder})
        info["chains"].append({"chain": i, "length": len(c["sequence"]), "copies": int(c["count"]), "n_uniref_hits": msa["n_uniref_hits"], "n_env_hits": msa["n_env_hits"],
                               "ticket": msa.get("ticket", ""), "msa_dir": folder})
    write_input_json(cfg["input_json"], cfg["name"], cfg["map_path"], cfg["resolution"], level, chains)
    return info


# ------------------------------------------------------------------------------------------------------------------------------ the one change of the source
DETECTION_FILE = os.path.join("src", "cryozeta", "runner", "inference_detection.py")
DETECTION_CHANGES = (("self.detection_model.to(self.device).to(torch.bfloat16)", "self.detection_model.to(self.device).to(torch.float32)"),
                     ("amp_dtype = torch.bfloat16", "amp_dtype = torch.float32"))


def patch_detection_precision(repo):
    """The detection network of CryoZeta is converted to bfloat16 and run under bfloat16 autocast without a choice (two lines of `inference_detection.py`). A GPU that cannot do
    bfloat16 (the preflight of the notebook tests a bfloat16 matrix product and 3D convolution) would stop there; on such a GPU the two lines are changed to float32. Nothing else
    is touched, and never on a GPU that can do bfloat16. Every line must occur exactly once (otherwise the source is not the pinned commit). Returns the changed text of the diff."""
    path = os.path.join(repo, DETECTION_FILE)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new = text
    lines = []
    for old, repl in DETECTION_CHANGES:
        if new.count(repl) == 1 and new.count(old) == 0:
            continue                                                   # changed already
        if new.count(old) != 1:
            raise RuntimeError("%s: the line `%s` occurs %d times (expected once): the source is not the pinned commit" % (DETECTION_FILE, old, new.count(old)))
        new = new.replace(old, repl)
        lines.append("-%s\n+%s" % (old, repl))
    if new != text:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------------------ the runner of one entry
class GpuPeak(threading.Thread):
    """Polls nvidia-smi for the memory used on one GPU and keeps the highest value (MB)."""

    def __init__(self, index, every=2.0):
        super().__init__(daemon=True)
        self.index, self.every, self.peak, self.total = index, every, None, None
        self._halt = threading.Event()

    def run(self):
        if not shutil.which("nvidia-smi"):
            return
        while not self._halt.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", str(self.index)],
                                     capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
                used, total = [int(x) for x in out[0].split(",")]
                self.peak = used if self.peak is None else max(self.peak, used)
                self.total = total
            except Exception:
                pass
            self._halt.wait(self.every)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


def _physical_gpu():
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


def stage_env(cfg):
    """The environment of the CryoZeta programs: what `env_setup.sh` (the activation script of the pixi project) sets, without pixi."""
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "LAYERNORM_TYPE", "PIXI_ENVIRONMENT_NAME"):
        env.pop(k, None)
    prefix = cfg["env_dir"]
    env["CONDA_PREFIX"] = prefix
    env["PATH"] = os.path.join(prefix, "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = os.path.join(prefix, "lib") + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["CUTLASS_PATH"] = prefix
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["USE_OPM_CHUNKED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if cfg.get("assets"):                                                  # where the package looks for the chemical components files (`CRYOZETA_ASSETS_DIR`, as in the demo scripts): without it
        env["CRYOZETA_ASSETS_DIR"] = cfg["assets"]                         # it looks next to its own files (`assets/` of a source checkout), finds nothing, and the program ends with exit code 0 and no model
    teaser = cfg.get("teaser_dir")
    if teaser:
        env["TEASERPP_DIR"] = teaser
        env["PYTHONPATH"] = os.path.join(teaser, "build", "python")
    if cfg.get("kernels") == "demo":                                       # the settings of inference_demo.sh (a GPU of the Ampere generation or later, a CUDA compiler)
        env["LAYERNORM_TYPE"] = "fast_layernorm"
        env["CUDA_HOME"] = prefix
    return env


def _flag(on):
    return "true" if on else "false"


def detection_cmd(cfg):
    return [os.path.join(cfg["env_dir"], "bin", "cryozeta-detection"), "json-run", cfg["input_json"], cfg["dump_dir"], "--device", "cuda",
            "--checkpoint-path", os.path.join(cfg["assets"], cfg["weights"]["detection"])]


def inference_cmd(cfg, interpolate, dtype, chunk):
    kernels = cfg.get("kernels") == "demo"
    return [os.path.join(cfg["env_dir"], "bin", "cryozeta-inference"), "--seeds", str(cfg["seed"]),
            "--load_checkpoint_path", os.path.join(cfg["assets"], cfg["weights"]["interpolate" if interpolate else "cryozeta"]),
            "--em_file_dir", cfg["dump_dir"], "--dump_dir", cfg["dump_dir"], "--input_json_path", cfg["input_json"],
            "--use_deepspeed_evo_attention", _flag(kernels), "--use_cuequivariance_attention", _flag(kernels), "--use_cuequivariance_multiplicative_update", _flag(kernels),
            "--use_cuequivariance_attention_pair_bias", _flag(kernels), "--use_opm_tilelang", "false",
            "--model.N_cycle", str(cfg["n_cycle"]), "--sample_diffusion.N_sample", str(cfg["n_sample"]), "--sample_diffusion.N_step", str(cfg["n_step"]),
            "--data.num_dl_workers", "1", "--use_interpolation", _flag(interpolate), "--overwrite", "true",
            "--dtype", dtype, "--infer_setting.chunk_size", str(chunk)]


def combine_cmd(cfg):
    return [os.path.join(cfg["env_dir"], "bin", "cryozeta-combine"), "--dump-dir", cfg["dump_dir"], "--input-json-path", cfg["input_json"], "--seeds", str(cfg["seed"]),
            "--num-select", str(cfg["num_select"])]


def run_stage(cfg, label, cmd, env, say):
    """Runs one program, streams its output (and writes it to logs/<label>.log), polls the GPU memory. Returns what happened; `oom` is true if the output says so anywhere."""
    log_dir = os.path.join(cfg["dump_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    gpu = GpuPeak(_physical_gpu())
    gpu.start()
    t0 = time.time()
    tail, oom, bars, last_progress = collections.deque(maxlen=60), False, 0, 0.0
    say("\n=== stage %s: %s" % (label, " ".join(os.path.basename(cmd[0]) if i == 0 else a for i, a in enumerate(cmd))[:500]))
    try:
        with open(os.path.join(log_dir, "%s.log" % label), "a", encoding="utf-8") as lf:
            lf.write("$ %s\n" % " ".join(cmd))
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=cfg["repo"], errors="replace")
            except OSError as ex:
                msg = "could not start %s: %s" % (cmd[0], ex)
                lf.write(msg + "\n")
                say(msg)
                return {"stage": label, "rc": 127, "seconds": 0.0, "gpu_peak_mb": None, "oom": False, "tail": [msg]}
            for line in p.stdout:
                line = line.rstrip("\n")
                lf.write(line + "\n")
                lf.flush()
                tail.append(line[:400])
                if "out of memory" in line.lower():
                    oom = True
                if "%|" in line:
                    bars += 1
                    if bars % 50 != 1:
                        continue
                say(line[:400])
                if time.time() - last_progress > 1.0:
                    last_progress = time.time()
                    _progress(cfg, "%s: %d s | %s" % (label, time.time() - t0, line.strip()[:160]))
            p.wait()
            p.stdout.close()
    finally:
        gpu.stop()
    return {"stage": label, "rc": p.returncode, "seconds": round(time.time() - t0, 1), "gpu_peak_mb": gpu.peak, "gpu_total_mb": gpu.total, "oom": oom, "tail": list(tail)}


def _progress(cfg, text):
    try:
        with open(cfg["progress_path"], "w", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        pass


def _entry_dir(cfg, *parts):
    return os.path.join(cfg["dump_dir"], cfg["name"], *parts)


def _predictions(cfg, stage_dir):
    return sorted(glob.glob(os.path.join(_entry_dir(cfg, stage_dir, "seed_%s" % cfg["seed"], "predictions"), "*_sample_*.cif")))


def _error_text(cfg, stage_dir):
    p = _entry_dir(cfg, stage_dir, "ERR", "%s.txt" % cfg["name"])
    if os.path.exists(p):
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


def _inference(cfg, interpolate, state, env, say, attempts):
    """CryoZeta (or CryoZeta-Interpolate) with the memory ladder; returns True when the predictions exist. `state` carries the settings that worked to the next stage."""
    label = "interpolate" if interpolate else "cryozeta"
    stage_dir = "CryoZeta-Interpolate" if interpolate else "CryoZeta"
    ladder = list(cfg.get("chunk_sizes") or CHUNK_LADDER)
    while len(attempts) < 8:
        chunk = ladder[state["chunk_index"]]
        shutil.rmtree(_entry_dir(cfg, stage_dir), ignore_errors=True)       # the files of an earlier attempt (its error file above all) must not be taken for the result of this one
        res = run_stage(cfg, "%s_try%d" % (label, len([a for a in attempts if a["stage"] == label]) + 1), inference_cmd(cfg, interpolate, state["dtype"], chunk), env, say)
        files = _predictions(cfg, stage_dir)
        err = _error_text(cfg, stage_dir)
        oom = res["oom"] or "out of memory" in err.lower()
        skipped = interpolate and not files and not err and any("no interpolation features" in ln for ln in res["tail"])
        ok = bool(files) and res["rc"] == 0 and not err
        attempts.append({"stage": label, "dtype": state["dtype"], "chunk_size": chunk, "rc": res["rc"], "seconds": res["seconds"], "gpu_peak_mb": res["gpu_peak_mb"], "oom": oom,
                         "predictions": len(files), "error": err[-600:] if err else "", "skipped": skipped})
        if ok or skipped:
            return True
        res["error_text"] = err
        attempts[-1]["tail"] = res["tail"][-12:]
        if res["rc"] == -9 and not oom:                                       # killed by the system (no RAM left): a smaller chunk or another number format does not help; the scheduler runs it again alone
            return False
        if oom and state["chunk_index"] + 1 < len(ladder):
            state["chunk_index"] += 1
            say("%s: out of GPU memory; the next attempt uses attention chunk size %d" % (label, ladder[state["chunk_index"]]))
            continue
        if not oom and state["dtype"] != "fp32":
            state["dtype"] = "fp32"
            say("%s: failed (not for memory); the next attempt computes in float32" % label)
            continue
        return False
    return False


def _copy_small_files(cfg):
    dst = os.path.join(cfg["dump_dir"], "summary_files")
    os.makedirs(dst, exist_ok=True)
    pairs = [(_entry_dir(cfg, "CryoZeta-Detection", "%s_timing.txt" % cfg["name"]), "detection_timing.txt"),
             (_entry_dir(cfg, "CryoZeta", "saved_data", "scores.csv"), "cryozeta_scores.csv"),
             (_entry_dir(cfg, "CryoZeta-Interpolate", "saved_data", "scores.csv"), "interpolate_scores.csv")]
    for stage_dir, tag in (("CryoZeta", "cryozeta"), ("CryoZeta-Interpolate", "interpolate")):
        for p in sorted(glob.glob(os.path.join(_entry_dir(cfg, stage_dir, "seed_%s" % cfg["seed"], "predictions"), "*summary_confidence*.json")))[:10]:
            pairs.append((p, "%s_%s" % (tag, os.path.basename(p))))
    for src, name in pairs:
        if os.path.exists(src) and os.path.getsize(src) < 2000000:
            shutil.copyfile(src, os.path.join(dst, name))


def run_entry(cfg, say=None):
    """One entry from the detection to the final models. Writes driver_summary.json; returns (exit code, summary): 0 the final models exist, 2 failed for lack of GPU memory, 3 failed, 4 a program was killed by the system (exit code -9: no RAM left)."""
    say = say or (lambda s: print(s, flush=True))
    t0 = time.time()
    os.makedirs(cfg["dump_dir"], exist_ok=True)
    env = stage_env(cfg)
    state = {"dtype": cfg["dtype"], "chunk_index": 0}
    stages, attempts = [], []
    summary = {"name": cfg["name"], "status": "failed", "failed_stage": "", "note": "", "settings": {k: cfg.get(k) for k in ("seed", "n_sample", "n_step", "n_cycle", "num_select", "dtype", "kernels", "chunk_sizes")},
               "stages": stages, "attempts": attempts}
    failing = None
    _progress(cfg, "input: contour level and MSA")
    try:
        summary["input"] = prepare_input(cfg, say)
    except Exception as ex:                                                  # the server, the header or the sequence: a failure of the input, not of the method
        failing = ("input", {"tail": [str(ex)[:400]], "oom": False})
        summary["note"] = "input: %s: %s" % (type(ex).__name__, str(ex)[:400])
    if failing is None:
        _progress(cfg, "detection: starting")
        det = run_stage(cfg, "detection", detection_cmd(cfg), env, say)
        stages.append({k: v for k, v in det.items() if k != "tail"})
        if det["rc"] != 0 or not os.path.exists(_entry_dir(cfg, "CryoZeta-Detection", "%s.pt" % cfg["name"])):
            failing = ("detection", det)
            summary["note"] = "detection: exit code %s%s%s" % (det["rc"], "; no detection file" if det["rc"] == 0 else "", "; CUDA out of memory" if det["oom"] else "")
    if failing is None:
        for interpolate in (False, True):
            label = "interpolate" if interpolate else "cryozeta"
            _progress(cfg, "%s: starting" % label)
            ok = _inference(cfg, interpolate, state, env, say, attempts)
            if not ok:
                last = attempts[-1]
                failing = (label, {"tail": last.get("tail", []), "oom": last["oom"], "rc": last.get("rc")})
                summary["note"] = "%s: no predictions after %d attempt(s) (%s)" % (label, len([a for a in attempts if a["stage"] == label]), "CUDA out of memory" if last["oom"] else (last["error"][-200:] or "exit code %s" % last["rc"]))
                break
    if failing is None:
        _progress(cfg, "combine: starting")
        comb = run_stage(cfg, "combine", combine_cmd(cfg), env, say)
        stages.append({k: v for k, v in comb.items() if k != "tail"})
        final = _entry_dir(cfg, "CryoZeta-Final", "%s_sample_0.cif" % cfg["name"])
        if comb["rc"] != 0 or not os.path.exists(final):
            failing = ("combine", comb)
            summary["note"] = "combine: exit code %s%s" % (comb["rc"], "; no final model" if comb["rc"] == 0 else "")
    _copy_small_files(cfg)
    summary["seconds"] = round(time.time() - t0, 1)
    summary["final_settings"] = dict(state, chunk_size=(cfg.get("chunk_sizes") or list(CHUNK_LADDER))[state["chunk_index"]])
    summary["gpu_peak_mb"] = max([s.get("gpu_peak_mb") or 0 for s in stages + attempts] or [0]) or None
    if failing is None:
        summary["status"] = "ok"
        code = 0
        say("\nDRIVER RESULT ok: %s, %.0f s, GPU memory peak %s MB, settings %s" % (cfg["name"], summary["seconds"], summary["gpu_peak_mb"], summary["final_settings"]))
    else:
        summary["failed_stage"] = failing[0]
        oom = bool(failing[1].get("oom"))
        code = 2 if oom else (4 if failing[1].get("rc") == -9 else 3)
        say("\nDRIVER RESULT failed: %s. The last lines of the stage:" % summary["note"])
        for line in (failing[1].get("tail") or [])[-25:]:
            say("    " + line)
        say("DRIVER RESULT failed: %s" % summary["note"])
    with open(cfg["summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _progress(cfg, "done: %s" % summary["status"])
    return code, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one entry, from a JSON file of settings")
    r.add_argument("config")
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    code, _summary = run_entry(cfg)
    return code


if __name__ == "__main__":
    sys.exit(main())
