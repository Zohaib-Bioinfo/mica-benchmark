#!/usr/bin/env python3
"""Unpacks the result zips of the Kaggle notebooks into results/raw/ (text files only) and keeps an index of what was read.

    python scripts/ingest_results.py <zip> [<zip> ...] --out results/raw
    python scripts/ingest_results.py --list method_status results/raw       # the status tables, oldest first (the order that analyze_results.py wants)

A result zip is named `<notebook>_result_<YYYYMMDD-HHMM>[_INCOMPLETE].zip`. Its folder is unpacked to `results/raw/<zip name without .zip>/`. Only small text files are kept
(.csv .json .txt .log .md .tsv .diff): the tables, the logs of the programs and the raw output of the scoring tools. Coordinate files (models of the methods and of MICA), maps, weights
and archives are never written to the repository: they are listed in the index as skipped, with their suffix. A name that would leave the folder (`..`, an absolute path) is refused.
`INDEX.csv` has one line per zip: the folder, the notebook, the time stamp, whether the zip is incomplete, the sha256 of the zip, the number of files kept and skipped.
A folder that holds a file `NOT_A_RESULT.txt` (the reason, in words) is a session whose runs did not test the method (an error of the set-up, for example a notebook that lacked a data file of the
method): it stays in the repository as part of the record, but `--list` does not give its status tables to the analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import zipfile

ALLOWED_SUFFIXES = (".csv", ".json", ".txt", ".log", ".md", ".tsv", ".diff")        # .diff: the changes that a notebook made to the source of a method (added 2026-11-01)
MAX_FILE_BYTES = 80 * 1000 * 1000
MAX_TOTAL_BYTES = 800 * 1000 * 1000
NAME = re.compile(r"^(?P<notebook>.+)_result_(?P<stamp>\d{8}-\d{4})(?P<incomplete>_INCOMPLETE)?\.zip$")
NOT_A_RESULT = "NOT_A_RESULT.txt"
INDEX_COLUMNS = ["folder", "notebook", "stamp", "incomplete", "zip_sha256", "zip_bytes", "files_kept", "files_skipped", "skipped_suffixes"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    """The path of a zip member below the folder, or None when it is not safe."""
    name = name.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def ingest(zip_path, out_root):
    """Unpacks one zip. Returns the index row."""
    name = os.path.basename(zip_path)
    m = NAME.match(name)
    if not m:
        raise ValueError("%s is not a result zip (<notebook>_result_<YYYYMMDD-HHMM>[_INCOMPLETE].zip)" % name)
    folder = os.path.join(out_root, name[:-4])
    kept, skipped, total = 0, {}, 0
    with zipfile.ZipFile(zip_path) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        for info in members:
            if safe_relative(info.filename) is None:
                raise ValueError("%s: the member %r would leave the folder; nothing was written" % (name, info.filename))
        tops = {safe_relative(i.filename).split("/", 1)[0] for i in members}
        strip = len(tops) == 1 and all("/" in safe_relative(i.filename) for i in members)
        for info in members:
            rel = safe_relative(info.filename)
            if strip:
                rel = rel.split("/", 1)[1]
            suffix = os.path.splitext(rel)[1].lower() or "(none)"
            if suffix not in ALLOWED_SUFFIXES:
                skipped[suffix] = skipped.get(suffix, 0) + 1
                continue
            if info.file_size > MAX_FILE_BYTES or total + info.file_size > MAX_TOTAL_BYTES:
                skipped["(too large)"] = skipped.get("(too large)", 0) + 1
                continue
            target = os.path.join(folder, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            total += info.file_size
            kept += 1
    return {"folder": name[:-4], "notebook": m.group("notebook"), "stamp": m.group("stamp"), "incomplete": bool(m.group("incomplete")), "zip_sha256": sha256_of(zip_path),
            "zip_bytes": os.path.getsize(zip_path), "files_kept": kept, "files_skipped": sum(skipped.values()),
            "skipped_suffixes": ";".join("%s:%d" % kv for kv in sorted(skipped.items()))}


def read_index(out_root):
    path = os.path.join(out_root, "INDEX.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_index(out_root, rows):
    rows = sorted(rows, key=lambda r: (r["stamp"], r["notebook"]))
    with open(os.path.join(out_root, "INDEX.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def status_files(out_root, name):
    """The paths of `name` (method_status.csv or mica_status.csv) in every unpacked result folder, oldest first; the folders that hold a `NOT_A_RESULT.txt` are left out."""
    rows = sorted(read_index(out_root), key=lambda r: (r["stamp"], r["notebook"]))
    rows = [r for r in rows if not os.path.exists(os.path.join(out_root, r["folder"], NOT_A_RESULT))]
    paths = [os.path.join(out_root, r["folder"], name) for r in rows]
    return [p for p in paths if os.path.exists(p)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zips", nargs="*")
    p.add_argument("--out", default="results/raw")
    p.add_argument("--list", nargs=2, metavar=("TABLE", "RAW"), help="print the paths of TABLE (method_status or mica_status) below RAW, oldest first")
    args = p.parse_args(argv)
    if args.list:
        table, raw = args.list
        print("\n".join(status_files(raw, table if table.endswith(".csv") else table + ".csv")))
        return 0
    if not args.zips:
        p.error("no zip given")
    os.makedirs(args.out, exist_ok=True)
    index = {r["folder"]: r for r in read_index(args.out)}
    for path in args.zips:
        row = ingest(path, args.out)
        index[row["folder"]] = row
        print("%s: %d files kept, %d skipped (%s)%s" % (row["folder"], row["files_kept"], row["files_skipped"], row["skipped_suffixes"] or "-", "  INCOMPLETE" if row["incomplete"] else ""))
    write_index(args.out, list(index.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
