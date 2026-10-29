#!/usr/bin/env python3
"""Compile the report and refuse to produce a PDF that is not finished:

    python3 scripts/build_report.py                      # report/report.tex -> report/MICA_benchmark_report.pdf
    python3 scripts/build_report.py --report report --out report/MICA_benchmark_report.pdf

Runs pdflatex, bibtex, pdflatex, pdflatex in `report/build/`. Before the run it checks the sources (placeholder in `config.tex`, `TODO` and similar marks in the text, a number macro
that the text uses and that has no value). After the run it reads the log and stops (exit code 1) on a LaTeX error, an undefined macro, reference or citation, or a label defined twice;
overfull boxes are listed (they do not stop the build). The numbers of the text are `\\newcommand`s in `report/generated/numbers.tex`, written by `make_report_tables.py`.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

MARKS = re.compile(r"\bTODO\b|\bTBD\b|\bFIXME\b|XXX|\?\?\?")
EMPTY_VALUES = {"", "n/a", "nan", "NaN", "--", "-"}
STANDARD_PLACEHOLDER = re.compile(r"(<[^>]*>|your[-_ ]?(name|repo|user|username)|example\.com|github\.com/(user|username|you)/)", re.I)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def number_macros(report):
    """{name: value} of the `\\newcommand{\\Name}{value}` lines of generated/numbers.tex."""
    path = os.path.join(report, "generated", "numbers.tex")
    if not os.path.exists(path):
        return None
    return dict(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", read(path), flags=re.M))


def text_files(report):
    files = [os.path.join(report, "report.tex"), os.path.join(report, "config.tex")]
    return [f for f in files if os.path.exists(f)] + sorted(glob.glob(os.path.join(report, "sections", "*.tex")))


def check_sources(report):
    """Problems in the sources that make a report unfinished (a list of strings; empty = fine)."""
    problems = []
    cfg = os.path.join(report, "config.tex")
    if not os.path.exists(cfg):
        return ["config.tex is missing"]
    m = re.search(r"\\newcommand\{\\RepoURL\}\{([^}]*)\}", read(cfg))
    if not m or not m.group(1).strip():
        problems.append("config.tex: \\RepoURL is not set")
    elif not m.group(1).startswith("https://github.com/") or STANDARD_PLACEHOLDER.search(m.group(1)):
        problems.append("config.tex: \\RepoURL is still a placeholder: %s" % m.group(1))
    for f in text_files(report):
        for n, line in enumerate(read(f).splitlines(), 1):
            code = line.split("%")[0] if "\\%" not in line else line
            if MARKS.search(code):
                problems.append("%s:%d: unfinished mark: %s" % (os.path.relpath(f, report), n, line.strip()[:80]))
    macros = number_macros(report)
    if macros is None:
        problems.append("generated/numbers.tex is missing (run scripts/make_report_tables.py)")
    else:
        used = {}
        for f in text_files(report):
            for name in re.findall(r"\\([A-Za-z]+)", re.sub(r"(?m)^\s*%.*$", "", read(f))):
                if name in macros:
                    used.setdefault(name, os.path.basename(f))
        for name, f in sorted(used.items()):
            if macros[name].strip() in EMPTY_VALUES:
                problems.append("%s uses \\%s, which has no value (%r)" % (f, name, macros[name]))
    if macros is not None:
        for f in text_files(report):
            body = re.sub(r"(?m)^\s*%.*$", "", read(f))
            for name in re.findall(r"=\s*\\([A-Za-z]+)\{\}", body):                    # `p = <0.001` would be wrong: a value that starts with the less-than sign needs no equals sign
                if macros.get(name, "").startswith("$<$"):
                    problems.append("%s: '= \\%s' would print '= <'" % (os.path.basename(f), name))
            for name in re.findall(r"\$p\$\s+\\([A-Za-z]+)\{\}", body):                # `p 0.005` is missing its equals sign
                if name in macros and not macros[name].startswith("$<$"):
                    problems.append("%s: '$p$ \\%s' has no relation sign" % (os.path.basename(f), name))
    return problems


def parse_log(text):
    """Errors and warnings of a pdflatex log: {errors, undefined_refs, undefined_cites, duplicate_labels, overfull, rerun}."""
    out = {"errors": [], "undefined_refs": [], "undefined_cites": [], "duplicate_labels": [], "overfull": [], "rerun": False}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("!"):
            where = ""
            for nxt in lines[i + 1:i + 8]:
                if nxt.startswith("l."):
                    where = nxt.strip()
                    break
            out["errors"].append((line[1:].strip() + " " + where).strip())
        m = re.search(r"Reference `([^']*)' on page (\d+) undefined", line)
        if m:
            out["undefined_refs"].append("%s (page %s)" % (m.group(1), m.group(2)))
        m = re.search(r"Citation `([^']*)' on page (\d+) undefined", line)
        if m:
            out["undefined_cites"].append("%s (page %s)" % (m.group(1), m.group(2)))
        m = re.search(r"Label `([^']*)' multiply defined", line)
        if m:
            out["duplicate_labels"].append(m.group(1))
        m = re.match(r"Overfull \\hbox \(([\d.]+)pt too wide\)(.*)", line)
        if m and float(m.group(1)) >= 1.0:
            out["overfull"].append("%spt%s" % (m.group(1), m.group(2)))
        if "Label(s) may have changed" in line:
            out["rerun"] = True
    return out


def problems_of(parsed):
    p = ["LaTeX error: " + e for e in parsed["errors"]]
    p += ["undefined reference: " + r for r in parsed["undefined_refs"]]
    p += ["undefined citation: " + c for c in parsed["undefined_cites"]]
    p += ["label defined twice: " + l for l in parsed["duplicate_labels"]]
    return p


def run(cmd, cwd, env):
    return subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")


def build(report, out_pdf, name="report"):
    problems = check_sources(report)
    if problems:
        return problems, None
    build_dir = os.path.join(report, "build")
    os.makedirs(build_dir, exist_ok=True)
    env = dict(os.environ, BIBINPUTS=os.path.abspath(report) + os.pathsep, BSTINPUTS=os.path.abspath(report) + os.pathsep,
               TEXINPUTS=os.path.abspath(report) + os.pathsep)
    tex = ["pdflatex", "-interaction=nonstopmode", "-output-directory=build", name + ".tex"]
    run(tex, report, env)
    b = run(["bibtex", name], build_dir, env)
    if "I couldn't open" in b.stdout or "error message" in b.stdout.lower():
        return ["bibtex: " + " | ".join(l for l in b.stdout.splitlines() if "couldn" in l or "rror" in l)[:300]], None
    log = os.path.join(build_dir, name + ".log")
    for _ in range(5):
        run(tex, report, env)
        if not (os.path.exists(log) and re.search(r"Label\(s\) may have changed|Rerun to get|Table widths have changed|Please \(re\)run", read(log))):
            break
    parsed = parse_log(read(log)) if os.path.exists(log) else {"errors": ["no log"], "undefined_refs": [], "undefined_cites": [], "duplicate_labels": [], "overfull": [], "rerun": False}
    problems = problems_of(parsed)
    if parsed["rerun"]:
        problems.append("labels still change after three runs")
    pdf = os.path.join(build_dir, name + ".pdf")
    if problems or not os.path.exists(pdf):
        return problems or ["no PDF was written"], parsed
    shutil.copyfile(pdf, out_pdf)
    return [], parsed


def pages(pdf):
    try:
        info = subprocess.run(["pdfinfo", pdf], stdout=subprocess.PIPE, text=True).stdout
        return int(re.search(r"Pages:\s+(\d+)", info).group(1))
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--report", default=os.path.join(here, "..", "report"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.report, "MICA_benchmark_report.pdf")
    problems, parsed = build(a.report, out)
    if parsed and parsed["overfull"]:
        print("overfull boxes of 1 pt or more (%d):" % len(parsed["overfull"]))
        for o in parsed["overfull"][:15]:
            print("  ", o)
    if problems:
        print("THE REPORT IS NOT FINISHED (%d problems):" % len(problems))
        for p in problems:
            print("  -", p)
        return 1
    print("wrote %s (%s pages)" % (out, pages(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
