#!/usr/bin/env python3
"""Tests for build_report.py (the check before the PDF is written and the reading of the LaTeX log).

    python -W error -m unittest test_build_report

The end-to-end tests need `pdflatex` and `bibtex`; they are skipped when these are not installed.
"""
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_report as br  # noqa: E402

GOOD_CONFIG = "\\newcommand{\\RepoURL}{https://github.com/Zohaib-Bioinfo/mica-benchmark}\n"
GOOD_NUMBERS = "\\newcommand{\\NEntries}{20}\n\\newcommand{\\PwHolm}{$<$0.001}\n\\newcommand{\\PtHolm}{0.024}\n"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class ReportDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.report = os.path.join(self.tmp, "report")
        write(os.path.join(self.report, "config.tex"), GOOD_CONFIG)
        write(os.path.join(self.report, "report.tex"), "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n")
        write(os.path.join(self.report, "generated", "numbers.tex"), GOOD_NUMBERS)
        write(os.path.join(self.report, "sections", "s1.tex"), "There are \\NEntries{} entries.\n")

    def section(self, text):
        write(os.path.join(self.report, "sections", "s1.tex"), text)

    def numbers(self, text):
        write(os.path.join(self.report, "generated", "numbers.tex"), text)

    def config(self, text):
        write(os.path.join(self.report, "config.tex"), text)


class TestCheckSources(ReportDir):
    def test_a_finished_report_has_no_problems(self):
        self.assertEqual(br.check_sources(self.report), [])

    def test_a_missing_config_is_a_problem(self):
        os.remove(os.path.join(self.report, "config.tex"))
        self.assertEqual(br.check_sources(self.report), ["config.tex is missing"])

    def test_the_repository_address_must_be_set(self):
        self.config("\\newcommand{\\RepoURL}{}\n")
        self.assertTrue(any("not set" in p for p in br.check_sources(self.report)))

    def test_placeholder_addresses_are_refused(self):
        for url in ("https://github.com/username/mica-benchmark", "https://github.com/YOUR-USER/repo", "https://example.com/mica",
                    "https://github.com/you/mica", "<repo url>", "https://gitlab.com/someone/mica"):
            self.config("\\newcommand{\\RepoURL}{%s}\n" % url)
            self.assertTrue(any("RepoURL" in p for p in br.check_sources(self.report)), url)

    def test_unfinished_marks_are_found_in_the_text(self):
        for mark in ("TODO", "TBD", "FIXME", "XXX", "???"):
            self.section("The result is %s here.\n" % mark)
            problems = br.check_sources(self.report)
            self.assertTrue(any("unfinished mark" in p and "s1.tex:1" in p for p in problems), mark)

    def test_a_mark_inside_a_latex_comment_is_not_a_problem(self):
        self.section("Fine text. % TODO for later, not printed\nMore text with 50\\% of the entries.\n")
        self.assertEqual(br.check_sources(self.report), [])

    def test_a_mark_after_an_escaped_percent_sign_is_still_found(self):
        self.section("Half of them (50\\%) TODO.\n")
        self.assertTrue(any("unfinished mark" in p for p in br.check_sources(self.report)))

    def test_a_used_macro_without_a_value_is_a_problem(self):
        for empty in ("", "n/a", "nan", "NaN", "--", "-"):
            self.numbers(GOOD_NUMBERS + "\\newcommand{\\TmMean}{%s}\n" % empty)
            self.section("The mean is \\TmMean{}.\n")
            problems = br.check_sources(self.report)
            self.assertTrue(any("TmMean" in p and "no value" in p for p in problems), repr(empty))

    def test_an_unused_macro_without_a_value_is_not_a_problem(self):
        self.numbers(GOOD_NUMBERS + "\\newcommand{\\TmMean}{n/a}\n")
        self.assertEqual(br.check_sources(self.report), [])

    def test_a_macro_used_only_in_a_comment_is_not_checked(self):
        self.numbers(GOOD_NUMBERS + "\\newcommand{\\TmMean}{n/a}\n")
        self.section("% The mean is \\TmMean{}.\nText.\n")
        self.assertEqual(br.check_sources(self.report), [])

    def test_a_missing_numbers_file_is_a_problem(self):
        os.remove(os.path.join(self.report, "generated", "numbers.tex"))
        self.assertTrue(any("numbers.tex is missing" in p for p in br.check_sources(self.report)))

    def test_an_equals_sign_before_a_less_than_value_is_a_problem(self):
        self.section("Holm $p$ = \\PwHolm{}.\n")
        problems = br.check_sources(self.report)
        self.assertTrue(any("PwHolm" in p and "'= <'" in p for p in problems), problems)

    def test_an_equals_sign_before_a_plain_value_is_fine(self):
        self.section("Holm $p$ = \\PtHolm{}.\n")
        self.assertEqual(br.check_sources(self.report), [])

    def test_a_less_than_value_after_the_p_symbol_is_fine(self):
        self.section("Holm $p$ \\PwHolm{}.\n")
        self.assertEqual(br.check_sources(self.report), [])

    def test_a_plain_p_value_without_a_relation_sign_is_a_problem(self):
        self.section("Holm $p$ \\PtHolm{}.\n")
        problems = br.check_sources(self.report)
        self.assertTrue(any("PtHolm" in p and "no relation sign" in p for p in problems), problems)


class TestParseLog(unittest.TestCase):
    LOG = "\n".join([
        "This is pdfTeX, Version 3.14159265",
        "! Undefined control sequence.",
        "l.42 The mean is \\Missing",
        "                  {} here.",
        "LaTeX Warning: Reference `sec:nowhere' on page 3 undefined on input line 80.",
        "LaTeX Warning: Citation `nobody2030' on page 4 undefined on input line 91.",
        "LaTeX Warning: Label `fig:one' multiply defined.",
        "Overfull \\hbox (0.4pt too wide) in paragraph at lines 5--6",
        "Overfull \\hbox (12.5pt too wide) in paragraph at lines 7--8",
        "Overfull \\hbox (1.0pt too wide) in alignment at lines 9--10",
        "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.",
    ])

    def test_errors_carry_the_line_of_the_source(self):
        p = br.parse_log(self.LOG)
        self.assertEqual(len(p["errors"]), 1)
        self.assertIn("Undefined control sequence.", p["errors"][0])
        self.assertIn("l.42", p["errors"][0])

    def test_undefined_references_citations_and_duplicate_labels(self):
        p = br.parse_log(self.LOG)
        self.assertEqual(p["undefined_refs"], ["sec:nowhere (page 3)"])
        self.assertEqual(p["undefined_cites"], ["nobody2030 (page 4)"])
        self.assertEqual(p["duplicate_labels"], ["fig:one"])

    def test_overfull_boxes_below_one_point_are_ignored(self):
        p = br.parse_log(self.LOG)
        self.assertEqual(len(p["overfull"]), 2)
        self.assertTrue(p["overfull"][0].startswith("12.5pt"))
        self.assertTrue(p["overfull"][1].startswith("1.0pt"))

    def test_the_rerun_warning_is_seen(self):
        self.assertTrue(br.parse_log(self.LOG)["rerun"])
        self.assertFalse(br.parse_log("All is well.\n")["rerun"])

    def test_a_clean_log_has_no_problems(self):
        p = br.parse_log("This is pdfTeX\nOutput written on report.pdf (27 pages).\n")
        self.assertEqual(br.problems_of(p), [])
        self.assertEqual((p["errors"], p["undefined_refs"], p["undefined_cites"], p["duplicate_labels"], p["overfull"]), ([], [], [], [], []))

    def test_problems_of_names_each_kind(self):
        problems = br.problems_of(br.parse_log(self.LOG))
        self.assertEqual(len(problems), 4)
        self.assertTrue(problems[0].startswith("LaTeX error: "))
        self.assertIn("undefined reference: sec:nowhere (page 3)", problems)
        self.assertIn("undefined citation: nobody2030 (page 4)", problems)
        self.assertIn("label defined twice: fig:one", problems)

    def test_an_exclamation_mark_inside_a_line_is_not_an_error(self):
        self.assertEqual(br.parse_log("Text with a surprise! in it\n")["errors"], [])


@unittest.skipUnless(shutil.which("pdflatex") and shutil.which("bibtex"), "pdflatex and bibtex are needed")
class TestBuild(ReportDir):
    TEX = ("\\documentclass{article}\n\\usepackage{natbib}\n\\input{config}\n\\input{generated/numbers}\n\\begin{document}\n"
           "\\section{One}\\label{sec:one}\nSee Section~\\ref{sec:one}, entries: \\NEntries{}, \\citep{ref1}. Code: \\RepoURL{}.\n"
           "\\bibliographystyle{abbrvnat}\n\\bibliography{refs}\n\\end{document}\n")
    BIB = "@article{ref1, author={Doe, J.}, title={A title}, journal={A journal}, year={2026}}\n"

    def setUp(self):
        super().setUp()
        write(os.path.join(self.report, "report.tex"), self.TEX)
        write(os.path.join(self.report, "refs.bib"), self.BIB)
        self.out = os.path.join(self.tmp, "out.pdf")

    def test_a_good_report_is_built_and_copied(self):
        problems, parsed = br.build(self.report, self.out)
        self.assertEqual(problems, [])
        self.assertTrue(os.path.getsize(self.out) > 1000)
        self.assertEqual(parsed["undefined_refs"], [])
        self.assertEqual(br.pages(self.out), 1)

    def test_an_undefined_reference_stops_the_build(self):
        write(os.path.join(self.report, "report.tex"), self.TEX.replace("\\ref{sec:one}", "\\ref{sec:missing}"))
        problems, _ = br.build(self.report, self.out)
        self.assertTrue(any("undefined reference: sec:missing" in p for p in problems), problems)
        self.assertFalse(os.path.exists(self.out))

    def test_an_undefined_citation_stops_the_build(self):
        write(os.path.join(self.report, "report.tex"), self.TEX.replace("\\citep{ref1}", "\\citep{nobody}"))
        problems, _ = br.build(self.report, self.out)
        self.assertTrue(any("nobody" in p for p in problems), problems)
        self.assertFalse(os.path.exists(self.out))

    def test_a_latex_error_stops_the_build(self):
        write(os.path.join(self.report, "report.tex"), self.TEX.replace("entries:", "entries: \\nosuchmacro{}"))
        problems, _ = br.build(self.report, self.out)
        self.assertTrue(any(p.startswith("LaTeX error") for p in problems), problems)
        self.assertFalse(os.path.exists(self.out))

    def test_an_unfinished_source_is_refused_before_latex_runs(self):
        self.section("TODO write this.\n")
        problems, parsed = br.build(self.report, self.out)
        self.assertTrue(any("unfinished mark" in p for p in problems))
        self.assertIsNone(parsed)
        self.assertFalse(os.path.isdir(os.path.join(self.report, "build")))

    def test_main_returns_one_for_a_problem_and_zero_for_a_good_report(self):
        self.assertEqual(br.main(["--report", self.report, "--out", self.out]), 0)
        os.remove(self.out)
        write(os.path.join(self.report, "report.tex"), self.TEX.replace("\\ref{sec:one}", "\\ref{sec:missing}"))
        self.assertEqual(br.main(["--report", self.report, "--out", self.out]), 1)


if __name__ == "__main__":
    unittest.main()
