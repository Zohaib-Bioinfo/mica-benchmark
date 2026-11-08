#!/usr/bin/env bash
# Regenerate the analysis, the figures, the tables and the PDF of the report from the unpacked result zips in results/raw/.
#
#   bash scripts/regenerate.sh            # everything, in place (results/, report/figures/, report/generated/, report/MICA_benchmark_report.pdf)
#   bash scripts/regenerate.sh --check    # the two analyses only, into a temporary folder; compares them with the committed tables and changes nothing
#
# Needs: python3 with numpy, pandas, scipy, matplotlib (and pdflatex + bibtex for the PDF). Run it from anywhere; it changes into the repository root.
# Two analyses are made from the same raw files (decision 12 of docs/SPEC.md and LOG.md, 2026-10-26 to 2026-10-28):
#   results/          the main analysis: the statuses of all sessions except the session without docked input; the scores of the baselines and of the MICA models
#   results/nodock/   the sensitivity analysis: MICA program 3 without docked AlphaFold input (8 entries) against the same baselines
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=results/raw
ENTRIES=data/entries_2026.csv

# status tables, oldest first (the order that analyze_results.py wants)
METHOD_STATUS=$(python3 scripts/ingest_results.py --list method_status "$RAW")
MICA_STATUS=$(python3 scripts/ingest_results.py --list mica_status "$RAW" | grep -v mica_gpu_nodock)
NODOCK_STATUS=$(ls "$RAW"/mica_gpu_nodock_result_*/mica_status.csv)

# scores, oldest session first (a later file replaces the rows of an earlier one with the same entry, method and model file)
scoring_folders() {   # the scoring sessions that hold scores: a folder without scoring/scores_long.csv (a stop before any model was scored) or with a NOT_A_RESULT.txt is left out
    for d in "$RAW"/scoring_*_result_*; do
        if [ -f "$d/scoring/scores_long.csv" ] && [ ! -f "$d/NOT_A_RESULT.txt" ]; then echo "$d"; fi
    done
}
scores() {   # $1: grep pattern for the folders to keep (-v: to drop)
    scoring_folders | "$@" | sed -E 's#^(.*_result_)([0-9]{8}-[0-9]{4})$#\2 \1\2#' | sort | cut -d' ' -f2- | sed 's#$#/scoring/scores_long.csv#'
}
SCORES_MAIN=$(scores grep -v nodock)
SCORES_NODOCK=$(scores grep -E "scoring_base|scoring_nodock|scoring_cryo")      # the baselines that were added later (CryoAtom, CryoZeta) are baselines of this analysis too

analyze() {  # $1: output folder, $2: status files, $3: score files
    # shellcheck disable=SC2086
    python3 scripts/analyze_results.py --entries "$ENTRIES" --status $2 --scores $3 --out "$1"
}

if [ "${1:-}" = "--check" ]; then
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    analyze "$TMP/main" "$METHOD_STATUS $MICA_STATUS" "$SCORES_MAIN" > "$TMP/main.log"
    analyze "$TMP/nodock" "$METHOD_STATUS $NODOCK_STATUS" "$SCORES_NODOCK" > "$TMP/nodock.log"
    bad=0
    for pair in "main:results" "nodock:results/nodock"; do
        new=$TMP/${pair%%:*}; old=${pair#*:}
        for f in "$new"/*; do
            if cmp -s "$f" "$old/$(basename "$f")"; then :; else echo "DIFFERENT: $old/$(basename "$f")"; bad=1; fi
        done
    done
    if [ "$bad" = 0 ]; then echo "All tables of results/ and results/nodock/ regenerate identically from results/raw/."; fi
    exit "$bad"
fi

analyze results "$METHOD_STATUS $MICA_STATUS" "$SCORES_MAIN"
analyze results/nodock "$METHOD_STATUS $NODOCK_STATUS" "$SCORES_NODOCK"
python3 scripts/make_figures.py --results results --out report/figures
python3 scripts/make_report_tables.py --results results --raw "$RAW" --data data --out report/generated
if command -v pdflatex > /dev/null && command -v bibtex > /dev/null; then
    python3 scripts/build_report.py
else
    echo "pdflatex or bibtex is not installed: the PDF was not rebuilt (the tables and macros of report/generated/ are up to date)."
fi
