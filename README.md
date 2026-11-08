# MICA cryo-EM benchmark on 2026 maps

A benchmark of MICA (Gyawali, Dhakal & Cheng, 2025) against ModelAngelo, EModelX(+AF) and two newer deep-learning model builders (EMProt, E3-CryoFold) on 20 EMDB/PDB entries whose maps and models were released in 2026, run on free compute (Kaggle and Google Colab). It is the practical task of the research-assistant selection of Prof. Jianlin Cheng. Author: Muhammad Zohaib (BS Bioinformatics, University of Agriculture Faisalabad; GitHub: Zohaib-Bioinfo).

**Report:** [report/MICA_benchmark_report.pdf](report/MICA_benchmark_report.pdf) (27 pages: 14 of text and references, 13 of appendix). **Log** of every decision, error and change: [LOG.md](LOG.md). **Scope**, locked before any run: [docs/SPEC.md](docs/SPEC.md).

**Status:** finished on 2026-10-30. CryoAtom, the third newer method of the plan, was dropped before the production runs (the first scope cut of the spec: neither of its two versions finished a pilot within 2 hours), so the benchmark has two newer methods.

## Result in brief

TM-score against the deposited model (US-align, normalised by the length of the deposited model) of the models that each method built. The last column is method minus MICA on the entries where both built a model (Tables 4 and 5 of the report).

| Method | Models built (of 20) | TM-score, median | TM-score, mean [95 % CI] | Difference to MICA, mean [95 % CI] (pairs) |
| --- | --- | --- | --- | --- |
| MICA | 17 | 0.94 | 0.94 [0.91, 0.96] | - |
| ModelAngelo | 19 | 0.91 | 0.79 [0.68, 0.90] | -0.15 [-0.28, -0.03] (17) |
| EModelX(+AF) | 18 | 0.94 | 0.94 [0.91, 0.96] | 0.00 [-0.02, 0.03] (17) |
| EMProt | 19 | 0.98 | 0.96 [0.92, 0.98] | 0.04 [0.02, 0.06] (17) |
| E3-CryoFold | 15 | 0.91 | 0.86 [0.79, 0.93] | -0.08 [-0.14, -0.03] (15) |

After Holm's correction only the difference to EMProt is clear on both statistical tests. EModelX(+AF) does not differ from MICA, E3-CryoFold is lower on the Wilcoxon test only, and ModelAngelo is lower on the maps above 3.0 Å (borderline) but not on the maps up to 3.0 Å; with at most 17 pairs, differences of a few hundredths cannot be resolved. Two deviations from MICA's own pipeline were forced by free compute: its docking step was replaced by a coarse, time-limited Phenix search, and its final real-space refinement could not run (Phenix 2.2.1 rejects the parameter file shipped with MICA), so the scored file is MICA's model before refinement. The report describes both with the evidence and the probable effect (Sections 2.2, 6.6 and 7). Three MICA runs did not finish on the free machine (two killed for lack of RAM, one at its time limit); failures stay in every table and are scored as zero in a sensitivity analysis.

## Layout

| Path | Contents |
| --- | --- |
| `docs/SPEC.md` | Locked scope: methods, data rules, metrics, statistics, compute plan, failure handling |
| `LOG.md` | Dated log, newest entry first |
| `data/` | The frozen entry list (`entries_2026.csv`, 20 entries: 10 up to 3.0 Å and 10 above 3.0 to 4.0 Å); the candidate pool it was drawn from (`pool/`, `pool_run1_prefilter/`); clusters, CC_mask, novelty and the selection log (`selection/`); the AlphaFold Server job file, the check of the returned predictions and the file hashes (`af3/`; the predictions themselves are not committed); `mica_map_geometry.csv` |
| `methods/` | The 29 methods considered, with versions, inputs and the reason each was included or excluded (`methods_table.csv`), and how the search ran (`SEARCH_LOG.md`) |
| `env/` | Environment files, one per method (exact pins; the header of each file says how it was resolved) |
| `scripts/` | All code, with a `test_*.py` for each part (overview below); `as_run/` holds notebooks that ran and have been replaced by newer builds |
| `results/` | `raw/`: the unpacked text outputs of the 23 sessions that produced results (`INDEX.csv` has the hash of each original zip); the tables of the main analysis (`metrics_per_entry.csv` is the per-entry table of every score); `nodock/`: the sensitivity analysis without docked AlphaFold input |
| `report/` | LaTeX sources (`report.tex`, `sections/`, `refs.bib`, `config.tex`), figures, the generated tables and number macros (`generated/`) and the PDF |

Scripts:

| Step | Files in `scripts/` |
| --- | --- |
| Candidate pool and selection (Colab) | `pull_pool.py`, `select_entries.py`, `compute_cc_mask.py`, `novelty_check.py`; the notebooks `pull_pool_colab.ipynb` and `select_colab.ipynb` (built by `make_colab_notebook.py` and `make_select_notebook.py`) |
| AlphaFold Server inputs | `make_af3_jobs.py`, `check_af3_results.py` |
| Runs on Kaggle (notebook generators) | `make_main_notebook.py` (ModelAngelo, E3-CryoFold, EMProt, EModelX(+AF); with `--methods cryoatom1` CryoAtom v1.0.0 in a notebook of its own, added on 2026-10-31 at the request of Prof. Cheng), `make_mica_notebook.py` (MICA: docking on CPU sessions, programs 2 and 3 on GPU sessions), `mica_dock_driver.py` (the docking driver), `mica_inputs.py` (MICA's input folder); pilots and diagnostics: `make_pilot_notebook.py`, `make_dockdiag_notebook.py`, `make_modelangelo_notebook.py`, `make_methods_notebook.py`, `make_survey_notebook.py`, `map_survey.py`; shared parts: `nbkit.py`, `nbtools.py` |
| Scoring | `make_scoring_notebook.py`, `score_models.py`, `evaluate_models.py` (US-align and Phenix `chain_comparison`) |
| Analysis and report | `ingest_results.py`, `analyze_results.py`, `make_figures.py`, `make_report_tables.py`, `build_report.py`, `regenerate.sh` |

## Regenerate the tables, figures and PDF

Every table, figure and number of the report comes from `results/raw/` with scripts of this repository; nothing in the text is typed in by hand (each number is a LaTeX macro in `report/generated/numbers.tex`, and `build_report.py` stops when a macro has no value or a reference is undefined).

```
pip install numpy pandas scipy matplotlib       # and pdflatex + bibtex (TeX Live) for the PDF
bash scripts/regenerate.sh --check              # about 10 seconds: the tables of results/ and results/nodock/ regenerate identically
bash scripts/regenerate.sh                      # everything, in place
```

`regenerate.sh` runs, in this order: `analyze_results.py` twice (main analysis into `results/`, sensitivity analysis into `results/nodock/`; seed 2026, 10,000 bootstrap resamples), `make_figures.py` (`report/figures/`), `make_report_tables.py` (`report/generated/`) and `build_report.py` (`report/MICA_benchmark_report.pdf`). The script lists the exact status and score files of each analysis.

## The notebooks as run

Every result in `results/raw/` comes from one Kaggle (or Colab) session of one notebook; the folder name is `<notebook>_result_<time>`. Notebooks are built by the generators above. The files in `scripts/as_run/` are the notebooks exactly as they ran (their generators have changed since, see its README); `test_mica_notebook.py` compares the committed MICA notebooks (`mica_dock3_*`, `mica_gpu*`) with a fresh build.

| Notebook | What it ran | Folder(s) in `results/raw/` |
| --- | --- | --- |
| `pull_pool_colab.ipynb`, `select_colab.ipynb` | candidate pool of 2026 releases; CC_mask, selection and novelty (Colab) | selection files in `data/` |
| `as_run/pilot_*.ipynb`, `as_run/dockdiag*.ipynb`, `as_run/survey.ipynb` | pilots of the five methods and diagnostics of MICA's docking on entry 26DK; map survey | not part of the tables; their findings are quoted in the report where stated (LOG 2026-10-19 to 2026-10-24) |
| `as_run/gpu_batch1.ipynb`, `gpu_batch2.ipynb`, `gpu_batch3.ipynb` | ModelAngelo, E3-CryoFold, EMProt and EModelX(+AF) on all 20 entries, two GPUs at once (batch 2 and 3: the open tasks) | `gpu_batch1_result_*`, `gpu_batch2_result_*`, `gpu_batch3_result_*` |
| `as_run/mica_cpu_a/b/c.ipynb` | MICA, docking pass 1: programs 1 and 2 on CPU sessions | `mica_cpu_*_result_*` |
| `as_run/mica_dock2_a/b/c/d.ipynb` | docking pass 2 (a parameter that Phenix rejected; see the report, Section 4.4) | `mica_dock2_*_result_*` |
| `mica_dock3_a/b/c/d.ipynb` | docking pass 3, the docking set-up that counts | `mica_dock3_*_result_*` |
| `mica_gpu.ipynb` | MICA program 3 on a GPU session (first round) | `mica_gpu_result_*` |
| `mica_gpu_nodock.ipynb` | the same eight entries without docked AlphaFold input (sensitivity) | `mica_gpu_nodock_result_*` |
| `mica_gpu2_a.ipynb`, `mica_gpu2_b.ipynb` | MICA program 3, one window at a time (the memory patch), the 12 entries without a model | `mica_gpu2_a_result_*_INCOMPLETE`, `mica_gpu2_b_result_*` |
| `mica_gpu3.ipynb` | program 3 for 9OC0 and 9VI4, the two entries that the time budget cut | `mica_gpu3_result_*` |
| `scoring_base.ipynb`, `scoring_mica.ipynb`, `scoring_mica2.ipynb`, `scoring_nodock.ipynb` | scoring of the models with US-align and Phenix `chain_comparison` (protocol v1.0 of LOG 2026-10-24) | `scoring_*_result_*` |
| `mica_gpu2_nodock_a/b.ipynb`, `scoring_mica3.ipynb`, `scoring_nodock2.ipynb`, `scoring.ipynb` | built and tested, **not run** (the GPU quota did not allow more sensitivity sessions; the other two were not needed; `scoring.ipynb` is the all-in-one build that the scope-specific ones replaced) | - |

## Reproduce the selection of the entries

`select_colab.ipynb` does it in one Colab run (Phenix from your own download link and password, typed in at run time): CC_mask for the entries that the rule asks for, the seeded selection round by round, the novelty check, one result zip. The same steps by hand (offline tests: `test_pull_pool.py`, `test_select_entries.py`, `test_compute_cc_mask.py`, `test_novelty_check.py`):

```
pip install gemmi requests
python scripts/pull_pool.py --freeze 2026-10-12 --out pool_out                                  # the pool of 2026 releases from RCSB and EMDB (484 candidates)
python scripts/select_entries.py cluster --pool data/pool --out data/selection --mmseqs mmseqs   # MMseqs2 clusters at 30 % identity
python scripts/select_entries.py select  --pool data/pool --out data/selection --entries data/entries_2026.csv --cc data/selection/cc_mask.csv
python scripts/compute_cc_mask.py --todo data/selection/cc_todo.csv --out data/selection/cc_mask.csv --phenix-env /path/to/phenix_env.sh
python scripts/novelty_check.py --entries data/entries_2026.csv --fasta data/pool/pool_candidate_sequences.fasta \
       --cryoem-ids data/pool/pool_ids_all_cryoem.txt --out data/selection/novelty
```

`select` exits with code 3 and writes `cc_todo.csv` while it still needs CC_mask values: run `compute_cc_mask.py`, then `select` again, until it exits with 0. The pool is defined by release date, so it gives the same 484 candidates on any day from the freeze date on; the frozen list and its hashes are in `LOG.md` (2026-10-14).

## Run the experiments again

You need a Kaggle account with GPU quota (T4 x2, Internet on), a Phenix download login (academic licence) stored as the Kaggle secrets `PHENIX_USER` and `PHENIX_PASSWORD`, and your own AlphaFold Server predictions of the 26 chain sequences (upload `data/af3/af3_server_jobs.json` on alphafoldserver.com, keep `model_0` of each job, check the files with `scripts/check_af3_results.py`, add them as a private Kaggle dataset). Build a notebook with its generator (for example `python scripts/make_main_notebook.py --batch batch1`, `python scripts/make_mica_notebook.py --pass3`), run it with Save and Run All, and unpack the result zip with `python scripts/ingest_results.py <zip> --out results/raw` (text files only: the models stay in the notebook output). The Phenix password and the AlphaFold predictions are never packed.

## Tests

```
cd scripts
python3 -W error -m unittest discover -p "test_*.py"
```

Tests that need an external tool are skipped when it is not available: `MMSEQS` (path of MMseqs2), `USALIGN` (US-align binary), `MICA_TEST_ENV_PYTHON` and `MICA_TEST_AF3_DIR` (end-to-end runs of the notebooks against stand-ins of the methods), `MICA_TEST_UV=1`, and `pdflatex` with `bibtex` (the build test of the report). The result of the last full run is in `LOG.md` (2026-10-30).

## Not in the repository

The AlphaFold 3 predictions (the terms of AlphaFold Server allow non-commercial research use only; the job list, the checks and the hashes are public), the Phenix installer and credentials, maps, network weights and the coordinate files of the models (they were written to the notebook outputs; `results/raw/INDEX.csv` lists what was skipped). The raw text output of every scoring tool is in `results/raw/scoring_*/scoring/raw/`.

## Methods

| Method | Reference | Code |
| --- | --- | --- |
| MICA | Gyawali, R., Dhakal, A. & Cheng, J. Multimodal deep learning integration of cryo-EM and AlphaFold3 for high-accuracy protein structure determination. *Commun. Chem.* 8, 320 (2025). [doi:10.1038/s42004-025-01718-5](https://doi.org/10.1038/s42004-025-01718-5) | [jianlin-cheng/MICA](https://github.com/jianlin-cheng/MICA) |
| ModelAngelo | Jamali, K. et al. Automated model building and protein identification in cryo-EM maps. *Nature* 628, 450-457 (2024). | [3dem/model-angelo](https://github.com/3dem/model-angelo) |
| EModelX(+AF) | Chen, S. et al. Protein complex structure modeling by cross-modal alignment between cryo-EM maps and protein sequences. *Nat. Commun.* 15, 8808 (2024). | [biomed-AI/EModelX](https://github.com/biomed-AI/EModelX) |
| CryoAtom (dropped, see below) | Su, B. et al. CryoAtom improves model building for cryo-EM. *Nat. Struct. Mol. Biol.* 33, 351-361 (2026). [doi:10.1038/s41594-025-01713-3](https://doi.org/10.1038/s41594-025-01713-3) | [YangLab-SDU/CryoAtom](https://github.com/YangLab-SDU/CryoAtom) |
| EMProt | Li, T., Chen, J., Li, H., Cao, H. & Huang, S.-Y. EMProt improves structure determination from cryo-EM maps. *Nat. Struct. Mol. Biol.* 33, 341-350 (2026). [doi:10.1038/s41594-025-01723-1](https://doi.org/10.1038/s41594-025-01723-1) | [huang-laboratory/EMProt](https://github.com/huang-laboratory/EMProt) |
| E3-CryoFold | Wang, J., Tan, C., Gao, Z., Zhang, G., Zhang, Y. & Li, S. Z. End-to-end cryo-EM complex structure determination with high accuracy and ultra-fast speed. *Nat. Mach. Intell.* 7, 1091-1103 (2025). [doi:10.1038/s42256-025-01056-0](https://doi.org/10.1038/s42256-025-01056-0) | [A4Bio/E3-CryoFold](https://github.com/A4Bio/E3-CryoFold) |

Two reserves were named in the spec and not needed: EM3DFold ([preprint](https://www.biorxiv.org/content/10.64898/2026.09.16.752067.abstract), [code](https://github.com/huang-laboratory/EM3DFold)), which passes the screen on paper but is not yet peer reviewed, and CryoZeta ([preprint](https://doi.org/10.64898/2026.02.13.705846), [code](https://github.com/kiharalab/CryoZeta)), whose README asks for 32 GB or more of GPU memory. CryoAtom passed the screen but was dropped after the pilot (report, Section 2.4). Every method considered, with the reason for each exclusion, is in `methods/methods_table.csv`.
