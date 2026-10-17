# MICA cryo-EM benchmark on 2026 maps

A benchmark of MICA against ModelAngelo, EModelX(+AF) and three newer deep-learning methods that build protein models from cryo-EM maps (CryoAtom, EMProt, E3-CryoFold). The test entries are EMDB maps and PDB models released in 2026, so none of them can be in the training data of the three baseline methods. Everything runs on free compute.

**Status:** Phase 0 done (scope locked 2026-10-10). Phase 2 done (newer methods chosen 2026-10-11). Phase 3 done: the candidate pool is frozen (484 entries, `data/pool/`) and the entry list is frozen (`data/entries_2026.csv`: 20 entries, 10 up to 3.0 A and 10 above 3.0 to 4.0 A, chosen on 2026-10-14 by `scripts/select_colab.ipynb`, CC_mask and novelty included). Phase 4 has started: the 26 AlphaFold Server predictions came back on 2026-10-16 and passed `scripts/check_af3_results.py`; the MICA pilot notebook (`scripts/pilot_mica.ipynb`, entry 26DK) is built and tested, its first real run is next. The locked decisions are in [docs/SPEC.md](docs/SPEC.md); the dated work log is [LOG.md](LOG.md).

## Layout

| Path | Contents |
| --- | --- |
| `docs/SPEC.md` | Locked scope: methods, data rules, metrics, statistics, compute plan, failure handling |
| `data/` | The frozen entry list (`entries_2026.csv`), the candidate pool it was drawn from (`pool/`), the clusters and the selection logs (`selection/`), the AlphaFold Server job files and the check of the returned predictions (`af3/`; the predictions themselves are not committed) |
| `methods/` | Methods considered, with versions, inputs and the reason each was included or excluded (`methods_table.csv`), and how the search ran (`SEARCH_LOG.md`) |
| `scripts/` | Phase 3: `pull_pool.py` (candidate pool from RCSB and EMDB), `select_entries.py` (clusters and the seeded selection rule), `compute_cc_mask.py` (Phenix CC_mask), `novelty_check.py` (identity to pre-2026 PDB chains), `make_af3_jobs.py` (AlphaFold Server jobs for the frozen list), `check_af3_results.py` (checks the returned predictions against the submitted sequences); Phase 4: `mica_inputs.py` (MICA's input folder), `evaluate_models.py` (US-align scores), `make_pilot_notebook.py` (builds `pilot_mica.ipynb`, the MICA pilot for Kaggle or Colab); the Colab notebooks that run them (`pull_pool_colab.ipynb`, `select_colab.ipynb`, built by `make_colab_notebook.py` and `make_select_notebook.py`) and a `test_*.py` for each; later: run wrappers, evaluation and analysis |
| `env/` | Environment files, one per method (so far `mica_requirements.txt`: Python 3.8.20, 171 exact pins, PyPI wheels, installed with `uv`) |
| `results/` | Raw per-entry metrics and summary tables |
| `report/` | Report source and the final PDF |
| `LOG.md` | Dated log, newest entry first |

Large files (maps, structures, model weights, outputs) are not committed. The scripts download them.

## Pull the candidate pool (Phase 3a)

`scripts/pull_pool.py` lists every cryo-EM entry that RCSB released from 2026-01-01 to the freeze date 2026-10-12, downloads each mmCIF file and applies the filters of `docs/SPEC.md` (decisions 3 and 4) that need no map: release window, single particle, resolution of 4.0 A or better, protein only, at most 700 modeled residues, at most 3 unique chain sequences, at most 5% UNK. It adds one rule so that an entry means one map and one model (exactly one EMDB map linked), checks the EMDB release date of that map, and writes the pool files. The map-model correlation (CC_mask, needs Phenix) is applied in Phase 3b. Every entry is judged on its own mmCIF file; server-side search filters are off by default (`--prefilter` turns them on), because the audit of the first real run found an entry that they wrongly removed (`data/pool_run1_prefilter/`). The pool is defined by release date, so the run gives the same pool on any day from the freeze date on.

```
pip install gemmi requests
python scripts/pull_pool.py --freeze 2026-10-12 --out pool_out
python scripts/test_pull_pool.py        # offline tests, mock servers
```

On Colab or Kaggle, upload `scripts/pull_pool_colab.ipynb` and run all cells; it packs the results into one zip. Rebuild the notebook after any change to the script with `python scripts/make_colab_notebook.py`, and test it with `python scripts/test_colab_notebook.py` (needs nbformat, nbclient and ipykernel). The log of the first real run and its file hashes are in `LOG.md`.

## Choose the entries (Phase 3b and 3c)

`select_colab.ipynb` does it in one Colab run: it installs Phenix (from your own download link and password, typed in at run time), computes CC_mask for the entries that the rule asks for, applies the rule of `docs/SPEC.md` (decisions 4, 5 and 9) round by round until the list is final, runs the novelty check and packs one zip. The same steps by hand:

```
python scripts/select_entries.py cluster --pool data/pool --out data/selection --mmseqs mmseqs     # MMseqs2 clusters at 30% identity
python scripts/select_entries.py select  --pool data/pool --out data/selection --entries data/entries_2026.csv --cc data/selection/cc_mask.csv
python scripts/compute_cc_mask.py --todo data/selection/cc_todo.csv --out data/selection/cc_mask.csv --phenix-env /path/to/phenix_env.sh
python scripts/novelty_check.py --entries data/entries_2026.csv --fasta data/pool/pool_candidate_sequences.fasta \
       --cryoem-ids data/pool/pool_ids_all_cryoem.txt --out data/selection/novelty
```

`compute_cc_mask.py` calls `phenix.map_model_cc ... --force` (Phenix 2.2.1 refuses it without `--force` and points to `phenix.map_correlations`); if that does not give a CC_mask it tries `phenix.map_correlations` and keeps the first variant that works for every later entry (column `variant` of `cc_mask.csv`). `select` exits with code 3 and writes `cc_todo.csv` while it still needs CC_mask values; run `compute_cc_mask.py`, then `select` again, until it exits with 0 (the first call without `--cc` is the first round). Tests (offline; MMseqs2 for the clustering and novelty tests via the `MMSEQS` variable or the PATH, otherwise those tests are skipped): `python scripts/test_select_entries.py`, `test_compute_cc_mask.py`, `test_novelty_check.py`, `test_select_notebook.py` (needs nbformat, nbclient and ipykernel). Rebuild the notebook after any change to its scripts or data with `python scripts/make_select_notebook.py`. The Phenix password is never stored in the repo, the notebook or the results.

## MICA pilot (Phase 4)

`scripts/pilot_mica.ipynb` runs MICA on one entry (default 26DK) on a free Kaggle or Colab GPU session and scores every model it writes:
machine and GPU check, Phenix install (user name and password from Kaggle or Colab Secrets, never stored), Python 3.8 environment from `env/mica_requirements.txt`,
MICA at its pinned commit (726d1bb) with its weights from Zenodo (MD5 compared with the one Zenodo lists), the entry's input folder (your own AlphaFold Server
predictions, the EMDB map), MICA's three programs with the arguments of `MICA_pipeline.sh`, then US-align (pinned commit) and Phenix `chain_comparison` on every model,
and one result zip with logs, timings, GPU memory peaks and the raw output of each tool. Rebuild it with `python scripts/make_pilot_notebook.py [--entries 26DK 9OC0]`;
test it with `python scripts/test_pilot_notebook.py` (stand-ins for Phenix, MICA and the servers; the end-to-end tests need `MICA_TEST_ENV_PYTHON`, `MICA_TEST_AF3_DIR` and `USALIGN`, see the file).

## Reproduce

To be completed in Phase 8, once the scripts exist. The aim is that every table and figure in the report regenerates from `results/metrics_per_entry.csv` with one command.

## Methods

| Method | Reference | Code |
| --- | --- | --- |
| MICA | Gyawali, R., Dhakal, A. & Cheng, J. Multimodal deep learning integration of cryo-EM and AlphaFold3 for high-accuracy protein structure determination. *Commun. Chem.* 8, 320 (2025). [doi:10.1038/s42004-025-01718-5](https://doi.org/10.1038/s42004-025-01718-5) | [jianlin-cheng/MICA](https://github.com/jianlin-cheng/MICA) |
| ModelAngelo | Jamali, K. et al. Automated model building and protein identification in cryo-EM maps. *Nature* 628, 450-457 (2024). | [3dem/model-angelo](https://github.com/3dem/model-angelo) |
| EModelX(+AF) | Chen, S. et al. Protein complex structure modeling by cross-modal alignment between cryo-EM maps and protein sequences. *Nat. Commun.* 15, 8808 (2024). | [biomed-AI/EModelX](https://github.com/biomed-AI/EModelX) |
| CryoAtom | Su, B. et al. CryoAtom improves model building for cryo-EM. *Nat. Struct. Mol. Biol.* 33, 351-361 (2026). [doi:10.1038/s41594-025-01713-3](https://doi.org/10.1038/s41594-025-01713-3) | [YangLab-SDU/CryoAtom](https://github.com/YangLab-SDU/CryoAtom) |
| EMProt | Li, T., Chen, J., Li, H., Cao, H. & Huang, S.-Y. EMProt improves structure determination from cryo-EM maps. *Nat. Struct. Mol. Biol.* 33, 341-350 (2026). [doi:10.1038/s41594-025-01723-1](https://doi.org/10.1038/s41594-025-01723-1) | [huang-laboratory/EMProt](https://github.com/huang-laboratory/EMProt) |
| E3-CryoFold | Wang, J., Tan, C., Gao, Z., Zhang, G., Zhang, Y. & Li, S. Z. End-to-end cryo-EM complex structure determination with high accuracy and ultra-fast speed. *Nat. Mach. Intell.* 7, 1091-1103 (2025). [doi:10.1038/s42256-025-01056-0](https://doi.org/10.1038/s42256-025-01056-0) | [A4Bio/E3-CryoFold](https://github.com/A4Bio/E3-CryoFold) |

The three newer methods passed the screen in decision 2 of the spec. Two reserves are used only if a selected method cannot be run: EM3DFold ([preprint](https://www.biorxiv.org/content/10.64898/2026.09.16.752067.abstract), [code](https://github.com/huang-laboratory/EM3DFold)), which passes the screen on paper but is not yet peer reviewed, and then CryoZeta ([preprint](https://doi.org/10.64898/2026.02.13.705846), [code](https://github.com/kiharalab/CryoZeta)), whose README asks for 32 GB or more of GPU memory. Every method considered, with the reason for each exclusion, is in `methods/methods_table.csv`. Which release of CryoAtom is run (the NSMB paper's code or the repository's current CryoAtom2 v2.1.0) is decided in Phase 4 and named in the report.
