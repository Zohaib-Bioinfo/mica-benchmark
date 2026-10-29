# Notebooks exactly as they were run

These are the notebooks that were run on Kaggle (or Colab) and whose result zips are in the analysis. They are kept as they were; do not rebuild them.
The generators in the folder above have moved on since (the notebooks built now stop at once when a required website cannot be reached, and the main notebook
has a `--lanes` option); the changes are listed in `LOG.md` (2026-10-24). `select_colab.ipynb` and `pull_pool_colab.ipynb` (the selection) stay in the folder above.

| notebook | run | result |
|---|---|---|
| `pilot_mica.ipynb` | first pilot, entry 26DK | stopped in docking (see LOG, 2026-10-19) |
| `dockdiag.ipynb`, `dockdiag2.ipynb` | docking tests on 26DK | LOG 2026-10-21/22 and 2026-10-24 |
| `pilot_methods.ipynb`, `pilot_modelangelo.ipynb` | pilots of the four other methods on 26DK | LOG 2026-10-21/22 |
| `survey.ipynb` | map survey, ran with Internet off, not rerun | LOG 2026-10-24 |
| `gpu_batch1.ipynb` | ModelAngelo, E3-CryoFold, EMProt, EModelX(+AF) on all 20 entries, 2 GPUs | LOG 2026-10-24 |
| `mica_cpu_a.ipynb`, `mica_cpu_b.ipynb`, `mica_cpu_c.ipynb` | MICA, pass 1: programs 1, 2 and (small maps) 3 on 20 entries, docking limit 45 minutes with Phenix's default search resolution | 14 of 59 domains docked, 12 entries without a docked domain: LOG 2026-10-25 (pass 2 replaces them) |
| `gpu_batch2.ipynb` | the 13 open baseline tasks, one program at a time | LOG 2026-10-25 (8 tasks are open: `gpu_batch3`) |
| `mica_dock2_a.ipynb`, `mica_dock2_b.ipynb`, `mica_dock2_c.ipynb`, `mica_dock2_d.ipynb` | MICA, docking pass 2: the same four sets of five entries, a search resolution added | Phenix 2.2.1-6174 rejected the added parameter (a True/False parameter where a number was passed); only the call of pass 1 ran again, 12 of 59 domains docked: LOG 2026-10-25/26 (pass 3, `mica_dock3_*` in the folder above, replaces it) |
| `gpu_batch3.ipynb` | the 8 baseline tasks that `gpu_batch2` left open | LOG 2026-10-25 |
