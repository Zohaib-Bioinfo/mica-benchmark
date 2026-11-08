# Notebooks exactly as they were run

These are the notebooks that were run on Kaggle (or Colab) and whose result zips are in the analysis. They are kept as they were; do not rebuild them.
The generators in the folder above have moved on since (the notebooks built now stop at once when a required website cannot be reached, and the main notebook
has a `--lanes` option); the changes are listed in `LOG.md` (2026-10-24; since 2026-11-07 a website that does not answer is also asked again, up to four times, before the notebook stops). `select_colab.ipynb` and `pull_pool_colab.ipynb` (the selection) stay in the folder above.

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
| `gpu_cryoatom_a.ipynb` | CryoAtom v1.0.0 on all 20 entries, 2 GPUs, one session of 8.7 hours | 16 models; 4 runs ended in CUDA out of memory (9OC0, 13BD, 9VBD, 9Z24): LOG 2026-10-31 and 2026-11-03 |
| `scoring_cryoatom.ipynb` | scoring of the CryoAtom models (CPU) | 32 models scored: LOG 2026-11-03 |
| `gpu_cryozeta_a.ipynb` | CryoZeta, first session | stopped in the set-up: the program could not find two data files that the notebook had not downloaded; the folder of its result carries `NOT_A_RESULT.txt` and no row of it is used: LOG 2026-11-03 |
| `gpu_cryozeta_b.ipynb` | CryoZeta, all 20 entries, 2 GPUs, 11 hours | 8 entries ok, 9WME cut by the end of the session (4.9 hours), 26DK killed for lack of RAM while the other GPU ran, 9ZLH not run (a download error), 9 entries not started: LOG 2026-11-04 |
| `scoring_cryozeta_b.ipynb` | scoring of the 8 CryoZeta models of batch b (CPU) | 40 models scored (8 entries x 5 samples): LOG 2026-11-04 |
| `gpu_cryozeta_c.ipynb` | CryoZeta, the 12 entries without a result, 2 GPUs, limit 9 hours (ended after 7.25 hours: no entry that was still waiting fitted into the time left) | 3 models (9ZLH, 9WTX, 9RWS), 3 failures (26DK killed for lack of RAM also alone; 9Z24 at the 6-hour limit; 13BD out of GPU memory), 9OC0 killed while the other GPU ran and not run again, 5 entries not started: LOG 2026-11-05 |
| `scoring_cryozeta_c.ipynb` | scoring of the 3 CryoZeta models of batch c (CPU) | 15 models scored (3 entries x 5 samples): LOG 2026-11-05 |
| `mica_dock3_a.ipynb`, `mica_dock3_b.ipynb`, `mica_dock3_c.ipynb`, `mica_dock3_d.ipynb` | MICA, pass 3: programs 1 and 2 (domain split and docking) on the CPU, the four sets of five entries | 229, 459, 229 and 275 minutes: LOG 2026-10-27 |
| `mica_gpu.ipynb` | MICA, program 3 on the GPU (T4 x2) for the 20 entries, `HOURS` 9 | 143 minutes; 8 entries ok, 3 failed, 9 deferred by the time guard: LOG 2026-10-27 |
| `mica_gpu_nodock.ipynb` | MICA, program 3 without docked AlphaFold input (the sensitivity analysis), `HOURS` 5 | 105 minutes, run at the same time as `mica_gpu`: LOG 2026-10-27 |
| `mica_gpu2_a.ipynb`, `mica_gpu2_b.ipynb` | MICA, pass 4: program 3 for the 12 entries without a MICA result, one window at a time (the memory patch of LOG 2026-10-27), `HOURS` 3.5 | 195 minutes (a, `_INCOMPLETE`) and 190 minutes (b); 9 more models: LOG 2026-10-28 |
| `mica_gpu3.ipynb` | MICA, pass 5: program 3 for 9OC0 and 9VI4, `HOURS` 4.2 | 186 minutes; both entries failed (9OC0 time limit, 9VI4 killed for lack of RAM): LOG 2026-10-28 |
| `gpu_cryozeta_d_try1.ipynb` | CryoZeta, batch d (the 6 entries without a result), first try, 2026-11-07 08:56 PKT | stopped after 25 seconds at the start check of the internet (one TLS time-out of the EBI server), no run; the folder of its result carries `NOT_A_RESULT.txt`: LOG 2026-11-07. This file is the notebook of the second try with the old check put back; nothing else differs |
| `gpu_cryozeta_d.ipynb` | CryoZeta, batch d, second try: the 6 entries without a result, 2 GPUs, `HOURS` 11 (it ended after 10.8 hours) | no model: 9OC0 killed for lack of RAM also when it ran alone, 9VBD and 9VUH at the 6-hour limit (failed), 9VI4 killed while the other GPU ran and not run again alone (no time left), 9WFL and 9WME cut by the end of the session after 4.4 and 4.1 hours: LOG 2026-11-07 |
| `scoring_cryozeta_d.ipynb` | scoring of the models of batch d (CPU) | stopped after the installation, because batch d built no model: nothing to score; the folder of its result carries `NOT_A_RESULT.txt`: LOG 2026-11-07 |
| `gpu_cryozeta_e` (no notebook file, see below) | CryoZeta, batch e: the notebook of batch d started again by Zohaib with the list 9VUH, 9WFL, 9WME (9VUH first and alone), 2 GPUs, `HOURS` 11 (it ended after 10.8 hours) | no model: 9VUH failed again at the 6-hour limit (second run of that entry, alone), 9WFL and 9WME cut by the end of the session after 4.7 hours: LOG 2026-11-08 |
| `scoring_cryozeta_e` (no notebook file, see below) | scoring of the models of batch e (CPU) | stopped after the installation, because batch e built no model: nothing to score; the folder of its result carries `NOT_A_RESULT.txt`: LOG 2026-11-08 |

The nine MICA notebooks of passes 3 to 5 (`mica_dock3_*`, `mica_gpu*`) were copied here on 2026-11-07 as they were in `scripts/` until then (equal to a fresh build of the generators of that time). The copies in `scripts/` were then rebuilt with the new start check, and the tests compare them with a fresh build as before. `mica_gpu2_nodock_a` and `mica_gpu2_nodock_b` were built but never run, so they are not here.

The two notebooks of batch e were made and run by Zohaib from those of batch d and are not kept here. Their run records are in `results/raw/`: the record of `gpu_cryozeta_e` lists the three tasks and its input files have the hashes of batch d; the input files of `scoring_cryozeta_e` have those of `scoring_cryozeta_d`. The zip of `gpu_cryozeta_e` was written under the name of batch d (`gpu_cryozeta_d_result_20261108-0417`) and renamed by Zohaib to `gpu_cryozeta_e_result_20261108-0417`; the repository uses the second name.
