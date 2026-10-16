# MICA Benchmark Phase 0 Spec

Locked 2026-10-10 · Owner: Muhammad Zohaib · Report deadline 2026-10-30 (send 2026-10-29)

Scope is locked: a free-compute benchmark of MICA against ModelAngelo, EModelX(+AF) and up to three newer methods on about 20 cryo-EM maps released in 2026. Any change after today is written in LOG.md first, with its reason.

## 1. Goal and deliverables

The goal is to measure how well MICA builds protein models on unseen 2026 maps, compared with the two methods in its own paper and with newer methods.

1. A PDF report of about 8-12 pages plus an appendix. It follows Prof. Cheng's order (methods, datasets, benchmarking procedure, evaluation metrics, results, conclusions) and adds a limitations section.
2. A public GitHub repo with the scripts, the frozen entry list, the environment files and the raw per-entry metrics. Every table in the report regenerates from it.

## 2. Locked decisions

| # | Decision | Locked value |
| --- | --- | --- |
| 1 | Methods | MICA, ModelAngelo, EModelX(+AF), plus up to 3 newer methods that pass the screen in row 2. Every version or commit is pinned in `methods/methods_table.csv`. |
| 2 | "Newer" screen | Published after the two baselines, or a new major release of one; deep learning; builds protein atomic models from a map plus sequence; public code and weights; runs on a free 16 GB GPU; documented defaults, no per-map tuning. Every exclusion is logged with a reason. |
| 3 | Release window | EMDB map and PDB model both released from 2026-01-01 to the freeze date 2026-10-12 (inclusive). Later releases are ignored. |
| 4 | Entry filters | Single-particle; resolution ≤4.0 Å; protein only (no nucleic-acid chains); ≤700 modeled residues in total; ≤3 unique chain sequences; ≤5% UNK residues; map-model CC_mask ≥0.70 (`phenix.map_model_cc`, identical settings for every entry). |
| 5 | Selection | MMseqs2 clusters at 30% identity on each entry's longest chain; one entry per cluster, no cluster used twice. Two strata: ≤3.0 Å and >3.0 to 4.0 Å. In each stratum, shuffle with seed 2026 and take the first 10 entries that pass every filter. Target 20 entries, floor 12. The list is committed to `data/entries_2026.csv` before any run. |
| 6 | AlphaFold inputs | Primary: the same AF3 predictions (AlphaFold Server, `model_0` per unique chain) go to every method that accepts predicted structures; ModelAngelo runs on map plus sequence only. Secondary, if time allows: each method in its own default AlphaFold mode on a subset. |
| 7 | Metrics | The paper's six: TM-score, aligned Cα length and sequence identity (US-align); Cα match and sequence match (Phenix `chain_comparison`); Cα quality score (Cα match × coverage, as the paper defines it). Added, because the email to Prof. Cheng listed them: Cα RMSD over aligned residues (US-align) and residue coverage (built residues ÷ deposited residues). Also completion rate and indicative runtime. Protein chains only. |
| 8 | Statistics | Each method is compared with MICA on entries where both produced a model. Primary test: paired Wilcoxon signed-rank on TM-score. For comparability with the paper: paired t-test on TM-score. 95% bootstrap CIs (10,000 resamples of entries), Holm correction across method pairs, results per stratum and pooled. With about 20 entries, effect sizes and CIs matter more than p-values. |
| 9 | Novelty check | For each entry, record the highest sequence identity of its chains to any pre-2026 PDB chain (MMseqs2). The main result uses all selected entries; the subset at ≤30% identity is a sensitivity analysis. |
| 10 | Compute | Free only, no paid services. Kaggle notebooks first (reported ~30 GPU-hours per week; to verify), Colab free as backup. Every run logs `nvidia-smi`, `nproc` and RAM. |
| 11 | Failures | Per-run limit 11 h, inside the 12 h session cap. Crashes, out-of-memory errors and timeouts count as failures and stay in every table with the reason. One rerun is allowed only if the platform killed the session. A sensitivity analysis scores failures as 0 on every metric. |
| 12 | Scope-cut order | If time or quota runs short: (1) fewer entries, dropped from the bottom of the seeded order, down to the floor of 12; (2) apply a ≤500-residue cap to the frozen list; (3) drop one newer method. Never cut the filters, the failure logging, the statistics or the report, and never drop an entry because of its results. |
| 13 | Report | PDF, about 8-12 pages plus an appendix, in Prof. Cheng's order plus limitations, with the repo link. Sent 2026-10-29 PKT. |

## 3. Checkpoints

1. 2026-10-12: candidate pool frozen. Every 2026 release up to this date is pulled.
2. 2026-10-16: entry list committed to `data/entries_2026.csv`; methods table final.
3. 2026-10-18: protocol frozen after a pilot run: commands, versions and metric definitions written down.
4. 2026-10-24: all full runs finished or logged as failures.
5. 2026-10-26: analysis done; every table and figure regenerates from the repo.
6. 2026-10-28: report draft complete.
7. 2026-10-29: QA done; report and repo link sent. 2026-10-30 is the buffer day.

## 4. To verify in Phase 1

- [ ] US-align and `chain_comparison`: exact command lines (multimer handling, distance cutoff). Reproduce the paper's numbers from MICA's released results before trusting the evaluation scripts.
- [ ] Cα quality score: the paper defines it as Cα match (precision) × coverage (built residues ÷ deposited residues); confirm against the Cryo2Struct reference code and its percent scale.
- [ ] Phenix: password received; the version works with MICA (the README example uses 1.20.1) and with `map_model_cc`.
- [ ] GPU fit: MICA, ModelAngelo and EModelX(+AF) run within 16 GB of GPU memory (the paper timed MICA on an 80 GB A100).
- [ ] AlphaFold Server: daily job cap, and whether one job may hold several chains.
- [ ] Kaggle: cores, RAM, GPU type, Internet toggle, weekly quota, whether CPU-only sessions avoid the GPU quota, and a storage plan (20 GB of saved disk against ~10 GB of ModelAngelo weights, Phenix and the environments).
- [ ] EModelX(+AF): the AF3 CIF converts to `ranked_0.pdb` and is accepted as a template.
- [ ] Licences: each tool's academic terms allow this use (ModelAngelo, EModelX, Phenix, newer methods).

## 5. Known risks

- Deposited models may have been built with ModelAngelo-like tools, which can favour them. The report states this as a caveat.
- Training cut-offs of newer methods are unknown, so homologues of 2026 proteins may sit in their training data. The identity column and the ≤30% subset measure this.
- About 20 entries of ≤700 residues limit how far the results generalise: the paper's standard test maps run up to 11,109 residues. The cap comes from compute and is stated in the report.
- MICA's Phenix docking is CPU-bound (85-97% of its runtime on 24 cores in the paper). On the few cores a free platform offers it may run several times slower, so docking goes in CPU-only sessions.
- Free GPUs differ between sessions, so runtimes are indicative only.
- Feeding AF3 models to EModelX(+AF) is a protocol choice, not its default. The secondary run in its default mode checks the effect.
- Phenix access is on the critical path: MICA's docking and refinement need it, and so does the CC_mask filter.

## 6. Working rules

- Free only: no paid tiers, no credit-card trials, one account per platform.
- Commit to GitHub daily and write a dated entry in LOG.md.
- Write each protocol step before running it. No per-map tuning. Never drop a failed run.
- Never edit raw outputs by hand. Every number in the report comes from a script in the repo.
- Understand every number and every choice; the Zoom interview will probe them. Answer honestly if asked about AI assistance.
