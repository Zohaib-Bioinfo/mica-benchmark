# Project log

Newest entry first. One entry per working day: what was done, what was decided, what broke, and the next actions.

## 2026-10-17: The MICA pilot notebook is built and tested (Phase 4)

**Done**
- GitHub repo of Zohaib checked against package v9: 104 files, all identical. `.gitignore` is missing there because the web upload drops files that start with a dot; he adds it by hand (Add file, Create new file, name `.gitignore`).
- `scripts/mica_inputs.py` (10 tests) builds the folder that MICA reads for one entry: the FASTA file with MICA's header format (`>26DK_1|Chains A|...`, chain letters run through the entry) and `AF3_results/<pdb>_<k>/fold_<pdb>_<k>_model_0.cif` from the files of AlphaFold Server. `scripts/evaluate_models.py` 0.1 (7 tests) scores a model with US-align (`-mol protein -outfmt 2`): TM-score normalised by the deposited model, C-alpha RMSD, aligned length, sequence identity, coverage.
- MICA environment without conda: `env/mica_requirements.txt` has 171 exact pins (Python 3.8.20 managed by `uv` 0.13.1, PyPI wheels; torch 1.8.1 from PyPI is the CUDA 10.2 build). Rebuilt here from that file in 65 s (wheels cached): the imports of MICA's own modules work, `run.py --help` works, MICA's bundled pulchra runs, and MICA's first program (domain split with Merizo) ran on the real 26DK prediction and gave 2 domains. US-align compiled from the pinned commit (`746ad8a`, version 20261012) in 45 s with `g++ -O3 -ffast-math -o USalign USalign.cpp -lm`.
- `scripts/make_pilot_notebook.py` builds `scripts/pilot_mica.ipynb` (35 tests, one of them builds the real Python environment with `uv`; with the earlier tests 213 pass). Its end-to-end tests run the whole notebook as a script against stand-ins: a Phenix download server with user name and password, a fake Phenix installer, a local git repository that plays MICA (programs with MICA's command lines), and local servers for the map, the deposited model, the weights and the Zenodo listing; they cover a complete run, a wrong password, a missing secret, a failing MICA program, a silent refinement failure, a failing `chain_comparison` and a second run in the same folders.
- Phenix documentation for the scoring tool (page "Comparison of CA positions in two models"): `phenix.chain_comparison target.pdb query.pdb`, the target first; `max_dist = 3.` (Angstrom) by default, `chain_type = PROTEIN`. It counts how many CA atoms of one model match CA atoms of the other (within `max_dist`), how many of them point in the same direction and how many residue names agree. The page does not describe an RMSD or a "sequence match" line; the pilot saves the raw output so that the parser is written from real output.

**Decided** (no locked value of the spec changes)
- The pilot runs MICA exactly as `MICA_pipeline.sh` does: the same three programs with the same arguments (the `python` of the MICA environment instead of the one on the PATH). Contour level and resolution come from MICA's own `utils/emdb_extractor.py` (EMDB API), with the EMDB header file as the second way; the values are recorded.
- The pilot scores every model MICA writes (C-alpha model, all-atom model, Phenix-refined model) and two controls: the AlphaFold prediction alone and the docked AlphaFold model. Which of MICA's files counts as its model in the benchmark is decided after the pilot.
- Time, GPU memory peak and lowest free RAM are recorded for each of MICA's three programs (decision 10 and the GPU-fit item of section 4 of the spec).

**Next actions**
- [ ] Zohaib: add `.gitignore`; run `scripts/pilot_mica.ipynb` on Kaggle (GPU, Internet on, two Phenix secrets, the AlphaFold zip as a private dataset) and send the result zip.
- [ ] After the pilot: fix what it shows; pin the MICA protocol and the evaluation commands by 2026-10-18; pilots for ModelAngelo, EModelX(+AF), CryoAtom, EMProt and E3-CryoFold; write the parser for `chain_comparison` and fix the C-alpha quality score.
- [ ] Recheck EM3DFold (tag, licence, journal status).

## 2026-10-16: The 26 AlphaFold Server predictions are back and checked (Phase 4 input complete)

**Done**
- Zohaib uploaded `data/af3/af3_server_jobs.json` with "Upload JSON" on alphafoldserver.com at 09:34 (PKT): all 26 jobs appeared in his history as saved drafts under their job names (screenshot), so the file format works on the server. He ran them and sent the 26 files `fold_<job>_model_0.cif` at 11:30 (`AlphaFold_ServerJOBS.zip`: 26 files in one folder, nothing else).
- New `scripts/check_af3_results.py` (17 tests). For every job in `data/af3/af3_jobs.csv` it checks: the file exists once, one model, one chain, amino acids only, the sequence equals the submitted sequence letter by letter, every coordinate is finite, every pLDDT lies between 0 and 100. Result: 26 of 26 ok (`data/af3/af3_results_check.csv`; sha256 of every file in `data/af3/af3_files_sha256.txt`). The 9-residue peptide 9V4D_E2 equals the submitted sequence (G in place of X).
- Confidence of the predictions (mean pLDDT of the C-alpha atoms per chain): lowest 62.4, median 87.2, highest 97.5. The chains with many low-confidence residues: 9VUH_E3 (mean 62.4; 43% of residues below 50), 9V4D_E1 (70.1; 25%), 9VUH_E2 (79.7; 22%) and 9KQM_E1 (75.7; 20%). They go to the methods unchanged (decision 6); the report can break the results down by pLDDT.
- Tests: 162 pass (25 + 7 + 68 + 17 + 11 + 10 + 7 + 17). On the newer sandbox host the AVX2 build of MMseqs2 18-8cc5c stopped with "Illegal instruction" (6 tests that run the real program failed); the SSE4.1 build of the same release (`mmseqs-linux-sse41.tar.gz`, commit `8cc5ce36...`) passes them all. This concerns only the tests here; the Colab runs used their own download.

**Decided** (no locked value of the spec changes)
- The predictions stay private (AlphaFold Server terms: non-commercial research use): `.gitignore` already excludes `*.cif` and `*.zip`, and now also `af3_models/` and `fold_*`. The public repo has the job list, the check table and the file hashes. No job is repeated.

**Next actions**
- [ ] Phase 4: pilot of every method on one small entry (26DK: 197-residue chain, 16 MB map) to pin versions and commands by 2026-10-18; MICA and ModelAngelo first. The 26 predictions go to the notebooks as a private upload (Colab file upload or a private Kaggle dataset).
- [ ] 2026-10-16: recheck EM3DFold (tag, licence, journal status).
- [ ] Zohaib: create the GitHub repo and push this folder; check that Kaggle has the phone verification (needed for Internet and GPU in notebooks).

## 2026-10-15: The entry list is frozen (20 entries); AlphaFold Server jobs prepared

**Done**
- Second Colab run of `scripts/select_colab.ipynb` (10:55 to 12:04 UTC on 2026-10-14, 69 minutes, 3 rounds; `selection_result_20261014-1204.zip`): FINAL, 20 entries, 10 up to 3.0 A and 10 above 3.0 to 4.0 A. Phenix 2.2.1 `phenix.map_model_cc ... --force` gave CC_mask for 39 of 43 evaluated entries, always through variant 1 (output line `  CC_mask  : x`, one per run, as expected). The novelty step ran (MMseqs2 18-8cc5c, `pdb_seqres` sha256 `3f5c502b...`; reference: 177,158 distinct protein sequences from pre-2026 entries after removing 14,077 entries that RCSB lists as released from 2026-01-01).
- Checked here, not only in Colab: the input hashes in `run_info.json` equal the repo files (so the script that ran is `compute_cc_mask.py` 1.1 of the repo); every selected row agrees with the pool and with `cc_mask.csv` (stratum, CC_mask >= 0.70, residues, chain counts); 20 distinct clusters, 20 distinct EMDB maps, 20 distinct PDB IDs; running `select_entries.py select` again on this machine with the returned `cc_mask.csv` gives the same 20 rows in the same order and a byte-identical `selection_log.csv`.
- Saved in the repo: `data/entries_2026.csv` (sha256 `cb695bb0...`), `data/selection/cc_mask.csv` (`1b35b862...`), `selection_log.csv`, `selection_info.json`, 43 Phenix logs in `cc_logs/`, `novelty/novelty.csv` and `novelty_info.json`, and the logs of both Colab runs in `colab_runs/` (the first one is the stop at the CC_mask test).
- `scripts/make_af3_jobs.py` (7 tests) writes `data/af3/`: 26 jobs, one per unique chain sequence of the 20 entries, ordered by the seeded order: `af3_jobs.csv`, `af3_jobs.fasta` and `af3_server_jobs.json` (for the button "Upload JSON" of AlphaFold Server; the format was checked on 2026-10-15 against the reader of this format in the AlphaFold 3 source, `src/alphafold3/common/folding_input.py` of `google-deepmind/alphafold3`, branch main, file sha256 `6e394fde...29f0`, function `from_alphafoldserver_fold_job`: a top-level list; per job only the keys `name`, `modelSeeds`, `sequences`, `dialect` = "alphafoldserver", `version` = 1; one `proteinChain` with `sequence` and `count` per entry of `sequences`; all 26 jobs pass these checks; it has not been tried on the server itself yet. Server defaults stay: `modelSeeds` empty, so the server picks a random seed; templates as the server uses them by default. Both go into the methods text). AlphaFold Server allows 30 jobs per day and 5,000 tokens per job (EBI training page), so one day is enough.

**Found**
- CC_mask by stratum, all evaluated entries: up to 3.0 A, 18 evaluated, 1 below 0.70, 2 not computable; above 3.0 A, 25 evaluated, 9 below 0.70 (0.54 to 0.69), 2 not computable. The final walk used 28 of the 43: 20 selected, 6 rejected for CC_mask below 0.70 (all above 3.0 A), 2 rejected because CC_mask could not be computed; 3 more were skipped because their cluster was already used (9ZKC, 9KQ5, 9WFC); 14 were evaluated but not needed (the batch of 16 per stratum). The CC_mask criterion mainly removes entries above 3.0 A; median CC_mask of the evaluated entries: 0.83 up to 3.0 A, 0.73 above.
- Phenix could not score 4 entries (9WNG, 23WN, 9XQN, 9VJ3): "The model contains atoms which are not in the scattering table "electron". Unknown atom types: O1-". Two of them were reached by the final walk: 9WNG, and 9XQN, which was the first entry of the seeded order above 3.0 A. By decision 4 an entry without a computable CC_mask is not accepted (selection log: `reject_cc_failed`). This is a limitation for the report: entries whose models use the atom type O1- (a charged oxygen) cannot be in the benchmark.
- Novelty (decision 9; highest identity to a pre-2026 PDB chain, chain covered to at least 50%): 13 of 20 entries at 0.95 or more (the protein is already in the PDB, usually in an older entry; 22FX is human apoferritin), 4 between 0.44 and 0.76 (9RWS 0.441, 9V4D 0.503, 9T5K 0.614, 9WTX 0.763) and 3 at 0.00: 9TKA, 9VBD, 26DK. The sensitivity subset of decision 9 (<= 30%) therefore has 3 entries: it can be described, not tested. The 2026 maps are new; for most entries the protein is not. The report states this next to the main result.
- Cluster sizes of the chosen entries (candidates in the entry's 30% cluster): up to 3.0 A: 1, 1, 39, 3, 6, 9, 7, 1, 8, 25; above 3.0 A: 11, 6, 2, 9, 1, 2, 1, 3, 4, 2. The two largest clusters (39 for 9JKK, 25 for 13BD) are common specimens: the rule gives one entry per cluster, so each is represented once.
- Modeled residues: median 512 (173 to 682) up to 3.0 A, 446 (191 to 534) above. Maps: 3.4 GB compressed in total; the largest is 9VI4 (1.8 A, 1,024 MB), then 9OC0 (333 MB) and 13BD (281 MB); a 1 GB map took 12 minutes to download, almost all of the time of that entry.

**Decided** (no locked value of the spec changes)
- The list is frozen as it is, so checkpoint 2 (2026-10-16) is met early. No entry is added or replaced. The four O1- failures are not repaired by editing models: that would change the procedure after the data were seen.
- AlphaFold inputs: one monomer job per unique chain sequence (26 jobs). 9V4D_E2 (a 9-residue peptide whose first residue is X in the deposited sequence) is submitted with G in place of X, flagged in `af3_jobs.csv` and to be named in the report. The AF3 predictions are not committed to the public repo (Server terms: non-commercial research use; only job lists and scores are public).

**Next actions**
- [ ] Zohaib: sign in at alphafoldserver.com, upload `data/af3/af3_server_jobs.json` with "Upload JSON", run the 26 jobs (30 per day), download the zips and send the 26 files `fold_*_model_0.cif`.
- [ ] Phase 4: pilot of every method on one small entry (26DK: 191 residues, 16 MB map) to pin versions and commands by 2026-10-18; MICA and ModelAngelo first.
- [ ] 2026-10-15: recheck EM3DFold (tag, licence, journal status).
- [ ] Zohaib: create the GitHub repo and push this folder (the entry list is public from then on).

## 2026-10-14: First Colab run of the selection notebook; Phenix 2.2.1 has retired `phenix.map_model_cc`

**Done**
- Zohaib ran `scripts/select_colab.ipynb` (02:51 to 03:11 UTC, 20 minutes; `selection_result_20261014-0311_INCOMPLETE.zip`). On the real servers this worked: the Phenix 2.2.1-6174 installer downloaded with his login (3.5 GB, sha256 `e2c76600...5885`), `bash installer -b -p /content/phenix-2.2.1-6174` installed it in 11 minutes, `phenix.version` printed "Version: 2.2.1", all 7 embedded files matched their hashes, and the first selection round ran (484 candidates; CC_mask needed for 32 entries, 16 per stratum). The Colab runtime had 2 CPU cores, 12.7 GB RAM, 94 GB free disk, Python 3.13.16.
- The CC_mask test on the first two entries (9T5K with a 250 MB map, 22FX with a 75 MB map) failed: `phenix.map_model_cc` exited with 1 and said "phenix.map_model_cc and phenix.model_map_cc are replaced by phenix.map_correlations. Add the keyword '--force' to use anyway". The notebook stopped there and packed the zip, as designed. The two failed rows say something about Phenix, not about the entries, so they were not used.
- Fix, `compute_cc_mask.py` 1.1. For the first entries it tries six ways to call Phenix, in this order, until one gives a CC_mask: (1) `phenix.map_model_cc <model> <map> resolution=R ignore_symmetry_conflicts=True --force`, (2) the same with `--force` first, (3) `phenix.map_correlations <model> <map> resolution=R ignore_symmetry_conflicts=True`, (4) `phenix.map_correlations` with full parameter names (`input_files.model`, `input_files.map_in_1`, `map_model_cc.resolution`, `map_model_cc.ignore_symmetry_conflicts`, as listed on the Phenix documentation page of that program), and (5, 6) the same programs without `ignore_symmetry_conflicts`, only if Phenix says it does not know that parameter. The variant that gives the first CC_mask is then the only one used for all later entries (column `variant` of `cc_mask.csv`; a later run reads it from there), so the settings stay identical for every entry. The old program prints one line `  CC_mask  : x` (as in cctbx's own test of it); the reader also accepts tables and files that Phenix wrote.
- Other changes: a gzip file cut off in the middle is a temporary problem (before it stopped the script); the files of the next two entries download in background threads while Phenix works on the current one (the first run needed 99 s and 222 s per entry, almost all of it download); `--diagnose` writes Phenix's help texts to `diagnostics.txt` when no variant works; the notebook moves the rows of a failed test to `cc_mask_FAILED_TEST.csv`, so a rerun in the same session does not count them as rejections.
- Tests: 138 pass (25 + 7 + 68 + 17 + 11 + 10). The stand-in for Phenix refuses without `--force`, like 2.2.1; new tests cover every variant, the remembered variant, the background downloads and the failed-test rows.

**Decided** (no locked value of the spec changes)
- The program named in decision 4 stays `phenix.map_model_cc`; where Phenix 2.2.1 needs it, with `--force`. If only the replacement works, the methods text says that CC_mask came from `phenix.map_correlations` (same quantity, Afonine et al. 2018). The notebook records the variant.

**Found**
- Not yet seen on real servers: the output of the CC_mask program, the RCSB all-methods search and the `pdb_seqres` download (the novelty step). Each step stops with a plain message and a zip with the log.
- Possible limit: a map of 1 GB or more may not fit into the 12.7 GB of RAM of a free Colab runtime when Phenix reads it; such an entry would show as "killed by signal 9" and, by decision 11, stays in the tables as a failure.

**Next actions**
- [ ] Zohaib: upload the new `scripts/select_colab.ipynb` to Colab, Runtime, Run all, type the Phenix user name and password when asked, keep the tab open, and send the new `selection_result_*.zip` (the test on two entries now shows within about 25 minutes whether CC_mask works).
- [ ] On return of the zip: read `run.log` first, then `cc_logs/` of the first entries (which variant worked, the raw CC_mask lines), verify counts and hashes, copy `entries_2026.csv`, `cc_mask.csv`, `selection_log.csv`, `selection_info.json` and `novelty/` into the repo, and record the numbers here.
- [ ] 2026-10-15: recheck EM3DFold (tag, licence, journal status). 2026-10-16: `data/entries_2026.csv` committed; methods table final. 2026-10-18: versions pinned and protocol frozen.

## 2026-10-13: Phase 3b and 3c, the full pool, the selection tools and the one-run notebook

**Done**
- The full pool run (script 1.1, sha256 `1b466057...`, the same file that was tested) ran in Colab on 2026-10-12 from 11:26 to 11:39 UTC: exit code 0, 0 download errors, canary OK, peak memory 3.2 GB. 6,116 cryo-EM entries in 41 weekly windows; one file was not parsed because of its size. Flow: 6,116, file size -1, single particle -415, resolution -481, protein only -1,242, modeled residues -3,410, unique sequences -19, UNK 0, exactly one EMDB map -63, EMDB map released in the window -1 = **484 candidates** (180 at up to 3.0 A, 304 above 3.0 to 4.0 A). The EMDB release date was verified for all 484.
- Checked outside the script: every candidate row satisfies every filter again (independent re-check of the csv, no failure); every candidate has exactly one longest chain in the FASTA file. Files, hashes and provenance: `data/pool/` and `data/pool/README.txt`.
- Run 1 against the full run: the 453 candidates of run 1 are all in the 484; the other 31 pass every file-based filter (225 to 690 modeled residues, 1.7 to 3.94 A, one to three protein entities). So the server-side prefilters lost 6% of the pool. Which one is responsible was not diagnosed and does not matter now; the first suspect (`protein_only`: protein plus sugar chains) is unconfirmed.
- Clustering (`select_entries.py cluster`): MMseqs2 release 18-8cc5c (commit 8cc5ce36...), `easy-cluster --min-seq-id 0.30 -c 0.8 --cov-mode 1 --threads 1`, one thread so the result repeats; input = the longest chain of each candidate, sorted by PDB ID. 484 candidates fall into **179 clusters** (99 single entries); the largest hold 39, 25, 15, 14, 12 entries. Files: `data/selection/candidate_clusters.csv`, `mmseqs_info.json`.
- New scripts, all tested offline: `select_entries.py` (clusters, seeded walk, rounds; 68 tests), `compute_cc_mask.py` (Phenix, one entry at a time, resume-safe; 10 tests with a stand-in for Phenix), `novelty_check.py` (11 tests, MMseqs2 for real), `make_select_notebook.py` and `select_colab.ipynb` (8 end-to-end runs of the notebook in a local Jupyter kernel with a fake Phenix installer and password check). All suites together: 129 tests pass.
- `scripts/select_colab.ipynb` does the rest of the selection in one Colab run: installs Phenix from the download link, tests CC_mask on two entries and shows Phenix's own output, then repeats {apply the rule, compute the CC_mask it asks for} until the list is final, runs the novelty check, and packs one zip (`selection_result_*.zip`). The password is typed in at run time, never written to a file, and every file is scanned for it before it goes into the zip.

**Decided** (no locked value of the spec changes)
- No EMDB map is used twice (entry skipped, like an entry whose cluster is used). In the full pool EMD-67166 is cited by 3 candidates and EMD-64067 by 2.
- The strata are walked in the order up to 3.0 A, then above 3.0 to 4.0 A, with one shared list of used clusters and maps. Decisions of an unfinished round are provisional (the first stratum may still use clusters the second one has already looked at); only the round that needs no more CC_mask defines the list. Tested on 60 random pools: the list does not depend on how many rounds it takes or on the batch size.
- CC_mask: `phenix.map_model_cc <model.cif> <map.map> resolution=<resolution_A of the entry> ignore_symmetry_conflicts=True` for every entry (a Phenix that does not know the last parameter gets the command without it, and the row says so in `variant`). Result "ok" with a number; "failed" if Phenix gave none, timed out (30 min) or was killed, or if the file is missing on the server (HTTP 404): a rejection that does not use up the cluster or the map. A temporary problem (download error, server busy) is not recorded, so the entry is tried again. Model from RCSB, map from EMDB (`.map.gz`).
- Phenix 2.2.1-6174 (Linux x86_64 installer) is the version of the benchmark; the notebook records the installer's sha256 and the output of `phenix.version` in `run_info.json`. Whether this version also runs MICA's docking and refinement is checked in Phase 4.
- Rounds ask for 16 entries per stratum at first (`--batch 16`), fewer later; only the number of Phenix runs per round changes, not the result.
- Novelty (decision 9): every distinct chain sequence of each chosen entry is searched (`mmseqs easy-search -s 7.5 -e 1e-3`) against all protein chains in `pdb_seqres.txt.gz` minus every entry that RCSB lists as released from 2026-01-01 on (all methods), duplicates removed. `max_id_to_pre2026_pdb` is the highest identity over alignments covering at least 50% of the query chain (0 if none); `max_id_any_coverage` is the highest over all alignments. Both are kept; the sensitivity analysis uses the first. Safety checks stop the run if the pool's 6,116 cryo-EM IDs are not all in the excluded list.
- The seeded order depends on Python's `random.shuffle`; a test pins the order of a reference list, and `selection_info.json` records the Python version.

**Found**
- The entry-level shuffle favors common specimens: 7 clusters hold 126 of the 484 candidates (the largest has 39). Locked rule, unchanged; to be reported as a limitation, with the cluster sizes of the chosen entries (column `cluster_size`).
- Map sizes in the pool: median 105 MB, 90th percentile 432 MB, largest 1,185 MB; 38 maps above 500 MB and 5 of 1 GB or more. Not a filter in the spec; large maps may fail in a method on a free GPU, and such failures stay in the tables (decision 11).
- Not yet seen on real servers: the Phenix installer (`bash installer -b -p DIR`), the output format of `phenix.map_model_cc`, the RCSB all-methods search and the `pdb_seqres` download. The notebook is built so that each of these stops early with a plain message and a zip that holds the log.

**Next actions**
- [x] Zohaib: run `scripts/select_colab.ipynb` in Colab and send `selection_result_*.zip` (done 2026-10-14; it stopped at the CC_mask test, see the entry above).
- [ ] On return of the zip: read `run.log` first and the raw Phenix output of the first two entries (`cc_logs/`), check the CC_mask regex against it, verify counts and hashes, copy `entries_2026.csv`, `cc_mask.csv`, `selection_log.csv`, `selection_info.json`, `novelty/` into the repo, and record the numbers here.
- [ ] 2026-10-15: recheck EM3DFold (tag, licence, journal status). 2026-10-16: `data/entries_2026.csv` committed; methods table final. 2026-10-18: versions pinned and protocol frozen.

## 2026-10-12 (evening): Phase 3a, first real run; the prefilters leaked, so the pool is pulled in full

**Done**
- Zohaib ran `scripts/pull_pool_colab.ipynb` (script version 1.0, sha256 `91e2add2...`) in Colab at 07:19-07:21 UTC. It ran to the end without errors in 80 seconds. Files of this run: `data/pool_run1_prefilter/` (log, run_info, flow, weekly counts).
- The canary read both real entries: 5A1A (EMD-2984, 4,088 modeled residues, released 2015-05-06) and 6VXX (EMD-21452 and EMD-21457, 2,916 residues, released 2020-03-11). So the parser reads real cryo-EM mmCIF files; the open point from the previous entry is closed for the fields the canary checks.
- Search: the first form of every part of the query was accepted (`exptl.method`, plain dates, `paginate`), and all four server-side prefilters were accepted. RCSB lists 6,116 cryo-EM entries released 2026-01-01 to 2026-10-12 in 41 weekly windows; the last window (2026-10-08 to 2026-10-12) is empty, as expected, since wwPDB releases on Wednesdays and the next one is 2026-10-14. No entry had a first-revision date outside its search week.
- After the prefilters 666 entries were downloaded (31 per second, no errors) and 453 passed every file-based filter and the EMDB check: 159 at up to 3.0 A and 294 above 3.0 to 4.0 A. The EMDB API answered for all 453 with `map_release` (status REL), so the fallback to `header_release` was not needed.
- Flow after the prefilters: 666, resolution -11, modeled residues -126, unique sequences -18, exactly one EMDB map -57, EMDB map released in the window -1.

**Found**
- The audit found a leak: of 100 sampled entries that the prefilters had removed, 1 (9VH3) passes every file-based filter. About 1% of the 5,450 removed entries may therefore belong in the pool, so the 453 candidates are incomplete. Which prefilter is responsible is not known. A likely reason, not checked, is the `protein_only` prefilter: RCSB's "Protein (only)" label may exclude entries that also carry sugar chains, while the spec and the file filter exclude only nucleic-acid chains.
- 6VXX lists two EMDB IDs because `_pdbx_database_related` also names a related map of the same paper. The rule "exactly one EMDB map" therefore also removes entries that only cite a related map. It removed 57 of 511 entries (36 with 2 links, 11 with 3, 6 with 4, the rest with 5 to 19). Kept as it is; stated in the report as a narrowing.
- Two EMDB maps are shared by several candidates (EMD-67166 by 3 entries, EMD-64067 by 2). Phase 3b must not select two entries of one map.
- Map file sizes (EMDB, uncompressed): median 95 MB, largest 1.2 GB. Not a filter in the spec; noted for the compute plan and the pilot run.

**Decided** (no locked value of the spec changes)
- The frozen pool is built without server-side prefilters: every one of the 6,116 entries is downloaded and judged on its own file. The run took 80 seconds for 666 entries, so the full pull is cheap. In script version 1.1 the prefilters are opt-in (`--prefilter`); the audit stays for that mode. Comparing run 1 with the full run will show which prefilter leaked; that is a diagnostic only.
- New filter `file_size` (version 1.1): an mmCIF file above 40 MB compressed is not parsed. It holds at least 150 MB of text, which is at least 40,000 residues, so it fails the 700-residue filter in any case; the rule only keeps the free Colab session (12 GB) from running out of memory. Checked with a stress run: 606 entries including six files of about 100 MB (0.9 million atoms each) ran in 22 seconds at a peak of 2.9 GB.
- Phase 3b adds one selection rule: no EMDB map is used twice (an entry whose map is already used is skipped, like an entry whose cluster is used).
- Phenix access: the download password arrived by email on 2026-10-12. It is a shared password that changes on the 1st of each month. It is not written in the repo or in this log; Colab and Kaggle get it from their secrets panel.

**Next actions**
- [ ] Zohaib: run the new `scripts/pull_pool_colab.ipynb` (version 1.1, no prefilters, zip `pool_full_2026-10-12.zip`) and send the zip.
- [ ] On return of the zip: check the log, record the hashes here, compare with run 1, and commit the pool files to `data/pool/`.
- [ ] Phase 3b: selection script (30% clusters of the longest chains, seeded walk, no cluster and no EMDB map used twice) and the CC_mask script for Colab with Phenix. Nothing of the CC_mask step can be tested without Phenix; the script is written so that a one-entry test run in Colab shows its output first.
- [ ] Phase 3c: novelty check against pre-2026 PDB chains (MMseqs2).
- [ ] 2026-10-16: `data/entries_2026.csv` committed; methods table final.

## 2026-10-12: Phase 3a, pool-pull script written and tested offline

**Done**
- Wrote `scripts/pull_pool.py` (version 1.0). It asks the RCSB Search API for every cryo-EM entry released 2026-01-01 to 2026-10-12 inclusive (week by week), downloads each entry's mmCIF file, applies the parts of decisions 3 and 4 of the spec that need no map, checks the EMDB release date of each surviving map, and writes `pool_all_entries.csv`, `pool_candidates.csv`, `pool_flow.csv` (entries left after each filter), `pool_candidate_sequences.fasta`, the ID lists, `search_weeks.csv` and `run_info.json` (versions, settings, hashes).
- Server-side search filters (single particle, resolution up to 4.1 A, protein only, up to 800 modeled residues) only shrink the download. Each is probed first and dropped if the API rejects it or returns nothing, and 100 of the entries they remove are downloaded and checked anyway (the audit). The filters that decide are applied to the files.
- A canary step parses two old single-particle entries (5A1A, 6VXX) before the search and stops the run if the parser reads empty fields from real files.
- `scripts/make_colab_notebook.py` builds `scripts/pull_pool_colab.ipynb`, which carries a copy of the script; one upload and "Run all" produces one zip.
- Tests, all passing: 24 tests of the script against mock RCSB, file and EMDB servers (`scripts/test_pull_pool.py`; filter edges, search-query fallbacks, truncated searches, resume, download errors, audit); 6 end-to-end runs of the notebook in a local Jupyter kernel against the same mocks, including the canary, search and download failures (`scripts/test_colab_notebook.py`).
- Parser check on real files: 13 X-ray and NMR mmCIF files from public test sets parse without error, and the modeled-residue count equals gemmi's own count of amino-acid residues in 11 of them. The other two differ only by non-standard residues inside a protein chain (a C-terminal NH2 cap; two modified residues) that the rule below counts as modeled.
- mmCIF tag names and enumerated values were checked against the wwPDB dictionary file `mmcif_pdbx_v50.dic`, and the request structure of the search (endpoint, query nodes, pagination, status codes) against the source of the official `rcsb-api` client.

**Decided** (operational rules; no locked value of the spec changes)
- An entry must link exactly one EMDB map. Entries that list several maps (focused, consensus or local-refinement maps) are dropped, so that an entry is one map and one model. This is a narrowing of decision 4 and is stated in the report.
- Modeled residues: distinct (asym ID, sequence position) pairs with coordinates in model 1, in entities of type polypeptide(L). Unique chain sequences: distinct deposited sequences among those entities. Longest chain (decision 5): the entity with the longest deposited sequence; a tie goes to the lower entity ID.
- Release date of a PDB model: the date of its first revision. Release date of an EMDB map: `map_release` from the EMDB API; if the API does not give it, `header_release` is used and the entry is flagged `header_date_only` in `emdb_release_verified`.
- Selection in Phase 3b: the two strata are processed in the order up to 3.0 A, then above 3.0 to 4.0 A, with one list of used clusters across both. Within a stratum the candidates are sorted by PDB ID and shuffled with `random.Random(2026)`; the walk follows that order, skips used clusters and applies the CC_mask gate. A CC_mask failure does not use up its cluster. Phase 3b computes CC_mask for about 20 candidates per stratum to start with, and for more if fewer than 10 pass.
- The pool is defined by release dates, so the pull gives the same pool on any day from 2026-10-12 on (wwPDB releases on Wednesdays; the next is 2026-10-14). The freeze date stays 2026-10-12.

**Not verified until the first real run**
- RCSB search attribute names, values and date format are written from the API documentation. The script tries the alternatives it knows, and the run log records which query form worked.
- The EMDB API key names (`map_release`, `header_release`, `current_status`, `size_kbytes`) are searched for anywhere in the JSON; the log and `emdb_release_verified` show if the fallback was needed.
- No real cryo-EM mmCIF file was available offline. The EMDB links (`_database_2`, `_pdbx_database_related`), `_em_experiment`, `_em_3d_reconstruction` and the revision history are tested on synthetic files written to the dictionary and are read for the first time on real cryo-EM files by the canary step.
- Real download speed, rate limits and the number of entries. The estimate is 15 to 40 minutes and several thousand entries in the window.

**Next actions**
- [ ] Zohaib: run `scripts/pull_pool_colab.ipynb` in Colab (steps in the chat) and send back the zip.
- [ ] On return of the zip: read the log first (canary lines, query form, prefilter audit, EMDB check), record the file hashes here, and fix anything the real servers exposed.
- [ ] Phase 3b: write the selection script (MMseqs2 clustering at 30% on the longest chain, seeded walk) and the CC_mask script (Phenix `map_model_cc`, identical settings for every entry). The Phenix password is on the critical path.
- [ ] Phase 3c: novelty check against pre-2026 PDB chains (MMseqs2).
- [ ] 2026-10-16: `data/entries_2026.csv` committed; methods table final.
- [ ] Open items from earlier entries still stand: Phenix password, GitHub repo, Kaggle checks, AlphaFold Server limit, the EM3DFold recheck on 2026-10-15, version pins on 2026-10-18.

## 2026-10-11 (later): Phase 2, EM3DFold code confirmed

**Done**
- The EM3DFold preprint page does not open from the workspace, so its "Code availability" statement was copied by hand: "The EM3DFold package is freely available at https://github.com/huang-laboratory/EM3DFold."
- Read the repository README. It builds protein, RNA and DNA models from a map plus FASTA files (`em3dfold build --map MAP.mrc --protein PROTEIN.fa --output OUT --device 0`) and has a sequence-free mode. It asks for a GPU with at least 12 GB VRAM, CUDA 11.8 or newer and Linux, and fetches its weights with `scripts/download.sh` (at least 6 GB of disk). It states no licence, no release or tag (103 commits), no per-chain predicted-structure input, and no Colab, runtime or tested hardware. It cites the work as "in submission" (Li and Huang, 2026).
- Regenerated `methods/methods_table.csv` (32 rows: 3 baseline, 3 yes, 2 reserve, 24 no) and updated `methods/SEARCH_LOG.md`.
- Tested whether EMDB and RCSB can be reached from the workspace. They cannot: every request is refused by the proxy. The 2026 candidate pool therefore has to be pulled by a script run in Kaggle or Colab with Internet on.

**Decided**
- Correction: "No new method passes the screen" (entry below) was wrong. EM3DFold passes on paper.
- EM3DFold is Reserve 1 and CryoZeta Reserve 2. Under the tie-break peer-reviewed methods come before preprints, so CryoAtom, EMProt and E3-CryoFold keep their places. Decision 1 of the spec is unchanged.
- Proposed CryoAtom default changed from "the repository's latest tag" to "the release that matches the NSMB paper, unless the CryoAtom2 training cutoff is shown to be before 2026-01-01". A model trained after 2026-01-01 could have seen the test entries. Details under "Training cutoffs" in `methods/SEARCH_LOG.md`. Still a Phase 4 decision.

**Not checked**
- The EM3DFold preprint text: training-data cutoff, runtime and its own comparisons.
- EM3DFold's licence and journal status.

**Next actions**
- [ ] 2026-10-15: recheck EM3DFold for a release tag, a licence and journal status.
- [ ] Open question for Zohaib: keep three newer methods (EM3DFold stays in reserve), or add EM3DFold as a fourth? A fourth needs a change to decision 1 of the spec and a log entry with the reason first. Default: keep three.
- [ ] Phase 3: pull the 2026 candidate pool with one script in Kaggle or Colab, apply the filters in decision 3, and write `data/entries_2026.csv`.
- [ ] Open items from the Phase 0 entry still stand: Phenix password, GitHub repo, Kaggle checks, AlphaFold Server limit.

## 2026-10-11 (afternoon): Phase 2, Scholar "Cited by" check screened

**Done**
- Ran the "Cited by" check by hand in Google Scholar, 2026 only, for MICA, CryoAtom, EMProt, E3-CryoFold and CryoAtom's earlier preprint (CryFold). 2026 counts: 9, 34, 9, 9 and 2. Raw notes: `methods/scholar_cited_by_2026-10-11.txt` (33 titles, 25 unique links).
- Screened every title in the notes against the spec and opened the pages that could be opened. Details in `methods/SEARCH_LOG.md`.

**Decided**
- No new method passes the screen. The three newer methods stay CryoAtom, EMProt and E3-CryoFold; CryoZeta stays the reserve. (Corrected in the entry above: EM3DFold passes on paper and is Reserve 1.)
- `methods/methods_table.csv` now has 32 rows. New: CryoForge (repairs another builder's model; no code stated), CryoNet.Discovery (retrieves known structures; no code stated) and five papers screened out by title. CoCoFold is identified as "Fine-tuning AlphaFold with limited cryo-EM observations" (raw particle images; 19-77 GB peak GPU memory). EM3DFold has its full title.
- Tie-break added to the screen (no locked value changes): if more than three methods pass, peer-reviewed publications come before preprints, then the earlier publication date. Never rank by benchmark results.
- CryoAtom: the repository README now describes CryoAtom2 (v2.1.0), a different model from the one in the NSMB paper (2026-04-14 preprint, not peer reviewed). Which release is run is a Phase 4 decision, and the report names the exact release.

**Not checked**
- Titles not copied from the Scholar lists (25 of 34 for CryoAtom, 3 of 9 for EMProt, 2 of 9 for E3-CryoFold) were not re-read.
- The EM3DFold preprint page could not be opened from the workspace (fetch error); a code or weights link may exist. (Resolved in the entry above: the code statement names a public GitHub repository.)
- Scholar records under other titles (for example MICA's earlier bioRxiv title) were not opened.

**Next actions**
- [x] Open the EM3DFold link from the notes file and look for a code or data availability statement (GitHub, Zenodo, Hugging Face). Write the exact sentence here. Done 2026-10-11; the sentence is in the entry above.
- [ ] Skim the model-building section of the Current Opinion in Structural Biology review ("Deep learning-based postprocessing and model building for cryo-electron microscopy maps") and note any method name that is not in the table.
- [ ] Optional: read the 25 uncopied CryoAtom titles once more.
- [ ] 2026-10-15: recheck EM3DFold (public code is confirmed; now for a release tag, a licence and journal status, as in the entry above).
- [ ] Read the LICENSE files; pin versions on 2026-10-18.

## 2026-10-11: Phase 2, newer methods chosen

**Done**
- Scanned the literature for deep-learning cryo-EM model builders published after ModelAngelo and EModelX, and applied the screen in `docs/SPEC.md` (decision 2).
- Read the paper page and the README of every finalist. Year, venue, inputs, weights, GPU need and licence are in `methods/methods_table.csv`; how the search ran, and what it could not reach, is in `methods/SEARCH_LOG.md`.
- Checked README hardware figures against a third-party measurement (CryoACE paper, Appendix H, ~500 residues on one H20: E3-CryoFold ~8 GB, ModelAngelo ~10 GB, CryoAtom ~12 GB).

**Decided**
- Newer methods: CryoAtom (Nat. Struct. Mol. Biol. 2026), EMProt (Nat. Struct. Mol. Biol. 2026) and E3-CryoFold (Nat. Mach. Intell. 2025). That fills the three slots in decision 1, so no spec text changes.
- CryoZeta is a reserve only: its README asks for 32 GB or more of GPU memory.
- Excluded, with reasons in the table: CryoACE (~38 GB; needs an aligned starting model), CryoBoltz (guided sampling; needs an aligned model), EM3DFold (no public code found; corrected later the same day, see the entry above), and the older or out-of-scope methods.
- Correction: E3-CryoFold was first excluded on an inferred link to EModelX. Its own paper (Nat. Mach. Intell. 7, 1091, published 2025-06-24) shows it is a separate, newer method, so it was screened properly and included.

**Not checked**
- Forward citations (who cites MICA, CryoAtom, EMProt): Google Scholar, Semantic Scholar, OpenAlex, Europe PMC and PubMed could not be reached from the workspace.
- GPU fit and licences are from READMEs and papers; nothing has been run yet.

**Next actions**
- [x] Google Scholar: open "Cited by" for MICA, CryoAtom, EMProt, E3-CryoFold and CryoAtom's earlier preprint (CryFold), restrict to 2026, record the result count for each list, and list any deep-learning model builder missing from the table (title and link). Done 2026-10-11; see the entry above.
- [ ] 2026-10-15: recheck EM3DFold (superseded: see the 2026-10-11 (later) entry).
- [ ] Read the LICENSE files: EMProt (README says MIT, paper says academic or noncommercial), EModelX, CryoAtom, E3-CryoFold.
- [ ] 2026-10-18: pin the version or commit of every method in `methods/methods_table.csv`.
- [ ] Open items from the Phase 0 entry still stand: Phenix password, GitHub repo, Kaggle checks, AlphaFold Server limit.

## 2026-10-10: Phase 0, scope locked

**Done**
- Decoded the task email into deliverables and read the MICA paper.
- Wrote and locked the Phase 0 spec in `docs/SPEC.md`: methods, "newer" screen, 2026 release window, entry filters, selection rule, AlphaFold input policy, metrics, statistics, compute plan, failure handling, scope-cut order, report format.
- Chose a strictly free-only compute plan: Kaggle first, Colab as backup, no paid services.
- Created the repo scaffold.
- Checked the spec against the MICA paper's Methods: residue coverage is its own metric (built residues ÷ deposited residues), not Cα match; added the CPU-bound docking risk and a GPU-fit check.

**Decided** (details and reasons in `docs/SPEC.md`)
- Release window: 2026-01-01 to the freeze date 2026-10-12.
- Target 20 entries (floor 12), at most one per 30% sequence cluster, two resolution strata (<=3.0 A and 3.0-4.0 A), at most 700 modeled residues.
- Same AF3 predictions for every method that accepts predicted structures; ModelAngelo on map plus sequence only.
- The paper's six metrics, plus Cα RMSD and residue coverage (the email to Prof. Cheng listed them) and runtime.

**Next actions**
- [ ] Request the Phenix password (needs an institutional email). It is on the critical path for MICA and for the CC_mask filter.
- [ ] Create the GitHub repo and push this folder.
- [ ] Kaggle: verify the account, switch Internet on, run `nproc`, `free -h` and `nvidia-smi`, and paste the output here.
- [ ] AlphaFold Server: sign in and note the daily job limit.
- [ ] Phase 1: read the MICA paper and the baselines' READMEs; work through the checklist in `docs/SPEC.md` section 4.
