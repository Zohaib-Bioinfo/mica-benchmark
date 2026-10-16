# Phase 2 search log: newer methods

Lookups made on 2026-10-10 and 2026-10-11. This file records how the list of newer methods in [methods_table.csv](methods_table.csv) was built, so the search can be repeated and checked.

## Question

Which deep-learning methods published after ModelAngelo and EModelX build protein atomic models from a cryo-EM map plus a sequence, have public code and weights, and can run on a free GPU? At most three are benchmarked (decision 1 in `docs/SPEC.md`).

## Screen

Decision 2 in `docs/SPEC.md`. A method is kept only if all of these hold:

1. Published after the two baselines, or a new major release of one of them.
2. Deep learning.
3. Builds protein atomic models from a map plus a sequence.
4. Public code and weights.
5. Runs on a free 16 GB GPU.
6. Documented defaults, so no per-map tuning.

Every exclusion is logged with its reason in `methods_table.csv`. A cell holds `n/r` (not recorded) when the method failed an earlier item and the field was not read. The `included` column takes `baseline`, `yes`, `reserve` or `no`.

Tie-break, added 2026-10-11 (no locked value changes): if more than three methods pass the screen, peer-reviewed publications come before preprints, and then the earlier publication date comes first. Methods are never ranked by benchmark results. Reserves follow the same order, with methods that pass on paper ahead of conditional ones.

## How the search ran

- Collected names from reviews and benchmark papers on cryo-EM model building (listed under "Reviews and benchmarks seen").
- Searched the web with queries built from each name plus "cryo-EM" and "model building"; with queries of the form "compared with ModelAngelo", "compared with EModelX" and "compared with CryoAtom"; and with searches restricted to biorxiv.org, arxiv.org, nature.com and github.com for 2025 and 2026 releases. The exact query strings were not saved.
- For every name that looked like a method, opened the paper page or preprint and the GitHub README. Year, venue, inputs, weights, GPU need and licence in the table come from those pages, read on the dates above.
- Compared each README's GPU figure with a third-party measurement where one exists: Appendix H of the CryoACE paper (arXiv 2606.31332) times several methods on one H20 GPU for a ~500-residue protein.
- Checked the ModelAngelo and EModelX repositories for a new major release (screen item 1). Neither lists a release or tag, so that route adds nothing.
- On 2026-10-11 the forward-citation gap was covered by hand in Google Scholar: "Cited by" lists restricted to 2026 for five papers. Titles and links were copied into [scholar_cited_by_2026-10-11.txt](scholar_cited_by_2026-10-11.txt) (raw, unedited). Counts of 2026 results: MICA 9, CryoAtom 34, EMProt 9, E3-CryoFold 9, CryoAtom's earlier preprint (CryFold) 2. All nine MICA results were copied. For the other lists only titles that looked like model-building methods, reviews or benchmarks were copied (9 of 34, 6 of 9, 7 of 9 and 2 of 2).
- The EM3DFold preprint page could not be opened from the workspace, so its "Code availability" statement was copied by hand on 2026-10-11: "The EM3DFold package is freely available at https://github.com/huang-laboratory/EM3DFold." The repository README was then read.

## Result

| Method | Newer | Deep learning | Map + sequence to atomic model | Public code and weights | Fits 16 GB | Defaults, no per-map tuning | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CryoAtom | yes (online 2025-11-14) | yes | yes | yes (MIT per README) | on paper (README 14 GB; third party ~12 GB) | yes, one documented command | **Include** |
| EMProt | yes (online 2025-12-08) | yes | yes; also takes per-chain predicted models | yes (licence wording differs) | on paper (README >10 GB) | yes | **Include** |
| E3-CryoFold | yes (2025-06-24) | yes | yes | yes (MIT per README) | on paper (third party ~8 GB) | to fix from the README example | **Include** |
| EM3DFold | yes (Sep 2026 preprint, "in submission") | yes | yes (protein, RNA, DNA) | yes (download script; no licence stated) | on paper (README 12 GB) | yes, one documented command | Reserve 1 (preprint) |
| CryoZeta | yes (2026 preprint) | yes | yes, with outside MSAs | yes (weights academic only) | no on paper (README 32 GB or more) | to check | Reserve 2 |
| CryoACE | yes (2026) | yes | no: needs an aligned starting model | yes | no (~38 GB) | no: resolution and threshold per map | Exclude |
| CryoBoltz | yes (2025) | yes | no: needs a pre-aligned model | yes | no (~32 GB, third party) | no | Exclude |

EM3DFold passes the screen on paper but is a preprint, so under the tie-break the three peer-reviewed methods keep their places. It is also the only one of the group whose README states no predicted-structure input, while EMProt takes per-chain AlphaFold models, which matters for the shared AF3 inputs (decision 6).

The 16 GB column rests on the READMEs plus one third-party table: in Appendix H of the CryoACE paper, a ~500-residue system on one H20 used about 8 GB (E3-CryoFold), 10 GB (ModelAngelo) and 12 GB (CryoAtom). These are another group's numbers for one protein size, not measurements from this project. The Phase 5 pilot measures them on the free GPUs.

## Screened from the Scholar lists (2026-10-11)

Every title in the notes file has a decision below or is already a row in `methods_table.csv`.

| Paper | What it is | Decision | Reason |
| --- | --- | --- | --- |
| EM3DFold (bioRxiv, Sep 2026) | Protein, RNA and DNA model building from maps with a language-model-powered network; appears in the CryoAtom, EMProt and E3-CryoFold lists | Reserve 1 | Passes the screen on paper once the code statement was checked by hand; a preprint, so the tie-break keeps the three peer-reviewed methods. |
| CryoForge (bioRxiv, posted 2026-08-20) | A correction layer that repairs a model made by another builder, with an evidence gate | Exclude | Not a builder from map + sequence; no code or weights stated. |
| CryoNet.Discovery (LangTaoSha preprint, 2026-07-27) | Matches a map to known structures by embedding retrieval; compared with ModelAngelo, cryoID and DomainFit | Exclude | Does not build models; no code or weights stated; not peer reviewed. |
| "Fine-tuning AlphaFold with limited cryo-EM observations" = CoCoFold (Commun. Chem. 9, 95, 2026-01-19) | Fine-tunes AlphaFold on raw particle images; compared with MICA, ModelAngelo, EModelX, DeepMainmast and DiffModeler | Exclude | The input is raw particle images, not a map; peak GPU memory 19-77 GB on an A100 80 GB. |
| CryoAtom2 (LangTaoSha preprint, 2026-04-14) | Newer model from the CryoAtom group for proteins and nucleic acids; the repository README now describes CryoAtom2 v2.1.0 | Not a separate method | A version question for CryoAtom; see "Notes for later phases". |
| CryoACE, CryoZeta | Already in the table | No change | |
| CryoCodex, EMAlign, HDOCK-Multimer, the ligand-modelling paper, TomoSwin3D | Map post-processing, map alignment, docking, ligand modelling, cryo-ET identification | Exclude | Screened by title only: none builds protein models from a map and a sequence. |
| DeepSRFusion, the XL-MS Encephalitozoon paper, the "AI-Physics-Experiment Trinity" paper | Super-resolution particle fusion, structure prediction with cross-linking data, protein dynamics modelling | Exclude | Screened by title only: no cryo-EM map-to-model builder. |
| Gap junction connexins, NAGPA, mycobacterial MCC, oocyte cytoplasmic lattice assembly, and a review of amphipathic environments for membrane-protein cryo-EM | Structure papers and a sample-preparation review that cite one of the four | Exclude | Not methods. |

## Reviews and benchmarks seen

Read, in part:

- "Connecting the dots: deep learning-based automated model building methods in cryo-EM" (bioRxiv 10.64898/2025.12.28.696775, Dec 2025). The table of methods was read; its reference list was not reachable, and the table gives no years or code links. It adds three names with no paper or code found (Cryo2Struct2, SEGEM++, Cryformer) and otherwise lists methods that are already in the methods table or earlier. Names in it that were not opened: DeepCryoRNA, EMRNA, EM2NA, DEMO-EM, DEMO-EM2, DEMO-EMfit, DEMO-EMol, CR-I-TASSER, CrAI, CryoJAM, CryoDomain, FFF. Revisit one only if a reader gives a reason.
- "A comprehensive survey and benchmark of deep learning-based methods for atomic model building from cryo-EM density maps" (bioRxiv 2025.03.02.641021). Benchmarks Phenix, ModelAngelo, EMBuild and DeepMainmast on 50 protein maps (1.8 to 7.8 Å). It predates the newer methods, so it is background for the report.
- "Advancing structure modeling from cryo-EM maps with deep learning" (Li, Terashi, Zhang, Kihara; Biochem. Soc. Trans. 53, 259-265, 2025). Covers methods up to 2024; no 2025 or later method.

Not opened (fetch refused or failed): "Deep learning-based postprocessing and model building for cryo-electron microscopy maps" (Curr. Opin. Struct. Biol.; it is in four of the five Scholar lists), "AI tools for cryo-EM: protein particle picking, density map enhancement, and atomic model building" (PubMed 42463260), the amyloid-fibril methods review, the University of Missouri thesis "Artificial intelligence for the identification and structure determination of macromolecules in cryo-electron microscopy and tomography", and the Bioinformatics benchmark below.

Related benchmarks to cite in the report where they are used:

- Benchmarking deep learning methods for Cα atom prediction in cryo-EM density maps. Bioinformatics 42(6), btag350.
- The CryoZeta benchmark set (Zenodo record 18603136).
- The hardware table in Appendix H of the CryoACE paper.

## Corrections made during the search

- E3-CryoFold was first excluded on an inferred link to EModelX, because its README and the SBGrid page both cite the EModelX paper. Its own paper (Nat. Mach. Intell. 7, 1091-1103, published 2025-06-24) shows it is a separate, newer method. An author correction of 2025-07-11 changes corresponding-author labels and adds one reference; it does not touch results, data or code. The method was then screened properly and included.
- EM3DFold was first logged as having no public code, because its preprint page could not be opened. The "Code availability" statement, copied by hand, names a GitHub repository, so it was re-screened and is now Reserve 1.
- CryoACE's ~38 GB figure comes from its paper (Appendix H). Its README states no GPU need.
- Cryformer first looked like a separate name with no paper. It matches the encoder module of the CryFold/CryoAtom network, although the Dec 2025 review lists it on its own.

## What could not be checked

- **Forward citations, in part.** The Scholar check covers five papers and 2026 only. The titles that were not copied (25 of 34 for CryoAtom, 3 of 9 for EMProt, 2 of 9 for E3-CryoFold) were judged by eye during the check and have not been re-read. Scholar records under other titles were not opened: MICA's earlier bioRxiv version appears under the title "Atomic Protein Structure Modeling from Cryo-EM Using Multi-Modal Deep Learning and AlphaFold3" on a third-party page (not confirmed to be the same preprint), and preprint versions of EMProt or E3-CryoFold under other titles, if any. Scholar's index also lags for the newest preprints.
- The EM3DFold preprint text was not read, only its code statement and the repository README. Its licence, runtime, journal status and training-data cutoff are unknown.
- Semantic Scholar, OpenAlex, Europe PMC, PMC and PubMed could not be reached from the workspace.
- The E3-CryoFold full text is paywalled; only the abstract, the availability statements and the third-party table were read.
- Nothing was installed or run. GPU, disk and licence facts come from READMEs and papers.

## Notes for later phases

- **Which CryoAtom.** The NSMB paper describes CryoAtom (code named CryFold v1.0, to confirm). The repository README now describes CryoAtom2 v2.1.0, and a 2026-04-14 preprint (not peer reviewed) describes CryoAtom2 as a different, inverse-folding-inspired model for proteins and nucleic acids. Decide in Phase 4 which tag or commit to run, state it in the report, and cite both papers. Proposed default (changed 2026-10-11 from "the repository's latest tag", for the reason under "Training cutoffs"): the release that matches the NSMB paper, unless the CryoAtom2 training cutoff is shown to be before 2026-01-01.
- **Versions.** Pin every method's version or commit on 2026-10-18 in `methods_table.csv`.
- **Training cutoffs.** Risk 2 in `docs/SPEC.md` covers homologues of 2026 proteins in a method's training data. A second risk is a method trained after 2026-01-01, which could have seen the test entries themselves. The three selected papers appeared online between June and December 2025, before the test window, but a repository can ship weights retrained since (CryoAtom's README now describes CryoAtom2 v2.1.0, and its preprint is dated 2026-04-14), and EM3DFold is a September 2026 preprint. For every method that is run, record the date of the weights file actually downloaded and what its paper or README says about the training set, and report both. If a release cannot be shown to predate 2026-01-01, keep it out of the headline comparison and report it separately.
- **GPU headroom.** CryoAtom's README asks for 14 GB, which is tight on a 15 GB T4. The 700-residue cap in decision 4 helps; a 16 GB P100, or two T4s with `-d`, are the fallbacks.
- **AlphaFold inputs.** EMProt takes one single-chain PDB or mmCIF per chain with `--chain`, so the AF3 `model_0.cif` files can go in directly. Run it this way as the primary mode, and without `--chain` as the secondary mode on a subset (decision 6). Check how it pairs each file with a FASTA entry. CryoAtom, E3-CryoFold and EM3DFold take map and sequence only, like ModelAngelo.
- **E3-CryoFold.** Take the protocol and `--t` from the README example for map plus sequence and write them into the protocol before the pilot. No per-map changes. The paper's code statement names A4Bio/CryoFold; check that it redirects to A4Bio/E3-CryoFold.
- **EMProt weights** sit on the authors' own server. Test the download from Kaggle early. EM3DFold's weights come from a download script; check it too before it is needed.
- **Reserves.** If one of the three selected methods cannot be installed or run, EM3DFold replaces it, then CryoZeta (after a pilot). Before EM3DFold is run, read its training cutoff (see "Training cutoffs"). Adding EM3DFold as a fourth method would change decision 1 of the spec (up to three newer methods), so it needs a log entry with a reason first.
- **Licences.** Read the LICENSE files: EMProt (README says MIT; the paper says academic or noncommercial), EModelX, CryoAtom, E3-CryoFold. EM3DFold states none. This benchmark is academic use.

## To do before 2026-10-16

1. Skim the model-building section of the Current Opinion in Structural Biology review for method names that are not in the table.
2. Optional: read the 25 uncopied CryoAtom titles once more.
3. For each selected method, note what its paper or README says about the training data and which weights file will be downloaded (see "Training cutoffs").
4. 2026-10-15: recheck EM3DFold for a release tag, a licence and journal status.
5. Read the LICENSE files named above.
6. 2026-10-18: pin versions.
