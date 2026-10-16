Frozen candidate pool of the MICA 2026 benchmark (spec decision 3, pool frozen 2026-10-12).
Made by scripts/pull_pool.py version 1.1 (sha256 1b466057b679ba5ccbc57299f863116e879dc72daec2c0e4faa0fe3c56b1f6a7), run in Google Colab
2026-10-12 11:26-11:39 UTC, without server-side prefilters: all 6,116 cryo-EM entries released 2026-01-01 to 2026-10-12 were downloaded
and judged on their own mmCIF file. Exit code 0, 0 download errors, canary OK. 484 candidates: 180 at up to 3.0 A, 304 above 3.0 to 4.0 A.
The pool is defined by release date, so a run on a later day with --freeze 2026-10-12 gives the same entries.
Run 1 (with prefilters) is kept in ../pool_run1_prefilter/ as an audit trail only; its 453 candidates are a subset of these 484.

File hashes (sha256), as received:
  f02bdd01ed5fe50855e7a2d1eb781b97eda2e9333ca7ec552462095cf1e13575  pool_all_entries.csv
  4bba9e1e1ab497a7ffdf6713fff6aed697245730f76a1637b19d3acf050737c3  pool_candidates.csv
  c914ce06feb39e07f3e60cdedc6a422d586c378097c2b938cb306494b7819795  pool_candidate_sequences.fasta
  c00bae1fbb4e12f6394cef35f63854f1f8dbaf1bf165e5ae6114663871754771  pool_flow.csv
  dca391ef5d0db171e4ded15d00a73472dc0b24e48d986f5f6dd639698aabc86c  pool_ids.txt
  dca391ef5d0db171e4ded15d00a73472dc0b24e48d986f5f6dd639698aabc86c  pool_ids_all_cryoem.txt
  ade89fa224c21850fb40f52b764ae59519c289694036b1042d65de9c3b6fde2f  search_weeks.csv
  a4763e2bc3036d59a405f52d7fa8cf9b85c89d54198ade7cf42be3ffc8679067  run_info.json
  b5034cc2041b5ecf88f1cdb50707fb3927ac80199187ceac68a8bc08b2181caf  run.log

Columns of pool_candidates.csv are described at the top of scripts/pull_pool.py; the sequence file has one record per modeled
protein entity (header: >ID_E<entity> len= copies= longest=0/1).
