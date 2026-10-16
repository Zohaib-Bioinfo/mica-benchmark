"""Tests for check_af3_results.py. Run from the scripts folder:  python -W error -m unittest test_check_af3_results"""
import contextlib
import csv
import hashlib
import io
import os
import shutil
import tempfile
import unittest

import check_af3_results as cr

THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
         "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}
HEADER = ["data_test", "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol", "_atom_site.label_atom_id", "_atom_site.label_alt_id",
          "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
          "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
          "_atom_site.auth_asym_id", "_atom_site.pdbx_PDB_model_num"]
JOB_COLUMNS = ["job_no", "job_name", "pdb_id", "entity", "stratum", "seed_rank", "length", "copies_in_entry", "sequence_submitted", "substitution"]
JOBS = [("9AAA_E1", "9AAA", "E1", "MKVLAGGW"), ("9AAA_E2", "9AAA", "E2", "GYKPLGLAY"), ("9BBB_E1", "9BBB", "E1", "ACDEFGHIK")]


def mmcif(seq, chains=("A",), plddt=90.0, bfactors=None, water=False, model_count=1):
    """A small mmCIF file: N, CA and C atoms of every residue, one chain per letter in `chains`."""
    lines = list(HEADER)
    n = 0
    for m in range(1, model_count + 1):
        for ci, ch in enumerate(chains):
            for i, aa in enumerate(seq, 1):
                for atom, el, dx in (("N", "N", 0.0), ("CA", "C", 1.4), ("C", "C", 2.5)):
                    n += 1
                    b = plddt if bfactors is None else bfactors[i - 1]
                    lines.append("ATOM %d %s %s . %s %s %d %d %.3f %.3f %.3f 1.00 %.2f %s %d" % (n, el, atom, THREE[aa], ch, ci + 1, i, dx + 3.8 * i, 10.0 * ci, 0.0, b, ch, m))
    if water:
        lines.append("HETATM %d O O . HOH A 9 99 5.000 5.000 5.000 1.00 50.00 A 1" % (n + 1))
    return "\n".join(lines) + "\n"


class CheckAf3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="af3chk_")
        self.addCleanup(shutil.rmtree, self.tmp)
        self.models = os.path.join(self.tmp, "models")
        os.makedirs(self.models)
        self.jobs_csv = os.path.join(self.tmp, "jobs.csv")
        with open(self.jobs_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(JOB_COLUMNS)
            for i, (name, pdb, ent, seq) in enumerate(JOBS, 1):
                w.writerow([i, name, pdb, ent, "le3.0", i, len(seq), 1, seq, ""])
        self.out = os.path.join(self.tmp, "out", "check.csv")
        self.manifest = os.path.join(self.tmp, "out", "sha.txt")

    def put(self, job_name, text, sub="", file_name=None):
        path = os.path.join(self.models, sub, file_name or "fold_%s_model_0.cif" % job_name.lower())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def put_all(self):
        for name, _pdb, _ent, seq in JOBS:
            self.put(name, mmcif(seq))

    def run_main(self, models=None):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = cr.main(["--jobs", self.jobs_csv, "--models", models or self.models, "--out", self.out, "--manifest", self.manifest])
        rows = []
        if os.path.exists(self.out):
            with open(self.out, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        return code, rows, buf.getvalue(), err.getvalue()

    def test_all_ok_writes_table_and_manifest(self):
        self.put_all()
        code, rows, out, _ = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertEqual([r["status"] for r in rows], ["ok"] * 3)
        self.assertEqual([r["job_name"] for r in rows], [j[0] for j in JOBS])
        self.assertEqual([r["residues"] for r in rows], ["8", "9", "9"])
        self.assertEqual(rows[0]["atoms"], "24")
        self.assertEqual(rows[0]["mean_plddt_ca"], "90.00")
        self.assertEqual(rows[0]["frac_ca_plddt_ge70"], "1.000")
        self.assertEqual(rows[0]["frac_ca_plddt_lt50"], "0.000")
        self.assertIn("3 jobs, 3 ok, 0 problems", out)
        with open(self.manifest, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 3)
        with open(os.path.join(self.models, "fold_9aaa_e1_model_0.cif"), "rb") as f:
            self.assertEqual(lines[0], "%s  fold_9aaa_e1_model_0.cif" % hashlib.sha256(f.read()).hexdigest())

    def test_plddt_statistics(self):
        self.put_all()
        self.put("9AAA_E1", mmcif("MKVLAGGW", bfactors=[95, 60, 40, 80, 20, 70, 50, 11]))
        code, rows, out, _ = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertEqual(rows[0]["mean_plddt_ca"], "53.25")          # 426 / 8
        self.assertEqual(rows[0]["frac_ca_plddt_ge70"], "0.375")     # 95, 80, 70
        self.assertEqual(rows[0]["frac_ca_plddt_lt50"], "0.375")     # 40, 20, 11

    def test_missing_file(self):
        self.put_all()
        os.remove(os.path.join(self.models, "fold_9aaa_e2_model_0.cif"))
        code, rows, out, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual([r["status"] for r in rows], ["ok", "missing", "ok"])
        self.assertIn("PROBLEM: 9AAA_E2: missing", out)
        with open(self.manifest, encoding="utf-8") as f:
            self.assertEqual(len(f.read().splitlines()), 2)

    def test_sequence_mismatch_names_the_position(self):
        self.put_all()
        self.put("9AAA_E1", mmcif("MKVLAGGY"))
        code, rows, out, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]["status"], "sequence_mismatch")
        self.assertIn("position 8", rows[0]["note"])
        self.assertEqual(rows[1]["status"], "ok")

    def test_shorter_model_is_a_mismatch(self):
        self.put_all()
        self.put("9BBB_E1", mmcif("ACDEFGHI"))
        code, rows, _, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual(rows[2]["status"], "sequence_mismatch")
        self.assertIn("8 residues, 9 submitted", rows[2]["note"])

    def test_two_chains_are_refused(self):
        self.put_all()
        self.put("9AAA_E1", mmcif("MKVLAGGW", chains=("A", "B")))
        code, rows, _, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]["status"], "not_one_chain")
        self.assertEqual(rows[0]["chains"], "2")

    def test_two_models_are_refused(self):
        self.put_all()
        self.put("9AAA_E1", mmcif("MKVLAGGW", model_count=2))
        code, rows, _, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]["status"], "not_one_model")

    def test_unreadable_file(self):
        self.put_all()
        self.put("9AAA_E1", "this is not a structure file\n")
        code, rows, _, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]["status"], "unreadable")
        self.assertEqual(rows[1]["status"], "ok")

    def test_pldd_outside_zero_to_hundred(self):
        self.put_all()
        self.put("9BBB_E1", mmcif("ACDEFGHIK", plddt=150.0))
        code, rows, _, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual(rows[2]["status"], "bad_numbers")
        self.assertIn("outside 0 to 100", rows[2]["note"])

    def test_residue_that_is_not_an_amino_acid(self):
        self.put_all()
        self.put("9AAA_E1", mmcif("MKVLAGGW", water=True))
        code, rows, _, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]["status"], "not_protein_only")
        self.assertIn("HOH", rows[0]["note"])

    def test_a_file_that_belongs_to_no_job_is_reported(self):
        self.put_all()
        self.put("9ZZZ_E1", mmcif("MKVL"))
        code, rows, out, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual([r["status"] for r in rows], ["ok"] * 3)
        self.assertIn("fold_9zzz_e1_model_0.cif: a file that belongs to no job", out)

    def test_a_file_name_twice_is_refused_for_that_job(self):
        self.put_all()
        self.put("9AAA_E1", mmcif("MKVLAGGW"), sub="second_copy")
        code, rows, out, _ = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]["status"], "file_twice")
        self.assertEqual(rows[1]["status"], "ok")
        self.assertNotIn("belongs to no job", out)

    def test_other_models_of_the_same_job_are_not_matched(self):
        self.put_all()
        self.put("9AAA_E1", mmcif("MKVLAGGW"), file_name="fold_9aaa_e1_model_1.cif")
        code, rows, out, _ = self.run_main()
        self.assertEqual(code, 2)                                    # model_1 is a file that belongs to no job
        self.assertEqual([r["status"] for r in rows], ["ok"] * 3)
        self.assertIn("fold_9aaa_e1_model_1.cif", out)

    def test_names_are_matched_without_regard_to_case_and_in_subfolders(self):
        for name, _pdb, _ent, seq in JOBS:
            self.put(name, mmcif(seq), sub="AlphaFold ServerJOBS", file_name="FOLD_%s_MODEL_0.CIF" % name)
        code, rows, out, _ = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertEqual(rows[0]["file"], "fold_9aaa_e1_model_0.cif")

    def test_missing_folder_and_missing_jobs_file_give_exit_2(self):
        code, _, _, err = self.run_main(models=os.path.join(self.tmp, "nowhere"))
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)
        os.remove(self.jobs_csv)
        code, _, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_a_job_name_that_occurs_twice_is_refused(self):
        with open(self.jobs_csv, "a", newline="", encoding="utf-8") as f:
            f.write("4,9aaa_e1,9AAA,E1,le3.0,1,8,1,MKVLAGGW,\n")
        code, _, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("occurs twice", err)

    def test_the_real_job_table_gives_the_file_names_that_the_server_used(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "af3", "af3_jobs.csv")
        jobs = cr.read_jobs(path)
        self.assertEqual(len(jobs), 26)
        self.assertEqual(cr.file_key(jobs[0]["job_name"]), "fold_9t5k_e1_model_0.cif")
        self.assertEqual(len({cr.file_key(j["job_name"]) for j in jobs}), 26)


if __name__ == "__main__":
    unittest.main()
