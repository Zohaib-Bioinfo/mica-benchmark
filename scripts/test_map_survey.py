#!/usr/bin/env python3
"""Tests for map_survey.py (numpy and scipy only).

The maps are written here, with a header of our own (any axis order, byte order, mode, start indices and origin), from a small helical model; one test lets mrcfile write
a map (in MICA_TEST_ENV_PYTHON) and reads it back, and one compares the atoms of a real mmCIF file (MICA_TEST_AF3_DIR) with Biopython's.

    MICA_TEST_ENV_PYTHON=... MICA_TEST_AF3_DIR=... python -W error -m unittest test_map_survey
"""
from __future__ import annotations

import gzip
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest

import numpy as np

import map_survey as ms

ENV_PY = os.environ.get("MICA_TEST_ENV_PYTHON")
AF3_DIR = os.environ.get("MICA_TEST_AF3_DIR")

CIF_HEAD = """data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def helix(n=40, shift=(0.0, 0.0, 0.0)):
    """C-alpha positions of an alpha helix (radius 2.3 A, rise 1.5 A, 100 degrees per residue), and N, C, O atoms next to each: (ca, other) arrays."""
    t = np.arange(n)
    ca = np.stack([2.3 * np.cos(np.radians(100 * t)), 2.3 * np.sin(np.radians(100 * t)), 1.5 * t], axis=1) + np.array(shift)
    other = [ca + d for d in (np.array([-1.2, 0.4, -0.7]), np.array([1.3, -0.3, 0.6]), np.array([1.8, -0.9, 1.2]))]
    return ca, other


def write_cif(path, copies=((0.0, 0.0, 0.0),), extra_rows=True):
    rows, i = [], 0
    for chain, shift in zip("ABCDEFG", copies):
        ca, (n_, c_, o_) = helix(40, shift)
        for k in range(len(ca)):
            for name, sym, xyz in (("N", "N", n_[k]), ("CA", "C", ca[k]), ("C", "C", c_[k]), ("O", "O", o_[k])):
                i += 1
                rows.append("ATOM %d %s %s . ALA %s 1 %d %.3f %.3f %.3f 1.00 %s 1" % (i, sym, name, chain, k + 1, xyz[0], xyz[1], xyz[2], chain))
    if extra_rows:
        rows.append("ATOM 9001 H H . ALA A 1 1 50.000 50.000 50.000 1.00 A 1")                  # a hydrogen
        rows.append("ATOM 9002 C CA B ALA A 1 1 60.000 60.000 60.000 0.50 A 1")                 # a second location
        rows.append("HETATM 9003 O O . HOH W 2 . 70.000 70.000 70.000 1.00 W 1")                # water
        rows.append("ATOM 9004 P P . DA D 3 1 80.000 80.000 80.000 1.00 D 1")                   # nucleic acid
        rows.append("ATOM 9005 C CA . ALA A 1 1 90.000 90.000 90.000 1.00 A 2")                 # a second model
    with open(path, "w") as f:
        f.write(CIF_HEAD + "\n".join(rows) + "\n#\n")


def density(shape_zyx, voxel, xyz, offset=(0.0, 0.0, 0.0), sigma=1.5):
    """Sum of Gaussians (peak 1) at the points xyz (A, model coordinates); the first voxel sits at `offset`."""
    vol = np.zeros(shape_zyx, dtype=np.float32)
    nz, ny, nx = shape_zyx
    reach = int(math.ceil(4 * sigma / voxel))
    for p in xyz:
        c = (np.asarray(p) - np.asarray(offset)) / voxel
        lo = np.maximum(np.floor(c - reach).astype(int), 0)
        hi = np.minimum(np.ceil(c + reach).astype(int) + 1, [nx, ny, nz])
        if np.any(hi <= lo):
            continue
        gx, gy, gz = (np.arange(lo[a], hi[a]) - c[a] for a in range(3))
        d2 = (gz[:, None, None] ** 2 + gy[None, :, None] ** 2 + gx[None, None, :] ** 2) * voxel ** 2
        vol[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] += np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
    return vol


def write_mrc(path, vol_zyx, voxel=1.0, nstart=(0, 0, 0), origin=(0.0, 0.0, 0.0), mode=2, axes=(1, 2, 3), little=True, nsymbt=0):
    """An MRC/CCP4 file with a header written by hand. vol_zyx is indexed [z][y][x]; axes = (mapc, mapr, maps) says which crystal axis runs fastest, next, slowest."""
    e = "<" if little else ">"
    nz, ny, nx = vol_zyx.shape
    length = {1: nx, 2: ny, 3: nz}
    mapc, mapr, maps = axes
    order = [maps, mapr, mapc]                                         # array axis -> crystal axis
    stored = np.transpose(vol_zyx, [{3: 0, 2: 1, 1: 2}[a] for a in order])
    dtype = {2: "f4", 1: "i2", 0: "i1", 12: "f2", 6: "u2"}[mode]
    stored = np.ascontiguousarray(stored.astype(e + dtype))
    ns, nr, nc = stored.shape
    head = bytearray(1024)
    head[0:40] = struct.pack(e + "10i", nc, nr, ns, mode, nstart[0], nstart[1], nstart[2], nx, ny, nz)
    head[40:64] = struct.pack(e + "6f", nx * voxel, ny * voxel, nz * voxel, 90.0, 90.0, 90.0)
    head[64:76] = struct.pack(e + "3i", mapc, mapr, maps)
    head[92:96] = struct.pack(e + "i", nsymbt)
    head[196:208] = struct.pack(e + "3f", *origin)
    head[208:212] = b"MAP "
    head[212:216] = bytes([0x44, 0x44, 0, 0]) if little else bytes([0x11, 0x11, 0, 0])
    with open(path, "wb") as f:
        f.write(bytes(head))
        f.write(b"\0" * nsymbt)
        f.write(stored.tobytes())


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="msurvey_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)


class ReadMrcTests(Base):
    def test_a_standard_float32_map_round_trips_with_its_header_values(self):
        rng = np.random.default_rng(1)
        vol = rng.random((10, 12, 14), dtype=np.float32)
        write_mrc(self.path("a.map"), vol, voxel=1.25, nstart=(3, -4, 5), origin=(1.5, 2.5, -3.5))
        got, info = ms.read_mrc(self.path("a.map"))
        np.testing.assert_array_equal(got, vol)
        self.assertEqual(info["grid_zyx"], (10, 12, 14))
        np.testing.assert_allclose(info["voxel"], (1.25, 1.25, 1.25))
        self.assertEqual(info["nstart"], (3, -4, 5))
        np.testing.assert_allclose(info["origin"], (1.5, 2.5, -3.5))
        self.assertEqual(info["mode"], 2)

    def test_axis_order_byte_order_and_data_type_do_not_matter(self):
        rng = np.random.default_rng(2)
        vol = rng.integers(-100, 100, size=(6, 8, 10)).astype(np.float32)
        for axes in ((1, 2, 3), (2, 1, 3), (3, 1, 2), (1, 3, 2), (2, 3, 1), (3, 2, 1)):
            for little in (True, False):
                for mode in (2, 1):
                    write_mrc(self.path("b.map"), vol, voxel=0.5, nstart=(1, 2, 3), axes=axes, little=little, mode=mode, nsymbt=80 if axes == (3, 1, 2) else 0)
                    got, info = ms.read_mrc(self.path("b.map"))
                    np.testing.assert_array_equal(got, vol, "axes %s little %s mode %s" % (axes, little, mode))
                    np.testing.assert_allclose(info["voxel"], (0.5, 0.5, 0.5))
                    self.assertEqual(info["grid_zyx"], (6, 8, 10), (axes, little, mode))
        # start indices follow the storage order: with axes (2, 1, 3) the first start belongs to the y axis
        write_mrc(self.path("c.map"), vol, axes=(2, 1, 3), nstart=(11, 22, 33))
        self.assertEqual(ms.read_mrc(self.path("c.map"))[1]["nstart"], (22, 11, 33))

    def test_anisotropic_voxels_come_from_the_cell_and_the_sampling(self):
        vol = np.ones((4, 5, 6), dtype=np.float32)
        write_mrc(self.path("d.map"), vol, voxel=2.0)
        with open(self.path("d.map"), "r+b") as f:                       # cell edges 6 x 20 x 4 A for a 6 x 5 x 4 grid: voxel 1 x 4 x 1
            f.seek(40)
            f.write(struct.pack("<3f", 6.0, 20.0, 4.0))
        np.testing.assert_allclose(ms.read_mrc(self.path("d.map"))[1]["voxel"], (1.0, 4.0, 1.0))

    def test_files_that_are_not_maps_are_refused_with_a_message(self):
        with open(self.path("e.map"), "wb") as f:
            f.write(b"x" * 100)
        with self.assertRaisesRegex(ValueError, "shorter than an MRC header"):
            ms.read_mrc(self.path("e.map"))
        with open(self.path("f.map"), "wb") as f:
            f.write(b"\x07" * 2000)
        with self.assertRaisesRegex(ValueError, "does not look like an MRC"):
            ms.read_mrc(self.path("f.map"))

    def test_gunzip_if_needed(self):
        write_mrc(self.path("g.map"), np.ones((3, 3, 3), dtype=np.float32))
        with open(self.path("g.map"), "rb") as f, gzip.open(self.path("g.map.gz"), "wb") as z:
            z.write(f.read())
        self.assertEqual(ms.gunzip_if_needed(self.path("g.map"), self.path("never")), self.path("g.map"))
        out = ms.gunzip_if_needed(self.path("g.map.gz"), self.path("sub/g_unzipped.map"))
        self.assertEqual(out, self.path("sub/g_unzipped.map"))
        np.testing.assert_array_equal(ms.read_mrc(out)[0], np.ones((3, 3, 3), dtype=np.float32))

    @unittest.skipUnless(ENV_PY, "needs MICA_TEST_ENV_PYTHON (a Python with numpy and mrcfile)")
    def test_a_map_written_by_mrcfile_is_read_the_same(self):
        code = ("import mrcfile, numpy as np, sys; a = np.random.default_rng(3).random((7, 9, 11)).astype('float32'); m = mrcfile.new(sys.argv[1], overwrite=True); m.set_data(a); "
                "m.voxel_size = (1.1, 1.2, 1.3); m.header.nxstart, m.header.nystart, m.header.nzstart = 2, 3, 4; m.header.origin.x = 5.5; m.close(); np.save(sys.argv[2], a)")
        subprocess.run([ENV_PY, "-c", code, self.path("m.map"), self.path("m.npy")], check=True)
        got, info = ms.read_mrc(self.path("m.map"))
        np.testing.assert_array_equal(got, np.load(self.path("m.npy")))
        np.testing.assert_allclose(info["voxel"], (1.1, 1.2, 1.3), rtol=1e-5)
        self.assertEqual(info["nstart"], (2, 3, 4))
        self.assertAlmostEqual(info["origin"][0], 5.5, places=4)


class AtomsTests(Base):
    def test_only_protein_atoms_of_the_first_model_in_the_first_location_are_read(self):
        write_cif(self.path("m.cif"), copies=((0, 0, 0), (30, 0, 0)))
        atoms = ms.read_atoms(self.path("m.cif"))
        self.assertEqual(atoms["ca"].shape, (80, 3))
        self.assertEqual(atoms["heavy"].shape, (320, 3))                 # no hydrogen, no second location, no water, no nucleic acid, no second model
        self.assertEqual(atoms["chains"], {"A": 40, "B": 40})

    def test_a_file_without_protein_atoms_is_refused(self):
        with open(self.path("n.cif"), "w") as f:
            f.write("data_x\n#\nloop_\n_entity.id\n_entity.type\n1 polymer\n#\n")
        with self.assertRaisesRegex(ValueError, "no protein atoms"):
            ms.read_atoms(self.path("n.cif"))

    @unittest.skipUnless(ENV_PY and AF3_DIR and any(f == "fold_26dk_e1_model_0.cif" for _r, _d, fs in os.walk(AF3_DIR or "") for f in fs), "needs the 26DK file of the AlphaFold Server")
    def test_a_real_file_gives_the_atoms_that_biopython_finds(self):
        cif = [os.path.join(r, f) for r, _d, fs in os.walk(AF3_DIR) for f in fs if f == "fold_26dk_e1_model_0.cif"][0]
        out = subprocess.run([ENV_PY, "-c", "import sys; from Bio.PDB import MMCIFParser; s = MMCIFParser(QUIET=True).get_structure('m', sys.argv[1]); "
                              "a = [x for x in s[0].get_atoms() if x.element != 'H']; print(len(a), sum(1 for x in a if x.get_id() == 'CA'))", cif],
                             capture_output=True, text=True, check=True).stdout.split()
        atoms = ms.read_atoms(cif)
        self.assertEqual((len(atoms["heavy"]), len(atoms["ca"])), (int(out[0]), int(out[1])))


class SurveyTests(Base):
    BOX = (120, 80, 80)                                             # z, y, x of the synthetic map; voxel 1 A

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.one = ((20.0, 40.0, 20.0),)
        cls.three = ((15.0, 20.0, 10.0), (40.0, 40.0, 10.0), (65.0, 60.0, 10.0))      # three parallel helices, far enough apart not to touch
        write_cif(cls.path_of("one.cif"), cls.one)
        write_cif(cls.path_of("three.cif"), cls.three[:1])          # the deposited model is one of the three copies
        for name, copies in (("one", cls.one), ("three", cls.three)):
            pts = [p for c in copies for arr in (helix(40, c)[0],) + tuple(helix(40, c)[1]) for p in arr]
            write_mrc(cls.path_of(name + ".map"), density(cls.BOX, 1.0, pts), voxel=1.0)
        cls.contour = 0.25

    @classmethod
    def path_of(cls, name):
        return os.path.join(cls.root, name)

    def test_one_copy_gives_one_copy_and_one_component(self):
        r = ms.survey(self.path("one.map"), self.path("one.cif"), contour=self.contour, resolution=3.5)
        self.assertEqual((r["n_chains"], r["n_ca"], r["n_heavy_atoms"]), (1, 40, 160))
        self.assertGreaterEqual(r["copies_of_model_estimate"], 1.0)
        self.assertLessEqual(r["copies_of_model_estimate"], 1.15)
        self.assertGreater(r["above_fraction_in_model"], 0.88)
        self.assertEqual((r["components_ge_500A3"], r["components_touching_model"]), (1, 1))
        self.assertEqual(r["frame"], "start")
        self.assertEqual(r["contour_source"], "EMDB header file")
        self.assertEqual(r["note"], "")
        self.assertEqual((r["grid_x"], r["grid_y"], r["grid_z"]), (80, 80, 120))
        self.assertEqual((r["box_x_A"], r["box_y_A"], r["box_z_A"]), (80.0, 80.0, 120.0))

    def test_three_copies_are_counted_and_only_one_touches_the_model(self):
        r = ms.survey(self.path("three.map"), self.path("three.cif"), contour=self.contour, resolution=3.5)
        self.assertGreater(r["copies_of_model_estimate"], 2.8)
        self.assertLess(r["copies_of_model_estimate"], 3.5)
        self.assertEqual(r["components_ge_5000A3"], 3)
        self.assertEqual(r["components_touching_model"], 1)
        self.assertAlmostEqual(r["largest_component_fraction"], 1 / 3, delta=0.06)
        self.assertLess(r["above_fraction_in_model"], 0.4)
        self.assertEqual(len(r["components"]), 3)
        self.assertTrue(all(c["volume_A3"] > 5000 for c in r["components"]))
        self.assertEqual(sorted(c["voxels_in_model_region"] > 0 for c in r["components"]), [False, False, True])

    def test_the_content_box_and_the_reflection_counts_follow_from_the_numbers(self):
        r = ms.survey(self.path("one.map"), self.path("one.cif"), contour=self.contour, resolution=3.5)
        self.assertLess(r["content_fraction_of_box"], 0.5)
        self.assertLess(r["reflections_content_M"], r["reflections_full_M"])
        self.assertAlmostEqual(r["reflections_full_M"], 80 * 80 * 120 * (4 * math.pi / 3) / 3.5 ** 3 / 2 / 1e6, places=3)
        self.assertAlmostEqual(r["docking_load_vs_26DK"], r["reflections_full_M"] * 1e6 / 915139, places=2)

    def test_the_formula_gives_the_count_that_phaser_printed_for_26dk(self):
        self.assertAlmostEqual(ms.reflections((265.6, 265.6, 265.6), 3.5), 915139, delta=500)

    def test_start_indices_and_the_origin_field_are_found_by_the_density_at_the_c_alpha_atoms(self):
        pts = [p for arr in (helix(40, self.one[0])[0],) + tuple(helix(40, self.one[0])[1]) for p in arr]
        # the first voxel lies at (3, 10, 4) A or (2.5, 12, 5) A or (2.5, 10, 5) A: the helix at (20, 40, 20) A stays inside the 80 x 80 x 120 A box in all three
        for label, nstart, origin in (("start", (3, 10, 4), (0.0, 0.0, 0.0)), ("origin", (0, 0, 0), (2.5, 12.0, 5.0)), ("start+origin", (1, 2, 3), (1.5, 8.0, 2.0))):
            offset = (np.array(nstart, float) + np.array(origin)).tolist()
            vol = density(self.BOX, 1.0, pts, offset=offset)
            write_mrc(self.path("f_%s.map" % label), vol, nstart=nstart, origin=origin)
            r = ms.survey(self.path("f_%s.map" % label), self.path("one.cif"), contour=self.contour, resolution=3.5)
            self.assertEqual(r["frame"], label)
            self.assertLessEqual(r["copies_of_model_estimate"], 1.15, label)
            self.assertGreater(r["ca_in_map_frac"], 0.99, label)
            self.assertEqual(r["note"], "", label)

    def test_a_model_that_does_not_fit_the_map_is_reported_in_the_note(self):
        write_cif(self.path("far.cif"), ((500.0, 500.0, 500.0),))
        r = ms.survey(self.path("one.map"), self.path("far.cif"), contour=self.contour, resolution=3.5)
        self.assertIn("lie inside the map", r["note"])
        self.assertEqual(r["copies_of_model_estimate"], "")

    def test_without_a_contour_level_mean_plus_three_sd_is_used_and_said(self):
        r = ms.survey(self.path("one.map"), self.path("one.cif"), contour=None, resolution=None)
        self.assertIn("mean + 3 std", r["contour_source"])
        self.assertIn("contour level taken as mean + 3 std", r["note"])
        self.assertNotIn("reflections_full_M", r)

    def test_a_map_without_density_above_the_contour_gives_empty_numbers_not_an_error(self):
        r = ms.survey(self.path("one.map"), self.path("one.cif"), contour=1e9, resolution=3.5)
        self.assertEqual(r["above_volume_A3"], 0)
        self.assertEqual(r["copies_of_model_estimate"], "")
        self.assertEqual(r["components"], [])

    def test_survey_columns_cover_every_key_the_survey_returns(self):
        r = ms.survey(self.path("three.map"), self.path("three.cif"), contour=self.contour, resolution=3.5)
        r.pop("components")
        self.assertEqual([k for k in r if k not in ms.SURVEY_COLUMNS], [])


if __name__ == "__main__":
    unittest.main()
