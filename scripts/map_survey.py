#!/usr/bin/env python3
"""What does a map hold, compared with the deposited model? (numpy and scipy only; no mrcfile, no Biopython, no Phenix)

Why it exists. In the first pilot, entry 26DK (one deposited chain of 191 residues) turned out to sit in a map that holds about fifteen copies of it, because the
deposited model is the asymmetric unit of a symmetric assembly. Every method then sees much more density than the one chain that the PDB model describes, the
Phenix docking of MICA works on a 265 A cube, and a method that builds every copy is scored against one chain. Before the full runs we need to know, for every
entry: how big the map is, how many copies of the deposited model it holds, and how many reflections Phaser would have to handle.

What it does.
  read_mrc(path)            the map as float32 [z][y][x] with grid, voxel size, start indices and origin (CCP4/MRC2014 header, any axis order, any byte order)
  read_atoms(cif)           protein atoms of the first model of an mmCIF file (an own small parser, standard residues only)
  choose_frame(...)         the offset of the map in the coordinates of the model: start indices, ORIGIN field or both, whichever puts the C-alpha atoms into
                            the highest density (a map and a model of one entry always agree in one of the three)
  survey(map, cif, ...)     the numbers per entry: volume above the contour level, the part of it within `radius` A of a model atom, the copies of the model that
                            the volume suggests, connected components, extent of the content, reflections that Phaser would get at the map resolution

    python scripts/map_survey.py <map or map.gz> <model.cif> [--contour X] [--resolution Y]       # one entry, prints the numbers
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shlex
import shutil
import struct
import sys
import time

import numpy as np

MODE_DTYPES = {0: "i1", 1: "i2", 2: "f4", 6: "u2", 12: "f2"}
AMINO_ACIDS = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "UNK",
               "SEC", "PYL"}
REFERENCE_REFLECTIONS = 915139          # Phaser's count for entry 26DK in the first docking test (265.6 A cube, 3.5 A)

SURVEY_COLUMNS = [
    "entry", "emdb_id", "resolution_table_A", "resolution_header_A", "contour_level", "contour_source", "grid_x", "grid_y", "grid_z", "voxel_A", "box_x_A", "box_y_A", "box_z_A",
    "map_mode", "map_min", "map_max", "map_mean", "map_std", "map_file_mb", "frame", "frame_scores", "ca_in_map_frac", "ca_mean_density_over_contour",
    "n_chains", "n_ca", "n_heavy_atoms", "model_extent_A", "above_volume_A3", "above_in_model_A3", "above_fraction_in_model", "copies_of_model_estimate",
    "components_ge_500A3", "components_ge_5000A3", "components_touching_model", "largest_component_A3", "largest_component_fraction", "content_x_A", "content_y_A",
    "content_z_A", "content_fraction_of_box", "reflections_full_M", "reflections_content_M", "docking_load_vs_26DK", "point_group", "number_of_copies_header",
    "oligomeric_state_header", "sample_name", "assembly_count", "assembly1_polymer_instances", "assembly1_oligomeric_count", "symmetry_rcsb", "map_sha256", "seconds", "note"]


# --------------------------------------------------------------------------------------------------------------------------------- the map
def read_mrc(path):
    """(volume, info). volume: float32 array indexed [z][y][x] (crystal axes), whatever the axis order of the file. info: grid_zyx, voxel (x, y, z in A),
    nstart (x, y, z, in voxels), origin (x, y, z, in A), cell (A), mode, axes (mapc, mapr, maps), little_endian, data_offset."""
    with open(path, "rb") as f:
        head = f.read(1024)
    if len(head) < 1024:
        raise ValueError("%s is shorter than an MRC header (%d bytes)" % (path, len(head)))
    parsed = None
    for little in (head[212] == 0x44, head[212] != 0x44):      # the machine stamp (0x44 little, 0x11 big endian) first, the other order as a second guess
        e = "<" if little else ">"
        nx, ny, nz, mode, nxs, nys, nzs, mx, my, mz = struct.unpack(e + "10i", head[0:40])
        if mode in MODE_DTYPES and 0 < nx < 20000 and 0 < ny < 20000 and 0 < nz < 20000:
            parsed = (e, nx, ny, nz, mode, nxs, nys, nzs, mx, my, mz)
            break
    if parsed is None:
        raise ValueError("%s: the header does not look like an MRC/CCP4 map (mode %s, grid %s x %s x %s)" % (path, mode, nx, ny, nz))
    e, nx, ny, nz, mode, nxs, nys, nzs, mx, my, mz = parsed
    cell = struct.unpack(e + "3f", head[40:52])
    mapc, mapr, maps = struct.unpack(e + "3i", head[64:76])
    if sorted((mapc, mapr, maps)) != [1, 2, 3]:
        mapc, mapr, maps = 1, 2, 3                                              # a header without axis order: the standard one
    nsymbt = struct.unpack(e + "i", head[92:96])[0]
    origin = struct.unpack(e + "3f", head[196:208])
    offset = 1024 + max(nsymbt, 0)
    raw = np.memmap(path, dtype=np.dtype(e + MODE_DTYPES[mode]), mode="r", offset=offset, shape=(nz, ny, nx))      # sections, rows, columns
    crystal_of_axis = [maps, mapr, mapc]
    perm = [crystal_of_axis.index(3), crystal_of_axis.index(2), crystal_of_axis.index(1)]
    vol = np.array(np.transpose(raw, perm), dtype=np.float32)
    del raw
    n_by_crystal = {mapc: nx, mapr: ny, maps: nz}
    m_by_crystal = {1: mx, 2: my, 3: mz}                                         # MX, MY, MZ and the cell edges are by crystal axis, NX, NY, NZ and the starts by storage order
    start_by_crystal = {mapc: nxs, mapr: nys, maps: nzs}
    voxel = []
    for k in (1, 2, 3):
        grid = m_by_crystal[k] if m_by_crystal[k] > 0 else n_by_crystal[k]
        voxel.append(cell[k - 1] / grid if cell[k - 1] > 0 else 1.0)
    info = {"grid_zyx": tuple(int(v) for v in vol.shape), "voxel": tuple(float(v) for v in voxel), "nstart": tuple(int(start_by_crystal[k]) for k in (1, 2, 3)),
            "origin": tuple(float(v) for v in origin), "cell": tuple(float(v) for v in cell), "mode": int(mode), "axes": (mapc, mapr, maps), "little_endian": e == "<",
            "data_offset": offset}
    return vol, info


def gunzip_if_needed(path, dest):
    """`path` itself when it is a plain file; otherwise the unzipped copy `dest` (made once)."""
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic != b"\x1f\x8b":
        return path
    if not (os.path.exists(dest) and os.path.getsize(dest) > 1024):
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        with gzip.open(path, "rb") as src, open(dest + ".part", "wb") as out:
            shutil.copyfileobj(src, out, 1 << 22)
        os.replace(dest + ".part", dest)
    return dest


# ------------------------------------------------------------------------------------------------------------------------------- the model
def read_atoms(cif_path):
    """Protein atoms of the first model of an mmCIF file: {"ca": (n, 3) array, "heavy": (m, 3) array, "chains": {chain id: residues with a C-alpha atom}}.
    Alternative locations other than the first are skipped, hydrogens too. Only the 20 standard amino acids (and MSE, SEC, PYL, UNK) count."""
    cols, ca, heavy, chains = [], [], [], {}
    state, first_model = 0, None
    with open(cif_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if state == 0:
                if line.startswith("loop_"):
                    cols, state = [], 1
                continue
            if state == 1:                                                            # the column names of a loop
                if line.startswith("_atom_site."):
                    cols.append(line.split()[0][len("_atom_site."):])
                    continue
                if line.startswith("_") or line.startswith("loop_") or not cols:
                    cols, state = [], (1 if line.startswith("loop_") else 0)
                    continue
                state = 2                                                             # first data line of an atom_site loop
            if state == 2:
                if not line.strip() or line.startswith(("#", "loop_", "_", "data_")):
                    break
                tok = line.split()
                if len(tok) != len(cols):
                    try:
                        tok = shlex.split(line)
                    except ValueError:
                        continue
                    if len(tok) != len(cols):
                        continue
                row = dict(zip(cols, tok))
                if row.get("group_PDB", "ATOM") not in ("ATOM", "HETATM"):
                    continue
                model = row.get("pdbx_PDB_model_num", "1")
                if first_model is None:
                    first_model = model
                elif model != first_model:
                    break
                if row.get("label_comp_id", "").upper() not in AMINO_ACIDS:                  # water, ligands, nucleic acids (MSE is a HETATM amino acid and stays)
                    continue
                if row.get("label_alt_id", ".") not in (".", "?", "A", ""):
                    continue
                if row.get("type_symbol", "C").upper() in ("H", "D"):
                    continue
                try:
                    xyz = (float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"]))
                except (KeyError, ValueError):
                    continue
                heavy.append(xyz)
                if row.get("label_atom_id", "").strip('"') == "CA":
                    ca.append(xyz)
                    chain = row.get("auth_asym_id", row.get("label_asym_id", "?"))
                    chains[chain] = chains.get(chain, 0) + 1
    if not cols or not heavy:
        raise ValueError("%s: no protein atoms found in an _atom_site loop" % cif_path)
    return {"ca": np.array(ca, dtype=np.float64).reshape(-1, 3), "heavy": np.array(heavy, dtype=np.float64).reshape(-1, 3), "chains": chains}


# ------------------------------------------------------------------------------------------------------------------------------ the frame
def frame_candidates(info):
    """The offsets (A) of the first voxel in model coordinates that a header can mean: the start indices, the ORIGIN field, or both."""
    voxel = np.array(info["voxel"], dtype=float)
    start = np.array(info["nstart"], dtype=float) * voxel
    origin = np.array(info["origin"], dtype=float)
    cands = {"start": start, "origin": origin, "start+origin": start + origin}
    out, seen = {}, []
    for name, v in cands.items():
        if not any(np.allclose(v, s) for s in seen):
            seen.append(v)
            out[name] = v
    return out


def density_at(vol, info, offset, xyz):
    """Trilinear density of the map at the points `xyz` (n, 3) in model coordinates; 0 outside the map. Also the fraction of the points inside."""
    from scipy.ndimage import map_coordinates
    voxel = np.array(info["voxel"], dtype=float)
    idx = (np.asarray(xyz, dtype=float) - np.asarray(offset, dtype=float)) / voxel           # (ix, iy, iz)
    nz, ny, nx = vol.shape
    inside = (idx[:, 0] >= 0) & (idx[:, 0] <= nx - 1) & (idx[:, 1] >= 0) & (idx[:, 1] <= ny - 1) & (idx[:, 2] >= 0) & (idx[:, 2] <= nz - 1)
    values = map_coordinates(vol, [idx[:, 2], idx[:, 1], idx[:, 0]], order=1, mode="constant", cval=0.0)
    return values, float(inside.mean()) if len(idx) else 0.0


def choose_frame(vol, info, ca_xyz):
    """(name, offset, scores) where scores = {name: mean density at the C-alpha atoms}; the best one wins, the first listed on a tie."""
    scores, best = {}, None
    for name, off in frame_candidates(info).items():
        values, _ = density_at(vol, info, off, ca_xyz)
        scores[name] = float(values.mean()) if len(values) else float("nan")
        if best is None or scores[name] > scores[best[0]] + 1e-9:
            best = (name, off)
    return best[0], best[1], scores


def model_mask(shape_zyx, info, offset, xyz, radius):
    """Boolean array [z][y][x]: voxels whose centre is within `radius` A of one of the points `xyz`."""
    voxel = np.array(info["voxel"], dtype=float)
    mask = np.zeros(shape_zyx, dtype=bool)
    nz, ny, nx = shape_zyx
    idx = (np.asarray(xyz, dtype=float) - np.asarray(offset, dtype=float)) / voxel
    reach = np.ceil(radius / voxel).astype(int) + 1
    r2 = radius * radius
    for c in idx:
        lo = np.maximum(np.floor(c - reach).astype(int), 0)
        hi = np.minimum(np.ceil(c + reach).astype(int) + 1, [nx, ny, nz])
        if np.any(hi <= lo):
            continue
        gx = (np.arange(lo[0], hi[0]) - c[0]) * voxel[0]
        gy = (np.arange(lo[1], hi[1]) - c[1]) * voxel[1]
        gz = (np.arange(lo[2], hi[2]) - c[2]) * voxel[2]
        d2 = gz[:, None, None] ** 2 + gy[None, :, None] ** 2 + gx[None, None, :] ** 2
        mask[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] |= d2 <= r2
    return mask


def reflections(edges_A, d):
    """Number of unique reflections of a P1 cell with these edges (A) up to resolution d (A), Friedel pairs counted once: V * 4 pi / 3 / d^3 / 2.
    Phaser reported 915,139 for the 265.6 A cube of entry 26DK at 3.5 A; this formula gives 915,000."""
    volume = float(edges_A[0]) * float(edges_A[1]) * float(edges_A[2])
    return volume * (4.0 * math.pi / 3.0) / (d ** 3) / 2.0


# ---------------------------------------------------------------------------------------------------------------------------- the survey
def survey(map_path, cif_path, contour=None, resolution=None, radius=3.0, min_small_A3=500.0, min_large_A3=5000.0, top_components=15):
    """The numbers of one entry (a dict with the keys of SURVEY_COLUMNS that this function knows, plus "components": a list of the largest components)."""
    t0 = time.time()
    vol, info = read_mrc(map_path)
    atoms = read_atoms(cif_path)
    voxel = np.array(info["voxel"], dtype=float)
    voxel_volume = float(np.prod(voxel))
    nz, ny, nx = vol.shape
    res = {"grid_x": nx, "grid_y": ny, "grid_z": nz, "voxel_A": round(float(np.mean(voxel)), 4), "map_mode": info["mode"],
           "box_x_A": round(nx * voxel[0], 1), "box_y_A": round(ny * voxel[1], 1), "box_z_A": round(nz * voxel[2], 1),
           "map_min": float(vol.min()), "map_max": float(vol.max()), "map_mean": float(vol.mean()), "map_std": float(vol.std())}
    note = []
    name, offset, scores = choose_frame(vol, info, atoms["ca"])
    ca_density, inside = density_at(vol, info, offset, atoms["ca"])
    res.update(frame=name, frame_scores=json.dumps({k: round(v, 5) for k, v in scores.items()}), ca_in_map_frac=round(inside, 4))
    if contour is None:
        contour = res["map_mean"] + 3.0 * res["map_std"]
        res["contour_source"] = "mean + 3 std (no level in the header file)"
        note.append("contour level taken as mean + 3 std")
    else:
        res["contour_source"] = "EMDB header file"
    res["contour_level"] = float(contour)
    res["ca_mean_density_over_contour"] = round(float(ca_density.mean()) / contour, 3) if contour else ""
    if inside < 0.9:
        note.append("only %.0f%% of the C-alpha atoms lie inside the map in the frame '%s': the model and the map may not be in one frame" % (100 * inside, name))
    if contour and float(ca_density.mean()) < 0.5 * contour:
        note.append("the C-alpha atoms sit at low density (mean %.3g, contour %.3g): check the frame or the contour level" % (float(ca_density.mean()), contour))
    above = vol > contour
    n_above = int(above.sum())
    mask = model_mask(vol.shape, info, offset, atoms["heavy"], radius)
    in_model = int((above & mask).sum())
    res.update(n_chains=len(atoms["chains"]), n_ca=int(len(atoms["ca"])), n_heavy_atoms=int(len(atoms["heavy"])),
               model_extent_A=round(float(np.linalg.norm(atoms["ca"].max(axis=0) - atoms["ca"].min(axis=0))), 1),
               above_volume_A3=round(n_above * voxel_volume), above_in_model_A3=round(in_model * voxel_volume))
    res["above_fraction_in_model"] = round(in_model / n_above, 4) if n_above else ""
    res["copies_of_model_estimate"] = round(n_above / in_model, 2) if in_model else ""
    comps = []
    if n_above:
        from scipy.ndimage import find_objects, label
        lab, n = label(above)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        touching = np.bincount(lab[mask & above], minlength=n + 1)[1:]
        order = np.argsort(-sizes)
        big_small = sizes * voxel_volume >= min_small_A3
        big_large = sizes * voxel_volume >= min_large_A3
        res.update(components_ge_500A3=int(big_small.sum()), components_ge_5000A3=int(big_large.sum()),
                   components_touching_model=int(((touching > 0) & big_small).sum()), largest_component_A3=round(float(sizes[order[0]]) * voxel_volume),
                   largest_component_fraction=round(float(sizes[order[0]]) / n_above, 4))
        objs = find_objects(lab, max_label=n)
        for rank, k in enumerate(order[:top_components], 1):
            sl = objs[k]
            sub = lab[sl] == (k + 1)
            zz, yy, xx = np.nonzero(sub)
            centre = np.array([xx.mean() + sl[2].start, yy.mean() + sl[1].start, zz.mean() + sl[0].start]) * voxel + offset
            comps.append({"rank": rank, "volume_A3": round(float(sizes[k]) * voxel_volume), "voxels_in_model_region": int(touching[k]),
                          "centre_x": round(float(centre[0]), 1), "centre_y": round(float(centre[1]), 1), "centre_z": round(float(centre[2]), 1)})
        del lab
        zs, ys, xs = np.nonzero(above.any(axis=(1, 2))), np.nonzero(above.any(axis=(0, 2))), np.nonzero(above.any(axis=(0, 1)))
        ext = [(a[0].max() - a[0].min() + 1) * v for a, v in ((xs, voxel[0]), (ys, voxel[1]), (zs, voxel[2]))]
        res.update(content_x_A=round(float(ext[0]), 1), content_y_A=round(float(ext[1]), 1), content_z_A=round(float(ext[2]), 1),
                   content_fraction_of_box=round(float(np.prod(ext)) / float(nx * ny * nz * voxel_volume), 4))
        if resolution:
            res["reflections_content_M"] = round(reflections(ext, resolution) / 1e6, 3)
    if resolution:
        full = reflections((nx * voxel[0], ny * voxel[1], nz * voxel[2]), resolution)
        res.update(reflections_full_M=round(full / 1e6, 3), docking_load_vs_26DK=round(full / REFERENCE_REFLECTIONS, 2))
    res.update(seconds=round(time.time() - t0, 1), note="; ".join(note), components=comps)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map")
    p.add_argument("cif")
    p.add_argument("--contour", type=float)
    p.add_argument("--resolution", type=float)
    p.add_argument("--radius", type=float, default=3.0)
    args = p.parse_args(argv)
    path = gunzip_if_needed(args.map, os.path.splitext(os.path.abspath(args.map))[0])
    res = survey(path, args.cif, args.contour, args.resolution, args.radius)
    comps = res.pop("components")
    print(json.dumps(res, indent=1))
    for c in comps:
        print(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
