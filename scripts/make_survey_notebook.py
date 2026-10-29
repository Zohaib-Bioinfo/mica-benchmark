#!/usr/bin/env python3
"""Build scripts/survey.ipynb: a CPU-only survey of what the map of every entry holds, compared with its deposited model.

Why it exists. The first pilot showed that entry 26DK (one deposited chain) sits in a map with about fifteen copies of that chain, that its 265 A cube costs the Phenix
docking of MICA hours, and that methods which build every copy of the map are scored against one chain. Before the full runs we need, for all 20 entries: the size of the
map, how many copies of the deposited model it holds, how many reflections Phaser would have to handle, and what EMDB and RCSB say about symmetry and assembly.

It downloads each map and model (as the first pilot did), reads the EMDB header file and the RCSB description, runs `map_survey.survey` and writes `survey.csv` (one line
per entry) and `survey_components.csv` (the largest connected pieces of density per entry). No Phenix, no GPU, no secrets, no private input; about 30 to 60 minutes.

    python make_survey_notebook.py            # writes scripts/survey.ipynb

Run it on Kaggle with the accelerator off, as "Save & Run All (Commit)".
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_pilot_notebook as P  # noqa: E402
import nbkit  # noqa: E402

NOTEBOOK = "survey"
NB_NAME = "survey.ipynb"
RCSB_API = "https://data.rcsb.org/rest/v1/core"
FILES = (("scripts/nbtools.py", "scripts/nbtools.py"),
         ("scripts/map_survey.py", "scripts/map_survey.py"),
         ("data/entries_2026.csv", "data/entries_2026.csv"))
NET_HOSTS = (("EBI FTP (EMDB maps and header files)", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-80558/header/emd-80558.xml", True),
             ("RCSB files (deposited models)", "https://files.rcsb.org/header/26DK.cif", True),
             ("RCSB data API (assemblies)", "https://data.rcsb.org/rest/v1/core/entry/26DK", False))

INTRO = """\
# MICA benchmark: what does the map of each entry hold? (CPU survey of @N_ENTRIES@ entries)

The first pilot (entry 26DK) showed that the deposited model can be one chain while the map holds about fifteen copies of it. That changes the cost of every method and the
way the models are scored. This notebook measures it for every entry of `data/entries_2026.csv`:

1. checks the machine and the internet (no GPU, no Phenix, no secrets and no private files are needed),
2. for each entry downloads the map (EMDB) and the deposited model (RCSB) and reads the EMDB header file (contour level, resolution, symmetry) and the RCSB assembly,
3. measures the grid and box size, the volume above the contour level, how much of it lies within 3 A of the deposited model, how many copies of the model that suggests,
   the connected pieces of density, and the number of reflections that Phaser would get at the map resolution,
4. writes `survey.csv` and `survey_components.csv`, refreshes a partial zip after every entry, and packs one zip.

**What you do (Kaggle, background mode)**: accelerator **None**, Internet **On**, then *Save Version -> Save & Run All (Commit)*. It runs on Kaggle's servers; you can close
the tab and your Wi-Fi does not matter. About 30 to 60 minutes. When the version shows *Succeeded*, open it and download the zip `survey_result_....zip` from its Output.
Send it to me. **Do not stop or cancel the run** unless it is stuck for more than 3 hours.
"""

STEP3_MD = """\
## Step 3. The survey

For each entry: the map and the deposited model are downloaded (the map is deleted again after the entry), the numbers are measured by `scripts/map_survey.py`, and
one line is added to `survey.csv` and the zip is refreshed. An entry that fails (download, unreadable file) is recorded with the reason and the next one follows."""

STEP3 = r'''
import map_survey

RCSB_API = "@RCSB_API@"
ENTRIES = @ENTRIES@                                   # [] = every entry of the table
MAPS_DIR = os.path.join(BIG, "maps")
COMPONENT_COLUMNS = ["entry", "rank", "volume_A3", "voxels_in_model_region", "centre_x", "centre_y", "centre_z"]
RESULTS = []
@FUNCTIONS@


def header_facts(text):
    """Symmetry and copy numbers from an EMDB header file (the tags are found by name, so the version of the schema does not matter)."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    local = lambda el: el.tag.rsplit("}", 1)[-1]
    texts = lambda name: "; ".join(dict.fromkeys(el.text.strip() for el in root.iter() if local(el) == name and el.text and el.text.strip()))
    sample = [el for el in root.iter() if local(el) == "sample"]
    names = [ch.text.strip() for el in sample[:1] for ch in el if local(ch) == "name" and ch.text]
    return {"point_group": texts("point_group"), "number_of_copies_header": texts("number_of_copies"), "oligomeric_state_header": texts("oligomeric_state"),
            "sample_name": (names[0] if names else "")[:120]}


def rcsb_facts(pdb):
    """What RCSB says about the assembly. Every field is optional: it only goes into the table."""
    import requests
    out = {}
    try:
        e = requests.get("%s/entry/%s" % (RCSB_API, pdb), timeout=60).json()
        out["assembly_count"] = e.get("rcsb_entry_info", {}).get("assembly_count", "")
    except Exception as ex:
        out["rcsb_note"] = "RCSB entry: %s" % str(ex)[:80]
    try:
        a = requests.get("%s/assembly/%s/1" % (RCSB_API, pdb), timeout=60).json()
        out["assembly1_polymer_instances"] = a.get("rcsb_assembly_info", {}).get("polymer_instance_count", "")
        out["assembly1_oligomeric_count"] = a.get("pdbx_struct_assembly", {}).get("oligomeric_count", "")
        out["symmetry_rcsb"] = "; ".join(("%s %s" % (s.get("symbol", ""), s.get("kind", ""))).strip() for s in (a.get("rcsb_struct_symmetry") or []))
    except Exception as ex:
        out["rcsb_note"] = (out.get("rcsb_note", "") + " RCSB assembly: %s" % str(ex)[:80]).strip()
    return out


def survey_entry(row):
    """(row of survey.csv, components) for one entry of the table."""
    entry, number = row["pdb_id"], row["emdb_id"].split("-")[-1]
    out = dict.fromkeys(map_survey.SURVEY_COLUMNS, "")
    out.update(entry=entry, emdb_id=row["emdb_id"], resolution_table_A=row["resolution_A"])
    t0 = time.time()
    gz = download(row["map_url"], os.path.join(BIG, "downloads", os.path.basename(row["map_url"])), 100000)
    map_path = map_survey.gunzip_if_needed(gz, os.path.join(MAPS_DIR, "emd_%s.map" % number))
    out["map_file_mb"] = round(os.path.getsize(map_path) / 1e6, 1)
    out["map_sha256"] = sha256_file(map_path)
    cif = download(row["cif_url"], os.path.join(WORK, "native", entry + ".cif"), 1000)
    notes, contour, resolution = [], None, None
    try:
        xml_url = re.sub(r"/map/[^/]*$", "/header/emd-%s.xml" % number, row["map_url"])
        with open(download(xml_url, os.path.join(BIG, "downloads", "emd-%s.xml" % number), 200), encoding="utf-8") as f:
            text = f.read()
        contour, resolution = contour_from_header_xml(text)
        out.update(header_facts(text))
    except Exception as ex:
        notes.append("EMDB header file: %s" % str(ex)[:120])
    out["resolution_header_A"] = resolution if resolution is not None else ""
    facts = rcsb_facts(entry)
    if facts.get("rcsb_note"):
        notes.append(facts.pop("rcsb_note"))
    out.update(facts)
    res = map_survey.survey(map_path, cif, contour, resolution if resolution is not None else float(row["resolution_A"]))
    comps = res.pop("components")
    out.update(res)
    out["note"] = "; ".join(x for x in [out.get("note", "")] + notes if x)
    out["seconds"] = round(time.time() - t0, 1)                      # downloads included
    return out, comps


def clean_up():
    for pattern in (os.path.join(MAPS_DIR, "*.map"), os.path.join(BIG, "downloads", "*.map.gz")):
        for p in glob.glob(pattern):
            try:
                os.remove(p)
            except OSError:
                pass


def summary_table(results):
    log("\n%-5s %-13s %6s %7s %5s %8s %7s %7s %6s %8s %8s  %s" % ("entry", "grid", "voxel", "box A", "res", "contour", "copies", "inside", "pieces", "Mrefl", "dock x", "symmetry / note"))
    for o in results:
        if not o.get("grid_x"):
            log("%-5s %s" % (o["entry"], o.get("note", "")))
            continue
        log("%-5s %-13s %6s %7s %5s %8.3g %7s %7s %6s %8s %8s  %s%s" % (
            o["entry"], "%sx%sx%s" % (o["grid_x"], o["grid_y"], o["grid_z"]), o["voxel_A"], o["box_x_A"], o["resolution_header_A"] or o["resolution_table_A"], o["contour_level"],
            o["copies_of_model_estimate"], o["above_fraction_in_model"], o["components_ge_5000A3"], o["reflections_full_M"], o["docking_load_vs_26DK"],
            o.get("point_group") or o.get("symmetry_rcsb") or "-", ("  | " + o["note"]) if o.get("note") else ""))
    log("copies = volume above the contour level / the part of it within 3 A of the deposited atoms; inside = that part as a fraction; pieces = connected pieces of at least 5000 A^3;")
    log("Mrefl = millions of reflections of the whole box at the map resolution (Phaser's work in the docking); dock x = Mrefl relative to 26DK, which did not finish in 20 minutes.")


def run_survey():
    rows = {r["pdb_id"]: r for r in read_csv("data/entries_2026.csv")}
    comps_all = []
    for entry in (ENTRIES or list(rows)):
        res = STATE["entries"][entry] = {"entry": entry, "status": "failed", "note": ""}
        log("\n" + "=" * 72 + "\nENTRY %s\n" % entry + "=" * 72)
        try:
            out, comps = survey_entry(rows[entry])
            RESULTS.append(out)
            comps_all += [dict(c, entry=entry) for c in comps]
            res.update(status="ok", note=out["note"])
            log("%s: grid %sx%sx%s, voxel %s A, box %s A, contour %s, about %s copies of the model, %s pieces, %s M reflections (%s s)" % (
                entry, out["grid_x"], out["grid_y"], out["grid_z"], out["voxel_A"], out["box_x_A"], out["contour_level"], out["copies_of_model_estimate"],
                out["components_ge_5000A3"], out["reflections_full_M"], out["seconds"]))
            if out["note"]:
                log("    note: %s" % out["note"])
        except Exception:
            import traceback
            tb = traceback.format_exc()
            res["note"] = tb.strip().splitlines()[-1][:300]
            log("ENTRY %s FAILED:\n%s" % (entry, tb))
            RESULTS.append({"entry": entry, "emdb_id": rows.get(entry, {}).get("emdb_id", ""), "resolution_table_A": rows.get(entry, {}).get("resolution_A", ""),
                            "note": "FAILED: " + res["note"]})
        finally:
            clean_up()
        write_csv("survey.csv", map_survey.SURVEY_COLUMNS, RESULTS)
        write_csv("survey_components.csv", COMPONENT_COLUMNS, comps_all)
        pack_now()
    summary_table(RESULTS)
    STATE["complete"] = bool(STATE["entries"]) and all(e["status"] == "ok" for e in STATE["entries"].values())
    if not STATE["complete"]:
        STATE["stopped_at"] = "failed entries: " + ", ".join(e["entry"] for e in STATE["entries"].values() if e["status"] != "ok")


if STATE["ok"]:
    try:
        run_survey()
    except Exception:
        import traceback
        STATE["stopped_at"] = "survey"
        log("the survey stopped by an error:\n%s" % traceback.format_exc())
    pack_now()
else:
    log("skipped: an earlier step stopped")
'''

EXTRA_INFO = '{"input_sha256": globals().get("EXPECTED", {}), "entries": sorted(STATE["entries"]), "map_survey": "scripts/map_survey.py"}'


def build(out_path, entries=(), entries_path=None, net_hosts=NET_HOSTS, rcsb_api=RCSB_API):
    """Writes the notebook. entries=() means every entry of the table. The other keyword arguments exist for the tests (local servers)."""
    file_cells, expected = nbkit.file_cells(FILES, entries_path)
    step6 = nbkit.fill(P.STEP6, {"@ENTRIES@": "[]"})
    functions = nbkit.function_sources(step6, ["read_csv", "write_csv", "download", "contour_from_header_xml"]).rstrip("\n")
    if entries_path:
        import csv
        with open(entries_path, newline="", encoding="utf-8") as f:
            n_entries = len(list(csv.DictReader(f)))
    else:
        import csv
        with open(os.path.join(nbkit.ROOT, "data", "entries_2026.csv"), newline="", encoding="utf-8") as f:
            n_entries = len(list(csv.DictReader(f)))
    mapping = {"@ENTRIES@": repr(list(entries)), "@N_ENTRIES@": str(len(entries) if entries else n_entries), "@RCSB_API@": rcsb_api, "@REQUIRE_GPU@": "False",
               "@NET_HOSTS@": repr(tuple(net_hosts)), "@EXPECTED@": repr(expected), "@NOTEBOOK@": NOTEBOOK, "@EXTRA_INFO@": EXTRA_INFO}
    fl = lambda text: nbkit.fill(text, mapping).strip("\n") + "\n"
    step3 = STEP3.replace("@FUNCTIONS@", functions)
    cells = [("intro", nbkit.md(nbkit.fill(INTRO, mapping))),
             ("step1-title", nbkit.md(P.STEP1_MD)), ("step1", nbkit.code(fl(P.STEP1))),
             ("step2-title", nbkit.md(P.STEP2_MD)), ("step2-dirs", nbkit.code(P.STEP2_DIRS.strip("\n") + "\n"))]
    cells += file_cells
    cells += [("step2-check", nbkit.code(fl(P.STEP2_CHECK))),
              ("robust-title", nbkit.md(nbkit.fill(nbkit.ROBUST_MD, mapping))), ("robust", nbkit.code(fl(nbkit.ROBUST))),
              ("step3-title", nbkit.md(STEP3_MD)), ("step3", nbkit.code(fl(step3))),
              ("final-title", nbkit.md(nbkit.FINAL_MD)), ("final", nbkit.code(fl(nbkit.FINAL)))]
    nbkit.assemble(out_path, NB_NAME, cells)
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, NB_NAME))
    p.add_argument("--entries", nargs="*", default=[])
    args = p.parse_args(argv)
    expected = build(args.out, entries=args.entries)
    print("wrote %s with %d embedded files" % (args.out, len(expected)))
    for path, digest in expected.items():
        print("  %s  %s" % (digest[:16], path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
