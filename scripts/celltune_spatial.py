#!/usr/bin/env python
"""Spatial analysis on CellTune's final cell types, relative to the H&E tumor region.

CellTune has no spatial-analysis module of its own (it computes spatial *features*
to classify cells, and exports the labels). This script is the step after export:

    CellTune population export  +  per-cell regions from celltune_regions.py
        -> composition by region
        -> where each type sits relative to the tumor edge (signed distance)
        -> pairwise enrichment, using akoyalib.spatial (tissue window, labels
           shuffled within tissue pieces, BH-corrected)
        -> an ark-analysis-ready cell table, tiled with halos (--ark-dir)

Two cell-type labels are left out of every spatial statistic by default
(--exclude Garbage,Ambiguous): CellTune's own convention is that they are not
cell types, so they should not be partners in an interaction.

Status: validated on synthetic data (tests/test_celltune_spatial.py) and the ark
export was checked against ark-analysis 0.7.2 on synthetic tiles. The CellTune
export file format has been read about but not seen: if your first real export
has different column names, pass --id-col / --type-col.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from akoyalib import spatial as sp          # noqa: E402

ID_CANDIDATES = ("cellID", "CellID", "cell_id", "label", "Label")


# -- reading CellTune's export --------------------------------------------------------------------

def read_table(path):
    p = pathlib.Path(path)
    # keep_default_na=False: cell-type or region names such as "None" or "NA" are real labels, not missing values
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, keep_default_na=False, na_values=[""])


def pick_id_col(df, given=None):
    if given:
        return given
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit(f"cannot find a cell-ID column among {list(df.columns)[:12]}...; pass --id-col")


def join_types(regions, pop, id_col, type_col, image, roi_map=None):
    """Attach the CellTune cell type to each row of the regions table.

    With `roi_map` (the roi_map.csv that make_rois.py crop wrote), a population export made from
    ROI images, whose `image` column holds ROI ids, is translated back to the slide id first. Cell
    IDs are the whole-slide IDs, so the join needs nothing else. A cell cut by two ROI edges can
    appear twice; the first label is kept.
    """
    id_col = pick_id_col(pop, id_col)
    if roi_map is not None and "image" in pop.columns:
        pop = pop.assign(image=pop["image"].astype(str).map(dict(zip(roi_map["image"], roi_map["sample_id"])))
                         .fillna(pop["image"].astype(str)))
    if type_col not in pop.columns:
        raise SystemExit(f"--type-col {type_col!r} is not a column of the population export; "
                         f"columns are {list(pop.columns)[:15]}...")
    if "image" in pop.columns:
        pop = pop[pop["image"].astype(str) == str(image)]
    pop = pop[[id_col, type_col]].rename(columns={id_col: "cellID", type_col: "cell_type"})
    pop = pop.drop_duplicates("cellID")
    out = regions.merge(pop, on="cellID", how="left")
    out["cell_type"] = out["cell_type"].fillna("Unclassified")
    return out


# -- splitting a marker-defined type by H&E region -------------------------------------------------

def parse_split(spec):
    """'Mesenchymal_Vim=Tumor_Putative,Stromal_Vim,Boundary_Vim' -> (type, in, out, boundary)."""
    try:
        src, rest = spec.split("=", 1)
        inside, outside, boundary = [x.strip() for x in rest.split(",")]
    except ValueError:
        raise SystemExit(f"--split wants TYPE=IN_LABEL,OUT_LABEL,BOUNDARY_LABEL, got {spec!r}")
    return src.strip(), inside, outside, boundary


def split_by_region(df, rules, focus):
    """Relabel cells of one marker-defined type by where they sit on the H&E.

    CellTune decides what a cell is from its markers alone. A vimentin-positive,
    CD45/CD31/PanCK-negative cell inside the H&E tumor and one in the lung stroma
    carry the same markers, so only the region can tell them apart. Doing that
    split here, after the classifier, keeps the H&E out of the model: if the region
    were a model input, "tumor cells cluster in the tumor" would partly be the
    model's own prior played back as a finding.

    Cells within the registration margin of the edge (`near_boundary`) get their
    own label instead of being forced to one side.
    """
    col_in, col_b = f"in_{focus}", "near_boundary"
    for c in (col_in, col_b):
        if c not in df.columns:
            raise SystemExit(f"--split needs column {c!r} from celltune_regions.py")
    out = df.copy()
    for src, lab_in, lab_out, lab_b in rules:
        m = out["cell_type"] == src
        out.loc[m & (out[col_b] == 0) & (out[col_in] == 1), "cell_type"] = lab_in
        out.loc[m & (out[col_b] == 0) & (out[col_in] == 0), "cell_type"] = lab_out
        out.loc[m & (out[col_b] == 1), "cell_type"] = lab_b
    return out


# -- the observation window -----------------------------------------------------------------------

def window_from_cells(xy_px, px_um, bin_px=32, close_um=60.0):
    """A tissue window built from where cells are, for when no stage-01 mask is at hand.

    Occupied bins, closed over `close_um` to bridge the gaps between cells, then
    connected components as tissue pieces. It is coarser than the stage-01 mask and
    will treat a large cell-free hole (an airway, a necrotic core) as outside the
    tissue, which for point-pattern purposes is correct. Prefer the stage-01 mask
    (--window) whenever you have it.
    """
    h = int(np.ceil(xy_px[:, 1].max() / bin_px)) + 1
    w = int(np.ceil(xy_px[:, 0].max() / bin_px)) + 1
    occ = np.zeros((h, w), bool)
    occ[(xy_px[:, 1] / bin_px).astype(int), (xy_px[:, 0] / bin_px).astype(int)] = True
    r = max(1, int(round(close_um / (bin_px * px_um))))
    mask = ndi.binary_closing(ndi.binary_dilation(occ, iterations=r), iterations=r)
    pieces, _ = ndi.label(mask)
    return sp.Window(mask, pieces.astype(np.int32), float(bin_px), px_um)


# -- summaries ------------------------------------------------------------------------------------

def composition(df, by="region"):
    t = pd.crosstab(df["cell_type"], df[by])
    frac = t / t.sum(axis=0)
    return t, frac


def distance_to_edge_table(df, focus, margin_um):
    """Where each cell type sits relative to the focus region's edge."""
    col = f"signed_um_{focus}"
    rows = []
    for ct, g in df.groupby("cell_type"):
        s = g[col].to_numpy()
        rows.append({"cell_type": ct, "n": len(g),
                     "median_signed_um": float(np.median(s)),
                     f"pct_deep_in_{focus}": 100 * float(np.mean(s > margin_um)),
                     "pct_margin_in": 100 * float(np.mean((s > 0) & (s <= margin_um))),
                     "pct_margin_out": 100 * float(np.mean((s <= 0) & (s > -margin_um))),
                     "pct_far_out": 100 * float(np.mean(s <= -margin_um))})
    return pd.DataFrame(rows).sort_values("cell_type")


def enrichment_long(res):
    q = sp.benjamini_hochberg(res["p"])
    types = res["types"]
    rows = []
    for i, a in enumerate(types):
        for j, b in enumerate(types):
            rows.append({"a": a, "b": b, "observed": int(res["observed"][i, j]),
                         "log2_oe": res["log2_oe"][i, j], "z": res["z"][i, j],
                         "p": res["p"][i, j], "q_bh": q[i, j]})
    return pd.DataFrame(rows)


# -- the ark-analysis export ----------------------------------------------------------------------

def tile_table(df, tile_um, halo_um, px_um, type_col="cell_type"):
    """Rows for ark-analysis, one block per tile, each tile carrying a halo.

    ark builds a dense cell-by-cell distance matrix per `fov`, so a whole slide
    cannot be one fov (1.47 million cells would need ~8.7 TB). Cutting tiles has
    one catch: a cell near a tile edge loses the neighbours that fall in the next
    tile, so its neighbourhood counts come out too low. The halo repeats those
    cells in this tile. `is_core` marks the cells whose neighbourhoods are exact
    (every neighbour within `halo_um` is present); keep only those from ark's output.
    """
    if halo_um < 0 or tile_um <= 0:
        raise ValueError("tile_um must be > 0 and halo_um >= 0")
    x, y = df["x_px"].to_numpy() * px_um, df["y_px"].to_numpy() * px_um
    ti, tj = (y // tile_um).astype(int), (x // tile_um).astype(int)
    parts = []
    for (i, j) in sorted(set(zip(ti.tolist(), tj.tolist()))):
        x0, y0 = j * tile_um, i * tile_um
        inside = ((x >= x0 - halo_um) & (x < x0 + tile_um + halo_um) &
                  (y >= y0 - halo_um) & (y < y0 + tile_um + halo_um))
        core = (ti == i) & (tj == j)
        g = df.loc[inside, ["cellID", "x_px", "y_px", "area_px", type_col]].copy()
        g.columns = ["label", "centroid-0", "centroid-1", "cell_size", "cell_meta_cluster"]
        g.insert(0, "fov", f"tile_r{i}_c{j}")
        g["is_core"] = core[inside]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# -- main -----------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", required=True, help="CSV from celltune_regions.py (has image, cellID, x_px, y_px, region, signed_um_*)")
    ap.add_argument("--populations", required=True, help="CellTune Export > Populations (or CellTable) .csv/.parquet")
    ap.add_argument("--type-col", default="Pred_AVG", help="column holding the final cell type (default Pred_AVG)")
    ap.add_argument("--id-col", default=None, help="cell-ID column in the population export (auto-detected)")
    ap.add_argument("--roi-map", default=None, help="roi_map.csv from make_rois.py crop, when the CellTune project was built from ROI images")
    ap.add_argument("--px-um", type=float, required=True)
    ap.add_argument("--focus", default="Tumor", help="region whose edge distances are reported")
    ap.add_argument("--split", action="append", default=[],
                    help="TYPE=IN_LABEL,OUT_LABEL,BOUNDARY_LABEL: relabel a marker-defined type by H&E region "
                         "(repeatable). e.g. Mesenchymal_Vim=Tumor_Putative,Stromal_Vim,Boundary_Vim")
    ap.add_argument("--exclude", default="Garbage,Ambiguous", help="cell types left out of all statistics")
    ap.add_argument("--window", default=None, help="stage-01 tissue-mask .npz (preferred); else built from cell positions")
    ap.add_argument("--radius-um", type=float, default=50.0, help="neighbour radius for enrichment")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--margin-um", type=float, default=50.0)
    ap.add_argument("--min-cells", type=int, default=100, help="types with fewer cells than this are not tested")
    ap.add_argument("--ark-dir", default=None, help="also write an ark-analysis cell table, tiled with halos, here")
    ap.add_argument("--tile-um", type=float, default=1000.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    regions = read_table(a.regions)
    image = regions["image"].iloc[0]
    roi_map = pd.read_csv(a.roi_map) if a.roi_map else None
    df = join_types(regions, read_table(a.populations), a.id_col, a.type_col, image, roi_map)
    if a.split:
        before = df["cell_type"].value_counts()
        df = split_by_region(df, [parse_split(x) for x in a.split], a.focus)
        after = df["cell_type"].value_counts()
        for src, lab_in, lab_out, lab_b in (parse_split(x) for x in a.split):
            print(f"split {src} ({int(before.get(src, 0)):,}): {lab_in} {int(after.get(lab_in, 0)):,}, "
                  f"{lab_out} {int(after.get(lab_out, 0)):,}, {lab_b} {int(after.get(lab_b, 0)):,}")
    # the one table every later step reads: type, region and position for every cell, Garbage and Ambiguous included
    keep_cols = [c for c in df.columns if c in ("image", "cellID", "cell_type", "region", "x_px", "y_px", "area_px",
                                                  "near_boundary") or c.startswith(("in_", "signed_um_"))]
    df[keep_cols].to_csv(out / "cell_types_final.csv", index=False)
    excl = {s.strip() for s in a.exclude.split(",") if s.strip()}
    n_all = len(df)
    df = df[~df["cell_type"].isin(excl)].reset_index(drop=True)
    print(f"{n_all:,} cells; {n_all - len(df):,} left out as {sorted(excl)}; {len(df):,} analysed")
    unc = int((df["cell_type"] == "Unclassified").sum())
    if unc:
        print(f"WARNING: {unc:,} cells had no type in the population export (labelled Unclassified)")

    ct, frac = composition(df)
    ct.to_csv(out / "composition_counts_by_region.csv")
    frac.to_csv(out / "composition_fraction_by_region.csv")

    focus_col = f"signed_um_{a.focus}"
    if focus_col in df.columns:
        distance_to_edge_table(df, a.focus, a.margin_um).to_csv(out / f"distance_to_{a.focus}_edge.csv", index=False)

    xy = df[["x_px", "y_px"]].to_numpy(float)
    window = sp.Window.from_stage01(a.window, a.px_um) if a.window else window_from_cells(xy, a.px_um)
    if not a.window:
        print("NOTE: no --window given; tissue window was built from cell positions (coarser than the stage-01 mask)")

    counts = df["cell_type"].value_counts()
    keep = df["cell_type"].map(counts) >= a.min_cells
    small = sorted(set(df.loc[~keep, "cell_type"]))
    if small:
        print(f"not tested for enrichment (< {a.min_cells} cells): {small}")
    scopes = {"all": keep}
    if "region" in df.columns:
        scopes[f"in_{a.focus}"] = keep & (df["region"] == a.focus)
        scopes[f"outside_{a.focus}"] = keep & (df["region"] != a.focus)
    for name, m in scopes.items():
        sub = df[m]
        if sub["cell_type"].nunique() < 2 or len(sub) < 2:
            print(f"enrichment [{name}]: fewer than two testable types, skipped")
            continue
        res = sp.enrichment(sub[["x_px", "y_px"]].to_numpy(float), sub["cell_type"].to_numpy(), window,
                            radius_um=a.radius_um, n_perm=a.n_perm, px_um=a.px_um)
        enrichment_long(res).to_csv(out / f"enrichment_{name}.csv", index=False)
        print(f"enrichment [{name}]: {len(sub):,} cells, {res['n_edges']:,} edges within {a.radius_um:g} um")

    if a.ark_dir:
        ad = pathlib.Path(a.ark_dir)
        ad.mkdir(parents=True, exist_ok=True)
        halo_um = a.radius_um
        t = tile_table(df, a.tile_um, halo_um, a.px_um)
        t.to_csv(ad / "ark_cell_table.csv", index=False)
        n_tiles = t["fov"].nunique()
        biggest = int(t.groupby("fov").size().max())
        gb = biggest ** 2 * 4 / 1e9
        print(f"ark table: {len(t):,} rows in {n_tiles} tiles; largest tile {biggest:,} cells "
              f"-> ~{gb:.1f} GB distance matrix. Use dist_lim = {a.radius_um / a.px_um:.0f} px; "
              f"keep only is_core == True from ark's output.")


if __name__ == "__main__":
    main()
