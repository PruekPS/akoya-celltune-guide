#!/usr/bin/env python
"""Choose and cut ROIs from a segmented slide, for labelling in CellTune.

Two sub-commands, run in this order for every slide.

  propose   Lay a grid of square ROIs over the slide, drop the ones that are mostly
            empty or sit on artifacts, sort the rest into strata (by H&E region when
            you have it), and draw a stratified random sample. Writes a CSV and a
            GeoJSON you can open in QuPath to look at before committing.

  crop      Cut each chosen ROI out of the slide into the folder layout CellTune's
            single-TIFF projects use:  <out>/<roi_id>/<Marker>.tif, segmentation_labels.tif
            (and optionally <Class>__Mask.tif). Cell IDs are copied unchanged from the
            whole-slide label image, so every label you make in an ROI maps back to
            the same cell on the full slide. roi_map.csv records each ROI's offset.

Why segment the whole slide first and crop afterwards: a cell keeps one ID for the
life of the project, ROI choice can use the whole-slide cell table (density, region,
QC), and a new ROI never needs a new segmentation run. ROIs are a labelling device,
not a statistical sample: see the training manual before using ROIs to compare groups.

Status: validated on synthetic data only (tests/test_make_rois.py).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import celltune_regions as cr          # noqa: E402


# -- propose ------------------------------------------------------------------------------------

def tile_grid(shape_hw, roi_px):
    """Top-left corners of a non-overlapping grid of roi_px squares that fit in the image."""
    h, w = shape_hw
    ys = np.arange(0, h - roi_px + 1, roi_px)
    xs = np.arange(0, w - roi_px + 1, roi_px)
    return [(int(y), int(x)) for y in ys for x in xs]


def tile_stats(cells, grid, roi_px, occ_bins=10):
    """Per tile: cell count and an occupancy proxy for how much of the tile is tissue.

    Occupancy is the share of an occ_bins x occ_bins sub-grid that holds at least one
    cell centroid. It stands in for a tissue mask: good enough to reject empty glass,
    not an area measurement. Pass a real tissue mask through --exclude if you have one.
    """
    x, y = cells["x_px"].to_numpy(), cells["y_px"].to_numpy()
    rows = []
    for (y0, x0) in grid:
        m = (x >= x0) & (x < x0 + roi_px) & (y >= y0) & (y < y0 + roi_px)
        n = int(m.sum())
        if n:
            bi = np.minimum(((y[m] - y0) / roi_px * occ_bins).astype(int), occ_bins - 1)
            bj = np.minimum(((x[m] - x0) / roi_px * occ_bins).astype(int), occ_bins - 1)
            occ = len(set(zip(bi.tolist(), bj.tolist()))) / occ_bins ** 2
        else:
            occ = 0.0
        rows.append({"y0": y0, "x0": x0, "n_cells": n, "occupancy": occ, "_mask": m})
    return rows


def assign_stratum(frac, focus, interface_min):
    """Name the stratum from the region mix of a tile.

    `frac` maps region name -> share of the tile's cells. With a focus region (e.g.
    Tumor) a tile that holds at least `interface_min` of both the focus and everything
    else is an `interface` tile; otherwise the majority region names the stratum.
    """
    if not frac:
        return "all"
    if focus:
        f = frac.get(focus, 0.0)
        if min(f, 1 - f) >= interface_min:
            return "interface"
    return max(frac, key=frac.get)


def propose(cells, shape_hw, roi_um, px_um, target_cells, regions_raster=None, priority=None,
            downsample=8, focus=None, interface_min=0.25, min_occupancy=0.6, min_cells=300,
            exclude_raster=None, exclude_max=0.05, validation_frac=0.2, seed=0):
    roi_px = int(round(roi_um / px_um))
    grid = tile_grid(shape_hw, roi_px)
    stats = tile_stats(cells, grid, roi_px)
    rng = np.random.default_rng(seed)

    reg = None
    if regions_raster is not None:
        idx = regions_raster[np.clip((cells["y_px"].to_numpy() / downsample).astype(int), 0, regions_raster.shape[0] - 1),
                             np.clip((cells["x_px"].to_numpy() / downsample).astype(int), 0, regions_raster.shape[1] - 1)]
        names = np.array(["Unassigned"] + list(priority))
        reg = names[idx]

    out = []
    for s in stats:
        keep = s["occupancy"] >= min_occupancy and s["n_cells"] >= min_cells
        if keep and exclude_raster is not None:
            y0, x0 = s["y0"] // downsample, s["x0"] // downsample
            sub = exclude_raster[y0:y0 + roi_px // downsample, x0:x0 + roi_px // downsample]
            keep = sub.size > 0 and float((sub > 0).mean()) <= exclude_max
        if not keep:
            continue
        frac = {}
        if reg is not None:
            vc = pd.Series(reg[s["_mask"]]).value_counts(normalize=True)
            frac = vc.to_dict()
        row = {"y0": s["y0"], "x0": s["x0"], "y1": s["y0"] + roi_px, "x1": s["x0"] + roi_px,
               "n_cells": s["n_cells"], "occupancy": round(s["occupancy"], 3),
               "stratum": assign_stratum(frac, focus, interface_min)}
        for k, v in frac.items():
            row[f"frac_{k}"] = round(float(v), 3)
        out.append(row)
    elig = pd.DataFrame(out)
    if elig.empty:
        return elig, elig

    per_roi = float(elig["n_cells"].median())
    n_total = max(1, int(np.ceil(target_cells / per_roi)))
    strata = sorted(elig["stratum"].unique())
    # Equal allocation across strata, so a stratum covering 5% of the slide still gets its share
    # (rare tissue contexts are where a classifier is weakest). A stratum with fewer eligible
    # tiles than its share gives the surplus back, and it is re-dealt to strata with room.
    cap = elig.groupby("stratum").size().to_dict()
    alloc = {st: 0 for st in strata}
    for _ in range(min(n_total, int(sum(cap.values())))):
        open_ = [st for st in strata if alloc[st] < cap[st]]
        alloc[min(open_, key=lambda st: alloc[st])] += 1      # one at a time to the emptiest open stratum
    chosen = []
    for st in strata:
        pool = elig[elig["stratum"] == st]
        chosen.append(pool.iloc[rng.permutation(len(pool))[:alloc[st]]])
    sel = pd.concat(chosen).reset_index(drop=True)

    # hold out a validation share, per stratum, so no stratum is missing from either role
    sel["role"] = "train"
    for st in strata:
        ix = sel.index[sel["stratum"] == st].to_numpy()
        n_val = int(round(validation_frac * len(ix)))
        if len(ix) >= 3 and n_val == 0 and validation_frac > 0:
            n_val = 1
        sel.loc[rng.permutation(ix)[:n_val], "role"] = "validation"
    sel = sel.sort_values(["y0", "x0"]).reset_index(drop=True)
    return sel, elig


# QuPath colours for the two roles, as [R, G, B]. QuPath 0.5.1 refuses to import a classification
# object that has a name but no colour ("Unable to parse PathClass", GsonTools.PathClassTypeAdapter),
# so the colour is not decoration: without it the whole file fails to load.
ROI_COLORS = {"train": [255, 200, 0], "validation": [0, 220, 255]}
ROI_COLOR_OTHER = [128, 128, 128]


def to_geojson(rois, sample_id):
    feats = []
    for r in rois.itertuples():
        ring = [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1], [r.x0, r.y0]]
        classification = {"name": f"ROI_{r.role}", "color": ROI_COLORS.get(r.role, ROI_COLOR_OTHER)}
        feats.append({"type": "Feature",
                      "properties": {"name": r.roi_id, "classification": classification,
                                     "stratum": r.stratum, "n_cells": int(r.n_cells)},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


def cmd_propose(a):
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # before the slow part, not after
    labels = cr.open_labels(a.labels)
    cells = cr.centroids_from_labels(labels)
    raster = priority = None
    if a.regions:
        priority = [s.strip() for s in a.priority.split(",") if s.strip()]
        raster = cr.rasterize(cr.load_regions(a.regions), labels.shape, a.downsample, priority)
    excl = None
    if a.exclude:
        # every polygon in the file counts, whatever class it was drawn as
        excl = cr.rasterize([("Exclude", rings) for _, rings in cr.load_regions(a.exclude)],
                            labels.shape, a.downsample, ["Exclude"])
    sel, elig = propose(cells, labels.shape, a.roi_um, a.px_um, a.target_cells, raster, priority,
                        a.downsample, a.focus, a.interface_min, a.min_occupancy, a.min_cells, excl,
                        a.exclude_max, a.validation_frac, a.seed)
    if sel.empty:
        sys.exit("no eligible ROIs: lower --min-occupancy / --min-cells, or check --px-um")
    sel.insert(0, "roi_id", [f"{a.sample_id}_R{i + 1:02d}" for i in range(len(sel))])
    sel.insert(0, "sample_id", a.sample_id)
    sel.to_csv(a.out, index=False)
    gj = pathlib.Path(a.out).with_suffix(".geojson")
    gj.write_text(json.dumps(to_geojson(sel, a.sample_id)))
    print(f"{len(elig)} eligible tiles of {a.roi_um:g} um; chose {len(sel)} "
          f"({int((sel.role == 'train').sum())} train, {int((sel.role == 'validation').sum())} validation), "
          f"~{int(sel.n_cells.sum()):,} cells")
    print(sel.groupby(["stratum", "role"]).size().unstack(fill_value=0).to_string())
    print(f"wrote {a.out} and {gj}")
    print("look before you commit: open the slide in QuPath, then File > Import objects from file... "
          "and pick the GeoJSON (it is data, not a script: do not run it in the script editor)")


# -- crop ---------------------------------------------------------------------------------------

def clean_marker(name):
    """CellTune marker names: letters, digits, underscore only."""
    return re.sub(r"[^A-Za-z0-9_]", "", name)


def write_tif(path, arr):
    import tifffile
    tifffile.imwrite(path, arr)


def crop_one(handle, labels, roi, out_dir, channel_names, region_geo=None, priority=None, px_um=None):
    y0, y1, x0, x1 = int(roi["y0"]), int(roi["y1"]), int(roi["x0"]), int(roi["x1"])
    d = pathlib.Path(out_dir) / roi["roi_id"]
    d.mkdir(parents=True, exist_ok=True)
    for ci, name in enumerate(channel_names):
        write_tif(d / f"{name}.tif", handle.read(ci, 0, y=(y0, y1), x=(x0, x1)))
    lab = np.asarray(labels[y0:y1, x0:x1])
    write_tif(d / "segmentation_labels.tif", lab.astype(np.uint32))
    if region_geo is not None:
        shifted = [(n, [r - np.array([x0, y0]) for r in rings]) for n, rings in region_geo]
        ras = cr.rasterize(shifted, lab.shape, 1, priority)
        for ci, cname in enumerate(priority, start=1):
            m = (ras == ci).astype(np.uint8)
            if m.any():
                write_tif(d / f"{cname}__Mask.tif", m)
    return lab


def cmd_crop(a):
    from akoyalib.imageio import ImageHandle
    h = ImageHandle(a.image, pixel_size_um=a.px_um)
    names = list(h.channel_names)
    if a.marker_map:
        mm = pd.read_csv(a.marker_map)
        lut = dict(zip(mm["Marker_orig"].str.lower(), mm["Marker"]))
        names = [lut.get(n.lower(), clean_marker(n)) for n in names]
    else:
        names = [clean_marker(n) for n in names]
    if len(set(names)) != len(names):
        sys.exit(f"channel names collide after cleaning: {names}")
    labels = cr.open_labels(a.labels)
    if tuple(labels.shape) != tuple(h.level_shape(0)[1:]):
        sys.exit(f"label image {tuple(labels.shape)} and slide {tuple(h.level_shape(0)[1:])} differ in size")
    rois = pd.read_csv(a.rois)
    if a.role != "all":
        rois = rois[rois["role"] == a.role]
    geo = priority = None
    if a.regions:
        geo = cr.load_regions(a.regions)
        priority = [s.strip() for s in a.priority.split(",") if s.strip()]
    # whole-slide area of every cell, once: a cell is "cut" when the crop holds fewer of its pixels
    whole = cr.centroids_from_labels(labels).set_index("cellID")["area_px"]
    rows, border_rows = [], []
    for roi in rois.to_dict("records"):
        lab = crop_one(h, labels, roi, a.out, names, geo, priority, a.px_um)
        ids, cnt = np.unique(lab[lab > 0], return_counts=True)
        cut = ids[cnt < whole.reindex(ids).to_numpy()]
        border_rows += [{"image": roi["roi_id"], "cellID": int(c)} for c in cut]
        rows.append({"image": roi["roi_id"], "sample_id": roi["sample_id"], "x0": int(roi["x0"]),
                     "y0": int(roi["y0"]), "px_um": h.pixel_size_um, "n_cells": int(len(ids)),
                     "n_cut_cells": int(len(cut)), "role": roi["role"], "stratum": roi["stratum"]})
    pd.DataFrame(rows).to_csv(pathlib.Path(a.out) / "roi_map.csv", index=False)
    pd.DataFrame(border_rows, columns=["image", "cellID"]).to_csv(pathlib.Path(a.out) / "cut_cells.csv", index=False)
    print(f"cropped {len(rows)} ROIs x {len(names)} channels into {a.out}")
    print("roi_map.csv holds each ROI's offset; cut_cells.csv lists cells the crop edge slices through "
          "(exclude them from labelling and from the training table).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="choose ROIs")
    p.add_argument("--labels", required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--px-um", type=float, required=True)
    p.add_argument("--roi-um", type=float, default=500.0, help="ROI side in um (default 500, about a MIBI-TOF field)")
    p.add_argument("--target-cells", type=int, default=30000, help="cells to label-pool per slide (default 30,000)")
    p.add_argument("--regions", default=None, help="regions GeoJSON in slide pixel coordinates (from VALIS); enables strata")
    p.add_argument("--priority", default=None, help="comma list of region classes, lowest priority first")
    p.add_argument("--focus", default=None, help="region class whose edge defines the `interface` stratum, e.g. Tumor")
    p.add_argument("--interface-min", type=float, default=0.25)
    p.add_argument("--exclude", default=None, help="GeoJSON of regions to avoid (folds, bubbles), class name irrelevant")
    p.add_argument("--exclude-max", type=float, default=0.05, help="max share of an ROI that may overlap --exclude")
    p.add_argument("--min-occupancy", type=float, default=0.6)
    p.add_argument("--min-cells", type=int, default=300)
    p.add_argument("--validation-frac", type=float, default=0.2)
    p.add_argument("--downsample", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="ROI CSV (a GeoJSON with the same stem is written beside it)")
    p.set_defaults(fn=cmd_propose)
    c = sub.add_parser("crop", help="cut the chosen ROIs into CellTune single-TIFF folders")
    c.add_argument("--image", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--rois", required=True)
    c.add_argument("--out", required=True, help="CellTune_Data/<PROJECT>/Images")
    c.add_argument("--px-um", type=float, default=None, help="override the pixel size read from the file")
    c.add_argument("--marker-map", default=None, help="CSV with Marker_orig,Marker (see panels/celltune_markers_*.csv)")
    c.add_argument("--role", default="all", choices=["all", "train", "validation"])
    c.add_argument("--regions", default=None, help="also write <Class>__Mask.tif per ROI")
    c.add_argument("--priority", default=None)
    c.set_defaults(fn=cmd_crop)
    a = ap.parse_args(argv)
    if a.cmd == "propose" and a.regions and not a.priority:
        sys.exit("--regions needs --priority")
    a.fn(a)


if __name__ == "__main__":
    main()
