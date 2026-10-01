#!/usr/bin/env python
"""Put H&E regions onto every cell, as columns CellTune can import.

The H&E is an adjacent section, so it cannot say what any one cell is. It can
say which part of the tissue a cell sits in (tumor, normal lung, necrosis). This
script turns regions drawn on the H&E, already warped onto the PhenoCycler image
(VALIS, see docs/celltune_he_and_spatial.html step H2), into one row per cell:

    image, cellID, x_px, y_px, region, in_<class>..., signed_um_<focus>, near_boundary

Why the centroid and not "what fraction of the cell lies inside the polygon":
registration between two different sections is off by tens of microns in places,
and a cell is ~10 um across. A fraction computed to the pixel would report
precision the registration does not have. The centroid plus a signed distance to
the region edge is the honest summary, and `near_boundary` marks the cells whose
region call is within the margin you say you can trust (--boundary-um).

Regions come from QuPath's GeoJSON export. Polygons and MultiPolygons are read,
holes are honoured, and the class name is taken from properties.classification.
Where regions overlap, the class listed later in --priority wins. A cell inside no
region is `Unassigned` (not "None": pandas reads the string None as missing).

Status: validated on synthetic data only (tests/test_celltune_regions.py). It
has not been run on a real warped H&E.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


# -- reading regions ----------------------------------------------------------------------------

def _class_name(props):
    c = (props or {}).get("classification")
    if isinstance(c, dict):
        return c.get("name")
    if isinstance(c, str):
        return c
    return (props or {}).get("name")


def load_regions(path):
    """-> list of (class_name, [ring, ring, ...]) with ring = Nx2 array of (x, y) px.

    The first ring is the exterior, any further rings are holes.
    """
    gj = json.loads(pathlib.Path(path).read_text())
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else (
        gj if isinstance(gj, list) else [gj])
    out = []
    for f in feats:
        g, name = f.get("geometry") or {}, _class_name(f.get("properties"))
        if name is None:
            continue
        polys = g["coordinates"] if g.get("type") == "MultiPolygon" else (
            [g["coordinates"]] if g.get("type") == "Polygon" else [])
        for rings in polys:
            out.append((name, [np.asarray(r, float) for r in rings]))
    return out


def rasterize(regions, shape_hw, downsample, priority):
    """Class-index raster at `downsample`-reduced resolution. 0 = no region.

    Index i+1 is priority[i]; later entries in `priority` overwrite earlier ones.
    """
    h, w = int(np.ceil(shape_hw[0] / downsample)), int(np.ceil(shape_hw[1] / downsample))
    raster = np.zeros((h, w), np.int16)
    for ci, cname in enumerate(priority, start=1):
        for name, rings in regions:
            if name != cname:
                continue
            img = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(img)
            for k, ring in enumerate(rings):
                pts = [tuple(p) for p in (ring / downsample)]
                if len(pts) >= 3:
                    d.polygon(pts, fill=255 if k == 0 else 0)
            raster[np.asarray(img) > 0] = ci
    return raster


# -- cell positions -----------------------------------------------------------------------------

def centroids_from_labels(labels, stripe=2048):
    """Centroid (x, y) in pixels and area for every label, without regionprops.

    `labels` is any 2-D integer array that supports slicing (numpy, memmap, zarr),
    read in row stripes so a 33,000 x 30,000 image never has to be one array's
    worth of temporaries. A bincount per stripe is exact.
    """
    h, w = labels.shape
    xs = np.arange(w, dtype=np.float64)
    n = 0
    cnt = sx = sy = None
    for r0 in range(0, h, stripe):
        blk = np.asarray(labels[r0:r0 + stripe]).astype(np.int64, copy=False)
        m = int(blk.max()) + 1
        if cnt is None or m > n:
            new = max(m, 1)
            pad = new - n
            cnt = np.zeros(new) if cnt is None else np.pad(cnt, (0, pad))
            sx = np.zeros(new) if sx is None else np.pad(sx, (0, pad))
            sy = np.zeros(new) if sy is None else np.pad(sy, (0, pad))
            n = new
        flat = blk.ravel()
        cnt[:m] += np.bincount(flat, minlength=m)
        sx[:m] += np.bincount(flat, weights=np.broadcast_to(xs, blk.shape).ravel(), minlength=m)
        rows = (np.arange(blk.shape[0], dtype=np.float64) + r0)[:, None]
        sy[:m] += np.bincount(flat, weights=np.broadcast_to(rows, blk.shape).ravel(), minlength=m)
    ids = np.nonzero(cnt[1:])[0] + 1                     # 0 is background
    return pd.DataFrame({"cellID": ids, "x_px": sx[ids] / cnt[ids], "y_px": sy[ids] / cnt[ids],
                         "area_px": cnt[ids].astype(np.int64)})


def open_labels(path):
    """A sliceable view of a label TIFF: memory-mapped when the file allows it."""
    import tifffile
    try:
        return tifffile.memmap(path)
    except Exception:
        return tifffile.imread(path)


# -- assigning cells to regions -----------------------------------------------------------------

def signed_distance_um(raster, class_index, work_um):
    """+ inside the class, - outside; distance in um to the nearest edge."""
    inside = raster == class_index
    if not inside.any():
        return np.full(raster.shape, -np.inf)
    return (ndi.distance_transform_edt(inside) - ndi.distance_transform_edt(~inside)) * work_um


def assign(cells, raster, priority, downsample, px_um, focus, boundary_um):
    ij_y = np.clip((cells["y_px"].to_numpy() / downsample).astype(int), 0, raster.shape[0] - 1)
    ij_x = np.clip((cells["x_px"].to_numpy() / downsample).astype(int), 0, raster.shape[1] - 1)
    idx = raster[ij_y, ij_x]
    out = cells.copy()
    out["region"] = np.where(idx > 0, np.array(["Unassigned"] + list(priority))[idx], "Unassigned")
    for ci, cname in enumerate(priority, start=1):
        out[f"in_{cname}"] = (idx == ci).astype(np.int8)
    if focus:
        fi = list(priority).index(focus) + 1
        sd = signed_distance_um(raster, fi, downsample * px_um)
        s = sd[ij_y, ij_x]
        out[f"signed_um_{focus}"] = np.round(s, 1)
        out["near_boundary"] = (np.abs(s) < boundary_um).astype(np.int8)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="CellTune label TIFF (0 = background)")
    p.add_argument("--regions", required=True, help="QuPath GeoJSON, already in PhenoCycler pixel coordinates")
    p.add_argument("--sample-id", required=True, help="written to the `image` column; must match the image name in the CellTune project")
    p.add_argument("--px-um", type=float, required=True, help="PhenoCycler microns per pixel (read it from stage 00, do not guess)")
    p.add_argument("--priority", required=True,
                   help="comma-separated class names, lowest priority first, e.g. Normal,Stroma,Tumor,Necrosis")
    p.add_argument("--focus", default="Tumor", help="class to give a signed distance to (default Tumor)")
    p.add_argument("--boundary-um", type=float, default=50.0,
                   help="cells closer than this to the focus edge are flagged near_boundary (default 50, ~ the registration error you should expect to see in the VALIS report)")
    p.add_argument("--downsample", type=int, default=8, help="raster reduction factor (default 8)")
    p.add_argument("--out", required=True, help="output CSV")
    a = p.parse_args(argv)

    priority = [s.strip() for s in a.priority.split(",") if s.strip()]
    if a.focus and a.focus not in priority:
        sys.exit(f"--focus {a.focus!r} is not in --priority {priority}")
    labels = open_labels(a.labels)
    regions = load_regions(a.regions)
    found = sorted({n for n, _ in regions})
    unused = [n for n in found if n not in priority]
    if unused:
        print(f"WARNING: classes in the GeoJSON that --priority does not list, so ignored: {unused}")
    missing = [n for n in priority if n not in found]
    if missing:
        print(f"WARNING: --priority names with no polygon in the GeoJSON: {missing}")

    raster = rasterize(regions, labels.shape, a.downsample, priority)
    cells = centroids_from_labels(labels)
    table = assign(cells, raster, priority, a.downsample, a.px_um, a.focus, a.boundary_um)
    table.insert(0, "image", a.sample_id)
    table.to_csv(a.out, index=False)
    print(f"{len(table):,} cells -> {a.out}")
    print(table["region"].value_counts().to_string())
    if a.focus:
        print(f"near_boundary (|distance to {a.focus} edge| < {a.boundary_um:g} um): "
              f"{int(table['near_boundary'].sum()):,} cells")


if __name__ == "__main__":
    main()
