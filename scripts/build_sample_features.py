#!/usr/bin/env python
"""One row per sample (or ROI), many features, for comparing groups downstream.

Turns the final cell table (cell_types_final.csv from celltune_spatial.py, one per
slide) into a wide feature table. The feature families follow the per-field table in
Liu, Calvet-Mirabent et al. (Angelo lab, HIV lymph nodes, 2026), generalised so none of
it depends on one tissue or panel:

  density      cells of each type per mm^2 of occupied tissue         dens__<type>
  proportion   share of all analysed cells                            prop__<type>
  groups       your own sums of types (lymphoid, myeloid, ...)        dens__<G>, prop__<G>
  ratios       your own A/B between types or groups                   ratio__<name>
  by region    the same densities inside each H&E region              dens__<type>__in_<region>
  diversity    mean Shannon index of the neighbours of each type      div__<type>__r<um>
  distance     mean distance to the k nearest cells of another type   knn<k>_um__<from>__to__<to>
  rings        what surrounds an index cell type, in distance bands   ring__<index>__<lo>-<hi>um__<type>
  enrichment   pairs of types nearer than chance (per unit)           enr_log2oe__A__B, enr_z__A__B
  functional   marker mean / fraction positive within chosen types    fn_mean__<m>__<type>, fn_pos_prop__...

Choices worth knowing about, because they change numbers:
  * Area is the area of occupied 50 um bins, not a tissue mask. Pass --area-bin-um to change it.
  * A type absent from a unit gets density and proportion 0 there (a real measurement). Features that
    need the type to be present (distances, diversity) stay NaN, as does a density in a region the unit
    does not contain.
  * A type with fewer than --min-type-cells in a unit gets NaN for its neighbour-based features.
  * A --group or --ratio name that is not a cell type (or a --group name) stops the run with a list of
    the real types, instead of silently counting 0 cells.
  * A ratio with a zero denominator is NaN, never infinity and never 0. (The published
    notebook fills these with 0, which makes "no denominator" look like "none of the numerator".)
  * Garbage, Ambiguous and Unclassified cells are left out of every feature.

Status: validated on synthetic data only (tests/test_build_sample_features.py).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from akoyalib import spatial as sp           # noqa: E402

DEFAULT_EXCLUDE = ("Garbage", "Ambiguous", "Unclassified")


# -- helpers ------------------------------------------------------------------------------------

def read_any(path):
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, keep_default_na=False, na_values=[""])


def occupied_bins(xy_um, bin_um):
    """Integer bin id (y_bin * W + x_bin) per cell, and the grid width."""
    bx = (xy_um[:, 0] // bin_um).astype(np.int64)
    by = (xy_um[:, 1] // bin_um).astype(np.int64)
    w = int(bx.max()) + 1 if len(bx) else 1
    return by * w + bx, w


def area_mm2(xy_um, bin_um):
    if len(xy_um) == 0:
        return 0.0
    ids, _ = occupied_bins(xy_um, bin_um)
    return len(np.unique(ids)) * (bin_um / 1000.0) ** 2


def region_area_mm2(xy_um, region, bin_um):
    """Occupied-bin area per region; a bin belongs to the region most of its cells carry."""
    ids, _ = occupied_bins(xy_um, bin_um)
    df = pd.DataFrame({"bin": ids, "region": np.asarray(region)})
    maj = df.groupby("bin")["region"].agg(lambda s: s.value_counts().index[0])
    return (maj.value_counts() * (bin_um / 1000.0) ** 2).to_dict()


def parse_group(spec):
    name, rest = spec.split("=", 1)
    return name.strip(), [t.strip() for t in rest.split("+") if t.strip()]


def parse_ratio(spec):
    name, rest = spec.split("=", 1)
    num, den = rest.split("/", 1)
    return name.strip(), [t.strip() for t in num.split("+")], [t.strip() for t in den.split("+")]


def check_names(group_specs, ratio_specs, all_types):
    """Stop on a --group/--ratio name that is neither a cell type nor a --group name.

    A typo, or a marker name such as CD3e where a cell type such as CD8_Tcell was meant, would
    otherwise count as 0 cells: the group comes out as 0 and the ratio as NaN, with no warning.
    """
    groups = dict(parse_group(g) for g in group_specs)
    known = set(all_types)
    bad = [(f"--group {g}", t) for g, members in groups.items() for t in members if t not in known]
    for spec in ratio_specs:
        name, num, den = parse_ratio(spec)
        bad += [(f"--ratio {name}", t) for t in num + den if t not in known and t not in groups]
    if bad:
        lines = "\n".join(f"  {where}: {t!r} is not a cell type" for where, t in bad)
        sys.exit(f"unknown names (use cell types exactly as in cell_types_final.csv, joined with +):\n{lines}\n"
                 f"cell types in these files: {', '.join(all_types)}")


def neighbour_pass(xy_um, codes, n_types, index_idx, radius_um, chunk=40000, edges=None):
    """Stream over index cells; accumulate what is needed, never hold a cells x types matrix for a slide.

    Returns, per index cell type code: summed Shannon index and the number of index cells that had
    a neighbour (for `div`), and, when `edges` is given, summed counts per (index type, ring, type).
    """
    tree = cKDTree(xy_um)
    shannon_sum = np.zeros(n_types)
    shannon_n = np.zeros(n_types)
    n_rings = 0 if edges is None else len(edges) - 1
    ring_sum = np.zeros((n_types, max(n_rings, 1), n_types))
    ring_n = np.zeros(n_types)
    for s in range(0, len(index_idx), chunk):
        ix = index_idx[s:s + chunk]
        sub = cKDTree(xy_um[ix])
        m = sub.sparse_distance_matrix(tree, radius_um, output_type="coo_matrix")
        rows, cols, d = m.row, m.col, m.data
        keep = cols != ix[rows]                               # drop each cell's pair with itself, by identity
        rows, cols, d = rows[keep], cols[keep], d[keep]
        # composition per index cell: count of each neighbour type
        cnt = np.zeros((len(ix), n_types))
        np.add.at(cnt, (rows, codes[cols]), 1)
        tot = cnt.sum(1)
        has = tot > 0
        p = np.where(has[:, None], cnt / np.where(tot[:, None] == 0, 1, tot[:, None]), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
        for c in range(n_types):
            sel = (codes[ix] == c) & has
            shannon_sum[c] += h[sel].sum()
            shannon_n[c] += sel.sum()
        if edges is not None:
            band = np.digitize(d, edges, right=True) - 1      # ring i is (edges[i], edges[i+1]]
            ok = (band >= 0) & (band < n_rings)
            ic = codes[ix][rows[ok]]
            np.add.at(ring_sum, (ic, band[ok], codes[cols[ok]]), 1)
        for c in range(n_types):
            ring_n[c] += (codes[ix] == c).sum()
    return shannon_sum, shannon_n, ring_sum, ring_n


def knn_mean(xy_from, xy_to, k, same):
    kk = k + 1 if same else k
    if len(xy_to) < kk or len(xy_from) == 0:
        return np.nan
    d, _ = cKDTree(xy_to).query(xy_from, k=kk)
    d = d[:, 1:] if same else d
    return float(np.mean(d))


# -- one unit -----------------------------------------------------------------------------------

def unit_features(cells, px_um, a, markers=None, pairs=None, all_types=None):
    """cells: one unit's rows with cell_type, x_px, y_px (and optionally region, cellID).

    `all_types` is every type in the cohort, so a type missing from this unit is reported as 0 cells
    rather than left out (which would leave a blank in the table and drop the unit from the test).
    """
    f = {}
    xy = cells[["x_px", "y_px"]].to_numpy(float) * px_um
    types_all = cells["cell_type"].to_numpy()
    n = len(cells)
    f["n_cells"] = n
    ar = area_mm2(xy, a.area_bin_um)
    f["area_mm2"] = ar
    counts = pd.Series(types_all).value_counts()
    types = [t for t in sorted(counts.index) if counts[t] >= a.min_type_cells]
    allt = sorted(set(all_types) | set(counts.index)) if all_types is not None else sorted(counts.index)

    for t in allt:
        f[f"dens__{t}"] = counts.get(t, 0) / ar if ar > 0 else np.nan
        f[f"prop__{t}"] = counts.get(t, 0) / n if n else np.nan
    groups = dict(parse_group(g) for g in a.group)
    for name, members in groups.items():
        c = int(sum(counts.get(t, 0) for t in members))
        f[f"dens__{name}"] = c / ar if ar > 0 else np.nan
        f[f"prop__{name}"] = c / n if n else np.nan

    def total(members):
        return sum(counts.get(t, 0) if t in counts.index else (groups_count.get(t, 0)) for t in members)
    groups_count = {g: int(sum(counts.get(t, 0) for t in m)) for g, m in groups.items()}
    for spec in a.ratio:
        name, num, den = parse_ratio(spec)
        nu, de = total(num), total(den)
        f[f"ratio__{name}"] = nu / de if de > 0 else np.nan

    if "region" in cells.columns and a.by_region:
        reg = cells["region"].to_numpy()
        rar = region_area_mm2(xy, reg, a.area_bin_um)
        for r in a.by_region:
            ra = rar.get(r, 0.0)
            f[f"area_mm2__in_{r}"] = ra
            m = reg == r
            for t in allt:
                f[f"dens__{t}__in_{r}"] = (int(((types_all == t) & m).sum()) / ra) if ra > 0 else np.nan

    codes_map = {t: i for i, t in enumerate(allt)}
    codes = np.array([codes_map[t] for t in types_all])
    nT = len(allt)

    if a.diversity_um or a.ring_index:
        rng = np.random.default_rng(0)
        idx = np.arange(n)
        if a.max_index_cells and n > a.max_index_cells:
            idx = np.sort(rng.choice(n, a.max_index_cells, replace=False))
        if a.diversity_um:
            sh, shn, _, _ = neighbour_pass(xy, codes, nT, idx, a.diversity_um)
            for t in allt:
                c = codes_map[t]
                f[f"div__{t}__r{a.diversity_um:g}"] = (sh[c] / shn[c]) if (shn[c] > 0 and counts.get(t, 0) >= a.min_type_cells) else np.nan
        if a.ring_index and a.ring_index in codes_map:
            edges = np.arange(0, a.ring_max_um + 1e-9, a.ring_width_um)
            c0 = codes_map[a.ring_index]
            sel = idx[codes[idx] == c0]
            _, _, rs, rn = neighbour_pass(xy, codes, nT, sel, a.ring_max_um, edges=edges)
            for ri in range(len(edges) - 1):
                for t in allt:
                    f[f"ring__{a.ring_index}__{edges[ri]:g}-{edges[ri + 1]:g}um__{t}"] = (
                        rs[c0, ri, codes_map[t]] / rn[c0]) if rn[c0] >= a.min_type_cells else np.nan

    if a.knn:
        for s in types:
            for t in types:
                f[f"knn{a.knn}_um__{s}__to__{t}"] = knn_mean(xy[types_all == s], xy[types_all == t], a.knn, s == t)

    if a.enrichment:
        from celltune_spatial import window_from_cells
        keep = np.isin(types_all, types)
        if keep.sum() > 10 and len(types) > 1:
            win = window_from_cells(cells[["x_px", "y_px"]].to_numpy(float)[keep], px_um)
            res = sp.enrichment(cells[["x_px", "y_px"]].to_numpy(float)[keep], types_all[keep], win,
                                radius_um=a.enrichment_um, n_perm=a.n_perm, px_um=px_um)
            for i, A in enumerate(res["types"]):
                for j, B in enumerate(res["types"]):
                    f[f"enr_log2oe__{A}__{B}"] = res["log2_oe"][i, j]
                    f[f"enr_z__{A}__{B}"] = res["z"][i, j]

    if markers is not None and pairs is not None and "cellID" in cells.columns:
        m = cells[["cellID", "cell_type"]].merge(markers, on="cellID", how="left")
        for mk, ty in pairs:
            if mk not in m.columns:
                continue
            v = m.loc[m["cell_type"] == ty, mk].astype(float)
            ok = counts.get(ty, 0) >= a.min_type_cells
            f[f"fn_mean__{mk}__{ty}"] = float(v.mean()) if ok else np.nan
            if a.pos_threshold is not None:
                f[f"fn_pos_prop__{mk}__{ty}"] = float((v > a.pos_threshold).mean()) if ok else np.nan
                f[f"fn_pos_dens__{mk}__{ty}"] = float((v > a.pos_threshold).sum() / ar) if (ok and ar > 0) else np.nan
    return f


# -- driver -------------------------------------------------------------------------------------

def assign_rois(cells, rois):
    """Rows for each ROI by centroid; a cell can belong to one ROI (they do not overlap)."""
    parts = []
    for r in rois.itertuples():
        m = (cells["x_px"] >= r.x0) & (cells["x_px"] < r.x1) & (cells["y_px"] >= r.y0) & (cells["y_px"] < r.y1)
        sub = cells[m].copy()
        sub["unit"] = r.roi_id
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else cells.iloc[0:0].assign(unit=[])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", required=True, help="cell_types_final.csv, one per slide (need image, cellID, cell_type, x_px, y_px)")
    ap.add_argument("--meta", required=True, help="CSV with sample_id, px_um and any grouping columns (e.g. group)")
    ap.add_argument("--rois", default=None, help="ROI CSV from make_rois.py; if given, the unit is the ROI, else the whole slide")
    ap.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE))
    ap.add_argument("--min-type-cells", type=int, default=10)
    ap.add_argument("--area-bin-um", type=float, default=50.0)
    ap.add_argument("--group", action="append", default=[], help="NAME=TYPE1+TYPE2 (repeatable)")
    ap.add_argument("--ratio", action="append", default=[], help="NAME=A+B/C+D, types or --group names (repeatable)")
    ap.add_argument("--by-region", action="append", default=[], help="region class to report separate densities for (repeatable)")
    ap.add_argument("--diversity-um", type=float, default=None, help="radius for neighbourhood Shannon diversity")
    ap.add_argument("--knn", type=int, default=0, help="k for mean distance to the k nearest cells of each type (0 = off)")
    ap.add_argument("--ring-index", default=None, help="index cell type for distance-band composition")
    ap.add_argument("--ring-width-um", type=float, default=10.0)
    ap.add_argument("--ring-max-um", type=float, default=100.0)
    ap.add_argument("--max-index-cells", type=int, default=0, help="subsample index cells for neighbour features (0 = all)")
    ap.add_argument("--enrichment", action="store_true", help="per-unit pairwise enrichment (slow)")
    ap.add_argument("--enrichment-um", type=float, default=20.0)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--markers", default=None, help="CSV: image, cellID, marker columns (positivity probability or intensity)")
    ap.add_argument("--pairs", default=None, help="CSV: marker, cell_type -- which marker/type combinations are biologically meaningful")
    ap.add_argument("--pos-threshold", type=float, default=None, help="call a cell positive above this value")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    meta = pd.read_csv(a.meta)
    cells = pd.concat([read_any(p) for p in a.cells], ignore_index=True)
    cells = cells.rename(columns={"image": "sample_id"}) if "image" in cells.columns else cells
    excl = {s.strip() for s in a.exclude.split(",") if s.strip()}
    cells = cells[~cells["cell_type"].isin(excl)].reset_index(drop=True)
    all_types = sorted(cells["cell_type"].unique())
    check_names(a.group, a.ratio, all_types)
    rois = pd.read_csv(a.rois) if a.rois else None
    markers = read_any(a.markers) if a.markers else None
    pairs = list(pd.read_csv(a.pairs)[["marker", "cell_type"]].itertuples(index=False, name=None)) if a.pairs else None
    if markers is not None and "image" in markers.columns:
        markers = markers.rename(columns={"image": "sample_id"})

    rows = []
    for sid, g in cells.groupby("sample_id"):
        px = float(meta.loc[meta["sample_id"] == sid, "px_um"].iloc[0]) if "px_um" in meta.columns else None
        if px is None:
            sys.exit("--meta needs a px_um column (microns per pixel for each sample)")
        mk = markers[markers["sample_id"] == sid] if markers is not None and "sample_id" in markers.columns else markers
        if rois is not None:
            gu = assign_rois(g, rois[rois["sample_id"] == sid])
            units = list(gu.groupby("unit"))
        else:
            units = [(sid, g)]
        for uid, ug in units:
            feats = unit_features(ug, px, a, mk, pairs, all_types)
            feats = {"unit_id": uid, "sample_id": sid, **feats}
            rows.append(feats)
    out = pd.DataFrame(rows)
    out = out.merge(meta.drop(columns=[c for c in ("px_um",) if c in meta.columns]), on="sample_id", how="left")
    out.to_csv(a.out, index=False)
    fam = pd.Series([c.split("__")[0] if "__" in c else "meta" for c in out.columns]).value_counts()
    print(f"{len(out)} units x {out.shape[1]} columns -> {a.out}")
    print(fam.to_string())


if __name__ == "__main__":
    main()
