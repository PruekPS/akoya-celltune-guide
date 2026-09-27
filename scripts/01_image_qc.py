"""Stage 01: image QC — tissue mask, artifact flags, per-channel signal and background.

What it does, per sample:
  1. Finds tissue from the nuclear stain at reduced resolution, keeping real
     air spaces (e.g. lung alveoli) as holes instead of filling them.
  2. Marks areas with no image data (exact zeros in every channel) and any
     exclusion polygons you drew in QuPath (GeoJSON).
  3. Splits tissue into square tiles and flags, by robust outlier statistics
     across the sample's own tiles:
       - possibly out of focus (low nuclear-stain sharpness)
       - possible fold or overlap (unusually bright nuclear stain)
       - possible antibody aggregate in a specific channel (bright outlier)
  4. For every channel, measures in-tissue signal against off-tissue
     background with exact histograms, and screens for the background
     problems PhenoCycler-Fusion's built-in subtraction can cause (Karnik et
     al., Leukemia 2025): many zeros inside tissue (possible over-subtraction)
     or bright off-tissue background (possible under-subtraction), and uneven
     background between areas (possible tile seams).
  5. Sets one display range per channel by rule (0.5th-99.9th percentile of
     in-tissue pixels), pooled across every sample QC'd so far, and writes a
     QuPath script that applies those same ranges -- so nobody tunes
     brightness by eye. Display ranges never change pixel values or any
     measurement; they only make images comparable between slides.

Tile flags are screening signals for a person to review in the report, not
verdicts. Focus and fold flags remove those tiles from the analysis mask used
by stage 02 (switch off with --keep-flagged); aggregate flags do not, because
they affect one channel, not the cells.

Usage:
    python scripts/01_image_qc.py --samples tests/spacec_demo/samples.csv --out results
    python scripts/01_image_qc.py --samples samples.csv --tile-um 200 --keep-flagged

Outputs per sample (<out>/<sample>/01_image_qc/):
    qc.json               summary + analysis-mask geometry, read by stage 02
    analysis_mask.npz     tissue minus no-data, exclusions and flagged tiles (reduced resolution)
    tiles.csv             per-tile metrics and flags
    channel_qc.csv        per-channel signal/background table
    tissue_regions.csv    connected tissue pieces (e.g. TMA cores)
    display_ranges.csv    this sample's in-tissue display ranges
and across samples (<out>/_batch/): display_ranges.csv, qupath_display_ranges.groovy
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import draw, filters, measure, morphology

from akoyalib import imageio, report, runs

STAGE = "01_image_qc"
STAGE_STATUS = "smoke-tested only"
DISPLAY_PCT = (0.5, 99.9)


def robust_z(v):
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    return np.zeros_like(v) if not mad else (v - med) / mad


def rasterize_geojson(path, shape, f):
    """QuPath GeoJSON (full-resolution pixel coordinates) -> bool mask at reduced resolution."""
    mask = np.zeros(shape, bool)
    data = json.loads(pathlib.Path(path).read_text())
    feats = data["features"] if isinstance(data, dict) and "features" in data else \
        (data if isinstance(data, list) else [data])
    n = 0
    for feat in feats:
        geom = feat.get("geometry", feat)
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else \
            [geom["coordinates"]] if geom["type"] == "Polygon" else []
        for poly in polys:
            for k, ring in enumerate(poly):
                ring = np.asarray(ring, float) / f
                rr, cc = draw.polygon(ring[:, 1], ring[:, 0], shape)
                mask[rr, cc] = (k == 0)  # outer ring adds, holes remove
            n += 1
    return mask, n


def tissue_mask(dapi, px_work, args):
    sm = filters.gaussian(np.log1p(dapi), sigma=args.smooth_um / px_work, preserve_range=True)
    valid = sm[dapi > 0]
    thr = filters.threshold_otsu(valid if valid.size else sm)
    mask = sm > thr
    mask = morphology.remove_small_objects(mask, max_size=int(args.min_tissue_um2 / px_work ** 2))
    holes = ndi.binary_fill_holes(mask) & ~mask
    small_holes = morphology.remove_small_objects(holes, max_size=int(args.fill_holes_um2 / px_work ** 2))
    mask |= holes & ~small_holes  # fill only holes smaller than the limit
    return mask, float(np.expm1(thr))


def qc_sample(row, args):
    sid = row["sample_id"]
    ingest = runs.require_stage(args.out, sid, "00_ingest", "ingest.json")
    if not ingest["ok_to_continue"]:
        sys.exit(f"[{sid}] stage 00 recorded FAIL checks; fix them (see reports/00_ingest.html) first.")
    channels = pd.read_csv(pathlib.Path(args.out) / sid / "00_ingest" / "channels.csv", keep_default_na=False)
    out_dir = runs.stage_dir(args.out, sid, STAGE)
    rep = report.Report("01", "Image QC", sid, STAGE_STATUS, args.run_note)
    handle = imageio.ImageHandle(ingest["image"], channel_names_file=ingest["channel_names_file"] or None,
                                 pixel_size_um=ingest["pixel_size_um"])
    px = ingest["pixel_size_um"]
    C, H, W = handle.level_shape(0)
    nuc = channels[(channels["segmentation"] == "nuclear") & (channels["status"] == "ok")].iloc[0]
    nuc_idx = int(nuc["index"])

    # -- reduced-resolution images: nuclear stain, no-data mask ----------------------------
    print(f"[{sid}] reduced-resolution pass over {C} channels")
    dapi_w, f = handle.read_downsampled(nuc_idx, args.work_side)
    px_w = px * f
    allzero = np.ones(dapi_w.shape, bool)
    for c in range(C):
        img, _ = handle.read_downsampled(c, args.work_side)
        allzero &= img == 0
    nodata = morphology.remove_small_objects(allzero, max_size=int(2500 / px_w ** 2))

    tissue, dapi_thr = tissue_mask(dapi_w, px_w, args)
    tissue &= ~nodata
    excl = np.zeros_like(tissue)
    n_poly = 0
    if row["exclusions"]:
        excl, n_poly = rasterize_geojson(row["exclusions"], tissue.shape, f)
        rep.check("info", f"Applied {n_poly} exclusion polygon(s) from {pathlib.Path(row['exclusions']).name}.")

    regions = measure.label(tissue)
    reg_rows = [{"region": p.label, "area_mm2": p.area * px_w ** 2 / 1e6,
                 "centroid_x_um": p.centroid[1] * px_w, "centroid_y_um": p.centroid[0] * px_w,
                 "bbox_um": [round(v * px_w) for v in p.bbox]} for p in measure.regionprops(regions)]
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(out_dir / "tissue_regions.csv", index=False)
    tissue_mm2 = tissue.sum() * px_w ** 2 / 1e6
    rep.check("ok" if tissue_mm2 > 0 else "fail",
              f"Tissue found: {tissue_mm2:.2f} mm² in {len(reg_df)} separate piece(s).")
    if nodata.any():
        rep.check("info", f"{nodata.sum() * px_w ** 2 / 1e6:.2f} mm² of the image holds no data "
                          f"(every channel exactly zero) and is excluded.")

    # -- full-resolution tile pass --------------------------------------------------------
    tile = max(32, int(round(args.tile_um / px)))
    ny, nx = int(np.ceil(H / tile)), int(np.ceil(W / tile))
    rows_map = np.minimum((np.arange(H) / f).astype(int), tissue.shape[0] - 1)
    cols_map = np.minimum((np.arange(W) / f).astype(int), tissue.shape[1] - 1)
    tile_tissue = np.zeros((ny, nx))
    tile_nodata = np.zeros((ny, nx))
    for ty in range(ny):
        rr = rows_map[ty * tile:(ty + 1) * tile]
        for tx in range(nx):
            cc = cols_map[tx * tile:(tx + 1) * tile]
            tile_tissue[ty, tx] = tissue[np.ix_(rr, cc)].mean()
            tile_nodata[ty, tx] = nodata[np.ix_(rr, cc)].mean()
    is_tissue_tile = tile_tissue >= 0.5
    is_bg_tile = (tile_tissue <= 0.02) & (tile_nodata <= 0.02)

    nb = runs.n_bins_for(handle.dtype)
    hist_in = np.zeros((C, nb), np.int64)
    hist_out = np.zeros((C, nb), np.int64)
    p999 = np.full((C, ny, nx), np.nan)
    med = np.full((C, ny, nx), np.nan)
    focus = np.full((ny, nx), np.nan)
    background_w = ~ndi.binary_dilation(tissue, iterations=max(1, int(20 / px_w))) & ~nodata  # >=20 µm off tissue
    print(f"[{sid}] full-resolution tile pass: {ny}×{nx} tiles of {tile}px ({args.tile_um} µm)")
    for c in range(C):
        for y0, y1, band in handle.iter_bands(c, tile):
            ty = y0 // tile
            band_mask = tissue[np.ix_(rows_map[y0:y1], cols_map)]
            band_bg = background_w[np.ix_(rows_map[y0:y1], cols_map)]
            runs.add_to_hist(hist_in[c], band[band_mask])
            runs.add_to_hist(hist_out[c], band[band_bg])
            for tx in range(nx):
                t = band[:, tx * tile:(tx + 1) * tile]
                if is_tissue_tile[ty, tx] or is_bg_tile[ty, tx]:
                    p999[c, ty, tx] = np.percentile(t, 99.9)
                    med[c, ty, tx] = np.median(t)
                if c == nuc_idx and is_tissue_tile[ty, tx]:
                    tf = t.astype(np.float32)
                    scale = max(np.percentile(tf, 99) - np.percentile(tf, 1), 1.0)
                    focus[ty, tx] = np.log10(np.var(filters.laplace(filters.gaussian(tf / scale, 1))) + 1e-12)

    # -- tile flags -------------------------------------------------------------------------
    tis = is_tissue_tile
    z_focus = np.full_like(focus, np.nan)
    z_focus[tis] = robust_z(focus[tis])
    z_dapi = np.full_like(focus, np.nan)
    z_dapi[tis] = robust_z(med[nuc_idx][tis])
    flag_focus = tis & (z_focus < -args.z_flag)
    flag_fold = tis & (z_dapi > args.z_flag)
    usable = channels["status"].eq("ok").to_numpy()
    agg = {}
    for c in range(C):
        if not usable[c] or c == nuc_idx:
            continue
        v = np.log1p(p999[c][tis])
        z = robust_z(v)
        tissue_p999 = runs.hist_percentiles(hist_in[c], [99.9])[0]
        hits = np.argwhere(tis)[(z > args.z_aggregate) & (np.expm1(v) > tissue_p999)]
        if len(hits):
            agg[c] = [tuple(h) for h in hits]

    tile_rows = []
    for ty in range(ny):
        for tx in range(nx):
            if not (tis[ty, tx] or is_bg_tile[ty, tx]):
                continue
            reasons = []
            if flag_focus[ty, tx]:
                reasons.append("focus")
            if flag_fold[ty, tx]:
                reasons.append("fold")
            aggs = [channels.loc[c, "marker"] for c, h in agg.items() if (ty, tx) in h]
            tile_rows.append({"tile_y": ty, "tile_x": tx, "x0_px": tx * tile, "y0_px": ty * tile, "size_px": tile,
                              "kind": "tissue" if tis[ty, tx] else "background",
                              "tissue_fraction": round(tile_tissue[ty, tx], 3),
                              "focus_log10": focus[ty, tx], "focus_z": z_focus[ty, tx],
                              "nuclear_median": med[nuc_idx, ty, tx], "nuclear_z": z_dapi[ty, tx],
                              "flags": ";".join(reasons), "aggregate_channels": ";".join(aggs)})
    tiles_df = pd.DataFrame(tile_rows)
    tiles_df.to_csv(out_dir / "tiles.csv", index=False)
    n_tis = int(tis.sum())
    for name, fl, what in (("focus", flag_focus, "possibly out of focus"), ("fold", flag_fold, "possible fold or overlap")):
        k = int(fl.sum())
        rep.check("warn" if k > 0.05 * n_tis else "ok" if k == 0 else "info",
                  f"{k} of {n_tis} tissue tiles ({100 * k / max(n_tis, 1):.1f}%) flagged {what}.")

    # -- analysis mask for stage 02 -----------------------------------------------------------
    analysis = tissue & ~excl
    if not args.keep_flagged:
        for ty, tx in np.argwhere(flag_focus | flag_fold):
            y0, y1 = int(ty * tile / f), int(min((ty + 1) * tile, H) / f) + 1
            x0, x1 = int(tx * tile / f), int(min((tx + 1) * tile, W) / f) + 1
            analysis[y0:y1, x0:x1] = False
    np.savez_compressed(out_dir / "analysis_mask.npz", mask=analysis, tissue=tissue, nodata=nodata,
                        exclusions=excl, downsample=f, regions=regions.astype(np.int32))
    analysis_mm2 = analysis.sum() * px_w ** 2 / 1e6
    rep.check("info", f"Analysis area passed to stage 02: {analysis_mm2:.2f} mm² "
                      f"({100 * analysis_mm2 / max(tissue_mm2, 1e-9):.1f}% of tissue).")

    # -- per-channel signal / background ----------------------------------------------------------
    ch_rows = []
    seam_vals = {}
    for c in range(C):
        tin = runs.hist_percentiles(hist_in[c], [0.5, 1, 50, 99, 99.9])
        bg = runs.hist_percentiles(hist_out[c], [50, 99])
        n_in = hist_in[c].sum()
        bg_tiles = med[c][is_bg_tile]
        bg_tiles = bg_tiles[np.isfinite(bg_tiles)]
        seam = float(np.percentile(bg_tiles, 90) - np.percentile(bg_tiles, 10)) if bg_tiles.size >= 10 else np.nan
        seam_vals[c] = seam
        sbr = tin[3] / max(bg[0], 1.0)
        flags = []
        # Compare the bright tail inside tissue against the background's OWN upper tail,
        # not against an absolute step in grey levels. The earlier test scaled "3 grey
        # levels at 8-bit" by the nominal bit depth, giving 768 counts for a 16-bit file
        # -- but a PhenoCycler qptiff uses only a few percent of that range (PS88's DAPI
        # p99.9 is 4,629 of 65,535), so 768 became a large signal threshold rather than a
        # negligible one. It called CD8, PD-L1 and PD-1 absent on PS88 when CD8's p99.9
        # of 428 sits 16x above its background p99 of 26.
        bg_hi = bg[1]                              # background p99
        bg_noise = max(bg_hi - bg[0], 1.0)         # the background's own spread
        zero_pct_in_tissue = 100 * hist_in[c][0] / max(n_in, 1)
        if usable[c] and c != nuc_idx:
            # p99.9, not p99: rare-cell markers (e.g. neutrophil CD15) stain <1% of pixels,
            # so their p99 sits at background even when the stain works.
            if tin[4] <= bg_hi + bg_noise:
                flags.append("no usable signal (tissue p99.9 no higher than background p99 plus its spread)")
            else:
                # A high zero fraction on its own is biology, not a fault: Ki67 marks only
                # cycling cells, CD8 only a rare subset, and on PS88 those two plus CD45
                # were the three highest simply because of what they stain. The signature
                # of over-subtraction is that the BACKGROUND has been clipped away too --
                # not just its median but its whole distribution, p99 included. A channel
                # whose off-tissue p99 is still above zero had a background left to it,
                # so its in-tissue zeros are where the marker is absent.
                if zero_pct_in_tissue > args.zero_pct and bg_hi <= 0:
                    flags.append("many zeros in tissue")
                if bg[0] >= 0.5 * max(tin[2], 1) and bg[0] >= 0.1 * tin[3]:
                    flags.append("bright off-tissue background")
                if sbr < 2:
                    flags.append("weak (p99 < 2× background)")
            if np.isfinite(seam) and seam > max(2.0, 0.25 * max(tin[3] - tin[2], 1)):
                flags.append("uneven background (possible seams)")
            if 100 * hist_in[c][nb - 1] / max(n_in, 1) > 0.1:
                flags.append("saturated >0.1%")
        ch_rows.append({"index": c, "marker": channels.loc[c, "marker"], "status": channels.loc[c, "status"],
                        "role": channels.loc[c, "role"], "tissue_p1": tin[1], "tissue_p50": tin[2],
                        "tissue_p99": tin[3], "tissue_p99_9": tin[4], "background_p50": bg[0],
                        "background_p99": bg[1], "signal_over_background_tail": tin[4] / max(bg[1], 1.0),
                        "p99_over_background": sbr, "pct_zero_in_tissue": zero_pct_in_tissue,
                        "background_spread_between_tiles": seam, "aggregate_tiles": len(agg.get(c, [])),
                        "display_low": tin[0], "display_high": max(tin[4], tin[0] + 1),
                        "flags": "; ".join(flags)})
    ch_df = pd.DataFrame(ch_rows)
    ch_df.to_csv(out_dir / "channel_qc.csv", index=False)
    ch_df[["index", "marker", "display_low", "display_high"]].to_csv(out_dir / "display_ranges.csv", index=False)
    np.savez_compressed(out_dir / "tissue_histograms.npz", hist_in=hist_in, hist_out=hist_out)

    for label, key in (("no usable signal — the stain looks absent or failed in this tissue", "no usable"),
                       ("many zeros inside tissue AND a background floored at zero — the pattern "
                        "over-subtraction leaves (Karnik et al. 2025). A high zero fraction on its own is "
                        "normal for a sparse marker and is not flagged",
                        "many zeros"),
                       ("off-tissue background at least half the tissue median — possible under-subtraction",
                        "bright off-tissue"),
                       ("weak signal: 99th percentile in tissue under twice the background", "weak"),
                       ("uneven background between areas — possible tile seams", "uneven"),
                       ("more than 0.1% saturated pixels inside tissue", "saturated")):
        hit = ch_df[ch_df["flags"].str.contains(key)]
        if len(hit):
            rep.check("warn", f"{len(hit)} channel(s) with {label}: {', '.join(hit['marker'])}.")
    if agg:
        rep.check("info", f"Possible aggregates (bright outlier tiles) in {len(agg)} channel(s): " +
                  ", ".join(f"{channels.loc[c, 'marker']} ({len(v)})" for c, v in
                            sorted(agg.items(), key=lambda kv: -len(kv[1]))[:12]) + ". Review the gallery.")

    qc = {"sample_id": sid, "pixel_size_um": px, "downsample": f, "work_pixel_um": px_w, "tile_px": tile,
          "tile_um": args.tile_um, "tissue_mm2": tissue_mm2, "analysis_mm2": analysis_mm2,
          "n_regions": len(reg_df), "nuclear_channel": nuc["marker"], "nuclear_index": nuc_idx,
          "dapi_threshold": dapi_thr, "tiles_flagged_focus": int(flag_focus.sum()),
          "tiles_flagged_fold": int(flag_fold.sum()), "flagged_tiles_excluded": not args.keep_flagged,
          "exclusion_polygons": n_poly}
    runs.write_json(out_dir / "qc.json", qc)

    # -- batch display ranges + QuPath script ---------------------------------------------------------
    batch = update_batch_display_ranges(args.out)

    # -- report ---------------------------------------------------------------------------------------
    lo_d, hi_d = ch_df.loc[nuc_idx, ["display_low", "display_high"]]
    fig, ax = plt.subplots(figsize=(10, 10 * dapi_w.shape[0] / dapi_w.shape[1] + 0.4))
    ax.imshow(imageio.to_display(dapi_w, lo_d, hi_d), cmap="gray")
    ax.contour(tissue, levels=[0.5], colors=[report.PALETTE["if"]], linewidths=0.9)
    if excl.any():
        ax.contour(excl, levels=[0.5], colors=[report.PALETTE["he"]], linewidths=0.9)
    for p in measure.regionprops(regions):
        ax.text(p.centroid[1], p.centroid[0], str(p.label), color="#FFD166", fontsize=12, ha="center", weight="bold")
    for fl, col in ((flag_focus, "#F4A261"), (flag_fold, "#E63946")):
        for ty, tx in np.argwhere(fl):
            ax.add_patch(mpatches.Rectangle((tx * tile / f, ty * tile / f), tile / f, tile / f,
                                            fill=False, ec=col, lw=1.2))
    ax.set_axis_off()
    ax.legend(handles=[mpatches.Patch(ec=report.PALETTE["if"], fc="none", label="tissue edge"),
                       mpatches.Patch(ec="#F4A261", fc="none", label="flag: focus"),
                       mpatches.Patch(ec="#E63946", fc="none", label="flag: fold/overlap")],
              loc="lower right", fontsize=8, framealpha=0.85)
    rep.h2("Tissue and flagged areas")
    rep.explain(f"Tissue is found from the nuclear stain ({nuc['marker']}) at {px_w:.2g} µm per pixel: smoothed "
                f"({args.smooth_um} µm), thresholded automatically (Otsu), pieces under "
                f"{args.min_tissue_um2:,.0f} µm² dropped, and only holes under {args.fill_holes_um2:,.0f} µm² filled, "
                f"so real air spaces such as lung alveoli stay outside the mask. Tiles are {args.tile_um} µm squares; "
                f"a tile is flagged when it sits more than {args.z_flag} robust standard deviations from this "
                f"sample's typical tile.")
    rep.figure(fig, f"Nuclear stain with tissue outline (blue), tissue pieces numbered, flagged tiles boxed. "
                    f"{len(reg_df)} piece(s), {tissue_mm2:.2f} mm².")
    if len(reg_df):
        rep.table(reg_df.round(3), caption="Tissue pieces. For a TMA each piece is usually one core.")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    for a, vals, thr, lab in ((axs[0], z_focus[tis], -args.z_flag, "sharpness (robust z)"),
                              (axs[1], z_dapi[tis], args.z_flag, "nuclear-stain brightness (robust z)")):
        a.hist(vals[np.isfinite(vals)], bins=40, color=report.PALETTE["if"])
        a.axvline(thr, color=report.PALETTE["fail"], ls="--", lw=1)
        a.set_xlabel(lab), a.set_ylabel("tissue tiles")
    fig.tight_layout()
    rep.figure(fig, "Distribution of the two tile scores across this sample's tissue tiles; dashed lines are the "
                    "flag thresholds. Sharpness is the variance of the Laplacian (edge strength) of the nuclear "
                    "stain after contrast normalisation.")
    gallery = tiles_df[tiles_df["flags"] != ""].copy()
    if len(gallery):
        gallery["sev"] = np.nanmax(np.abs(gallery[["focus_z", "nuclear_z"]].to_numpy()), axis=1)
        gallery = gallery.sort_values("sev", ascending=False).head(8)
        fig, axs = plt.subplots(1, len(gallery), figsize=(2.2 * len(gallery), 2.6))
        for a, (_, t) in zip(np.atleast_1d(axs), gallery.iterrows()):
            crop = handle.read(nuc_idx, 0, y=(t.y0_px, t.y0_px + tile), x=(t.x0_px, t.x0_px + tile))
            a.imshow(imageio.to_display(crop, lo_d, hi_d), cmap="gray")
            a.set_title(f"{t.flags}\nfocus z={t.focus_z:.1f}, bright z={t.nuclear_z:.1f}", fontsize=7.5)
            a.set_axis_off()
        fig.tight_layout()
        rep.figure(fig, "The most extreme flagged tiles at full resolution (nuclear stain). Judge whether each flag "
                        "is a real artifact or tissue that simply looks different (e.g. a dense follicle).")

    rep.h2("Signal and background, channel by channel")
    rep.explain("All values are exact, from every full-resolution pixel. <strong>tissue_p50 / p99</strong>: median "
                "and 99th percentile inside tissue. <strong>background_p50</strong>: median off tissue (at least "
                "20 µm from any tissue edge). <strong>p99_over_background</strong>: a simple signal-to-background "
                "ratio. <strong>pct_zero_in_tissue</strong>: share of tissue pixels at exactly zero; high values can "
                "mean background subtraction removed real signal, or that the marker is truly absent from this "
                "tissue, which only the images can tell apart. <strong>background_spread_between_tiles</strong>: "
                "difference between the 90th and 10th percentile of off-tissue tile medians; large values suggest "
                "tile seams or uneven illumination. This seam measure is <em>not yet shown to work</em>: seams "
                "visible in a nearly empty channel can be single grey-level steps that only appear because the "
                "display is stretched, and it has not yet been tested on a slide with real seams. "
                "Flags are screening signals, not verdicts.")
    show = ch_df.drop(columns=["display_low", "display_high"]).copy()
    show["_o"] = show["flags"].eq("")
    rep.table(show.sort_values(["_o", "index"]).drop(columns="_o"), floatfmt="{:,.3g}")
    if agg:
        pairs = sorted(((c, h) for c, hs in agg.items() for h in hs),
                       key=lambda ch: -p999[ch[0], ch[1][0], ch[1][1]])[:8]
        fig, axs = plt.subplots(1, len(pairs), figsize=(2.2 * len(pairs), 2.6))
        for a, (c, (ty, tx)) in zip(np.atleast_1d(axs), pairs):
            crop = handle.read(c, 0, y=(ty * tile, (ty + 1) * tile), x=(tx * tile, (tx + 1) * tile))
            a.imshow(imageio.to_display(crop, *ch_df.loc[c, ["display_low", "display_high"]]), cmap="magma")
            a.set_title(f"{channels.loc[c, 'marker']}\ntile ({ty},{tx})", fontsize=7.5)
            a.set_axis_off()
        fig.tight_layout()
        rep.figure(fig, "Brightest outlier tiles, per channel, at full resolution. Tiny intense specks with no cell "
                        "shape suggest antibody aggregates; cell-shaped signal is probably real biology (e.g. a "
                        "cluster of rare cells).")

    rep.h2("Display ranges: set by rule, not by eye")
    rep.explain("Brightness/contrast only changes how an image is <em>shown</em>, never its pixel values, so "
                "adjusting it in QuPath cannot change any measurement. Bias enters where a person decides "
                "something from what they see: where a threshold goes, which cells count as positive, which cells "
                "are clicked to train a classifier. So the pipeline shows every channel with one rule: its 0.5th to "
                "99.9th percentile of in-tissue pixels, pooled across every sample QC'd into this output folder. "
                "Nothing in the analysis is decided by eye from these images; they are for checking. "
                f"The same ranges are written as a QuPath script, <code>{batch['groovy']}</code>, so your QuPath view "
                "matches the reports (display only; not yet tested inside QuPath).")
    order = [c for c in range(C) if usable[c]]
    rep.figure(display_sheet(handle, ch_df, order, batch["ranges"]),
               f"Every usable channel with the pooled display range ({batch['n_samples']} sample(s) pooled so far).")
    rep.set_provenance(
        inputs={"image": f"{ingest['image']} (sha256 {ingest['sha256']})", "stage 00": "00_ingest/ingest.json",
                "exclusions": row["exclusions"] or "(none)"},
        params={k: v for k, v in vars(args).items() if k not in ("samples", "out", "sample_id")},
        outputs={"qc.json": out_dir / "qc.json", "analysis_mask.npz": out_dir / "analysis_mask.npz",
                 "channel_qc.csv": out_dir / "channel_qc.csv", "tiles.csv": out_dir / "tiles.csv",
                 "batch display ranges": batch["csv"], "QuPath script": batch["groovy"]},
        repo_dir=runs.REPO_DIR)
    path = rep.write(runs.report_path(args.out, sid, STAGE))
    handle.close()
    print(f"[{sid}] report: {path}")


def display_sheet(handle, ch_df, order, ranges, per_row=8, side=320):
    import math
    rows = math.ceil(len(order) / per_row)
    _, h, w = handle.level_shape(0)
    fig, axes = plt.subplots(rows, per_row, figsize=(per_row * 1.9, rows * (1.9 * h / w + 0.35)))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, order):
        thumb, _ = handle.read_downsampled(c, side)
        m = ch_df.loc[c, "marker"]
        lo, hi = ranges.get(m, (ch_df.loc[c, "display_low"], ch_df.loc[c, "display_high"]))
        ax.imshow(imageio.to_display(thumb, lo, hi), cmap="gray")
        ax.set_title(f"{m}  [{lo:.0f}–{hi:.0f}]", fontsize=7.5)
        ax.set_axis_off()
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


def update_batch_display_ranges(out):
    """Pool in-tissue histograms of every sample QC'd under <out> and rewrite the batch ranges."""
    out = pathlib.Path(out)
    pooled, n = {}, 0
    for qc_dir in sorted(out.glob(f"*/{STAGE}")):
        hist_file, ch_file = qc_dir / "tissue_histograms.npz", qc_dir / "channel_qc.csv"
        if not (hist_file.exists() and ch_file.exists()):
            continue
        hist_in = np.load(hist_file)["hist_in"]
        ch = pd.read_csv(ch_file, keep_default_na=False)
        for c, m in zip(ch["index"], ch["marker"]):
            if not m:
                continue
            h = hist_in[c]
            if m in pooled and pooled[m].size != h.size:
                continue  # mixed bit depths: keep the first
            pooled[m] = pooled.get(m, np.zeros_like(h)) + h
        n += 1
    ranges = {}
    for m, h in pooled.items():
        lo, hi = runs.hist_percentiles(h, list(DISPLAY_PCT))
        ranges[m] = (lo, max(hi, lo + 1))
    bdir = out / "_batch"
    bdir.mkdir(parents=True, exist_ok=True)
    csv = bdir / "display_ranges.csv"
    pd.DataFrame([{"marker": m, "display_low": lo, "display_high": hi} for m, (lo, hi) in ranges.items()]) \
        .to_csv(csv, index=False)
    groovy = bdir / "qupath_display_ranges.groovy"
    lines = ["// Generated by scripts/01_image_qc.py - display ranges set by rule "
             f"({DISPLAY_PCT[0]}th-{DISPLAY_PCT[1]}th percentile of in-tissue pixels, pooled over {n} sample(s)).",
             "// Changes only how channels are displayed in QuPath; pixel values and measurements are untouched.",
             "// QuPath does not save display ranges with the project: run this each session (Automate > Script editor).",
             "// Channel names must match the names QuPath shows; for .qptiff files QuPath may show filter names,",
             "// in which case rename channels first (Akoya's guide, section 3.1) or edit the names below.",
             "// Not yet tested inside QuPath."]
    lines += [f"setChannelDisplayRange('{m}', {lo:.0f}, {hi:.0f})" for m, (lo, hi) in ranges.items()]
    groovy.write_text("\n".join(lines) + "\n")
    return {"ranges": ranges, "csv": csv, "groovy": groovy, "n_samples": n}


def main():
    p = runs.base_parser(__doc__)
    p.add_argument("--work-side", type=int, default=4096, help="longer side (px) of the reduced-resolution image")
    p.add_argument("--smooth-um", type=float, default=5.0, help="smoothing before thresholding tissue (µm)")
    p.add_argument("--min-tissue-um2", type=float, default=20000, help="drop tissue pieces smaller than this (µm²)")
    p.add_argument("--fill-holes-um2", type=float, default=2000,
                   help="fill holes smaller than this (µm²); larger holes, e.g. alveoli, stay out")
    p.add_argument("--tile-um", type=float, default=200.0, help="QC tile size (µm)")
    p.add_argument("--z-flag", type=float, default=3.5, help="robust z threshold for focus/fold flags")
    p.add_argument("--z-aggregate", type=float, default=5.0, help="robust z threshold for aggregate flags")
    p.add_argument("--zero-pct", type=float, default=50.0,
                   help="flag a channel when more than this %% of tissue pixels are exactly zero")
    p.add_argument("--keep-flagged", action="store_true",
                   help="keep focus/fold-flagged tiles in the analysis mask (default: exclude them)")
    args = p.parse_args()
    for _, row in runs.select_samples(args).iterrows():
        qc_sample(row, args)


if __name__ == "__main__":
    main()
