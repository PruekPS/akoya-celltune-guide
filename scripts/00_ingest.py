"""Stage 00: ingest and validate one multiplexed image per sample.

Opens the image lazily (never loads a whole slide), resolves channel names
and pixel size, matches every image channel to the sample's panel sheet,
records a SHA-256 checksum of the input so every later result traces to an
exact file, and measures each channel's intensity distribution. Writes a
hand-off file for stage 01 and an HTML report whose contact sheet shows
every channel under its assigned marker name, for a person to confirm.

Nothing here modifies the input image.

Usage:
    python scripts/00_ingest.py --samples tests/spacec_demo/samples.csv --out results
    python scripts/00_ingest.py --samples samples.csv --sample-id K7M2_lung_01 --run-note "first real slide"

Outputs per sample:
    <out>/<sample>/00_ingest/ingest.json    image facts + checks, read by stages 01-02
    <out>/<sample>/00_ingest/channels.csv   one row per image channel
    <out>/<sample>/reports/00_ingest.html
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from akoyalib import imageio, panel as panel_mod, report, runs

STAGE = "00_ingest"
STAGE_STATUS = "smoke-tested only"  # update when validated on a real Fusion .qptiff
PIXEL_BUDGET = 50_000_000          # per channel: read everything below this, sample tiles above
SAMPLE_FRACTION = 0.05             # area fraction sampled when above the budget
PLAUSIBLE_UM = (0.1, 2.0)          # µm/pixel range expected for PhenoCycler-Fusion scans


def sampled_tiles(h, w, tile=1024, fraction=SAMPLE_FRACTION, seed=0):
    tiles = runs.tile_grid(h, w, tile)
    if h * w <= PIXEL_BUDGET:
        return tiles, 1.0
    rng = np.random.default_rng(seed)
    n = max(20, int(len(tiles) * fraction))
    pick = sorted(rng.choice(len(tiles), size=min(n, len(tiles)), replace=False))
    chosen = [tiles[i] for i in pick]
    area = sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in chosen)
    return chosen, area / (h * w)


def channel_stats(handle, tiles):
    """Exact per-channel histograms over the given full-resolution tiles."""
    nb = runs.n_bins_for(handle.dtype)
    rows, hists = [], np.zeros((handle.n_channels, nb), np.int64)
    maxval = nb - 1
    whole = len(tiles) == len(runs.tile_grid(*handle.level_shape(0)[1:], 1024))
    for c in range(handle.n_channels):
        if whole:  # small image: every pixel, read band by band (each stored chunk once)
            for _, _, band in handle.iter_bands(c, 1024):
                runs.add_to_hist(hists[c], band)
        else:
            for y0, y1, x0, x1 in tiles:
                runs.add_to_hist(hists[c], handle.read(c, 0, y=(y0, y1), x=(x0, x1)))
        hist = hists[c]
        total = hist.sum()
        p1, p50, p99, p999 = runs.hist_percentiles(hist, [1, 50, 99, 99.9])
        nz = np.nonzero(hist)[0]
        rows.append({"p1": p1, "p50": p50, "p99": p99, "p99_9": p999,
                     "max": int(nz[-1]) if nz.size else 0,
                     "pct_zero": 100.0 * hist[0] / total,
                     "pct_saturated": 100.0 * hist[maxval] / total})
    return pd.DataFrame(rows), hists


def contact_sheet(handle, table, per_row=8, side=360):
    n = handle.n_channels
    rows = math.ceil(n / per_row)
    _, h, w = handle.level_shape(0)
    fig, axes = plt.subplots(rows, per_row, figsize=(per_row * 1.9, rows * (1.9 * h / w + 0.45)))
    axes = np.atleast_1d(axes).ravel()
    colour = {"ok": report.PALETTE["ink"], "failed": report.PALETTE["fail"],
              "unconfirmed": report.PALETTE["warn"], "": report.PALETTE["warn"]}
    for c in range(n):
        ax = axes[c]
        thumb, _ = handle.read_downsampled(c, max_side=side)
        r = table.iloc[c]
        ax.imshow(imageio.to_display(thumb, r["p1"], max(r["p99_9"], r["p1"] + 1)), cmap="gray",
                  interpolation="nearest")
        label = r["marker"] or "UNMATCHED"
        ax.set_title(f"{c}: {label}\n({r['image_name']})", fontsize=7.5,
                     color=colour.get(r.get("status", ""), report.PALETTE["ink"]))
        ax.set_xticks([]), ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


def ingest_sample(row, args):
    sid = row["sample_id"]
    out_dir = runs.stage_dir(args.out, sid, STAGE)
    rep = report.Report("00", "Ingest & validate", sid, STAGE_STATUS, args.run_note)
    print(f"[{sid}] opening {row['image']}")

    panel = panel_mod.read_panel(row["panel"])
    handle = imageio.ImageHandle(row["image"], channel_names_file=row["channel_names"] or None,
                                 pixel_size_um=row["pixel_size_um"] or None)
    _, H, W = handle.level_shape(0)

    # -- channels ↔ panel ------------------------------------------------------
    matched, not_in_image = panel_mod.match_channels(handle.channel_names, panel)
    table = matched.merge(panel[["marker", "role", "segmentation", "status", "notes"]],
                          on="marker", how="left").fillna("")
    table.loc[table["marker"] == "", "status"] = ""

    if handle.channel_names_source.startswith("placeholder"):
        rep.check("fail", "The image carries no channel names and no channel-names file was given. "
                          "Add a <code>channel_names</code> file to samples.csv.")
    else:
        rep.check("info", f"Channel names read from {handle.channel_names_source}.")
    how = table["how"].value_counts().to_dict()
    if how.get("order"):
        rep.check("warn", "No channel matched the panel by name, so channels were assigned to markers "
                          "<strong>by position</strong>. Check every panel in the contact sheet below.")
    unmatched = table[table["how"] == "unmatched"]
    if len(unmatched):
        rep.check("warn", f"{len(unmatched)} image channel(s) have no panel row and will be ignored: "
                          f"{', '.join(unmatched['image_name'])}.")
    else:
        rep.check("ok", f"All {len(table)} image channels matched a panel marker "
                        f"({how.get('exact', 0)} by name, {how.get('alias', 0)} by alias).")
    if not_in_image:
        rep.check("warn", f"Panel markers not found in the image: {', '.join(not_in_image)}.")
    nuclear = table[(table["segmentation"] == "nuclear") & (table["status"] == "ok")]
    if nuclear.empty:
        rep.check("fail", "No usable nuclear channel (segmentation=nuclear, status=ok). Stage 02 cannot run.")
    membrane = table[(table["segmentation"] == "membrane") & (table["status"] == "ok")]
    rep.check("ok" if len(membrane) else "warn",
              f"Segmentation inputs: nuclear = {', '.join(nuclear['marker']) or 'none'}; "
              f"membrane = {', '.join(membrane['marker']) or 'none'}.")
    bad = table[table["status"].isin(["failed", "unconfirmed"])]
    if len(bad):
        rep.check("info", "Kept visible but excluded from analysis by the panel sheet: " +
                  ", ".join(f"{m} ({s})" for m, s in zip(bad["marker"], bad["status"])) + ".")

    # -- pixel size ---------------------------------------------------------------
    px = handle.pixel_size_um
    if px is None:
        rep.check("fail", "No pixel size in the file's metadata. Add <code>pixel_size_um</code> to samples.csv.")
    else:
        level = "info" if handle.pixel_size_source == "explicit override" else "ok"
        rep.check(level, f"Pixel size {px:.4g} µm, from {handle.pixel_size_source}.")
        if not PLAUSIBLE_UM[0] <= px <= PLAUSIBLE_UM[1]:
            rep.check("warn", f"Pixel size {px:.4g} µm is outside the {PLAUSIBLE_UM[0]}-{PLAUSIBLE_UM[1]} µm "
                              f"range expected for PhenoCycler-Fusion; check the value.")

    # -- bit depth, checksum, intensity statistics ------------------------------------
    if handle.dtype == np.uint8:
        rep.check("warn", "The image is 8-bit (256 intensity levels). If this is an export, look for a "
                          "higher-bit-depth original: coarse levels limit how well dim markers separate "
                          "from background.")
    print(f"[{sid}] checksum")
    sha = report.sha256sum(handle.path)
    tiles, frac = sampled_tiles(H, W)
    print(f"[{sid}] intensity statistics over {frac:.0%} of the image")
    stats, hists = channel_stats(handle, tiles)
    table = pd.concat([table, stats], axis=1)
    np.savez_compressed(out_dir / "channel_histograms.npz", hists=hists, fraction_read=frac)

    sat = table[(table["pct_saturated"] > 0.1) & (table["status"] == "ok")]
    if len(sat):
        rep.check("warn", "More than 0.1% of pixels are at the maximum value (saturated) in: " +
                  ", ".join(f"{m} ({p:.2g}%)" for m, p in zip(sat["marker"], sat["pct_saturated"])) +
                  ". Saturated pixels cap measured intensity.")
    else:
        rep.check("ok", "No usable channel has more than 0.1% saturated pixels.")
    empty = table[(table["p99_9"] <= table["p1"]) & (table["status"] == "ok")]
    if len(empty):
        rep.check("warn", f"Channels with essentially no signal range: {', '.join(empty['marker'])}.")

    fails = sum(1 for lvl, _ in rep.checks if lvl == "fail")
    table.to_csv(out_dir / "channels.csv", index=False)
    facts = {
        "sample_id": sid, "image": str(handle.path), "image_bytes": handle.path.stat().st_size,
        "image_mtime": handle.path.stat().st_mtime, "sha256": sha, "format": handle.format,
        "axes": handle.axes, "dtype": str(handle.dtype), "n_channels": handle.n_channels,
        "levels": [list(handle.level_shape(lvl)) for lvl in range(handle.n_levels)],
        "pixel_size_um": px, "pixel_size_source": handle.pixel_size_source,
        "channel_names_source": handle.channel_names_source, "species": row["species"],
        "panel": row["panel"], "panel_sha256": report.sha256sum(row["panel"]),
        "channel_names_file": row["channel_names"], "exclusions": row["exclusions"],
        "stats_fraction_read": frac, "ok_to_continue": fails == 0,
    }
    runs.write_json(out_dir / "ingest.json", facts)

    # -- report ---------------------------------------------------------------------
    rep.h2("The image")
    size_mm = f"{H * px / 1000:.2f} × {W * px / 1000:.2f} mm" if px else "unknown (no pixel size)"
    rep.table(pd.DataFrame([
        ("File", handle.path.name), ("Format", handle.format), ("Size on disk", f"{facts['image_bytes'] / 1e9:.2f} GB"),
        ("Channels", handle.n_channels), ("Pixels (full resolution)", f"{H:,} × {W:,}"),
        ("Physical size", size_mm), ("Pyramid levels", handle.n_levels), ("Bit depth", str(handle.dtype)),
        ("Pixel size", f"{px} µm ({handle.pixel_size_source})" if px else "not found"),
        ("Species", row["species"]), ("SHA-256", sha[:16] + "…"),
    ], columns=["", "value"]))
    rep.h2("Channel by channel")
    rep.explain("Each image channel matched to the panel sheet. <strong>role</strong> decides how later stages "
                "use it; <strong>status</strong> failed/unconfirmed keeps a marker out of analysis without hiding it. "
                f"Intensities are measured over {frac:.0%} of full-resolution pixels "
                "(the whole image when it is small enough), before any tissue masking, so off-tissue background "
                "counts too. pct_saturated is the share of pixels at the maximum value.")
    cols = ["index", "image_name", "marker", "how", "role", "segmentation", "status",
            "p1", "p50", "p99", "p99_9", "pct_zero", "pct_saturated"]
    rep.table(table[cols], floatfmt="{:,.3g}")
    rep.h2("Contact sheet: does each name match its image?")
    rep.explain("Every channel at low resolution under the marker name it was assigned. Brightness is set by one "
                "rule for all channels (1st to 99.9th percentile of that channel), not by eye. A nuclear stain "
                "should look like nuclei, a membrane marker like outlines, a vessel marker like vessels. "
                "Titles in red are markers the panel marks as failed; amber, unconfirmed or unmatched.")
    rep.figure(contact_sheet(handle, table),
               "All channels, same display rule. Numbers are image channel indices; the image's own channel "
               "name is in brackets.")
    rep.set_provenance(
        inputs={"image": f"{handle.path} (sha256 {sha})", "panel": row["panel"],
                "channel_names": row["channel_names"] or "(from image metadata)",
                "samples sheet": args.samples},
        params={"pixel_budget_per_channel": PIXEL_BUDGET, "sample_fraction_above_budget": SAMPLE_FRACTION,
                "plausible_pixel_um": PLAUSIBLE_UM},
        outputs={"ingest.json": out_dir / "ingest.json", "channels.csv": out_dir / "channels.csv"},
        repo_dir=runs.REPO_DIR)
    path = rep.write(runs.report_path(args.out, sid, STAGE))
    handle.close()
    print(f"[{sid}] report: {path}" + ("" if fails == 0 else f"  ({fails} FAIL check(s) - fix before stage 01)"))
    return fails == 0


def main():
    args = runs.base_parser(__doc__).parse_args()
    ok = [ingest_sample(row, args) for _, row in runs.select_samples(args).iterrows()]
    sys.exit(0 if all(ok) else 2)


if __name__ == "__main__":
    main()
