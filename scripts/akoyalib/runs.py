"""Plumbing shared by every stage: CLI arguments, output layout, sample selection,
stage-to-stage hand-off files, and exact intensity histograms.

Output layout (one folder per sample, one sub-folder per stage):

    <out>/<sample_id>/00_ingest/...
    <out>/<sample_id>/01_image_qc/...
    <out>/<sample_id>/reports/00_ingest.html
    <out>/_batch/...            cross-sample outputs (e.g. display ranges)
"""
import argparse
import os
import json
import pathlib
import sys

import numpy as np

from . import panel as panel_mod

REPO_DIR = pathlib.Path(__file__).resolve().parents[2]


def base_parser(description):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", required=True, help="samples.csv (see scripts/akoyalib/panel.py)")
    p.add_argument("--sample-id", action="append", default=[],
                   help="run only this sample (repeatable); default: every sample in the sheet")
    p.add_argument("--out", default="results", help="output root (default: results/)")
    p.add_argument("--run-note", default="",
                   help="what this run is, shown in every report header (e.g. 'SPACEc public demo')")
    return p


def select_samples(args):
    samples = panel_mod.read_samples(args.samples)
    if args.sample_id:
        unknown = sorted(set(args.sample_id) - set(samples["sample_id"]))
        if unknown:
            sys.exit(f"--sample-id not in {args.samples}: {unknown}")
        samples = samples[samples["sample_id"].isin(args.sample_id)]
    return samples.reset_index(drop=True)


def stage_dir(out, sample_id, stage):
    d = pathlib.Path(out) / sample_id / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_path(out, sample_id, stage):
    return pathlib.Path(out) / sample_id / "reports" / f"{stage}.html"


def write_json(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return str(o)
    raise TypeError(type(o))


def deepcell_token():
    """Return DEEPCELL_ACCESS_TOKEN with surrounding whitespace removed, or ''.

    A token pasted into a file with a line break inside the quotes produces an
    invalid HTTP header, and the library's error message then quotes the token
    value back - printing a secret into a log. Stripping here removes both the
    failure and that exposure. The value is never printed by this code.
    """
    token = os.environ.get("DEEPCELL_ACCESS_TOKEN", "").strip()
    if token:
        os.environ["DEEPCELL_ACCESS_TOKEN"] = token
    return token


def require_stage(out, sample_id, stage, filename):
    """Load a previous stage's hand-off file or stop with a clear instruction."""
    path = pathlib.Path(out) / sample_id / stage / filename
    if not path.exists():
        sys.exit(f"[{sample_id}] {path} not found - run stage {stage} for this sample first.")
    return read_json(path)


# -- exact histograms ----------------------------------------------------------

def n_bins_for(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return 256
    if dtype == np.uint16:
        return 65536
    raise ValueError(f"histograms are implemented for uint8/uint16 images, not {dtype}; "
                     f"convert the image or extend runs.n_bins_for")


def add_to_hist(hist, pixels):
    """Accumulate integer pixel values into a bincount histogram in place."""
    hist += np.bincount(np.asarray(pixels).ravel(), minlength=hist.size)[:hist.size]


def hist_percentiles(hist, qs):
    """Percentiles (0-100) from an integer-value histogram; exact up to the integer bin."""
    total = hist.sum()
    if total == 0:
        return [float("nan")] * len(qs)
    cdf = np.cumsum(hist) / total
    return [float(np.searchsorted(cdf, q / 100.0, side="left")) for q in qs]


def tile_grid(height, width, tile):
    """(y0, y1, x0, x1) tiles covering the image, row-major."""
    return [(y, min(y + tile, height), x, min(x + tile, width))
            for y in range(0, height, tile) for x in range(0, width, tile)]
