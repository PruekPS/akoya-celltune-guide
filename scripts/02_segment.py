"""Stage 02: segment nuclei and whole cells, and compare methods.

Why several methods: a cell outline decides which pixels -- and so which
marker signal -- are credited to each cell. Outlines that are too big pull in
a neighbour's signal (spillover); too small, and membrane markers are missed.
There is no ground truth in a new image, so this stage runs more than one
method, measures how much they disagree, and shows the disagreements on the
image for a person to judge.

Methods (choose with --methods; default "cellsam,expansion"):
  cellsam      CellSAM (Marks/Israel et al., Nat Methods 2025), the default whole-cell
               method: a segmentation foundation model that finds cells directly from
               the image rather than growing them from nuclei. Input is (blank, nuclear,
               whole-cell), the format CellSAM documents for multiplexed images. Pure
               PyTorch, so it runs on current GPUs; weights download from
               users.deepcell.org, so DEEPCELL_ACCESS_TOKEN must be set. Practically
               needs a GPU (SAM backbone). See envs/environment-cellsam.yml.
               With --cellsam-merge-nested: also folds predicted cells that are mostly
               swallowed by a dilated neighbour into it (CellSAM sometimes returns two
               overlapping labels for one true cell). Ported from a hand-tuned fix
               (CellSAM_local/cellsam/templates/run_segmentation-Copy1.ipynb) already
               checked by eye on PS88 crops; here it runs block by block so it scales to
               a whole slide instead of one small field of view.
  nuclei       StarDist (Schmidt et al., MICCAI 2018; pretrained 2D_versatile_fluo)
               on the nuclear stain. Always run: it provides the nuclei, and the
               nucleus-based methods below grow from it.
  expansion    each nucleus grown by a fixed distance (--expand-um), stopping where
               it meets a neighbour. This is what Akoya's QuPath StarDist workflow and
               most PhenoCycler studies do, and the baseline CellSAM is compared against.
  propagation  each nucleus grown outward over the membrane composite image, so
               neighbouring cells meet at the bright membrane between them rather than
               halfway. Growth always reaches --expand-um; beyond that it continues only
               onto pixels with membrane signal (--min-membrane), up to --max-grow-um.
  mesmer       Optional. Mesmer whole-cell segmentation (Greenwald et al., Nat
               Biotechnol 2022). Its last release (DeepCell 0.12.10, Aug 2024) still
               pins TensorFlow 2.8 and Python < 3.11, so it needs its own environment
               (envs/environment-mesmer.yml) and runs on CPU: TF 2.8 predates the CUDA
               current GPUs need. Kept because it is the method most PhenoCycler studies
               use, which makes it a useful independent yardstick.

Membrane composite: every panel channel with segmentation=membrane and
status=ok, each scaled by its in-tissue display range from stage 01, clipped
to 0-1 and averaged. Only cells whose nucleus centre lies in stage 01's
analysis mask are kept.

Optional reference comparison (--reference): a CSV of cell centres from
another segmentation of the same image (columns x_px, y_px, optional region,
optional area_px). With --reference-align translate, each reference region is
located in this image by a translation search, which handles references made
on cropped images. Reports the share of reference cells that have one of our
nuclei within --match-um, and compares cell areas of matched pairs.

Usage:
    python scripts/02_segment.py --samples tests/spacec_demo/samples.csv --out results \\
        --reference tests/spacec_demo/mesmer_reference.csv --reference-align translate

Outputs (<out>/<sample>/02_segment/):
    nuclei.tif, cells_<method>.tif   label images (0 = background), tiled + zlib, open in QuPath/Fiji
    cells_<method>.csv               one row per cell: centre (px and µm), area, nucleus area, region
    segment.json                     counts and parameters, read by stage 03
"""
import os
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, measure, segmentation

from akoyalib import imageio, report, runs

STAGE = "02_segment"
STAGE_STATUS = {"nuclei": "smoke-tested only", "expansion": "smoke-tested only",
                "propagation": "smoke-tested only", "mesmer": "built, never run"}
BLOCK = 2048


# -- helpers ------------------------------------------------------------------------------

class LazyNormalized:
    """Array-like view of one channel, scaled so lo->0 and hi->1, read block by block."""

    def __init__(self, handle, channel, lo, hi):
        self.h, self.c, self.lo, self.hi = handle, channel, float(lo), float(hi)
        _, H, W = handle.level_shape(0)
        self.shape, self.ndim, self.dtype = (H, W), 2, np.dtype(np.float32)

    def __getitem__(self, idx):
        ys, xs = idx[0], idx[1]
        y0, y1, _ = ys.indices(self.shape[0])
        x0, x1, _ = xs.indices(self.shape[1])
        a = self.h.read(self.c, 0, y=(y0, y1), x=(x0, x1)).astype(np.float32)
        return (a - self.lo) / (self.hi - self.lo)


def blocks(H, W, block=BLOCK):
    for y0 in range(0, H, block):
        for x0 in range(0, W, block):
            yield y0, min(y0 + block, H), x0, min(x0 + block, W)


def pad(y0, y1, x0, x1, halo, H, W):
    return max(0, y0 - halo), min(H, y1 + halo), max(0, x0 - halo), min(W, x1 + halo)


def mask_block(mask_w, f, py0, py1, px0, px1):
    rr = np.minimum((np.arange(py0, py1) / f).astype(int), mask_w.shape[0] - 1)
    cc = np.minimum((np.arange(px0, px1) / f).astype(int), mask_w.shape[1] - 1)
    return mask_w[np.ix_(rr, cc)]


def membrane_block(handle, mem, py0, py1, px0, px1):
    acc = None
    for c, lo, hi in mem:
        a = np.clip((handle.read(c, 0, y=(py0, py1), x=(px0, px1)).astype(np.float32) - lo) / (hi - lo), 0, 1)
        acc = a if acc is None else acc + a
    return acc / len(mem)


def label_stats(labels, H, W, n_max, other=None):
    """Exact per-label pixel count and centroid (and overlap with `other` labels of the same id)."""
    cnt = np.zeros(n_max + 1, np.int64)
    sy = np.zeros(n_max + 1)
    sx = np.zeros(n_max + 1)
    inter = np.zeros(n_max + 1, np.int64) if other is not None else None
    for y0, y1, x0, x1 in blocks(H, W):
        lab = np.asarray(labels[y0:y1, x0:x1])
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = lab > 0
        l = lab[m]
        cnt += np.bincount(l, minlength=n_max + 1)[:n_max + 1]
        sy += np.bincount(l, weights=yy[m], minlength=n_max + 1)[:n_max + 1]
        sx += np.bincount(l, weights=xx[m], minlength=n_max + 1)[:n_max + 1]
        if other is not None:
            o = np.asarray(other[y0:y1, x0:x1])
            same = m & (o == lab)
            inter += np.bincount(lab[same], minlength=n_max + 1)[:n_max + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return cnt, sy / cnt, sx / cnt, inter


def save_labels(path, arr):
    tifffile.imwrite(path, arr, compression="zlib", tile=(512, 512), metadata=None)


# -- methods --------------------------------------------------------------------------------

def run_nuclei(handle, nuc_idx, lo, hi, work_dir, args):
    from stardist.models import StarDist2D
    _, H, W = handle.level_shape(0)
    model = StarDist2D.from_pretrained(args.stardist_model)
    scale = None if abs(args.pixel_um / args.stardist_target_um - 1) < 0.05 else args.pixel_um / args.stardist_target_um
    out = np.lib.format.open_memmap(work_dir / "nuclei_raw.npy", mode="w+", dtype=np.int32, shape=(H, W))
    model.predict_instances_big(LazyNormalized(handle, nuc_idx, lo, hi), axes="YX", block_size=BLOCK,
                                min_overlap=128, context=96, labels_out=out, show_progress=False,
                                prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh, scale=scale)
    return out, scale


def filter_relabel(labels, keep_ids, H, W, path):
    """Keep only keep_ids, renumbered 1..n in order; written to a new memmap."""
    lut = np.zeros(int(labels.max()) + 1 if labels.size else 1, np.int32)
    lut[keep_ids] = np.arange(1, len(keep_ids) + 1, dtype=np.int32)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(H, W))
    for y0, y1, x0, x1 in blocks(H, W):
        out[y0:y1, x0:x1] = lut[np.asarray(labels[y0:y1, x0:x1])]
    return out


def run_expansion(nuclei, px, args, path):
    H, W = nuclei.shape
    d = max(1, int(round(args.expand_um / px)))
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(H, W))
    for y0, y1, x0, x1 in blocks(H, W):
        py0, py1, px0, px1 = pad(y0, y1, x0, x1, d + 2, H, W)
        e = segmentation.expand_labels(np.asarray(nuclei[py0:py1, px0:px1]), distance=d)
        out[y0:y1, x0:x1] = e[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
    return out, d


def run_propagation(nuclei, handle, mem, mask_w, f, px, args, path):
    H, W = nuclei.shape
    g = max(1, int(round(args.max_grow_um / px)))
    d_min = min(g, max(1, int(round(args.expand_um / px))))
    halo = 2 * g + 4
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(H, W))
    for y0, y1, x0, x1 in blocks(H, W):
        py0, py1, px0, px1 = pad(y0, y1, x0, x1, halo, H, W)
        nuc = np.asarray(nuclei[py0:py1, px0:px1])
        if not nuc.any():
            out[y0:y1, x0:x1] = 0
            continue
        raw_memb = membrane_block(handle, mem, py0, py1, px0, px1)
        memb = filters.gaussian(raw_memb, sigma=1)
        dist = ndi.distance_transform_edt(nuc == 0)
        # Always allowed up to the expansion distance; beyond it only onto pixels with membrane
        # signal (gated on a smoother image). Without the gate cells grew into empty space to the
        # full cap; gating on the sigma=1 image left ragged edges and islands (both seen on the demo).
        reach = (dist <= d_min) | ((dist <= g) & (filters.gaussian(raw_memb, sigma=2) >= args.min_membrane))
        allowed = (reach & mask_block(mask_w, f, py0, py1, px0, px1)) | (nuc > 0)
        elevation = memb + args.distance_weight * dist / g
        lab = segmentation.watershed(elevation, markers=nuc, mask=allowed)
        # keep only the part of each cell connected to its own nucleus (drops detached islands)
        pieces = measure.label(lab, background=0, connectivity=1)
        keep = np.unique(pieces[nuc > 0])
        lab[~np.isin(pieces, keep)] = 0
        out[y0:y1, x0:x1] = lab[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
    return out, g


def blockwise_instances(predict, handle, nuc_idx, nuc_range, mem, mask_w, f, path, block, halo, resume=False):
    """Run a whole-cell segmenter block by block and stitch the labels.

    `predict(nuclear, membrane)` takes two float images scaled 0-1 and returns
    an integer label image of the same shape. Blocks overlap by `halo`; a cell
    belongs to the block containing its centre, and its full footprint is
    written wherever the output is still empty, so cells crossing a block edge
    stay whole. Used by both CellSAM and Mesmer, so their labels are stitched
    identically and the comparison between them is about the models.

    Resumable, because a whole slide on CPU outlives a wall-time limit (PS88:
    ~67 h projected against a 36 h limit). Label ids are handed out in block
    order as one contiguous range per block, so the file after block k is fully
    described by the pair (k, next id). That pair is written to
    `<name>.progress.json` after every block, once the block's labels are
    flushed, and `resume=True` continues from it. A file from a run that predates
    the progress record is resumed by reconstructing the pair from the labels
    themselves. Without `resume`, the file is recreated from zeros, as before.
    """
    _, H, W = handle.level_shape(0)
    path = pathlib.Path(path)
    prog = path.with_name(path.stem + ".progress.json")
    todo = list(blocks(H, W, block))
    start, next_id = 0, 1
    if resume and path.exists():
        out = np.load(path, mmap_mode="r+")
        if out.shape != (H, W) or out.dtype != np.int32:
            raise ValueError(f"{path.name} is {out.dtype} {out.shape}, expected int32 {(H, W)}; cannot resume into it")
        start, next_id, how = _resume_point(out, todo, H, W, block, halo, prog)
        cleared = _clear_stale(out, todo, start, next_id, H, W, halo)
        if start >= len(todo):
            print(f"  {path.name} is already complete ({how}); nothing left to segment", flush=True)
        else:
            print(f"  resuming {path.name} at block {start + 1:,} of {len(todo):,}, next id {next_id:,} ({how})"
                  + (f"; cleared {cleared:,} pixels so that block is redone from a clean window" if cleared else ""),
                  flush=True)
    else:
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(H, W))
        prog.unlink(missing_ok=True)
    for bi in range(start, len(todo)):
        y0, y1, x0, x1 = todo[bi]
        py0, py1, px0, px1 = pad(y0, y1, x0, x1, halo, H, W)
        if mask_block(mask_w, f, py0, py1, px0, px1).any():
            nuc = np.clip((handle.read(nuc_idx, 0, y=(py0, py1), x=(px0, px1)).astype(np.float32) - nuc_range[0])
                          / (nuc_range[1] - nuc_range[0]), 0, 1)
            memb = membrane_block(handle, mem, py0, py1, px0, px1) if mem else np.zeros_like(nuc)
            lab = predict(nuc, memb)
            ids = np.unique(lab[lab > 0])
            if ids.size:
                yy, xx = np.mgrid[py0:py1, px0:px1]
                cy = ndi.mean(yy, lab, ids)
                cx = ndi.mean(xx, lab, ids)
                owned = ids[(cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)]
                lut = np.zeros(int(lab.max()) + 1, np.int32)
                lut[owned] = np.arange(next_id, next_id + len(owned), dtype=np.int32)
                next_id += len(owned)
                region = out[py0:py1, px0:px1]
                new = lut[lab]
                write = (new > 0) & (region == 0)
                region[write] = new[write]
                out.flush()
        _save_progress(prog, bi, next_id, block, halo, H, W)
    out.flush()
    _save_progress(prog, len(todo) - 1, next_id, block, halo, H, W, done=True)
    return out


def _save_progress(prog, block_index, next_id, block, halo, H, W, done=False):
    """Atomically record that blocks up to `block_index` are written and flushed."""
    tmp = prog.with_name(prog.name + ".tmp")
    tmp.write_text(json.dumps({"block_index": int(block_index), "next_id": int(next_id), "block": int(block),
                               "halo": int(halo), "H": int(H), "W": int(W), "done": bool(done)}))
    os.replace(tmp, prog)


def _resume_point(out, todo, H, W, block, halo, prog):
    """(first block to run, next id, how it was found)."""
    if prog.exists():
        rec = json.loads(prog.read_text())
        if (rec.get("block"), rec.get("halo"), rec.get("H"), rec.get("W")) != (block, halo, H, W):
            raise ValueError(
                f"{prog.name} was written with block={rec.get('block')}, halo={rec.get('halo')} on a "
                f"{rec.get('H')}x{rec.get('W')} image, but this run uses block={block}, halo={halo} on {H}x{W}. "
                f"Which block owns a cell depends on both, so resuming would mix two segmentations.")
        if rec.get("done"):
            return len(todo), int(rec["next_id"]), "progress record marks it done"
        return int(rec["block_index"]) + 1, int(rec["next_id"]), "from progress record"
    return _reconstruct_resume_point(out, todo, H, W, block, halo)


def _reconstruct_resume_point(out, todo, H, W, block, halo):
    """Recover (block, next id) from the labels of a run that kept no progress record.

    The highest id written, M, belongs to the last block that wrote anything, and
    that block's cells hold the contiguous ids [first, M]. Its written cells lie
    inside its padded window with centres in its core, so walking down from M
    while ids stay contiguous and centred in that core finds `first`. If the walk
    stops early (a cell of that block that got no pixels of its own leaves a gap),
    the only consequence is unused ids; no cell is lost or duplicated.
    """
    ncols = -(-W // block)
    M, band_y = 0, 0
    for y0 in range(0, H, block):
        m = int(np.asarray(out[y0:min(y0 + block, H)]).max())
        if m > M:
            M, band_y = m, y0
    if M == 0:
        return 0, 1, "file held no labels; starting from the first block"
    wy0, wy1 = max(0, band_y - block), min(H, band_y + 2 * block)
    yy, xx = np.nonzero(np.asarray(out[wy0:wy1]) == M)
    s = int((yy.mean() + wy0) // block) * ncols + int(xx.mean() // block)
    y0, y1, x0, x1 = todo[s]
    py0, py1, px0, px1 = pad(y0, y1, x0, x1, halo, H, W)
    region = np.asarray(out[py0:py1, px0:px1])
    cand = np.unique(region[(region > 0) & (region <= M)])[::-1]
    gy, gx = np.mgrid[py0:py1, px0:px1]
    cy = np.asarray(ndi.mean(gy, region, cand))
    cx = np.asarray(ndi.mean(gx, region, cand))
    first, expected = M + 1, M
    for cid, ok in zip(cand.tolist(), ((cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)).tolist()):
        if cid != expected or not ok:
            break
        first, expected = cid, expected - 1
    return s, int(first), f"reconstructed from labels: block {s + 1:,} owned ids {first:,}-{M:,}"


def _clear_stale(out, todo, start, next_id, H, W, halo):
    """Zero ids >= next_id inside the start block's padded window.

    Only that block can have written them -- either half-way through when the job
    died, or completely but before its progress record was saved. Redoing the
    block from a clean window means its fresh ids cannot collide with a partial
    first attempt, even if the model's output differs slightly between runs.
    """
    if start >= len(todo):
        return 0
    y0, y1, x0, x1 = todo[start]
    py0, py1, px0, px1 = pad(y0, y1, x0, x1, halo, H, W)
    region = out[py0:py1, px0:px1]
    stale = region >= next_id
    n = int(stale.sum())
    if n:
        region[stale] = 0
        out.flush()
    return n


def run_cellsam(handle, nuc_idx, nuc_range, mem, mask_w, f, px, path, args):
    """CellSAM whole-cell segmentation (Marks/Israel et al., Nat Methods 2025).

    Input per block is (H, W, 3) as (blank, nuclear, whole-cell), the format
    CellSAM documents for multiplexed images; the third channel is our membrane
    composite. Weights download from users.deepcell.org, so DEEPCELL_ACCESS_TOKEN
    must be set. Needs a GPU to be practical: the model is a SAM backbone.
    """
    import torch
    from cellSAM import get_model, segment_cellular_image

    # Pin torch to the allocation. On PS88 with only OMP_NUM_THREADS set, ~29 of 32
    # cores stayed busy yet CellSAM ran ~7-8x slower per core-second than on the
    # demo; library pools sized from the node rather than the job are the leading
    # suspect. --threads 0 means: use the SLURM allocation, else the library default.
    n_threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if n_threads:
        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:          # already fixed by an earlier parallel op in this process
            pass
    print(f"  torch threads: {torch.get_num_threads()} intra-op, {torch.get_num_interop_threads()} inter-op",
          flush=True)

    device = args.cellsam_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        print("  CellSAM on CPU: expect this to be slow (SAM backbone).")
    model = get_model()
    if device != "cpu":
        model = model.to(device)

    empty = []

    def predict(nuc, memb):
        img = np.stack([np.zeros_like(nuc), nuc, memb], axis=-1)
        return mask_or_empty(
            lambda: segment_cellular_image(img, model=model, device=device,
                                           bbox_threshold=args.cellsam_bbox_threshold,
                                           normalize=not args.cellsam_no_normalize),
            nuc.shape, empty)

    out = blockwise_instances(predict, handle, nuc_idx, nuc_range, mem, mask_w, f, path,
                              block=args.cellsam_block_px, halo=args.cellsam_block_px // 8,
                              resume=args.resume)
    if empty:
        print(f"  CellSAM found nothing in {len(empty)} block(s); those were left empty.", flush=True)
    return out


def mask_or_empty(call, shape, empty_blocks=None):
    """A block's labels, or an empty block where the model found no cells.

    CellSAM returns None when it finds nothing, and then crashes inside its own
    `fill_holes_and_remove_small_masks` on `None.ndim` -- which killed PS82's
    stage 02 at block 174 of about 3,000, 12 minutes in (job 42263229). A block
    with no cells is normal on a slide with sparse tissue, not a failure, so it
    becomes an array of zeros and is counted. Only the two exception types that
    bug raises are caught; anything else still stops the run.
    """
    try:
        mask, _, _ = call()
    except (AttributeError, TypeError):
        mask = None
    if mask is None:
        if empty_blocks is not None:
            empty_blocks.append(True)
        return np.zeros(shape, np.int32)
    return np.asarray(mask, np.int32)


def merge_nested_labels(raw, H, W, px, out_path, threshold=0.9, dilation_um=5.0, search_um=50.0, block=BLOCK):
    """Fold predicted cells that are mostly swallowed by a dilated neighbour into it.

    Ported from a hand-tuned fix (`merge_nested_labels_radius` in
    CellSAM_local/cellsam/templates/run_segmentation-Copy1.ipynb) already
    checked by eye on PS88 crops: CellSAM sometimes returns two overlapping
    labels for what is really one cell. For an ordered pair (big, small),
    `small` is folded into `big` when `threshold` or more of `small`'s area
    lies inside `big`'s mask grown by `dilation_um`; only pairs within
    `search_um` of each other are ever compared.

    Run block by block, like `blockwise_instances`. The original notebook
    version calls `regionprops` on the whole label image at once and compares
    every pair of cells -- fine for one small field of view, but a whole slide
    holds hundreds of thousands of cells: the full array does not fit in
    memory as a `regionprops` call, and an all-pairs comparison never
    finishes. Each label's fate is decided once, by whichever block's core
    (not halo) contains its centroid -- the same ownership rule
    `blockwise_instances` uses for stitching -- so two blocks never resolve
    the same pair two different ways. A label already folded into another
    cannot itself be chosen as a new absorber, so chains (A into B into C)
    collapse to their root when the mapping is applied, in one pass.

    Returns (merged label memmap, number of predicted cells folded away).
    """
    dilation_px = max(1, int(round(dilation_um / px)))
    search_px = max(dilation_px, int(round(search_um / px)))
    halo = search_px + dilation_px + 2

    lut = {}  # absorbed label -> the label it was folded into (not yet root-resolved)

    def root(label):
        seen = []
        while label in lut:
            seen.append(label)
            label = lut[label]
        for s in seen:
            lut[s] = label  # path-compress so later look-ups are O(1)
        return label

    for y0, y1, x0, x1 in blocks(H, W, block):
        py0, py1, px0, px1 = pad(y0, y1, x0, x1, halo, H, W)
        crop = np.asarray(raw[py0:py1, px0:px1])
        if not crop.any():
            continue
        regions = {r.label: r for r in measure.regionprops(crop)}
        if len(regions) < 2:
            continue
        ids = np.fromiter(regions.keys(), dtype=np.int64)
        centres = np.array([[regions[int(i)].centroid[0] + py0, regions[int(i)].centroid[1] + px0] for i in ids])
        owned = ids[(centres[:, 0] >= y0) & (centres[:, 0] < y1) & (centres[:, 1] >= x0) & (centres[:, 1] < x1)]
        if not owned.size:
            continue
        tree = cKDTree(centres)
        pos_of = {int(v): k for k, v in enumerate(ids)}
        for oid in owned.tolist():
            if oid in lut:
                continue  # already folded into something by an earlier block
            other = regions[oid]
            other_mask, other_area, osl = other.image, other.area, other.slice
            nearby = tree.query_ball_point(centres[pos_of[oid]], r=search_px)
            best = None
            for j in nearby:
                cand = int(ids[j])
                if cand == oid or cand in lut:
                    continue
                big = regions[cand]
                cmask = np.zeros(crop.shape, bool)
                cmask[big.slice][big.image] = True
                dil = ndi.binary_dilation(cmask, iterations=dilation_px)
                frac = np.logical_and(dil[osl], other_mask).sum() / other_area
                if frac >= threshold and (best is None or frac > best[1]):
                    best = (cand, frac)
            if best is not None:
                lut[oid] = best[0]

    n_merged = len(lut)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int32, shape=(H, W))
    remap = None
    if n_merged:
        remap = np.arange(int(raw.max()) + 1, dtype=np.int32)
        for label in lut:
            remap[label] = root(label)
    for y0, y1, x0, x1 in blocks(H, W, block):
        chunk = np.asarray(raw[y0:y1, x0:x1])
        out[y0:y1, x0:x1] = remap[chunk] if remap is not None else chunk
    out.flush()
    return out, n_merged


def run_mesmer(handle, nuc_idx, nuc_range, mem, mask_w, f, px, path):
    """Blockwise Mesmer whole-cell segmentation (optional; needs DeepCell + token)."""
    from deepcell.applications import Mesmer  # imported only where available
    app = Mesmer()

    def predict(nuc, memb):
        lab = app.predict(np.stack([nuc, memb], -1)[None], image_mpp=px, compartment="whole-cell")
        return np.asarray(lab[0, ..., 0], np.int32)

    return blockwise_instances(predict, handle, nuc_idx, nuc_range, mem, mask_w, f, path,
                               block=BLOCK, halo=128)


# -- reference comparison ---------------------------------------------------------------------

def _point_map(xy, shape, bin_px):
    m = np.zeros(shape, np.float32)
    iy = np.clip((xy[:, 1] // bin_px).astype(int), 0, shape[0] - 1)
    ix = np.clip((xy[:, 0] // bin_px).astype(int), 0, shape[1] - 1)
    np.add.at(m, (iy, ix), 1)
    return ndi.gaussian_filter(m, 1.0)


def align_translate(ours_xy, ref_xy, H, W, bin_px, match_px):
    """Translation (and x/y orientation) that overlays reference centres onto ours.

    Works on individual cell positions at fine resolution (bin_px ~ 2 µm): at the
    true offset the same cells coincide and the cross-correlation has a sharp
    peak; at wrong offsets only tissue-scale density overlaps, which is flat.
    Coarser bins only see tissue outlines, and two similar TMA cores then look
    alike (this failed on the first run on the SPACEc demo).

    Returns (dx, dy, swapped, peak_z). peak_z is how many robust standard
    deviations the peak stands above the rest of the correlation map; a real
    alignment is far above ~10.
    """
    from scipy.signal import fftconvolve
    A = _point_map(ours_xy, (int(np.ceil(H / bin_px)), int(np.ceil(W / bin_px))), bin_px)
    best = None
    for swapped in (False, True):
        r = ref_xy[:, ::-1] if swapped else ref_xy
        origin = r.min(axis=0)
        rr = r - origin
        B = _point_map(rr, (int(rr[:, 1].max() // bin_px) + 1, int(rr[:, 0].max() // bin_px) + 1), bin_px)
        corr = fftconvolve(A, B[::-1, ::-1], mode="full")
        iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
        body = corr[corr > 0.05 * corr.max()]
        med = np.median(body)
        mad = np.median(np.abs(body - med)) * 1.4826 or 1.0
        z = float((corr[iy, ix] - med) / mad)
        ox = (ix - (B.shape[1] - 1)) * bin_px - origin[0]
        oy = (iy - (B.shape[0] - 1)) * bin_px - origin[1]
        if best is None or z > best[3]:
            best = (ox, oy, swapped, z)
    ox, oy, swapped, z = best
    r = ref_xy[:, ::-1] if swapped else ref_xy
    tree = cKDTree(ours_xy)
    shifted = r + [ox, oy]
    d, j = tree.query(shifted, distance_upper_bound=match_px)
    ok = np.isfinite(d)
    resid = np.median(ours_xy[j[ok]] - shifted[ok], axis=0) if ok.any() else np.zeros(2)
    return ox + resid[0], oy + resid[1], swapped, z


def chance_rate(ours_xy, ref_xy_aligned, max_d, shift_px, n=12, seed=0):
    """Share of reference cells 'found' after deliberately misaligning them by shift_px.

    In dense tissue a random point often has a nucleus within a few µm, so a
    match rate means nothing without this baseline.
    """
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        s = rng.uniform(*shift_px)
        ri, _, _ = match_one_to_one(ours_xy, ref_xy_aligned + [s * np.cos(ang), s * np.sin(ang)], max_d)
        rates.append(len(ri) / len(ref_xy_aligned))
    return float(np.mean(rates))


def overlap_counts(A, B, H, W, b_max):
    """Pixel overlap between two independent label images, as {(a, b): pixels}."""
    counts = {}
    stride = int(b_max) + 1
    for y0, y1, x0, x1 in blocks(H, W):
        a = np.asarray(A[y0:y1, x0:x1]).ravel()
        b = np.asarray(B[y0:y1, x0:x1]).ravel()
        m = (a > 0) & (b > 0)
        if not m.any():
            continue
        key, cnt = np.unique(a[m].astype(np.int64) * stride + b[m].astype(np.int64), return_counts=True)
        for k, c in zip(key.tolist(), cnt.tolist()):
            counts[k] = counts.get(k, 0) + c
    return counts, stride


def compare_two_methods(name_a, df_a, lab_a, name_b, df_b, lab_b, H, W, px, args, seed=0):
    """Compare two segmentations with independent labels (e.g. CellSAM vs expansion).

    Cells are paired one-to-one by centre distance, with the same
    shift-the-points chance baseline used for external references, then the
    pixel overlap (IoU) of each pair is measured. Neither method is ground
    truth: this measures how much they disagree and where, not who is right.
    """
    a_xy = df_a[["x_px", "y_px"]].to_numpy()
    b_xy = df_b[["x_px", "y_px"]].to_numpy()
    # match_one_to_one(ours, ref) returns (index into ref, index into ours) -- in that
    # order. Here `ours` is A and `ref` is B, so the first array indexes B, not A.
    bi, ai, dist = match_one_to_one(a_xy, b_xy, args.tight_um / px)
    chance = chance_rate(a_xy, b_xy, args.tight_um / px, shift_px=(20 / px, 60 / px), seed=seed)
    # The loose radius as well, because in dense tissue it is nearly meaningless on its
    # own: on the tonsil demo, 5 um paired 92.6% of cells against an 80.3% chance rate
    # (+12 points), while 2 um paired 80.6% against 19.6% (+61 points). Reporting only
    # the loose number would have read as near-perfect agreement.
    bi_l, _, _ = match_one_to_one(a_xy, b_xy, args.match_um / px)
    chance_l = chance_rate(a_xy, b_xy, args.match_um / px, shift_px=(20 / px, 60 / px), seed=seed)

    counts, stride = overlap_counts(lab_a, lab_b, H, W, int(df_b["cell_id"].max()))
    area_a = df_a.set_index("cell_id")["area_um2"] if "area_um2" in df_a else df_a.set_index("cell_id")["nucleus_area_um2"]
    area_b = df_b.set_index("cell_id")["area_um2"] if "area_um2" in df_b else df_b.set_index("cell_id")["nucleus_area_um2"]
    ious, ratios = [], []
    for i, j in zip(ai, bi):
        ida, idb = int(df_a["cell_id"].iloc[i]), int(df_b["cell_id"].iloc[j])
        inter = counts.get(ida * stride + idb, 0) * px ** 2
        aa, bb = float(area_a.get(ida, 0)), float(area_b.get(idb, 0))
        if aa > 0 and bb > 0:
            ious.append(inter / (aa + bb - inter))
            ratios.append(aa / bb)
    ious, ratios = np.array(ious), np.array(ratios)
    return {
        "method A": name_a, "method B": name_b, f"{name_a} cells": len(df_a), f"{name_b} cells": len(df_b),
        f"paired within {args.tight_um:g} µm": len(ai),
        f"% of {name_a} paired": 100 * len(ai) / max(len(df_a), 1),
        f"% of {name_b} paired": 100 * len(ai) / max(len(df_b), 1),
        "expected by chance %": 100 * chance,
        "margin over chance (pts)": 100 * (len(ai) / max(len(df_b), 1) - chance),
        f"% of {name_b} paired within {args.match_um:g} µm": 100 * len(bi_l) / max(len(df_b), 1),
        f"chance % at {args.match_um:g} µm": 100 * chance_l,
        "median IoU of pairs": float(np.median(ious)) if ious.size else np.nan,
        "% pairs IoU > 0.5": float(100 * np.mean(ious > 0.5)) if ious.size else np.nan,
        f"median area {name_a}/{name_b}": float(np.median(ratios)) if ratios.size else np.nan,
        "_ious": ious,
    }


def match_one_to_one(ours_xy, ref_xy, max_d):
    """Greedy one-to-one matching by distance.

    Returns (ref_index, our_index, distance): the FIRST array indexes ref_xy (the
    second argument) and the SECOND indexes ours_xy (the first argument). Getting
    this backwards indexes one table with the other's positions, which raises
    IndexError when ref is the longer table and silently mispairs cells when it
    is not.
    """
    tree = cKDTree(ours_xy)
    d, j = tree.query(ref_xy, k=3, distance_upper_bound=max_d)
    cand = [(d[i, k], i, j[i, k]) for i in range(len(ref_xy)) for k in range(3) if np.isfinite(d[i, k])]
    cand.sort()
    used_r, used_o, out = set(), set(), []
    for dist, i, o in cand:
        if i in used_r or o in used_o:
            continue
        used_r.add(i), used_o.add(o)
        out.append((i, o, dist))
    arr = np.array(out) if out else np.zeros((0, 3))
    return arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2]


# -- figures ----------------------------------------------------------------------------------

def roi_panel(handle, nuc_idx, nuc_range, mem, label_sets, rois, roi_px, titles):
    ncol = 1 + len(label_sets)
    fig, axs = plt.subplots(len(rois), ncol, figsize=(2.3 * ncol, 2.3 * len(rois)))
    axs = np.atleast_2d(axs)
    colours = [(1, 1, 1), (1, 0.82, 0.2), (1, 0.3, 0.85), (0.3, 1, 0.6)]
    for r, (y0, x0, tag) in enumerate(rois):
        y1, x1 = y0 + roi_px, x0 + roi_px
        nuc = np.clip((handle.read(nuc_idx, 0, y=(y0, y1), x=(x0, x1)).astype(np.float32) - nuc_range[0])
                      / (nuc_range[1] - nuc_range[0]), 0, 1)
        memb = membrane_block(handle, mem, y0, y1, x0, x1) if mem else np.zeros_like(nuc)
        rgb = np.dstack([memb * 0.2, memb, nuc])
        axs[r, 0].imshow(np.clip(rgb, 0, 1))
        axs[r, 0].set_ylabel(tag, fontsize=8)
        for k, (name, lab) in enumerate(label_sets, start=1):
            img = rgb.copy() * 0.85
            sub = np.asarray(lab[y0:y1, x0:x1])
            bnd = segmentation.find_boundaries(sub, mode="inner")
            img[bnd] = colours[min(k - 1, len(colours) - 1)]
            axs[r, k].imshow(np.clip(img, 0, 1))
        for a in axs[r]:
            a.set_xticks([]), a.set_yticks([])
    for k, t in enumerate(titles):
        axs[0, k].set_title(t, fontsize=9)
    fig.tight_layout(pad=0.3)
    return fig


# -- main per sample ----------------------------------------------------------------------------

def segment_sample(row, args):
    sid = row["sample_id"]
    ingest = runs.require_stage(args.out, sid, "00_ingest", "ingest.json")
    qc = runs.require_stage(args.out, sid, "01_image_qc", "qc.json")
    base = pathlib.Path(args.out) / sid
    channels = pd.read_csv(base / "00_ingest" / "channels.csv", keep_default_na=False)
    ch_qc = pd.read_csv(base / "01_image_qc" / "channel_qc.csv", keep_default_na=False)
    hist_in = np.load(base / "01_image_qc" / "tissue_histograms.npz")["hist_in"]
    masks = np.load(base / "01_image_qc" / "analysis_mask.npz")
    mask_w, f, regions_w = masks["mask"], float(masks["downsample"]), masks["regions"]
    out_dir = runs.stage_dir(args.out, sid, STAGE)
    handle = imageio.ImageHandle(ingest["image"], channel_names_file=ingest["channel_names_file"] or None,
                                 pixel_size_um=ingest["pixel_size_um"])
    px = ingest["pixel_size_um"]
    args.pixel_um = px
    _, H, W = handle.level_shape(0)
    nuc_idx = qc["nuclear_index"]
    nuc_range = runs.hist_percentiles(hist_in[nuc_idx], [1, 99.8])
    mem_rows = channels[(channels["segmentation"] == "membrane") & (channels["status"] == "ok")]
    mem = [(int(c), ch_qc.loc[int(c), "display_low"], ch_qc.loc[int(c), "display_high"]) for c in mem_rows["index"]]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    status = "smoke-tested only" if "mesmer" not in methods else "built, never run"
    rep = report.Report("02", "Segment nuclei and whole cells", sid, status, args.run_note)
    timings, results, skipped = {}, {}, {}

    # -- nuclei --------------------------------------------------------------------------------
    t0 = time.time()
    print(f"[{sid}] StarDist nuclei on {H:,}×{W:,} px")
    raw, scale = run_nuclei(handle, nuc_idx, *nuc_range, out_dir, args)
    n_raw = int(raw.max())
    cnt, cy, cx, _ = label_stats(raw, H, W, n_raw)
    ids = np.arange(n_raw + 1)
    inside = (cnt > 0)
    ry = np.minimum((np.nan_to_num(cy) / f).astype(int), mask_w.shape[0] - 1)
    rx = np.minimum((np.nan_to_num(cx) / f).astype(int), mask_w.shape[1] - 1)
    inside &= mask_w[ry, rx]
    inside[0] = False
    min_px = args.min_nucleus_um2 / px ** 2
    small = inside & (cnt < min_px)
    keep = ids[inside & ~small]
    nuclei = filter_relabel(raw, keep, H, W, out_dir / "nuclei.npy")
    del raw
    (out_dir / "nuclei_raw.npy").unlink()
    n_nuc = len(keep)
    timings["nuclei"] = time.time() - t0
    rep.check("ok" if n_nuc else "fail",
              f"StarDist found {n_raw:,} nuclei; {n_nuc:,} kept after removing {int((cnt[1:] > 0).sum() - inside[1:].sum()):,} "
              f"outside the analysis area and {int(small.sum()):,} smaller than {args.min_nucleus_um2} µm².")
    label_sets = [("nuclei", nuclei)]

    # -- whole-cell methods --------------------------------------------------------------------------
    if "cellsam" in methods:
        if not runs.deepcell_token():
            skipped["cellsam"] = ("DEEPCELL_ACCESS_TOKEN is not set; CellSAM downloads its weights from "
                                  "users.deepcell.org (register there and set the token yourself)")
        else:
            try:
                t0 = time.time()
                raw_name = "cells_cellsam_raw.npy" if args.cellsam_merge_nested else "cells_cellsam.npy"
                cellsam_raw = run_cellsam(handle, nuc_idx, nuc_range, mem, mask_w, f, px,
                                          out_dir / raw_name, args)
                timings["cellsam"] = time.time() - t0
                if args.stop_after_cellsam:
                    # Split-job mode: the GPU's work is done and everything that
                    # follows -- the nested-label merge above all -- is single-
                    # threaded CPU. Holding a GPU through it wastes the allocation
                    # and trips HiPerGator's idle-GPU policy (0% for an hour), which
                    # killed job 43518458 after CellSAM had already succeeded. Stop
                    # here and let a CPU job finish with --resume, which reads
                    # cells_cellsam_raw.progress.json and skips straight past
                    # segmentation.
                    print(f"  --stop-after-cellsam: wrote {out_dir / raw_name}. "
                          f"Finish on CPU with the same --cellsam-block-px plus --resume.",
                          flush=True)
                    return
                if args.cellsam_merge_nested:
                    t0 = time.time()
                    merged, n_merged = merge_nested_labels(
                        cellsam_raw, H, W, px, out_dir / "cells_cellsam.npy",
                        threshold=args.merge_threshold, dilation_um=args.merge_dilation_um,
                        search_um=args.merge_search_um)
                    timings["cellsam_merge"] = time.time() - t0
                    results["cellsam"] = merged
                    rep.check("info", f"Nested-label merge folded {n_merged:,} predicted cells into a "
                                      f"neighbour (≥{args.merge_threshold:.0%} of the smaller one's area "
                                      f"inside the larger one dilated by {args.merge_dilation_um:g} µm; "
                                      f"candidates within {args.merge_search_um:g} µm of each other).")
                else:
                    results["cellsam"] = cellsam_raw
                label_sets.append(("cellsam", results["cellsam"]))
            except ImportError as exc:
                skipped["cellsam"] = f"CellSAM not importable here ({exc}); use envs/environment-cellsam.yml"
    if "expansion" in methods:
        t0 = time.time()
        exp, d = run_expansion(nuclei, px, args, out_dir / "cells_expansion.npy")
        results["expansion"] = exp
        timings["expansion"] = time.time() - t0
        label_sets.append((f"expansion ({args.expand_um} µm)", exp))
    if "propagation" in methods:
        if not mem:
            skipped["propagation"] = "no membrane channels (segmentation=membrane, status=ok) in the panel"
        else:
            t0 = time.time()
            prop, g = run_propagation(nuclei, handle, mem, mask_w, f, px, args, out_dir / "cells_propagation.npy")
            results["propagation"] = prop
            timings["propagation"] = time.time() - t0
            label_sets.append((f"propagation (≤{args.max_grow_um} µm)", prop))
    if "mesmer" in methods:
        if not runs.deepcell_token():
            skipped["mesmer"] = "DEEPCELL_ACCESS_TOKEN is not set (register at users.deepcell.org and set it yourself)"
        else:
            try:
                t0 = time.time()
                results["mesmer"] = run_mesmer(handle, nuc_idx, nuc_range, mem, mask_w, f, px,
                                               out_dir / "cells_mesmer.npy")
                timings["mesmer"] = time.time() - t0
                label_sets.append(("mesmer", results["mesmer"]))
            except ImportError as exc:
                skipped["mesmer"] = f"DeepCell not importable here ({exc}); use envs/environment-mesmer.yml on HiPerGator"
    for m, why in skipped.items():
        rep.check("info", f"Method <strong>{m}</strong> not run: {why}.")

    # -- per-cell tables ----------------------------------------------------------------------------------
    nuc_cnt, nuc_cy, nuc_cx, _ = label_stats(nuclei, H, W, n_nuc)

    def region_of(yv, xv):
        yy = np.minimum((np.nan_to_num(yv) / f).astype(int), regions_w.shape[0] - 1)
        xx = np.minimum((np.nan_to_num(xv) / f).astype(int), regions_w.shape[1] - 1)
        return regions_w[yy, xx]

    tables = {}
    nuc_df = pd.DataFrame({"cell_id": np.arange(1, n_nuc + 1), "x_px": nuc_cx[1:], "y_px": nuc_cy[1:],
                           "x_um": nuc_cx[1:] * px, "y_um": nuc_cy[1:] * px,
                           "nucleus_area_um2": nuc_cnt[1:] * px ** 2,
                           "region": region_of(nuc_cy[1:], nuc_cx[1:])})
    nuc_df.to_csv(out_dir / "cells_nuclei.csv", index=False)
    tables["nuclei"] = nuc_df
    other = results.get("propagation") if "expansion" in results else None
    for name, lab in results.items():
        n = int(lab.max())
        cnt_m, cy_m, cx_m, inter = label_stats(lab, H, W, n, other=other if name == "expansion" else None)
        df = pd.DataFrame({"cell_id": np.arange(1, n + 1), "x_px": cx_m[1:], "y_px": cy_m[1:],
                           "x_um": cx_m[1:] * px, "y_um": cy_m[1:] * px, "area_um2": cnt_m[1:] * px ** 2,
                           "region": region_of(cy_m[1:], cx_m[1:])})
        if name in ("expansion", "propagation"):  # share ids with nuclei
            df["nucleus_area_um2"] = nuc_cnt[1:n + 1] * px ** 2
            df["nc_ratio"] = df["nucleus_area_um2"] / df["area_um2"]
        if inter is not None:
            prop_cnt = label_stats(other, H, W, n)[0]
            df["iou_vs_propagation"] = inter[1:] / (cnt_m[1:] + prop_cnt[1:] - inter[1:])
        df = df[df["area_um2"] > 0]
        df.to_csv(out_dir / f"cells_{name}.csv", index=False)
        tables[name] = df

    # -- save label images ------------------------------------------------------------------------------
    for name, lab in [("nuclei", nuclei)] + list(results.items()):
        save_labels(out_dir / (f"cells_{name}.tif" if name != "nuclei" else "nuclei.tif"), lab)

    # -- CellTune export ------------------------------------------------------------------------------
    # celltune.org/documentation/getting-started/data-preparation/segmentation: a single 2D TIFF, cell
    # IDs as an unsigned integer type, 0 = background, named "<image name>_segmentation_labels.tif".
    # CellTune's own project setup reads the original image itself (this pipeline hands it the qptiff
    # unchanged -- QPTIFF is one of the pyramidal formats CellTune's Images step documents directly), so
    # this file is the one thing that actually needs reshaping to match CellTune's convention. It is
    # written under our own out_dir, not CellTune's CellTune_Data tree (whose location on this machine
    # we don't know): copy or symlink it into <project>/Segmentations/ once that project exists.
    if args.celltune_export:
        lab = nuclei if args.celltune_method == "nuclei" else results.get(args.celltune_method)
        if lab is None:
            rep.check("warn", f"--celltune-export requested method '{args.celltune_method}' but it did not "
                              f"run for {sid} (ran: {', '.join(['nuclei'] + list(results))}); no CellTune "
                              f"file was written.")
        else:
            celltune_dir = out_dir / "celltune"
            celltune_dir.mkdir(exist_ok=True)
            ct_path = celltune_dir / f"{sid}_segmentation_labels.tif"
            save_labels(ct_path, np.asarray(lab).astype(np.uint32))
            rep.check("info", f"CellTune-formatted segmentation written to <code>{ct_path}</code> (uint32, "
                              f"cell IDs with 0 = background, from method '{args.celltune_method}'). Copy or "
                              f"symlink it into your CellTune project's <code>Segmentations/</code> folder as "
                              f"<code>{sid}_segmentation_labels.tif</code>; point CellTune's own Images step "
                              f"at the original qptiff directly, unmodified.")

    # -- report: summary checks ---------------------------------------------------------------------------
    rep.h2("What was segmented")
    rep.explain("Nuclei come from StarDist on the nuclear stain, scaled so its 1st–99.8th in-tissue percentiles map "
                "to 0–1 (one rule for the whole slide, not per tile). Whole cells are grown from those nuclei in two "
                "ways. <strong>Expansion</strong> adds the same margin to every nucleus, stopping where neighbours "
                "meet: fast and predictable, but it ignores where membranes really are. <strong>Propagation</strong> "
                "floods outward over the membrane image so that neighbouring cells split along the bright membrane "
                "between them, with growth capped at a maximum distance. Both keep one cell per nucleus, so "
                "they can be compared cell by cell.")
    summ = []
    for name, df in tables.items():
        area = df["area_um2"] if "area_um2" in df else df["nucleus_area_um2"]
        row_s = {"method": name, "cells": len(df), "median area µm²": area.median(),
                 "IQR area µm²": f"{area.quantile(.25):.0f}–{area.quantile(.75):.0f}",
                 "median equiv. diameter µm": 2 * np.sqrt(area.median() / np.pi),
                 "runtime s": timings.get(name, np.nan)}
        if "nc_ratio" in df:
            row_s["median N/C"] = df["nc_ratio"].median()
        summ.append(row_s)
    rep.table(pd.DataFrame(summ), caption="Median equivalent diameter is the diameter of a circle with the median "
                                          "area. N/C is nucleus area divided by cell area.")
    dens = nuc_df.groupby("region").size().rename("nuclei").to_frame()
    reg_csv = base / "01_image_qc" / "tissue_regions.csv"
    if reg_csv.exists():
        reg = pd.read_csv(reg_csv).set_index("region")
        dens["area mm²"] = reg["area_mm2"]
        dens["nuclei per mm²"] = dens["nuclei"] / dens["area mm²"]
    rep.table(dens.reset_index(), caption="Nuclei per tissue piece (from stage 01).")
    med_d = 2 * np.sqrt(nuc_df["nucleus_area_um2"].median() / np.pi)
    rep.check("ok" if 4 <= med_d <= 12 else "warn",
              f"Median nucleus diameter {med_d:.1f} µm (typical nuclei are about 5–10 µm; far outside that range "
              f"usually means a wrong pixel size or StarDist scale).")

    # -- morphology distributions -----------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    bins = np.linspace(0, 400, 81)
    axs[0].hist(nuc_df["nucleus_area_um2"], bins=bins, histtype="step", lw=1.4, label="nuclei")
    for name in results:
        axs[0].hist(tables[name]["area_um2"], bins=bins, histtype="step", lw=1.4, label=name)
    axs[0].set_xlabel("area (µm²)"), axs[0].set_ylabel("cells"), axs[0].legend(fontsize=8)
    for name in ("expansion", "propagation"):
        if name in tables:
            axs[1].hist(tables[name]["nc_ratio"].clip(0, 1), bins=40, histtype="step", lw=1.4, label=name)
    axs[1].set_xlabel("nucleus / cell area"), axs[1].legend(fontsize=8)
    if "iou_vs_propagation" in tables.get("expansion", {}):
        axs[2].hist(tables["expansion"]["iou_vs_propagation"].dropna(), bins=40, color=report.PALETTE["hema"])
        axs[2].set_xlabel("overlap of the two outlines (IoU)")
    else:
        axs[2].axis("off")
    fig.tight_layout()
    rep.figure(fig, "Left: area distributions. Centre: how much of each cell the nucleus fills. Right: for each "
                    "nucleus, the overlap between its expansion and propagation outlines (intersection over union: "
                    "1 = identical, 0.5 = they share half their combined area).")

    if "iou_vs_propagation" in tables.get("expansion", {}):
        iou = tables["expansion"]["iou_vs_propagation"]
        ratio = tables["propagation"].set_index("cell_id")["area_um2"] / tables["expansion"].set_index("cell_id")["area_um2"]
        rep.check("info", f"Expansion vs propagation: median outline overlap (IoU) {iou.median():.2f}; "
                          f"{(iou < 0.5).mean():.0%} of cells overlap less than half; propagation cells are "
                          f"{ratio.median():.2f}× the expansion area (median).")

    # -- CellSAM vs the StarDist-based methods ------------------------------------------------------------------
    cross = []
    if "cellsam" in results:
        for other_name in ("expansion", "propagation"):
            if other_name in results:
                cross.append(compare_two_methods("cellsam", tables["cellsam"], results["cellsam"],
                                                 other_name, tables[other_name], results[other_name],
                                                 H, W, px, args, seed=args.seed))
    if cross:
        rep.h2("CellSAM against the StarDist-based cells")
        rep.explain("CellSAM finds whole cells directly from the image; the other methods grow cells from StarDist "
                    "nuclei. Their labels are independent, so cells are paired one-to-one by centre distance "
                    f"(within {args.match_um:g} µm) and then compared pixel by pixel. <strong>Expected by chance</strong> "
                    "is the pairing rate after shifting one set 20–60 µm in random directions: in dense tissue a "
                    "cell almost always has <em>some</em> neighbour nearby, so a high pairing rate means little on "
                    "its own. <strong>IoU</strong> is the shared area over the combined area of a pair: 1 is identical "
                    "outlines, 0.5 means they share half. Neither method is ground truth.")
        show_cols = [c for c in cross[0] if not c.startswith("_")]
        rep.table(pd.DataFrame([{k: c[k] for k in show_cols} for c in cross]), floatfmt="{:,.3g}")
        fig, ax = plt.subplots(figsize=(5.2, 3))
        for c in cross:
            if c["_ious"].size:
                ax.hist(c["_ious"], bins=np.linspace(0, 1, 41), histtype="step", lw=1.5,
                        label=f"cellsam vs {c['method B']}")
        ax.set_xlabel("IoU of paired cells"), ax.set_ylabel("pairs"), ax.legend(fontsize=8)
        fig.tight_layout()
        rep.figure(fig, "Outline agreement for paired cells. A peak near 1 means the two methods draw nearly the "
                        "same cells; a long tail towards 0 means they disagree about where cells end.")
        for c in cross:
            other = c["method B"]
            n_a = c["cellsam cells"]
            n_b = c[other + " cells"]
            tight_pct = c["% of " + other + " paired"]
            margin = c["margin over chance (pts)"]
            loose_pct = c[f"% of {other} paired within {args.match_um:g} µm"]
            loose_chance = c[f"chance % at {args.match_um:g} µm"]
            area_ratio = c["median area cellsam/" + other]
            rep.check("info" if margin >= 20 else "warn",
                      f"CellSAM found {n_a:,} cells against {n_b:,} from {other}. Within {args.tight_um:g} µm, "
                      f"{tight_pct:.0f}% of {other} cells pair with a CellSAM cell against {c['expected by chance %']:.0f}% "
                      f"by chance — a margin of {margin:+.0f} points. Median outline overlap IoU "
                      f"{c['median IoU of pairs']:.2f}; CellSAM cells are {area_ratio:.2f}× their area."
                      + ("" if margin >= 20 else " The margin over chance is thin, so this agreement is weak evidence."))
            rep.check("info",
                      f"At the looser {args.match_um:g} µm radius the same comparison gives {loose_pct:.0f}% paired, but "
                      f"chance alone gives {loose_chance:.0f}% in tissue this dense, so read the {args.tight_um:g} µm "
                      f"number above and treat this one as an upper bound only.")

    # -- ROI gallery: random + where the methods disagree most ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    roi_px = int(round(args.roi_um / px))
    cand = nuc_df[(nuc_df["x_px"] > roi_px) & (nuc_df["x_px"] < W - roi_px) &
                  (nuc_df["y_px"] > roi_px) & (nuc_df["y_px"] < H - roi_px)]
    rois = []
    for _, c in cand.sample(n=min(args.n_rois, len(cand)), random_state=args.seed).iterrows():
        rois.append((int(c.y_px - roi_px / 2), int(c.x_px - roi_px / 2), "random"))
    if "iou_vs_propagation" in tables.get("expansion", {}):
        e = tables["expansion"].copy()
        e["gy"], e["gx"] = (e["y_px"] // roi_px).astype(int), (e["x_px"] // roi_px).astype(int)
        g = e.groupby(["gy", "gx"]).agg(iou=("iou_vs_propagation", "median"), n=("cell_id", "size"))
        g = g[g["n"] >= 15].sort_values("iou").head(3)
        for (gy, gx), r in g.iterrows():
            rois.append((int(min(gy * roi_px, H - roi_px)), int(min(gx * roi_px, W - roi_px)),
                         f"low overlap (IoU {r.iou:.2f})"))
    if rois:
        titles = ["nuclear (blue) + membrane (green)"] + [n for n, _ in label_sets]
        rep.h2("Outlines on the image")
        rep.explain(f"{args.roi_um:.0f} µm squares: {sum(r[2] == 'random' for r in rois)} chosen at random (seed "
                    f"{args.seed}, recorded so the same squares come back on a rerun) and up to 3 where expansion "
                    "and propagation disagree most. Judge which outline follows the membrane.")
        rep.figure(roi_panel(handle, nuc_idx, nuc_range, mem, label_sets, rois, roi_px, titles),
                   "White: nuclei. Yellow: expansion. Magenta: propagation.")

    # -- hand-drawn ground truth: the only comparison where one side is right -------------------------------------
    if args.ground_truth:
        from akoyalib import groundtruth as gtmod
        gt = gtmod.read_qupath_geojson(args.ground_truth)
        rep.h2("Scored against hand-drawn cells")
        rep.explain(
            f"Reference: <code>{pathlib.Path(args.ground_truth).name}</code> — {len(gt.cells):,} outlines you drew "
            f"across {len(gt.regions)} labelled field(s). Unlike the method-versus-method comparison above, one side "
            "here is ground truth, so the numbers are F1 at a stated outline-overlap (IoU) threshold rather than "
            "centre distances against a chance baseline: two unrelated cells do not overlap by half their combined "
            "area, so no chance correction is needed. Scoring happens only inside the fields you drew, because that "
            "is where labelling is exhaustive — a cell found outside them is not a false positive. Cells clipped by "
            "a field's border are dropped from both sides. F1 at IoU 0.5 is the number the published segmentation "
            "benchmarks report, so it can be read next to them.")

        gt_rows, gt_detail = [], {}
        for name, lab in [("nuclei", results.get("nuclei"))] + [(n, l) for n, l in label_sets]:
            if lab is None:
                continue
            try:
                out = gtmod.score_in_regions(gt, lab)
            except ValueError as e:
                rep.check("warn", f"Could not score {name} against the hand labels: {e}")
                break
            pooled = out["pooled"]
            gt_detail[name] = out
            row = {"method": name, "hand-drawn cells": pooled["n_ground_truth"],
                   "found": pooled["n_predicted"],
                   "count ratio": pooled["count_ratio"],
                   "median IoU of matches": pooled["median_matched_IoU"],
                   "split": pooled["split_ground_truth_cells"],
                   "merged": pooled["merged_predictions"]}
            for t, r in pooled["per_threshold"].items():
                row[f"F1@{t:g}"] = r["F1"]
            gt_rows.append(row)

        if gt_rows:
            gt_df = pd.DataFrame(gt_rows)
            rep.table(gt_df, floatfmt="{:,.3g}")
            best = gt_df.loc[gt_df["F1@0.5"].idxmax()]
            spread = gt_df["F1@0.5"].max() - gt_df["F1@0.5"].min()
            rep.check("info" if spread >= 0.05 else "warn",
                      f"Best F1 at IoU 0.5: {best['method']} at {best['F1@0.5']:.2f}, over {len(gt_df)} methods "
                      f"spanning {spread:.2f}."
                      + ("" if spread >= 0.05 else " That spread is small enough that this ground truth cannot "
                         "separate the methods; label more crowded fields, or choose on speed and stability "
                         "instead and say so."))
            rep.check("info",
                      "Split and merged count the errors F1 hides: split is hand-drawn cells that one method cut "
                      "into pieces, merged is predictions covering more than one hand-drawn cell.")
            per_region = pd.DataFrame([{"method": m, "field": r["region"], "cells": r["n_ground_truth"],
                                        "F1@0.5": r["per_threshold"][0.5]["F1"],
                                        "median IoU": r["median_matched_IoU"]}
                                       for m, out in gt_detail.items() for r in out["per_region"]])
            rep.h3("By field")
            rep.explain("Pooled numbers hide where methods differ. Crowded fields are where they diverge, so read "
                        "these before choosing.")
            rep.table(per_region, floatfmt="{:,.3g}")
            gt_df.to_csv(out_dir / "ground_truth_scores.csv", index=False)
            per_region.to_csv(out_dir / "ground_truth_scores_by_field.csv", index=False)

    # -- reference comparison -------------------------------------------------------------------------------------
    ref_summary = None
    if args.reference:
        ref = pd.read_csv(args.reference)
        rep.h2("Comparison with an existing segmentation")
        rep.explain(f"Reference: <code>{pathlib.Path(args.reference).name}</code>. {args.reference_note} "
                    f"Each reference cell counts as found when one of our nuclei lies within {args.match_um} µm of "
                    "its centre (one-to-one matching, closest pairs first). Only this direction is meaningful when "
                    "the reference was filtered after segmentation: cells it dropped cannot count against us, so "
                    "the opposite rate is not reported.")
        ours = nuc_df[["x_px", "y_px"]].to_numpy()
        rows_r, pairs, dists, aligned = [], [], [], []
        groups = ref.groupby("region") if "region" in ref else [("all", ref)]
        for reg_name, rdf in groups:
            rxy = rdf[["x_px", "y_px"]].to_numpy(float)
            swapped, z = False, float("nan")
            if args.reference_align == "translate":
                ox, oy, swapped, z = align_translate(ours, rxy, H, W, bin_px=max(1, int(round(2 / px))),
                                                     match_px=args.match_um / px)
                rxy = (rxy[:, ::-1] if swapped else rxy) + [ox, oy]
            else:
                ox = oy = 0.0
            ri, oi, dist = match_one_to_one(ours, rxy, args.match_um / px)
            chance = chance_rate(ours, rxy, args.match_um / px, shift_px=(20 / px, 60 / px), seed=args.seed)
            ti, _, _ = match_one_to_one(ours, rxy, args.tight_um / px)
            chance_t = chance_rate(ours, rxy, args.tight_um / px, shift_px=(20 / px, 60 / px), seed=args.seed)
            our_reg = int(pd.Series(nuc_df["region"].to_numpy()[oi]).mode().iloc[0]) if len(oi) else -1
            rows_r.append({"reference region": reg_name, "our tissue piece": our_reg,
                           "offset x,y (µm)": f"{ox * px:.0f}, {oy * px:.0f}" + (" (x/y swapped)" if swapped else ""),
                           "reference cells": len(rdf),
                           "our nuclei in piece": int((nuc_df["region"] == our_reg).sum()) if our_reg > 0 else 0,
                           f"found ≤{args.tight_um:g} µm %": 100 * len(ti) / len(rdf),
                           f"chance ≤{args.tight_um:g} µm %": 100 * chance_t,
                           f"found ≤{args.match_um:g} µm %": 100 * len(ri) / len(rdf),
                           f"chance ≤{args.match_um:g} µm %": 100 * chance,
                           "median distance µm": np.median(dist) * px if len(dist) else np.nan,
                           "alignment peak z": z})
            dists.append(dist * px)
            found_t = np.zeros(len(rxy), bool)
            found_t[ti] = True
            aligned.append(pd.DataFrame({"x": rxy[:, 0], "y": rxy[:, 1], "found": found_t}))
            if "area_px" in rdf:
                pairs.append(pd.DataFrame({"ref_area_um2": rdf["area_px"].to_numpy()[ri] * px ** 2,
                                           "cell_id": nuc_df["cell_id"].to_numpy()[oi],
                                           "dist_um": dist * px}))
        rtab = pd.DataFrame(rows_r)
        rep.table(rtab, caption=f"Found % = share of reference cells with one of our nuclei within the given "
                                f"distance (one-to-one). Chance % = the same after shifting the reference 20–60 µm in "
                                f"random directions. In dense tissue chance is high at {args.match_um:g} µm, so the "
                                f"tight {args.tight_um:g} µm columns are the informative ones. The alignment peak z is "
                                f"kept for the record but is not well calibrated (it reads low even for correct "
                                f"alignments); judge alignment by the tight-distance columns and the distance histogram.")
        pieces = rtab["our tissue piece"].tolist()
        if len(set(pieces)) < len(pieces):
            rep.check("fail", "Two reference regions landed on the same tissue piece: the alignment is not "
                              "trustworthy, so the numbers below are not interpretable.")
        ft, ct = f"found ≤{args.tight_um:g} µm %", f"chance ≤{args.tight_um:g} µm %"
        bad = rtab[rtab[ft] < 3 * rtab[ct]]
        if len(bad):
            rep.check("fail", f"Alignment not convincing for reference region(s) "
                              f"{', '.join(map(str, bad['reference region']))}: tight matches are under 3× chance.")
        w = rtab["reference cells"] / rtab["reference cells"].sum()
        tot, tot_ch = float((rtab[ft] * w).sum() / 100), float((rtab[ct] * w).sum() / 100)
        tot5 = float((rtab[f"found ≤{args.match_um:g} µm %"] * w).sum() / 100)
        rep.check("ok" if tot >= 0.8 else "warn",
                  f"{tot:.0%} of the {int(rtab['reference cells'].sum()):,} reference cells have one of our nuclei "
                  f"within {args.tight_um:g} µm (chance: {tot_ch:.0%}); {tot5:.0%} within {args.match_um:g} µm.")
        n_ref = int(rtab["reference cells"].sum())
        n_ours = int(rtab["our nuclei in piece"].sum())
        if n_ours < 0.9 * n_ref:
            rep.check("warn", f"We found {n_ours:,} nuclei in these tissue pieces against {n_ref:,} reference cells "
                              f"({n_ours / n_ref - 1:+.0%}), even though the reference was already filtered. Dense "
                              f"areas may be under-segmented; see the gallery and consider a lower --prob-thresh.")
        al = pd.concat(aligned).reset_index(drop=True)
        # nuclear-stain value at every reference centre: are our misses in bright / saturated areas?
        al["nuclear_value"] = np.nan
        yi = np.clip(al["y"].round().astype(int), 0, H - 1).to_numpy()
        xi = np.clip(al["x"].round().astype(int), 0, W - 1).to_numpy()
        for y0b, y1b, band in handle.iter_bands(nuc_idx, BLOCK):
            sel = (yi >= y0b) & (yi < y1b)
            al.loc[sel, "nuclear_value"] = band[yi[sel] - y0b, xi[sel]]
        # crowding: reference cells within 10 µm of each reference cell
        al["neighbours_10um"] = [len(n) - 1 for n in cKDTree(al[["x", "y"]].to_numpy())
                                 .query_ball_point(al[["x", "y"]].to_numpy(), r=10 / px)]
        al.to_csv(out_dir / "reference_matches.csv", index=False)
        top = float(np.iinfo(handle.dtype).max) if np.issubdtype(handle.dtype, np.integer) else np.inf
        hit, mis = al[al["found"]], al[~al["found"]]
        med_hit, med_miss = hit["nuclear_value"].median(), mis["nuclear_value"].median()
        sat_hit = float((hit["nuclear_value"] >= top).mean() * 100)
        sat_miss = float((mis["nuclear_value"] >= top).mean() * 100)
        brighter = med_miss > 1.2 * med_hit or sat_miss > 2 * max(sat_hit, 0.5)
        # sensitivity of our detection to crowding: found rate by neighbour count. Compare the
        # most crowded cells with moderately crowded ones (medians of neighbour counts are too
        # coarse to see this: on the SPACEc demo both were 5 while the rate fell from 79% to 52%).
        by_nb = al.groupby(al["neighbours_10um"].clip(upper=8))["found"].mean()
        rate_mod = al.loc[al["neighbours_10um"].between(1, 3), "found"].mean()
        rate_dense = al.loc[al["neighbours_10um"] >= 7, "found"].mean()
        crowded = bool(rate_dense < rate_mod - 0.10)
        nb_hit, nb_miss = f"{rate_mod:.0%}", f"{rate_dense:.0%}"
        explain = ("brightness/saturation" if brighter else "") + (" and " if brighter and crowded else "") + \
                  ("crowding" if crowded else "")
        rep.check("warn" if (brighter or crowded) else "info",
                  f"Reference cells we missed vs found: nuclear stain median {med_miss:.0f} vs {med_hit:.0f} "
                  f"({sat_miss:.1f}% vs {sat_hit:.1f}% saturated). Found rate {nb_hit} for cells with 1–3 "
                  f"reference neighbours within 10 µm, {nb_miss} for cells with 7 or more. " +
                  (f"The misses are associated with {explain}." if explain else
                   "Neither brightness nor crowding separates misses from finds."))
        rep.table(pd.DataFrame({"reference neighbours within 10 µm": [f"{int(k)}" + ("+" if k == 8 else "")
                                                                      for k in by_nb.index],
                                "reference cells": al.groupby(al["neighbours_10um"].clip(upper=8)).size().to_numpy(),
                                f"found by us ≤{args.tight_um:g} µm": [f"{v:.0%}" for v in by_nb.to_numpy()]}),
                  caption="How often we find a reference cell, by how crowded it is. A falling rate means our "
                          "nuclear segmentation struggles where cells are packed.")
        miss = al[~al["found"]].copy()
        if len(miss):
            miss["gy"], miss["gx"] = (miss["y"] // roi_px).astype(int), (miss["x"] // roi_px).astype(int)
            worst = miss.groupby(["gy", "gx"]).size().sort_values(ascending=False).head(3)
            fig, axs = plt.subplots(1, len(worst), figsize=(3.6 * len(worst), 3.8))
            for a, ((gy, gx), n_miss) in zip(np.atleast_1d(axs), worst.items()):
                y0, x0 = int(min(gy * roi_px, H - roi_px)), int(min(gx * roi_px, W - roi_px))
                y1, x1 = y0 + roi_px, x0 + roi_px
                nuc_img = np.clip((handle.read(nuc_idx, 0, y=(y0, y1), x=(x0, x1)).astype(np.float32)
                                   - nuc_range[0]) / (nuc_range[1] - nuc_range[0]), 0, 1)
                memb = membrane_block(handle, mem, y0, y1, x0, x1) if mem else np.zeros_like(nuc_img)
                rgb = np.clip(np.dstack([memb * 0.2, memb, nuc_img]), 0, 1) * 0.85
                rgb[segmentation.find_boundaries(np.asarray(nuclei[y0:y1, x0:x1]), mode="inner")] = (1, 1, 1)
                a.imshow(rgb)
                inroi = al[(al["x"] >= x0) & (al["x"] < x1) & (al["y"] >= y0) & (al["y"] < y1)]
                for fnd, mk, col in ((True, "o", "#2EC4B6"), (False, "x", "#FF3B30")):
                    s = inroi[inroi["found"] == fnd]
                    a.scatter(s["x"] - x0, s["y"] - y0, marker=mk, s=14, c=col, linewidths=1.1)
                a.set_title(f"{n_miss} reference cells without a nucleus ≤{args.tight_um:g} µm", fontsize=8)
                a.set_xticks([]), a.set_yticks([])
            fig.tight_layout()
            rep.figure(fig, f"Where the two segmentations disagree most ({args.roi_um:.0f} µm squares with the most "
                            f"unmatched reference cells). White: our nuclei. Teal circles: reference cells we matched. "
                            f"Red crosses: reference cells with none of our nuclei within {args.tight_um:g} µm. Neither "
                            f"segmentation is ground truth: a red cross over a visible nucleus is our miss; one over "
                            f"membrane with no nucleus may be a cell whose nucleus lies outside this section, which "
                            f"a nucleus-first method cannot find.")
        d_all = np.concatenate(dists) if dists else np.array([])
        if d_all.size:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.hist(d_all, bins=np.linspace(0, args.match_um, 26), color=report.PALETTE["if"])
            ax.set_xlabel("distance, reference centre to our nucleus centre (µm)"), ax.set_ylabel("matched pairs")
            fig.tight_layout()
            rep.figure(fig, "Distances of matched pairs. True matches pile up near zero; chance matches spread "
                            "evenly across the disc, so they rise towards the right edge.")
        ref_summary = {"found_fraction_tight": tot, "chance_fraction_tight": tot_ch, "found_fraction": tot5,
                       "tight_um": args.tight_um, "match_um": args.match_um, "per_region": rows_r}
        if pairs:
            pr = pd.concat(pairs)
            pr = pr[pr["dist_um"] <= args.tight_um]  # tight pairs only: chance pairs would blur the comparison
            fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
            comp = []
            axs[0].hist(pr["ref_area_um2"], bins=bins, histtype="step", lw=1.6, color="k", label="reference")
            for name in results:
                m = pr.merge(tables[name][["cell_id", "area_um2"]], on="cell_id")
                axs[0].hist(m["area_um2"], bins=bins, histtype="step", lw=1.4, label=name)
                axs[1].hist(np.log2(m["area_um2"] / m["ref_area_um2"]), bins=np.linspace(-3, 3, 61),
                            histtype="step", lw=1.4, label=name)
                comp.append({"method": name, "matched cells": len(m),
                             "median area ratio (ours / reference)": float(np.median(m["area_um2"] / m["ref_area_um2"])),
                             "within ±25% of reference area": float(np.mean(np.abs(m["area_um2"] / m["ref_area_um2"] - 1) <= 0.25))})
            axs[0].set_xlabel("cell area (µm²), matched cells"), axs[0].legend(fontsize=8)
            axs[1].axvline(0, color="k", lw=0.8)
            axs[1].set_xlabel("log2(our area / reference area)"), axs[1].legend(fontsize=8)
            fig.tight_layout()
            rep.figure(fig, f"Whole-cell areas of cell pairs matched within {args.tight_um:g} µm. Right: 0 means the "
                            f"same area; +1 means ours is twice the reference, −1 half.")
            rep.table(pd.DataFrame(comp), floatfmt="{:,.3g}")
            ref_summary["area_comparison"] = comp

    seg = {"sample_id": sid, "methods_run": ["nuclei"] + list(results), "methods_skipped": skipped,
           "n_nuclei": n_nuc, "n_nuclei_raw": n_raw, "stardist_model": args.stardist_model,
           "stardist_scale": scale, "expand_um": args.expand_um, "max_grow_um": args.max_grow_um,
           "membrane_channels": [channels.loc[c, "marker"] for c, _, _ in mem],
           "nuclear_range": nuc_range, "timings_s": timings, "reference": ref_summary}
    runs.write_json(out_dir / "segment.json", seg)
    for p in out_dir.glob("*.npy"):
        p.unlink()
    rep.set_provenance(
        inputs={"image": f"{ingest['image']} (sha256 {ingest['sha256']})", "analysis mask": "01_image_qc/analysis_mask.npz",
                "reference": args.reference or "(none)"},
        params={k: v for k, v in vars(args).items() if k not in ("samples", "out", "sample_id")},
        outputs={p.name: p for p in sorted(out_dir.iterdir())}, repo_dir=runs.REPO_DIR)
    path = rep.write(runs.report_path(args.out, sid, STAGE))
    handle.close()
    print(f"[{sid}] {n_nuc:,} nuclei; report: {path}")


def main():
    p = runs.base_parser(__doc__)
    p.add_argument("--methods", default="cellsam,expansion",
                   help="comma list from: cellsam (default whole-cell method), expansion, propagation, "
                        "mesmer (optional). StarDist nuclei always run.")
    p.add_argument("--cellsam-device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                   help="where CellSAM runs; auto picks cuda, then mps (Apple), then cpu")
    p.add_argument("--cellsam-bbox-threshold", type=float, default=0.4,
                   help="CellSAM cell-detection confidence; lower finds more cells (its main precision/recall dial)")
    p.add_argument("--cellsam-block-px", type=int, default=512,
                   help="block size for CellSAM; smaller suits densely packed tissue")
    p.add_argument("--cellsam-no-normalize", action="store_true",
                   help="feed CellSAM our globally scaled images instead of its own per-block normalisation "
                        "(its normalisation is per block, which can differ between blocks)")
    p.add_argument("--cellsam-merge-nested", action="store_true",
                   help="after CellSAM, fold predicted cells that are mostly swallowed by a dilated neighbour "
                        "into it -- CellSAM sometimes returns two overlapping labels for one true cell. Ported "
                        "from CellSAM_local's hand-tuned merge_nested_labels_radius, already checked by eye on "
                        "PS88 crops; here it runs block by block so it scales to a whole slide.")
    p.add_argument("--merge-threshold", type=float, default=0.9,
                   help="nested-label merge: fraction of the smaller label's area that must lie inside the "
                        "larger one (dilated by --merge-dilation-um) before it is folded in")
    p.add_argument("--merge-dilation-um", type=float, default=5.0,
                   help="nested-label merge: how far to grow the larger label before testing containment (µm)")
    p.add_argument("--merge-search-um", type=float, default=50.0,
                   help="nested-label merge: only label pairs within this distance of each other are compared (µm)")
    p.add_argument("--stop-after-cellsam", action="store_true",
                   help="exit once CellSAM has written its labels, before the nested-label merge and "
                        "everything downstream. For splitting a GPU CellSAM job from the CPU work that "
                        "follows it; finish with the same --cellsam-block-px plus --resume.")
    p.add_argument("--celltune-export", action="store_true",
                   help="also write the chosen segmentation as a CellTune-format label TIFF (uint32, "
                        "0=background) under <out>/<sample>/02_segment/celltune/<sample>_segmentation_labels.tif. "
                        "See celltune.org/documentation/getting-started/data-preparation/segmentation.")
    p.add_argument("--celltune-method", default="cellsam",
                   help="which segmentation to export for CellTune: one of --methods, or 'nuclei'")
    p.add_argument("--stardist-model", default="2D_versatile_fluo")
    p.add_argument("--stardist-target-um", type=float, default=0.5,
                   help="pixel size StarDist is run at; images at other sizes are rescaled")
    p.add_argument("--prob-thresh", type=float, default=None, help="StarDist probability threshold (model default)")
    p.add_argument("--nms-thresh", type=float, default=None, help="StarDist overlap threshold (model default)")
    p.add_argument("--min-nucleus-um2", type=float, default=10.0, help="drop nuclei smaller than this")
    p.add_argument("--expand-um", type=float, default=3.0, help="expansion margin (µm)")
    p.add_argument("--max-grow-um", type=float, default=6.0, help="propagation: maximum growth from the nucleus (µm)")
    p.add_argument("--distance-weight", type=float, default=0.5,
                   help="propagation: how strongly distance from the nucleus resists growth (0-1 scale of membrane)")
    p.add_argument("--min-membrane", type=float, default=0.1,
                   help="propagation: beyond --expand-um, grow only onto pixels whose membrane composite is at "
                        "least this (0-1 scale of the membrane channels' display range)")
    p.add_argument("--roi-um", type=float, default=60.0, help="side of each gallery square (µm)")
    p.add_argument("--n-rois", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ground-truth", default="",
                   help="QuPath GeoJSON of hand-drawn cells, plus one annotation per labelled field classified "
                        "'Ground truth'. Scores every method by F1 at IoU thresholds inside those fields.")
    p.add_argument("--reference", default="", help="CSV of reference cell centres (x_px, y_px[, region, area_px])")
    p.add_argument("--reference-align", choices=["none", "translate"], default="none")
    p.add_argument("--reference-note", default="", help="one sentence on where the reference came from")
    p.add_argument("--match-um", type=float, default=5.0, help="max distance to count a reference cell as found")
    p.add_argument("--tight-um", type=float, default=2.0,
                   help="tight match distance, where chance matches are rare (used for the headline and areas)")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted CellSAM segmentation (e.g. one stopped by a wall-time limit) "
                        "instead of starting it over; uses cells_cellsam.progress.json when present, otherwise "
                        "reconstructs the resume point from the label file. StarDist nuclei are recomputed.")
    p.add_argument("--threads", type=int, default=0,
                   help="CPU threads for CellSAM; 0 = the SLURM allocation, or the library default off-cluster")
    args = p.parse_args()
    for _, row in runs.select_samples(args).iterrows():
        segment_sample(row, args)


if __name__ == "__main__":
    main()
