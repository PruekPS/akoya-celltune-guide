"""Read hand-drawn cells from a QuPath GeoJSON export and score a segmentation against them.

This is the reference side of the stage-02 bake-off. Nothing here compares two
automatic methods to each other -- `02_segment.compare_two_methods` does that,
and neither of its inputs is ground truth. Here one side *is* ground truth:
outlines a person drew.

Two things make that different from a method-versus-method comparison.

**Labelling is exhaustive only inside the fields that were drawn.** A nucleus
the pipeline found outside those fields is not a false positive, it is simply
outside the labelled area. So every score is computed inside an explicit
region, and cells touching the region border are dropped from both sides --
they are cut off by the border, not by the segmenter.

**Agreement is measured by outline overlap, not by centre distance.** Centroid
matching needs a chance baseline because in dense tissue any point has a
neighbour; IoU at a stated threshold does not, because overlapping by more than
half the combined area is not something two unrelated cells do. Reporting F1
across a sweep of IoU thresholds is what the segmentation benchmarks use
(Sankaranarayanan et al., Commun Biol 2025, doi:10.1038/s42003-025-08184-8;
Schmidt et al., MICCAI 2018), so our numbers can be read next to theirs.

Expected export from QuPath:

    File -> Export objects as GeoJSON
      * All objects
      * Export as FeatureCollection
      * Exclude measurements  (keeps the file small; nothing here reads them)

Coordinates come out in full-resolution image pixels, which is what the label
images from stage 02 are in, so no rescaling is applied. If you annotate on a
downsampled image, say so -- that would need a scale factor and this module
deliberately does not guess one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon as sk_polygon

# QuPath writes annotations (hand-drawn areas) and detections (cells) into one
# FeatureCollection. These are the classification names this module accepts as
# "the area that was labelled exhaustively"; anything else is treated as a cell.
REGION_CLASSES = {"ground truth", "groundtruth", "gt", "gt region", "ground truth region",
                  "labelled region", "labeled region", "region"}

# QuPath leaves an annotation unclassified unless one is set, and then its *name* is
# the only label it carries. The first real export (PS88, 2026-09-25) named the field
# "Ground truth" with no classification, so the region was read as a cell and the file
# scored as "1 outline in 0 fields" without raising anything. Names count too.
REGION_NAMES = REGION_CLASSES


@dataclass
class Shape:
    """One polygon with optional holes, in image pixel coordinates."""
    rings: list                      # [exterior, hole, hole, ...] each (N, 2) as (x, y)
    classification: str = ""
    object_type: str = ""
    name: str = ""

    @property
    def bbox(self):
        x, y = self.rings[0][:, 0], self.rings[0][:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    @property
    def centroid(self):
        r = self.rings[0]
        return float(r[:, 0].mean()), float(r[:, 1].mean())


@dataclass
class GroundTruth:
    cells: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    points: list = field(default_factory=list)     # (x, y) arrays: nuclei marked by clicking
    path: str = ""

    def __len__(self):
        return len(self.cells)

    @property
    def marked_points(self):
        """Clicked nuclei centres as an (N, 2) array of (x, y), empty if none."""
        return np.asarray(self.points, float).reshape(-1, 2) if self.points else np.zeros((0, 2))


def _rings(geom):
    """Polygon rings from a GeoJSON geometry, as a list of lists of (N, 2) arrays."""
    t = geom.get("type")
    if t == "Polygon":
        return [[np.asarray(r, float) for r in geom["coordinates"]]]
    if t == "MultiPolygon":
        return [[np.asarray(r, float) for r in poly] for poly in geom["coordinates"]]
    if t == "GeometryCollection":
        out = []
        for g in geom.get("geometries", []):
            out.extend(_rings(g))
        return out
    return []   # Point / LineString: not an outline, ignore


def _points(geom):
    """Clicked positions from a GeoJSON geometry, as a list of (x, y) arrays.

    QuPath's Points tool writes every click of one annotation into a single
    MultiPoint, so one feature can carry hundreds of nuclei. Marking centres is
    far quicker than drawing outlines, and it answers the question the stage-02
    bake-off actually turns on -- how many cells there are and where -- though it
    cannot say anything about outline shape.
    """
    t = geom.get("type")
    if t == "Point":
        return [np.asarray(geom["coordinates"], float)[:2]]
    if t == "MultiPoint":
        return [np.asarray(c, float)[:2] for c in geom["coordinates"]]
    if t == "GeometryCollection":
        out = []
        for g in geom.get("geometries", []):
            out.extend(_points(g))
        return out
    return []


def _classification(props):
    c = props.get("classification")
    if isinstance(c, dict):
        return str(c.get("name", "") or "")
    if isinstance(c, str):
        return c
    return ""


def read_qupath_geojson(path, region_classes=REGION_CLASSES):
    """Read hand-drawn cells and the regions they were drawn in.

    A feature becomes a *region* if its classification is one of
    `region_classes`, or if it is a QuPath annotation while detections are also
    present in the file. Everything else is a cell.
    """
    with open(path) as fh:
        doc = json.load(fh)

    feats = doc.get("features") if isinstance(doc, dict) and doc.get("type") == "FeatureCollection" else None
    if feats is None:
        feats = doc if isinstance(doc, list) else [doc]
        if not isinstance(doc, list) and doc.get("type") == "Feature":
            feats = [doc]

    parsed, clicked = [], []
    for f in feats:
        geom, props = f.get("geometry") or {}, f.get("properties") or {}
        obj = str(props.get("objectType", "") or "")
        cls = _classification(props)
        nm = str(props.get("name", "") or "")
        for rings in _rings(geom):
            parsed.append(Shape(rings=rings, classification=cls, object_type=obj, name=nm))
        clicked.extend(_points(geom))

    has_detections = any(s.object_type == "detection" for s in parsed)
    cells, regions = [], []
    for s in parsed:
        by_class = s.classification.strip().lower() in region_classes
        by_name = s.name.strip().lower() in REGION_NAMES
        by_type = has_detections and s.object_type == "annotation"
        # Nobody draws cell outlines in the same file as they click nuclei, so in a
        # points export every polygon is a labelled field whatever it is called. The
        # PS82 export (2026-09-26) came back named "Grouth truth"; matching the name
        # alone would have silently scored it as "473 marks in 0 fields".
        by_points = bool(clicked) and s.object_type != "detection"
        (regions if (by_class or by_name or by_type or by_points) else cells).append(s)

    gt = GroundTruth(cells=cells, regions=regions, points=clicked, path=str(path))
    if not gt.cells and not clicked:
        raise ValueError(f"{path}: no cell outlines and no marked points found. Export with "
                         f"'All objects' and 'Export as FeatureCollection' ticked.")
    return gt


def rasterize(shapes, y0, y1, x0, x1, shrink=0.0):
    """Label image of `shapes` inside the window, one integer id per shape (0 = background).

    Later shapes overwrite earlier ones where they overlap, which is what QuPath
    itself does; hand-drawn cells rarely overlap. `shrink` is unused for now and
    kept out of the signature's meaning deliberately -- eroding ground truth to
    make a score look better is exactly the kind of knob that should not exist.
    """
    if shrink:
        raise NotImplementedError("shrinking ground truth is not supported on purpose")
    h, w = int(y1 - y0), int(x1 - x0)
    out = np.zeros((h, w), np.int32)
    for i, s in enumerate(shapes, start=1):
        ext = s.rings[0]
        rr, cc = sk_polygon(ext[:, 1] - y0, ext[:, 0] - x0, shape=(h, w))
        if rr.size == 0:
            continue
        out[rr, cc] = i
        for hole in s.rings[1:]:
            hr, hc = sk_polygon(hole[:, 1] - y0, hole[:, 0] - x0, shape=(h, w))
            if hr.size:
                out[hr, hc] = 0
    return out


def region_window(region, pad=0):
    """Integer (y0, y1, x0, x1) window covering one region shape."""
    xmin, ymin, xmax, ymax = region.bbox
    return (int(np.floor(ymin)) - pad, int(np.ceil(ymax)) + pad + 1,
            int(np.floor(xmin)) - pad, int(np.ceil(xmax)) + pad + 1)


def drop_border_touching(labels):
    """Remove objects touching the window edge, and report how many went.

    A cell clipped by the border has a truncated area on one side and a full
    area on the other, so its IoU is wrong through no fault of either method.
    Both ground truth and prediction get the same treatment.
    """
    if labels.size == 0:
        return labels, 0
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    touching = set(int(v) for v in np.unique(edge) if v > 0)
    if not touching:
        return labels, 0
    out = labels.copy()
    out[np.isin(out, list(touching))] = 0
    return out, len(touching)


def iou_matrix(gt, pred):
    """Pairwise IoU between the objects of two label images of the same shape.

    Returns (matrix, gt_ids, pred_ids) with matrix[i, j] the IoU of gt_ids[i]
    against pred_ids[j]. Computed from a joint histogram so it stays linear in
    pixels rather than quadratic in objects.
    """
    gt_ids = np.array([i for i in np.unique(gt) if i > 0])
    pred_ids = np.array([i for i in np.unique(pred) if i > 0])
    if gt_ids.size == 0 or pred_ids.size == 0:
        return np.zeros((gt_ids.size, pred_ids.size)), gt_ids, pred_ids

    gi = np.searchsorted(gt_ids, gt.ravel())
    pi = np.searchsorted(pred_ids, pred.ravel())
    valid = (gt.ravel() > 0) & (pred.ravel() > 0)
    inter = np.zeros((gt_ids.size, pred_ids.size), np.int64)
    if valid.any():
        np.add.at(inter, (gi[valid], pi[valid]), 1)

    gt_area = np.array([(gt == i).sum() for i in gt_ids], np.int64)
    pred_area = np.array([(pred == j).sum() for j in pred_ids], np.int64)
    union = gt_area[:, None] + pred_area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(union > 0, inter / union, 0.0)
    return m, gt_ids, pred_ids


def score(gt_labels, pred_labels, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9), drop_border=True):
    """F1 of a segmentation against hand-drawn ground truth, per IoU threshold.

    One-to-one matching that maximizes total IoU (Hungarian), then a pair counts
    as a true positive only if its IoU clears the threshold. At 0.5 this is the
    standard instance-segmentation F1 reported by the segmentation benchmarks.

    Also returns the errors a single F1 hides: `split` is how many predicted
    objects overlap one ground-truth cell by at least 10% (over-segmentation),
    `merged` the reverse (under-segmentation).
    """
    if drop_border:
        gt_labels, n_gt_border = drop_border_touching(gt_labels)
        pred_labels, n_pred_border = drop_border_touching(pred_labels)
    else:
        n_gt_border = n_pred_border = 0

    m, gt_ids, pred_ids = iou_matrix(gt_labels, pred_labels)
    n_gt, n_pred = gt_ids.size, pred_ids.size

    matched_iou = np.array([])
    if n_gt and n_pred:
        r, c = linear_sum_assignment(-m)
        matched_iou = m[r, c]
        matched_iou = matched_iou[matched_iou > 0]

    per_t = {}
    for t in thresholds:
        tp = int((matched_iou >= t).sum())
        fp, fn = n_pred - tp, n_gt - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_t[t] = {"threshold": t, "TP": tp, "FP": fp, "FN": fn,
                    "precision": prec, "recall": rec,
                    "F1": 2 * prec * rec / max(prec + rec, 1e-12)}

    split = merged = 0
    if n_gt and n_pred:
        overlap = m >= 0.10
        split = int((overlap.sum(axis=1) > 1).sum())     # one GT cell hit by several predictions
        merged = int((overlap.sum(axis=0) > 1).sum())    # one prediction covering several GT cells

    return {
        "n_ground_truth": int(n_gt), "n_predicted": int(n_pred),
        "count_ratio": (n_pred / n_gt) if n_gt else float("nan"),
        "mean_matched_IoU": float(matched_iou.mean()) if matched_iou.size else float("nan"),
        "median_matched_IoU": float(np.median(matched_iou)) if matched_iou.size else float("nan"),
        "split_ground_truth_cells": split, "merged_predictions": merged,
        "dropped_touching_border_gt": int(n_gt_border),
        "dropped_touching_border_pred": int(n_pred_border),
        "per_threshold": per_t,
        "F1@0.5": per_t.get(0.5, {}).get("F1", float("nan")),
    }


def _min_distance_to_ring(pts, ring):
    """Shortest distance from each point to a closed polygon boundary."""
    if pts.size == 0:
        return np.zeros(0)
    a, b = ring[:-1], ring[1:]
    seg = b - a
    length2 = (seg ** 2).sum(axis=1)
    length2[length2 == 0] = 1e-12
    t = (((pts[:, None, :] - a[None]) * seg[None]).sum(axis=2) / length2[None]).clip(0, 1)
    proj = a[None] + t[..., None] * seg[None]
    return np.sqrt(((pts[:, None, :] - proj) ** 2).sum(axis=2)).min(axis=1)


def match_points(gt_xy, pred_xy, radius):
    """One-to-one nearest matching within `radius`; returns (gt index, pred index, distance).

    Hungarian assignment on distance, then pairs beyond the radius are discarded.
    Greedy nearest-neighbour would double-count a prediction that happens to be
    closest to two marks, which inflates recall exactly where cells are crowded
    and the answer matters most.
    """
    if gt_xy.size == 0 or pred_xy.size == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    d = np.sqrt(((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2))
    cost = np.where(d <= radius, d, radius * 1e3)
    gi, pi = linear_sum_assignment(cost)
    keep = d[gi, pi] <= radius
    return gi[keep], pi[keep], d[gi[keep], pi[keep]]


def score_points(gt_xy, pred_xy, radius, shift_range=(40.0, 120.0), n_shifts=8, seed=0):
    """Detection accuracy of a segmentation against clicked nuclei centres.

    A prediction counts as finding a marked nucleus when it is the one-to-one
    nearest within `radius` pixels. There is no IoU here: a click has no area, so
    this measures whether the right number of cells were found in the right
    places, not whether their outlines are the right shape.

    Centre matching needs a chance baseline, because in crowded tissue almost any
    point has some prediction near it. The baseline shifts every prediction by a
    random offset between `shift_range` pixels and rematches; a real result has to
    beat it by a wide margin to mean anything. This mirrors what stage 02 does
    when comparing two automatic methods.
    """
    gt_xy, pred_xy = np.asarray(gt_xy, float), np.asarray(pred_xy, float)
    gi, pi, dist = match_points(gt_xy, pred_xy, radius)
    n_gt, n_pred, tp = len(gt_xy), len(pred_xy), len(gi)
    prec = tp / max(n_pred, 1)
    rec = tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    rng = np.random.default_rng(seed)
    chance = []
    for _ in range(n_shifts if (n_gt and n_pred) else 0):
        ang = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(*shift_range)
        moved = pred_xy + np.array([np.cos(ang), np.sin(ang)]) * mag
        c_gi, _, _ = match_points(gt_xy, moved, radius)
        c_tp = len(c_gi)
        c_prec, c_rec = c_tp / max(n_pred, 1), c_tp / max(n_gt, 1)
        chance.append(2 * c_prec * c_rec / max(c_prec + c_rec, 1e-12))
    chance_f1 = float(np.mean(chance)) if chance else float("nan")

    return {"n_ground_truth": int(n_gt), "n_predicted": int(n_pred),
            "count_ratio": (n_pred / n_gt) if n_gt else float("nan"),
            "TP": int(tp), "FP": int(n_pred - tp), "FN": int(n_gt - tp),
            "precision": float(prec), "recall": float(rec), "F1": float(f1),
            "chance_F1": chance_f1, "margin_over_chance": float(f1 - chance_f1) if chance else float("nan"),
            "median_match_distance_px": float(np.median(dist)) if dist.size else float("nan"),
            "match_radius_px": float(radius)}


def score_points_in_regions(gt, pred_xy, radius, edge_margin=None, shift_range=(40.0, 120.0),
                            n_shifts=8, seed=0):
    """Score clicked nuclei region by region, then pooled.

    Only marks inside a labelled region are scored, and marks within
    `edge_margin` of that region's border are dropped along with predictions in
    the same strip: a nucleus at the very edge may have its true match just
    outside the field, which would be counted as a miss through no fault of the
    segmenter. `edge_margin` defaults to the match radius.
    """
    from matplotlib.path import Path as MplPath

    if not gt.regions:
        raise ValueError(
            f"{gt.path}: no labelled region found. Draw a rectangle around each field that was "
            f"marked exhaustively and either classify it 'Ground truth' or name it 'Ground truth', "
            f"so scoring knows where the marking is complete.")
    marks = gt.marked_points
    if marks.size == 0:
        raise ValueError(f"{gt.path}: no marked points found.")
    pred_xy = np.asarray(pred_xy, float).reshape(-1, 2)
    margin = radius if edge_margin is None else edge_margin

    per_region, pooled_gt, pooled_pred = [], [], []
    for i, region in enumerate(gt.regions, start=1):
        ring = region.rings[0]
        path = MplPath(ring)
        in_gt = path.contains_points(marks) & (_min_distance_to_ring(marks, ring) > margin)
        in_pred = path.contains_points(pred_xy) & (_min_distance_to_ring(pred_xy, ring) > margin)
        g, p = marks[in_gt], pred_xy[in_pred]
        if g.size == 0:
            continue
        res = score_points(g, p, radius, shift_range, n_shifts, seed)
        res["region"] = region.name or f"region {i}"
        per_region.append(res)
        pooled_gt.append(g)
        pooled_pred.append(p)

    if not per_region:
        raise ValueError(f"{gt.path}: no marked points fell inside a labelled region.")
    pooled = score_points(np.vstack(pooled_gt), np.vstack(pooled_pred) if pooled_pred else np.zeros((0, 2)),
                          radius, shift_range, n_shifts, seed)
    return {"pooled": pooled, "per_region": per_region, "mode": "points"}


def score_in_regions(gt, pred_label_image, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Score every labelled region of a GroundTruth against a full-slide label image.

    `pred_label_image` may be a numpy array or anything sliceable the same way
    (a memmap from stage 02, for instance), indexed [y, x] in full-resolution
    pixels. Regions are scored separately and then pooled, because a mean of
    per-region F1 hides that one crowded field is where the methods differ.
    """
    if not gt.regions:
        raise ValueError(
            f"{gt.path}: no labelled region found. Draw a rectangle around each field you "
            f"labelled and classify it 'Ground truth', so scoring knows where labelling was "
            f"exhaustive. Without it, every cell found outside your labelled fields counts "
            f"as a false positive.")

    per_region, pooled_gt, pooled_pred = [], [], []
    offset_gt = offset_pred = 0
    for k, region in enumerate(gt.regions):
        y0, y1, x0, x1 = region_window(region)
        H, W = pred_label_image.shape[:2]
        y0, y1 = max(0, y0), min(H, y1)
        x0, x1 = max(0, x0), min(W, x1)
        if y1 <= y0 or x1 <= x0:
            continue

        inside = [c for c in gt.cells
                  if x0 <= c.centroid[0] < x1 and y0 <= c.centroid[1] < y1]
        g = rasterize(inside, y0, y1, x0, x1)
        p = np.asarray(pred_label_image[y0:y1, x0:x1]).astype(np.int32, copy=True)

        # Keep only the region's own interior -- a hand-drawn field may be any shape --
        # and select both sides the same way. Ground-truth cells were chosen by centroid
        # above, so predictions are too: cropping predictions by overlap instead would
        # count a cell that merely leans into the field, giving it more objects than the
        # ground truth has and depressing precision for a reason that is not segmentation.
        mask = rasterize([region], y0, y1, x0, x1) > 0
        g[~mask] = 0
        p[~mask] = 0
        keep = []
        for pid in np.unique(p):
            if pid == 0:
                continue
            ys, xs = np.nonzero(p == pid)
            if mask[int(round(ys.mean())), int(round(xs.mean()))]:
                keep.append(pid)
        p[~np.isin(p, keep)] = 0

        s = score(g, p, thresholds=thresholds)
        s["region"] = region.name or f"region {k + 1}"
        s["window"] = (y0, y1, x0, x1)
        s["area_px"] = int(mask.sum())
        per_region.append(s)

        gg, pp = g.copy(), p.copy()
        gg[gg > 0] += offset_gt
        pp[pp > 0] += offset_pred
        offset_gt, offset_pred = int(gg.max(initial=0)), int(pp.max(initial=0))
        pooled_gt.append(gg.ravel())
        pooled_pred.append(pp.ravel())

    if not per_region:
        raise ValueError(f"{gt.path}: every labelled region fell outside the image.")

    pooled = score(np.concatenate(pooled_gt)[None, :], np.concatenate(pooled_pred)[None, :],
                   thresholds=thresholds, drop_border=False)
    pooled["region"] = "all regions pooled"
    return {"per_region": per_region, "pooled": pooled}


def score_two_labellings(a, b, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Agreement between two hand labellings of the same fields -- the noise floor.

    The bake-off gate is that a method's margin over its rivals must exceed the
    spread between two passes of the same human over the same field. Without
    this number a bake-off cannot tell "method A is better" from "the labels
    wobble by that much", and the whole comparison rests on an assumption.

    `a` supplies the regions (use the first pass). Cells from both labellings are
    rasterized in the same windows and scored exactly as a method would be, so
    the resulting F1 is directly comparable with the per-method scores.
    """
    if not a.regions:
        raise ValueError(f"{a.path}: the first labelling must carry the 'Ground truth' regions.")

    per_region, pooled_a, pooled_b = [], [], []
    off_a = off_b = 0
    for k, region in enumerate(a.regions):
        y0, y1, x0, x1 = region_window(region)
        mask = rasterize([region], y0, y1, x0, x1) > 0

        def inside(cells):
            return [c for c in cells if x0 <= c.centroid[0] < x1 and y0 <= c.centroid[1] < y1]

        ga = rasterize(inside(a.cells), y0, y1, x0, x1)
        gb = rasterize(inside(b.cells), y0, y1, x0, x1)
        ga[~mask] = 0
        gb[~mask] = 0

        s = score(ga, gb, thresholds=thresholds)
        s["region"] = region.name or f"region {k + 1}"
        per_region.append(s)

        aa, bb = ga.copy(), gb.copy()
        aa[aa > 0] += off_a
        bb[bb > 0] += off_b
        off_a, off_b = int(aa.max(initial=0)), int(bb.max(initial=0))
        pooled_a.append(aa.ravel())
        pooled_b.append(bb.ravel())

    if not per_region:
        raise ValueError("no overlapping labelled region between the two labellings")

    pooled = score(np.concatenate(pooled_a)[None, :], np.concatenate(pooled_b)[None, :],
                   thresholds=thresholds, drop_border=False)
    pooled["region"] = "all regions pooled"
    return {"per_region": per_region, "pooled": pooled,
            "noise_floor_F1@0.5": pooled["per_threshold"][0.5]["F1"]}
