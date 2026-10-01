"""Distances between cells and interactions between cell types.

Four things decide whether a spatial result means anything, and all four are
easy to get wrong in ways that still produce a plausible number. This module
takes a position on each.

**The observation window is the tissue, not its bounding box.** A convex hull
around lung tissue swallows the airways and the slide background, so every
population looks clustered because the window says cells could have been in
places no cell can be. Sridhar et al. measured this: a misspecified window
inflated Ripley's K area-under-curve by up to 649.6% and produced 100% false
rejection of complete spatial randomness (doi:10.34133/csbj.0141). Stage 01
already emits the tissue-occupancy mask; everything here uses it.

**The null shuffles labels, never positions.** Asking "is A near B more than
expected" must hold the tissue geometry fixed and permute only what each cell
*is*. Shuffling positions instead asks a question about tissue architecture
that nobody wanted answered.

**Permutation is stratified by tissue piece.** PS88 is eight separate pieces of
lung. A label permuted from piece 1 to piece 7 creates a "neighbour" pair that
no biology could produce, which shrinks the null's spread and inflates every z.

**Distances near the section edge are right-censored.** If a cell's nearest
T cell is 80 um away but the tissue ends 30 um away, the true nearest T cell may
lie in the part of the tumour that went to another slide. Those measurements are
counted and reported separately rather than silently averaged in.

Abundance is handled by construction: every enrichment is observed over expected
from the same graph with shuffled labels, so a population being common cannot by
itself look like an interaction.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi


# -- the window ---------------------------------------------------------------------------------

class Window:
    """The tissue, as stage 01 measured it: where a cell could have been.

    `mask` is boolean at `downsample`-reduced resolution; `pieces` labels its
    connected components, which are the strata for permutation.
    """

    def __init__(self, mask, pieces, downsample, px_um):
        self.mask = np.asarray(mask, bool)
        self.pieces = np.asarray(pieces, np.int32)
        self.downsample = float(downsample)
        self.px_um = float(px_um)
        self.work_um = self.px_um * self.downsample
        # distance (in um) from each tissue pixel to the nearest non-tissue pixel
        self._edge_dist = ndi.distance_transform_edt(self.mask) * self.work_um

    @classmethod
    def from_stage01(cls, npz_path, px_um):
        z = np.load(npz_path)
        mask = z["mask"]
        pieces = z["regions"] if "regions" in z else ndi.label(mask)[0]
        return cls(mask, pieces, float(z["downsample"]), px_um)

    @property
    def area_um2(self):
        return float(self.mask.sum()) * self.work_um ** 2

    def _ij(self, xy_px):
        """Full-resolution pixel coordinates -> indices into the reduced mask."""
        j = np.clip((xy_px[:, 0] / self.downsample).astype(int), 0, self.mask.shape[1] - 1)
        i = np.clip((xy_px[:, 1] / self.downsample).astype(int), 0, self.mask.shape[0] - 1)
        return i, j

    def inside(self, xy_px):
        i, j = self._ij(xy_px)
        return self.mask[i, j]

    def piece_of(self, xy_px):
        i, j = self._ij(xy_px)
        return self.pieces[i, j]

    def distance_to_edge_um(self, xy_px):
        i, j = self._ij(xy_px)
        return self._edge_dist[i, j]

    def sample_uniform(self, n, rng):
        """n points drawn uniformly from the tissue, in full-resolution pixels.

        Used to get the expectation under complete spatial randomness *in this
        window*, which sidesteps edge-correction formulae that assume a
        rectangle. An analytic CSR expectation on a shape like lung would be
        wrong in the direction that makes everything look clustered.
        """
        ii, jj = np.nonzero(self.mask)
        k = rng.integers(0, ii.size, size=n)
        # jitter inside the reduced pixel so points are not on a lattice
        y = (ii[k] + rng.random(n)) * self.downsample
        x = (jj[k] + rng.random(n)) * self.downsample
        return np.column_stack([x, y])


# -- distances between cell types ---------------------------------------------------------------

def nearest_of_type(xy_from, xy_to, exclude_self=False):
    """Distance from each row of `xy_from` to the nearest point in `xy_to`.

    `exclude_self` for a population against itself, where the nearest point is
    the cell itself at distance zero.
    """
    if len(xy_to) == 0 or len(xy_from) == 0:
        return np.full(len(xy_from), np.nan)
    k = 2 if exclude_self else 1
    tree = cKDTree(xy_to)
    d, _ = tree.query(xy_from, k=k)
    return (d[:, 1] if exclude_self else (d[:, 0] if d.ndim > 1 else d)).astype(float)


def contact_fractions(d_um, radii=(10, 20, 30, 50, 100)):
    """Share of cells with a partner within each radius. NaNs are excluded."""
    ok = np.isfinite(d_um)
    n = int(ok.sum())
    return {f"within_{r:g}um": (float(np.mean(d_um[ok] <= r)) if n else np.nan) for r in radii}


def censoring(d_um, edge_um):
    """Flag distances that could be shortened by tissue the section does not contain.

    A distance longer than the cell's own distance to the tissue edge is
    right-censored: the true nearest partner may lie outside the section.
    """
    return np.isfinite(d_um) & (d_um > edge_um)


def pair_distances(xy, labels, window, pairs=None, radii=(10, 20, 30, 50, 100),
                   n_perm=200, seed=0, px_um=1.0):
    """Nearest-neighbour distance from every type to every other, against a null.

    Returns one row per ordered pair (from -> to) with the observed median
    distance, contact fractions, the share of measurements that are censored,
    and a permutation z and p computed by shuffling labels within tissue pieces.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    types = [t for t in sorted(set(labels.tolist())) if t is not None]
    piece = window.piece_of(xy)
    edge = window.distance_to_edge_um(xy)
    xy_um = xy * px_um
    pairs = pairs or [(a, b) for a in types for b in types]

    rows = []
    for a, b in pairs:
        ia, ib = labels == a, labels == b
        if ia.sum() == 0 or ib.sum() == 0:
            continue
        d = nearest_of_type(xy_um[ia], xy_um[ib], exclude_self=(a == b))
        cens = censoring(d, edge[ia])
        obs = float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan

        null = np.empty(n_perm)
        for p in range(n_perm):
            perm = _permute_within(labels, piece, rng)
            ja, jb = perm == a, perm == b
            dn = nearest_of_type(xy_um[ja], xy_um[jb], exclude_self=(a == b))
            null[p] = np.nanmedian(dn) if np.isfinite(dn).any() else np.nan
        mu, sd = float(np.nanmean(null)), float(np.nanstd(null))
        z = (obs - mu) / sd if sd > 0 else np.nan
        # two-sided empirical p with the +1 correction, so p is never 0
        more = int(np.sum(np.abs(null - mu) >= abs(obs - mu)))
        p_emp = (more + 1) / (n_perm + 1)

        row = {"from": a, "to": b, "n_from": int(ia.sum()), "n_to": int(ib.sum()),
               "median_um": obs, "null_median_um": mu,
               "closer_than_chance": bool(obs < mu), "z": z, "p": p_emp,
               "pct_censored": 100 * float(cens.mean()) if cens.size else np.nan}
        row.update(contact_fractions(d, radii))
        rows.append(row)
    return rows


def _permute_within(labels, strata, rng):
    """Shuffle labels inside each stratum, never across them."""
    out = np.array(labels, copy=True)
    for s in np.unique(strata):
        m = strata == s
        idx = np.nonzero(m)[0]
        out[idx] = labels[rng.permutation(idx)]
    return out


# -- interactions between types -----------------------------------------------------------------

def neighbour_pairs(xy_um, radius_um):
    """Index pairs (i, j), i < j, of cells within `radius_um` of each other."""
    tree = cKDTree(xy_um)
    pairs = tree.query_pairs(radius_um, output_type="ndarray")
    return pairs


def enrichment(xy, labels, window, radius_um=70.0, n_perm=1000, seed=0, px_um=1.0):
    """Which type pairs touch more (or less) than chance, on a fixed graph.

    The graph is built once from the real positions and never rebuilt: only the
    labels move. That is what makes the answer about arrangement rather than
    about density -- a population being twice as common doubles its edges under
    the null as well, so the ratio is unmoved.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    types = sorted(set(labels.tolist()))
    idx = {t: i for i, t in enumerate(types)}
    T = len(types)
    piece = window.piece_of(xy)
    pairs = neighbour_pairs(xy * px_um, radius_um)
    if pairs.size == 0:
        return {"types": types, "observed": np.zeros((T, T)), "log2_oe": np.full((T, T), np.nan),
                "z": np.full((T, T), np.nan), "p": np.full((T, T), np.nan),
                "n_edges": 0, "radius_um": radius_um, "n_perm": n_perm}

    def counts(lab):
        c = np.zeros((T, T), np.int64)
        a = np.array([idx[v] for v in lab[pairs[:, 0]]])
        b = np.array([idx[v] for v in lab[pairs[:, 1]]])
        np.add.at(c, (a, b), 1)
        np.add.at(c, (b, a), 1)          # undirected: count each edge both ways
        return c

    obs = counts(labels)
    null = np.empty((n_perm, T, T))
    for p in range(n_perm):
        null[p] = counts(_permute_within(labels, piece, rng))

    mu, sd = null.mean(0), null.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_oe = np.log2(np.where(obs > 0, obs, np.nan) / np.where(mu > 0, mu, np.nan))
        z = np.where(sd > 0, (obs - mu) / sd, np.nan)
    more = (np.abs(null - mu) >= np.abs(obs - mu)).sum(0)
    p_emp = (more + 1) / (n_perm + 1)

    return {"types": types, "observed": obs, "expected": mu, "log2_oe": log2_oe,
            "z": z, "p": p_emp, "n_edges": int(len(pairs)),
            "radius_um": radius_um, "n_perm": n_perm}


def benjamini_hochberg(p):
    """FDR-adjusted q values. With 16 types there are 256 ordered pairs; at
    p < 0.05 uncorrected, 13 of them are expected to be 'significant' by luck."""
    p = np.asarray(p, float)
    flat = p.ravel()
    ok = np.isfinite(flat)
    q = np.full(flat.shape, np.nan)
    if ok.sum() == 0:
        return q.reshape(p.shape)
    vals = flat[ok]
    order = np.argsort(vals)
    n = vals.size
    ranked = vals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q.reshape(p.shape)


# -- neighbourhood composition and niches ---------------------------------------------------------

def neighbourhood_composition(xy, labels, radius_um=25.0, px_um=1.0):
    """For each cell, the make-up of its neighbourhood as fractions per type.

    The cell itself is excluded, so a rare type surrounded by a common one does
    not get credit for its own presence.
    """
    labels = np.asarray(labels)
    types = sorted(set(labels.tolist()))
    idx = {t: i for i, t in enumerate(types)}
    code = np.array([idx[v] for v in labels])
    xy_um = xy * px_um
    tree = cKDTree(xy_um)
    comp = np.zeros((len(xy), len(types)))
    for i, nb in enumerate(tree.query_ball_point(xy_um, radius_um)):
        nb = [j for j in nb if j != i]
        if nb:
            np.add.at(comp[i], code[nb], 1)
    total = comp.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, comp / total, 0.0)
    return frac, types, total.ravel().astype(int)


def niches(composition, k=7, seed=0):
    """Cluster neighbourhood compositions into recurring niches."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(composition)
    return km.labels_, km.cluster_centers_


# -- is a population clustered at all? ------------------------------------------------------------

def clustering_index(xy, window, n_sim=20, seed=0, px_um=1.0):
    """Observed mean nearest-neighbour distance over its expectation under CSR.

    Below 1 means clustered, near 1 means indistinguishable from random, above 1
    means dispersed. The expectation is simulated by scattering the same number
    of points uniformly *inside the tissue mask*, so the shape of the tissue is
    accounted for rather than assumed rectangular.

    This is the check that matters before any tumour-referenced claim: a
    "tumour" compartment that is not spatially clustered is not a tumour, and
    every distance measured from it is meaningless.
    """
    n = len(xy)
    if n < 3:
        return {"n": n, "observed_um": np.nan, "expected_um": np.nan, "index": np.nan}
    obs = float(np.mean(nearest_of_type(xy * px_um, xy * px_um, exclude_self=True)))
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_sim):
        pts = window.sample_uniform(n, rng) * px_um
        sims.append(np.mean(nearest_of_type(pts, pts, exclude_self=True)))
    exp = float(np.mean(sims))
    return {"n": n, "observed_um": obs, "expected_um": exp,
            "index": obs / exp if exp > 0 else np.nan,
            "expected_sd_um": float(np.std(sims))}
