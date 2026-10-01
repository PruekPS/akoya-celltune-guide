#!/usr/bin/env python
"""Compare a feature table between groups, counting samples and not ROIs.

The unit of replication is the animal or slide, never the ROI or the cell. Ten ROIs from
one slide are ten looks at one biological sample; testing them as ten samples shrinks the
p-values by a factor that grows with how different slides are from each other. This script
fits, for every feature,

    feature ~ group + (1 | sample)          a random intercept per sample

(the same model the published HIV lymph-node analysis fits with R's nlme::lme), applies
Benjamini-Hochberg across features, and reports how many samples each group had.

Rules it enforces so a table of p-values cannot overstate what the data support:
  * a feature needs at least --min-samples samples in every group, else its p is NaN
    ("descriptive only"): with 2 vs 2 animals there is no valid test, only an effect size
  * a unit is one row; if every sample has exactly one row it falls back to a rank test
    on samples (Mann-Whitney), because a random intercept is unidentifiable then
  * if the mixed model does not converge it falls back to a rank test on per-sample means,
    and says so in the `method` column

Needs statsmodels. Status: validated on synthetic data only (tests/test_feature_stats.py),
including a demonstration that testing ROIs as independent samples is anti-conservative and that
statsmodels' default Wald z p-value is too small at few samples (the t-based p used here is not).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from akoyalib.spatial import benjamini_hochberg        # noqa: E402

META_COLS = {"unit_id", "sample_id", "n_cells", "area_mm2"}


def fit_one(df, feature, group_col, sample_col, min_samples):
    d = df[[feature, group_col, sample_col]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = sorted(d[group_col].unique())
    res = {"feature": feature, "n_rows": len(d)}
    if len(groups) != 2:
        res.update(method="needs_two_groups", p=np.nan)
        return res
    g0, g1 = groups
    ns = d.groupby(group_col)[sample_col].nunique()
    res.update(group_ref=g0, group_test=g1, n_samples_ref=int(ns.get(g0, 0)), n_samples_test=int(ns.get(g1, 0)))
    sm = d.groupby([group_col, sample_col])[feature].mean().reset_index()
    a = sm.loc[sm[group_col] == g0, feature].to_numpy()
    b = sm.loc[sm[group_col] == g1, feature].to_numpy()
    res["mean_ref"], res["mean_test"] = (float(a.mean()) if len(a) else np.nan), (float(b.mean()) if len(b) else np.nan)
    res["estimate"] = res["mean_test"] - res["mean_ref"]
    if min(len(a), len(b)) < min_samples:
        res.update(method=f"descriptive_only(<{min_samples}_samples_per_group)", p=np.nan)
        return res
    if len(d) == d[sample_col].nunique():                 # one row per sample: nothing to nest
        res.update(method="mann_whitney_on_samples", p=float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))
        return res
    try:
        import statsmodels.formula.api as smf
        dd = d.rename(columns={feature: "y", group_col: "g", sample_col: "s"})
        dd["g"] = pd.Categorical(dd["g"], categories=groups)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = smf.mixedlm("y ~ g", dd, groups=dd["s"]).fit(reml=True)
        term = [t for t in m.params.index if t.startswith("g[T.")][0]
        est, se = float(m.params[term]), float(m.bse[term])
        # statsmodels reports a Wald z p-value, which is too small with few samples (measured: 11.7% false
        # positives at 4 v 4 samples with no real effect). A group effect is carried by samples, not ROIs, so
        # use t with (samples - 2) degrees of freedom, as nlme::lme does: 6% on the same simulation.
        df_between = int(d[sample_col].nunique()) - 2
        res.update(method="mixed_model", estimate=est, se=se, df=df_between,
                   p=float(2 * stats.t.sf(abs(est / se), df_between)))
    except Exception as e:                                 # singular fit, non-convergence, missing statsmodels
        res.update(method=f"mann_whitney_on_sample_means(mixed model failed: {type(e).__name__})",
                   p=float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))
    return res


def run(df, group_col, sample_col, min_samples, features=None):
    feats = features or [c for c in df.columns if c not in META_COLS | {group_col, sample_col}
                         and pd.api.types.is_numeric_dtype(df[c])]
    out = pd.DataFrame([fit_one(df, f, group_col, sample_col, min_samples) for f in feats])
    out["q_bh"] = benjamini_hochberg(out["p"].to_numpy(float))
    return out.sort_values("p")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="features.csv from build_sample_features.py")
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--sample-col", default="sample_id")
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.features)
    out = run(df, a.group_col, a.sample_col, a.min_samples)
    out.to_csv(a.out, index=False)
    m = out["method"].value_counts()
    print(f"{len(out)} features -> {a.out}")
    print(m.to_string())
    print(f"q < 0.05: {int((out['q_bh'] < 0.05).sum())}")


if __name__ == "__main__":
    main()
