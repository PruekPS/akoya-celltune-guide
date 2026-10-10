# 10. Spatial analysis and comparing groups

Once CellTune has predicted a type for every cell on the slide (page 9), this
page turns that into: where each type sits relative to the tissue, whether
types sit nearer each other than chance, one row of numbers per sample, and an
honest statistical comparison between groups. Four scripts, run in order.

This is the most optional page in this guide. If you only needed a
CellTune-ready segmentation, you are already done at page 6. Keep going if you
also want tissue-region context or group comparisons out of CellTune's
predictions.

---

## 1. H&E regions, on the slide's own coordinates

Your panel may have no marker specific to the thing you actually care about —
no tumor marker is a common case. An adjacent H&E section can say *where* the
tumor is, even though it cannot say what any one cell is (it is a different
section, a few microns away). This step turns H&E regions into a column on
every cell:

```bash
python scripts/celltune_regions.py \
    --labels results/<your-sample-id>/02_segment/celltune/<your-sample-id>_segmentation_labels.tif \
    --regions <your-sample-id>_regions_warped.geojson \
    --sample-id <your-sample-id> --px-um <your pixel size> \
    --priority Normal,Necrosis,Tumor --focus Tumor --boundary-um 50 \
    --out <your-sample-id>_regions.csv
```

Before this will work, you need:

- Regions **drawn in QuPath** on the H&E (tumor, normal, necrosis — name
  classes consistently across slides, since the script matches by name), and
  on the slide itself for anything to exclude (folds, bubbles).
- Those H&E polygons **registered onto the PhenoCycler slide** — a different
  section needs its own coordinate system mapped onto this one. This guide
  does not cover registration itself; **VALIS** (github.com/MathOnco/valis) is
  a documented tool for it. Exclusion polygons drawn directly on the slide
  need no registration.

`--priority` lists classes lowest-priority first — where two regions overlap,
the one named later wins. `--focus` names the class that gets a signed
distance-to-edge column in µm (**positive inside it, negative outside**, 0 on
the edge); `near_boundary` flags cells
within `--boundary-um` of that edge, where the region call is least certain.
A cell outside every drawn region is `Unassigned`.

**Why a centroid, not "what fraction of the cell is inside the polygon":**
registration between two sections is off by tens of microns in places, and a
cell is roughly 10 µm across — a fraction computed to the pixel would claim
precision the registration does not have.

---

## 2. Join CellTune's types to those regions

```bash
python scripts/celltune_spatial.py \
    --regions <your-sample-id>_regions.csv \
    --populations <CellTune export>.csv \
    --roi-map CellTune_Data/<PROJECT>/Images/roi_map.csv \
    --px-um <your pixel size> --focus Tumor \
    --split Mesenchymal_Vim=Tumor_Putative,Stromal_Vim,Boundary_Vim \
    --out spatial_<your-sample-id>
```

`--populations` is CellTune's own Export → Populations (or CellTable) file;
`--type-col` defaults to `Pred_AVG`, the averaged-probability call from page
9. `--roi-map` is only needed if the CellTune project was built from ROI
folders (page 7) rather than one whole-slide file.

**`--split`** is for exactly the case page 9 raised: a cell type defined by a
marker that is not specific to what you actually mean by it. The example above
splits a marker-only `Mesenchymal_Vim` call into three by region — cells
inside the tumor region become `Tumor_Putative`, cells elsewhere become
`Stromal_Vim`, and cells near the tumor edge (within `--boundary-um` of page
1's output) become `Boundary_Vim` rather than a guess either way. Two cell
types (`Garbage`, `Ambiguous` by default, via `--exclude`) are left out of
every statistic, since CellTune's own convention is that they are not real
cell types.

Also computes composition by region, pairwise enrichment between types
(`--radius-um`, permutation-tested within tissue and false-discovery
corrected), and, with `--ark-dir`, a halo-tiled cell table ready for
**ark-analysis** if you want its own neighbourhood tools downstream.

---

## 3. One feature row per sample

```bash
python scripts/build_sample_features.py \
    --cells spatial_*/cell_types_final.csv --meta meta.csv \
    --group "Lymphoid=<TypeA>+<TypeB>" --ratio "A_over_B=<TypeA>/<TypeB>" \
    --by-region Tumor --diversity-um 30 --knn 1 \
    --out features.csv
```

`--group` and `--ratio` take **cell-type names exactly as they appear in
your CellTune export** (for example `CD8_Tcell`), joined with `+` — not
marker names, and no commas. A name that is not a cell type stops the run
with a list of the real cell types, so a typo cannot quietly turn into a
group of 0 cells. A `--ratio` side may also use a `--group` name.

`--meta` is a CSV you write: `sample_id`, `px_um`, and whatever grouping
column you intend to compare on (e.g. `group`). Pass `--rois` (the CSV from
page 7) if you want one row per ROI instead of per slide — see the warning in
step 4 before doing that.

| Feature family | Flag | What it gives you |
|---|---|---|
| Density, proportion | *(always)* | cells of each type per mm² / share of all cells |
| Your own groups | `--group NAME=TYPE1+TYPE2` | sums across types you define, repeatable |
| Your own ratios | `--ratio NAME=A+B/C+D` | repeatable; a zero denominator is `NaN`, never 0 or infinity |
| By region | `--by-region <class>` | the same densities, restricted to one H&E region, repeatable |
| Neighbourhood diversity | `--diversity-um` | Shannon index of each cell's neighbours |
| Nearest-neighbour distance | `--knn <k>` | mean distance to the k nearest cells of another type |
| Distance bands | `--ring-index`, `--ring-width-um`, `--ring-max-um` | composition of what surrounds one index type, by distance |
| Enrichment | `--enrichment` | pairs of types nearer each other than chance (slow) |
| Functional markers | `--markers`, `--pairs` | mean / fraction-positive within chosen types |

A type that is absent from a slide (or ROI) gets density and proportion
**0** there: zero cells is a real measurement, and leaving it blank would drop
that slide from step 4. What stays `NaN` (blank) is what cannot be measured: a
density in a region the slide does not contain, a distance or diversity
around a type that is absent, a ratio with a zero denominator, and
neighbour-based features of a type with fewer than `--min-type-cells`
(default 10) cells in that unit.

---

## 4. Compare groups honestly

```bash
python scripts/feature_stats.py \
    --features features.csv --group-col group --ref-group <your control group> --out stats.csv
```

`--ref-group` names the reference: every `estimate` is the other group minus
this one. Without it, the alphabetically first group is the reference (with
`WT` and `KO` that is `KO`, which flips every sign), and the script prints
which one it used.

**The unit of replication is the animal or slide, never the ROI and never the
cell.** Ten ROIs from one slide are ten looks at one biological sample, not
ten samples — testing them as independent shrinks p-values by a factor that
grows with how different slides are from each other. This script fits a
random-intercept-per-sample model for every feature, then corrects across
features (Benjamini-Hochberg).

Guardrails it enforces rather than leaving to you to remember:

- A feature needs at least `--min-samples` (default 3) samples in **every**
  group, or its p-value is reported as `NaN` — "descriptive only." Two
  animals per group is a real effect size, not a valid test.
- If every sample contributes exactly one row, a random intercept cannot be
  fit, so it falls back to a rank test (Mann-Whitney) on samples.
- If the mixed model gives any warning (it did not converge, or the
  animal-to-animal spread came out as zero, which is common), it falls back
  to Student's t-test on the per-animal averages with (animals − 2) degrees
  of freedom — the same test the model makes when every animal has the same
  number of ROIs — and the `method` column says so. Nothing fails silently.

Needs `statsmodels`, which is not in `environment-cellsam.yml`:

```bash
conda run -n akoya-cellsam pip install statsmodels
```

---

## Method credit

The feature families in step 3 follow Liu, Calvet-Mirabent et al. (Angelo
lab, HIV lymph nodes, 2026), generalised so none of it depends on one tissue
or panel.

---

Next: [11. Changing things](11-changing-things.md)
