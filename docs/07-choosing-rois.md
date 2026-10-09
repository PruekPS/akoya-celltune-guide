# 7. Choosing and cutting ROIs

Stage 02 segmented the **whole slide** — every cell, wherever it sits. CellTune
cannot usefully load the whole slide at once: its own feature calculation,
Zarr conversion, memory use and review time all grow with how many cells are
in the project, and one whole slide can hold over a million. This page cuts
smaller regions (ROIs) out of the slide for CellTune to actually work with.

**Segment once, whole; cut ROIs afterwards.** Not the other way round. Every
cell keeps the same ID it had in the whole-slide segmentation, so a label you
make inside an ROI is a label for that exact cell on the full slide — and
choosing where ROIs go can use the whole-slide cell table (how dense, which
tissue region, whether it passed QC).

---

## The two steps

```bash
# 1. propose — look at this before you commit to anything
pipeline/sbatch_mail.sh --account=<GROUP> --qos=<GROUP> \
    pipeline/run_make_rois.slurm propose results <your-sample-id>

# 2. crop — only after you've looked at step 1's output
pipeline/sbatch_mail.sh --account=<GROUP> --qos=<GROUP> \
    pipeline/run_make_rois.slurm crop results <your-sample-id> \
    --out CellTune_Data/<PROJECT>/Images \
    --marker-map panels/celltune_markers_mouse_io_template.csv
```

**Where things go.** The proposal (a `.csv` and a `.geojson`) is written to
`results/<your-sample-id>/rois/`, next to that slide's other results, and `crop`
reads it from there — you do not type either path. That is deliberate: the box
positions and cell IDs are only valid for the exact label image they were
proposed from, so if you segment the slide again (a different block size, the
merge switched on) the old proposal is stale and belongs to the old run. The
cropped images are different: one CellTune project pools ROIs from several
slides and the TIFFs are large, so you name that folder yourself with `--out`.
Pass `--out` to `propose` or `--rois` to `crop` only if you want a different
location.

These do **not** chain automatically — that gap between them is deliberate,
so you look at what was proposed before cutting anything.

Both read `--image`, `--px-um` and `--labels` for `<your-sample-id>` from what
stages 00 and 02 already wrote (`ingest.json`, and the CellTune label file from
page 6). You never type those three by hand.

---

## Step 1: propose

Lays a grid of square tiles over the slide (500 µm by default — about a
MIBI-TOF field, roughly 2,300 cells at typical PhenoCycler density), drops
tiles that are mostly empty glass or too close to an exclusion zone, and
draws a stratified sample from what is left:

| Rule | Default | Why |
|---|---|---|
| Eligibility | ≥60% occupied, ≥300 cells | rejects empty glass and folds |
| How many | enough to reach `--target-cells` (30,000) | roughly 13 ROIs at 2,300 cells each |
| Strata | one per H&E region, plus `interface` where two regions mix | rare tissue contexts are where a classifier is weakest |
| Allocation | equal across strata | a context covering 5% of the slide still gets its share |
| Validation | 20% of ROIs per stratum, held out | never trained on — your honest test set |
| Randomness | fixed `--seed` | same command, same ROIs; a different seed gives a different draw |

Without H&E regions, everything is one stratum — drop `--regions`,
`--priority` and `--focus` entirely and the rest still works.

Example output, on a real whole-slide run:

```
509 eligible tiles of 500 um; chose 11 (9 train, 2 validation), ~31,670 cells
role     train  validation
stratum
all         9           2
wrote results/PS88/rois/PS88_rois.csv and results/PS88/rois/PS88_rois.geojson
look before you commit: open the slide in QuPath, then File > Import objects from file... and pick the GeoJSON (it is data, not a script: do not run it in the script editor)
```

**Open the GeoJSON over the slide in QuPath.** Copy `results/<your-sample-id>/rois/<your-sample-id>_rois.geojson`
to your own computer, open the slide in QuPath **first**, then choose
**File → Import objects from file…** and pick the GeoJSON. The ROIs arrive as
annotations — yellow for train, cyan for validation — and the annotation list on
the left names them `<your-sample-id>_R01`, `_R02`, and so on. If you cannot see
them, press **Zoom to fit**: each box is about 1 mm across on a slide that is
many millimetres wide.

> **It is data, not a script.** Dropping the file onto QuPath while no slide is
> open, or pasting it into Automate → Script editor, treats it as code and fails
> with `Unexpected input: '{"type":'`. That error means "wrong menu", not "bad
> file". If instead the import itself fails with `Unable to parse PathClass`,
> your copy of `make_rois.py` is older than the colour fix — QuPath refuses
> a classification that has a name but no colour — so update the script and run
> `propose` again.

Every box should sit on tissue,
spread out, away from folds and bubbles. If it does not look right, change the
rule (`--roi-um`, `--exclude`, `--seed`) and run propose again — never
hand-pick ROIs because they "look like good tumor". That bakes your own
expectation into the training set before the classifier ever sees anything.

"No eligible ROIs" means `--min-occupancy` or `--min-cells` is too strict for
your tissue, or `--px-um` is wrong — the wrapper prints the pixel size it used
at the top of the log.

---

## Step 2: crop

Cuts each chosen ROI into the folder layout CellTune's single-TIFF projects
use:

```
CellTune_Data/<PROJECT>/Images/
├── <your-sample-id>_R01/
│   ├── DAPI.tif, CD45.tif, ... one TIFF per marker
│   └── segmentation_labels.tif
├── <your-sample-id>_R02/
│   └── ...
├── roi_map.csv     each ROI's offset on the slide
└── cut_cells.csv   cells the ROI edge slices through
```

Marker names are cleaned to CellTune's rule (letters, digits, underscore only)
through `--marker-map`, a CSV with `Marker_orig,Marker` columns —
`panels/celltune_markers_mouse_io_template.csv` is a worked example for a
mouse immuno-oncology panel; write your own for a different panel the same
way you wrote your segmentation panel on page 3. Without `--marker-map`,
vendor names are cleaned automatically, and the run stops if two channels
collide after cleaning rather than silently merging them.

**Cells in `cut_cells.csv` are cells the crop sliced in half at an ROI edge.**
Their signal is incomplete — exclude them when you sample cells to label, and
leave them out of any table you build from the labels later.

Check it before trusting it: in QuPath, open one ROI's channels next to the
original slide at the same spot — they must match — and open
`segmentation_labels.tif` over a channel to confirm outlines still follow real
cells, not cut through them.

---

## Flags worth knowing

| Flag | Default | Effect |
|---|---|---|
| `--roi-um` | 500 | ROI side length, in µm |
| `--target-cells` | 30000 | cells to pool across all chosen ROIs |
| `--regions` | *(none)* | H&E regions GeoJSON, already warped onto the slide (page 10) |
| `--focus` | *(none)* | which region class defines the `interface` stratum, e.g. `Tumor` |
| `--exclude` | *(none)* | GeoJSON of folds/bubbles/torn edges to avoid |
| `--validation-frac` | 0.2 | share of ROIs per stratum held out, never trained on |
| `--role` *(crop only)* | `all` | cut only `train` or only `validation` ROIs |

Full list: `python scripts/make_rois.py propose --help` /
`python scripts/make_rois.py crop --help`.

---

## Two kinds of ROI — do not mix them up

- **Labelling ROIs** (this page) teach the classifier. Chosen by the rule
  above, never by outcome, and spread across slides and tissue contexts.
- **Analysis units** are what you compare between groups later (page 10) —
  for a whole slide that is the slide itself, not an ROI. Never choose an
  analysis region because "the classifier found something there": that is
  selecting on the result.

---

Next: [8. Loading into CellTune](08-into-celltune.md)
