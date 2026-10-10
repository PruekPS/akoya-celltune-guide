# 9. Training CellTune's classifier

This page is about the CellTune app itself, not the cluster. Everything here
happens on your own computer, for every panel and slide the same way — only
two tables change between projects.

---

## What CellTune is actually doing

A cell's raw marker brightness is not trustworthy on its own: signal from a
bright neighbour leaks into the cell next to it (spillover), a 2-D section
cuts cells through at random points, and a marker fades from bright to dim
with no natural cut-off. CellTune (Bussi et al., *Nature Methods* 2026) does
not try to find one threshold per marker. It trains a classifier from
examples you label, then shows you the cells it is least sure about.

| Idea | What it means | Why it helps |
|---|---|---|
| Spatial features | For each marker: mean, max, variance and pixel count inside the cell, just outside it, and further out | Lets the model see "bright in this cell but brighter next door" — what spillover looks like |
| Landmarks | A small set of cells so clear-cut nobody would argue with them | Seeds the first model without hand-labelling thousands of cells |
| Query by committee | Two different models (XGBoost, CatBoost) train on your labels; cells where they disagree are sampled next | Labelling effort goes where it actually changes the model |
| Honest labels | `Unidentified`, `Garbage`, `Ambiguous` are real options, not failures | Forcing a label onto a genuinely ambiguous cell teaches the model noise |

Things worth knowing before you start:

- **CatBoost is slow without an NVIDIA GPU** — the docs say 1–2 hours per
  training on CPU. Apple Silicon is not NVIDIA, so a Mac trains on CPU; a
  Windows machine with an NVIDIA GPU is faster.
- Free disk: roughly **2× your image size per project** (CellTune builds its
  own Zarr copy). At least 32 GB RAM is recommended.
- Free for academic non-commercial use; the app itself is compiled, not open
  source.

---

## The two tables you write once per panel

| File | Columns | Rules |
|---|---|---|
| Marker table | marker name, vendor name, expected expression, lineage (1/0) | Names: letters/digits/underscore only, matching across datasets |
| Cell-type table | `CellType`, `PrimaryMarker`, `SecondaryMarker`, `TertiaryMarker`, display colour | `PrimaryMarker` supports `&` (and), `|` (or), `!` (not); capitalised singular type names |

`panels/celltune_markers_mouse_io_template.csv` is a worked marker table for a
mouse immuno-oncology panel — copy it and edit for yours, the same way you
wrote your segmentation panel on page 3. Page 7 ("Writing your marker table") walks through
editing it and checking it against your slide before you crop.

**Write your own cell-type table; this guide does not ship one.** Which
markers define which cell type is a scientific call about your own biology,
not something to copy from someone else's panel. Two things worth deciding
deliberately while you write it:

- A marker that stains diffusely rather than crisply (check stage 01's
  report) is better set **non-lineage**, so it is excluded from the features
  that drive classification rather than quietly degrading them.
- If your panel has **no marker that is specific to your cell type of
  interest** — for example no tumor marker, with tumor cells sharing a marker
  with normal mesenchymal cells — do not force a marker-only definition.
  Define it broadly by marker (e.g. `Mesenchymal_Vim`) and split it by H&E
  region afterward, after CellTune's export, rather than pretending the marker
  alone settles it.

---

## Create the project and calculate features

New Project (light mode) → an empty folder inside `CellTune_Projects`, named
something like `<Tissue>_<MonDDYY>` → point it at `Images/` (page 8). Pick the
segmentation, drop channels you do not want. Then **Calculate → Cell
Features**, with:

- arcsinh factor **0.1** (the documented value for fluorescence — 100 is for
  mass imaging, not PhenoCycler),
- your real pixel size,
- at most 16 cores, default erode/dilate/environment distances.

Be patient here — CellTune's own documentation says the progress bar for this
step is still being improved, so a long pause is expected, not a hang.

---

## Explore, then set landmarks

**Explore** first: run the initial flowSOM clustering and look at the
heatmap. These clusters are for discovery, not final labels — check that
every type you expect shows up, and note anything you did not expect.

**Landmarks**: a strict gate — high on the type's primary marker, low on the
markers of other types. Where spillover is a concern, add "high in the cell,
low in its neighbours." View at least 30 gated cells before turning a gate
into a population. Automated landmarking from your cell-type table does this
for you, starting strict and relaxing until it has roughly 20 cells per type —
still review a sample before training. Rare types are usually worth labelling
by hand instead.

---

## The training cycle

1. Create a classifier using lineage-marker features.
2. Add your landmark populations as labels. Train with both XGBoost and
   CatBoost.
3. **Plot Confusions** shows where the two models disagree — that is where
   labelling helps most.
4. **Sample & Review** about 150–300 cells (border cells excluded, automatic
   channel selection on). Label each one: confirm, correct, or mark
   `Ambiguous`/`Garbage`.
5. Add the new labels, retrain, repeat.

**Stop when** the two models agree above roughly 85% for well-defined types,
90% for nuclear markers, 65% for fuzzy ones — and you have roughly 100–200
labels per type. Training produces four population sets; `Pred_AVG` (the
averaged probabilities) is the one to use.

---

## Predict every cell on the slide

Training ran on your labelled ROIs. The last step applies the trained
classifier to **every cell on the whole slide**, not just the ROIs — ROIs
were a labelling device (page 7), and the thing you actually analyse is the
full prediction. Do this through CellTune's own prediction step (or its
open-source algorithm notebooks, which can run on the cluster if the app
itself cannot reach your whole-slide data). Export the result as one table:
one row per cell, with its predicted type.

---

This is where the guide ends: you have CellTune's prediction for every cell
on the slide. Analysis after that (regions, spatial statistics, group
comparisons, figures) is not covered here.

Next: [10. Changing things](10-changing-things.md)
