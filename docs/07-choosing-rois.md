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
    --marker-map panels/celltune_markers_<your-panel>.csv   # your table, see below
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

Each channel's TIFF is named from your **marker table** (`--marker-map`) —
see the next section. Without `--marker-map`, names are cleaned automatically
(`F4/80` → `F480`, `PD-L1` → `PDL1`), and the run stops if two channels collide
after cleaning rather than silently merging them.

### Writing your marker table (`--marker-map`)

This one CSV does two jobs. `crop` uses it to name each channel's TIFF, and
CellTune uses **the same file** later as its marker table (page 9). Write it
once per panel and use it for every slide stained with that panel.

`panels/celltune_markers_mouse_io_template.csv` is a worked example for a
mouse immuno-oncology panel. **Do not use it unchanged unless your panel is
that exact panel** — the names must match your slides, not ours.

| Column | What goes in it | Rules |
|---|---|---|
| `Marker` | the clean name: the TIFF's file name, and the name CellTune shows | letters, digits and `_` only — no spaces, `-`, `/` or `.`; each name once; keep it the same across all your slides |
| `Marker_orig` | the channel name **exactly as your slide file spells it** | upper/lower case does not matter; spaces, hyphens and slashes do |
| `Expected_Expression` | free text: which cells should be positive | for you and CellTune's display, not used by `crop`; put the text in `"quotes"` if it contains a comma |
| `Lineage` | `1` or `0` | `1` = a crisp marker that defines a cell type (CD45, CD3e, CD8 …); `0` = everything else: DAPI, state markers (Ki67, PD-1), and any channel that stained poorly |

**1. Find your slide's channel names.** `crop` reads them from the slide file
itself, not from your page 3 panel. Stage 00 recorded them, in the
`image_name` column:

```bash
python -c "import pandas as pd; print(pd.read_csv('results/<your-sample-id>/00_ingest/channels.csv')['image_name'].to_string())"
```

Every name printed needs one row in your table, spelled the same way in
`Marker_orig`.

**2. Copy the example and edit the copy.** Never edit the template itself — a
later `git pull` would then clash with your edits.

```bash
cp panels/celltune_markers_mouse_io_template.csv panels/celltune_markers_<your-panel>.csv
nano panels/celltune_markers_<your-panel>.csv
```

Keep the header line. One row per channel: delete rows for markers you do not
have, add rows for ones you do, and fix `Marker_orig` wherever your slide
spells a name differently. A channel your page 3 panel marked `failed` still
gets a row — `crop` writes every channel — but give it `Lineage` `0` so it
cannot drive cell typing. Save with `Ctrl`+`O`, `Enter`, exit with `Ctrl`+`X`.

**3. Check it against the slide before cropping.** Set the two paths on the
first line of the command, then paste it all:

```bash
python - panels/celltune_markers_<your-panel>.csv results/<your-sample-id>/00_ingest/channels.csv <<'EOF'
import csv, re, sys
marker_map, channels = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(marker_map, newline="")))
lut = {r["Marker_orig"].lower(): r["Marker"] for r in rows}
image = [r["image_name"] for r in csv.DictReader(open(channels, newline=""))]
problems = 0
for r in rows:
    if not re.fullmatch(r"[A-Za-z0-9_]+", r["Marker"]):
        print(f"BAD NAME   {r['Marker']!r}: Marker may only use letters, digits and _"); problems += 1
    if r.get("Lineage", "").strip() not in ("0", "1"):
        print(f"BAD LINEAGE {r['Marker']}: Lineage must be 1 or 0, not {r.get('Lineage')!r}"); problems += 1
dupes = {m for m in [r["Marker"] for r in rows] if [r["Marker"] for r in rows].count(m) > 1}
for m in sorted(dupes):
    print(f"DUPLICATE  Marker {m!r} appears more than once"); problems += 1
for n in image:
    if n.lower() in lut:
        print(f"ok         {n!r:22} -> {lut[n.lower()]}.tif")
    else:
        print(f"NOT IN MAP {n!r:22} -> {re.sub(r'[^A-Za-z0-9_]', '', n)}.tif (cleaned automatically; add a row)"); problems += 1
for o in sorted(set(lut) - {n.lower() for n in image}):
    print(f"UNUSED     Marker_orig {o!r} matches no channel on this slide (typo, or not in your panel)")
print("no problems" if problems == 0 else f"{problems} problem(s) to fix before cropping")
EOF
```

What each line means:

| Line | Means | Fix |
|---|---|---|
| `ok` | that channel will be saved under this name | nothing |
| `NOT IN MAP` | no row matches this channel; it falls back to the automatic name, which may not match your `Marker` column | add a row, or correct that row's `Marker_orig` |
| `UNUSED` | a row matches no channel on this slide — usually a spelling difference | correct `Marker_orig`, or delete the row if you do not have that marker |
| `BAD NAME` / `DUPLICATE` / `BAD LINEAGE` | the table breaks a rule above | fix that row |

Run it until it ends with `no problems`, then crop with
`--marker-map panels/celltune_markers_<your-panel>.csv`. If all your slides
were stained with the same panel, checking one slide is enough — but if any
slide was scanned with a different channel list, check that one too.

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
