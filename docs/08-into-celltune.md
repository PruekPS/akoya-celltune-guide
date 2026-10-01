# 8. Loading into CellTune

CellTune runs on your own computer (macOS Apple Silicon, or Windows with a
GPU — there is no Linux build, so this cannot happen on the cluster). Before
launching it, **switch to light mode**: CellTune does not support dark mode
yet, and its own documentation says to change this first.

There are two ways to point CellTune at your data. Use the ROI folders from
page 7 unless you have a specific reason not to.

---

## The normal way: the ROI folders from page 7

```bash
scp -r <your-username>@<cluster-login-host>:/blue/<GROUP>/<USER>/akoya-pipeline/CellTune_Data/<PROJECT>/Images ~/CellTune_Projects/<PROJECT>_data/
```

In CellTune: **New Project** → a fresh, empty folder inside your
`CellTune_Projects` folder → point it at the `Images/` folder you just copied
down. CellTune sees each ROI subfolder as one image, already paired with its
own `segmentation_labels.tif` — no separate segmentation step needed, because
page 7's `crop` already put the right file in each folder. Confirm channels
(every ROI shares the same marker set, since they all came from one
`--marker-map`), then **Calculate → Cell Features**.

---

## The other way: one whole slide, one label file

Only worth it for a **small test slide** — the reasoning on page 7 for why a
whole slide does not scale into one CellTune project still applies here.

| CellTune wants | Give it | Convert it? |
|---|---|---|
| The image | your original `.qptiff` | **No — unmodified** |
| The segmentation | `<sample>_segmentation_labels.tif` from page 6 | already in the right format |

The segmentation file is a single 2-D TIFF the same width and height as your
slide: every pixel of cell 1 holds `1`, cell 2 holds `2`, background is `0`,
stored as **uint32** so there is room for millions of cells. That is exactly
CellTune's documented format, so nothing needs converting.

Getting it off the cluster (roughly 4 GB for a whole slide, and you need the
`.qptiff` too, which is larger again):

```bash
scp <your-username>@<cluster-login-host>:/blue/<GROUP>/<USER>/akoya-pipeline/results/<your-sample-id>/02_segment/celltune/<your-sample-id>_segmentation_labels.tif ~/Desktop/
```

Copy or link it into your CellTune project's `Segmentations/` folder,
**keeping the filename exactly as it is** — CellTune matches segmentation to
image by name:

```
<your CellTune project>/
├── Segmentations/
│   └── <your-sample-id>_segmentation_labels.tif
└── ...
```

Then add the image through CellTune's own Images step, pointing at the
original `.qptiff`.

---

## If CellTune will not accept the data

| Symptom | Likely cause | Fix |
|---|---|---|
| "unsupported data type" | not uint32 | re-check with the snippet on page 6 |
| Cells appear in the wrong place | image and segmentation are different sizes | make sure both come from the same slide, and that the `.qptiff` was not cropped or re-exported |
| Nothing loads (whole-slide way) | filename does not match the image | rename to match, keeping `_segmentation_labels.tif` |
| Nothing loads (ROI way) | channel names collide, or an ROI folder is incomplete | re-run `crop` — it refuses to write colliding names rather than merging them silently |
| Everything is one blob | background is not 0 | re-check with the snippet on page 6 |
| App looks visually broken | not in light mode | switch before launching — a real, documented requirement |

For anything specific to CellTune itself, its own documentation at
**celltune.org** is the authority — this guide only covers getting your data
into a shape it accepts.

---

Next: [9. Training CellTune's classifier](09-celltune-training.md)
