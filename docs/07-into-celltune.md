# 7. Loading into CellTune

CellTune needs **two** things. Only one of them is something you made.

| CellTune wants | Give it | Convert it? |
|---|---|---|
| The image | your original `.qptiff` | **No — unmodified** |
| The segmentation | `<sample>_segmentation_labels.tif` | already in the right format |

CellTune reads QPTIFF directly. Do not convert, flatten or re-export your
slide — point CellTune at the file that came off the instrument.

---

## What the segmentation file is

A single 2-D TIFF the same width and height as your slide, where:

- every pixel belonging to cell 1 holds the number `1`,
- every pixel of cell 2 holds `2`, and so on,
- background is `0`,
- the numbers are stored as **uint32** (unsigned 32-bit integers), so there is
  room for millions of cells.

That is exactly the format CellTune documents for segmentation input, which is
why no conversion is needed.

---

## Getting it off the cluster

The file is large — roughly 4 GB for a whole slide. From a terminal **on your
own computer**:

```bash
scp <your-username>@<cluster-login-host>:/blue/<GROUP>/<USER>/akoya-pipeline/results/<your-sample-id>/02_segment/celltune/<your-sample-id>_segmentation_labels.tif ~/Desktop/
```

You also need the `.qptiff` locally if it is not already, which is larger
again.

---

## Putting it in your CellTune project

Copy or link the file into your CellTune project's `Segmentations/` folder,
**keeping the filename exactly as it is**:

```
<your CellTune project>/
├── Segmentations/
│   └── <your-sample-id>_segmentation_labels.tif
└── ...
```

The name matters: CellTune matches the segmentation to the image by name. If
you rename the file, rename it consistently with how the image is registered in
CellTune.

Then in CellTune, add the image through its own Images step, pointing at the
original `.qptiff`.

---

## If CellTune will not accept the file

| Symptom | Likely cause | Fix |
|---|---|---|
| "unsupported data type" | not uint32 | re-check with the snippet on page 6 |
| Cells appear in the wrong place | image and segmentation are different sizes | make sure both come from the same slide, and that the `.qptiff` was not cropped or re-exported |
| Nothing loads | filename does not match the image | rename to match, keeping `_segmentation_labels.tif` |
| Everything is one blob | background is not 0 | re-check with the snippet on page 6 |

For anything specific to CellTune itself, its own documentation at
**celltune.org** is the authority — this guide only covers producing the file.

---

Next: [8. Changing things](08-changing-things.md)
