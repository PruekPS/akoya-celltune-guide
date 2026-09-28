# 6. Stage 02 — segmentation, and the CellTune file

This is the long step. It finds every cell and writes the label image CellTune
needs.

Two methods run:

- **StarDist** finds nuclei from the nuclear channel. Fast, a few minutes.
- **CellSAM** finds whole cells directly from the image. This is the slow part
  and the one that benefits from a GPU.

A **nested-label merge** is available but **off by default**. CellSAM does
sometimes draw two outlines for one cell, and the merge folds the inner into
the outer — but measured against hand-clicked nuclei it made the result
slightly worse, and it is very slow. See "The nested-label merge" below.

---

## The recommended way: two jobs

```bash
ACCOUNT=<GROUP> QOS=<GROUP> pipeline/run_stage_02_split.sh samples/my_slide.csv results
```

That submits **two** jobs:

1. **GPU job** — CellSAM only, then stops.
2. **CPU job** — the merge, the cell tables, the report and the CellTune
   export. Starts automatically, and only if job 1 succeeded.

### Why two jobs and not one

Only CellSAM uses the GPU. Everything after it is single-threaded CPU work, and
on a large slide the merge alone can run over half an hour. Many clusters
**cancel a job whose GPU sits idle** — a common threshold is 0% utilisation for
one hour. Holding a GPU through the CPU phase therefore risks losing the job
*after* the expensive part has already succeeded. Splitting avoids that, and
frees the GPU for someone else.

If job 1 succeeds and job 2 fails, you do not have to redo segmentation — see
"Resuming" below.

---

## If you have no GPU

```bash
pipeline/sbatch_mail.sh --account=<GROUP> --qos=<GROUP> \
    --cpus-per-task=14 --time=24:00:00 \
    pipeline/run_stage_02.slurm samples/my_slide.csv results
```

One job, all on CPU. Expect **14–16 hours** for a whole slide instead of about
1.5 hours on a GPU. It works; it is just slow, and nothing can cancel it for
being idle.

---

## Block size — the setting worth understanding

The slide is too big for the model at once, so it is cut into square blocks.
**CellSAM resizes every block to exactly 1024×1024 pixels before looking at
it.** Block size therefore decides the resolution the model works at, not just
how the image is cut up.

At a 0.5 µm pixel size, a 10 µm cell reaches the model as:

| Block | What happens | Cell reaches model as |
|---|---|---|
| 512 px | upscaled ×2 | ~40 px |
| **1024 px** | **untouched** | **~20 px** |
| 2048 px | downscaled ×2 | ~10 px |

**512 is the default here, and it is the default because it was measured.** On
one slide, scored against 734 hand-clicked nuclei in two fields:

| Block | Cells found | Recall | Precision | F1 |
|---|---|---|---|---|
| **512 px** | 1,473,195 | **0.830** | 0.686 | **0.751** |
| 1024 px | 643,653 | 0.510 | 0.728 | 0.599 |

At 1024 the model misses nearly half the cells a person marked. The tempting
argument — "1024 is the model's native size, so nothing is resampled" — is
true and still leads to the wrong answer: upscaling a 512 px block shows the
model cells at roughly 40 px instead of 20 px, and it detects small cells much
better that way.

Bigger blocks are faster and worse. **Do not raise this for speed without
re-scoring against hand labels on your own tissue** — the best value depends on
your pixel size and how big your cells are, so measure rather than inherit this
number.

To change it:

```bash
CELLSAM_BLOCK=512 ACCOUNT=<GROUP> QOS=<GROUP> \
    pipeline/run_stage_02_split.sh samples/my_slide.csv results
```

**Keep it the same across slides you intend to compare.** Block seams move when
it changes, so results are not directly comparable across different values.

---

## The nested-label merge, and why it is off

CellSAM occasionally returns two overlapping outlines for one cell. The merge
folds the smaller into the larger when most of it sits inside the larger one
grown by a few micrometres. That sounds like a clear win. It was measured, on
one slide, against 734 hand-clicked nuclei in two fields:

| | recall | precision | F1 |
|---|---|---|---|
| **No merge** | 0.830 | 0.686 | **0.751** |
| Merge | 0.798 | 0.689 | 0.739 |

It folded 7.7–11% of labels and came out **slightly worse**, in both fields and
at both matching radii. Recall falls about 0.03 while precision gains 0.003 —
so most of what it folds away are real cells absorbed into a neighbour, not the
duplicate outlines it was written to remove.

It is also slow: **43–51 minutes for one 1-megapixel crop**, single-threaded, on
a slide of over 1,000 megapixels.

Switch it on with `MERGE_NESTED=1` only if you have measured it on your own
tissue and it helps there. Denser or sparser tissue than lung may behave
differently — but measure before paying for it.

## Watching it

```bash
squeue -u $USER
```

You will see both jobs. The second sits in `PD` with reason `(Dependency)`
until the first finishes — that is correct, not a fault.

CellSAM writes its progress to a file you can read at any time:

```bash
cat results/<your-sample-id>/02_segment/cells_cellsam_raw.progress.json
```

```json
{"block_index": 709, "next_id": 532823, "block": 1024, "halo": 128,
 "H": 33120, "W": 30720, "done": false}
```

`block_index` counts blocks finished. Total blocks is roughly
`(H ÷ block) × (W ÷ block)` — here about 990. So 709 of 990 done.

---

## The file you came for

When both jobs finish:

```bash
ls -la results/<your-sample-id>/02_segment/celltune/
```

```
<your-sample-id>_segmentation_labels.tif
```

Check it is what CellTune expects:

```bash
python -c "
import tifffile, numpy as np
a = tifffile.imread('results/<your-sample-id>/02_segment/celltune/<your-sample-id>_segmentation_labels.tif')
print('dtype      :', a.dtype, '(should be uint32)')
print('shape      :', a.shape, '(should match your slide)')
print('background :', (a == 0).any(), '(should be True)')
print('cells      :', len(np.unique(a)) - 1)
"
```

You want `uint32`, a shape matching your slide, background present, and a cell
count in the hundreds of thousands for a whole slide.

> **If the folder is missing**, the export was skipped. It is logged as a
> *warning*, not an error, so the job can still say `COMPLETED`. Read the log:
> `grep -i celltune logs/akoya_02_cpu_*.out`. The usual cause is asking to
> export a method that did not run.

---

## Resuming

If the CPU job fails but CellSAM had finished, do **not** start over. The raw
labels are on disk. Confirm:

```bash
cat results/<your-sample-id>/02_segment/cells_cellsam_raw.progress.json
```

If it says `"done": true`, finish with:

```bash
pipeline/sbatch_mail.sh --account=<GROUP> --qos=<GROUP> \
    --cpus-per-task=14 --time=06:00:00 \
    pipeline/run_stage_02.slurm samples/my_slide.csv results --resume --cellsam-block-px 1024
```

`--resume` reads the progress file and skips segmentation entirely.

**The block size must match the original run.** If they differ, the saved
labels will not line up.

---

Next: [7. Loading into CellTune](07-into-celltune.md)
