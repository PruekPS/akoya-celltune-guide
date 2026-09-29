# Akoya → CellTune: segmentation, step by step

This repository takes a **PhenoCycler-Fusion `.qptiff` slide** and produces the
**label image CellTune needs**, running on an HPC cluster with SLURM.

It is written for someone who has **never used a terminal or a cluster before**.
Every step is spelled out, including how to log in and how to check you are in
the right folder. If a step assumes something, it says so.

Work through the pages in order.

---

## What you will end up with

```
results/<your-slide>/02_segment/celltune/<your-slide>_segmentation_labels.tif
```

A label image: every cell filled with its own number, background 0. That file
plus your **original, unmodified `.qptiff`** are the two things CellTune wants.

---

## The pages, in order

| # | Page | What it covers |
|---|------|----------------|
| 1 | [Getting on the cluster](docs/01-getting-started.md) | Accounts, logging in, two-factor, what a terminal prompt means |
| 2 | [Finding your way around](docs/02-your-project-folder.md) | `cd`, `pwd`, `ls`, where files live, how to be sure you are in the right place |
| 3 | [Installing the software](docs/03-installation.md) | Conda environment, the DeepCell token, one-time setup |
| 4 | [Writing your samples.csv](docs/04-samples-csv.md) | The one file you must write yourself, column by column |
| 5 | [Stages 00 and 01](docs/05-stage-00-01.md) | Reading the slide, image quality checks |
| 6 | [Stage 02 → CellTune](docs/06-stage-02-celltune.md) | Segmentation, GPU vs CPU, the CellTune file |
| 7 | [Loading into CellTune](docs/07-into-celltune.md) | Where the file goes, what to point CellTune at |
| 8 | [Changing things](docs/08-changing-things.md) | Different slide, different folder, different settings |
| 9 | [When it goes wrong](docs/09-troubleshooting.md) | Real errors, what they mean, what to do |

---

## The short version

Once you have done the setup once, a new slide is four commands:

```bash
ssh <your-username>@<cluster-login-host>          # log in
cd /blue/<GROUP>/<USER>/akoya-pipeline            # go to the project
nano samples/my_slide.csv                          # describe your slide
ACCOUNT=<GROUP> QOS=<GROUP> pipeline/run_stage_02_split.sh samples/my_slide.csv results
```

Anything in `<angle brackets>` is a placeholder you replace with your own value.
Page 1 explains what each one is.

---

## What is in this repository

```
scripts/       the pipeline itself (Python)
  00_ingest.py     reads the .qptiff, records what is in it
  01_image_qc.py   per-channel quality checks, tissue mask
  02_segment.py    segmentation, and the CellTune export
  akoyalib/        shared helper code
pipeline/      SLURM job scripts (how to run on the cluster)
panels/        marker panel template
samples/       example samples.csv
environment-cellsam.yml   the software environment
docs/          the nine pages above
```

## Method credits

Segmentation uses **CellSAM** (Israel, Marks et al., *Nature Methods* 2025) for
whole cells and **StarDist** (Schmidt et al., MICCAI 2018) for nuclei.
The output format follows **CellTune**'s documented segmentation input
(celltune.org). This repository is the wrapper that runs them on whole slides.

## Scope

This covers stages 00–02 only: from a raw slide to a CellTune-ready segmentation.
Marker quantification, normalisation and cell typing are not included here.
