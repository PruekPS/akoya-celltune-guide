# Akoya → CellTune: segmentation, ROIs, and training, step by step

This repository takes a **PhenoCycler-Fusion `.qptiff` slide** all the way to
**a trained CellTune classifier and a per-sample feature table**, running the
heavy lifting on an HPC cluster with SLURM and the labelling in CellTune's own
desktop app.

It is written for someone who has **never used a terminal or a cluster before**.
Every step is spelled out, including how to log in and how to check you are in
the right folder. If a step assumes something, it says so.

Work through the pages in order.

---

## What you will end up with

A CellTune-ready label image:

```
results/<your-slide>/02_segment/celltune/<your-slide>_segmentation_labels.tif
```

— every cell filled with its own number, background 0 — plus, if you follow
the whole guide: a set of ROIs cut for labelling, a trained CellTune classifier
applied to every cell on the slide, and one feature row per sample ready for
an honest group comparison. Pages 1–6 get you the label image; 7–10 are
everything after it.

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
| 7 | [Choosing and cutting ROIs](docs/07-choosing-rois.md) | Why ROIs, the rule for picking them, cutting them for CellTune |
| 8 | [Loading into CellTune](docs/08-into-celltune.md) | ROI folders or one whole slide, what to point CellTune at |
| 9 | [Training CellTune's classifier](docs/09-celltune-training.md) | What CellTune does, the two tables you write, the labelling cycle |
| 10 | [Spatial analysis and comparing groups](docs/10-spatial-and-features.md) | H&E regions, enrichment, a feature table, honest statistics |
| 11 | [Changing things](docs/11-changing-things.md) | Different slide, different folder, different settings |
| 12 | [When it goes wrong](docs/12-troubleshooting.md) | Real errors, what they mean, what to do |

---

## The short version

Once you have done the setup once, a new slide through segmentation is four
commands:

```bash
ssh <your-username>@<cluster-login-host>          # log in
cd /blue/<GROUP>/<USER>/akoya-pipeline            # go to the project
nano samples/my_slide.csv                          # describe your slide
ACCOUNT=<GROUP> QOS=<GROUP> pipeline/run_stage_02_split.sh samples/my_slide.csv results
```

Through a CellTune-ready set of ROIs, two more:

```bash
pipeline/sbatch_mail.sh --account=<GROUP> --qos=<GROUP> \
    pipeline/run_make_rois.slurm propose results my_slide --out rois/my_slide_rois.csv
# ... look at rois/my_slide_rois.geojson in QuPath first ...
pipeline/sbatch_mail.sh --account=<GROUP> --qos=<GROUP> \
    pipeline/run_make_rois.slurm crop results my_slide \
    --rois rois/my_slide_rois.csv --out CellTune_Data/my_project/Images
```

Anything in `<angle brackets>` is a placeholder you replace with your own value.
Page 1 explains what each one is.

---

## What is in this repository

```
scripts/       the pipeline itself (Python)
  00_ingest.py              reads the .qptiff, records what is in it
  01_image_qc.py            per-channel quality checks, tissue mask
  02_segment.py             segmentation, and the CellTune export
  make_rois.py              choose and cut ROIs for CellTune labelling
  celltune_regions.py       H&E regions -> a column on every cell
  celltune_spatial.py       join CellTune's cell types to those regions, enrichment
  build_sample_features.py one feature row per sample/ROI
  feature_stats.py          honest group comparison (random intercept per sample)
  akoyalib/                 shared helper code
pipeline/      SLURM job scripts (how to run on the cluster)
panels/        marker panel template, CellTune marker-table template
samples/       example samples.csv
environment-cellsam.yml   the software environment
docs/          the twelve pages above
```

## Method credits

Segmentation uses **CellSAM** (Israel, Marks et al., *Nature Methods* 2025) for
whole cells and **StarDist** (Schmidt et al., MICCAI 2018) for nuclei. Cell
typing uses **CellTune** (Bussi et al., *Nature Methods* 2026); the feature
families in page 10 follow Liu, Calvet-Mirabent et al. (Angelo lab, HIV lymph
nodes, 2026), generalised. This repository is the wrapper that runs all of it
on whole slides and ties the pieces together.

## Scope

Pages 1–6: from a raw slide to a CellTune-ready segmentation. Pages 7–10:
choosing ROIs, loading and training CellTune, and basic spatial analysis /
group comparisons on its output.

**Not covered:** registering an H&E section onto the slide (VALIS is linked
from page 10, but using it is not walked through here), designing your own
cell-type table (a scientific call only you can make — page 9 explains the
format, not the biology), and the earlier marker-quantification/normalisation
pipeline (`03_quantify.py`/`04_normalize.py`), which predates CellTune and
is not needed to reach it.
