#!/bin/bash
# Choose and cut ROIs from a segmented slide, for CellTune labelling -- the
# pipeline wrapper around scripts/make_rois.py (see its own docstring, and
# docs/celltune_training_manual.html section 3, for what this is and why: a
# slide is segmented once, whole, and ROIs are cut from it afterwards for
# CellTune labelling, not the other way round).
#
# Two sub-commands, like the script itself. This job does NOT chain them --
# propose is meant to be looked at (open the GeoJSON in QuPath) before you
# commit to cropping, so that is a deliberate, separate step in between.
#
#   # 1. propose ROIs for a slide that already has stage 02's CellTune export
#   pipeline/sbatch_mail.sh --account=<ACCOUNT> --qos=<QOS> \
#       pipeline/run_make_rois.slurm propose results_osa PS88 \
#       --roi-um 500 --target-cells 30000
#   # -> <out_dir>/PS88/rois/PS88_rois.csv and PS88_rois.geojson
#   # ... copy the geojson to your computer, open the slide in QuPath, then
#   # File > Import objects from file...; re-run with a different
#   # --seed/--exclude if it does not look right, THEN:
#
#   # 2. cut the chosen ROIs into CellTune's single-TIFF folder layout
#   pipeline/sbatch_mail.sh --account=<ACCOUNT> --qos=<QOS> \
#       pipeline/run_make_rois.slurm crop results_osa PS88 \
#       --out CellTune_Data/<PROJECT>/Images \
#       --marker-map panels/celltune_markers_mouse_io_template.csv
#
# Where things go. The proposal (csv + geojson) lives with the slide's own
# results, <out_dir>/<SLIDE>/rois/, because the box positions and cell IDs are
# only valid for the exact label image they were proposed from: re-segment (a
# different block size, the merge switched on) and they are stale. propose
# writes there by default and crop reads from there by default. The cropped
# CellTune images are NOT per slide -- one CellTune project pools ROIs from
# several slides and the TIFFs are large -- so crop's --out stays yours to
# name, e.g. CellTune_Data/<PROJECT>/Images.
#
# This wrapper's only job is to stop you typing --image/--px-um/--labels from
# memory for <SLIDE> -- 00_Indexing.md is explicit that a wrong --px-um
# silently rescales every distance later, and a label image one pixel off the
# slide's own size breaks both make_rois.py and CellTune. Both are read from
# what stages 00 and 02 already wrote, the same way every stage here reads
# what came before it rather than taking your word for it:
#   --image, --px-um   <out_dir>/<SLIDE>/00_ingest/ingest.json
#   --labels           <out_dir>/<SLIDE>/02_segment/celltune/<SLIDE>_segmentation_labels.tif
# Everything else -- --roi-um, --target-cells, --regions, --priority, --focus,
# --exclude, --marker-map, --role, --seed, ... -- is scripts/make_rois.py's own
# CLI, passed straight through unchanged; see
# `python scripts/make_rois.py propose --help` / `crop --help`. Passing
# --px-um, --labels, or (for propose) --out / (for crop) --rois yourself
# overrides the default, since argparse keeps the last occurrence of a flag.
#
# No GPU: propose and crop are numpy/pandas/tifffile I/O, nothing that needs
# one. Same conda environment as every other stage regardless (no separate
# env exists for this). Scikit-image pulls in Pillow, which celltune_regions.py
# needs for --regions/--exclude/--priority; if that import ever fails on a
# fresh environment, add pillow explicitly to envs/environment-cellsam.yml --
# this has not been hit in practice, just not separately verified.
#
# Needs: stage 00 (for ingest.json) and stage 02 run with --celltune-export
# (for the label TIFF) already done for <SLIDE>. Status: scripts/make_rois.py
# is covered by synthetic tests (tests/test_make_rois.py) and has been run on
# real slides (PS88 here; PS104 by a student). Still look at the GeoJSON in
# QuPath, and at one cropped ROI beside the slide, before trusting either.
#SBATCH --job-name=akoya_rois
# Email when the job ends, fails, or can never start because a job it depends
# on failed. Submit through pipeline/sbatch_mail.sh -- this directive alone
# does not reliably reach a job on this cluster.
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
CMD="${1:?usage: sbatch pipeline/run_make_rois.slurm <propose|crop> <out_dir> <SLIDE> [extra make_rois.py args]}"
OUT="${2:?usage: sbatch pipeline/run_make_rois.slurm <propose|crop> <out_dir> <SLIDE> [extra make_rois.py args]}"
SLIDE="${3:?usage: sbatch pipeline/run_make_rois.slurm <propose|crop> <out_dir> <SLIDE> [extra make_rois.py args]}"
shift 3
mkdir -p logs

case "$CMD" in
  propose|crop) ;;
  *) echo "first argument must be 'propose' or 'crop', not '$CMD'" >&2; exit 2 ;;
esac

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${AKOYA_ENV:-akoya-cellsam}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Check every input first, same as every other stage here -- cheap to fail
# now, expensive to discover after a crop has started writing TIFFs.
INGEST="$OUT/$SLIDE/00_ingest/ingest.json"
[ -f "$INGEST" ] || { echo "  FAIL   no $INGEST -- run stage 00 for $SLIDE first" >&2; exit 2; }
IMAGE=$(python -c "import json; print(json.load(open('$INGEST'))['image'])")
PXUM=$(python -c "import json; print(json.load(open('$INGEST'))['pixel_size_um'])")
echo "  OK     stage 00 ingest.json: image=$IMAGE px_um=$PXUM"

LABELS="$OUT/$SLIDE/02_segment/celltune/${SLIDE}_segmentation_labels.tif"
[ -f "$LABELS" ] || {
  echo "  FAIL   no $LABELS -- run stage 02 with --celltune-export (or CELLTUNE=1, the" >&2
  echo "         run_stage_02.slurm default) for $SLIDE first" >&2
  exit 2
}
echo "  OK     stage 02 CellTune export: $LABELS"

# Default location for the proposal: with the slide's own results, because it is
# only valid for this exact label image (see the header). Either default is
# skipped when you pass the flag yourself.
ROI_DIR="$OUT/$SLIDE/rois"
ROI_CSV="$ROI_DIR/${SLIDE}_rois.csv"
case " $* " in *" --out "*) USER_OUT=1 ;; *) USER_OUT=0 ;; esac
case " $* " in *" --rois "*) USER_ROIS=1 ;; *) USER_ROIS=0 ;; esac

if [ "$CMD" = "propose" ]; then
  DEFAULTS=()
  if [ "$USER_OUT" = "0" ]; then
    mkdir -p "$ROI_DIR"
    DEFAULTS=(--out "$ROI_CSV")
    echo "  OK     proposal will be written to $ROI_CSV (and the .geojson beside it)"
  fi
  echo "make_rois.py propose --labels $LABELS --sample-id $SLIDE --px-um $PXUM ${DEFAULTS[*]:-} $*"
  python scripts/make_rois.py propose --labels "$LABELS" --sample-id "$SLIDE" --px-um "$PXUM" ${DEFAULTS[@]+"${DEFAULTS[@]}"} "$@"
else
  DEFAULTS=()
  if [ "$USER_ROIS" = "0" ]; then
    [ -f "$ROI_CSV" ] || {
      echo "  FAIL   no $ROI_CSV -- run 'propose' for $SLIDE first (or pass --rois <csv>)" >&2
      exit 2
    }
    DEFAULTS=(--rois "$ROI_CSV")
    echo "  OK     using the proposal at $ROI_CSV"
  fi
  echo "make_rois.py crop --image $IMAGE --labels $LABELS --px-um $PXUM ${DEFAULTS[*]:-} $*"
  python scripts/make_rois.py crop --image "$IMAGE" --labels "$LABELS" --px-um "$PXUM" ${DEFAULTS[@]+"${DEFAULTS[@]}"} "$@"
fi
