#!/bin/bash
# Stage 02 on its own: segment a slide with CellSAM and write a CellTune-ready label image.
#
# Stages 00 and 01 must already have run for the sample (this script does not
# re-checksum the qptiff or redo image QC). Use pipeline/run_stages_00_01.slurm
# for a brand-new slide, then this.
#
#   # CPU (no GPU allocation; ~14-16 h for a whole slide)
#   pipeline/sbatch_mail.sh --account=<ACCOUNT> --qos=<QOS> \
#       pipeline/run_stage_02.slurm samples.csv results_osa
#
#   # GPU -- add the allocation on the command line; sbatch flags override the
#   # #SBATCH lines below, and the script switches itself to cuda when it sees one.
#   pipeline/sbatch_mail.sh --account=<ACCOUNT> --qos=<QOS> \
#       --partition=hpg-rtx6000 --gres=gpu:1 --cpus-per-task=8 --time=06:00:00 \
#       pipeline/run_stage_02.slurm samples.csv results_osa
#
# GPU notes (checked 2026-09-26): the <GROUP> QoS allows gres/gpu=2 -- it was 0
# until mid-September, which is why every CellSAM run so far has been on CPU and
# why pipeline/run_cellsam_compare.slurm hard-codes --cellsam-device cpu. The
# GPU partitions are hpg-rtx6000, hpg-b200, hpg-turin and hwgui; there is no
# partition called "gpu", and the default (hpg-default) has none. The -b QoS
# (<GROUP>-b) has gres/gpu=0, so it cannot be used for GPU work.
#
# THE IDLE-GPU POLICY. The cluster cancels a job whose GPU sits at 0%
# utilisation for an hour (docs.rc.ufl.edu/scheduler/idle_gpu_policy/). What
# trips it here is the NESTED-LABEL MERGE, not CellSAM and not the block size.
# Evidence from job 43475675: CellSAM finished all 3,899 blocks at 512 px in
# about 2 h 13 m, and the job was cancelled 1 h 41 m later, during the
# single-threaded merge. The merge printed nothing at the time, so the log
# looked as if CellSAM had stalled -- the progress record showed otherwise, and
# reading it rather than the log is what settled this.
#
# The merge is now OFF by default (see the measurement below), which removes
# this problem at the source. When it is switched on, the GPU/CPU split
# (pipeline/run_stage_02_split.sh) keeps it off the GPU. Raising the block size
# was also tried as a fix
# and is REJECTED: it costs a great deal of accuracy for no necessary gain --
# see the block-size note below. Two further safeguards live here: a hard check
# that torch can actually see the GPU, so a CPU fallback inside a GPU job fails
# in seconds rather than an hour, and a sampler logging utilisation every two
# minutes so the next run is judged on evidence. Nothing here keeps the GPU
# artificially busy; that would dodge the policy rather than satisfy it, and
# splitting is what UF Research Computing themselves recommend. StarDist is not
# a factor -- it is 0.08 h of a 15.6 h PS81 run, with CellSAM at 99.4%.
#
# CPU baseline to beat, measured: PS81 15.7 h / 12.3 GB, PS88 ~14 h / 19.4 GB,
# PS82 7.5 h, all at 14-16 pinned threads. Wall limits: <GROUP> 31 days,
# <GROUP>-b 4 days. Group caps: <GROUP> cpu=14, mem=112000M, so --mem stays
# at or under 100gb.
#
# Knobs (environment variables, so the positional args stay simple):
#   METHODS          what stage 02 runs          (default: cellsam,expansion)
#   CELLTUNE         1 writes the CellTune TIFF  (default: 1)
#   CELLTUNE_METHOD  which method to export      (default: cellsam)
#   MERGE_NESTED     1 folds nested CellSAM labels into their neighbour (default: 0 -- see below)
#   CELLSAM_DEVICE   force cuda/cpu/mps          (default: cuda if a GPU is allocated, else cpu)
#   RUN_NOTE         text shown in the report header
# Anything after the two positional arguments is passed straight to 02_segment.py,
# so --resume, --cellsam-bbox-threshold and friends still work.
#
# Needs the akoya-cellsam environment and a DeepCell token in
# ~/.config/deepcell_token (CellSAM fetches its weights from users.deepcell.org).
# SLURM does not read ~/.bashrc, hence the explicit source. No `set -x`, no `env`:
# the token must never reach a log.
#
# Real account/QoS/paths belong in your untracked NOTES.local.md, not in this file.
#SBATCH --job-name=akoya_02
# Email when the job ends, fails, or can never start because a job it depends on
# failed. HiPerGator drops this directive, so submit via pipeline/sbatch_mail.sh,
# which re-applies it with scontrol after submission and verifies it took.
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
SAMPLES="${1:?usage: sbatch pipeline/run_stage_02.slurm samples.csv [out_dir] [extra 02 args]}"
OUT="${2:-results}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true
NOTE="${RUN_NOTE:-Stage 02 on HiPerGator $(date +%F)}"
mkdir -p logs

if [ -f "$HOME/.config/deepcell_token" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.config/deepcell_token"
fi
if [ -z "${DEEPCELL_ACCESS_TOKEN:-}" ]; then
  echo "DEEPCELL_ACCESS_TOKEN is not set; CellSAM cannot fetch its weights." >&2
  exit 2
fi
echo "DeepCell token: present (${#DEEPCELL_ACCESS_TOKEN} characters)"   # length only, never the value

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check every input first. A 15-hour job that dies in its tenth second for a
# missing samples sheet has cost a queue slot and a day of waiting.
# shellcheck disable=SC1091
source pipeline/preflight.sh
preflight "$SAMPLES" "${AKOYA_ENV:-akoya-cellsam}" || exit 2

conda activate "${AKOYA_ENV:-akoya-cellsam}"
# Environment libraries first: the node's /lib64 libstdc++ is older than the
# GLIBCXX conda-forge's scipy needs.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Pin every thread pool to the allocation, not just OpenMP: MKL, OpenBLAS,
# numexpr and TensorFlow (StarDist loads it into the same process) otherwise
# size their pools from the node rather than the job, which cost ~7-8x per
# core-second on the first PS88 run.
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTRAOP_THREADS="$OMP_NUM_THREADS" TF_NUM_INTEROP_THREADS=2

# Pick the device from the allocation rather than trusting the submitter to keep
# two flags in step: SLURM sets these only when a GPU was actually granted.
if [ -n "${CELLSAM_DEVICE:-}" ]; then
  DEVICE="$CELLSAM_DEVICE"
elif [ -n "${SLURM_JOB_GPUS:-}${SLURM_GPUS_ON_NODE:-}${CUDA_VISIBLE_DEVICES:-}" ]; then
  DEVICE="cuda"
else
  DEVICE="cpu"
fi
echo "CellSAM device: $DEVICE  (SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}' CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}')"

GPU_SAMPLER_PID=""
if [ "$DEVICE" = "cuda" ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
  # Fail in seconds, not after an hour. A GPU allocation whose torch cannot see
  # the card runs the whole slide on CPU inside a GPU job -- which wastes the
  # allocation and gets cancelled by the idle-GPU policy anyway.
  if ! python - <<'PYEOF'
import sys, torch
if not torch.cuda.is_available():
    print(f"  torch {torch.__version__} cannot see a GPU "
          f"(torch.cuda.is_available() is False, built for CUDA {torch.version.cuda}).", file=sys.stderr)
    sys.exit(1)
print(f"  torch {torch.__version__} sees {torch.cuda.device_count()} GPU(s): "
      f"{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")
PYEOF
  then
    echo "ERROR: --gres=gpu was requested but this environment's torch has no usable CUDA." >&2
    echo "       Rebuild the env from envs/environment-cellsam.yml (it pins the cu128 wheels)," >&2
    echo "       or submit without --gres/--partition to run on CPU." >&2
    exit 2
  fi

  # Sample utilisation into the log so the next run is diagnosable rather than
  # guessed at. Reading a counter does not keep the GPU busy and is not meant to:
  # see the block-size note below for the actual remedy.
  ( while true; do
      echo "[gpu $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used \
            --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed')"
      sleep 120
    done ) &
  GPU_SAMPLER_PID=$!
  trap 'if [ -n "$GPU_SAMPLER_PID" ]; then kill "$GPU_SAMPLER_PID" 2>/dev/null || true; fi' EXIT
fi

ARGS=(--methods "${METHODS:-cellsam,expansion}" --cellsam-device "$DEVICE")
# An `if` rather than `[ ... ] && ...`: under `set -e` the one-liner form only
# survives a false test by a subtle exemption, and this file must not depend on it.
if [ "${MERGE_NESTED:-0}" = "1" ]; then
  ARGS+=(--cellsam-merge-nested)
fi
if [ "${CELLTUNE:-1}" = "1" ]; then
  ARGS+=(--celltune-export --celltune-method "${CELLTUNE_METHOD:-cellsam}")
fi

# THE NESTED-LABEL MERGE IS OFF BY DEFAULT, and this is why. Measured on PS88
# against 734 hand-clicked nuclei in two fields, at the 512 px block:
#                   recall   precision   F1
#   unmerged        0.8297     0.6858   0.7509
#   merged          0.7984     0.6886   0.7394     (3 um match radius)
#   unmerged        0.9069     0.7438   0.8173
#   merged          0.8725     0.7500   0.8066     (5 um match radius)
# It folds 7.7-11% of labels and makes the result slightly WORSE -- worse in
# both fields at both radii. Recall falls by about 0.03 while precision gains
# 0.003, so the labels being folded are mostly real cells absorbed into their
# neighbours, not the duplicate outlines the merge was written to remove.
# It is also extremely slow: 43-51 minutes for a single 1-megapixel crop, on a
# slide of 1,017 megapixels, single-threaded.
# Switch it on with MERGE_NESTED=1 only if you have re-measured it on your own
# tissue and it helps there. One slide is thin evidence for a benefit; it is
# ample evidence against paying hours for a loss.
#
# Block size. CellSAM resizes EVERY block to exactly 1024x1024 before the model
# sees it -- cellSAM/sam_inference.py: `imgs = [T.Resize((1024, 1024))(img) ...]`
# -- so the block size decides the resolution the model actually works at, not
# just how the slide is cut up. On these slides (0.5055 um/px) a 10 um cell
# reaches SAM as:
#     512 px blocks  -> 3,840 blocks on PS88, upscaled x2.00, cell ~40 px
#    1024 px blocks  ->   960 blocks on PS88, no resampling, cell ~20 px
#    2048 px blocks  ->   240 blocks on PS88, DOWNSCALED x0.50, cell ~10 px
# MEASURED ON PS88, 2026-09-27, against hand-clicked nuclei in two fields
# (734 marks). Block size is NOT just a tiling choice -- it changes how big a
# cell looks to the model, and that changes the answer a great deal:
#     512 px  -> 1,473,195 cells   recall 0.830  precision 0.686  F1 0.751
#    1024 px  ->   643,653 cells   recall 0.510  precision 0.728  F1 0.599
# At 1024 CellSAM misses nearly half the cells that were marked by hand. The
# "native input size, nothing resampled" argument for 1024 sounded right and is
# wrong in practice: upscaling a 512 px block presents cells at ~40 px instead
# of ~20 px, and the model detects small cells far better at that size.
# **512 is the default because it is the setting that matches ground truth.**
# Do not raise it for speed without re-scoring against hand labels.
#
# Seam placement depends on this, and the halo scales with it (stage 02 uses
# block/8), so it is printed rather than applied silently. CPU keeps stage 02's
# own 512 px default, which is what the existing PS81/PS82/PS88 results used --
# a GPU run at 1024 is therefore NOT directly comparable with those. Set
# CELLSAM_BLOCK to override; pass --cellsam-block-px yourself and this keeps out
# of the way.
case " $* " in
  *" --cellsam-block-px "*) BLOCK_SET_BY_USER=1 ;;
  *)                        BLOCK_SET_BY_USER=0 ;;
esac
if [ "$BLOCK_SET_BY_USER" = "0" ]; then
  if [ "$DEVICE" = "cuda" ]; then
    CELLSAM_BLOCK="${CELLSAM_BLOCK:-512}"
  else
    CELLSAM_BLOCK="${CELLSAM_BLOCK:-512}"
  fi
  ARGS+=(--cellsam-block-px "$CELLSAM_BLOCK")
  echo "CellSAM block size: ${CELLSAM_BLOCK} px (device $DEVICE); CellSAM resizes every block to"
  echo "  1024x1024, so this is the resolution the model works at. Seams and the halo depend on it:"
  echo "  keep it fixed across slides you intend to compare. Existing PS81/PS82/PS88 runs used 512."
fi

echo "stage 02: ${ARGS[*]} $*"
python scripts/02_segment.py --samples "$SAMPLES" --out "$OUT" --run-note "$NOTE" "${ARGS[@]}" "$@"

echo
echo "Report:   $OUT/<sample>/reports/02_segment.html"
if [ "${CELLTUNE:-1}" = "1" ]; then
  echo "CellTune: $OUT/<sample>/02_segment/celltune/<sample>_segmentation_labels.tif"
  echo "          uint32, 0 = background. Copy or symlink it into your CellTune"
  echo "          project's Segmentations/ folder keeping that filename, and point"
  echo "          CellTune's Images step at the original qptiff, unmodified."
  echo "          If the file is missing, read the report: an export for a method"
  echo "          that did not run is logged as a warning, not an error."
fi
