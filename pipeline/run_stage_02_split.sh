#!/bin/bash
# Stage 02 as two chained jobs: CellSAM on a GPU, then everything else on CPU.
#
#   pipeline/run_stage_02_split.sh samples.csv results_gpu1024 [extra 02 args...]
#
# Why split. CellSAM is the only part of stage 02 that uses the GPU. Everything
# after it -- the nested-label merge above all -- is single-threaded CPU work,
# and on PS88 the merge alone ran past 35 minutes. Holding a GPU through that
# wastes the allocation and trips HiPerGator's idle-GPU policy, which cancels a
# job whose GPU sits at 0% for an hour (docs.rc.ufl.edu/scheduler/idle_gpu_policy/).
# That is exactly how job 43518458 died on 2026-09-27: CellSAM finished all 990
# blocks in 1 h 31 m, then the merge idled the GPU for half an hour and the job
# was killed with the segmentation already on disk. Splitting is also what UF
# Research Computing recommend for this shape of workload.
#
# Step 1 (GPU)  02_segment.py ... --stop-after-cellsam   -> cells_cellsam_raw.npy
# Step 2 (CPU)  02_segment.py ... --resume               -> merge, tables, report,
#                                                           CellTune export
# Step 2 runs with --dependency=afterok, so it starts only if step 1 succeeded,
# and its --resume reads cells_cellsam_raw.progress.json and skips segmentation
# entirely. StarDist nuclei are recomputed in step 2 (about 6 minutes on PS88);
# that is a known, accepted cost of the split, not an oversight.
#
# Both steps must use the SAME --cellsam-block-px or the resume will not line up
# with the raw file; this script passes it to both and defaults to 512, which is the
# size that matched hand-labelled ground truth best (see run_stage_02.slurm).
#
# Override any of these from the environment:
#   ACCOUNT QOS GPU_PARTITION GPU_CPUS GPU_TIME GPU_MEM CPU_CPUS CPU_TIME CPU_MEM
#   CELLSAM_BLOCK METHODS CELLTUNE CELLTUNE_METHOD MERGE_NESTED RUN_NOTE
#
# The nested-label merge is off by default; it measured slightly worse than no
# merge against hand labels and costs hours. See run_stage_02.slurm.
set -euo pipefail

SAMPLES="${1:?usage: pipeline/run_stage_02_split.sh samples.csv [out_dir] [extra 02 args...]}"
OUT="${2:-results}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true

# Your SLURM account and QoS. Ask your cluster's support desk or your PI if you
# do not know them; on many clusters `sacctmgr show assoc user=$USER format=Account,QOS`
# prints them. Set them once in your shell instead of typing them every time:
#     export ACCOUNT=myaccount QOS=myqos
ACCOUNT="${ACCOUNT:?set ACCOUNT to your SLURM account, e.g. export ACCOUNT=myaccount}"
QOS="${QOS:?set QOS to your SLURM QoS, e.g. export QOS=myqos}"
GPU_PARTITION="${GPU_PARTITION:-hpg-rtx6000}"
GPU_CPUS="${GPU_CPUS:-8}";   GPU_TIME="${GPU_TIME:-08:00:00}";  GPU_MEM="${GPU_MEM:-100gb}"
CPU_CPUS="${CPU_CPUS:-14}";  CPU_TIME="${CPU_TIME:-08:00:00}";  CPU_MEM="${CPU_MEM:-100gb}"
BLOCK="${CELLSAM_BLOCK:-512}"

[ -f pipeline/sbatch_mail.sh ] || { echo "run this from the repository root" >&2; exit 2; }
[ -f "$SAMPLES" ] || { echo "no such samples sheet: $SAMPLES" >&2; exit 2; }

echo "stage 02, split into two jobs"
echo "  samples : $SAMPLES"
echo "  out     : $OUT"
echo "  block   : ${BLOCK} px (both steps; must match or --resume will not line up)"
echo

# Step 1: GPU, CellSAM only. CELLSAM_BLOCK is exported so run_stage_02.slurm
# uses the same value, and --stop-after-cellsam is passed through to the script.
# CELLTUNE=0 because step 1 returns before the export -- step 2 writes it.
J1=$(CELLSAM_BLOCK="$BLOCK" CELLTUNE=0 pipeline/sbatch_mail.sh \
      --account="$ACCOUNT" --qos="$QOS" \
      --partition="$GPU_PARTITION" --gres=gpu:1 \
      --cpus-per-task="$GPU_CPUS" --mem="$GPU_MEM" --time="$GPU_TIME" \
      --job-name=akoya_02_gpu \
      pipeline/run_stage_02.slurm "$SAMPLES" "$OUT" --stop-after-cellsam "$@" | tail -1)
echo "step 1 (GPU, CellSAM)      : job $J1"

# Step 2: CPU, everything after CellSAM. No --gres, so no idle-GPU policy.
J2=$(CELLSAM_BLOCK="$BLOCK" CELLSAM_DEVICE=cpu pipeline/sbatch_mail.sh \
      --account="$ACCOUNT" --qos="$QOS" \
      --cpus-per-task="$CPU_CPUS" --mem="$CPU_MEM" --time="$CPU_TIME" \
      --job-name=akoya_02_cpu \
      --dependency=afterok:"$J1" \
      pipeline/run_stage_02.slurm "$SAMPLES" "$OUT" --resume "$@" | tail -1)
echo "step 2 (CPU, merge+export) : job $J2   (starts only if $J1 succeeds)"
echo
echo "Watch:   squeue -u \$USER"
echo "Logs:    logs/akoya_02_gpu_${J1}.out  then  logs/akoya_02_cpu_${J2}.out"
echo "Result:  $OUT/<sample>/02_segment/celltune/<sample>_segmentation_labels.tif"
