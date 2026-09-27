#!/bin/bash
# Stages 00 (ingest) and 01 (image QC) only, for every sample in a samples.csv.
#
#   sbatch --account=<ACCOUNT> --qos=<QOS> pipeline/run_stages_00_01.slurm samples.csv results
#
# Why 00-01 on their own: these two read the image lazily and finish in minutes
# even on a whole slide, so they are safe to run unattended and they produce the
# report that says whether the slide is usable at all. Stage 02 is the expensive
# one (CellSAM on CPU) and gets its own job once 01 says the slide is worth it.
#
# Group limits (<GROUP>, 2026-09-11): cpu=14, mem=112000M TOTAL across running
# jobs, so keep --mem at or under 100gb. gres/gpu=0.
#SBATCH --job-name=akoya_00_01
# Email the account's address when the job ends, fails, or can never start because a
# job it depends on failed (without INVALID_DEPEND that last case sends nothing).
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
SAMPLES="${1:?usage: sbatch pipeline/run_stages_00_01.slurm samples.csv [out_dir]}"
OUT="${2:-results}"
NOTE="${RUN_NOTE:-Stages 00-01 on HiPerGator $(date +%F)}"
mkdir -p logs

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check every input before anything expensive starts. There is no plain `akoya`
# environment on HiPerGator -- that name exists only on the Mac -- so the env is
# a parameter with a default that is actually present here.
# shellcheck disable=SC1091
source pipeline/preflight.sh
preflight "$SAMPLES" "${AKOYA_ENV:-akoya-cellsam}" || exit 2

conda activate "$AKOYA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/00_ingest.py   --samples "$SAMPLES" --out "$OUT" --run-note "$NOTE"
python scripts/01_image_qc.py --samples "$SAMPLES" --out "$OUT" --run-note "$NOTE"

echo "Reports: $OUT/<sample>/reports/00_ingest.html and 01_image_qc.html"
