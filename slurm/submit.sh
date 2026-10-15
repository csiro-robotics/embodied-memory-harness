#!/bin/bash
# Submit one arm of the benchmark to SLURM.
#
# Usage:  slurm/submit.sh <arm> [extra sbatch args]
#         arm is one of: no_memory kg_hybrid macla wiki_v3 (reads configs/arm_<arm>.yaml)
# Example: SMOKE=3 slurm/submit.sh macla
#
# Site-specific settings (paths, account, partition) come from slurm/site.env, which is
# git-ignored. Create it once from slurm/site.env.example.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$(cd "$HERE/.." && pwd)
SITE_ENV=$HERE/site.env
if [ ! -f "$SITE_ENV" ]; then
    echo "ERROR: $SITE_ENV not found. Copy slurm/site.env.example to slurm/site.env and fill it in."
    exit 1
fi
# shellcheck source=/dev/null
source "$SITE_ENV"

ARM=${1:?usage: slurm/submit.sh <arm> [extra sbatch args]}
shift
CONFIG=$CODE/configs/arm_$ARM.yaml
[ -f "$CONFIG" ] || { echo "ERROR: no config at $CONFIG"; exit 1; }

mkdir -p "$RUNS" "$ALFWORLD_STAGE" 2>/dev/null || true
# SLURM exits with code 53 and writes no log at all if it cannot open the log file, so
# check write access here, where the error can still be read.
for d in "$RUNS" "$ALFWORLD_STAGE"; do
    [ -w "$d" ] || { echo "ERROR: $d is not writable by $USER (set RUNS / ALFWORLD_STAGE in site.env)"; exit 1; }
done
for d in "$ALFWORLD_SRC" "${TRAIN_TASKS:-$TRAIN500_TASKS}" "$OLLAMA_MODELS_DIR"; do
    [ -r "$d" ] || { echo "ERROR: $d is not readable"; exit 1; }
done
# shellcheck disable=SC2086
sbatch ${SBATCH_ARGS:-} \
    --job-name="arm-$ARM" \
    --output="$RUNS/%x-%j.out" \
    --export=ALL,CODE="$CODE",CONFIG="$CONFIG",SITE_ENV="$SITE_ENV" \
    "$@" "$HERE/arm.slurm"
