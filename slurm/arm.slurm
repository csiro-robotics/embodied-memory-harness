#!/bin/bash
#SBATCH --job-name=arm
#SBATCH --time=2-00:00:00         # 48h: ~10-14h train (500 eps) + ~2h eval
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
# Site-specific options (account, partition, node exclusions, log path) are NOT set here.
# Submit through slurm/submit.sh, which reads them from slurm/site.env:
#
#   cp slurm/site.env.example slurm/site.env      # once; fill in your paths
#   slurm/submit.sh wiki_v3                        # one arm
#   for ARM in no_memory kg_hybrid macla wiki_v3; do slurm/submit.sh $ARM; done
#
# One memory arm per submission. The ONLY thing that varies between arms is the config
# (its memory: block). The seed gives every arm the SAME deterministic order over the
# 500-task train set (see src/alfworld_env.select_and_order_games).
#
# Optional environment overrides (pass them in front of submit.sh, e.g. SMOKE=3 slurm/submit.sh macla):
#   SMOKE=N          cap both phases at N episodes (shakedown, never a result)
#   TRAIN_EPS, EVAL_EPS, EVAL_SPLIT   iteration-round overrides (each recorded as a deviation)
#   TRAIN_TASKS=<dir>  training game folders (default TRAIN500_TASKS from site.env)
#   EVAL_SEEN=N      also evaluate on N valid_seen tasks (140 = all) into eval_seen/
#   The 1500-game run: TRAIN_TASKS=<train1500_v1> TRAIN_EPS=1500 EVAL_SEEN=140, and pass
#   --time=4-00:00:00 for kg_hybrid (~68h), e.g.
#   TRAIN_TASKS=... TRAIN_EPS=1500 EVAL_SEEN=140 slurm/submit.sh kg_hybrid --time=4-00:00:00
#   SKIP_TRAIN=1 TRAIN_OUT=<run>/train   evaluate an existing trained store
#   SKIP_EVAL=1      train only
#   VERBOSE=1        DEBUG logging
#   REPLICATE=N      replicate number (DEC-15): seed 42 + N - 1, i.e. a different training order

set -euo pipefail

# ---- Site settings -----------------------------------------------------------
: "${SITE_ENV:?SITE_ENV is not set; submit with slurm/submit.sh}"
: "${CODE:?CODE is not set; submit with slurm/submit.sh}"
: "${CONFIG:?CONFIG is not set; submit with slurm/submit.sh}"
# shellcheck source=/dev/null
source "$SITE_ENV"
: "${RUNS:?set RUNS in site.env}"
: "${ALFWORLD_SRC:?set ALFWORLD_SRC in site.env}"
: "${TRAIN500_TASKS:?set TRAIN500_TASKS in site.env}"
: "${ALFWORLD_STAGE:?set ALFWORLD_STAGE in site.env}"
: "${OLLAMA_MODELS_DIR:?set OLLAMA_MODELS_DIR in site.env}"

OUT=$RUNS/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
mkdir -p "$OUT"

echo "===== JOB START (unified arm) ====="
echo "Host:   $(hostname)"
echo "JobID:  $SLURM_JOB_ID"
echo "Job:    $SLURM_JOB_NAME"
echo "Time:   $(date)"
echo "Config: $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for _m in ${SITE_MODULES:-}; do module load "$_m"; done

# ---- ALFWorld data dir (train500_v2 as the train split) ---------------------
# TRAIN_TASKS, set when queuing, swaps in another training set (e.g. train1500_v1).
TRAIN_TASKS=${TRAIN_TASKS:-$TRAIN500_TASKS}
# One staging folder per job: parallel arms used to share one and raced while building it
# (job 60486809 died on "cp: File exists"). The folder only holds symlinks and one config.
STAGE=$ALFWORLD_STAGE/job-$SLURM_JOB_ID
echo "--- Configuring ALFWorld data dir at $STAGE ---"
mkdir -p "$STAGE/json_2.1.1"
ln -sfn "$TRAIN_TASKS"                           "$STAGE/json_2.1.1/train"
ln -sfn "$ALFWORLD_SRC/json_2.1.1/valid_unseen"  "$STAGE/json_2.1.1/valid_unseen"
ln -sfn "$ALFWORLD_SRC/json_2.1.1/valid_seen"    "$STAGE/json_2.1.1/valid_seen"
ln -sfn "$ALFWORLD_SRC/json_2.1.1/valid_train"   "$STAGE/json_2.1.1/valid_train" 2>/dev/null || true
ln -sfn "$ALFWORLD_SRC/logic"                    "$STAGE/logic" 2>/dev/null || true

cp "$ALFWORLD_SRC/base_config.yaml" "$STAGE/base_config.yaml"
# os.walk does not follow our symlink at the json_2.1.1 level; point data_path directly
# at json_2.1.1/train so the train games are discovered.
sed -i 's|data_path: "${ALFWORLD_DATA}/json_2.1.1"|data_path: "${ALFWORLD_DATA}/json_2.1.1/train"|' \
    "$STAGE/base_config.yaml"
echo "  json_2.1.1/train -> $(readlink "$STAGE/json_2.1.1/train")"
# Count game files, not folders: train500_v2 holds 500 games in 438 task folders.
echo "  Training games: $(find -L "$TRAIN_TASKS" -name game.tw-pddl | wc -l)"
export ALFWORLD_DATA=$STAGE
trap 'rm -rf "$STAGE"' EXIT   # symlinks and one config only; replaced below once ollama starts

# ---- Python environment --------------------------------------------------------
PYBIN=python
if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    # nounset off around activation: some env activate.d hooks read unset variables.
    set +u
    conda activate "$CONDA_ENV"
    set -u
    echo "Conda env: $CONDA_ENV"
fi
echo "Python: $(command -v $PYBIN)"
$PYBIN -c "import alfworld, openai, yaml, numpy" \
    || { echo "ERROR: required python deps missing in this environment"; exit 1; }

# ---- Ollama ------------------------------------------------------------------
export OLLAMA_MODELS=$OLLAMA_MODELS_DIR
mkdir -p "$OLLAMA_MODELS"
# Per-job port: SLURM can pack several arms onto one node, and a fixed port would make
# the second job silently share the first job's server. Start from the job id and walk
# up to the first free port.
_free_port() {
    local p=$1
    while ss -ltn 2>/dev/null | grep -q ":${p} "; do p=$((p + 1)); done
    echo "$p"
}
# Each job takes a pair of ports (action, memory) at an even offset from its job id, so
# consecutive job ids never pick overlapping pairs on a shared node (H19: jobs 60491700 and
# 60491701 collided when the pairs were one port apart, and one job's calls reached the
# other's server). The check after start-up stops the job if either server did not bind.
OLLAMA_PORT=$(_free_port $((11434 + (SLURM_JOB_ID % 2500) * 2)))
# Two servers per job, same GPU, same settings (H19). The action server answers only the
# action model's requests; the memory server answers every memory-side call (the wiki's
# reads and consolidation, the KG's extraction and embeddings, MACLA's refinement).
# Ollama reuses cached state from the previous request, so a memory call between two
# action calls changed the action model's reply to a byte-identical prompt. With its own
# server, the action model sees the same request sequence whenever the arms send the
# same prompts. Every arm gets both servers, so no arm runs in a different setup.
OLLAMA_MEMORY_PORT=$(_free_port $((OLLAMA_PORT + 1)))
echo "Ollama ports: action $OLLAMA_PORT, memory $OLLAMA_MEMORY_PORT"

# The `ollama` CLI (model check, run_env.txt) talks to the action server. The module runs
# Ollama in an Apptainer image that sets its own OLLAMA_HOST (port 11434); only the
# APPTAINERENV_ form overrides it inside the container. Each `serve` below sets its own.
export OLLAMA_HOST=127.0.0.1:${OLLAMA_PORT}
export APPTAINERENV_OLLAMA_HOST=127.0.0.1:${OLLAMA_PORT}
# runner.build_llm honours these over the config's base_url.
export OLLAMA_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}/v1"
export OLLAMA_MEMORY_BASE_URL="http://127.0.0.1:${OLLAMA_MEMORY_PORT}/v1"
export APPTAINERENV_OLLAMA_KEEP_ALIVE=-1
export OLLAMA_KEEP_ALIVE=-1
# One context window for every request of every arm, set on the server (H1). The
# OpenAI-compatible /v1 endpoint ignores a num_ctx sent with a request, so Run 10 ran every
# request at Ollama's default of 4096 tokens per slot with 2 parallel slots, whatever the
# config said. run_pipeline.py checks that this value equals the config's num_ctx.
export OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-32768}
export APPTAINERENV_OLLAMA_CONTEXT_LENGTH=$OLLAMA_CONTEXT_LENGTH
# One slot, so the whole window belongs to the one request the pipeline sends at a time.
export OLLAMA_NUM_PARALLEL=1
export APPTAINERENV_OLLAMA_NUM_PARALLEL=1
# Fixed hash seed (H5): set and dict-key iteration order is then the same in every job.
export PYTHONHASHSEED=0
# Hugging Face models (MACLA's MiniLM, pinned revision, P11) load from the local cache only,
# so a job never contacts the Hub or picks up a newer copy mid-study.
export HF_HUB_OFFLINE=1
# An arm that also serves an embedding model (kg_hybrid) needs both models resident,
# or ollama may evict the chat model for the embedder on every retrieval.
if grep -q "embed_model:" "$CONFIG"; then
    export APPTAINERENV_OLLAMA_MAX_LOADED_MODELS=2
    export OLLAMA_MAX_LOADED_MODELS=2
    echo "OLLAMA_MAX_LOADED_MODELS=2 (config declares an embed_model)"
fi

# Stagger starts: parallel arms share one model directory.
sleep $((RANDOM % 30))

# The server logs go into the run folder so they are kept with the results (H12).
OLLAMA_HOST=127.0.0.1:${OLLAMA_PORT} APPTAINERENV_OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT} \
    ollama serve > "$OUT/ollama_server.log" 2>&1 &
OLLAMA_PID=$!
OLLAMA_HOST=127.0.0.1:${OLLAMA_MEMORY_PORT} APPTAINERENV_OLLAMA_HOST=0.0.0.0:${OLLAMA_MEMORY_PORT} \
    ollama serve > "$OUT/ollama_memory_server.log" 2>&1 &
OLLAMA_MEMORY_PID=$!
trap 'kill $OLLAMA_PID $OLLAMA_MEMORY_PID 2>/dev/null || true; rm -rf "$STAGE"' EXIT
echo "Ollama PIDs: action $OLLAMA_PID, memory $OLLAMA_MEMORY_PID (logs in $OUT)"
sleep 5
for pair in "$OLLAMA_PID:ollama_server.log" "$OLLAMA_MEMORY_PID:ollama_memory_server.log"; do
    if ! kill -0 "${pair%%:*}" 2>/dev/null || grep -q "address already in use" "$OUT/${pair#*:}"; then
        echo "ERROR: an Ollama server did not start (see $OUT/${pair#*:}); another job may hold its port"
        exit 1
    fi
done

for port in $OLLAMA_PORT $OLLAMA_MEMORY_PORT; do
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:$port/api/tags" > /dev/null 2>&1; then
            echo "Ollama on port $port ready after ${i}s"; break
        fi
        sleep 1
    done
done

# ---- Model cache check ---------------------------------------------------------
# Retried because parallel servers scanning one model directory can make `ollama list`
# briefly return an incomplete set.
MODELS="qwen3:14b"
if grep -q "embed_model:" "$CONFIG"; then
    EMBED=$(grep -m1 'embed_model:' "$CONFIG" | sed 's/.*embed_model:[[:space:]]*//' | tr -d '"' | awk '{print $1}')
    [ -n "$EMBED" ] && MODELS="$MODELS $EMBED"
fi
echo "Models required: $MODELS"
for MODEL in $MODELS; do
    FOUND=0
    for attempt in $(seq 1 12); do
        if ollama list 2>/dev/null | grep -qF "$MODEL"; then FOUND=1; break; fi
        sleep 5
    done
    if [ "$FOUND" -ne 1 ]; then
        echo "ERROR: model $MODEL not found in $OLLAMA_MODELS"
        echo "  Run on a login node: OLLAMA_MODELS=$OLLAMA_MODELS ollama pull $MODEL"
        exit 1
    fi
    echo "Model $MODEL: cached OK"
done

# ---- Run environment record (R3) --------------------------------------------------
# Everything needed to say later exactly what this run used.
{
    echo "job_id: $SLURM_JOB_ID"
    echo "job_name: $SLURM_JOB_NAME"
    echo "host: $(hostname)"
    echo "start: $(date -Is)"
    echo "config: $CONFIG"
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
    echo "git_commit: $(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty: $(git -C "$CODE" status --porcelain 2>/dev/null | wc -l) changed files"
    echo "ollama_version: $(ollama --version 2>&1 | tail -1)"
    echo "ollama_context_length: $OLLAMA_CONTEXT_LENGTH"
    echo "ollama_num_parallel: $OLLAMA_NUM_PARALLEL"
    echo "pythonhashseed: $PYTHONHASHSEED"
    echo "python: $($PYBIN --version 2>&1)"
    echo "alfworld: $($PYBIN -c "import importlib.metadata as m; print(m.version('alfworld'))")"
    echo "--- ollama list (model digests) ---"
    ollama list
    echo "--- pip freeze ---"
    $PYBIN -m pip freeze 2>/dev/null
} > "$OUT/run_env.txt"
cp "$CONFIG" "$OUT/arm_config.yaml"
echo "Run environment recorded in $OUT/run_env.txt"

# ---- Pre-warm the model ------------------------------------------------------------
# Loads the model on both servers at the context length, before the first timed episode.
echo "Pre-warming model (context length $OLLAMA_CONTEXT_LENGTH)..."
for port in $OLLAMA_PORT $OLLAMA_MEMORY_PORT; do
    curl -sf "http://127.0.0.1:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3:14b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
        --max-time 120 > /dev/null \
        && echo "Model warm-up done on port $port." || echo "Warm-up timed out on $port (continuing)."
done

# ---- Flags ---------------------------------------------------------------------------
SKIP_TRAIN_FLAG=""
if [ -n "${SKIP_TRAIN:-}" ] && [ -n "${TRAIN_OUT:-}" ]; then
    SKIP_TRAIN_FLAG="--skip-train"
    OUT=$(dirname "$TRAIN_OUT")
    echo "  Skipping train — reusing memory from $TRAIN_OUT"
fi
SKIP_EVAL_FLAG=""
[ -n "${SKIP_EVAL:-}" ] && SKIP_EVAL_FLAG="--skip-eval"
VERBOSE_FLAG=""
[ "${VERBOSE:-0}" = "1" ] && VERBOSE_FLAG="--verbose"
SMOKE_FLAG=""
[ -n "${SMOKE:-}" ] && SMOKE_FLAG="--smoke ${SMOKE}"
ROUND_FLAGS=""
[ -n "${TRAIN_EPS:-}" ]  && ROUND_FLAGS="$ROUND_FLAGS --train-episodes ${TRAIN_EPS}"
[ -n "${EVAL_EPS:-}" ]   && ROUND_FLAGS="$ROUND_FLAGS --eval-episodes ${EVAL_EPS}"
[ -n "${EVAL_SPLIT:-}" ] && ROUND_FLAGS="$ROUND_FLAGS --eval-split ${EVAL_SPLIT}"
[ -n "${EVAL_SEEN:-}" ]  && ROUND_FLAGS="$ROUND_FLAGS --eval-seen ${EVAL_SEEN}"
# DEC-15: replicate N trains in its own game order (seed 42 + N - 1).
REPLICATE_FLAG="--replicate ${REPLICATE:-1}"
echo "replicate: ${REPLICATE:-1}" >> "$OUT/run_env.txt"
echo "train_tasks: ${TRAIN_TASKS}" >> "$OUT/run_env.txt"

echo "===== RUNNING UNIFIED ARM ====="
echo "  Config:  $CONFIG"
echo "  Output:  $OUT"
[ -n "${SMOKE:-}" ] && echo "  MODE:    SMOKE (${SMOKE} episodes/phase) — shakedown only, not a result"

$PYBIN "$CODE/run_pipeline.py" \
    --config "$CONFIG" \
    --out    "$OUT" \
    $SKIP_TRAIN_FLAG \
    $SKIP_EVAL_FLAG \
    $SMOKE_FLAG \
    $ROUND_FLAGS \
    $REPLICATE_FLAG \
    $VERBOSE_FLAG

echo "===== JOB DONE ====="
echo "end: $(date -Is)" >> "$OUT/run_env.txt"

# H1 check: the context the server actually used, and how many prompts it cut to fit.
# check_run.py fails a run whose truncated_prompts is not 0.
{
    echo "requested_context_length: $OLLAMA_CONTEXT_LENGTH"
    echo "truncated_prompts: $(cat "$OUT"/ollama_server.log "$OUT"/ollama_memory_server.log \
        | grep -c 'truncating input prompt' || true)"
    for log in ollama_server.log ollama_memory_server.log; do
        echo "--- context settings in $log (count, setting) ---"
        grep -oE "n_ctx_per_seq *= *[0-9]+|n_ctx *= *[0-9]+|--ctx-size [0-9]+|--parallel [0-9]+" \
            "$OUT/$log" | sort | uniq -c || true
    done
} > "$OUT/ollama_context_check.txt"
cat "$OUT/ollama_context_check.txt"
echo "Report: $OUT/report.json"
