#!/bin/bash
# HealthBench local vLLM job
#
# Workflow:
#   1. Start local vLLM server.
#   2. Wait until the server is ready.
#   3. Run main.py.
#   4. Stop vLLM automatically when finished.
#
# Pick the task and options here, then submit:
#   sbatch slurm/run_vllm.sbatch
#
# Examples:
#   TASK_ARGS="--task train_capo_prompt --dev-size 300" sbatch slurm/run_vllm.sbatch
#   MODEL_ALIAS=qwen2_5_1_5b TASK_ARGS="--task generate_scores --grader qwen2_5_1_5b" sbatch slurm/run_vllm.sbatch
#   TASK_ARGS="--task generate_responses --limit 5" sbatch slurm/run_vllm.sbatch

#SBATCH --job-name=hb_vllm
#SBATCH --output=logs/hb_vllm_%j.out
#SBATCH --error=logs/hb_vllm_%j.err
#SBATCH --partition=universe
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Exclude nodes with known CUDA issues until an admin repairs them.
# helios/atlas: /dev/nvidia-uvm returns EIO, so cuInit fails with CUDA error 999.
# prometheus: excluded by course/cluster convention.
#SBATCH --exclude=prometheus,helios,atlas

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# shellcheck disable=SC1091
source slurm/vllm_env.sh
setup_vllm_env
check_cuda

MODEL_ALIAS="${MODEL_ALIAS:-gemma_2_9b_it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

MODEL_PATH=$(uv run python - <<PY
from src.config import resolve_model
print(resolve_model("$MODEL_ALIAS"))
PY
)

echo "Serving model alias: $MODEL_ALIAS"
echo "Resolved model path: $MODEL_PATH"
nvidia-smi

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

stdbuf -o0 -e0 uv run vllm serve "$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    --served-model-name "$MODEL_ALIAS" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager &
VLLM_PID=$!

echo "Waiting for vLLM..."
VLLM_READY=0
for _ in $(seq 1 120); do
    if curl -s "http://localhost:8000/v1/models" > /dev/null; then
        VLLM_READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited before the server became ready."
        exit 1
    fi
    sleep 5
done

if [[ "$VLLM_READY" -ne 1 ]]; then
    echo "ERROR: vLLM did not become ready within 10 minutes."
    exit 1
fi

echo "vLLM is ready. Starting main.py..."
export MODEL_ALIAS
export VLLM_SERVER_URL="http://localhost:8000/v1"
export LLM_API_BASE_URL="$VLLM_SERVER_URL"

TASK_ARGS="${TASK_ARGS:---task generate_responses --limit 5 --max-workers 1}"

# shellcheck disable=SC2086
/usr/bin/time -v uv run python main.py $TASK_ARGS

echo "HealthBench vLLM job finished."
