#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server. Everything is overridable by env vars.
#   ./serve.sh
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct-AWQ ./serve.sh                 # quantized
#   EXTRA_ARGS="--tensor-parallel-size 2" ./serve.sh                # 2 GPUs
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-half}"            # T4 has no bfloat16 support -> use half (fp16)
MAX_LEN="${MAX_LEN:-4096}"        # max prompt+output tokens per request
GPU_UTIL="${GPU_UTIL:-0.85}"      # fraction of GPU memory vLLM may use (weights + KV cache)
MAX_SEQS="${MAX_SEQS:-64}"        # max requests batched together

exec vllm serve "$MODEL" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  --port "$PORT" \
  ${EXTRA_ARGS:-}
