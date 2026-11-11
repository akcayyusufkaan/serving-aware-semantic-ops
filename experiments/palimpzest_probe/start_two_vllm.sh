#!/usr/bin/env bash
# Starts two small vLLM servers that stay awake together on a 4 GB GPU (WSL2).
# Palimpzest sends requests to both models interleaved, so both must be up.
# Usage: ./start_two_vllm.sh   |   ./start_two_vllm.sh stop
set -euo pipefail
UTIL=${UTIL:-0.35}   # try 0.30 if startup reports insufficient memory
LOGDIR=${LOGDIR:-./vllm_logs}
mkdir -p "$LOGDIR"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "vllm serve" || true
  exit 0
fi

export VLLM_USE_FLASHINFER_SAMPLER=0

start() {  # $1 = Hugging Face id, $2 = port
  vllm serve "$1" --port "$2" --gpu-memory-utilization "$UTIL" \
    --max-model-len 2048 --enforce-eager > "$LOGDIR/vllm_$2.log" 2>&1 &
  echo "started $1 on port $2 (log: $LOGDIR/vllm_$2.log)"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$2/health" > /dev/null; then echo "ready: $2"; return 0; fi
    sleep 2
  done
  echo "port $2 not ready after 6 minutes, check the log"; exit 1
}

# Sequential, so the second server sees the memory the first one took.
start Qwen/Qwen2.5-0.5B-Instruct 8001
start HuggingFaceTB/SmolLM2-360M-Instruct 8002
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
