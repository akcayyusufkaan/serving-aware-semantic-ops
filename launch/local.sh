#!/usr/bin/env bash
# Any GPU machine with Docker and the NVIDIA Container Toolkit (e.g. a WSL2 laptop).
# Usage: ./launch/local.sh configs/smoke.yaml
set -euo pipefail
CONFIG=${1:-configs/smoke.yaml}
IMAGE=semops-harness:dev

docker build -t "$IMAGE" .

mkdir -p "$HOME/hf-cache" results data
# Configs are mounted so editing them does not require a rebuild.
docker run --rm --gpus all --ipc=host \
  -v "$HOME/hf-cache:/models" \
  -v "$PWD/results:/results" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/configs:/app/configs:ro" \
  -e HF_TOKEN -e RUN_ENV=laptop \
  "$IMAGE" --config "$CONFIG"
