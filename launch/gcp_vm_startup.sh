#!/usr/bin/env bash
# GCE startup script. Settings come from instance metadata:
#   semops-image     image to run (Artifact Registry)
#   semops-bucket    GCS bucket name for data/ and results/
#   semops-config    config inside the image (default: configs/cuad.yaml)
#   semops-shutdown  "false" keeps the VM running after the run (default: true)
# Uses the GPU when one is present. Startup scripts run on every boot, so a restarted
# spot VM pulls earlier results back and the runner skips finished runs.
set -uo pipefail

md() { curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

IMAGE=$(md semops-image)
BUCKET=$(md semops-bucket)
CONFIG=$(md semops-config || echo configs/cuad.yaml)
SHUTDOWN=$(md semops-shutdown || echo true)
WORK=/opt/semops
RESULTS="gs://${BUCKET}/results"

mkdir -p "$WORK/hf-cache" "$WORK/results" "$WORK/data"

if ! command -v docker > /dev/null; then
  apt-get update -q && apt-get install -y -q docker.io
fi
gcloud auth configure-docker "${IMAGE%%/*}" --quiet
docker pull "$IMAGE"

gcloud storage rsync -r "$RESULTS" "$WORK/results" 2> /dev/null || true
gcloud storage rsync -r "gs://${BUCKET}/data" "$WORK/data" 2> /dev/null || true

HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token 2> /dev/null || true)
export HF_TOKEN

GPU_ARGS=()
if command -v nvidia-smi > /dev/null; then GPU_ARGS=(--gpus all); fi

docker run --rm "${GPU_ARGS[@]}" --ipc=host \
  -v "$WORK/hf-cache:/models" \
  -v "$WORK/results:/results" \
  -v "$WORK/data:/data:ro" \
  -e HF_TOKEN -e RUN_ENV=gcp -e IMAGE_DIGEST="$IMAGE" \
  "$IMAGE" --config "$CONFIG"
echo "harness exited with status $?"

gcloud storage rsync -r "$WORK/results" "$RESULTS"

if [ "$SHUTDOWN" = "true" ]; then shutdown -h now; fi
