#!/usr/bin/env bash
# GCE startup script. Settings come from instance metadata:
#   semops-image     image to run (Artifact Registry)
#   semops-bucket    GCS bucket name for data/ and results/
#   semops-config    config inside the image (default: configs/cuad.yaml)
#   semops-on-finish delete (default) | stop | none: what the VM does after uploading results
# Uses the GPU when one is present. Startup scripts run on every boot, so a restarted
# spot VM pulls earlier results back and the runner skips finished runs.
set -uo pipefail

md() { curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

IMAGE=$(md semops-image)
BUCKET=$(md semops-bucket)
CONFIG=$(md semops-config || echo configs/cuad.yaml)
ON_FINISH=$(md semops-on-finish || echo delete)
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
if lspci | grep -qi nvidia; then
  # Deep Learning VM images ship the driver; wait until it is loaded.
  for _ in $(seq 1 60); do nvidia-smi > /dev/null 2>&1 && break; sleep 10; done
  nvidia-smi
  if ! docker info 2> /dev/null | grep -qi nvidia; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
      gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -q && apt-get install -y -q nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
  fi
  GPU_ARGS=(--gpus all)
fi

LOG="$WORK/results/vm_$(date +%Y%m%dT%H%M%S).log"
docker run --rm "${GPU_ARGS[@]}" --ipc=host \
  -v "$WORK/hf-cache:/models" \
  -v "$WORK/results:/results" \
  -v "$WORK/data:/data:ro" \
  -e HF_TOKEN -e RUN_ENV=gcp -e IMAGE_DIGEST="$IMAGE" \
  "$IMAGE" --config "$CONFIG" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
echo "harness exited with status $STATUS"

gcloud storage rsync -r "$WORK/results" "$RESULTS"
# Picked up by the e-mail alert, if one is set up.
gcloud logging write semops "run finished: config=$CONFIG status=$STATUS results=$RESULTS" --severity=NOTICE || true

NAME=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
case "$ON_FINISH" in
  delete) gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || shutdown -h now ;;
  stop) shutdown -h now ;;
esac
