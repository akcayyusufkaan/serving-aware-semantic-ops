#!/usr/bin/env bash
# Builds the vLLM image on Cloud Build and starts a g2-standard-8 (1x L4) VM that runs one config.
# Run from Cloud Shell in the repository root after gcp_cpu_test.sh has set up the project:
#   PROJECT=<project-id> ./launch/gcp_gpu_run.sh                 # configs/smoke_l4.yaml
#   PROJECT=<project-id> CONFIG=configs/cuad.yaml ./launch/gcp_gpu_run.sh
# SPOT=1 uses a spot VM (cheaper, may be preempted; the run resumes on restart).
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT to your project id}
REGION=${REGION:-europe-west4}
ZONE=${ZONE:-europe-west4-a}
BUCKET=${BUCKET:-${PROJECT}-semops}
REPO=${REPO:-semops}
VM=${VM:-semops-l4}
CONFIG=${CONFIG:-configs/smoke_l4.yaml}
TAG=${TAG:-$(git rev-parse --short HEAD)}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/semops-harness:${TAG}"

gcloud config set project "$PROJECT"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  gcloud builds submit --region="$REGION" --config=cloudbuild.gpu.yaml --substitutions=_IMAGE="$IMAGE" .
fi

SPOT_ARGS=()
if [ "${SPOT:-0}" = "1" ]; then SPOT_ARGS=(--provisioning-model=SPOT --instance-termination-action=STOP); fi

gcloud compute instances create "$VM" --zone="$ZONE" \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE "${SPOT_ARGS[@]}" \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --scopes=cloud-platform \
  --metadata-from-file=startup-script=launch/gcp_vm_startup.sh \
  --metadata="semops-image=$IMAGE,semops-bucket=$BUCKET,semops-config=$CONFIG"

EXP=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['experiment'])")
cat <<MSG

VM $VM started with $CONFIG. Check progress and results:
  gcloud compute instances describe $VM --zone=$ZONE --format="value(status)"
  gcloud storage cat gs://$BUCKET/results/$EXP/runs.jsonl | wc -l
The VM stops itself when the run ends. Delete it afterwards:
  gcloud compute instances delete $VM --zone=$ZONE
MSG
