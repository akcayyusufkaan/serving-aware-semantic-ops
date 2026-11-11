#!/usr/bin/env bash
# Runs one config on a g2-standard-8 (1x L4) VM. The config is uploaded to GCS, so changing it
# needs no image rebuild. Run from Cloud Shell in the repository root:
#   PROJECT=<project-id> CONFIG=configs/smoke_l4.yaml ./launch/gcp_gpu_run.sh
# By default this submits a Cloud Build job that builds the image and then creates the VM, and
# returns at once. SKIP_BUILD=1 reuses the image tagged with the current commit and creates the
# VM directly; adding RETRY=1 keeps retrying zones on Cloud Build for up to 3 hours. SPOT=1 uses a spot VM. The VM uploads results, deletes itself, and is stopped
# after MAX_RUN in any case.
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT to your project id}
REGION=${REGION:-europe-west4}            # Artifact Registry, bucket and Cloud Build
ZONES=${ZONES:-"us-central1-a us-central1-b us-central1-c europe-west4-a europe-west4-b europe-west4-c"}
BUCKET=${BUCKET:-${PROJECT}-semops}
REPO=${REPO:-semops}
VM=${VM:-semops-l4}
CONFIG=${CONFIG:-configs/smoke_l4.yaml}
MAX_RUN=${MAX_RUN:-3h}
TAG=${TAG:-$(git rev-parse --short HEAD)}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/semops-harness:${TAG}"
SPOT_ARGS=""
if [ "${SPOT:-0}" = "1" ]; then SPOT_ARGS="--provisioning-model=SPOT"; fi

gcloud config set project "$PROJECT"

# Cloud Build and the VM run as the default compute service account: Cloud Build creates the VM,
# and the VM reads the HF token, writes a log entry and deletes itself.
SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
for role in roles/compute.instanceAdmin.v1 roles/iam.serviceAccountUser roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
    --role="$role" --condition=None --quiet > /dev/null
done
if gcloud secrets describe hf-token > /dev/null 2>&1; then
  gcloud secrets add-iam-policy-binding hf-token --member="serviceAccount:$SA" \
    --role=roles/secretmanager.secretAccessor > /dev/null
fi

CONFIG_URL="gs://${BUCKET}/configs/$(basename "$CONFIG")"
gcloud storage cp "$CONFIG" "$CONFIG_URL"
EXP=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment'])")

if [ "${SKIP_BUILD:-0}" = "1" ] && [ "${RETRY:-0}" = "1" ]; then
  # Keep trying for up to 3 hours on Cloud Build; Cloud Shell can be closed.
  gcloud builds submit --async --region="$REGION" --config=cloudbuild.vm.yaml \
    --substitutions="^;^_IMAGE=$IMAGE;_BUCKET=$BUCKET;_CONFIG=$CONFIG_URL;_VM=$VM;_ZONES=$ZONES;_SPOT_ARGS=$SPOT_ARGS;_MAX_RUN=$MAX_RUN" .
  echo "Retrying in Cloud Build; you can close Cloud Shell. Status: gcloud builds list --region=$REGION --limit=1"
elif [ "${SKIP_BUILD:-0}" = "1" ]; then
  IMAGE="$IMAGE" BUCKET="$BUCKET" CONFIG="$CONFIG_URL" VM="$VM" ZONES="$ZONES" \
    MAX_RUN="$MAX_RUN" SPOT_ARGS="$SPOT_ARGS" bash launch/gcp_create_vm.sh
else
  gcloud builds submit --async --region="$REGION" --config=cloudbuild.gpu.yaml \
    --substitutions="^;^_IMAGE=$IMAGE;_BUCKET=$BUCKET;_CONFIG=$CONFIG_URL;_VM=$VM;_ZONES=$ZONES;_SPOT_ARGS=$SPOT_ARGS;_MAX_RUN=$MAX_RUN" .
  echo "Build submitted; you can close Cloud Shell. Build status: gcloud builds list --region=$REGION --limit=1"
fi

cat <<MSG

The VM deletes itself after uploading results and is stopped after $MAX_RUN in any case. To check:
  gcloud compute instances list                                 # empty = VM gone
  gcloud storage cat gs://$BUCKET/results/$EXP/runs.jsonl | wc -l
MSG
