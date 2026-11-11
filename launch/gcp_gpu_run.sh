#!/usr/bin/env bash
# Starts a Cloud Build job that builds the vLLM image and then creates a g2-standard-8 (1x L4) VM.
# Returns immediately; the VM runs the config, uploads results to GCS and stops itself.
# Run from Cloud Shell in the repository root after gcp_cpu_test.sh has set up the project:
#   PROJECT=<project-id> ./launch/gcp_gpu_run.sh                 # configs/smoke_l4.yaml
#   PROJECT=<project-id> CONFIG=configs/cuad.yaml SPOT=1 ./launch/gcp_gpu_run.sh
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT to your project id}
REGION=${REGION:-europe-west4}
ZONE=${ZONE:-europe-west4-a}
BUCKET=${BUCKET:-${PROJECT}-semops}
REPO=${REPO:-semops}
VM=${VM:-semops-l4}
CONFIG=${CONFIG:-configs/smoke_l4.yaml}
MAX_RUN=${MAX_RUN:-3h}   # hard limit: the VM is stopped after this even if the run hangs
TAG=${TAG:-$(git rev-parse --short HEAD)}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/semops-harness:${TAG}"
SPOT_ARGS=""
if [ "${SPOT:-0}" = "1" ]; then SPOT_ARGS="--provisioning-model=SPOT"; fi

gcloud config set project "$PROJECT"

# Cloud Build and the VM run as the default compute service account: Cloud Build creates the VM,
# and the VM writes a log entry and deletes itself when the run ends.
SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
for role in roles/compute.instanceAdmin.v1 roles/iam.serviceAccountUser roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
    --role="$role" --condition=None --quiet > /dev/null
done

gcloud builds submit --async --region="$REGION" --config=cloudbuild.gpu.yaml \
  --substitutions="_IMAGE=$IMAGE,_BUCKET=$BUCKET,_CONFIG=$CONFIG,_VM=$VM,_ZONE=$ZONE,_SPOT_ARGS=$SPOT_ARGS,_MAX_RUN=$MAX_RUN" .

EXP=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment'])")
cat <<MSG

Build submitted; you can close Cloud Shell. The VM deletes itself after uploading results,
and is stopped after $MAX_RUN in any case. To check:
  gcloud builds list --region=$REGION --limit=1             # SUCCESS = VM created
  gcloud compute instances list                             # empty = VM gone
  gcloud storage cat gs://$BUCKET/results/$EXP/runs.jsonl | wc -l
MSG
