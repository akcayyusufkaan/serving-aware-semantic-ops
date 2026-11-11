#!/usr/bin/env bash
# Sets up the GCP resources and starts a CPU-only test VM running the stub vLLM image.
# Run from Cloud Shell in the repository root:  PROJECT=<project-id> ./launch/gcp_cpu_test.sh
# Exercises the same path the GPU run will use: Artifact Registry, Secret Manager,
# the startup script, results in GCS and the automatic shutdown.
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT to your project id}
REGION=${REGION:-europe-west4}
ZONE=${ZONE:-europe-west4-a}
BUCKET=${BUCKET:-${PROJECT}-semops}
REPO=${REPO:-semops}
VM=${VM:-semops-cpu-test}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/semops-harness:cpu-test"

gcloud config set project "$PROJECT"
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com cloudbuild.googleapis.com

gcloud artifacts repositories describe "$REPO" --location="$REGION" > /dev/null 2>&1 ||
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION"
gcloud storage buckets describe "gs://$BUCKET" > /dev/null 2>&1 ||
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access

# Built on Cloud Build rather than locally: Cloud Shell's network can be slow for image pulls.
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  gcloud builds submit --region="$REGION" --config=cloudbuild.cpu.yaml --substitutions=_IMAGE="$IMAGE" .
fi

# The VM runs as the default compute service account.
SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
  --role=roles/artifactregistry.reader --condition=None --quiet > /dev/null
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA" \
  --role=roles/storage.objectAdmin > /dev/null
if gcloud secrets describe hf-token > /dev/null 2>&1; then
  gcloud secrets add-iam-policy-binding hf-token --member="serviceAccount:$SA" \
    --role=roles/secretmanager.secretAccessor > /dev/null
fi

gcloud compute instances create "$VM" --zone="$ZONE" \
  --machine-type=e2-standard-2 --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=20GB --scopes=cloud-platform \
  --metadata-from-file=startup-script=launch/gcp_vm_startup.sh \
  --metadata="semops-image=$IMAGE,semops-bucket=$BUCKET,semops-config=configs/smoke.yaml"

cat <<MSG

VM $VM started. Follow the startup script:
  gcloud compute instances get-serial-port-output $VM --zone=$ZONE | grep startup-script
Results (8 lines expected):
  gcloud storage cat gs://$BUCKET/results/smoke/runs.jsonl
The VM stops itself when the run ends. Delete it afterwards:
  gcloud compute instances delete $VM --zone=$ZONE
MSG
