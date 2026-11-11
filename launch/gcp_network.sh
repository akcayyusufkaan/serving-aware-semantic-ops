#!/usr/bin/env bash
# One-time network setup for VMs without external IPs: Google APIs through Private Google Access,
# everything else (apt, Hugging Face, NVIDIA repos) through Cloud NAT.
#   PROJECT=<project-id> ./launch/gcp_network.sh
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT}
REGION=${REGION:-europe-west4}
gcloud config set project "$PROJECT"
gcloud compute networks subnets update default --region="$REGION" --enable-private-ip-google-access
gcloud compute routers describe semops-router --region="$REGION" > /dev/null 2>&1 ||
  gcloud compute routers create semops-router --network=default --region="$REGION"
gcloud compute routers nats describe semops-nat --router=semops-router --region="$REGION" > /dev/null 2>&1 ||
  gcloud compute routers nats create semops-nat --router=semops-router --region="$REGION" \
    --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges
