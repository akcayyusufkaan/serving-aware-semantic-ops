#!/usr/bin/env bash
# Downloads CUAD, builds the workload files with scripts/prepare_cuad.py and uploads them to
# gs://<bucket>/data/, which the VM syncs to /data. Run from Cloud Shell in the repository root:
#   PROJECT=<project-id> ./launch/gcp_upload_cuad.sh            # N=25 contracts by default
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT}
BUCKET=${BUCKET:-${PROJECT}-semops}
N=${N:-25}
TMP=$(mktemp -d)
curl -sSL -o "$TMP/data.zip" https://github.com/TheAtticusProject/cuad/raw/main/data.zip
unzip -q -o "$TMP/data.zip" -d "$TMP"
python3 scripts/prepare_cuad.py --cuad-test "$TMP/test.json" --n "$N" --out-dir data
gcloud storage cp "data/cuad_${N}.jsonl" "data/cuad_${N}_labels.json" "gs://${BUCKET}/data/"
rm -rf "$TMP"
