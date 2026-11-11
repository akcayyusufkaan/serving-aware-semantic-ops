#!/usr/bin/env bash
# Creates the L4 VM in the first zone that has capacity. Used by gcp_gpu_run.sh and Cloud Build.
# Env: IMAGE, BUCKET, CONFIG (gs:// URL or path in the image), VM, ZONES, MAX_RUN, SPOT_ARGS,
# ROUNDS (passes over ZONES, default 1), WAIT (seconds between passes, default 600) and MACHINE.
# Each zone's region needs Private Google Access and Cloud NAT (launch/gcp_network.sh).
set -uo pipefail
ROUNDS=${ROUNDS:-1}
# 48 GB RAM: switching between two 8B models briefly holds both models' weights in CPU memory
# (~15 GB each), which does not fit next to everything else on a 32 GB g2-standard-8.
MACHINE=${MACHINE:-g2-standard-12}
WAIT=${WAIT:-600}
for ROUND in $(seq 1 "$ROUNDS"); do
for Z in $ZONES; do
  echo "[$ROUND/$ROUNDS] trying $Z"
  # shellcheck disable=SC2086
  if gcloud compute instances create "$VM" --zone="$Z" \
      --machine-type="$MACHINE" --accelerator=type=nvidia-l4,count=1 \
      --maintenance-policy=TERMINATE $SPOT_ARGS \
      --max-run-duration="$MAX_RUN" --instance-termination-action=STOP \
      --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
      --boot-disk-size=200GB --scopes=cloud-platform --no-address \
      --metadata-from-file=startup-script=launch/gcp_vm_startup.sh \
      --metadata="semops-image=$IMAGE,semops-bucket=$BUCKET,semops-config=$CONFIG" 2> /tmp/vm_err; then
    echo "VM $VM created in $Z"
    exit 0
  fi
  grep -oE "STOCKOUT|QUOTA_EXCEEDED|message: .{0,160}" /tmp/vm_err | head -1
done
[ "$ROUND" -lt "$ROUNDS" ] && sleep "$WAIT"
done
echo "no zone had capacity"
exit 1
