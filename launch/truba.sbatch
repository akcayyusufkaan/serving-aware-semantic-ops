#!/bin/bash
# TRUBA (Slurm + Apptainer). Convert the image first:
#   apptainer pull semops-harness.sif docker://<registry>/semops-harness:<tag>
# Compute nodes may have no internet: download models to ~/hf-cache on a login node.
# TODO: check partition, account and GPU syntax against the TRUBA docs.
#SBATCH -J semops
#SBATCH -p <gpu-partition>
#SBATCH -A <account>
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o logs/%j.out

mkdir -p results data logs
export HF_HUB_OFFLINE=1
# Resubmitting after a time limit resumes from the last finished run.
apptainer exec --nv \
  --bind "$HOME/hf-cache:/models" \
  --bind "$PWD/results:/results" \
  --bind "$PWD/data:/data" \
  --bind "$PWD/configs:/app/configs" \
  --env RUN_ENV=truba \
  semops-harness.sif \
  bash -c "cd /app && python3 -m harness.runner --config configs/cuad.yaml"
