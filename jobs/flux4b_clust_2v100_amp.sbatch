#!/usr/bin/env bash
# One worker trains; GPU1 holds the frozen text encoder. Validation stays serial.
# Submit: sbatch jobs/flux4b_clust_2v100_amp.sbatch
# Resume: sbatch jobs/flux4b_clust_2v100_amp.sbatch --resume
# Inspect: squeue -j JOBID; cancel: scancel JOBID
#SBATCH --job-name=rsrch-4b-v100-amp
#SBATCH --account=proj_1892
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=2-02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/nasilaev/rsrch_new
#SBATCH --output=/home/nasilaev/rsrch_new/logs/clust/%x-%j.out
#SBATCH --error=/home/nasilaev/rsrch_new/logs/clust/%x-%j.err
set -euo pipefail
# Site RealMemory=1: memory directives are intentionally omitted.
export BA_CLUST_CONFIG=configs/clust/flux4b_2v100_amp.yaml
exec bash scripts/run_clust_v100.sh \
  --run runs/flux4b_clust_2v100_amp_qkvo_r128_20k_20261003 "$@"
