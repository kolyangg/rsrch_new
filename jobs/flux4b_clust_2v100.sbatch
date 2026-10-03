#!/usr/bin/env bash
# Create logs/clust BEFORE sbatch. One task launches the serial controller;
# torchrun starts exactly two workers for each training segment.
#SBATCH --job-name=rsrch-new-4b-v100
#SBATCH --account=proj_1892
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100:2
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/nasilaev/rsrch_new
#SBATCH --output=/home/nasilaev/rsrch_new/logs/clust/%x-%j.out
#SBATCH --error=/home/nasilaev/rsrch_new/logs/clust/%x-%j.err

# This cluster advertises RealMemory=1: omit --mem/--mem-per-cpu.
# Launch only after user approval, data receipt and environment preparation:
#   mkdir -p logs/clust
#   sbatch jobs/flux4b_clust_2v100.sbatch
# Resume: sbatch jobs/flux4b_clust_2v100.sbatch --resume
# Inspect: squeue -j JOBID; sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
# Cancel: scancel JOBID. Resume uses the last complete 500-update checkpoint.
set -euo pipefail
exec bash scripts/run_clust_v100.sh \
  --run runs/flux4b_clust_2v100_fp16_qkvo_r128_20k_20261003 "$@"
