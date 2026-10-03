#!/usr/bin/env bash
# Companion logger: sbatch jobs/clust_comet.sbatch TRAINING_JOB_ID RUN_DIRECTORY
# One CPU, no GPU; exits when the target job ends, with bounded final uploads.
#SBATCH --job-name=rsrch-comet
#SBATCH --account=proj_1892
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/nasilaev/rsrch_new
#SBATCH --output=/home/nasilaev/rsrch_new/logs/clust/comet-service-%j.log
set -euo pipefail
[[ $# == 2 ]] || { echo 'Expected TRAINING_JOB_ID RUN_DIRECTORY' >&2; exit 2; }
source scripts/activate_clust_env.sh
export PYTHONUNBUFFERED=1 COMET_DISPLAY_SUMMARY_LEVEL=0
exec python -u -m scripts.upload_clust_comet --job-id "$1" --run "$2"
