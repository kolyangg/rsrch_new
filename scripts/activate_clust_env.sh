#!/usr/bin/env bash
# Source this before cluster setup or execution; never install in base/PhotoMaker.
source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh
conda activate /home/nasilaev/.conda/envs/rsrch_new
export PYTHONNOUSERSITE=1
export BA_ROOT="/home/nasilaev/rsrch_new"
export BA_ENVS_DIR="$BA_ROOT/envs/clust-v100"
export BA_FLUX_PYTHON="$CONDA_PREFIX/bin/python"
[[ "$CONDA_PREFIX" == /home/nasilaev/.conda/envs/rsrch_new ]] || return 1
