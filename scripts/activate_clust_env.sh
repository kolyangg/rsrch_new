#!/usr/bin/env bash
# Source this before cluster setup or execution; never install in base/PhotoMaker.
source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh
conda activate /home/nasilaev/.conda/envs/rsrch_new
# InsightFace's GCC-built extension also needs this runtime in the metric venvs.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONNOUSERSITE=1
export BA_ROOT="/home/nasilaev/rsrch_new"
export BA_ENVS_DIR="$BA_ROOT/envs/clust-v100"
export BA_FLUX_PYTHON="$CONDA_PREFIX/bin/python"
[[ "$CONDA_PREFIX" == /home/nasilaev/.conda/envs/rsrch_new ]] || return 1
