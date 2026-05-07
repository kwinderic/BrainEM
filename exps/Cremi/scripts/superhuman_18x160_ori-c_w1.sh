#!/bin/bash
source .sh
source "$(cd "$(dirname "$0")/../.." && pwd)/common.sh"

# ---- Method config ----
ARCH="superhuman"
MAIN_PY="projects/Baselines/main.py"
CONFIG="projects/Baselines/configs/base/SUPERHUMAN-BASE.yaml"
TRAIN_BASE_CONFIG="projects/Baselines/configs/base/CREMI-Base.yaml"
CHECKPOINT=""
MODEL_EXTRA_ARGS=(
    MODEL.WEIGHT_OPT '[["1", "0"]]'
)

# ---- Setup ----
setup_workdir

# ---- Train (uncomment to train) ----
# run_train 4

# ---- Eval ----
eval_on Wafer4
eval_on AC3-AC4
eval_on Flywire
