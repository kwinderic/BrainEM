#!/bin/bash
source .sh
source "$(cd "$(dirname "$0")/../.." && pwd)/common.sh"

# ---- Method config ----
ARCH="sam2aff_pt_fg"
MAIN_PY="projects/SAM2AFF/main.py"
CONFIG="projects/SAM2AFF/configs/SAM2AFF-16x256.yaml"
TRAIN_BASE_CONFIG="projects/SAM2AFF/configs/CREMI-Base.yaml"
CHECKPOINT=""
MODEL_EXTRA_ARGS=(
    MODEL.CONV_AFTER_FUSION True
    MODEL.IS_FREEZE_ENCODER False
    MODEL.SAM2_CONFIG large
    DATASET.MEAN 0.5
    DATASET.STD 0.5
    MODEL.WEIGHT_OPT '[["0", "0"]]'
    MODEL.FILTERS '[16, 64, 64, 64, 64]'
    MODEL.FG_STAGE_NUM 1
)

# ---- Setup ----
setup_workdir

# ---- Train (uncomment to train) ----
# run_train 4

# ---- Eval ----
eval_on Wafer4
eval_on AC3-AC4
eval_on Flywire
