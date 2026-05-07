#!/bin/bash
source init.sh
source "$(cd "$(dirname "$0")/../.." && pwd)/common.sh"

# ---- Method config ----
ARCH="sam2aff_pt_fg"
MAIN_PY="projects/SAM2AFF/main.py"
CONFIG="projects/SAM2AFF/configs/SAM2AFF-16x256.yaml"
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

# ---- Train & Eval (volume determined by @suffix in filename) ----
TRAIN_BASE_CONFIG="projects/Baselines/configs/base/Train_Pool_All.yaml"
TRAIN_GPUS="0,1,2,3"
# run_train 4
# eval_on Flywire_1
# eval_on AC3_100
# eval_on Wafer4_test
# eval_on m93
eval_on fib25-1
# eval_on cremiA_test
# eval_on cremiB_test
# eval_on cremiC_test
# eval_on Flywire_2