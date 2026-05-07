#!/bin/bash
source init.sh
source "$(cd "$(dirname "$0")/../.." && pwd)/common.sh"

# ---- Method config ----
ARCH="superhuman"
MAIN_PY="projects/Baselines/main.py"
CONFIG="projects/Baselines/configs/base/SUPERHUMAN-BASE.yaml"
MODEL_EXTRA_ARGS=(
    MODEL.WEIGHT_OPT '[["1", "0"]]'
    SOLVER.SAMPLES_PER_BATCH 1
    SOLVER.ITERATION_SAVE 20000
    SOLVER.ITERATION_TOTAL 200000
    SOLVER.ITERATION_VAL 20000
)

# ---- Setup ----
setup_workdir

# ---- Train & Eval (volume determined by @suffix in filename) ----
TRAIN_BASE_CONFIG="projects/Baselines/configs/base/Train_Pool_All.yaml"
TRAIN_GPUS="2,3"
# run_train 2
# eval_on Flywire_1
# eval_on AC3_100
# eval_on Wafer4_test
# eval_on m93
eval_on fib25-1
# eval_on cremiA_test
# eval_on cremiB_test
# eval_on cremiC_test
# eval_on Flywire_2
