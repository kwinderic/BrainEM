#!/bin/bash
# SuperHuman, Cross-Source track: train on the multi-source pool, then
#   evaluate on SNEMI, FIB25, Wafer4 and M93.
#
# Same method and same pipeline as superhuman@cremiAll.sh; only the training
# config differs (TrainPoolAll instead of CremiAll) and the eval targets are the
# four cross-source volumes.
#
# Outputs: exps/BrainEM/outputs/superhuman/poolAll/
# Run from the repo root.

# Locate exps/_lib/common.sh by walking up from this script.
_dir="$(cd "$(dirname "$0")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -f "$_dir/_lib/common.sh" ]; do
    _dir="$(dirname "$_dir")"
done
source "$_dir/_lib/common.sh"

# ---- Method config ----
ARCH="superhuman"
MAIN_PY="projects/Baselines/main.py"
CONFIG="projects/Baselines/configs/base/SUPERHUMAN-BASE.yaml"
MODEL_EXTRA_ARGS=(
    # ---- affinity target / loss (shared by the methods using this objective) ----
    MODEL.OUT_PLANES 3
    MODEL.TARGET_OPT '["2"]'
    MODEL.LABEL_EROSION 1
    MODEL.LOSS_OPTION '[["WeightedBCEWithLogitsLoss", "DiceLoss"]]'
    MODEL.LOSS_WEIGHT '[[1.0, 0.5]]'
    MODEL.WEIGHT_OPT '[["1", "0"]]'
    MODEL.OUTPUT_ACT '[["none", "sigmoid"]]'
    MODEL.NORM_MODE gn
    INFERENCE.OUTPUT_ACT '["sigmoid"]'
    # ---- solver ----
    SOLVER.SAMPLES_PER_BATCH 1
    SOLVER.ITERATION_TOTAL 200000
    SOLVER.ITERATION_SAVE  20000
    SOLVER.ITERATION_VAL   20000
)

# ---- Setup ----
setup_workdir

# ---- Train on the multi-source pool ----
TRAIN_BASE_CONFIG="configs/datasets/TrainPoolAll.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"   # override from the shell, or unset to auto-pick
run_train 2

# ---- Eval on the four cross-source volumes ----
eval_on SNEMI
eval_on FIB25
eval_on Wafer4
eval_on M93
