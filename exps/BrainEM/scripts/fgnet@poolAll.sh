#!/bin/bash
# FGNet, Cross-Source track: train on the multi-source pool, then
#   evaluate on SNEMI, FIB25, Wafer4 and M93.
#
# Pipeline per test volume: infer -> waterz -> eval -> vis.
# Outputs: exps/BrainEM/outputs/fgnet/poolAll/
#
# Run from the repo root: the SAM2 backbone weights are resolved relative to it
# (projects/FGNet/sam2aff/sam2/checkpoints/ -- see README for the download).

# Locate exps/_lib/common.sh by walking up from this script.
_dir="$(cd "$(dirname "$0")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -f "$_dir/_lib/common.sh" ]; do
    _dir="$(dirname "$_dir")"
done
source "$_dir/_lib/common.sh"

# ---- Method config ----
ARCH="sam2aff_pt_fg"
MAIN_PY="projects/FGNet/main.py"
CONFIG="projects/FGNet/configs/FGNet-16x256.yaml"
MODEL_EXTRA_ARGS=(
    # ---- FGNet-specific knobs ----
    MODEL.CONV_AFTER_FUSION True
    MODEL.IS_FREEZE_ENCODER False
    MODEL.SAM2_CONFIG large       # sam2.1_hiera_large.pt
    MODEL.FILTERS '[16, 64, 64, 64, 64]'
    MODEL.FG_STAGE_NUM 1
    DATASET.MEAN 0.5
    DATASET.STD 0.5
    # ---- affinity target / loss (same objective as the baselines) ----
    MODEL.OUT_PLANES 3
    MODEL.TARGET_OPT '["2"]'
    MODEL.LABEL_EROSION 1
    MODEL.LOSS_OPTION '[["WeightedBCEWithLogitsLoss", "DiceLoss"]]'
    MODEL.LOSS_WEIGHT '[[1.0, 0.5]]'
    MODEL.WEIGHT_OPT '[["0", "0"]]'
    MODEL.OUTPUT_ACT '[["none", "sigmoid"]]'
    MODEL.NORM_MODE gn
    # ---- solver ----
    SOLVER.SAMPLES_PER_BATCH 1
    SOLVER.ITERATION_TOTAL 100000
    SOLVER.ITERATION_SAVE  10000
    SOLVER.ITERATION_VAL   10000
    INFERENCE.OUTPUT_ACT '["sigmoid"]'
    INFERENCE.SAMPLES_PER_BATCH 1
)

# ---- Setup ----
setup_workdir

# ---- Train on the multi-source pool ----
TRAIN_BASE_CONFIG="configs/datasets/TrainPoolAll.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"   # override from the shell, or unset to auto-pick
run_train 4

# ---- Eval on the four cross-source volumes ----
eval_on SNEMI
eval_on FIB25
eval_on Wafer4
eval_on M93
