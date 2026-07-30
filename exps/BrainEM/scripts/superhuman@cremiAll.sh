#!/bin/bash
# SuperHuman baseline: train on the CREMI A+B+C pool (4 nm), then evaluate on
# CREMI A/B/C and the two FlyWire crops.
#
# Pipeline per test volume: infer -> waterz -> eval -> vis.
# Outputs: exps/BrainEM/outputs/superhuman/cremiAll/
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
    # MODEL.FILTERS stays at the config default; it must match the checkpoint
    # loaded at inference time.
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

# ---- Train on the CREMI A+B+C pool ----
TRAIN_BASE_CONFIG="configs/datasets/CremiAll.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"   # override from the shell, or unset to auto-pick
run_train 2

# ---- Eval (infer -> waterz -> eval -> vis) ----
# CHECKPOINT defaults to ${WORK_DIR}/checkpoint_best.pth.tar.
eval_on CremiA
eval_on CremiB
eval_on CremiC
# FlyWire is same-source with CREMI (both FAFB). The full crop needs hours of
# inference and ~180 GB RAM for the watershed; see README.
eval_on FlywireSmall
eval_on Flywire
