#!/bin/bash
# MALA, Cross-Source track: train on the multi-source pool, then
#   evaluate on SNEMI, FIB25, Wafer4 and M93.
#
# The only valid-padding network here: output (56x56x56) is smaller than input
# (84x268x268) and inference strides by the output size with no overlap.
#
# Outputs: exps/BrainEM/outputs/mala/poolAll/
# Run from the repo root.

# Locate exps/_lib/common.sh by walking up from this script.
_dir="$(cd "$(dirname "$0")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -f "$_dir/_lib/common.sh" ]; do
    _dir="$(dirname "$_dir")"
done
source "$_dir/_lib/common.sh"

# ---- Method config ----
ARCH="mala"
MAIN_PY="projects/Baselines/main.py"
CONFIG="projects/Baselines/configs/base/MALA-BASE.yaml"

# Geometry and solver live in MALA-BASE.yaml, which keeps the original paper's
# 84x268x268 -> 56x56x56 shapes (the SuperHuman repo's port shrinks these to
# 53x268x268 -> 25x56x56; the yaml notes both). MALA trains with MalisLoss, its
# defining component, rather than the BCE+Dice the other baselines share -- so
# unlike them this script does not override the loss.
MODEL_EXTRA_ARGS=(
    SOLVER.SAMPLES_PER_BATCH 1
    SYSTEM.NUM_CPUS 0
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
