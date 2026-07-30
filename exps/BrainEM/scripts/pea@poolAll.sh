#!/bin/bash
# PEA, Cross-Source track: train on the multi-source pool, then
#   evaluate on SNEMI, FIB25, Wafer4 and M93.
#
# Uses four multi-scale affinity targets and computes its own loss inside
# forward(); the LOSS_* entries in PEA-BASE.yaml are placeholders. See README.
#
# Outputs: exps/BrainEM/outputs/pea/poolAll/
# Run from the repo root.

# Locate exps/_lib/common.sh by walking up from this script.
_dir="$(cd "$(dirname "$0")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -f "$_dir/_lib/common.sh" ]; do
    _dir="$(dirname "$_dir")"
done
source "$_dir/_lib/common.sh"

# ---- Method config ----
ARCH="pea"
MAIN_PY="projects/Baselines/main.py"
CONFIG="projects/Baselines/configs/base/PEA-BASE.yaml"

# Architecture, the four multi-scale targets and the solver all live in
# PEA-BASE.yaml, matching the reference implementation's 3d_cremi_h160 recipe
# (Adam lr 1e-4, amsgrad, eps 1e-2, 100k iters, crop 18x160x160). Only the
# per-run batch split is set here.
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
# PEA writes affinity as float32; post_process.py accepts either that or the
# uint8/vol0 layout the other methods emit.
eval_on SNEMI
eval_on FIB25
eval_on Wafer4
eval_on M93
