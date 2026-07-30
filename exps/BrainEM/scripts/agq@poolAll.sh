#!/bin/bash
# AGQ, Cross-Source track: train on the multi-source pool, then
#   evaluate on SNEMI, FIB25, Wafer4 and M93.
#
# Predicts affinity alongside learned instance queries, so it takes two targets:
# "9-1" (instance label, for the query decoder) and "2" (affinity). Only the
# affinity head feeds post-processing. See README.
#
# Outputs: exps/BrainEM/outputs/agq/poolAll/
# Run from the repo root.

# Locate exps/_lib/common.sh by walking up from this script.
_dir="$(cd "$(dirname "$0")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -f "$_dir/_lib/common.sh" ]; do
    _dir="$(dirname "$_dir")"
done
source "$_dir/_lib/common.sh"

# ---- Method config ----
ARCH="affinity_knet"
MAIN_PY="projects/AGQ/main.py"
CONFIG="projects/AGQ/configs/AGQ-Base.yaml"

# Query count, mask initialisation and the solver live in AGQ-Base.yaml
# (Adam lr 1e-4, 100k iters, crop 17x257x257, 100 learned masks).
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
