#!/bin/bash
# AGQ (affinity_knet): train on the CREMI A+B+C pool, then evaluate.
#
# Predicts affinity alongside learned instance queries, so it takes two targets:
# "9-1" (instance label, for the query decoder) and "2" (affinity). Only the
# affinity head feeds post-processing. See README.
#
# Outputs: exps/BrainEM/outputs/agq/cremiAll/
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

# ---- Train on the CREMI A+B+C pool ----
TRAIN_BASE_CONFIG="configs/datasets/CremiAll.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"   # override from the shell, or unset to auto-pick
run_train 2

# ---- Eval (infer -> waterz -> eval -> vis) ----
eval_on CremiA
eval_on CremiB
eval_on CremiC
# FlyWire is same-source with CREMI (both FAFB). The full crop needs hours of
# inference and ~180 GB RAM for the watershed; see README.
eval_on FlywireSmall
eval_on Flywire
