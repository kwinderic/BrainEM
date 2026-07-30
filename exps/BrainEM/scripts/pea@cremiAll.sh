#!/bin/bash
# PEA (Pixel-Embedded Affinity): train on the CREMI A+B+C pool, then evaluate.
#
# Uses four multi-scale affinity targets and computes its own loss inside
# forward(); the LOSS_* entries in PEA-BASE.yaml are placeholders. See README.
#
# Outputs: exps/BrainEM/outputs/pea/cremiAll/
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

# ---- Train on the CREMI A+B+C pool ----
TRAIN_BASE_CONFIG="configs/datasets/CremiAll.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"   # override from the shell, or unset to auto-pick
run_train 2

# ---- Eval (infer -> waterz -> eval -> vis) ----
# PEA writes affinity as float32; post_process.py accepts either that or the
# uint8/vol0 layout the other methods emit.
eval_on CremiA
eval_on CremiB
eval_on CremiC
# FlyWire is same-source with CREMI (both FAFB). The full crop needs hours of
# inference and ~180 GB RAM for the watershed; see README.
eval_on FlywireSmall
eval_on Flywire
