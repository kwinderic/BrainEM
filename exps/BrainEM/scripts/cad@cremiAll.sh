#!/bin/bash
# CAD (Co-detection of Affinities and Densities): train on the CREMI A+B+C pool,
# then evaluate.
#
# 2D dual-stream U-Net coupled to a 3D U-Net; computes its own loss inside
# forward() from four multi-scale affinity targets. See README.
#
# Outputs: exps/BrainEM/outputs/cad/cremiAll/
# Run from the repo root.

# Locate exps/_lib/common.sh by walking up from this script.
_dir="$(cd "$(dirname "$0")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -f "$_dir/_lib/common.sh" ]; do
    _dir="$(dirname "$_dir")"
done
source "$_dir/_lib/common.sh"

# ---- Method config ----
ARCH="cad"
MAIN_PY="projects/Baselines/main.py"
CONFIG="projects/Baselines/configs/base/CAD-BASE.yaml"

# Architecture, the CAD_* loss weights and the solver live in CAD-BASE.yaml
# (Adam lr 1e-4, amsgrad, 200k iters, crop 18x160x160, 2D branch from 10k).
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
