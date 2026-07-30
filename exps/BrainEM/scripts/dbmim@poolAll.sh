#!/bin/bash
# dbMiM, Cross-Source track. Stage 2/2: supervised finetuning on the
# multi-source pool from the ViT encoder produced by dbmim_pretrain.sh, then
# evaluate on SNEMI, FIB25, Wafer4 and M93.
#
# Pipeline per test volume: infer -> waterz -> eval -> vis.
# Outputs: exps/BrainEM/outputs/dbmim/poolAll/
# Run from the repo root.

# Locate exps/_lib/common.sh by walking up from this script.
_dir="$(cd "$(dirname "$0")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -f "$_dir/_lib/common.sh" ]; do
    _dir="$(dirname "$_dir")"
done
source "$_dir/_lib/common.sh"

# ---- Method config ----
ARCH="dbmim_unetr_aniso_em"
MAIN_PY="projects/dbMiM/main.py"
CONFIG="projects/dbMiM/configs/dbMiM-Base.yaml"

# ViT encoder from stage 1. Run dbmim_pretrain.sh first, or point this at a
# released encoder checkpoint.
PRETRAIN_ENC="exps/BrainEM/outputs/dbmim_pretrain/checkpoint_best.pth.tar"

# Solver settings live in dbMiM-Base.yaml, following the reference R48 finetune
# config (AdamW 8e-5, encoder 1e-5, constant LR, 20k iters). batch_size is per
# rank there, so 2/rank is kept. Where that config and the paper differ we follow
# the config; see README.
MODEL_EXTRA_ARGS=(
    MODEL.DBMIM_PRETRAIN "${PRETRAIN_ENC}"
    SOLVER.SAMPLES_PER_BATCH 2
    SOLVER.ITERATION_TOTAL 20000
    SOLVER.ITERATION_SAVE 10000
    SYSTEM.NUM_CPUS 16
)

# ---- Setup ----
setup_workdir

# ---- Train on the multi-source pool ----
TRAIN_BASE_CONFIG="configs/datasets/TrainPoolAll.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"   # override from the shell, or unset to auto-pick
run_train 3

# ---- Eval on the four cross-source volumes ----
# No validation loader is built here, so no checkpoint_best.pth.tar is written;
# eval points at the final numbered checkpoint.
CHECKPOINT="${WORK_DIR}/checkpoint_20000.pth.tar"
eval_on SNEMI
eval_on FIB25
eval_on Wafer4
eval_on M93
