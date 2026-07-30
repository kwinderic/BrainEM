#!/bin/bash
# SegNeuron stage 2/2: supervised finetuning on the CREMI A+B+C pool, initialized
# from the self-supervised encoder produced by segneuron_pretrain.sh.
#
# Two heads (affinity + foreground) fused inside forward() at inference, so
# evaluation goes through eval_segneuron rather than eval_on.
#
# Outputs: exps/BrainEM/outputs/segneuron/cremiAll/
# Run from the repo root.

# Locate exps/_lib/common.sh by walking up from this script.
_dir="$(cd "$(dirname "$0")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -f "$_dir/_lib/common.sh" ]; do
    _dir="$(dirname "$_dir")"
done
source "$_dir/_lib/common.sh"

# ---- Method config ----
ARCH="segneuron"
MAIN_PY="projects/SegNeuron/main.py"
CONFIG="projects/SegNeuron/configs/SegNeuron-Base.yaml"

# Encoder from stage 1. Run segneuron_pretrain.sh first, or point this at a
# released encoder checkpoint.
PRETRAIN_ENC="exps/BrainEM/outputs/segneuron_pretrain/checkpoint_best.pth.tar"

# Solver settings live in SegNeuron-Base.yaml. The official batch_size 8 is a
# global value under DataParallel; on DDP it is GPUs x samples-per-rank.
MODEL_EXTRA_ARGS=(
    MODEL.SEGNEURON_PRETRAIN "${PRETRAIN_ENC}"
    SOLVER.SAMPLES_PER_BATCH 1
    SOLVER.ITERATION_TOTAL 100000
)

# ---- Setup ----
setup_workdir

# ---- Train on the CREMI A+B+C pool ----
TRAIN_BASE_CONFIG="configs/datasets/CremiAll.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5}"   # override from the shell, or unset to auto-pick
run_train 6

# ---- Eval (SegNeuron infer -> waterz -> eval -> vis) ----
# No validation loader is built here, so no checkpoint_best.pth.tar is written;
# eval points at the final numbered checkpoint.
SEGNEURON_CKPT="${WORK_DIR}/checkpoint_100000.pth.tar"
eval_segneuron CremiA
eval_segneuron CremiB
eval_segneuron CremiC
eval_segneuron FlywireSmall
eval_segneuron Flywire
