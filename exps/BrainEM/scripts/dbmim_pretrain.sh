#!/bin/bash
# dbMiM stage 1/2: self-supervised pretraining (masked image modeling on
# unlabeled EM). Produces the ViT encoder that dbmim@cremiAll.sh finetunes.
#
# Outputs: exps/BrainEM/outputs/dbmim_pretrain/
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

# Values below follow the reference config pretrain_public_em_membrane_r16.yaml.
# Its batch_size is per rank (it trains under DDP), so 2/rank is kept here.
# Where that config and the paper text differ we follow the config: 160k iters
# (paper 200k), lr 1.5e-4 (paper 1e-4), AdamW (paper Adam).
#
# SSL_POLICY_ENABLED False uses a fixed mask_ratio, as in the reference's
# plain_mae / edgemask / mixedmask configs. Set it True to enable the learnable
# mask policy.
MODEL_EXTRA_ARGS=(
    MODEL.SSL_PRETRAIN True
    # Labels are dropped so VolumeDataset yields images only; the volume list
    # comes from --config-base.
    DATASET.LABEL_NAME None
    DATASET.VAL_IMAGE_NAME None
    DATASET.VAL_LABEL_NAME None
    MODEL.SSL_MASK_RATIO 0.75
    MODEL.SSL_MEMBRANE_WEIGHT 1.35
    MODEL.SSL_MEMBRANE_AXIS_WEIGHTS '[0.25, 1.0, 1.0]'
    MODEL.SSL_MEMBRANE_CLIP 5.0
    MODEL.SSL_STRUCTURE_WEIGHT 0.2
    MODEL.SSL_STRUCTURE_AXIS_WEIGHTS '[0.5, 1.0, 1.0]'
    MODEL.SSL_VIS_INTERVAL 2000
    MODEL.SSL_POLICY_ENABLED False
    MODEL.SSL_POLICY_FREEZE_AFTER 40000
    MODEL.INPUT_SIZE '[32, 160, 160]'
    SOLVER.BASE_LR 0.00015
    SOLVER.WEIGHT_DECAY 0.05
    SOLVER.SAMPLES_PER_BATCH 2
    SOLVER.ITERATION_TOTAL 160000
    SOLVER.ITERATION_SAVE 10000
    # Per-node total, split across ranks.
    SYSTEM.NUM_CPUS 16
)

# ---- Setup ----
setup_workdir

# ---- Train ----
TRAIN_BASE_CONFIG="configs/datasets/TrainPoolAll.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"   # override from the shell, or unset to auto-pick
run_train 3
