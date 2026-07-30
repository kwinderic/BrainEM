#!/bin/bash
# SegNeuron stage 1/2: self-supervised pretraining (masked inpainting + HOG
# reconstruction). Produces the encoder that segneuron@cremiAll.sh finetunes --
# its down* layers are loaded via MODEL.SEGNEURON_PRETRAIN.
#
# Outputs: exps/BrainEM/outputs/segneuron_pretrain/
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

# SSL pretraining is driven entirely by the MODEL.SSL_* knobs below.
MODEL_EXTRA_ARGS=(
    MODEL.SSL_PRETRAIN True
    # SSL reads the unlabeled pool through the framework dataloader, so the
    # volume list comes from --config-base (configs/datasets/SSLPoolAll.yaml) and
    # the label entries must be None: VolumeDataset then yields images only.
    DATASET.LABEL_NAME None
    DATASET.VAL_IMAGE_NAME None
    DATASET.VAL_LABEL_NAME None
    # Official loss: 0.2 * MSE(image) + MSE(HOG).
    MODEL.SSL_IMG_WEIGHT 0.2
    MODEL.SSL_VIS_INTERVAL 4000
    MODEL.FILTERS '[32, 64, 96, 128, 256]'
    MODEL.INPUT_SIZE '[20, 128, 128]'
    # Paper: Adam lr 1e-3, global batch 8, 400k iters. Set to 200k here to fit
    # the compute budget; raise ITERATION_TOTAL to 400000 to match the paper.
    SOLVER.NAME Adam
    SOLVER.LR_SCHEDULER_NAME WarmupCosineLR
    SOLVER.BASE_LR 0.001
    SOLVER.WARMUP_ITERS 0
    SOLVER.SAMPLES_PER_BATCH 2
    SOLVER.ITERATION_TOTAL 200000
    SOLVER.ITERATION_SAVE 10000
    # Per-node total, split across the DDP ranks. The per-sample HOG is
    # CPU-bound, so this needs headroom.
    SYSTEM.NUM_CPUS 64
)

# ---- Setup ----
setup_workdir

# ---- Train ----
TRAIN_BASE_CONFIG="configs/datasets/SSLPoolAll.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"   # override from the shell, or unset to auto-pick
run_train 4
