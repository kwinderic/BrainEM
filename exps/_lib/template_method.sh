#!/bin/bash
# ============================================================================
# Template for a method script.
# Copy this file, rename to <method>[@<subset>].sh, place under any
# exps/<workflow>/scripts/, fill in the variables below.
#
# File-name convention (parsed by setup_workdir in common.sh):
#   <method>.sh             -> outputs/<method>/
#   <method>@<subset>.sh    -> outputs/<method>/<subset>/
# ============================================================================

# Locate exps/_lib/common.sh by walking up from this script.
_dir="$(cd "$(dirname "$0")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -f "$_dir/_lib/common.sh" ]; do
    _dir="$(dirname "$_dir")"
done
source "$_dir/_lib/common.sh"

# ---- Method config ----
ARCH="<arch_name>"                                       # MODEL.ARCHITECTURE
MAIN_PY="projects/<Project>/main.py"
CONFIG="projects/<Project>/configs/<config>.yaml"
MODEL_EXTRA_ARGS=(
    # Same-ARCH ablation knobs go here. Examples:
    # MODEL.SAM2_CONFIG large
    # MODEL.FILTERS '[16, 64, 64, 64, 64]'
    # MODEL.IS_FREEZE_ENCODER False
)

# ---- Setup ----
setup_workdir

# ---- Train ----
TRAIN_BASE_CONFIG="configs/datasets/TrainPoolAll.yaml"
TRAIN_GPUS="0,1,2,3"            # comment out to auto-pick via wait_gpus
# run_train 4

# ---- Eval ----
# Pass the dataset name (yaml stem) of any file in configs/datasets/.
# Currently available: SNEMI, CremiA, CremiB, CremiC, Wafer4, Flywire,
# FIB25, M93, TrainPoolAll.
# eval_on CremiA
# eval_on FIB25
