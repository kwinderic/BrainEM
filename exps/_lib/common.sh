#!/bin/bash
# exps/_lib/common.sh — repo-level pipeline entry points.
#
# Single source of truth for dataset paths is configs/datasets/<Name>.yaml.
# This file reads INFERENCE.{INPUT_PATH,IMAGE_NAME,LABEL_PATH} from there and
# wires them into postprocess / vis so bash and yaml never disagree.
#
# Method script protocol:
#   1. source this file (use exps/_lib/template_method.sh as starting point)
#   2. Set: ARCH, MAIN_PY, CONFIG, MODEL_EXTRA_ARGS (array)
#      Optional: CHECKPOINT (default ${WORK_DIR}/checkpoint_best.pth.tar)
#                TRAIN_GPUS / INFER_GPUS (default = wait_gpus auto-assign)
#                CONFIGS_ROOT (default configs/datasets)
#   3. Call setup_workdir
#   4. Call run_train [NUM_GPUS] and/or eval_on <DatasetName> [thresh]
#
# DatasetName must match a yaml in configs/datasets/, e.g.
#   eval_on CremiA   ->  configs/datasets/CremiA.yaml
#
# All paths below are relative to the repo root -- run scripts from there.

# Shell helpers (wait_gpus / save_log / print_self), resolved next to this file.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# ---------- Setup ----------
setup_workdir() {
    script_name="$0"
    script_name="${script_name##*/}"
    script_name="${script_name%.*}"

    # The exp root is the nearest ancestor containing a scripts/ subdir, so a
    # script works at any depth under it.
    local _script_dir
    _script_dir="$(cd "$(dirname "$0")" && pwd)"
    EXP_DIR="${_script_dir}"
    while [ "${EXP_DIR}" != "/" ] && [ ! -d "${EXP_DIR}/scripts" ]; do
        EXP_DIR="$(dirname "${EXP_DIR}")"
    done
    if [ "${EXP_DIR}" = "/" ]; then
        echo "ERROR: setup_workdir cannot find exp root (no ancestor has a scripts/ subdir)" >&2
        return 1
    fi

    # Optional EXP_GROUP nests runs under a sub-tree of outputs/.
    local out_root="${EXP_DIR}/outputs"
    if [ -n "${EXP_GROUP:-}" ]; then
        out_root="${out_root}/${EXP_GROUP}"
    fi

    # method@subset.sh -> outputs/[<group>/]method/subset/
    if [[ "${script_name}" == *@* ]]; then
        METHOD_NAME="${script_name%%@*}"
        SUB_NAME="${script_name#*@}"
        WORK_DIR="${out_root}/${METHOD_NAME}/${SUB_NAME}"
    else
        METHOD_NAME="${script_name}"
        SUB_NAME=""
        WORK_DIR="${out_root}/${script_name}"
    fi

    echo "script_name: ${script_name}"
    echo "WORK_DIR:    ${WORK_DIR}"

    mkdir -p "${WORK_DIR}"
    save_log "${WORK_DIR}/log.txt"
    print_self "$0"
}

# ---------- Dataset resolver ----------
# Maps a DatasetName -> yaml path; reads INFERENCE.{INPUT_PATH,IMAGE_NAME,LABEL_PATH}
# from yaml and exports IMAGE_PATH / LABEL_PATH / DATASET_YAML.
CONFIGS_ROOT="${CONFIGS_ROOT:-configs/datasets}"

_dataset_paths() {
    local name="$1"
    DATASET_YAML="${CONFIGS_ROOT}/${name}.yaml"
    if [ ! -f "${DATASET_YAML}" ]; then
        echo "ERROR: dataset config not found: ${DATASET_YAML}" >&2
        echo "       Available:" >&2
        ls "${CONFIGS_ROOT}"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//; s/^/         /' >&2
        return 1
    fi

    eval "$(python3 - "${DATASET_YAML}" <<'PY'
import os, sys, yaml
yaml_path = sys.argv[1]
with open(yaml_path) as f:
    cfg = yaml.safe_load(f) or {}
inf = cfg.get('INFERENCE', {}) or {}
ds  = cfg.get('DATASET',   {}) or {}
root = inf.get('INPUT_PATH') or ds.get('INPUT_PATH') or ''
img  = inf.get('IMAGE_NAME') or ''
image_path = os.path.join(root, img) if root and img else ''
label_path = inf.get('LABEL_PATH') or ''
def _q(v): return str(v).replace("'", "'\\''")
print(f"IMAGE_PATH='{_q(image_path)}'")
print(f"LABEL_PATH='{_q(label_path)}'")
PY
)"
    return 0
}

# ---------- Inference ----------
run_infer() {
    local name="$1"
    _dataset_paths "${name}" || return 1
    local output_dir="${WORK_DIR}/${name}"
    local ckpt="${CHECKPOINT:-${WORK_DIR}/checkpoint_best.pth.tar}"

    echo "==================== Inference: ${name} ===================="
    echo "Dataset yaml: ${DATASET_YAML}"
    echo "Checkpoint:   ${ckpt}"
    if [ -n "${INFER_GPUS:-}" ]; then
        local num_gpus
        num_gpus=$(echo "${INFER_GPUS}" | tr ',' '\n' | wc -l)
        export CUDA_VISIBLE_DEVICES="${INFER_GPUS}"
    else
        local num_gpus=1
    fi
    python -u \
        ${MAIN_PY} \
            --config-base "${DATASET_YAML}" \
            --config-file ${CONFIG} \
            --inference \
            --checkpoint "${ckpt}" \
            MODEL.ARCHITECTURE ${ARCH} \
            DATASET.OUTPUT_PATH "${output_dir}" \
            INFERENCE.OUTPUT_PATH "${output_dir}" \
            SYSTEM.NUM_GPUS ${num_gpus} \
            SOLVER.SAMPLES_PER_BATCH 1 \
            "${MODEL_EXTRA_ARGS[@]}"
}

# ---------- Postprocess + eval ----------
run_postprocess() {
    local name="$1"
    local thresh="${2:-0.50}"
    _dataset_paths "${name}" || return 1
    local output_dir="${WORK_DIR}/${name}"

    echo "==================== Post-process: ${name} (thresh=${thresh}) ===================="
    python scripts/post_process.py \
        "${output_dir}/result.h5" \
        "${output_dir}" \
        "${thresh}"

    if [ -n "${LABEL_PATH}" ] && [ -f "${LABEL_PATH}" ]; then
        python scripts/eval.py \
            eval_full \
            "${LABEL_PATH}" \
            "${output_dir}/segments.npy" \
            "${output_dir}/result.h5"
    else
        echo "Skip eval: LABEL_PATH not set or missing (${LABEL_PATH})"
    fi
}


run_eval_only() {
    local name="$1"
    local thresh="${2:-0.50}"
    _dataset_paths "${name}" || return 1
    local output_dir="${WORK_DIR}/${name}"

    # Segmentation metrics only (VOI / ARAND). Skips the affinity argument so
    # eval.py does not re-read result.h5 for voxel-level accuracy -- that pass
    # dominates runtime on the 400x4000x4000 FlyWire crop.
    if [ -n "${LABEL_PATH}" ] && [ -f "${LABEL_PATH}" ]; then
        python scripts/eval.py \
            eval_full \
            "${LABEL_PATH}" \
            "${output_dir}/segments.npy"
    else
        echo "Skip eval: LABEL_PATH not set or missing (${LABEL_PATH})"
    fi
}


# ---------- Visualize ----------
run_vis() {
    local name="$1"
    _dataset_paths "${name}" || return 1
    local output_dir="${WORK_DIR}/${name}"

    echo "==================== Visualize: ${name} ===================="
    python scripts/vis_slices.py \
        --image "${IMAGE_PATH}" \
        --label "${LABEL_PATH}" \
        --affinity "${output_dir}/result.h5" \
        --segment "${output_dir}/segments.npy" \
        --output_dir "${output_dir}/vis"
}

# ---------- Full pipeline ----------
eval_on() {
    local name="$1"
    local thresh="${2:-0.50}"
    run_infer "${name}"
    
    run_postprocess "${name}" "${thresh}"
    run_vis "${name}"
}


# ---------- postprocess pipeline ----------
postprocess_on() {
    local name="$1"
    local thresh="${2:-0.50}"
    
    run_postprocess "${name}" "${thresh}"
    run_vis "${name}"
}

eval_only_on() { run_eval_only "$1" "${2:-0.50}"; }
vis_on()       { run_vis "$1"; }

# ---------- waterz threshold sweep ----------
# Usage: run_sweep <DatasetName> [thresholds] [extra args to sweep_waterz.py...]
# One watershed, then waterz walks all thresholds incrementally, so the sweep
# costs about the same as a single postprocess at the largest threshold.
# Metrics for every threshold go to ${WORK_DIR}/<name>/sweep_waterz.csv.
run_sweep() {
    local name="$1"
    local thresholds="${2:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}"
    # Drop <name> and <thresholds>; anything left is forwarded to the python tool.
    if [ "$#" -ge 2 ]; then shift 2; else shift "$#"; fi
    _dataset_paths "${name}" || return 1
    local output_dir="${WORK_DIR}/${name}"

    echo "==================== Waterz sweep: ${name} (thresholds=${thresholds}) ===================="
    local label_arg=()
    if [ -n "${LABEL_PATH}" ] && [ -f "${LABEL_PATH}" ]; then
        label_arg=(--label "${LABEL_PATH}")
    else
        echo "No LABEL_PATH (${LABEL_PATH}) — sweeping without metrics"
    fi
    python -u scripts/sweep_waterz.py \
        --res_h5 "${output_dir}/result.h5" \
        --out_dir "${output_dir}" \
        --thresholds "${thresholds}" \
        "${label_arg[@]}" \
        "$@"
}

# infer -> sweep thresholds (no vis; pick a threshold from the csv, then
# postprocess_on/vis_on if you want the segmentation + slice figures).
sweep_on() {
    local name="$1"
    run_infer "${name}" || return 1
    run_sweep "$@"
}

# ---------- SegNeuron pipeline (MNet dual-head, mask fused into affinity) ----------
# SegNeuron's forward() fuses its affinity and foreground heads at inference and
# returns 3-channel affinity, so the standard waterz postprocess + eval follow.
# Requires: SEGNEURON_CKPT set to the model checkpoint.
eval_segneuron() {
    local name="$1"
    local thresh="${2:-0.50}"
    _dataset_paths "${name}" || return 1
    local output_dir="${WORK_DIR}/${name}"
    mkdir -p "${output_dir}"
    local ckpt="${SEGNEURON_CKPT:-${WORK_DIR}/checkpoint_best.pth.tar}"

    echo "==================== Inference (SegNeuron): ${name} ===================="
    echo "Dataset yaml: ${DATASET_YAML}"
    echo "Checkpoint:   ${ckpt}"
    # Training uses two targets ['2','0']; forward() fuses them at inference and
    # returns 3-channel affinity, so the target list is overridden to one entry
    # (otherwise SplitActivation would try to split 3 channels into 3+1).
    local num_gpus=1
    if [ -n "${INFER_GPUS:-}" ]; then
        num_gpus=$(echo "${INFER_GPUS}" | tr ',' '\n' | wc -l)
        export CUDA_VISIBLE_DEVICES="${INFER_GPUS}"
    fi
    python -u ${MAIN_PY} \
        --config-base "${DATASET_YAML}" \
        --config-file ${CONFIG} \
        --inference \
        --checkpoint "${ckpt}" \
        MODEL.ARCHITECTURE ${ARCH} \
        DATASET.OUTPUT_PATH "${output_dir}" \
        INFERENCE.OUTPUT_PATH "${output_dir}" \
        MODEL.TARGET_OPT "['2']" \
        MODEL.WEIGHT_OPT "[['0']]" \
        MODEL.LOSS_OPTION "[['WeightedMSE']]" \
        MODEL.OUTPUT_ACT "[['none']]" \
        INFERENCE.OUTPUT_ACT "['none']" \
        MODEL.OUT_PLANES 3 \
        SYSTEM.NUM_GPUS ${num_gpus} \
        SOLVER.SAMPLES_PER_BATCH "${INFER_BATCH:-8}" \
        "${MODEL_EXTRA_ARGS[@]}"

    run_postprocess "${name}" "${thresh}"   # standard waterz + eval on result.h5
    run_vis "${name}"
}

# ---------- Train ----------
# Usage: run_train [NUM_GPUS]
# Set TRAIN_BASE_CONFIG before calling (path to a dataset yaml or training-pool yaml).
# Set TRAIN_GPUS="2,3,4,5" to pin GPUs (otherwise wait_gpus auto-picks).
run_train() {
    local num_gpus="${1:-4}"
    local port=$(( ( RANDOM % 1000 ) + 2000 ))

    echo "==================== Train: ${script_name} (${num_gpus} GPUs) ===================="
    if [ -n "${TRAIN_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    else
        wait_gpus ${num_gpus}
    fi
    # Optional pretrain checkpoint (set PRETRAIN_CKPT to a path to enable finetune).
    # Distinct from CHECKPOINT, which is used by run_infer/eval_on for inference.
    local ckpt_arg=()
    if [ -n "${PRETRAIN_CKPT:-}" ]; then
        echo "Init from pretrain checkpoint: ${PRETRAIN_CKPT}"
        ckpt_arg=(--checkpoint "${PRETRAIN_CKPT}")
    fi
    OMP_NUM_THREADS=8 \
    python -u -m torch.distributed.run \
        --nproc_per_node=${num_gpus} \
        --master_port=${port} \
        ${MAIN_PY} \
            --distributed \
            --config-base "${TRAIN_BASE_CONFIG}" \
            --config-file ${CONFIG} \
            "${ckpt_arg[@]}" \
            MODEL.ARCHITECTURE ${ARCH} \
            DATASET.OUTPUT_PATH "${WORK_DIR}" \
            MONITOR.WANDB_NAME "${script_name}" \
            "${MODEL_EXTRA_ARGS[@]}"
}

# ---------- Train (single-process, no DDP) ----------
# One GPU (TRAIN_GPUS or wait_gpus 1), calling main.py directly. For trainers
# that are not DDP-aware; none of the eight methods here need it.
run_train_single() {
    echo "==================== Train (single): ${script_name} ===================="
    if [ -n "${TRAIN_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    else
        wait_gpus 1
    fi
    local ckpt_arg=()
    if [ -n "${PRETRAIN_CKPT:-}" ]; then
        echo "Init from pretrain checkpoint: ${PRETRAIN_CKPT}"
        ckpt_arg=(--checkpoint "${PRETRAIN_CKPT}")
    fi
    OMP_NUM_THREADS=8 \
    python -u ${MAIN_PY} \
        --config-base "${TRAIN_BASE_CONFIG}" \
        --config-file ${CONFIG} \
        "${ckpt_arg[@]}" \
        MODEL.ARCHITECTURE ${ARCH} \
        DATASET.OUTPUT_PATH "${WORK_DIR}" \
        MONITOR.WANDB_NAME "${script_name}" \
        "${MODEL_EXTRA_ARGS[@]}"
}
