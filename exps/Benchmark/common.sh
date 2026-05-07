#!/bin/bash
# Benchmark/common.sh - Shared functions for benchmark experiments
#
# Method scripts should:
#   1. source 
#   2. source this file
#   3. Set variables: ARCH, MAIN_PY, CONFIG, MODEL_EXTRA_ARGS (array)
#      Optional: CHECKPOINT (defaults to ${WORK_DIR}/checkpoint_best.pth.tar)
#      Optional: TRAIN_GPUS, INFER_GPUS (defaults to wait_gpus auto-assign)
#   4. Call setup_workdir
#   5. Call eval_on DATASET for each test dataset
#
# Example:
#   source 
#   source "$(cd "$(dirname "$0")/../.." && pwd)/common.sh"
#   ARCH="superhuman"
#   MAIN_PY="projects/Baselines/main.py"
#   CONFIG="projects/Baselines/configs/base/SUPERHUMAN-BASE.yaml"
#   CHECKPOINT="/path/to/checkpoint_best.pth.tar"
#   MODEL_EXTRA_ARGS=( MODEL.WEIGHT_OPT '[["1", "0"]]' )
#   setup_workdir
#   eval_on Wafer4
#   eval_on AC3-AC4

# ===================== Setup =====================

setup_workdir() {
    script_name="$0"
    script_name="${script_name##*/}"
    script_name="${script_name%.*}"

    EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

    # Support @ separator: method_name@sub_name -> outputs/method_name/sub_name
    if [[ "${script_name}" == *@* ]]; then
        METHOD_NAME="${script_name%%@*}"
        SUB_NAME="${script_name#*@}"
        WORK_DIR="${EXP_DIR}/outputs/${METHOD_NAME}/${SUB_NAME}"
    else
        METHOD_NAME="${script_name}"
        SUB_NAME=""
        WORK_DIR="${EXP_DIR}/outputs/${script_name}"
    fi

    echo "script_name: ${script_name}"
    echo "WORK_DIR: ${WORK_DIR}"

    mkdir -p "${WORK_DIR}"
    save_log "${WORK_DIR}/log.txt"
    print_self "$0"
}

# ===================== Dataset Registry =====================
# Sets: EVAL_BASE_CONFIG, LABEL_PATH, IMAGE_PATH, EXTRA_INFER_ARGS (array)

_dataset_config() {
    local dataset="$1"
    EXTRA_INFER_ARGS=()

    case "${dataset}" in
        Wafer4_test)
            EVAL_BASE_CONFIG="projects/Baselines/configs/base/WAFER4-Base.yaml"
            LABEL_PATH="datasets/wafer4/wafer4_labels_test.h5"
            IMAGE_PATH="datasets/wafer4/wafer4_inputs_test.h5"
            EXTRA_INFER_ARGS=(INFERENCE.IMAGE_NAME wafer4_inputs_test.h5)
            ;;
        AC3_100)
            EVAL_BASE_CONFIG="projects/Baselines/configs/base/SNEMI-Base.yaml"
            LABEL_PATH="datasets/AC3-AC4/AC3_labels.h5"
            IMAGE_PATH="datasets/AC3-AC4/AC3_inputs.h5"
            ;;
        Flywire_1)
            EVAL_BASE_CONFIG="projects/Baselines/configs/base/Flywire-Base.yaml"
            LABEL_PATH="datasets/EM/Flywire/flywire_1_labels.h5"
            IMAGE_PATH="datasets/EM/Flywire/flywire_1.h5"
            ;;
        Flywire_2)
            EVAL_BASE_CONFIG="projects/Baselines/configs/base/Flywire-2-Base.yaml"
            LABEL_PATH="datasets/EM/Flywire/flywire_2_labels.h5"
            IMAGE_PATH="datasets/EM/Flywire/flywire_2_inputs.h5"
            ;;
        # ---- Add new datasets below ----
        cremiA_test)
            EVAL_BASE_CONFIG="projects/Baselines/configs/base/CREMI-Base-A.yaml"
            LABEL_PATH="datasets/cremi/sample_A_test_labels.h5"
            IMAGE_PATH="datasets/cremi/sample_A_test_inputs.h5"
            ;;
        cremiB_test)
            EVAL_BASE_CONFIG="projects/Baselines/configs/base/CREMI-Base-B.yaml"
            LABEL_PATH="datasets/cremi/sample_B_test_labels.h5"
            IMAGE_PATH="datasets/cremi/sample_B_test_inputs.h5"
            ;;
        cremiC_test)
            EVAL_BASE_CONFIG="projects/Baselines/configs/base/CREMI-Base-C.yaml"
            LABEL_PATH="datasets/cremi/sample_C_test_labels.h5"
            IMAGE_PATH="datasets/cremi/sample_C_test_inputs.h5"
            ;;
        m93)
            EVAL_BASE_CONFIG="projects/Baselines/configs/base/93-Base.yaml"
            INFER_IMAGE_PATH="9_3_78_97_Stack.tif"
            LABEL_PATH="datasets/pyw/93/9_3_78_97_Labels.tif"
            IMAGE_PATH="datasets/pyw/93/9_3_78_97_Stack.tif"
            ;;
        fib25-1)
            EVAL_BASE_CONFIG="projects/Baselines/configs/base/FIB25-1-Base.yaml"
            LABEL_PATH="datasets/EM/FIB25/0_label.tif"
            IMAGE_PATH="datasets/EM/FIB25/0.tif"
            ;;    
        *)
            echo "ERROR: Unknown dataset '${dataset}'"
            echo "Available: Wafer4_test, AC3_100, Flywire_1, Flywire_2, cremiA_test, cremiB_test, cremiC_test, m93, fib25-1"
            return 1
            ;;
    esac
}

# ===================== Inference =====================

run_infer() {
    local dataset="$1"
    _dataset_config "${dataset}" || return 1
    local output_dir="${WORK_DIR}/${dataset}"
    local ckpt="${CHECKPOINT:-${WORK_DIR}/checkpoint_best.pth.tar}"

    echo "==================== Inference: ${dataset} ===================="
    echo "Checkpoint: ${ckpt}"
    if [ -n "${INFER_GPUS:-}" ]; then
        local num_gpus=$(echo "${INFER_GPUS}" | tr ',' '\n' | wc -l)
        export CUDA_VISIBLE_DEVICES="${INFER_GPUS}"
    else
        # wait_gpus 1
        local num_gpus=1
    fi
    python -u \
        ${MAIN_PY} \
            --config-base "${EVAL_BASE_CONFIG}" \
            --config-file ${CONFIG} \
            --inference \
            --checkpoint "${ckpt}" \
            MODEL.ARCHITECTURE ${ARCH} \
            DATASET.OUTPUT_PATH "${output_dir}" \
            INFERENCE.OUTPUT_PATH "${output_dir}" \
            SYSTEM.NUM_GPUS ${num_gpus} \
            SOLVER.SAMPLES_PER_BATCH 1 \
            "${MODEL_EXTRA_ARGS[@]}" \
            "${EXTRA_INFER_ARGS[@]}" \
            
}

# ===================== Post-process + Eval =====================

run_postprocess() {
    local dataset="$1"
    local thresh="${2:-0.50}"
    _dataset_config "${dataset}" || return 1
    local output_dir="${WORK_DIR}/${dataset}"

    echo "==================== Post-process: ${dataset} (thresh=${thresh}) ===================="
    python scripts/post_process.py \
        "${output_dir}/result.h5" \
        "${output_dir}" \
        "${thresh}"

    python scripts/eval.py \
        eval_full \
        "${LABEL_PATH}" \
        "${output_dir}/segments.npy" \
        "${output_dir}/result.h5"
}

# ===================== Visualize =====================

run_vis() {
    local dataset="$1"
    _dataset_config "${dataset}" || return 1
    local output_dir="${WORK_DIR}/${dataset}"

    echo "==================== Visualize: ${dataset} ===================="
    python scripts/vis_slices.py \
        --image "${IMAGE_PATH}" \
        --label "${LABEL_PATH}" \
        --affinity "${output_dir}/result.h5" \
        --segment "${output_dir}/segments.npy" \
        --output_dir "${output_dir}/vis"
}

# ===================== Full Pipeline =====================
# Usage: eval_on DATASET [THRESH]

eval_on() {
    local dataset="$1"
    local thresh="${2:-0.50}"

    run_infer "${dataset}"
    run_postprocess "${dataset}" "${thresh}"
    run_vis "${dataset}"
}

postprocess_on() {
    local dataset="$1"
    local thresh="${2:-0.50}"

    run_postprocess "${dataset}" "${thresh}"
}

vis_on() {
    local dataset="$1"
    local thresh="${2:-0.50}"

    run_vis "${dataset}"
}
# ===================== Train =====================
# Usage: run_train [NUM_GPUS]
# Set TRAIN_BASE_CONFIG before calling
# Set TRAIN_GPUS="2,3,4,9" to pin specific GPUs
run_train() {
    local num_gpus="${1:-4}"
    local port=$(( ( RANDOM % 1000 ) + 2000 ))

    echo "==================== Train: ${script_name} (${num_gpus} GPUs) ===================="
    if [ -n "${TRAIN_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    else
        wait_gpus ${num_gpus}
    fi
    OMP_NUM_THREADS=8 \
    python -u -m torch.distributed.run \
        --nproc_per_node=${num_gpus} \
        --master_port=${port} \
        ${MAIN_PY} \
            --distributed \
            --config-base "${TRAIN_BASE_CONFIG}" \
            --config-file ${CONFIG} \
            MODEL.ARCHITECTURE ${ARCH} \
            DATASET.OUTPUT_PATH "${WORK_DIR}" \
            MONITOR.WANDB_NAME "${script_name}" \
            SOLVER.SAMPLES_PER_BATCH 1 \
            "${MODEL_EXTRA_ARGS[@]}"
}
