#!/bin/bash
# exps/_lib/env.sh — shell helpers used by every method script.
#
# Sourced by _lib/common.sh, so method scripts get these for free. Nothing here
# is machine-specific: it only shells out to nvidia-smi and coreutils.

# ---------- Wait until NUM_GPUS cards are idle, then pin them ----------
# Usage: wait_gpus [num_gpus] [poll_sec]
# "Idle" = under 2 GB in use. Exports CUDA_VISIBLE_DEVICES with the free ids.
# Set TRAIN_GPUS / INFER_GPUS in the method script to skip this and pin by hand.
wait_gpus() {
    local num_gpus=${1:-8}
    local refresh_interval=${2:-60}
    local gpu_info free_gpus gpu_count index total used

    echo "$(date '+%Y-%m-%d %H:%M:%S') - need ${num_gpus} GPU(s), polling every ${refresh_interval}s"

    _free_gpus() {
        gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
        local out=()
        while IFS=',' read -r index used; do
            if ((used < 2000)); then out+=("${index}"); fi
        done <<< "${gpu_info}"
        echo "${out[@]}"
    }

    free_gpus=($(_free_gpus))
    gpu_count=${#free_gpus[@]}
    while ((gpu_count < num_gpus)); do
        echo "$(date '+%Y-%m-%d %H:%M:%S') - free: ${free_gpus[*]:-none}"
        sleep "${refresh_interval}"
        free_gpus=($(_free_gpus))
        gpu_count=${#free_gpus[@]}
    done

    export CUDA_VISIBLE_DEVICES="$(IFS=, ; echo "${free_gpus[*]}")"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}

# ---------- Block until a file appears ----------
# Usage: wait_file <path> [poll_sec]
wait_file() {
    local filename="$1"
    local refresh_interval="${2:-60}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - waiting for ${filename}"
    while [ ! -f "${filename}" ]; do sleep "${refresh_interval}"; done
    echo "$(date '+%Y-%m-%d %H:%M:%S') - found ${filename}"
}

# ---------- Tee all further stdout/stderr into a log file ----------
# Usage: save_log <path>          (CLEAN_LOG=True truncates instead of appending)
save_log() {
    local log_file="$1"
    if [ "${CLEAN_LOG:-False}" = "True" ]; then : > "${log_file}"; fi
    exec &> >(tee -a "${log_file}")
    echo "========================================"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - logging to ${log_file}"
    echo "----------------------------------------"
}

# ---------- Echo the running script into the log ----------
# Records the exact hyperparameters a run used, next to its outputs.
print_self() {
    local script_file="${1:-$0}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - script: ${script_file}"
    echo "----------------------------------------"
    cat "${script_file}"
    echo "----------------------------------------"
}
