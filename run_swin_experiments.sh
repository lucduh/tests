#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_swin_experiments.sh TEACHER_CHECKPOINT TRAIN_JSON TEST_JSON [RESEARCH_ROOT]
#
# Optional environment variables:
#   DEPTHS="10 8 6 4"
#   LATENCY_BATCH_SIZES="1,2,4"
#   LATENCY_IMAGE_SIZES="1280x960 1920x1440 2560x1920"
#   TRAIN_IMAGE_SIZE="1280x960"

TEACHER_CHECKPOINT=${1:?teacher checkpoint is required}
TRAIN_JSON=${2:?training JSON is required}
TEST_JSON=${3:?held-out test JSON is required}
RESEARCH_ROOT=${4:-/domino/datasets/local/donut/research}

RUNS_DIR="$RESEARCH_ROOT/runs"
EVALUATION_DIR="$RESEARCH_ROOT/results/evaluation"
LATENCY_DIR="$RESEARCH_ROOT/results/encoder_latency"

DEPTHS=${DEPTHS:-"10 8 6 4"}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-42}
TRAIN_IMAGE_SIZE=${TRAIN_IMAGE_SIZE:-"1280x960"}
LATENCY_IMAGE_SIZES=${LATENCY_IMAGE_SIZES:-"1280x960"}
LATENCY_BATCH_SIZES=${LATENCY_BATCH_SIZES:-"1,2,4"}

TRAIN_IMAGE_HEIGHT=${TRAIN_IMAGE_SIZE%x*}
TRAIN_IMAGE_WIDTH=${TRAIN_IMAGE_SIZE#*x}

benchmark_model() {
    local name=$1
    local checkpoint=$2

    for image_size in $LATENCY_IMAGE_SIZES; do
        local image_height=${image_size%x*}
        local image_width=${image_size#*x}

        uv run python benchmark_encoder_latency.py "$checkpoint" \
            --name "$name" \
            --out "$LATENCY_DIR" \
            --batch-sizes "$LATENCY_BATCH_SIZES" \
            --image-height "$image_height" \
            --image-width "$image_width"
    done
}

uv run python evaluate_distillation.py \
    "$TEACHER_CHECKPOINT" \
    "$TEST_JSON" \
    --name teacher \
    --out "$EVALUATION_DIR"
benchmark_model teacher "$TEACHER_CHECKPOINT"

for depth in $DEPTHS; do
    for method in nodistill distill; do
        run_name="swin-stage2-d${depth}-${method}-s${SEED}"
        distillation_flag=()
        if [[ "$method" == "nodistill" ]]; then
            distillation_flag=(--no-distillation)
        fi

        uv run python train.py "$TEACHER_CHECKPOINT" \
            --data-json "$TRAIN_JSON" \
            --output-dir "$RUNS_DIR" \
            --run-name "$run_name" \
            --stage-depth "$depth" \
            --image-height "$TRAIN_IMAGE_HEIGHT" \
            --image-width "$TRAIN_IMAGE_WIDTH" \
            --max-epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --seed "$SEED" \
            "${distillation_flag[@]}"

        checkpoint="$RUNS_DIR/$run_name/best"
        uv run python evaluate_distillation.py \
            "$checkpoint" \
            "$TEST_JSON" \
            --name "$run_name" \
            --out "$EVALUATION_DIR"
        benchmark_model "$run_name" "$checkpoint"
    done
done
