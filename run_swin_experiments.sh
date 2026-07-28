#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_swin_experiments.sh TEACHER_CHECKPOINT TRAIN_JSON TEST_JSON [OUTPUT_DIR]
#
# The test JSON must be held out from training. Every run uses the same split,
# seed, optimizer settings, and evaluation metric.

TEACHER_CHECKPOINT=${1:?teacher checkpoint is required}
TRAIN_JSON=${2:?training JSON is required}
TEST_JSON=${3:?held-out test JSON is required}
OUTPUT_DIR=${4:-results/swin_experiments}

DEPTHS=${DEPTHS:-"10 8 6 4"}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-42}

mkdir -p "$OUTPUT_DIR/evaluations"

uv run python evaluate_distillation.py \
    "$TEACHER_CHECKPOINT" \
    "$TEST_JSON" \
    --output "$OUTPUT_DIR/evaluations/teacher.json"

for depth in $DEPTHS; do
    for method in nodistill distill; do
        run_name="swin-stage2-d${depth}-${method}-s${SEED}"
        distillation_flag=()
        if [[ "$method" == "nodistill" ]]; then
            distillation_flag=(--no-distillation)
        fi

        uv run python train.py "$TEACHER_CHECKPOINT" \
            --data-json "$TRAIN_JSON" \
            --output-dir "$OUTPUT_DIR" \
            --run-name "$run_name" \
            --stage-depth "$depth" \
            --max-epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --seed "$SEED" \
            "${distillation_flag[@]}"

        uv run python evaluate_distillation.py \
            "$OUTPUT_DIR/$run_name/best" \
            "$TEST_JSON" \
            --output "$OUTPUT_DIR/evaluations/$run_name.json"
    done
done
