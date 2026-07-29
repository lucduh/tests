# Donut encoder research

Research code for building, distilling, evaluating, and benchmarking smaller Donut Swin encoders.

Architecture details are documented in [`docs/encoder_architecture.md`](docs/encoder_architecture.md). The planned experiment sequence is in [`docs/encoder_experiment_plan.md`](docs/encoder_experiment_plan.md).

## Running the experiment sweep

```bash
LATENCY_IMAGE_SIZES="1280x960 1920x1440 2560x1920" \
LATENCY_BATCH_SIZES="1,2,4" \
./run_swin_experiments.sh \
  TEACHER_CHECKPOINT \
  TRAIN_JSON \
  HELD_OUT_TEST_JSON
```

The test JSON must be held out from training and model selection.

## Research storage

Research artifacts are kept separate from Donut's production training runs and general benchmarks:

```text
/domino/datasets/local/donut/research/
├── runs/
│   └── <run_name>/
│       ├── best/
│       ├── last/
│       └── train.json
└── results/
    ├── evaluation/
    │   ├── teacher.json
    │   └── <run_name>.json
    └── encoder_latency/
        ├── teacher__<height>x<width>.json
        └── <run_name>__<height>x<width>.json
```

These paths are defined once in `research_paths.py`.

### Runs

`runs/<run_name>/` owns the checkpoints and training history for one student configuration:

- `best/`: checkpoint with the lowest validation objective
- `last/`: checkpoint after the final epoch
- `train.json`: configuration, run metadata, epoch losses, and best validation loss

Run names include the architecture, training method, and seed. For example:

```text
swin-stage2-d6-distill-s42
swin-stage2-d6-nodistill-s42
```

Using a unique run name prevents results from different experiments from overwriting each other.

### Evaluation records

`evaluate_distillation.py` evaluates exactly one checkpoint and saves one record:

```json
{
  "meta": {
    "model_id": "...",
    "device": "cuda",
    "dtype": "bfloat16",
    "torch": "...",
    "transformers": "...",
    "timestamp": "..."
  },
  "config": {
    "name": "swin-stage2-d6-distill-s42",
    "checkpoint": ".../best",
    "data_json": "...",
    "documents": 100,
    "max_new_tokens": 128
  },
  "summary": {
    "strict_macro_field_f1": 0.95,
    "parameters": {
      "total": 123,
      "encoder": 70,
      "decoder": 53
    }
  }
}
```

The evaluator intentionally knows nothing about teacher/student roles. The experiment script assigns a stable model name, and the notebook uses that name to join records.

### Encoder-latency records

`benchmark_encoder_latency.py` saves one file per model and image resolution. Each file contains all requested batch sizes:

```json
{
  "meta": {},
  "config": {
    "name": "swin-stage2-d6-distill-s42",
    "checkpoint": ".../best",
    "image_height": 1280,
    "image_width": 960,
    "batch_sizes": [1, 2, 4],
    "warmup": 3,
    "runs": 10
  },
  "results": [
    {
      "batch_size": 1,
      "encoder_latency_ms": 30.5,
      "latency_ms_runs": [30.1, 30.8]
    }
  ]
}
```

`encoder_latency_ms` is the latency of the complete batch. Raw run latencies are retained so later analysis is not limited to a precomputed average.

## Record writing

Training, evaluation, and latency records reuse Donut's existing helpers:

```python
from donut.runio import run_meta, save_record
```

- `run_meta` records model, device, dtype, library versions, and timestamp.
- `save_record` creates the output directory and writes a self-describing JSON record.
- Separate files allow partial sweeps and additional resolutions to accumulate without creating a monolithic result file.

## Analysis notebook

Open:

```text
notebooks/distillation_results.ipynb
```

The notebook reads the evaluation and encoder-latency directories and shows:

- Strict macro field F1
- Total, encoder, and decoder parameter counts
- Parameter reduction from the teacher
- Encoder latency across batch sizes and resolutions
- Speedup relative to the teacher
- Quality/latency trade-offs

The comparison point can be changed in the notebook:

```python
SELECTED_IMAGE_SIZE = (1280, 960)
SELECTED_BATCH_SIZE = 1
```

## Individual commands

Train a student:

```bash
uv run python train.py TEACHER_CHECKPOINT \
  --stage-depth 6 \
  --run-name swin-stage2-d6-distill-s42
```

Disable distillation:

```bash
uv run python train.py TEACHER_CHECKPOINT \
  --stage-depth 6 \
  --no-distillation \
  --run-name swin-stage2-d6-nodistill-s42
```

Evaluate one model:

```bash
uv run python evaluate_distillation.py CHECKPOINT HELD_OUT_TEST_JSON \
  --name MODEL_NAME
```

Benchmark one encoder:

```bash
uv run python benchmark_encoder_latency.py CHECKPOINT \
  --name MODEL_NAME \
  --batch-sizes 1,2,4 \
  --image-height 1280 \
  --image-width 960
```
