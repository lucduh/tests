# Encoder compression experiment plan

## Objective

Find a Donut Swin encoder on the quality, latency, and model-size Pareto frontier.

The primary quality metric is strict macro field F1. Compression is represented by parameter count, and runtime performance is measured using encoder latency on the target GPU.

Every comparison must use the same:

- Trained teacher checkpoint
- Training and held-out test data
- Random seed and train/validation split
- Attention backend and precision
- Image resolution and batch size for latency comparisons

Parameter count does not depend on image resolution, but latency and quality can. Broad architecture screening should use the small resolution; finalists must be evaluated at all target resolutions.

## Current baseline

The original Swin stage depths are:

```text
[2, 2, 14, 2]
```

The current student method removes blocks from stage 2 while preserving regular/shifted block pairs. The initial sweep compares stage-2 depths `10, 8, 6, 4`, each with:

- Task-only training
- Task loss plus logit and final encoder-state distillation

This establishes whether distillation is necessary and where quality first declines.

## Phase 1: complete stage-2 depth pruning

### Test depth 2

Add the aggressive architecture:

```text
[2, 2, 2, 2]
```

This retains one of the seven original regular/shifted block pairs and removes 12 of the 14 stage-2 blocks.

### Search block selections

Depth alone does not identify an architecture. With seven original block pairs, the number of pair-preserving choices is:

| Student depth | Pairs retained | Combinations |
|---:|---:|---:|
| 6 | 3 | 35 |
| 4 | 2 | 21 |
| 2 | 1 | 7 |

Do not fully train every combination.

#### Screening

For each candidate selection:

1. Build the pruned student from the trained checkpoint.
2. Evaluate it without training on a fixed calibration subset.
3. Rank it using task loss or final encoder-state distance from the teacher.
4. Retain only the best few candidates at each depth.

Candidates with the same depth have the same parameter count and nearly identical latency, so screening only needs to rank quality.

#### Full training

At each depth, train:

- The best screened selection
- The uniformly spaced selection
- One contrasting selection, such as early-only, middle-only, or late-only

Compare task-only and distilled training for the finalists rather than every screened candidate.

## Phase 2: prune stage 3

Stage 2 is long, but stage 3 has hidden dimension 1024 and therefore contains many parameters per block. Stage-2-only pruning is capped while both stage-3 blocks remain.

Starting from the best stage-2 students, test:

```text
[2, 2, 4, 2]
[2, 2, 2, 2]
[2, 2, 4, 1]
[2, 2, 2, 1]
```

For a one-block stage 3, independently test:

```text
keep stage-3 block 0
keep stage-3 block 1
```

The regular/shifted pair constraint cannot be preserved with one block, so both alternatives need empirical evaluation.

The main aggressive candidate is:

```text
[2, 2, 2, 1]
```

It attacks both the long stage and the parameter-heavy final stage while preserving the 1024-dimensional encoder/decoder interface.

## Phase 3: prune stages 0 and 1

Early-stage blocks contain fewer parameters but process many more spatial tokens. They may matter more for latency than their parameter counts suggest.

Starting from the best phase-2 architecture, test one early-stage change at a time:

```text
[1, 2, d2, 1]
[2, 1, d2, 1]
[1, 1, d2, 1]
```

Measure real latency rather than inferring speed from removed parameters.

## Phase 4: reduce FFN width

The feed-forward network contains most parameters inside each Swin block:

```text
dimension → 4 × dimension → dimension
```

After selecting stage depths, test MLP expansion ratios:

```text
4
3
2
```

For pretrained initialization, score each intermediate FFN neuron using both adjacent projections, for example:

```text
norm(fc1 neuron) × norm(fc2 neuron)
```

Retain the highest-scoring neurons and physically construct smaller linear layers. Masking neurons without changing matrix shapes does not provide dense-kernel acceleration.

## Phase 5: strengthen distillation when needed

The current objective is:

```text
task cross-entropy
+ decoder-logit KL
+ final encoder-state MSE
```

Keep this baseline until aggressive architectures begin losing quality. Then test additions individually.

### Stage-output distillation

Match the output of each student stage to the corresponding teacher stage. This supervises the full encoder hierarchy while preserving a simple layer mapping.

### Retained-block distillation

For a retained student block copied from a particular teacher block, match its output to that teacher block's output. This is more complex and should only be attempted if stage-output matching is insufficient.

Every stronger distillation method must be compared with task-only training on the same student architecture. Otherwise architecture redundancy and distillation benefit cannot be separated.

## Phase 6: reduce encoder width

Once depth and FFN pruning are understood, build a narrower encoder. For example:

```text
Original dimensions: [128, 256, 512, 1024]
Student dimensions:  [96, 192, 384, 768]
```

Add a final learned projection:

```text
encoder 768 → decoder interface 1024
```

This keeps MBart unchanged while reducing attention, FFN, and patch-merging dimensions throughout the encoder. Width reduction is more invasive than structured depth pruning and should not be mixed into early experiments.

## Resolution and latency validation

Use the small resolution for broad screening. Promote only the teacher and Pareto-optimal students to the full resolution grid:

```text
1280×960
1920×1440
2560×1920
```

For each resolution, benchmark the operational batch sizes, initially:

```text
1, 2, 4
```

Compare every student with the teacher at the same resolution and batch size. The checkpoint's processor size must be explicitly changed for each resolution; supplying a larger source image alone is not sufficient.

Large images may change quality as well as latency. A shallower encoder has fewer shifted-window transformations over a larger spatial grid, while the higher resolution also makes small text more legible.

If a small-resolution-trained finalist loses quality at larger resolutions, progressively fine-tune it at medium and then large resolution rather than repeating the complete architecture search at full resolution.

## Experiment sequence

1. Complete the initial depth `10, 8, 6, 4` sweep.
2. Screen and train depth-2 block-pair choices.
3. Screen alternative block selections for depths 6 and 4.
4. Select the best stage-2 architecture at each parameter budget.
5. Test stage-3 depth 1, including both possible retained blocks.
6. Test early-stage depth reductions one stage at a time.
7. Reduce FFN width on the best depth architecture.
8. Add stronger distillation only when the current objective stops preserving F1.
9. Attempt encoder-width reduction after the structured depth/FFN Pareto frontier is established.
10. Validate finalists at all image resolutions and batch sizes.

## Repetition and selection

Use one seed for broad screening. After identifying the best two or three candidates, repeat them with at least two additional seeds.

Select models using the held-out trade-off among:

- Strict macro field F1
- Encoder latency at the deployment resolution and batch size
- Encoder and total parameter count

Do not combine these into an arbitrary single score. Keep the three axes visible and choose from the Pareto frontier.
