# Donut Swin encoder and student architecture

## Original encoder

Donut uses a hierarchical Swin Transformer with approximately this configuration:

```text
Patch size:       4×4
Stage depths:     [2, 2, 14, 2]
Stage dimensions: [128, 256, 512, 1024]
Window size:      10×10
```

The image follows this path:

```text
Image
  ↓ patch embedding, stride 4
Stage 0: 2 blocks, dimension 128
  ↓ patch merging
Stage 1: 2 blocks, dimension 256
  ↓ patch merging
Stage 2: 14 blocks, dimension 512
  ↓ patch merging
Stage 3: 2 blocks, dimension 1024
  ↓
Final visual tokens
```

`stage_index=2` selects the third stage, which contains 14 blocks. It is the long and expensive stage, but it is not technically the final stage: stage 3 follows it.

## Spatial resolutions

For a `1280×960` input:

| Location | Grid | Tokens | Dimension |
|---|---:|---:|---:|
| Patch embedding | `320×240` | 76,800 | 128 |
| Stage 1 | `160×120` | 19,200 | 256 |
| Stage 2 | `80×60` | 4,800 | 512 |
| Stage 3/final | `40×30` | 1,200 | 1024 |

Stage 2 runs 14 blocks over 4,800 tokens. For a `2560×1920` input, it processes 19,200 tokens. Swin attention is restricted to local windows, but every stage-2 token still passes through all 14 attention and feed-forward blocks.

## One Swin block

A Swin block approximately performs:

```text
hidden states
    ↓ LayerNorm
window self-attention
    ↓ residual connection
    ↓ LayerNorm
feed-forward network
    ↓ residual connection
```

The stage-2 hidden dimension is 512, and its feed-forward network is approximately:

```text
512 → 2048 → 512
```

One stage-2 block has roughly 3.15 million parameters:

- Q/K/V and attention output projections: about 1.05 million
- Feed-forward projections: about 2.10 million
- Layer norms and biases: relatively small

Removing six blocks therefore removes roughly 19 million parameters.

## Shifted and non-shifted blocks

Swin blocks alternate between regular and shifted windows:

```text
block 0: regular windows
block 1: shifted windows
block 2: regular windows
block 3: shifted windows
...
```

Regular windows communicate within local regions. Shifted windows allow information to cross the previous window boundaries. The student therefore retains complete pairs:

```text
[regular block, shifted block]
```

## `EncoderStudentConfig`

```python
@dataclass
class EncoderStudentConfig:
    stage_index: int = 2
    depth: int = 8
    kept_blocks: tuple[int, ...] | None = None
```

### `stage_index`

Selects the Swin stage using zero-based indexing:

```text
0 → original depth 2
1 → original depth 2
2 → original depth 14
3 → original depth 2
```

The default is stage 2 because it is the long stage.

### `depth`

Specifies how many blocks the selected stage should retain. For example:

```python
depth=8
```

changes the architecture from:

```text
[2, 2, 14, 2]
```

to:

```text
[2, 2, 8, 2]
```

`depth` is used only when `kept_blocks` is not supplied. Automatic selection currently requires an even depth so complete regular/shifted pairs can be retained.

### `kept_blocks`

Allows the original blocks to be selected explicitly:

```python
EncoderStudentConfig(
    stage_index=2,
    kept_blocks=(0, 1, 6, 7, 12, 13),
)
```

When supplied, `kept_blocks` overrides `depth`. This example creates a six-block third stage using three complete block pairs.

## Automatic block selection

`uniformly_spaced_block_pairs` treats the original 14 blocks as seven pairs:

```text
pair 0: blocks 0, 1
pair 1: blocks 2, 3
pair 2: blocks 4, 5
pair 3: blocks 6, 7
pair 4: blocks 8, 9
pair 5: blocks 10, 11
pair 6: blocks 12, 13
```

It retains pairs spread across the original stage rather than simply keeping the first blocks.

For depth 8:

```text
pair indices:  0, 2, 4, 6
block indices: 0, 1, 4, 5, 8, 9, 12, 13
```

For depth 6:

```text
pair indices:  0, 3, 6
block indices: 0, 1, 6, 7, 12, 13
```

For depth 4:

```text
pair indices:  0, 6
block indices: 0, 1, 12, 13
```

This preserves transformations from the beginning, middle, and end of the trained stage.

## How the student is created

### 1. Load the trained checkpoint

```python
student = DonutModel.load(...)
```

The student starts from trained teacher weights rather than random weights.

### 2. Prepare the model

```python
student.prepare_for_training()
student.set_image_size(*image_size)
```

This configures the tokenizer vocabulary, task token, decoder embeddings, and image resolution.

### 3. Access the selected stage

```python
stages = student.model.encoder.encoder.layers
stage = stages[architecture.stage_index]
```

`encoder.encoder.layers` contains the four Swin stages. The `cast(Any, ...)` in the implementation has no runtime effect; it only helps the type checker understand Hugging Face's dynamic model structure.

### 4. Physically replace the blocks

```python
stage.blocks = torch.nn.ModuleList(
    [stage.blocks[index] for index in kept]
)
```

This is structured pruning. Removed blocks are no longer part of the model, so their parameters, forward operations, and gradients disappear. `ModuleList` ensures PyTorch registers the retained blocks and their parameters.

### 5. Update the saved configuration

```python
depths[architecture.stage_index] = len(kept)
student.model.encoder.config.depths = depths
student.model.config.encoder.depths = depths
```

This ensures that loading the saved checkpoint reconstructs the smaller architecture rather than the original 14-block stage.

## What does not change

This pruning method does not change:

- Image resolution
- Number of visual tokens
- Window size
- Hidden dimensions
- Patch merging
- Final visual-token shape
- MBart decoder
- Vocabulary

The output remains:

```text
batch × final_visual_tokens × 1024
```

The unchanged output shape allows the student to use the original decoder and makes direct final-state distillation against the teacher possible.

## Expected acceleration

With depth 8, stage 2 changes from 14 attention and feed-forward blocks to 8. This removes approximately 43% of stage-2 block computation:

```text
1 - 8/14 ≈ 42.9%
```

It does not remove 43% of total encoder latency because patch embedding and the other stages remain unchanged. Unlike final visual-token pruning, however, this removes computation from inside the encoder and should directly reduce encoder latency.
