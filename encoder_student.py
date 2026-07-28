from dataclasses import dataclass
from typing import Any, cast

import torch
from donut import DonutModel


@dataclass
class EncoderStudentConfig:
    stage_index: int = 2
    depth: int = 8
    kept_blocks: tuple[int, ...] | None = None


def uniformly_spaced_block_pairs(original_depth, student_depth):
    """Select complete non-shifted/shifted Swin block pairs."""
    if student_depth <= 0 or student_depth > original_depth:
        raise ValueError(f"Student depth must be between 1 and {original_depth}")
    if original_depth % 2 or student_depth % 2:
        raise ValueError(
            "Swin pair selection requires even original and student depths"
        )

    original_pairs = original_depth // 2
    kept_pairs = student_depth // 2
    if kept_pairs == 1:
        pair_indices = [0]
    else:
        pair_indices = [
            round(index * (original_pairs - 1) / (kept_pairs - 1))
            for index in range(kept_pairs)
        ]

    return tuple(
        block_index
        for pair_index in pair_indices
        for block_index in (2 * pair_index, 2 * pair_index + 1)
    )


def create_encoder_student(
    checkpoint,
    architecture,
    *,
    device,
    attention_backend,
    image_size,
):
    student = DonutModel.load(
        str(checkpoint),
        device=device,
        dtype=torch.float32,
        attention_backend=attention_backend,
    )
    student.prepare_for_training()
    student.set_image_size(*image_size)

    encoder = cast(Any, student.model.encoder)
    stages = encoder.encoder.layers
    if architecture.stage_index >= len(stages):
        raise ValueError(f"Invalid encoder stage index: {architecture.stage_index}")

    stage = stages[architecture.stage_index]
    kept = architecture.kept_blocks or uniformly_spaced_block_pairs(
        len(stage.blocks), architecture.depth
    )
    if kept != tuple(sorted(set(kept))) or not kept or kept[-1] >= len(stage.blocks):
        raise ValueError(f"Invalid block indices: {kept}")

    stage.blocks = torch.nn.ModuleList([stage.blocks[index] for index in kept])
    depths = list(student.model.encoder.config.depths)
    depths[architecture.stage_index] = len(kept)
    student.model.encoder.config.depths = depths
    student.model.config.encoder.depths = depths
    return student
