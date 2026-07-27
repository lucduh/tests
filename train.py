import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from donut import DonutModel
from donut.constants import (
    DATA_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_LENGTH,
    TRAIN_RUNS_DIR,
)
from donut.dataset import DonutDataset, load_samples
from donut.model import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


@dataclass
class Config:
    data_json: Path = DATA_DIR / "train.json"
    val_split: float = 0.1
    teacher_path: Path = Path("test")
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    temperature: float = 2.0
    alpha: float = 1.0
    beta: float = 0.5
    max_epochs: int = 5
    seed: int = 42
    device: str = "cuda"
    precision: str = "bf16"
    attention_backend: str = "sdpa"
    run_name: str = "Testing"
    output_dir: Path = TRAIN_RUNS_DIR
    stage_2_kept_blocks: list[int] = field(
        default_factory=lambda: [0, 1, 4, 5, 8, 9, 12, 13]
    )


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_model(checkpoint, config):
    donut = DonutModel.load(
        str(checkpoint),
        device=config.device,
        dtype=torch.float32,
        attention_backend=config.attention_backend,
    )
    donut.prepare_for_training()
    donut.set_image_size(*config.image_size)
    return donut


def create_student_teacher_models(config):
    teacher = load_model(config.teacher_path, config)
    student = load_model(config.teacher_path, config)

    stage = student.model.encoder.encoder.layers[2]
    kept = config.stage_2_kept_blocks
    if kept != sorted(set(kept)) or not kept or kept[-1] >= len(stage.blocks):
        raise ValueError(f"Invalid stage-2 block indices: {kept}")

    stage.blocks = torch.nn.ModuleList([stage.blocks[index] for index in kept])
    depths = list(student.model.encoder.config.depths)
    depths[2] = len(kept)
    student.model.encoder.config.depths = depths
    student.model.config.encoder.depths = depths

    teacher.model.eval().requires_grad_(False)
    return teacher, student


def get_loaders(config, processor, generator):
    samples = load_samples(config.data_json)
    if len(samples) < 2:
        raise ValueError(
            "At least two samples are required for a train/validation split"
        )

    random.shuffle(samples)
    split = min(len(samples) - 1, max(1, int(len(samples) * (1 - config.val_split))))
    train_samples, val_samples = samples[:split], samples[split:]

    train_dataset = DonutDataset(train_samples, processor, config.max_length)
    val_dataset = DonutDataset(val_samples, processor, config.max_length)
    pin_memory = config.device.startswith("cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def masked_kl_div(student_logits, teacher_logits, labels, temperature):
    valid = labels.ne(-100)
    student_log_probs = torch.nn.functional.log_softmax(
        student_logits[valid].float() / temperature, dim=-1
    )
    teacher_probs = torch.nn.functional.softmax(
        teacher_logits[valid].float() / temperature, dim=-1
    )
    return (
        torch.nn.functional.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean"
        )
        * temperature**2
    )


def encoder_distillation_loss(student_states, teacher_states):
    student_states = torch.nn.functional.layer_norm(
        student_states.float(), student_states.shape[-1:]
    )
    teacher_states = torch.nn.functional.layer_norm(
        teacher_states.float(), teacher_states.shape[-1:]
    )
    return torch.nn.functional.mse_loss(student_states, teacher_states)


def get_loss(teacher, student, pixel_values, labels, config):
    with autocast(config.device, config.precision):
        with torch.no_grad():
            teacher_outputs = teacher(
                pixel_values=pixel_values, labels=labels, return_dict=True
            )
        student_outputs = student(
            pixel_values=pixel_values, labels=labels, return_dict=True
        )

    task_loss = student_outputs.loss
    logit_loss = masked_kl_div(
        student_outputs.logits,
        teacher_outputs.logits,
        labels,
        config.temperature,
    )
    encoder_loss = encoder_distillation_loss(
        student_outputs.encoder_last_hidden_state,
        teacher_outputs.encoder_last_hidden_state,
    )
    total_loss = task_loss + config.alpha * logit_loss + config.beta * encoder_loss

    return {
        "total_loss": total_loss,
        "task_loss": task_loss,
        "logit_loss": logit_loss,
        "encoder_loss": encoder_loss,
    }


def mean_losses(loss_sums, n_batches):
    return {name: value / n_batches for name, value in loss_sums.items()}


def train_one_epoch(teacher, student, optimizer, scheduler, loader, config):
    student.train()
    loss_sums = {name: 0.0 for name in DISTILLATION_LOSSES}

    for batch in tqdm(loader, desc="training", leave=False):
        pixel_values = batch["pixel_values"].to(config.device)
        labels = batch["labels"].to(config.device)

        optimizer.zero_grad()
        losses = get_loss(teacher, student, pixel_values, labels, config)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        for name, loss in losses.items():
            loss_sums[name] += loss.item()

    return mean_losses(loss_sums, len(loader))


@torch.no_grad()
def evaluate(teacher, student, loader, config):
    teacher.eval()
    student.eval()
    loss_sums = {name: 0.0 for name in DISTILLATION_LOSSES}

    for batch in tqdm(loader, desc="validation", leave=False):
        pixel_values = batch["pixel_values"].to(config.device)
        labels = batch["labels"].to(config.device)
        losses = get_loss(teacher, student, pixel_values, labels, config)

        for name, loss in losses.items():
            loss_sums[name] += loss.item()

    return mean_losses(loss_sums, len(loader))


DISTILLATION_LOSSES = (
    "total_loss",
    "task_loss",
    "logit_loss",
    "encoder_loss",
)


def main(config):
    generator = seed_everything(config.seed)
    teacher_donut, student_donut = create_student_teacher_models(config)
    teacher = teacher_donut.model
    student = student_donut.model

    train_loader, val_loader = get_loaders(config, student_donut.processor, generator)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=len(train_loader) * config.max_epochs,
    )

    best_val_loss = float("inf")
    for epoch in range(1, config.max_epochs + 1):
        train_losses = train_one_epoch(
            teacher, student, optimizer, scheduler, train_loader, config
        )
        val_losses = evaluate(teacher, student, val_loader, config)

        train_summary = ", ".join(
            f"{name}={value:.4f}" for name, value in train_losses.items()
        )
        val_summary = ", ".join(
            f"{name}={value:.4f}" for name, value in val_losses.items()
        )
        print(f"Epoch {epoch}: train {train_summary}")
        print(f"Epoch {epoch}: val   {val_summary}")

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            student_donut.save_checkpoint(config.output_dir / config.run_name / "best")

    student_donut.save_checkpoint(config.output_dir / config.run_name / "last")


if __name__ == "__main__":
    main(Config())
