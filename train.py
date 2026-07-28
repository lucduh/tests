import argparse
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from donut import DonutModel
from donut.constants import DATA_DIR, DEFAULT_IMAGE_SIZE, DEFAULT_MAX_LENGTH
from donut.dataset import DonutDataset, load_samples
from donut.model import autocast
from donut.runio import run_meta, save_record
from encoder_student import EncoderStudentConfig, create_encoder_student
from research_paths import RESEARCH_RUNS_DIR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


@dataclass
class Config:
    teacher_path: Path
    data_json: Path = DATA_DIR / "train.json"
    val_split: float = 0.1
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
    distillation: bool = True
    stage_index: int = 2
    stage_depth: int = 8
    kept_blocks: tuple[int, ...] | None = None
    run_name: str = "swin-distillation"
    output_dir: Path = RESEARCH_RUNS_DIR


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def create_teacher(config):
    teacher = DonutModel.load(
        str(config.teacher_path),
        device=config.device,
        dtype=torch.float32,
        attention_backend=config.attention_backend,
    )
    teacher.prepare_for_training()
    teacher.set_image_size(*config.image_size)
    teacher.model.eval().requires_grad_(False)
    return teacher


def get_loaders(config, processor, generator):
    samples = load_samples(config.data_json)
    if len(samples) < 2:
        raise ValueError(
            "At least two samples are required for a train/validation split"
        )

    random.shuffle(samples)
    split = min(len(samples) - 1, max(1, int(len(samples) * (1 - config.val_split))))
    train_samples, val_samples = samples[:split], samples[split:]

    pin_memory = config.device.startswith("cuda")
    train_loader = DataLoader(
        DonutDataset(train_samples, processor, config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        DonutDataset(val_samples, processor, config.max_length),
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
        student_outputs = student(
            pixel_values=pixel_values, labels=labels, return_dict=True
        )
        task_loss = student_outputs.loss

        if teacher is None:
            return {"total_loss": task_loss, "task_loss": task_loss}

        with torch.no_grad():
            teacher_outputs = teacher(
                pixel_values=pixel_values, labels=labels, return_dict=True
            )

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


def train_one_epoch(teacher, student, optimizer, scheduler, loader, config):
    student.train()
    loss_sums = Counter()

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

    return {name: value / len(loader) for name, value in loss_sums.items()}


@torch.no_grad()
def evaluate(teacher, student, loader, config):
    student.eval()
    loss_sums = Counter()

    for batch in tqdm(loader, desc="validation", leave=False):
        pixel_values = batch["pixel_values"].to(config.device)
        labels = batch["labels"].to(config.device)
        losses = get_loss(teacher, student, pixel_values, labels, config)

        for name, loss in losses.items():
            loss_sums[name] += loss.item()

    return {name: value / len(loader) for name, value in loss_sums.items()}


def train(config):
    generator = seed_everything(config.seed)
    architecture = EncoderStudentConfig(
        stage_index=config.stage_index,
        depth=config.stage_depth,
        kept_blocks=config.kept_blocks,
    )
    student_donut = create_encoder_student(
        config.teacher_path,
        architecture,
        device=config.device,
        attention_backend=config.attention_backend,
        image_size=config.image_size,
    )
    teacher_donut = create_teacher(config) if config.distillation else None
    teacher = teacher_donut.model if teacher_donut is not None else None
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

    run_dir = config.output_dir / config.run_name
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, config.max_epochs + 1):
        train_losses = train_one_epoch(
            teacher, student, optimizer, scheduler, train_loader, config
        )
        val_losses = evaluate(teacher, student, val_loader, config)
        history.append(
            {"epoch": epoch, "train": train_losses, "validation": val_losses}
        )

        print(
            f"Epoch {epoch}: [TRAIN] "
            + ", ".join(f"{name}={value:.4f}" for name, value in train_losses.items())
        )
        print(
            f"Epoch {epoch}: [VAL]   "
            + ", ".join(f"{name}={value:.4f}" for name, value in val_losses.items())
        )

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            student_donut.save_checkpoint(run_dir / "best")

    student_donut.save_checkpoint(run_dir / "last")
    save_record(
        run_dir,
        "train.json",
        {
            "meta": run_meta(config.device, None, str(config.teacher_path)),
            "config": asdict(config),
            "best_val_loss": best_val_loss,
            "epochs": history,
        },
    )


def parse_blocks(value):
    return tuple(int(item) for item in value.split(",")) if value else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_path", type=Path)
    parser.add_argument("--data-json", type=Path, default=DATA_DIR / "train.json")
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RUNS_DIR)
    parser.add_argument("--run-name", default="swin-distillation")
    parser.add_argument("--stage-index", type=int, default=2)
    parser.add_argument("--stage-depth", type=int, default=8)
    parser.add_argument("--kept-blocks", type=parse_blocks)
    parser.add_argument("--no-distillation", action="store_true")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    train(
        Config(
            teacher_path=args.teacher_path,
            data_json=args.data_json,
            image_size=(args.image_height, args.image_width),
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta,
            max_epochs=args.max_epochs,
            seed=args.seed,
            device=args.device,
            precision=args.precision,
            attention_backend=args.attention_backend,
            distillation=not args.no_distillation,
            stage_index=args.stage_index,
            stage_depth=args.stage_depth,
            kept_blocks=args.kept_blocks,
            run_name=args.run_name,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
