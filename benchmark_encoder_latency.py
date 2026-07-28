import argparse
import time
from pathlib import Path

import torch
from donut import DonutModel
from donut.constants import DEFAULT_IMAGE_SIZE
from donut.runio import parse_ints, run_meta, save_record
from research_paths import ENCODER_LATENCY_DIR


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(encoder, pixel_values, device, warmup, runs):
    with torch.inference_mode():
        for _ in range(warmup):
            encoder(pixel_values)

        times = []
        for _ in range(runs):
            synchronize(device)
            start = time.perf_counter()
            encoder(pixel_values)
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000)

    return times


def default_model_name(checkpoint):
    path = Path(checkpoint)
    return path.parent.name if path.name in {"best", "last"} else path.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", type=Path, default=ENCODER_LATENCY_DIR)
    parser.add_argument("--name")
    parser.add_argument("--batch-sizes", type=parse_ints, default=[1, 2, 4])
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    donut = DonutModel.load(
        args.checkpoint,
        device=args.device,
        attention_backend="sdpa",
    )
    donut.set_image_size(args.image_height, args.image_width)
    model = donut.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    results = []
    for batch_size in args.batch_sizes:
        pixel_values = torch.rand(
            batch_size,
            3,
            args.image_height,
            args.image_width,
            device=device,
            dtype=dtype,
        )
        latency_ms_runs = measure_latency(
            model.encoder,
            pixel_values,
            device,
            args.warmup,
            args.runs,
        )
        results.append(
            {
                "batch_size": batch_size,
                "encoder_latency_ms": sum(latency_ms_runs) / len(latency_ms_runs),
                "latency_ms_runs": latency_ms_runs,
            }
        )

    name = args.name or default_model_name(args.checkpoint)
    record = {
        "meta": run_meta(args.device, None, args.checkpoint),
        "config": {
            "name": name,
            "checkpoint": args.checkpoint,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "results": results,
    }
    filename = f"{name}__{args.image_height}x{args.image_width}.json"
    save_record(args.out, filename, record)


if __name__ == "__main__":
    main()
