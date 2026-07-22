import math
import time
from pathlib import Path

import pandas as pd
import torch
from donut import DonutModel
from donut.model import decoder_start_ids, init_shift_tokens_from_decoder
from transformers.modeling_outputs import BaseModelOutput

IMAGE_SIZES = [(1280, 960), (1920, 1440), (2560, 1920)]  # (height, width)
KEEP_RATIOS = [1.0, 0.75, 0.5, 0.25, 0.1]
MAX_NEW_TOKENS = 32
N_WARMUP = 1
N_RUNS = 3
SEED = 42
OUT_DIR = Path("results")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_ms(fn, device, n_runs):
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    times = [(time.perf_counter() - start) * 1000]

    for _ in range(1, n_runs):
        synchronize(device)
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)

    mean = sum(times) / len(times)
    std = math.sqrt(sum((value - mean) ** 2 for value in times) / len(times))
    return result, mean, std


def random_prune(
    hidden_states: torch.Tensor, keep_ratio: float, permutation: torch.Tensor
) -> torch.Tensor:
    """Keep a random subset while retaining the tokens' original spatial order."""
    n_keep = max(1, round(hidden_states.shape[1] * keep_ratio))
    indices = permutation[:n_keep].sort().values
    return hidden_states.index_select(1, indices)


def benchmark_generation(donut: DonutModel) -> list[dict]:
    model = donut.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    init_shift_tokens_from_decoder(model)
    start_ids = decoder_start_ids(model)
    records = []

    for height, width in IMAGE_SIZES:
        pixels = torch.rand((1, 3, height, width), device=device, dtype=dtype)

        def encode(pixel_values: torch.Tensor):
            with torch.inference_mode():
                return model.encoder(pixel_values, return_dict=True)

        encode(pixels)  # warm-up
        encoder_outputs, encoder_ms, encoder_std_ms = timed_ms(
            lambda: encode(pixels), device, N_RUNS
        )
        hidden_states = encoder_outputs.last_hidden_state
        n_tokens = hidden_states.shape[1]
        generator = torch.Generator(device=device).manual_seed(SEED)
        permutation = torch.randperm(n_tokens, generator=generator, device=device)
        size_records = []
        for keep_ratio in KEEP_RATIOS:
            pruned_states = random_prune(hidden_states, keep_ratio, permutation)
            pruned_outputs = BaseModelOutput(
                last_hidden_state=pruned_states  # ty: ignore[invalid-argument-type]
            )

            def generate():
                with torch.inference_mode():
                    return model.generate(  # ty: ignore[invalid-argument-type, missing-argument]
                        encoder_outputs=pruned_outputs,
                        decoder_input_ids=start_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        min_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                    )

            for _ in range(N_WARMUP):
                generate()
            _, generation_ms, generation_std_ms = timed_ms(generate, device, N_RUNS)
            size_records.append(
                {
                    "image_height": height,
                    "image_width": width,
                    "keep_ratio": keep_ratio,
                    "prune_ratio": 1.0 - keep_ratio,
                    "visual_tokens": pruned_states.shape[1],
                    "encoder_ms": encoder_ms,
                    "encoder_std_ms": encoder_std_ms,
                    "generation_ms": generation_ms,
                    "generation_std_ms": generation_std_ms,
                }
            )

        baseline_ms = size_records[0]["generation_ms"]
        for record in size_records:
            record["generation_speedup"] = baseline_ms / record["generation_ms"]
            # The encoder is not pruned in this experiment.
            record["estimated_end_to_end_ms"] = encoder_ms + record["generation_ms"]
            record["estimated_end_to_end_speedup"] = (
                encoder_ms + baseline_ms
            ) / record["estimated_end_to_end_ms"]
        records.extend(size_records)

        pixels = torch.empty(0, device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return records


def main() -> None:
    torch.manual_seed(SEED)
    donut = DonutModel.load(attention_backend="sdpa")
    records = benchmark_generation(donut)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_pickle(OUT_DIR / "token_pruning_benchmark.pkl")


if __name__ == "__main__":
    main()
