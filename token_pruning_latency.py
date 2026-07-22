import time
from pathlib import Path

import pandas as pd
import torch
from donut import DonutModel
from donut.model import decoder_start_ids, init_shift_tokens_from_decoder
from transformers.modeling_outputs import BaseModelOutput

IMAGE_SIZES = [(1280, 960), (1920, 1440), (2560, 1920)]  # (height, width)
BATCH_SIZES = [1, 2, 4]
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

    return result, sum(times) / len(times)


def peak_memory_mb(fn, device):
    if device.type != "cuda":
        return None

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024**2


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
    records = []

    for height, width in IMAGE_SIZES:
        for batch_size in BATCH_SIZES:
            start_ids = decoder_start_ids(model, batch_size)
            pixels = torch.rand(
                (batch_size, 3, height, width), device=device, dtype=dtype
            )

            def encode():
                with torch.inference_mode():
                    return model.encoder(pixels, return_dict=True)

            encode()
            encoder_outputs, encoder_ms = timed_ms(encode, device, N_RUNS)
            hidden_states = encoder_outputs.last_hidden_state
            n_tokens = hidden_states.shape[1]
            generator = torch.Generator(device=device).manual_seed(SEED)
            permutation = torch.randperm(n_tokens, generator=generator, device=device)

            for keep_ratio in KEEP_RATIOS:
                pruned_states = random_prune(hidden_states, keep_ratio, permutation)
                pruned_outputs = BaseModelOutput(
                    last_hidden_state=pruned_states  # ty: ignore[invalid-argument-type]
                )

                def generate(n_tokens):
                    with torch.inference_mode():
                        return model.generate(  # ty: ignore[invalid-argument-type, missing-argument]
                            encoder_outputs=pruned_outputs,
                            decoder_input_ids=start_ids,
                            max_new_tokens=n_tokens,
                            min_new_tokens=n_tokens,
                            do_sample=False,
                            use_cache=True,
                        )

                for _ in range(N_WARMUP):
                    generate(1)
                    generate(MAX_NEW_TOKENS)

                _, first_token_ms = timed_ms(lambda: generate(1), device, N_RUNS)
                _, generation_ms = timed_ms(
                    lambda: generate(MAX_NEW_TOKENS), device, N_RUNS
                )
                peak_mb = peak_memory_mb(lambda: generate(MAX_NEW_TOKENS), device)

                records.append(
                    {
                        "image_height": height,
                        "image_width": width,
                        "batch_size": batch_size,
                        "keep_ratio": keep_ratio,
                        "visual_tokens": pruned_states.shape[1],
                        "encoder_ms": encoder_ms,
                        "first_token_decoder_ms": first_token_ms,
                        "generation_ms": generation_ms,
                        "peak_memory_mb": peak_mb,
                    }
                )

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
