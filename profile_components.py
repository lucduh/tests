import time
from pathlib import Path

import pandas as pd
import torch
from donut import DonutModel
from donut.model import decoder_start_ids, init_shift_tokens_from_decoder

IMAGE_SIZE = (1280, 960)  # (height, width)
BATCH_SIZE = 1
N_WARMUP = 2
N_RUNS = 10
OUT_PATH = Path("results/component_profile.pkl")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device):
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            fn()

        times = []
        result = None
        for _ in range(N_RUNS):
            synchronize(device)
            start = time.perf_counter()
            result = fn()
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000)

    return result, sum(times) / len(times)


def profile_encoder(model, pixels, device):
    encoder = model.encoder
    records = []

    (hidden_states, dimensions), latency = timed(
        lambda: encoder.embeddings(pixels), device
    )
    records.append({"component": "encoder_embeddings", "latency_ms": latency})

    for index, stage in enumerate(encoder.encoder.layers):
        outputs, latency = timed(lambda: stage(hidden_states, dimensions), device)
        hidden_states = outputs[0]
        dimensions = outputs[2][-2:]
        records.append({"component": f"encoder_stage_{index}", "latency_ms": latency})

    return hidden_states, records


def profile_decoder(model, encoder_hidden_states, device):
    bart = model.decoder
    decoder = bart.model.decoder
    input_ids = decoder_start_ids(model, BATCH_SIZE)
    records = []

    def embed():
        hidden = decoder.embed_tokens(input_ids)
        hidden = hidden + decoder.embed_positions(input_ids)
        return decoder.layernorm_embedding(hidden)

    hidden_states, latency = timed(embed, device)
    records.append({"component": "decoder_embeddings", "latency_ms": latency})

    for index, layer in enumerate(decoder.layers):

        def self_attention():
            normalized = layer.self_attn_layer_norm(hidden_states)
            output, _ = layer.self_attn(hidden_states=normalized)
            return hidden_states + output

        hidden_states, latency = timed(self_attention, device)
        records.append(
            {
                "component": f"decoder_layer_{index}_self_attention",
                "latency_ms": latency,
            }
        )

        def cross_attention():
            normalized = layer.encoder_attn_layer_norm(hidden_states)
            output, _ = layer.encoder_attn(
                hidden_states=normalized,
                key_value_states=encoder_hidden_states,
            )
            return hidden_states + output

        hidden_states, latency = timed(cross_attention, device)
        records.append(
            {
                "component": f"decoder_layer_{index}_cross_attention",
                "latency_ms": latency,
            }
        )

        def feed_forward():
            normalized = layer.final_layer_norm(hidden_states)
            output = layer.fc2(layer.activation_fn(layer.fc1(normalized)))
            return hidden_states + output

        hidden_states, latency = timed(feed_forward, device)
        records.append(
            {
                "component": f"decoder_layer_{index}_feed_forward",
                "latency_ms": latency,
            }
        )

    _, latency = timed(lambda: bart.lm_head(decoder.layer_norm(hidden_states)), device)
    records.append({"component": "decoder_lm_head", "latency_ms": latency})
    return records


def main():
    donut = DonutModel.load(attention_backend="sdpa")
    model = donut.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    init_shift_tokens_from_decoder(model)

    height, width = IMAGE_SIZE
    pixels = torch.rand((BATCH_SIZE, 3, height, width), device=device, dtype=dtype)

    encoder_hidden_states, encoder_records = profile_encoder(model, pixels, device)
    decoder_records = profile_decoder(model, encoder_hidden_states, device)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(encoder_records + decoder_records).to_pickle(OUT_PATH)


if __name__ == "__main__":
    main()
