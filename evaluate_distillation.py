import argparse
import json
from pathlib import Path

import torch
from donut import DonutModel
from donut.constants import DEFAULT_MAX_NEW_TOKENS
from donut.dataset import load_samples, parse_prediction
from donut.metrics import field_stats
from PIL import Image
from tqdm import tqdm


def parameter_counts(model):
    def count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    return {
        "total": count(model),
        "encoder": count(model.encoder),
        "decoder": count(model.decoder),
    }


def ground_truth(sample):
    return {
        field["field_name"].split("/")[-1]: field.get("annotator_text", "").strip()
        for field in sample["fields"]
        if field.get("annotator_text", "").strip()
    }


def evaluate(checkpoint, samples, device, max_new_tokens):
    donut = DonutModel.load(checkpoint, device=device, attention_backend="sdpa")
    dtype = next(donut.model.parameters()).dtype
    results = []

    for sample in tqdm(samples, desc=Path(checkpoint).name):
        image = Image.open(sample["image"]).convert("RGB")
        pixels = donut.processor(image, return_tensors="pt").pixel_values.to(
            device=device, dtype=dtype
        )
        with torch.inference_mode():
            output_ids = donut.hf_generate(pixels, max_new_tokens=max_new_tokens)

        decoded = donut.processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=False
        )
        results.append(
            {
                "image": str(sample["image"]),
                "gt": ground_truth(sample),
                "pred": parse_prediction(decoded, donut.processor),
            }
        )

    stats = field_stats(results, soft=False)
    field_f1 = [values["f1"] for values in stats.values() if values["f1"] is not None]
    if not field_f1:
        raise ValueError("The evaluation data contains no scoreable fields")

    return {
        "score": sum(field_f1) / len(field_f1),
        "parameters": parameter_counts(donut.model),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_checkpoint")
    parser.add_argument("student_checkpoint")
    parser.add_argument("data_json", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/distillation_evaluation.json")
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    samples = load_samples(args.data_json)
    record = {
        "metric": "strict_macro_field_f1",
        "data_json": str(args.data_json),
        "documents": len(samples),
        "teacher": evaluate(
            args.teacher_checkpoint, samples, args.device, args.max_new_tokens
        ),
        "student": evaluate(
            args.student_checkpoint, samples, args.device, args.max_new_tokens
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
