"""Train four next-word models and write metrics, plots, and samples.

Usage:
    python run_experiment.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from src.config import (
    ARCHITECTURES,
    ARCH_LABELS,
    BATCH_SIZE,
    BOOKS,
    CANDIDATE_SEEDS,
    CHECKPOINT_DIR,
    DROPOUT,
    EMBED_DIM,
    EPOCHS,
    GENERATED_WORDS,
    GRAD_CLIP,
    HIDDEN_DIM,
    LEARNING_RATE,
    MIN_FREQ,
    RESULTS_DIR,
    SEED,
    SEQ_LEN,
    STRIDE,
    TEMPERATURES,
    TRAIN_FRACTION,
)
from src.data import UNK, encode_seed, make_loader, prepare_datasets
from src.generate import detokenize, diversity_metrics, generate_tokens, mean_metrics
from src.plotting import save_comparison_bars, save_temperature_plot, save_training_curves
from src.train import train_model


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_seeds(stoi: dict[str, int]) -> list[dict]:
    chosen = []
    for text in CANDIDATE_SEEDS:
        ids = encode_seed(text, stoi)
        if ids is None:
            print(f"  skipping seed with unknown tokens: {text}")
            continue
        chosen.append({"text": text, "ids": ids})
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        raise RuntimeError("Fewer than three seeds were fully covered by the vocabulary.")
    return chosen


def write_samples_markdown(samples: list[dict], path: Path) -> None:
    lines = [
        "# Generated samples",
        "",
        "Each continuation is 70 words sampled from the same three seeds.",
        "The seed is shown in brackets. Text is lowercase because that is how the models were trained.",
        "The unknown-token class is masked at decoding time so the samples are real words.",
        "Loss, perplexity, and accuracy still count that class during training and validation.",
        "",
    ]
    current = None
    for sample in samples:
        if sample["architecture"] != current:
            current = sample["architecture"]
            lines.extend(["", f"## {ARCH_LABELS[current]}", ""])
        lines.append(f"### Temperature {sample['temperature']}")
        lines.append("")
        lines.append(f"*Seed:* {sample['seed']}")
        lines.append("")
        lines.append(sample["text"])
        lines.append("")
        metrics = sample["metrics"]
        lines.append(
            "Distinct-1 {:.3f}, distinct-2 {:.3f}, immediate repetition {:.3f}, unknown-token rate {:.3f}.".format(
                metrics["distinct_1"],
                metrics["distinct_2"],
                metrics["repetition_rate"],
                metrics["unk_rate"],
            )
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = choose_device()
    print(f"Device: {device}", flush=True)

    prepared = prepare_datasets()
    stats = prepared["stats"]
    print(
        f"Tokens {stats['total_tokens']:,} | vocab {stats['vocab_size']:,} | "
        f"train windows {stats['train_windows']:,} | val windows {stats['val_windows']:,}",
        flush=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    results = {}
    models = {}
    for arch in ARCHITECTURES:
        print(f"\nTraining {ARCH_LABELS[arch]}", flush=True)
        model, result = train_model(
            arch,
            vocab_size=len(prepared["itos"]),
            train_loader=make_loader(prepared["train_x"], prepared["train_y"], shuffle=True),
            val_loader=make_loader(prepared["val_x"], prepared["val_y"], shuffle=False),
            device=device,
        )
        models[arch] = model
        results[arch] = result
        torch.save(
            {
                "architecture": arch,
                "state_dict": model.state_dict(),
                "itos": prepared["itos"],
                "config": {
                    "embed_dim": EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT,
                    "seq_len": SEQ_LEN,
                },
            },
            CHECKPOINT_DIR / f"{arch}.pt",
        )
        print(
            f"  best epoch {result['best_epoch']}: "
            f"val ppl {result['best_val_perplexity']:.1f}, "
            f"val acc {result['best_val_accuracy']:.3f}, "
            f"time {result['train_seconds']:.1f}s, "
            f"params {result['parameters']:,}",
            flush=True,
        )

    seeds = choose_seeds(prepared["stoi"])
    samples = []
    temperature_summary: dict[str, dict] = {arch: {} for arch in ARCHITECTURES}
    for arch, model in models.items():
        for temperature in TEMPERATURES:
            metric_rows = []
            for seed in seeds:
                generated = generate_tokens(
                    model,
                    seed["ids"],
                    prepared["itos"],
                    device,
                    temperature=temperature,
                    banned_ids=[prepared["stoi"][UNK]],
                )
                metrics = diversity_metrics(generated)
                metric_rows.append(metrics)
                seed_tokens = [prepared["itos"][index] for index in seed["ids"]]
                samples.append(
                    {
                        "architecture": arch,
                        "temperature": temperature,
                        "seed": seed["text"],
                        "text": f"[{detokenize(seed_tokens)}] {detokenize(generated)}",
                        "metrics": metrics,
                    }
                )
            temperature_summary[arch][str(temperature)] = mean_metrics(metric_rows)

    save_training_curves(results)
    save_comparison_bars(results)
    save_temperature_plot(temperature_summary)
    write_samples_markdown(samples, RESULTS_DIR / "generated_samples.md")

    payload = {
        "system_title": "Best Beloved: Autoregressive Text Generation on Kipling's Animal Tales",
        "device": str(device),
        "seed": SEED,
        "hyperparameters": {
            "seq_len": SEQ_LEN,
            "stride": STRIDE,
            "min_freq": MIN_FREQ,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "epochs": EPOCHS,
            "train_fraction": TRAIN_FRACTION,
            "optimizer": "Adam",
            "loss": "cross-entropy on the next word",
        },
        "books": BOOKS,
        "dataset": stats,
        "models": results,
        "seeds": [seed["text"] for seed in seeds],
        "temperatures": TEMPERATURES,
        "generated_words": GENERATED_WORDS,
        "generation": {
            "temperatures": TEMPERATURES,
            "words": GENERATED_WORDS,
            "unk_masked_at_decoding": True,
            "checkpoint": "best validation loss",
        },
        "temperature_summary": temperature_summary,
        "samples": samples,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nWrote results to", RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
