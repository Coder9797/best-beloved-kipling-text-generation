"""Comparison figures for loss, perplexity, accuracy, time, and temperature."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ARCHITECTURES, ARCH_LABELS, PLOT_DIR

COLORS = {
    "rnn": "#4C78A8",
    "birnn": "#F58518",
    "lstm": "#54A24B",
    "gru": "#E45756",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def save_training_curves(results: dict[str, dict], output_dir: Path = PLOT_DIR) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for arch in ARCHITECTURES:
        history = results[arch]["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], color=COLORS[arch], linestyle="--", linewidth=1.4)
        axes[0].plot(epochs, [row["val_loss"] for row in history], color=COLORS[arch], linewidth=2.0, label=ARCH_LABELS[arch])
        axes[1].plot(epochs, [row["val_perplexity"] for row in history], color=COLORS[arch], linewidth=2.0, label=ARCH_LABELS[arch])
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].plot([], [], color="0.4", linestyle="--", label="Train")
    axes[0].plot([], [], color="0.4", linewidth=2, label="Validation")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_title("Validation perplexity")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def save_comparison_bars(results: dict[str, dict], output_dir: Path = PLOT_DIR) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Simple RNN", "BiRNN", "LSTM", "GRU"]
    colors = [COLORS[arch] for arch in ARCHITECTURES]
    perplexities = [results[arch]["best_val_perplexity"] for arch in ARCHITECTURES]
    accuracies = [100 * results[arch]["best_val_accuracy"] for arch in ARCHITECTURES]
    seconds = [results[arch]["train_seconds"] for arch in ARCHITECTURES]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    series = [
        (perplexities, "Best validation perplexity", "Perplexity"),
        (accuracies, "Best validation accuracy", "Accuracy (%)"),
        (seconds, "Wall-clock training time", "Seconds"),
    ]
    for axis, (values, title, ylabel) in zip(axes, series):
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", labelrotation=0)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def save_temperature_plot(temperature_summary: dict, output_dir: Path = PLOT_DIR) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    temperatures = sorted({float(temp) for arch_rows in temperature_summary.values() for temp in arch_rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for arch in ARCHITECTURES:
        rows = temperature_summary[arch]
        distinct = [rows[str(temp)]["distinct_2"] for temp in temperatures]
        repetition = [rows[str(temp)]["repetition_rate"] for temp in temperatures]
        axes[0].plot(temperatures, distinct, marker="o", color=COLORS[arch], label=ARCH_LABELS[arch])
        axes[1].plot(temperatures, repetition, marker="o", color=COLORS[arch], label=ARCH_LABELS[arch])
    axes[0].set_title("Diversity (distinct-2)")
    axes[0].set_ylabel("Unique bigrams / bigrams")
    axes[1].set_title("Immediate repetition")
    axes[1].set_ylabel("Fraction of repeated tokens")
    for axis in axes:
        axis.set_xlabel("Sampling temperature")
        axis.set_xticks(temperatures)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "temperature_effects.png", dpi=150)
    plt.close(fig)
