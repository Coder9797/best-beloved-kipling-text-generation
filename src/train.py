"""Train one next-word model and score it on the held-out windows."""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import EPOCHS, GRAD_CLIP, LEARNING_RATE, SEED, WEIGHT_DECAY
from src.models import NextWordModel


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def run_epoch(
    model: NextWordModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_count += batch_size

    mean_loss = total_loss / max(total_count, 1)
    return {
        "loss": mean_loss,
        "perplexity": perplexity(mean_loss),
        "accuracy": total_correct / max(total_count, 1),
    }


def train_model(
    arch: str,
    vocab_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
) -> tuple[NextWordModel, dict]:
    set_seed(SEED)
    model = NextWordModel(vocab_size, arch).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_perplexity": train_metrics["perplexity"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_perplexity": val_metrics["perplexity"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        marker = ""
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            marker = "  *"
        print(
            f"  {arch:5} epoch {epoch:02d}/{epochs}  "
            f"train loss {row['train_loss']:.3f} ppl {row['train_perplexity']:.1f} acc {row['train_accuracy']:.3f}  "
            f"val loss {row['val_loss']:.3f} ppl {row['val_perplexity']:.1f} acc {row['val_accuracy']:.3f}  "
            f"({row['epoch_seconds']:.1f}s){marker}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError(f"Training produced no checkpoint for {arch}.")
    model.load_state_dict(best_state)
    result = {
        "architecture": arch,
        "parameters": model.parameter_count(),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "train_seconds": time.perf_counter() - started,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": perplexity(best_val_loss),
        "best_val_accuracy": history[best_epoch - 1]["val_accuracy"],
        "best_train_loss": history[best_epoch - 1]["train_loss"],
        "best_train_perplexity": history[best_epoch - 1]["train_perplexity"],
        "best_train_accuracy": history[best_epoch - 1]["train_accuracy"],
    }
    return model, result
