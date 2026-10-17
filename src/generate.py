"""Temperature sampling and qualitative diversity measures."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.config import GENERATED_WORDS, SEQ_LEN
from src.data import UNK
from src.models import NextWordModel


def detokenize(tokens: list[str]) -> str:
    pieces: list[str] = []
    for token in tokens:
        if token in ".,!?;:" and pieces:
            pieces[-1] = pieces[-1] + token
        else:
            pieces.append(token)
    return " ".join(pieces)


@torch.no_grad()
def generate_tokens(
    model: NextWordModel,
    seed_ids: list[int],
    itos: list[str],
    device: torch.device,
    temperature: float,
    n_words: int = GENERATED_WORDS,
    seq_len: int = SEQ_LEN,
    banned_ids: list[int] | None = None,
) -> list[str]:
    """Sample the next word from softmax(logits / temperature).

    ``banned_ids`` is used to keep the unknown-token bucket out of the printed
    text. Validation metrics still score that class; this mask is decoding only
    and is applied to every model.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    model.eval()
    context = list(seed_ids)
    generated: list[str] = []
    for _ in range(n_words):
        window = context[-seq_len:]
        inputs = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(inputs)[0] / temperature
        if banned_ids:
            logits = logits.clone()
            logits[banned_ids] = float("-inf")
        probabilities = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probabilities, num_samples=1).item())
        token = itos[next_id]
        generated.append(token)
        context.append(next_id)
    return generated


def diversity_metrics(tokens: list[str]) -> dict[str, float]:
    total = len(tokens)
    if total == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_rate": 0.0, "unk_rate": 0.0}
    distinct_1 = len(set(tokens)) / total
    bigrams = list(zip(tokens, tokens[1:]))
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    repeats = sum(1 for previous, current in zip(tokens, tokens[1:]) if previous == current)
    unk_rate = sum(token == UNK for token in tokens) / total
    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repetition_rate": repeats / max(total - 1, 1),
        "unk_rate": unk_rate,
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
