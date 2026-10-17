"""Load Kipling's animal tales and build next-word training windows."""

from __future__ import annotations

import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BATCH_SIZE,
    BOOKS,
    MIN_FREQ,
    RAW_DIR,
    SEED,
    SEQ_LEN,
    STRIDE,
    TRAIN_FRACTION,
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*|[.,!?;:]")
START_RE = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)

UNK = "<unk>"


def strip_gutenberg(text: str) -> str:
    start = START_RE.search(text)
    end = END_RE.search(text)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Could not find Project Gutenberg start/end markers.")
    return text[start.end() : end.start()]


def normalize(text: str) -> str:
    """Lowercase the prose and map curly punctuation onto ASCII tokens."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": " ",
        "\u201d": " ",
        "\u2014": " ",
        "\u2013": " ",
        "\u00a0": " ",
        "\u00ad": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.lower()
    text = text.replace("'", "'")
    text = re.sub(r"[^a-z0-9'.,!?;:\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def load_books() -> list[dict]:
    books = []
    for meta in BOOKS:
        raw = (RAW_DIR / meta["filename"]).read_text(encoding="utf-8", errors="replace")
        tokens = tokenize(normalize(strip_gutenberg(raw)))
        books.append({**meta, "tokens": tokens})
    return books


def split_book(tokens: list[str], train_fraction: float = TRAIN_FRACTION) -> tuple[list[str], list[str]]:
    """Chronological split so overlapping windows cannot leak into validation."""
    cut = int(len(tokens) * train_fraction)
    cut = min(max(cut, SEQ_LEN + 1), len(tokens) - (SEQ_LEN + 1))
    return tokens[:cut], tokens[cut:]


def build_vocab(train_tokens: list[str], min_freq: int = MIN_FREQ) -> tuple[dict[str, int], list[str]]:
    counts = Counter(train_tokens)
    itos = [UNK]
    for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= min_freq:
            itos.append(token)
    stoi = {token: index for index, token in enumerate(itos)}
    return stoi, itos


def encode(tokens: list[str], stoi: dict[str, int]) -> list[int]:
    unk_id = stoi[UNK]
    return [stoi.get(token, unk_id) for token in tokens]


def make_windows(ids: list[int], seq_len: int = SEQ_LEN, stride: int = STRIDE) -> tuple[torch.Tensor, torch.Tensor]:
    """Each example is a context window and the single word that follows it."""
    inputs = []
    targets = []
    last = len(ids) - seq_len
    for start in range(0, last, stride):
        inputs.append(ids[start : start + seq_len])
        targets.append(ids[start + seq_len])
    if not inputs:
        raise ValueError("Not enough tokens to build a single training window.")
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def make_loader(inputs: torch.Tensor, targets: torch.Tensor, shuffle: bool) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(SEED)
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def describe_corpus(books: list[dict], train_tokens: list[str], val_tokens: list[str], itos: list[str]) -> dict:
    all_tokens = train_tokens + val_tokens
    sentences = []
    current = 0
    for token in all_tokens:
        current += 1
        if token in ".!?":
            if current > 1:
                sentences.append(current)
            current = 0
    counts = Counter(train_tokens)
    return {
        "books": [
            {
                "title": book["title"],
                "year": book["year"],
                "url": book["url"],
                "tokens": len(book["tokens"]),
            }
            for book in books
        ],
        "total_tokens": len(all_tokens),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "unique_train_tokens": len(counts),
        "vocab_size": len(itos),
        "min_freq": MIN_FREQ,
        "type_token_ratio_train": round(len(counts) / max(len(train_tokens), 1), 4),
        "mean_sentence_length_tokens": round(sum(sentences) / len(sentences), 2) if sentences else None,
        "sentence_count": len(sentences),
        "most_frequent_train_tokens": counts.most_common(15),
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
    }


def prepare_datasets() -> dict:
    books = load_books()
    train_tokens: list[str] = []
    val_tokens: list[str] = []
    for book in books:
        train_part, val_part = split_book(book["tokens"])
        book["train_tokens"] = len(train_part)
        book["val_tokens"] = len(val_part)
        train_tokens.extend(train_part)
        val_tokens.extend(val_part)

    stoi, itos = build_vocab(train_tokens)
    train_x, train_y = make_windows(encode(train_tokens, stoi))
    val_x, val_y = make_windows(encode(val_tokens, stoi))
    stats = describe_corpus(books, train_tokens, val_tokens, itos)
    stats["train_windows"] = int(train_x.shape[0])
    stats["val_windows"] = int(val_x.shape[0])
    unk_rate = float((val_y == stoi[UNK]).float().mean()) if len(val_y) else 0.0
    stats["val_target_unk_rate"] = round(unk_rate, 4)
    return {
        "stoi": stoi,
        "itos": itos,
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "stats": stats,
    }


def encode_seed(text: str, stoi: dict[str, int]) -> list[int] | None:
    tokens = tokenize(normalize(text))
    if len(tokens) < 4:
        return None
    unk_id = stoi[UNK]
    ids = [stoi.get(token, unk_id) for token in tokens]
    if any(index == unk_id for index in ids):
        return None
    return ids
