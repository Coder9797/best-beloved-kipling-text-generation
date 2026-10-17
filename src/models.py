"""Next-word models with a shared embedding and output layer.

The training target is the word immediately after a fixed context window.
A bidirectional net is therefore allowed to read that window in both
directions. It never receives the target word, so the label does not leak.
At generation time the same window is encoded again, the sampled word is
appended, and the oldest word is dropped.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ARCH_LABELS, DROPOUT, EMBED_DIM, HIDDEN_DIM


class NextWordModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        arch: str,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if arch not in ARCH_LABELS:
            raise ValueError(f"Unknown architecture: {arch}")
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even so the bidirectional RNN can split it.")

        self.arch = arch
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        if arch == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif arch == "birnn":
            # Each direction uses hidden_dim / 2, so the concatenated state
            # matches the unidirectional hidden size.
            self.rnn = nn.RNN(
                embed_dim,
                hidden_dim // 2,
                batch_first=True,
                bidirectional=True,
            )
        elif arch == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif arch == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dim() >= 2:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.input_dropout(self.embed(token_ids))
        _output, hidden = self.rnn(embedded)
        summary = self._summarise(hidden)
        return self.fc(self.output_dropout(summary))

    def _summarise(self, hidden: torch.Tensor | tuple) -> torch.Tensor:
        if self.arch == "lstm":
            hidden = hidden[0]
        if self.arch == "birnn":
            # hidden[0] has read the window left to right.
            # hidden[1] has read the same window right to left.
            return torch.cat([hidden[0], hidden[1]], dim=-1)
        return hidden[-1]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
