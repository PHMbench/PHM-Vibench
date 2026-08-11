"""Clean-room TSL-style Transformer and LSTM classifier."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _TSLBlock(nn.Module):
    """Parallel self-attention and LSTM branches followed by an FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        lstm_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.branch_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm_projection = (
            nn.Identity()
            if lstm_hidden_dim == d_model
            else nn.Linear(lstm_hidden_dim, d_model)
        )
        self.branch_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.branch_norm(x)
        attention, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        recurrent, _ = self.lstm(normalized)
        x = x + self.branch_dropout(attention + self.lstm_projection(recurrent))
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    """Patch-based TSL-style classifier using the repository input convention."""

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        del metadata

        self.input_dim = int(getattr(args, "input_dim", 1))
        self.seq_len = int(getattr(args, "seq_len", 128))
        self.patch_size = int(getattr(args, "patch_size", 16))
        self.d_model = int(getattr(args, "d_model", 64))
        self.n_heads = int(getattr(args, "n_heads", getattr(args, "num_heads", 4)))
        self.num_layers = int(getattr(args, "num_layers", 2))
        self.d_ff = int(getattr(args, "d_ff", 128))
        self.lstm_hidden_dim = int(getattr(args, "lstm_hidden_dim", self.d_model))
        self.dropout_rate = float(getattr(args, "dropout", 0.1))
        self.num_classes = getattr(args, "num_classes", None)

        positive_values = {
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "patch_size": self.patch_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "lstm_hidden_dim": self.lstm_hidden_dim,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"TSLTransformer requires positive values for: {', '.join(invalid)}")
        if self.seq_len % self.patch_size != 0:
            raise ValueError("model.seq_len must be divisible by model.patch_size")
        if self.d_model % self.n_heads != 0:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if not isinstance(self.num_classes, int) or self.num_classes <= 1:
            raise ValueError("model.num_classes must resolve to one integer greater than one")

        self.num_patches = self.seq_len // self.patch_size
        self.patch_projection = nn.Linear(
            self.patch_size * self.input_dim,
            self.d_model,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.d_model)
        )
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.blocks = nn.ModuleList(
            [
                _TSLBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    lstm_hidden_dim=self.lstm_hidden_dim,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "TSLTransformer expects input shaped [batch, length, channels], "
                f"received {tuple(x.shape)}"
            )
        batch_size, seq_len, channels = x.shape
        if seq_len != self.seq_len or channels != self.input_dim:
            raise ValueError(
                "TSLTransformer input mismatch: expected "
                f"[batch, {self.seq_len}, {self.input_dim}], received {tuple(x.shape)}"
            )

        patches = x.reshape(
            batch_size,
            self.num_patches,
            self.patch_size * self.input_dim,
        )
        tokens = self.patch_projection(patches)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.input_dropout(tokens + self.position_embedding)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)[:, 0]

    def forward(
        self,
        x: torch.Tensor,
        file_id: Any = None,
        task_id: Any = None,
        return_feature: bool = False,
    ):
        del file_id
        if task_id not in {None, False, "classification"}:
            raise ValueError(f"TSLTransformer supports classification only, got {task_id!r}")
        features = self.encode(x)
        logits = self.classifier(features)
        if return_feature:
            return logits, features
        return logits
