from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _infer_count(metadata: Any, column: str, default: int) -> int:
    df = getattr(metadata, "df", metadata)
    if df is None or not hasattr(df, "columns") or column not in df.columns:
        return int(default)
    values = df[column].dropna()
    if values.empty:
        return int(default)
    return max(int(values.max()) + 1, int(default))


class ConditionEncoder(nn.Module):
    """Encode V0 PHM generative conditions: fault_label and domain_id."""

    def __init__(
        self,
        metadata: Any = None,
        embedding_dim: int = 32,
        num_fault_classes: int | None = None,
        num_domains: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        n_fault = int(num_fault_classes or _infer_count(metadata, "Label", 2))
        n_domain = int(num_domains or _infer_count(metadata, "Domain_id", 2))
        self.fault_embedding = nn.Embedding(max(n_fault, 1), self.embedding_dim)
        self.domain_embedding = nn.Embedding(max(n_domain, 1), self.embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2 + 1, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, condition: dict[str, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        if not isinstance(condition, dict):
            raise ValueError("condition must be a dict with fault_label and domain_id")
        for key in ["fault_label", "domain_id"]:
            if key not in condition:
                raise ValueError(f"condition missing required key: {key}")
        fault = condition["fault_label"].long()
        domain = condition["domain_id"].long()
        if fault.ndim != 1 or domain.ndim != 1:
            fault = fault.view(-1)
            domain = domain.view(-1)
        if fault.max().item() >= self.fault_embedding.num_embeddings:
            raise ValueError("fault_label exceeds configured embedding size")
        if domain.max().item() >= self.domain_embedding.num_embeddings:
            raise ValueError("domain_id exceeds configured embedding size")
        t_vec = t.float().view(-1, 1)
        encoded = torch.cat(
            [self.fault_embedding(fault), self.domain_embedding(domain), t_vec],
            dim=1,
        )
        return self.proj(encoded)
