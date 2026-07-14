from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

import torch
import torch.nn as nn


def _metadata_rows(metadata: Any) -> Iterable[Mapping[str, Any]]:
    """Yield metadata rows without importing pandas at module import time."""

    if metadata is None:
        return []

    dataframe = getattr(metadata, "df", None)
    if dataframe is not None and hasattr(dataframe, "iterrows"):
        return [row.to_dict() for _, row in dataframe.iterrows()]

    if isinstance(metadata, Mapping):
        values = metadata.values()
    elif hasattr(metadata, "values") and callable(metadata.values):
        values = metadata.values()
    else:
        return []

    return [row for row in values if isinstance(row, Mapping)]


def _infer_cardinality(metadata: Any, key: str, default: int) -> int:
    values: list[int] = []
    for row in _metadata_rows(metadata):
        value = row.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            values.append(parsed)
    return max(values) + 1 if values else int(default)


def _resolve_cardinality(
    explicit: int | None,
    metadata: Any,
    key: str,
    default: int,
) -> int:
    cardinality = int(explicit) if explicit is not None else _infer_cardinality(
        metadata,
        key,
        default,
    )
    if cardinality <= 0:
        raise ValueError(f"{key} cardinality must be positive, got {cardinality}")
    return cardinality


class ConditionEncoder(nn.Module):
    """Encode the v0.2.1 PHM condition contract.

    The direct model inputs are deliberately limited to ``fault_label`` and
    ``domain_id``. Operating variables such as RPM/load may be recorded through
    a domain map but are not direct control inputs in this release slice.
    """

    def __init__(
        self,
        metadata: Any = None,
        embedding_dim: int = 32,
        num_fault_classes: int | None = None,
        num_domains: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )

        self.num_fault_classes = _resolve_cardinality(
            num_fault_classes,
            metadata,
            "Label",
            2,
        )
        self.num_domains = _resolve_cardinality(
            num_domains,
            metadata,
            "Domain_id",
            2,
        )

        self.fault_embedding = nn.Embedding(
            self.num_fault_classes,
            self.embedding_dim,
        )
        self.domain_embedding = nn.Embedding(
            self.num_domains,
            self.embedding_dim,
        )
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim * 2 + 1, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(
        self,
        condition: dict[str, torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(condition, dict):
            raise ValueError(
                "condition must be a dict containing fault_label and domain_id"
            )
        for key in ("fault_label", "domain_id"):
            if key not in condition:
                raise ValueError(f"condition missing required key: {key}")

        weight = self.fault_embedding.weight
        t = torch.as_tensor(t, device=weight.device, dtype=weight.dtype).reshape(-1)
        if not torch.isfinite(t).all():
            raise ValueError("t contains NaN/Inf")

        fault = torch.as_tensor(
            condition["fault_label"],
            device=t.device,
            dtype=torch.long,
        ).reshape(-1)
        domain = torch.as_tensor(
            condition["domain_id"],
            device=t.device,
            dtype=torch.long,
        ).reshape(-1)

        if fault.numel() != t.numel() or domain.numel() != t.numel():
            raise ValueError(
                "condition batch mismatch: "
                f"fault={fault.numel()}, domain={domain.numel()}, t={t.numel()}"
            )
        if fault.numel() == 0:
            raise ValueError("condition batch must not be empty")
        if torch.any(fault < 0) or torch.any(domain < 0):
            raise ValueError("fault_label and domain_id must be non-negative")
        if int(fault.max().item()) >= self.num_fault_classes:
            raise ValueError(
                "fault_label exceeds configured embedding size: "
                f"max={int(fault.max().item())}, size={self.num_fault_classes}"
            )
        if int(domain.max().item()) >= self.num_domains:
            raise ValueError(
                "domain_id exceeds configured embedding size: "
                f"max={int(domain.max().item())}, size={self.num_domains}"
            )

        encoded = torch.cat(
            (
                self.fault_embedding(fault),
                self.domain_embedding(domain),
                t.unsqueeze(1),
            ),
            dim=1,
        )
        return self.projection(encoded)
