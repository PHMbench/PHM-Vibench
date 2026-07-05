from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY_PATH = Path("configs/registry/generative_frontier_methods.yaml")
ALLOWED_CLAIM_STATUS = {"benchmark-valid", "exploratory", "docs-only"}
ALLOWED_RUNTIME_STATUS = {"not_implemented", "implemented", "runtime_pilot", "toy_only"}
ALLOWED_INTEGRATION_LEVEL = {
    "runtime_pilot",
    "runtime_pilot_after_tfm",
    "runtime_pilot_later",
    "research_pilot",
    "toy_only",
    "project_card_only",
    "backbone_pilot",
}
REQUIRED_FIELDS = {
    "title",
    "family",
    "year",
    "reference",
    "integration_level",
    "runtime_status",
    "claim_status",
    "supports_one_step",
    "requires_ot",
    "requires_mamba",
    "promotion_requirements",
}


@dataclass(frozen=True)
class FrontierMethodSpec:
    method_id: str
    title: str
    family: str
    year: int
    reference: str
    integration_level: str
    runtime_status: str
    claim_status: str
    supports_one_step: bool
    requires_ot: bool
    requires_mamba: bool
    promotion_requirements: tuple[str, ...]

    @property
    def blocks_benchmark_valid(self) -> bool:
        return self.claim_status != "benchmark-valid" or self.integration_level in {
            "project_card_only",
            "toy_only",
            "research_pilot",
            "backbone_pilot",
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["promotion_requirements"] = list(self.promotion_requirements)
        data["blocks_benchmark_valid"] = self.blocks_benchmark_valid
        return data


def load_frontier_method_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, FrontierMethodSpec]:
    registry_path = Path(path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"frontier method registry not found: {registry_path}")
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    methods = data.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("frontier method registry must contain a non-empty methods mapping")

    loaded: dict[str, FrontierMethodSpec] = {}
    for method_id, raw_spec in methods.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"frontier method {method_id} must be a mapping")
        missing = sorted(REQUIRED_FIELDS.difference(raw_spec))
        if missing:
            raise ValueError(f"frontier method {method_id} is missing fields: {missing}")
        claim_status = str(raw_spec["claim_status"])
        runtime_status = str(raw_spec["runtime_status"])
        integration_level = str(raw_spec["integration_level"])
        if claim_status not in ALLOWED_CLAIM_STATUS:
            raise ValueError(f"frontier method {method_id} has invalid claim_status={claim_status!r}")
        if runtime_status not in ALLOWED_RUNTIME_STATUS:
            raise ValueError(
                f"frontier method {method_id} has invalid runtime_status={runtime_status!r}"
            )
        if integration_level not in ALLOWED_INTEGRATION_LEVEL:
            raise ValueError(
                f"frontier method {method_id} has invalid integration_level={integration_level!r}"
            )
        if claim_status == "benchmark-valid" and integration_level in {
            "project_card_only",
            "toy_only",
            "research_pilot",
            "backbone_pilot",
        }:
            raise ValueError(
                f"frontier method {method_id} cannot be benchmark-valid at "
                f"integration_level={integration_level!r}"
            )
        requirements = raw_spec["promotion_requirements"]
        if not isinstance(requirements, list) or not all(
            isinstance(item, str) and item for item in requirements
        ):
            raise ValueError(
                f"frontier method {method_id} promotion_requirements must be non-empty strings"
            )
        loaded[str(method_id)] = FrontierMethodSpec(
            method_id=str(method_id),
            title=str(raw_spec["title"]),
            family=str(raw_spec["family"]),
            year=int(raw_spec["year"]),
            reference=str(raw_spec["reference"]),
            integration_level=integration_level,
            runtime_status=runtime_status,
            claim_status=claim_status,
            supports_one_step=bool(raw_spec["supports_one_step"]),
            requires_ot=bool(raw_spec["requires_ot"]),
            requires_mamba=bool(raw_spec["requires_mamba"]),
            promotion_requirements=tuple(requirements),
        )
    return loaded


def get_frontier_method(
    method_id: str,
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> FrontierMethodSpec:
    registry = load_frontier_method_registry(path)
    try:
        return registry[method_id]
    except KeyError as exc:
        known = ", ".join(sorted(registry))
        raise KeyError(f"unknown frontier method {method_id!r}; known methods: {known}") from exc


def export_registry_snapshot(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[dict[str, Any]]:
    registry = load_frontier_method_registry(path)
    return [registry[method_id].to_dict() for method_id in sorted(registry)]
