from __future__ import annotations

from typing import Any


FORBIDDEN_SYNTHETIC_SOURCE_SPLITS = {"val", "valid", "validation", "test", "target_test"}


def build_utility_protocol_metadata(
    *,
    protocol_id: str,
    synthetic_source_split: str,
    reference_split: str,
    allow_test_reference_eval: bool = False,
    augmentation_ratio: float | None = None,
) -> dict[str, Any]:
    """Validate and describe TSTR/TRTS/augmentation utility protocol usage."""
    source = str(synthetic_source_split).strip().lower()
    reference = str(reference_split).strip().lower()
    if source in FORBIDDEN_SYNTHETIC_SOURCE_SPLITS:
        raise ValueError(
            "synthetic utility data cannot be sourced from validation/test splits; "
            f"got synthetic_source_split={synthetic_source_split!r}"
        )
    if reference in {"test", "target_test"} and not allow_test_reference_eval:
        raise ValueError("test utility reference requires allow_test_reference_eval=true")
    if augmentation_ratio is not None and augmentation_ratio < 0.0:
        raise ValueError("augmentation_ratio must be non-negative")
    return {
        "utility_protocol_id": protocol_id,
        "synthetic_source_split": source,
        "reference_split": "val" if reference == "valid" else reference,
        "allow_test_reference_eval": bool(allow_test_reference_eval),
        "augmentation_ratio": augmentation_ratio,
        "metrics": ["tstr_accuracy", "trts_accuracy"],
    }
