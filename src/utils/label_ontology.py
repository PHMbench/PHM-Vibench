"""Classification label-ontology validation shared by runtime boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real
from typing import Any, Iterable


def metadata_rows(metadata: Any) -> list[dict[str, Any]]:
    """Return metadata as a non-empty list of plain row mappings."""

    source = getattr(metadata, "df", metadata)
    if isinstance(source, Mapping):
        raw_rows = list(source.values())
    elif hasattr(source, "to_dict"):
        try:
            raw_rows = source.to_dict(orient="records")
        except TypeError:
            raw_rows = source.to_dict("records")
    else:
        raise TypeError(
            "metadata must be a mapping, a pandas-like table, or expose a .df table"
        )

    if not raw_rows:
        raise ValueError("metadata must contain at least one row")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping) and hasattr(row, "to_dict"):
            row = row.to_dict()
        if not isinstance(row, Mapping):
            raise TypeError(
                f"metadata row {index} must be a mapping, got {type(row).__name__}"
            )
        rows.append(dict(row))
    return rows


def _normalize_label(value: Any, *, context: str) -> int:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, bool):
        raise ValueError(f"{context} labels must be integers, not boolean values")
    if isinstance(value, Integral):
        label = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not isfinite(numeric) or not numeric.is_integer():
            raise ValueError(
                f"{context} labels must be finite integers, got {value!r}"
            )
        label = int(numeric)
    else:
        raise ValueError(
            f"{context} labels must be numeric integers, got {value!r}"
        )

    if label < 0:
        raise ValueError(f"{context} labels must be non-negative, got {label}")
    return label


def validate_zero_based_contiguous_labels(
    labels: Iterable[Any],
    *,
    context: str,
) -> int:
    """Validate ``{0, 1, ..., K-1}`` and return ``K``."""

    normalized = [_normalize_label(value, context=context) for value in labels]
    if not normalized:
        raise ValueError(f"{context} has no labels")

    observed = sorted(set(normalized))
    expected = list(range(observed[-1] + 1))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        raise ValueError(
            f"{context} labels must be zero-based and contiguous: "
            f"expected={expected}, observed={observed}, missing={missing}. "
            "PHMFactory does not silently re-encode labels."
        )
    return len(expected)


def validate_metadata_label_ontology(
    metadata: Any,
    *,
    group_field: str,
    require_labels: bool,
) -> dict[Any, int]:
    """Validate label ontologies independently for each metadata group."""

    grouped_labels: dict[Any, list[Any]] = {}
    missing_label_groups: set[Any] = set()

    for index, row in enumerate(metadata_rows(metadata)):
        if group_field not in row:
            raise KeyError(
                f"metadata row {index} is missing required field {group_field!r}"
            )
        group = row[group_field]
        try:
            hash(group)
        except TypeError as exc:
            raise TypeError(
                f"metadata field {group_field!r} must be scalar and hashable, "
                f"got {group!r}"
            ) from exc

        if "Label" not in row:
            missing_label_groups.add(group)
            grouped_labels.setdefault(group, [])
            continue
        grouped_labels.setdefault(group, []).append(row["Label"])

    results: dict[Any, int] = {}
    for group, labels in grouped_labels.items():
        if labels and group in missing_label_groups:
            raise ValueError(
                f"metadata group {group_field}={group!r} mixes rows with and "
                "without Label values"
            )
        if not labels:
            if require_labels:
                raise ValueError(
                    f"metadata group {group_field}={group!r} has no Label values"
                )
            continue
        results[group] = validate_zero_based_contiguous_labels(
            labels,
            context=f"metadata group {group_field}={group!r}",
        )

    if require_labels and not results:
        raise ValueError("metadata contains no classification label ontology")
    return results


__all__ = [
    "metadata_rows",
    "validate_metadata_label_ontology",
    "validate_zero_based_contiguous_labels",
]
