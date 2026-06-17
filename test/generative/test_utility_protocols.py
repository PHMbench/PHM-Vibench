from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.paperpack_generative import build_paperpack
from src.task_factory.Components.generative.metrics.utility_protocol import (
    build_utility_protocol_metadata,
)


def test_utility_protocol_rejects_synthetic_test_source() -> None:
    with pytest.raises(ValueError, match="cannot be sourced from validation/test"):
        build_utility_protocol_metadata(
            protocol_id="tstr_smoke",
            synthetic_source_split="test",
            reference_split="train",
        )


def test_utility_protocol_requires_explicit_test_reference_permission() -> None:
    with pytest.raises(ValueError, match="allow_test_reference_eval=true"):
        build_utility_protocol_metadata(
            protocol_id="tstr_smoke",
            synthetic_source_split="train",
            reference_split="test",
        )


def test_utility_protocol_metadata_is_paperpack_indexed(tmp_path: Path) -> None:
    metadata = build_utility_protocol_metadata(
        protocol_id="tstr_trts_dummy",
        synthetic_source_split="train",
        reference_split="val",
        augmentation_ratio=1.0,
    )
    assert "utility_classifier_tstr_accuracy" in metadata["metrics"]
    assert "tstr_accuracy" in metadata["deprecated_metric_aliases"]
    metrics_path = tmp_path / "run" / "generative_eval_metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tstr_nearest_centroid_accuracy",
            "tstr_nearest_centroid_accuracy_status",
            "tstr_nearest_centroid_accuracy_reason",
            "trts_nearest_centroid_accuracy",
            "trts_nearest_centroid_accuracy_status",
            "trts_nearest_centroid_accuracy_reason",
            "utility_protocol_id",
            "utility_source_split",
            "utility_reference_split",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "tstr_nearest_centroid_accuracy": "0.5",
                "tstr_nearest_centroid_accuracy_status": "ok",
                "trts_nearest_centroid_accuracy": "0.25",
                "trts_nearest_centroid_accuracy_status": "ok",
                "utility_protocol_id": metadata["utility_protocol_id"],
                "utility_source_split": metadata["synthetic_source_split"],
                "utility_reference_split": metadata["reference_split"],
            }
        )

    paperpack = build_paperpack(tmp_path / "run")
    run_index = (paperpack / "appendix" / "run_index.csv").read_text(encoding="utf-8")
    utility_table = (paperpack / "tables" / "table_utility_mean_std.csv").read_text(
        encoding="utf-8"
    )

    assert "tstr_trts_dummy" in run_index
    assert "tstr_nearest_centroid_accuracy" in utility_table
    assert "trts_nearest_centroid_accuracy" in utility_table
