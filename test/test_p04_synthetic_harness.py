from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.p04.generate_synthetic import (
    AMPLITUDES,
    DATASET_ID,
    DATASET_NAME,
    DIAGNOSIS_FREQUENCIES,
    DOMAIN_ID,
    DRAWS_PER_CELL,
    FREQUENCY_MULTIPLIERS,
    MASTER_SEED,
    MECHANISMS,
    NOISE_PSD_SLOPES,
    PARTITION_CELL_COUNTS,
    PARTITION_NAMES,
    SAMPLE_ID_BASE,
    SNR_LEVELS_DB,
    enumerate_cells,
    generate_dataset,
    render_sample,
)


def test_constrained_cell_allocation_is_deterministic_and_complete() -> None:
    first = enumerate_cells()
    second = enumerate_cells()

    assert first == second
    assert len(first) == 4 * 4 * 36 == 576
    assert len({cell.allocation_sha256 for cell in first}) == 576
    assert {cell.mechanism for cell in first} == set(MECHANISMS)

    expected_support = (
        set(NOISE_PSD_SLOPES),
        set(SNR_LEVELS_DB),
        set(AMPLITUDES),
        set(FREQUENCY_MULTIPLIERS),
    )
    for diagnosis_label in range(len(DIAGNOSIS_FREQUENCIES)):
        for mechanism_id in range(len(MECHANISMS)):
            pair = [
                cell
                for cell in first
                if cell.diagnosis_label == diagnosis_label
                and cell.mechanism_id == mechanism_id
            ]
            assert Counter(cell.partition for cell in pair) == PARTITION_CELL_COUNTS
            for partition in PARTITION_NAMES:
                selected = [cell for cell in pair if cell.partition == partition]
                observed = tuple(
                    {
                        cell.factor_values()[axis]
                        for cell in selected
                    }
                    for axis in range(4)
                )
                assert observed == expected_support


def test_sample_rendering_is_byte_deterministic_and_uses_frozen_identity() -> None:
    cells = enumerate_cells()
    for mechanism_id in range(len(MECHANISMS)):
        cell = next(cell for cell in cells if cell.mechanism_id == mechanism_id)
        first = render_sample(cell, draw=3)
        second = render_sample(cell, draw=3)

        assert first.array.dtype == np.float32
        assert first.array.shape == (512, 2)
        assert np.array_equal(first.array, second.array)
        assert first.manifest == second.manifest
        assert first.audit == second.audit
        assert first.metadata["Dataset_id"] == DATASET_ID == 904
        assert first.metadata["Domain_id"] == DOMAIN_ID == 0
        assert first.metadata["Name"] == DATASET_NAME == "P04_Synthetic"
        assert isinstance(first.metadata["Id"], int)
        assert isinstance(first.metadata["Nuisance_cell"], int)


def test_full_generate_and_validate_contract(tmp_path: Path) -> None:
    output = tmp_path / "synthetic_v1"

    result = generate_dataset(output)

    assert result["status"] == "passed"
    assert result["sample_count"] == 4608
    assert result["root"] == str(output.resolve())
    assert len(result["generator_manifest_sha256"]) == 64
    assert len(result["partition_manifest_sha256"]) == 64
    assert len(result["metadata_file_sha256"]) == 64
    assert (output / "metadata.csv").is_file()
    raw_files = sorted((output / "raw" / DATASET_NAME).glob("*.npy"))
    assert len(raw_files) == 4608
    sample = np.load(raw_files[0], allow_pickle=False)
    assert sample.dtype == np.float32
    assert sample.shape == (512, 2)

    with (output / "metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    assert len(metadata) == 4608
    assert int(metadata[0]["Id"]) == SAMPLE_ID_BASE
    assert int(metadata[-1]["Id"]) == SAMPLE_ID_BASE + 4608 - 1
    assert {row["Partition"] for row in metadata} == set(PARTITION_NAMES)
    assert {row["Name"] for row in metadata} == {DATASET_NAME}

    partition_manifest = json.loads(
        (output / "partition_manifest.json").read_text(encoding="utf-8")
    )
    assert partition_manifest["schema_version"] == 1
    assert partition_manifest["strategy"] == "grouped_metadata"
    assert partition_manifest["task_type"] == "Default_task"
    assert partition_manifest["seed"] == MASTER_SEED
    assert partition_manifest["metadata_file_sha256"] == result["metadata_file_sha256"]
    assert partition_manifest["partition_map"] == {
        "train": "train",
        "val": "optimization_validation",
        "test": "intervention",
    }

    all_ids: list[int] = []
    all_groups: list[str] = []
    expected_samples = {
        partition: cell_count * 16 * DRAWS_PER_CELL
        for partition, cell_count in PARTITION_CELL_COUNTS.items()
    }
    for partition in PARTITION_NAMES:
        record = partition_manifest["partitions"][partition]
        assert record["sample_count"] == expected_samples[partition]
        assert record["group_count"] == PARTITION_CELL_COUNTS[partition] * 16
        assert all(isinstance(sample_id, int) for sample_id in record["ids"])
        all_ids.extend(record["ids"])
        all_groups.extend(record["groups"])
    assert len(all_ids) == len(set(all_ids)) == 4608
    assert len(all_groups) == len(set(all_groups)) == 576

    extrema = result["validation"]["observed_extrema"]
    assert extrema["minimum_low_frequency_power_ratio"] >= 0.80
    assert extrema["minimum_harmonic_combined_power_ratio"] >= 0.70
    assert extrema["minimum_impulse_envelope_kurtosis"] >= 5.0
    assert extrema["maximum_residual_nonzero_autocorrelation"] <= 0.30
    assert extrema["maximum_noise_slope_absolute_error"] <= 0.10

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_dataset(output)
