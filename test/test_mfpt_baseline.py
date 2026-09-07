from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from phmfactory.config import analyze_config
from scripts.prepare_mfpt_baseline import (
    EXPECTED_FILES,
    PROVIDER_REVISION,
    TEST_FILES,
    TRAIN_FILES,
    prepare_dataset,
    verify_provider_tree,
)
from src.data_factory.reader.RM_007_MFPT import read, read_record


def _write_mfpt_record(
    path: Path,
    *,
    signal: np.ndarray | None = None,
    sample_rate: float = 48_828.0,
    include_gs: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bearing = {
        "sr": sample_rate,
        "rate": 25.0,
        "load": 50.0,
    }
    if include_gs:
        bearing["gs"] = (
            np.linspace(-1.0, 1.0, 256, dtype=np.float64)
            if signal is None
            else signal
        )
    savemat(
        path,
        {
            "bearing": bearing,
            "BPFO": 81.125,
            "BPFI": 118.875,
            "FTF": 14.838,
            "BSF": 63.91,
        },
    )


def _provider_tree(root: Path) -> Path:
    for split, filenames in (("train_data", TRAIN_FILES), ("test_data", TEST_FILES)):
        for index, filename in enumerate(filenames):
            signal = np.sin(
                np.linspace(0.0, 8.0 * np.pi, 256, dtype=np.float64)
                + index
            )
            _write_mfpt_record(
                root / split / filename,
                signal=signal,
                sample_rate=48_828.0 if index % 2 == 0 else 97_656.0,
            )
    return root


def test_mfpt_reader_returns_signal_and_physical_metadata(tmp_path: Path) -> None:
    path = tmp_path / "baseline_1.mat"
    _write_mfpt_record(path)

    record = read_record(path)
    signal = read(path)

    assert signal.shape == (256, 1)
    assert np.array_equal(signal, record["signal"])
    assert record["sample_rate_hz"] == 48_828.0
    assert record["shaft_rate_hz"] == 25.0
    assert record["load"] == 50.0
    assert record["BPFO"] == 81.125
    assert record["BPFI"] == 118.875
    assert np.isfinite(signal).all()


def test_mfpt_reader_rejects_missing_or_invalid_scientific_fields(
    tmp_path: Path,
) -> None:
    missing_gs = tmp_path / "missing_gs.mat"
    _write_mfpt_record(missing_gs, include_gs=False)
    with pytest.raises(KeyError, match="bearing.gs|field 'gs'"):
        read_record(missing_gs)

    bad_rate = tmp_path / "bad_rate.mat"
    _write_mfpt_record(bad_rate, sample_rate=0.0)
    with pytest.raises(ValueError, match="bearing.sr.*positive"):
        read_record(bad_rate)

    nonfinite = tmp_path / "nonfinite.mat"
    signal = np.ones(256, dtype=np.float64)
    signal[10] = np.nan
    _write_mfpt_record(nonfinite, signal=signal)
    with pytest.raises(FloatingPointError, match="bearing.gs.*NaN or Inf"):
        read_record(nonfinite)

    missing_frequency = tmp_path / "missing_frequency.mat"
    savemat(
        missing_frequency,
        {
            "bearing": {
                "gs": np.ones(256),
                "sr": 48_828.0,
                "rate": 25.0,
                "load": 0.0,
            },
            "BPFI": 118.875,
            "FTF": 14.838,
            "BSF": 63.91,
        },
    )
    with pytest.raises(KeyError, match="BPFO"):
        read_record(missing_frequency)


def test_mfpt_provider_tree_requires_exact_public_file_set(tmp_path: Path) -> None:
    provider = _provider_tree(tmp_path / "provider")
    assert verify_provider_tree(provider) == provider.resolve()

    unexpected = provider / "test_data" / "unexpected.mat"
    _write_mfpt_record(unexpected)
    with pytest.raises(ValueError, match="unexpected=.*unexpected.mat"):
        verify_provider_tree(provider)


def test_mfpt_preparation_builds_exact_metadata_without_overwrite(
    tmp_path: Path,
) -> None:
    provider = _provider_tree(tmp_path / "provider")
    output = tmp_path / "prepared"

    metadata_path = prepare_dataset(
        provider,
        output,
        source_revision=PROVIDER_REVISION,
    )

    assert metadata_path == output / "metadata_mfpt.csv"
    assert {
        path.relative_to(output / "raw" / "RM_007_MFPT").as_posix()
        for path in (output / "raw" / "RM_007_MFPT").rglob("*.mat")
    } == EXPECTED_FILES

    with metadata_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert sum(row["Provider_Split"] == "train" for row in rows) == 14
    assert sum(row["Provider_Split"] == "test" for row in rows) == 6
    assert {int(row["Label"]) for row in rows} == {0, 1, 2}
    assert {int(row["Domain_id"]) for row in rows} == {0, 1}
    assert {int(row["Channels"]) for row in rows} == {1}
    assert {row["Source_Revision"] for row in rows} == {PROVIDER_REVISION}
    assert {row["License"] for row in rows} == {"CC BY-NC-SA 4.0"}
    assert {float(row["Sample_Rate"]) for row in rows} == {48_828.0, 97_656.0}

    with pytest.raises(FileExistsError, match="never overwrites user data"):
        prepare_dataset(provider, output, source_revision=PROVIDER_REVISION)


def test_mfpt_baseline_config_freezes_file_level_protocol(tmp_path: Path) -> None:
    analysis = analyze_config(
        "configs/baselines/01_mfpt/mfpt_global_average_linear.yaml",
        override_values=(
            f"data.data_dir={tmp_path / 'mfpt'}",
            f"environment.output_dir={tmp_path / 'results'}",
            f"data.split.manifest_path={tmp_path / 'results' / 'split_manifest.json'}",
        ),
    )
    config = analysis.effective_config

    assert config["pipeline"] == "Pipeline_01_Fault_Diagnosis"
    assert config["environment"]["seed"] == 17
    assert config["environment"]["iterations"] == 3
    assert config["model"] == {
        "type": "Baseline",
        "name": "GlobalAverageLinear",
        "input_dim": 1,
    }
    assert config["task"]["target_system_id"] == [7]
    assert config["task"]["source_domain_id"] == [0]
    assert config["task"]["target_domain_id"] == [1]
    assert config["data"]["split"]["strategy"] == "grouped_metadata"
    assert config["data"]["split"]["group_key"] == "File"
    assert config["data"]["split"]["stratify_key"] == "Label"
    assert config["data"]["split"]["test_policy"] == "task_defined"
    assert config["trainer"]["device"] == "cpu"
