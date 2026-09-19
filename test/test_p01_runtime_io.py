"""Synthetic checks for P01's explicit H5 channel and physical-unit binding."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from experiments.p01.fusion_data import read_records
from experiments.p01.window_io import window_record
from src.data_factory.H5DataDict import H5DataDict


RPM_PATTERN = r"(?P<rpm>[0-9]+(?:\.[0-9]+)?)rpm, [0-9.]+Nm, [0-9.]+N"


@pytest.fixture
def h5_binding(tmp_path):
    """The three synthetic channels have deliberately disjoint value ranges."""
    waveform = np.stack(
        [np.arange(20) + offset for offset in (1000, 2000, 3000)], axis=1
    ).astype(np.float64)[..., None]
    rows, assignments = [], []
    groups = [
        ("bearing_a", "update", 0),
        ("bearing_b", "update", 1),
        ("bearing_c", "validation", 0),
        ("bearing_d", "assessment", 1),
        ("bearing_e", "test", 0),
    ]
    descriptions = {
        "0": "1500rpm, 0.7Nm, 1000N",
        "1": "900rpm, 0.7Nm, 1000N",
        "2": "1500rpm, 0.7Nm, 400N",
    }
    h5_path = tmp_path / "synthetic.h5"
    with h5py.File(h5_path, "w") as archive:
        for domain, description in descriptions.items():
            for group, split, label in groups:
                acquisition_id = str(len(rows) + 1)
                archive.create_dataset(acquisition_id, data=waveform)
                rows.append({
                    "Id": acquisition_id,
                    "Label": label,
                    "Domain_id": domain,
                    "Domain_description": description,
                    "Sample_rate": 64000,
                })
                assignments.append({
                    "Id": acquisition_id, "Physical_group": group, "Partition": split,
                })
    metadata_path = tmp_path / "metadata.csv"
    protocol_path = tmp_path / "protocol.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    pd.DataFrame(assignments).to_csv(protocol_path, index=False)
    dataset = {
        "format": "vibench_h5",
        "metadata_file": str(metadata_path),
        "protocol_file": str(protocol_path),
        "h5_file": str(h5_path),
        "source_domains": ["0", "1"],
        "domain_sequence": ["2"],
        "columns": {
            "id": "Id", "unit_id": "Physical_group", "split": "Partition",
            "label": "Label", "domain": "Domain_id",
            "sample_rate_hz": "Sample_rate",
            "rotation_speed_rpm": "Domain_description",
        },
        "rotation_speed_pattern": RPM_PATTERN,
    }
    data = {
        "layout": "LC", "squeeze_axes": [2], "channel_indices": [2],
        "window_size": 5, "windows_per_unit": 3,
    }
    return dataset, {"model": {"num_classes": 2}, "data": data}, waveform


def test_vibration_channel_is_selected_without_axis_or_value_repair(h5_binding):
    dataset, config, waveform = h5_binding
    record = {"path": dataset["h5_file"], "h5_key": "1"}

    windows = window_record(record, dataset, config["data"])

    expected = np.stack([waveform[start:start + 5, 2, 0] for start in (0, 7, 15)])
    assert windows.shape == (3, 5, 1)
    assert windows.dtype == torch.float32
    torch.testing.assert_close(windows[:, :, 0], torch.from_numpy(expected).float(),
                               rtol=0, atol=0)


@pytest.mark.parametrize("channels", [[], [3], [-1], [2, 2], [True], [2.0]])
def test_invalid_declared_channels_fail(h5_binding, channels):
    dataset, config, _ = h5_binding
    config["data"]["channel_indices"] = channels
    with pytest.raises(ValueError, match="channel_indices"):
        window_record({"path": dataset["h5_file"], "h5_key": "1"}, dataset,
                      config["data"])


def test_metadata_preflight_reads_keys_but_never_waveforms(h5_binding, monkeypatch):
    dataset, config, _ = h5_binding
    observed_keys = []
    original_contains = H5DataDict.__contains__

    def contains(signals, key):
        observed_keys.append(key)
        return original_contains(signals, key)

    def forbid_waveform(signals, key):
        raise AssertionError(f"Metadata preflight read waveform {key}")

    monkeypatch.setattr(H5DataDict, "__contains__", contains)
    monkeypatch.setattr(H5DataDict, "__getitem__", forbid_waveform)
    records = read_records(dataset, config)

    assert observed_keys == [str(i) for i in range(1, 16)]
    assert len(records) == 15
    assert {record["sample_rate_hz"] for record in records} == {64000.0}
    assert {record["domain"]: record["rotation_speed_rpm"] for record in records} == {
        "0": 1500.0, "1": 900.0, "2": 1500.0,
    }
    assert {record["unit_id"] for record in records} == {
        "bearing_a", "bearing_b", "bearing_c", "bearing_d", "bearing_e",
    }
    assert len({record["acquisition_id"] for record in records}) == 15


@pytest.mark.parametrize("description", [
    "1500, 0.7Nm, 1000N",
    "1500Hz, 0.7Nm, 1000N",
    "prefix 1500rpm, 0.7Nm, 1000N",
    "1500rpm, 0.7Nm, 1000N trailing",
])
def test_rpm_extraction_requires_the_complete_declared_unit_pattern(h5_binding, description):
    dataset, config, _ = h5_binding
    frame = pd.read_csv(dataset["metadata_file"], dtype=str)
    frame.loc[0, "Domain_description"] = description
    frame.to_csv(dataset["metadata_file"], index=False)

    with pytest.raises(ValueError, match="RPM metadata does not match"):
        read_records(dataset, config)


def test_rpm_pattern_must_name_the_measured_quantity(h5_binding):
    dataset, config, _ = h5_binding
    dataset["rotation_speed_pattern"] = r"([0-9]+)rpm, [0-9.]+Nm, [0-9.]+N"
    with pytest.raises(ValueError, match="named rpm group"):
        read_records(dataset, config)


def test_runtime_imports_need_only_benchmark_root_from_other_cwd(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from experiments.p01.window_io import window_record; "
        "from experiments.p01.fusion_data import read_records; "
        "assert callable(window_record) and callable(read_records)"
    )
    result = subprocess.run([sys.executable, "-c", program, str(root)], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
