"""Synthetic MAT association checks; these are not PU qualification evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from scripts import tii_pu_observations as diagnostic


def write_mat(path: Path, *, xindex=2, raster="HostService", times=None, values=None,
              unit=np.array([]), missing_xindex=False):
    time = np.array([0., .1, .21, .3]) if times is None else np.asarray(times)
    signal = dict(Name="vibration_1", Raster="HostService", XIndex=xindex,
                  DownSampling=1, Unit=unit, Data=np.arange(4.) if values is None else values)
    if missing_xindex:
        del signal["XIndex"]
    axes = [dict(Name="mechanical", Raster="Mechanical", Unit="s", Data=np.arange(4.)),
            dict(Name=np.array([]), Raster=raster, Unit=np.array([]), Data=time)]
    other = dict(Name="current", Raster="Mechanical", XIndex=1, Unit="A", Data=np.arange(4.))
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {path.stem: dict(Y=np.array([other, signal], dtype=object),
                                 X=np.array(axes, dtype=object))})


def observe(tmp_path, **kwargs):
    path = tmp_path / "N09_M07_F10_K001_1.mat"
    write_mat(path, **kwargs)
    return diagnostic.observe_record(path, dict(metadata_id=0, original_file=path.name,
        bearing_id="K001", operating_condition="N09_M07_F10", metadata_fs=10., metadata_length=4))


def population():
    metadata = pd.DataFrame(dict(Id=[47567, 0], Name=["RM_027_PU"] * 2, Dataset_id=[20] * 2,
        File=["K001/N09_M07_F10_K001_1.mat", "K002/N15_M07_F10_K002_1.mat"],
        Sample_rate=[10.] * 2, Sample_lenth=[4] * 2, Label=[0, 1]))
    inventory = pd.DataFrame(dict(file=["RM_027_PU.h5"] * 2, dataset=["RM_027_PU"] * 2,
        dataset_id=[20] * 2, metadata_id=[47567, 0], recording_id=[47567, 0],
        original_recording=metadata.File, group=["K001", "K002"], metadata_rate=[10.] * 2,
        metadata_length=[4] * 2, label=[0, 1]))
    return metadata, inventory


def test_name_index_raster_mapping_uses_linked_axis_and_preserves_empty_units(tmp_path):
    row = observe(tmp_path)
    assert row["observation_status"] == "observed"
    assert row["y_index_1based"] == 2 and row["x_index_1based"] == 2
    assert row["time_raster"] == row["signal_raster"] == "HostService"
    assert row["time_name_original"] == row["signal_unit_original"] == row["time_unit_original"] == "[]"
    assert row["signal_unit_status"] == row["time_unit_status"] == "empty"
    assert row["last_time"] == .3 and row["span_average_rate_per_stored_unit"] == 10.
    assert row["stored_dt_exactly_nonuniform"] is True
    assert "conditional" in row["rate_hz_status"]
    assert row["metadata_id"] == 0


@pytest.mark.parametrize("kwargs", [dict(xindex=0), dict(xindex=1.5), dict(xindex=3),
    dict(xindex=np.nan), dict(missing_xindex=True), dict(raster="different"),
    dict(times=[0., .1, .2]), dict(times=np.ones((2, 2)))])
def test_invalid_association_retains_row_and_never_computes_rate(tmp_path, kwargs):
    row = observe(tmp_path, **kwargs)
    assert row["metadata_id"] == 0 and row["n_signal"] == 4
    assert row["observation_status"] == "failed" and row["error"]
    assert row["time_association_status"] == "unresolved"
    assert row["span_average_rate_hz"] is None


@pytest.mark.parametrize("times,duplicates,reversals", [([0., .1, .1, .3], 1, 0), ([0., .2, .1, .3], 0, 1)])
def test_duplicate_or_reversed_stored_time_is_reported_without_repair(tmp_path, times, duplicates, reversals):
    row = observe(tmp_path, times=times)
    assert row["observation_status"] == "failed"
    assert row["duplicate_time_steps"] == duplicates and row["reversed_time_steps"] == reversals
    assert row["n_time"] == 4 and row["span_average_rate_hz"] is None


def test_nonfinite_axis_and_missing_unit_remain_explicit(tmp_path):
    row = observe(tmp_path, times=[0., np.nan, .2, .3])
    assert row["time_finite"] is False and row["observation_status"] == "failed"
    assert row["span_average_rate_hz"] is None
    original, status = diagnostic._unit(object())
    assert original is None and status == "absent"
    assert diagnostic._unit(type("Unit", (), {"Unit": ""})()) == ('""', "empty")


@pytest.mark.parametrize("changed", ["missing_id", "extra_id", "duplicate_id", "wrong_file", "missing_group"])
def test_inventory_population_changes_fail_before_mat_read(changed):
    metadata, inventory = population()
    if changed == "missing_id":
        metadata.loc[1, "Id"] = np.nan
    elif changed == "extra_id":
        metadata.loc[1, "Id"] = 1
    elif changed == "duplicate_id":
        inventory.loc[1, "metadata_id"] = 47567
    elif changed == "wrong_file":
        inventory.loc[1, "original_recording"] = "other.mat"
    else:
        inventory.loc[1, "group"] = np.nan
    with pytest.raises(ValueError):
        diagnostic.matched_records(metadata, inventory)


def test_cli_retains_failed_record_has_complete_coverage_and_refuses_overwrite(tmp_path, monkeypatch):
    metadata, inventory = population()
    root, output = tmp_path / "raw_inputs", tmp_path / "outputs"
    root.mkdir()
    metadata.to_csv(root / "metadata.csv", index=False)
    inventory_path = tmp_path / "inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    write_mat(root / "raw" / "RM_027_PU" / metadata.File.iloc[0])
    write_mat(root / "raw" / "RM_027_PU" / metadata.File.iloc[1], xindex=0)
    monkeypatch.setattr(diagnostic, "EXPECTED_RECORDS", 2)
    args = ["--data-root", str(root), "--metadata-file", "metadata.csv", "--inventory", str(inventory_path),
            "--output-dir", str(output)]
    assert diagnostic.main(args) == 1
    rows = pd.read_csv(output / "record_observations.csv")
    assert rows.metadata_id.tolist() == [47567, 0]
    assert rows.observation_status.tolist() == ["observed", "failed"]
    summary = json.loads((output / "observation_summary.json").read_text())
    assert summary["inspected_records"] == 2 and summary["coverage_complete"] is True
    assert summary["qualification"] == "UNRESOLVED" and summary["failed_records"] == 1
    assert set(p.name for p in output.iterdir()) == set(diagnostic.OUTPUTS)
    assert diagnostic.main(args) == 2
    assert pd.read_csv(output / "record_observations.csv").metadata_id.tolist() == [47567, 0]
