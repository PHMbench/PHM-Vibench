"""Synthetic contract checks; no industrial accuracy or qualification evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from src.data_factory.p08_data import ConditionEncoder, load_records, split_records, windows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inventory(tmp_path):
    records, ontology = [], []
    for system in ("001", "013", "019"):
        for label, definition in enumerate(("healthy", "inner race fault", "outer race fault")):
            ontology.append(dict(system_id=system, raw_label=f"0{label}", label=label,
                                 definition=definition, source="fixture definition"))
            for unit in range(3):
                record_id = f"{system}-{unit}-{label}"
                records.append(dict(record_id=record_id, system_id=system,
                                    physical_unit_id=f"0{unit}", raw_label=f"0{label}",
                                    sampling_rate=12000, signal_path=f"{record_id}.npy",
                                    signal_key="001", channel=1, rpm=1000 + 100 * unit,
                                    material="NA" if unit == 0 else "steel"))
    _write_csv(tmp_path / "records.csv", records)
    _write_csv(tmp_path / "ontology.csv", ontology)
    config = dict(evidence_kind="tensor_fixture", record_inventory=str(tmp_path / "records.csv"),
                  ontology_file=str(tmp_path / "ontology.csv"), window_points=4,
                  stride_points=3, normalization="none")
    return config, records, ontology


def test_metadata_is_lazy_and_preserves_lexical_identity(inventory):
    config, _, _ = inventory
    with patch("src.data_factory.p08_data.H5DataDict") as h5, patch("numpy.load") as npy:
        records = load_records(config)
    h5.assert_not_called()
    npy.assert_not_called()
    assert records[0]["system_id"] == "001"
    assert records[0]["physical_unit_id"] == "00"
    assert records[0]["signal_key"] == "001"
    assert records[0]["raw_label"] == "00"
    assert records[0]["material"] == "NA"
    assert records[0]["label"] == 0
    assert Path(records[0]["signal_path"]).parent == Path(config["record_inventory"]).parent


@pytest.mark.parametrize("field,value,message", [
    ("sampling_rate", "nan", "finite numeric"), ("sampling_rate", "inf", "finite numeric"),
    ("sampling_rate", "0", "strictly positive"), ("channel", "0.5", "integer"),
    ("channel", "-1", "integer"), ("physical_unit_id", "", "nonempty"),
    ("raw_label", "undeclared", "ontology mapping"),
])
def test_invalid_inventory_fails_before_signal_access(inventory, field, value, message):
    config, rows, _ = inventory
    rows[0][field] = value
    _write_csv(Path(config["record_inventory"]), rows)
    with patch("numpy.load") as npy, pytest.raises(ValueError, match=message):
        load_records(config)
    npy.assert_not_called()


@pytest.mark.parametrize("conflict", ["record_id", "raw_channel", "cross_unit"])
def test_duplicate_or_conflicting_record_identity_is_rejected(inventory, conflict):
    config, rows, _ = inventory
    if conflict == "record_id":
        rows[1]["record_id"] = rows[0]["record_id"]
        match = "Duplicate record_id"
    elif conflict == "raw_channel":
        rows[1].update(signal_path=rows[0]["signal_path"], physical_unit_id=rows[0]["physical_unit_id"],
                       signal_key="different-ignored-npy-key")
        match = "Duplicate signal identity"
    else:
        rows[1].update(signal_path=rows[0]["signal_path"], channel=0)
        match = "conflicting system/unit/label ownership"
    _write_csv(Path(config["record_inventory"]), rows)
    with pytest.raises(ValueError, match=match):
        load_records(config)


def test_h5_absolute_key_cannot_alias_another_unit(inventory):
    config, rows, _ = inventory
    rows[0]["signal_path"] = "same.h5"
    rows[1].update(signal_path="same.h5", signal_key="/001")
    _write_csv(Path(config["record_inventory"]), rows)
    with pytest.raises(ValueError, match="conflicting system/unit/label ownership"):
        load_records(config)


@pytest.mark.parametrize("change,message", [
    ("meaning", "conflicting physical definitions"), ("fraction", "integer"),
    ("classes", "numbered 0..C-1"), ("source", "ontology.source"),
])
def test_ontology_must_define_shared_documented_physical_classes(inventory, change, message):
    config, _, rows = inventory
    if change == "meaning":
        rows[3]["definition"] = "different fault"
    elif change == "fraction":
        rows[0]["label"] = "0.5"
    elif change == "classes":
        for row in rows:
            if row["label"] == 2:
                row["label"] = 3
    else:
        rows[0]["source"] = ""
    _write_csv(Path(config["ontology_file"]), rows)
    with pytest.raises(ValueError, match=message):
        load_records(config)


def test_industrial_requires_approved_per_system_qualification(inventory):
    config, rows, _ = inventory
    config["evidence_kind"] = "industrial"
    for row in rows:
        row["signal_path"] = row["signal_path"].replace(".npy", ".h5")
    _write_csv(Path(config["record_inventory"]), rows)
    with pytest.raises(ValueError, match="CSV path"):
        load_records(config)
    qualification = [dict(system_id=system, approved="true", ontology_source="defined classes",
                          group_source="physical unit log", channel_source="sensor layout",
                          condition_source="deployment measured rpm/material with units")
                     for system in ("001", "013", "019")]
    path = Path(config["record_inventory"]).parent / "qualification.csv"
    config["qualification_file"] = str(path)
    _write_csv(path, qualification)
    assert len(load_records(config)) == len(rows)
    qualification[1]["approved"] = "False"
    _write_csv(path, qualification)
    with pytest.raises(ValueError, match="not approved"):
        load_records(config)
    _write_csv(path, qualification[:1])
    with pytest.raises(ValueError, match="Missing approved qualification"):
        load_records(config)


def test_split_is_deterministic_grouped_target_excluded_and_complete(inventory):
    records = load_records(inventory[0])
    train, val, test = split_records(records, "019", 0.3, 42)
    assert (train, val, test) == split_records(list(reversed(records)), "019", 0.3, 42)
    groups = lambda rows: {(r["system_id"], r["physical_unit_id"]) for r in rows}
    assert groups(train).isdisjoint(groups(val) | groups(test))
    assert groups(val).isdisjoint(groups(test))
    assert {r["system_id"] for r in train} == {"001", "013"}
    assert {r["system_id"] for r in val} == {"001", "013"}
    assert {r["system_id"] for r in test} == {"019"}
    assert sorted(r["record_id"] for r in train + val + test) == sorted(r["record_id"] for r in records)


@pytest.mark.parametrize("fraction", [0, 1, 0.99, float("nan")])
def test_invalid_or_empty_source_partition_fails(inventory, fraction):
    with pytest.raises(ValueError):
        split_records(load_records(inventory[0]), "019", fraction, 42)


def test_split_cannot_exclude_a_common_class_from_source_training():
    records = [dict(record_id=f"{system}-{label}", system_id=system,
                    physical_unit_id=str(label), label=label)
               for system in ("source", "target") for label in range(3)]
    with pytest.raises(ValueError, match="loses common labels"):
        split_records(records, "target", 0.3, 42)


def test_condition_fit_rejects_target_and_zero_iqr(inventory):
    records = load_records(inventory[0])
    encoder = ConditionEncoder(["rpm"], ["material"])
    with pytest.raises(ValueError, match="rejects target"):
        encoder.fit(records, "019")
    with pytest.raises(ValueError, match="zero/nonfinite IQR"):
        encoder.fit([{**row, "rpm": 1000} for row in records if row["system_id"] != "019"], "019")


@pytest.mark.parametrize("field", ["system_id", "dataset", "Id", "signal_path", "raw_label",
                                   "sampling_rate", "sampling_frequency", "physical_unit_id", "DomainId",
                                   "signal_key", "recording_name", "filename"])
def test_identifiers_cannot_enter_condition_vectors(field):
    with pytest.raises(ValueError, match="disallowed information"):
        ConditionEncoder([field], [])


def test_target_transform_is_samplewise_and_does_not_refit(inventory):
    train, val, target = split_records(load_records(inventory[0]), "019", 0.3, 42)
    encoder = ConditionEncoder(["rpm"], ["material"]).fit(train, "019")
    state = encoder.state_dict()
    extreme = {**target[0], "rpm": 99999, "material": "unseen"}
    encoded = encoder.transform([extreme])
    assert torch.equal(encoded[0], encoder.transform([val[0], extreme, target[-1]])[1])
    assert encoded[0, 1:4].tolist() == [1, 0, 1]
    category_width = len(state["categorical"]["material"])
    assert encoded[0, 4 + category_width:].tolist() == [1, 1, 0]
    assert encoder.state_dict() == state
    restored = ConditionEncoder.from_state_dict(json.loads(json.dumps(state)))
    assert torch.equal(restored.transform([extreme]), encoded)
    assert encoder.transform([]).shape == (0, encoder.output_dim)


def test_missing_and_explicit_default_states_have_distinct_bits(inventory):
    train, _, target = split_records(load_records(inventory[0]), "019", 0.3, 42)
    encoder = ConditionEncoder(["rpm"], ["material"]).fit(train, "019")
    missing = encoder.transform(target, "missing")
    default = encoder.transform(target, "default")
    assert missing[0, :4].tolist() == [0, 0, 0, 0]
    assert default[0, :4].tolist() == [0, 0, 1, 0]
    assert missing[0, -3:].tolist() == [1, 0, 0]
    assert default[0, -3:].tolist() == [1, 0, 1]
    natural_missing = encoder.transform([{**target[0], "rpm": "", "material": None}])
    assert torch.equal(natural_missing, missing[:1])


def test_serialized_encoder_rejects_target_in_fitted_sources(inventory):
    train, _, _ = split_records(load_records(inventory[0]), "019", 0.3, 42)
    state = ConditionEncoder(["rpm"], ["material"]).fit(train, "019").state_dict()
    state["source_systems"].append("019")
    with pytest.raises(ValueError, match="target-excluded source systems"):
        ConditionEncoder.from_state_dict(state)


def test_restored_numeric_stats_are_canonical_and_categories_remain_strings(inventory):
    train, _, target = split_records(load_records(inventory[0]), "019", 0.3, 42)
    encoder = ConditionEncoder(["rpm"], ["material"]).fit(train, "019")
    state = encoder.state_dict()
    state["continuous"]["rpm"]["median"] = str(state["continuous"]["rpm"]["median"])
    restored = ConditionEncoder.from_state_dict(state)
    assert torch.equal(restored.transform(target), encoder.transform(target))
    state["categorical"]["material"] = [1, 2]
    with pytest.raises(ValueError, match="Invalid source vocabulary"):
        ConditionEncoder.from_state_dict(state)


def test_explicit_invalid_condition_values_are_not_treated_as_missing(inventory):
    train, _, target = split_records(load_records(inventory[0]), "019", 0.3, 42)
    encoder = ConditionEncoder(["rpm"], ["material"]).fit(train, "019")
    with pytest.raises(ValueError, match="finite numeric"):
        encoder.transform([{**target[0], "rpm": "nan"}])


def test_complete_windows_preserve_selected_channel_and_no_padding(inventory):
    config = inventory[0]
    record = load_records(config)[0]
    raw = np.stack([100 + np.arange(12), np.arange(12)], axis=1)
    np.save(record["signal_path"], raw)
    result = windows(record, config)
    assert result.shape == (3, 4, 1)
    assert result[..., 0].tolist() == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
    normalized = windows(record, {**config, "normalization": "per_window_standardize"})
    torch.testing.assert_close(normalized.mean(1), torch.zeros(3, 1), atol=1e-7, rtol=0)
    torch.testing.assert_close(normalized.std(1, correction=0), torch.ones(3, 1))
    assert np.array_equal(np.load(record["signal_path"]), raw)


def test_h5_exact_key_and_channel_are_used(inventory, tmp_path):
    config = inventory[0]
    record = load_records(config)[0]
    path = tmp_path / "signal.h5"
    with h5py.File(path, "w", libver="latest") as file:
        file.create_dataset("001", data=np.arange(24).reshape(12, 2))
        file.create_dataset("1", data=np.zeros((12, 2)))
    result = windows({**record, "signal_path": str(path)}, config)
    assert result[0, :, 0].tolist() == [1, 3, 5, 7]


@pytest.mark.parametrize("scenario,message", [
    ("channel", "cannot supply channel"), ("short", "no complete window"),
    ("nan", "nonfinite"), ("constant", "constant/nonfinite"),
])
def test_invalid_window_inputs_fail_without_fallback(inventory, scenario, message):
    config = inventory[0]
    record = load_records(config)[0]
    raw = np.stack([np.arange(12), np.arange(12)], axis=1).astype(float)
    if scenario == "channel":
        record["channel"] = 2
    elif scenario == "short":
        raw = raw[:3]
    elif scenario == "nan":
        raw[0, 1] = np.nan
    else:
        raw[:] = 1
        config["normalization"] = "per_window_standardize"
    np.save(record["signal_path"], raw)
    with pytest.raises(ValueError, match=message):
        windows(record, config)


def test_shared_reader_explicit_dtype_preserves_numeric_looking_identifiers(tmp_path):
    from src.data_factory.data_utils import read_metadata_table
    table = tmp_path / "typed.csv"
    table.write_text("record_id,label\n001,0\n002,1\n")
    typed = read_metadata_table(table, dtype={"record_id": "string"})
    assert typed.record_id.tolist() == ["001", "002"]
    assert typed.label.tolist() == [0, 1]
    assert read_metadata_table(table).record_id.tolist() == [1, 2]
