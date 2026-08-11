from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.p04.run_g050_decisive import (
    PARTITIONS,
    _apply_probe,
    _checkpoint_contract,
    _decision,
    _delta_rid,
    _hungarian,
    _knockout_contrast,
    _load_admitted_data,
    _load_config,
    _window_starts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/p04/g050_decisive.yaml"


@pytest.fixture(scope="module")
def config():  # type: ignore[no-untyped-def]
    return _load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def admitted(config):  # type: ignore[no-untyped-def]
    return _load_admitted_data(config, None)


def test_exact_16_source_contract_is_disjoint_before_windowing(admitted) -> None:  # type: ignore[no-untyped-def]
    assert len(admitted.sources) == 16
    source_sets = {
        partition: {
            source.source_id
            for source in admitted.sources
            if source.partition == partition
        }
        for partition in PARTITIONS
    }
    assert all(len(values) == 4 for values in source_sets.values())
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            assert source_sets[left].isdisjoint(source_sets[right])
    for partition in PARTITIONS:
        assert len(admitted.partitions[partition]) == 116
        sources = [source for source in admitted.sources if source.partition == partition]
        assert {source.label for source in sources} == {0, 1, 2, 3}


def test_all_windows_are_exact_nonoverlapping_and_keep_source_metadata(admitted) -> None:  # type: ignore[no-untyped-def]
    for source in admitted.sources:
        assert len(source.window_starts) == 29
        assert source.window_starts[0] == 0
        assert source.window_starts[-1] + 4096 == source.signal_length
        assert all(
            right - left >= 4096
            for left, right in zip(source.window_starts, source.window_starts[1:])
        )
        assert source.sample_rate_hz == 12_000
        assert source.rotation_speed_rpm in {1797, 1772, 1750, 1730}
        assert source.load_hp == source.domain
    for partition in PARTITIONS:
        data = admitted.partitions[partition]
        assert data.x.shape == (116, 4096, 1)
        assert len(data.source_id) == 116
        assert torch.isfinite(data.x).all()
        assert torch.all(data.sample_rate_hz == 12_000)


def test_normalization_is_fitted_once_from_train_only(admitted) -> None:  # type: ignore[no-untyped-def]
    train = admitted.partitions["train"].x.double()
    # Reversing the frozen affine transform recovers the admitted train windows.
    reconstructed = train * admitted.normalization_std + admitted.normalization_mean
    assert float(reconstructed.mean()) == pytest.approx(
        admitted.normalization_mean, rel=0.0, abs=1e-8
    )
    assert float(reconstructed.std(unbiased=False)) == pytest.approx(
        admitted.normalization_std, rel=1e-7, abs=1e-8
    )
    assert admitted.contract["normalization"]["fit_partition"] == "train"
    assert admitted.contract["normalization"]["per_window_refit"] is False


def test_98_and_99_keep_nominal_and_missing_raw_rpm_distinct(admitted) -> None:  # type: ignore[no-untyped-def]
    by_file = {source.file_id: source for source in admitted.sources}
    assert by_file[98].raw_rpm is None
    assert by_file[99].raw_rpm is None
    assert by_file[98].rotation_speed_rpm == 1772
    assert by_file[99].rotation_speed_rpm == 1750
    assert all(
        by_file[file_id].raw_rpm is not None
        for file_id in set(by_file) - {98, 99}
    )


def test_config_freezes_disjoint_probe_ids_and_independent_slot_permutations(config) -> None:  # type: ignore[no-untyped-def]
    match_ids = {probe["id"] for probe in config["probes"]["match"]}
    eval_ids = {probe["id"] for probe in config["probes"]["eval"]}
    assert match_ids.isdisjoint(eval_ids)
    permutations = config["protocol"]["slot_permutation_by_seed"]
    assert set(permutations) == {20, 21, 22}
    assert len({tuple(value) for value in permutations.values()}) == 3
    assert all(sorted(value) == [0, 1, 2, 3] for value in permutations.values())


def test_probe_generation_consumes_sampling_rate_and_rotation_speed() -> None:
    x = torch.ones(2, 512, 1)
    physical = {
        "sample_rate_hz": torch.tensor([12_000.0, 24_000.0]),
        "rotation_speed_rpm": torch.tensor([1_800.0, 3_600.0]),
        "load_hp": torch.tensor([0.0, 1.0]),
    }
    probe = {"transform": "low_order", "order": 2.0}
    transformed = _apply_probe(
        x, physical, probe, relative_rms=0.2, batch_offset=0
    )
    torch.testing.assert_close(transformed[0], transformed[1])


def test_contract_rejects_unknown_label_map_before_data_use(config) -> None:  # type: ignore[no-untyped-def]
    invalid = copy.deepcopy(config)
    invalid["data"]["label_map"].pop(3)
    with pytest.raises(ValueError, match="label map"):
        _load_admitted_data(invalid, None)


def test_contract_rejects_ambiguous_partition_map(config) -> None:  # type: ignore[no-untyped-def]
    invalid = copy.deepcopy(config)
    invalid["data"]["partition_by_domain"][3] = "P_match"
    with pytest.raises(ValueError, match="domain partition map"):
        _load_admitted_data(invalid, None)


@pytest.mark.parametrize(
    ("length", "window", "count"),
    [(10_000, 4096, 3), (8191, 4096, 2)],
)
def test_window_contract_rejects_overlap(length: int, window: int, count: int) -> None:
    with pytest.raises(ValueError, match="non-overlapping"):
        _window_starts(length, window, count)


def test_primary_estimand_helpers_match_frozen_equations() -> None:
    assert _delta_rid(0.9, 0.4, 0.7, 0.5) == pytest.approx(0.2)
    assert _knockout_contrast([1.0, 0.2, 0.3, 0.4], 0) == pytest.approx(0.7)
    assignment = _hungarian(
        np.asarray(
            [
                [0.1, 0.9, 0.0, 0.0],
                [0.8, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.7, 0.2],
                [0.0, 0.0, 0.1, 0.8],
            ]
        )
    )
    assert assignment == [1, 0, 2, 3]


def _decision_fixture(p0: float, p1: float, p2: float, null: float, delta_int: float):  # type: ignore[no-untyped-def]
    pairs = [
        {
            "held_out_similarity": value,
            "random_assignment_null_mean": null,
            "probe_label_permutation_null_mean": null,
        }
        for value in (p0 - 0.01, p0, p0 + 0.01)
    ]
    matching = {
        "arms": {
            "P0": {"RID_HO": p0, "seed_pairs": pairs},
            "P1": {
                "RID_HO": p1,
                "seed_pairs": [{"held_out_similarity": p1}] * 3,
            },
            "P2": {
                "RID_HO": p2,
                "seed_pairs": [{"held_out_similarity": p2}] * 3,
            },
        },
        "P0_null_mean": null,
        "delta_rid": _delta_rid(p0, p1, p2, null),
    }
    interventions = {"delta_int": delta_int}
    runs = {
        ("P0", seed): {
            "partition_metrics": {
                "P_eval": {"collapsed": False, "maximum_expert_usage": 0.4}
            }
        }
        for seed in (20, 21, 22)
    }
    return matching, interventions, runs


def test_decision_mapping_distinguishes_continue_reposition_and_stop() -> None:
    assert _decision(*_decision_fixture(0.9, 0.4, 0.5, 0.3, 0.1))["decision"] == "continue"
    assert _decision(*_decision_fixture(0.9, 0.4, 0.5, 0.3, -0.1))["decision"] == "reposition"
    # Generic capacity or shuffled alignment explaining P0 requires stop/merge,
    # even when P0 remains above its random null.
    assert _decision(*_decision_fixture(0.7, 0.8, 0.6, 0.3, 0.1))["decision"] == "stop_or_merge"


def test_checkpoint_contract_binds_run_kind_and_physics(config) -> None:  # type: ignore[no-untyped-def]
    smoke = _checkpoint_contract(config, "P0", 20, 1)
    pilot = _checkpoint_contract(config, "P0", 20, 15)
    assert smoke["run_kind"] == "smoke"
    assert pilot["run_kind"] == "pilot"
    assert smoke != pilot
    assert pilot["model"]["compatibility_alpha"] == 1.0
    assert pilot["model"]["low_order_cutoff"] == 4.0
    assert pilot["data"]["speed"]["mode"] == "nominal_by_domain"


def test_config_rejects_literal_match_eval_probe_reuse(config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    invalid = copy.deepcopy(config)
    reused = copy.deepcopy(invalid["probes"]["match"][0])
    reused["id"] = invalid["probes"]["eval"][0]["id"]
    invalid["probes"]["eval"][0] = reused
    path = tmp_path / "invalid_probe.yaml"
    path.write_text(yaml.safe_dump(invalid, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="identical transform"):
        _load_config(path)
