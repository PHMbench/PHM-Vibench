from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from src.Pipeline_06_generative import _condition_counts, _select_condition
from src.config_schema.models import ExperimentConfig


def _metadata():
    return {
        1: {"Label": 0, "Domain_id": 0},
        2: {"Label": 1, "Domain_id": 1},
    }


def _base_config(generative: dict) -> dict:
    return {
        "pipeline": "Pipeline_06_generative",
        "environment": {"project": "x", "output_dir": "results", "seed": 0, "iterations": 1},
        "data": {"data_dir": "data", "metadata_file": "metadata_dummy.csv"},
        "model": {"type": "generative_model", "name": "phm_cfm_mlp1d"},
        "task": {
            "type": "generative",
            "name": "conditional_flow_matching",
            "generative": generative,
        },
        "trainer": {"name": "Default_trainer", "device": "cpu", "gpus": 0, "num_epochs": 1},
    }


def test_grid_condition_sampling_counts_all_pairs() -> None:
    gen_cfg = SimpleNamespace(
        condition_sampling_policy="grid",
        condition_grid=SimpleNamespace(
            fault_label=[0, 1],
            domain_id=[0, 1],
            samples_per_condition=3,
        ),
    )

    condition = _select_condition(gen_cfg, _metadata(), 2, 0, "cpu")

    assert condition["fault_label"].shape == (12,)
    assert condition["domain_id"].shape == (12,)
    assert _condition_counts(condition) == {
        "fault=0,domain=0": 3,
        "fault=0,domain=1": 3,
        "fault=1,domain=0": 3,
        "fault=1,domain=1": 3,
    }


def test_explicit_condition_sampling_preserves_counts() -> None:
    gen_cfg = SimpleNamespace(
        condition_sampling_policy="explicit",
        explicit_conditions=[
            SimpleNamespace(fault_label=0, domain_id=1, count=2),
            SimpleNamespace(fault_label=1, domain_id=0, count=4),
        ],
    )

    condition = _select_condition(gen_cfg, _metadata(), 99, 0, "cpu")

    assert _condition_counts(condition) == {
        "fault=0,domain=1": 2,
        "fault=1,domain=0": 4,
    }


def test_train_distribution_sampling_uses_metadata_pairs() -> None:
    gen_cfg = SimpleNamespace(condition_sampling_policy="train_distribution")

    condition = _select_condition(gen_cfg, _metadata(), 20, 123, "cpu")

    counts = _condition_counts(condition)
    assert sum(counts.values()) == 20
    assert set(counts).issubset({"fault=0,domain=0", "fault=1,domain=1"})


def test_first_metadata_repeated_keeps_smoke_behavior() -> None:
    gen_cfg = SimpleNamespace(condition_sampling_policy="first_metadata_repeated")

    condition = _select_condition(gen_cfg, _metadata(), 5, 0, "cpu")

    assert torch.equal(condition["fault_label"], torch.zeros(5, dtype=torch.long))
    assert torch.equal(condition["domain_id"], torch.zeros(5, dtype=torch.long))


def test_invalid_condition_policy_fails_schema_validation() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            _base_config({"mode": "train", "condition_sampling_policy": "unknown"})
        )


def test_grid_policy_requires_grid_schema() -> None:
    with pytest.raises(ValidationError, match="condition_grid"):
        ExperimentConfig.model_validate(
            _base_config({"mode": "train", "condition_sampling_policy": "grid"})
        )


def test_explicit_policy_requires_condition_rows_schema() -> None:
    with pytest.raises(ValidationError, match="explicit_conditions"):
        ExperimentConfig.model_validate(
            _base_config({"mode": "train", "condition_sampling_policy": "explicit"})
        )
