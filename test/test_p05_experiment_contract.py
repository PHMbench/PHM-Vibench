from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

from src.configs.p05_contract import validate_p05_experiment_contract


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _ns(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_ns(item) for item in value]
    return value


def _pilot(arm="P05-M", dataset="CWRU"):
    matrix = yaml.safe_load(
        open(
            "configs/experiments/p05/protocol/pilot_matrix_p05_v1.yaml",
            encoding="utf-8",
        )
    )
    job = next(
        value
        for value in matrix["jobs"]
        if value["arm"] == arm and value["dataset"] == dataset
    )

    def merge(left, right):
        result = deepcopy(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    config = merge(matrix["common_config"], matrix["datasets"][dataset]["config"])
    config = merge(config, matrix["arms"][arm]["config"])
    config = merge(config, job["config"])
    config["task"]["p05_run_phase"] = "pilot"
    sections = ("environment", "data", "model", "task", "trainer")
    return {name: _ns(config[name]) for name in sections}


def _validate(config):
    return validate_p05_experiment_contract(
        config["environment"],
        config["data"],
        config["model"],
        config["task"],
        config["trainer"],
        SimpleNamespace(runtime_identity={}),
    )


def _tuning_arm(arm: str, dataset: str):
    config = _pilot("P05-B0", dataset)
    classes = 2 if dataset == "XJTU" else 4
    config["task"].p05_arm_id = arm
    config["task"].p05_run_phase = "tuning"
    config["task"].p05_trace_export = False
    config["task"].lr = 3.0e-4
    config["trainer"].p05_pilot_mode = False
    config["trainer"].num_epochs = 60
    config["trainer"].early_stopping = True
    config["trainer"].patience = 10
    config["model"].uxfd.neural_residual = _ns(
        {
            "enable": arm == "P05-B1",
            "hidden_dim": 29 if classes == 2 else 26,
        }
    )
    config["model"].uxfd.anfis = _ns(
        {
            "enable": arm == "P05-B3",
            "num_features": 8,
            "num_membership_functions": 3,
            "num_rules": 10,
            "antecedent_temperature": 1.0,
            "min_width": 1.0e-4,
            "firing_epsilon": 1.0e-12,
        }
    )
    return config


@pytest.mark.parametrize("arm", ["P05-M", "P05-B0"])
@pytest.mark.parametrize("dataset", ["CWRU", "XJTU"])
def test_frozen_pilot_contract_is_accepted(arm, dataset) -> None:
    contract = _validate(_pilot(arm, dataset))

    assert contract is not None
    assert contract.arm_id == arm
    assert contract.dataset == dataset
    assert contract.phase == "pilot"
    assert contract.trace_export is (arm == "P05-M")


@pytest.mark.parametrize("arm", ["P05-B1", "P05-B3"])
@pytest.mark.parametrize("dataset", ["CWRU", "XJTU"])
def test_frozen_b1_b3_tuning_contract_is_accepted(arm, dataset) -> None:
    contract = _validate(_tuning_arm(arm, dataset))

    assert contract is not None
    assert contract.arm_id == arm
    assert contract.dataset == dataset
    assert contract.phase == "tuning"
    assert contract.trace_export is False


@pytest.mark.parametrize("dataset", ["CWRU", "XJTU"])
def test_b1_contract_rejects_hidden_width_drift(dataset) -> None:
    config = _tuning_arm("P05-B1", dataset)
    config["model"].uxfd.neural_residual.hidden_dim = float(
        config["model"].uxfd.neural_residual.hidden_dim
    )

    with pytest.raises(ValueError, match="neural_residual.hidden_dim"):
        _validate(config)


def test_b1_contract_requires_explicit_hidden_width() -> None:
    config = _tuning_arm("P05-B1", "XJTU")
    del config["model"].uxfd.neural_residual.hidden_dim

    with pytest.raises(ValueError, match="neural_residual.hidden_dim"):
        _validate(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_features", 7),
        ("num_membership_functions", 4),
        ("num_rules", 9),
        ("antecedent_temperature", 0.5),
        ("min_width", 1.0e-3),
        ("firing_epsilon", 1.0e-9),
    ],
)
def test_b3_contract_rejects_anfis_drift(field, value) -> None:
    config = _tuning_arm("P05-B3", "XJTU")
    setattr(config["model"].uxfd.anfis, field, value)

    with pytest.raises((TypeError, ValueError), match=f"anfis.{field}"):
        _validate(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("task", "loss"), "CE", "CE_weighted"),
        (("environment", "seed"), 42, "20260801"),
        (("trainer", "num_epochs"), 4, "5"),
        (("trainer", "p05_pilot_mode"), False, "p05_pilot_mode"),
        (("task", "p05_trace_export"), False, "p05_trace_export=True"),
        (("model", "internal_instance_normalization"), True, "False"),
    ],
)
def test_pilot_contract_rejects_scientific_drift(path, value, message) -> None:
    config = _pilot("P05-M", "XJTU")
    setattr(config[path[0]], path[1], value)

    with pytest.raises((TypeError, ValueError), match=message):
        _validate(config)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("environment", "wandb", True, "environment.wandb"),
        ("environment", "swanlab", True, "environment.swanlab"),
        ("data", "allow_download", True, "data.allow_download"),
        ("data", "cache_mode", "write", "data.cache_mode"),
        ("data", "batch_size", 32, "data.batch_size"),
        ("data", "window_size", 2048, "data.window_size"),
        ("data", "stride", 2048, "data.stride"),
        ("data", "num_workers", 1, "data.num_workers"),
        ("data", "drop_last_train", True, "data.drop_last_train"),
        ("data", "noise_snr", 30, "data.noise_snr"),
        ("model", "skip_connection", False, "model.skip_connection"),
        ("trainer", "precision", 16, "trainer.precision"),
    ],
)
def test_preconstruction_contract_rejects_io_and_runtime_drift(
    section, field, value, message
) -> None:
    config = _pilot("P05-M", "XJTU")
    setattr(config[section], field, value)

    with pytest.raises((TypeError, ValueError), match=message):
        _validate(config)


def test_tuning_and_decisive_phase_bind_stage_seed_and_trace() -> None:
    tuning = _pilot("P05-M", "XJTU")
    tuning["task"].p05_run_phase = "tuning"
    tuning["task"].p05_trace_export = False
    tuning["task"].lr = 3.0e-4
    tuning["trainer"].p05_pilot_mode = False
    tuning["trainer"].num_epochs = 60
    tuning["trainer"].early_stopping = True
    tuning["trainer"].patience = 10
    assert _validate(tuning).trace_export is False

    decisive = _pilot("P05-M", "XJTU")
    decisive["task"].p05_run_phase = "decisive"
    decisive["environment"].stage = "fit_validate_test"
    decisive["environment"].seed = 42
    decisive["trainer"].p05_pilot_mode = False
    decisive["trainer"].num_epochs = 100
    decisive["trainer"].early_stopping = True
    decisive["trainer"].patience = 15
    assert _validate(decisive).seed == 42

    decisive["environment"].seed = 43
    with pytest.raises(ValueError, match="seed"):
        _validate(decisive)


def test_non_evidence_configuration_is_ignored() -> None:
    assert (
        validate_p05_experiment_contract(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(p05_evidence_mode=False),
            SimpleNamespace(),
            None,
        )
        is None
    )
