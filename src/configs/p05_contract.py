"""Fail-closed phase and arm validation for frozen P05 GPU experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


REGISTERED_FINAL_SEEDS = frozenset({42, 123, 456, 789, 1024})
REGISTERED_LEARNING_RATES = frozenset({1.0e-3, 3.0e-4})
REGISTERED_NEURAL_ARMS = frozenset({"P05-M", "P05-B0", "P05-B1", "P05-B3"})


@dataclass(frozen=True)
class P05ExperimentContract:
    arm_id: str
    dataset: str
    dataset_id: int
    phase: str
    seed: int
    trace_export: bool


def _get(value: Any, dotted: str, default: Any = None) -> Any:
    current = value
    for part in dotted.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    return current


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"P05 requires {name}={expected!r}, got {value!r}")


def _float(value: Any, expected: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"P05 {name} must be numeric")
    if not math.isfinite(float(value)) or not math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(f"P05 requires {name}={expected!r}, got {value!r}")


def _validate_common(
    args_environment: Any,
    args_data: Any,
    args_model: Any,
    args_task: Any,
    args_trainer: Any,
) -> tuple[str, str, int, str]:
    arm_id = _get(args_task, "p05_arm_id")
    if arm_id not in REGISTERED_NEURAL_ARMS:
        raise ValueError(f"unregistered P05 neural arm: {arm_id!r}")
    phase = _get(args_task, "p05_run_phase")
    if phase not in {"pilot", "tuning", "decisive"}:
        raise ValueError("task.p05_run_phase must be pilot, tuning, or decisive")
    target_ids = _get(args_task, "target_system_id")
    if target_ids == [1]:
        dataset, dataset_id, classes, windows = "CWRU", 1, 4, 16
    elif target_ids == [2]:
        dataset, dataset_id, classes, windows = "XJTU", 2, 2, 4
    else:
        raise ValueError("P05 target_system_id must be exactly [1] or [2]")

    exact_values = (
        (_get(args_environment, "iterations"), 1, "environment.iterations"),
        (_get(args_environment, "wandb"), False, "environment.wandb"),
        (_get(args_environment, "swanlab"), False, "environment.swanlab"),
        (_get(args_data, "p05_evidence_mode"), True, "data.p05_evidence_mode"),
        (_get(args_data, "allow_download"), False, "data.allow_download"),
        (_get(args_data, "cache_mode"), "read_only_verified", "data.cache_mode"),
        (_get(args_data, "batch_size"), 64, "data.batch_size"),
        (_get(args_data, "window_size"), 4096, "data.window_size"),
        (_get(args_data, "stride"), 4096, "data.stride"),
        (
            _get(args_data, "window_sampling_strategy"),
            "evenly_spaced",
            "data.window_sampling_strategy",
        ),
        (
            _get(args_data, "normalization"),
            "train_channel_standardization",
            "data.normalization",
        ),
        (_get(args_data, "dtype"), "float32", "data.dtype"),
        (_get(args_data, "num_workers"), 0, "data.num_workers"),
        (_get(args_data, "drop_last_train"), False, "data.drop_last_train"),
        (_get(args_data, "noise_snr"), None, "data.noise_snr"),
        (
            _get(args_data, "split_strategy"),
            "preassigned_metadata",
            "data.split_strategy",
        ),
        (
            _get(args_data, "split.strategy"),
            "preassigned_metadata",
            "data.split.strategy",
        ),
        (_get(args_data, "split.split_key"), "Protocol_Split", "data.split.split_key"),
        (_get(args_data, "split.group_key"), "Protocol_Group", "data.split.group_key"),
        (_get(args_data, "split.seed"), 20260801, "data.split.seed"),
        (_get(args_data, "split.test_policy"), "partition", "data.split.test_policy"),
        (_get(args_model, "type"), "X_model", "model.type"),
        (_get(args_model, "name"), "TSPN_UXFD", "model.name"),
        (_get(args_model, "in_dim"), 4096, "model.in_dim"),
        (_get(args_model, "out_dim"), 4096, "model.out_dim"),
        (_get(args_model, "in_channels"), 2, "model.in_channels"),
        (_get(args_model, "out_channels"), 4, "model.out_channels"),
        (_get(args_model, "scale"), 1, "model.scale"),
        (_get(args_model, "skip_connection"), True, "model.skip_connection"),
        (
            _get(args_model, "internal_instance_normalization"),
            False,
            "model.internal_instance_normalization",
        ),
        (_get(args_model, "device"), "cuda", "model.device"),
        (_get(args_model, "num_classes"), classes, "model.num_classes"),
        (_get(args_data, "num_window"), windows, "data.num_window"),
        (_get(args_task, "type"), "Default_task", "task.type"),
        (_get(args_task, "name"), "Default_task", "task.name"),
        (_get(args_task, "loss"), "CE_weighted", "task.loss"),
        (_get(args_task, "metrics"), ["acc", "f1_macro"], "task.metrics"),
        (_get(args_task, "optimizer"), "adam", "task.optimizer"),
        (_get(args_task, "scheduler"), None, "task.scheduler"),
        (_get(args_task, "p05_evidence_mode"), True, "task.p05_evidence_mode"),
        (_get(args_task, "sample_weight_key"), "sample_weight", "task.sample_weight_key"),
        (_get(args_trainer, "p05_evidence_mode"), True, "trainer.p05_evidence_mode"),
        (_get(args_trainer, "name"), "Default_trainer", "trainer.name"),
        (_get(args_trainer, "device"), "cuda", "trainer.device"),
        (_get(args_trainer, "accelerator"), "gpu", "trainer.accelerator"),
        (_get(args_trainer, "devices"), 1, "trainer.devices"),
        (_get(args_trainer, "gpus"), 1, "trainer.gpus"),
        (_get(args_trainer, "num_nodes"), 1, "trainer.num_nodes"),
        (_get(args_trainer, "num_processes"), 1, "trainer.num_processes"),
        (_get(args_trainer, "strategy"), "auto", "trainer.strategy"),
        (_get(args_trainer, "precision"), 32, "trainer.precision"),
        (_get(args_trainer, "deterministic"), True, "trainer.deterministic"),
        (_get(args_trainer, "monitor"), "val_loss", "trainer.monitor"),
        (_get(args_trainer, "monitor_mode"), "min", "trainer.monitor_mode"),
        (_get(args_trainer, "save_top_k"), 1, "trainer.save_top_k"),
    )
    for actual, expected, name in exact_values:
        _exact(actual, expected, name=name)
    _float(_get(args_data, "train_ratio"), 0.8, name="data.train_ratio")
    _float(_get(args_task, "weight_decay"), 1.0e-4, name="task.weight_decay")
    signal_configs = _get(args_model, "signal_processing_configs")
    signal_keys = (
        set(signal_configs)
        if isinstance(signal_configs, dict)
        else set(vars(signal_configs)) if hasattr(signal_configs, "__dict__") else set()
    )
    if signal_keys != {"layer1"} or _get(
        args_model, "signal_processing_configs.layer1"
    ) != ["I"]:
        raise ValueError("P05 requires model.signal_processing_configs={layer1:[I]}")
    if _get(args_model, "feature_extractor_configs") != ["Mean", "Std"]:
        raise ValueError("P05 requires model.feature_extractor_configs=[Mean,Std]")
    return arm_id, phase, dataset_id, dataset


def _validate_arm(args_model: Any, *, arm_id: str, classes: int) -> None:
    fuzzy = _get(args_model, "uxfd.fuzzy.enable", False)
    neural = _get(args_model, "uxfd.neural_residual.enable", False)
    anfis = _get(args_model, "uxfd.anfis.enable", False)
    logic = _get(args_model, "uxfd.logic.enable", False)
    operator_attention = _get(args_model, "uxfd.operator_attention.enable", False)
    enable_sp2d = _get(args_model, "uxfd.enable_sp2d", False)
    for name, value in (
        ("fuzzy.enable", fuzzy),
        ("neural_residual.enable", neural),
        ("anfis.enable", anfis),
        ("logic.enable", logic),
        ("operator_attention.enable", operator_attention),
        ("enable_sp2d", enable_sp2d),
    ):
        if type(value) is not bool:
            raise TypeError(f"model.uxfd.{name} must be a literal boolean")
    expected = {
        "P05-M": (True, False, False),
        "P05-B0": (False, False, False),
        "P05-B1": (False, True, False),
        "P05-B3": (False, False, True),
    }[arm_id]
    if (fuzzy, neural, anfis) != expected:
        raise ValueError(f"P05 arm {arm_id} has an invalid head-enable tuple")
    if logic or operator_attention or enable_sp2d:
        raise ValueError("P05 central arms forbid logic, operator attention, and SP2D")

    if arm_id in {"P05-M", "P05-B0"}:
        frozen_fuzzy = {
            "num_fuzzy_features": 8,
            "num_membership_functions": 3,
            "num_rules": 10,
            "logit_scale": 0.5,
            "antecedent_temperature": 1.0,
            "min_width": 1.0e-4,
            "firing_epsilon": 1.0e-12,
        }
        for name, expected_value in frozen_fuzzy.items():
            actual = _get(args_model, f"uxfd.fuzzy.{name}")
            if isinstance(expected_value, float):
                _float(actual, expected_value, name=f"model.uxfd.fuzzy.{name}")
            else:
                _exact(actual, expected_value, name=f"model.uxfd.fuzzy.{name}")
    if arm_id == "P05-B1":
        expected_hidden = 29 if classes == 2 else 26
        configured = _get(args_model, "uxfd.neural_residual.hidden_dim", None)
        _exact(
            configured,
            expected_hidden,
            name="model.uxfd.neural_residual.hidden_dim",
        )
    if arm_id == "P05-B3":
        frozen_anfis = {
            "num_features": 8,
            "num_membership_functions": 3,
            "num_rules": 10,
            "antecedent_temperature": 1.0,
            "min_width": 1.0e-4,
            "firing_epsilon": 1.0e-12,
        }
        for name, expected_value in frozen_anfis.items():
            actual = _get(args_model, f"uxfd.anfis.{name}")
            if isinstance(expected_value, float):
                _float(actual, expected_value, name=f"model.uxfd.anfis.{name}")
            else:
                _exact(actual, expected_value, name=f"model.uxfd.anfis.{name}")


def validate_p05_experiment_contract(
    args_environment: Any,
    args_data: Any,
    args_model: Any,
    args_task: Any,
    args_trainer: Any,
    runtime_contract: Any,
) -> P05ExperimentContract | None:
    """Validate a complete frozen P05 experiment before data/model construction."""

    if runtime_contract is None:
        if _get(args_task, "p05_evidence_mode", False) is True:
            raise ValueError("task P05 evidence mode requires an accepted runtime preflight")
        return None
    arm_id, phase, dataset_id, dataset = _validate_common(
        args_environment,
        args_data,
        args_model,
        args_task,
        args_trainer,
    )
    classes = 2 if dataset_id == 2 else 4
    _validate_arm(args_model, arm_id=arm_id, classes=classes)
    seed = _get(args_environment, "seed")
    if type(seed) is not int or seed < 0:
        raise TypeError("P05 environment.seed must be a non-negative integer")
    lr = _get(args_task, "lr")
    trace_export = _get(args_task, "p05_trace_export")
    if type(trace_export) is not bool:
        raise TypeError("task.p05_trace_export must be a literal boolean")

    if phase == "pilot":
        _exact(seed, 20260801, name="environment.seed")
        _float(lr, 1.0e-3, name="task.lr")
        _exact(_get(args_environment, "stage"), "fit_validate_only", name="environment.stage")
        _exact(_get(args_trainer, "num_epochs"), 5, name="trainer.num_epochs")
        _exact(_get(args_trainer, "early_stopping"), False, name="trainer.early_stopping")
        _exact(_get(args_trainer, "p05_pilot_mode"), True, name="trainer.p05_pilot_mode")
        if arm_id not in {"P05-M", "P05-B0"}:
            raise ValueError("P05 pilot is frozen to P05-M and P05-B0")
        expected_trace = arm_id == "P05-M"
    elif phase == "tuning":
        _exact(seed, 20260801, name="environment.seed")
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or float(lr) not in REGISTERED_LEARNING_RATES
        ):
            raise ValueError("P05 tuning lr must be 1e-3 or 3e-4")
        _exact(_get(args_environment, "stage"), "fit_validate_only", name="environment.stage")
        _exact(_get(args_trainer, "num_epochs"), 60, name="trainer.num_epochs")
        _exact(_get(args_trainer, "early_stopping"), True, name="trainer.early_stopping")
        _exact(_get(args_trainer, "patience"), 10, name="trainer.patience")
        _exact(_get(args_trainer, "p05_pilot_mode"), False, name="trainer.p05_pilot_mode")
        expected_trace = False
    else:
        if seed not in REGISTERED_FINAL_SEEDS:
            raise ValueError("P05 decisive seed is not registered")
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or float(lr) not in REGISTERED_LEARNING_RATES
        ):
            raise ValueError("P05 decisive lr must be a registered tuned candidate")
        _exact(_get(args_environment, "stage"), "fit_validate_test", name="environment.stage")
        _exact(_get(args_trainer, "num_epochs"), 100, name="trainer.num_epochs")
        _exact(_get(args_trainer, "early_stopping"), True, name="trainer.early_stopping")
        _exact(_get(args_trainer, "patience"), 15, name="trainer.patience")
        _exact(_get(args_trainer, "p05_pilot_mode"), False, name="trainer.p05_pilot_mode")
        expected_trace = arm_id == "P05-M"
    if trace_export is not expected_trace:
        raise ValueError(
            f"P05 {phase} arm {arm_id} requires p05_trace_export={expected_trace}"
        )
    return P05ExperimentContract(
        arm_id=arm_id,
        dataset=dataset,
        dataset_id=dataset_id,
        phase=phase,
        seed=seed,
        trace_export=trace_export,
    )


__all__ = [
    "P05ExperimentContract",
    "REGISTERED_FINAL_SEEDS",
    "REGISTERED_LEARNING_RATES",
    "REGISTERED_NEURAL_ARMS",
    "validate_p05_experiment_contract",
]
