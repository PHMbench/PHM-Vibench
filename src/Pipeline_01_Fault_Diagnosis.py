"""Maintained fault-diagnosis Pipeline using the shared classification runtime."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from src.runtime import (
    ClassificationContext,
    ClassificationHooks,
    run_classification_pipeline,
)


_RENDERER_FIELDS = (
    "n_fft",
    "hop_length",
    "win_length",
    "window",
    "window_periodic",
    "center",
    "pad_mode",
    "normalized",
    "onesided",
    "representation",
    "scaling",
    "resize",
    "normalization",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_ready(item)
            for key, item in vars(value).items()
        }
    if hasattr(value, "item"):
        return value.item()
    return value


def _renderer_from_config(args_model: Any) -> dict[str, Any]:
    renderer = getattr(args_model, "renderer", None)
    if renderer is None:
        raise ValueError("P01 grouped protocol requires model.renderer configuration.")
    missing = [field for field in _RENDERER_FIELDS if not hasattr(renderer, field)]
    if missing:
        raise ValueError(f"P01 renderer configuration is missing field(s) {missing}.")
    return {field: getattr(renderer, field) for field in _RENDERER_FIELDS}


def build_p01_data_protocol_summary(
    data_factory: Any,
    args_model: Any,
    model: Any = None,
    loader_probe: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    grouped = getattr(data_factory, "grouped_protocol", None)
    split = getattr(data_factory, "split_summary", None)
    if not grouped or not isinstance(split, dict):
        raise ValueError("P01 data protocol summary requires an active grouped split.")

    renderer = _renderer_from_config(args_model)
    if model is not None:
        identity = getattr(model, "renderer_identity", None)
        observed = identity() if callable(identity) else None
        if observed is not None and observed != renderer:
            raise ValueError(
                "Executed model renderer differs from the frozen renderer config."
            )

    return _json_ready(
        {
            "status": "succeeded",
            "scope": "C01_data_protocol_only",
            "endpoint": {
                "name": grouped["endpoint"],
                "admitted_labels": grouped["admitted_labels"],
                "excluded_label_0_reason": grouped[
                    "excluded_label_0_reason"
                ],
                "inferential_unit": grouped["inferential_unit"],
                "verified_run_identity": grouped["verified_run_identity"],
                "observation_hierarchy": grouped["observation_hierarchy"],
                "identity_limit": grouped["identity_limit"],
                "target_label_access_boundary": grouped[
                    "target_label_access_boundary"
                ],
            },
            "split": split,
            "renderer": {
                "identity": renderer,
                "matched_conditions": ["M2", "M3", "M4", "M5"],
                "data_fitting_boundary": (
                    "none: renderer parameters are frozen configuration values"
                ),
            },
            "loader_probe": loader_probe,
            "provenance": provenance,
            "scientific_boundary": (
                "This artifact validates identity, grouping, fitting boundaries, "
                "and paired-view construction only; it contains no performance result."
            ),
        }
    )


def write_p01_data_protocol_summary(
    path: str | Path,
    data_factory: Any,
    args_model: Any,
    model: Any = None,
    loader_probe: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_p01_data_protocol_summary(
        data_factory,
        args_model,
        model=model,
        loader_probe=loader_probe,
        provenance=provenance,
    )
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def _trainable_parameters(module: Any) -> int:
    if module is None:
        return 0
    return sum(
        int(parameter.numel())
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _p01_parameter_counts(model: Any) -> dict[str, int]:
    names = (
        "encoder_1d",
        "project_1d",
        "renderer",
        "encoder_2d",
        "project_2d",
        "attention",
        "head",
    )
    counts = {
        name: _trainable_parameters(getattr(model, name, None))
        for name in names
    }
    counts["total"] = _trainable_parameters(model)
    if sum(counts[name] for name in names) != counts["total"]:
        raise RuntimeError("P01 parameter component counts do not sum to total")
    return counts


def _macro_f1(
    truth: list[int], predictions: list[int], label_order: tuple[int, ...]
) -> float:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Macro-F1 requires paired, non-empty observations")
    scores = []
    for label in label_order:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(truth, predictions)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(truth, predictions)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(truth, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return float(sum(scores) / len(scores))


def _best_checkpoint_path(trainer: Any) -> str:
    candidates = [
        str(getattr(callback, "best_model_path", "") or "")
        for callback in getattr(trainer, "callbacks", [])
        if hasattr(callback, "best_model_path")
    ]
    paths = [path for path in candidates if path]
    if len(paths) != 1:
        raise RuntimeError(
            "P01 grouped evaluation requires exactly one selected checkpoint path"
        )
    return paths[0]


def build_p01_grouped_result_rows(
    context: ClassificationContext,
) -> list[dict[str, Any]]:
    """Build C02 rows from frozen-checkpoint predictions at the group boundary."""
    grouped_evaluation = getattr(
        context.args_task, "grouped_evaluation", None
    )
    if grouped_evaluation is None or not bool(
        getattr(grouped_evaluation, "enabled", False)
    ):
        if context.result is None:
            raise RuntimeError("P01 result rows require trainer.test output")
        return [dict(context.result)]

    condition = str(getattr(context.args_model, "condition", ""))
    if condition not in {"M1", "M2"}:
        raise ValueError(
            "C02 grouped evaluation admits only the M1 and M2 conditions"
        )
    model = context.model
    if condition == "M1":
        forbidden = ("renderer", "encoder_2d", "project_2d")
        view_path = "waveform_1d_encoder_only"
        renderer_identity: Any = {
            "status": "not_applicable",
            "reason": "M1 has no 2D renderer or 2D encoder branch",
        }
    else:
        forbidden = ("encoder_1d", "project_1d")
        view_path = "deterministic_renderer_then_2d_encoder_only"
        renderer_identity = getattr(model, "renderer_identity")()
    present = [name for name in forbidden if getattr(model, name, None) is not None]
    if present:
        raise RuntimeError(
            f"Condition {condition} unexpectedly contains forbidden branch(es) {present}"
        )

    identity_reader = getattr(context.task, "label_contract_identity", None)
    label_identity = identity_reader() if callable(identity_reader) else None
    if not isinstance(label_identity, dict):
        raise RuntimeError("C02 grouped evaluation requires a label contract")
    raw_label_order = tuple(int(value) for value in label_identity["raw_labels"])
    training_indices = tuple(
        int(value) for value in label_identity["training_indices"]
    )
    if training_indices != tuple(range(len(raw_label_order))):
        raise RuntimeError("C02 training indices must be contiguous from zero")

    expected_aggregation = (
        "mean_softmax_windows_then_argmax_per_domain_condition_block"
    )
    aggregation = str(getattr(grouped_evaluation, "aggregation", ""))
    if aggregation != expected_aggregation:
        raise ValueError(
            "Unsupported P01 grouped aggregation; expected "
            f"{expected_aggregation!r}"
        )
    metric_name = str(getattr(grouped_evaluation, "primary_metric", ""))
    if metric_name != "condition_block_macro_f1":
        raise ValueError(
            "C02 grouped evaluation primary_metric must be "
            "'condition_block_macro_f1'"
        )

    record_reader = getattr(context.task, "grouped_evaluation_records", None)
    records = record_reader() if callable(record_reader) else None
    if not isinstance(records, list) or not records:
        raise RuntimeError(
            "Post-checkpoint test produced no grouped evaluation records"
        )

    target_domains = tuple(
        int(value) for value in getattr(context.args_task, "target_domain_id", [])
    )
    if len(target_domains) != 2 or len(set(target_domains)) != 2:
        raise ValueError("C02 requires exactly two distinct target domains")
    observed_domains = {int(record["domain_id"]) for record in records}
    if observed_domains != set(target_domains):
        raise ValueError(
            "Grouped test records do not exactly match configured target domains: "
            f"observed={sorted(observed_domains)}, expected={sorted(target_domains)}"
        )

    grouped_records: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        domain_id = int(record["domain_id"])
        group_id = str(record["physical_group_id"])
        raw_label = int(record["raw_label"])
        training_label = int(record["training_label"])
        if raw_label not in raw_label_order:
            raise ValueError(f"Grouped record has inadmissible raw label {raw_label}")
        expected_training_label = raw_label_order.index(raw_label)
        if training_label != expected_training_label:
            raise ValueError(
                "Grouped record raw/training labels violate the frozen bijection"
            )
        grouped_records[(domain_id, group_id)].append(record)

    expected_groups = int(
        getattr(grouped_evaluation, "required_groups_per_domain", 0)
    )
    expected_per_class = int(
        getattr(grouped_evaluation, "required_groups_per_class_domain", 0)
    )
    expected_windows = int(
        getattr(grouped_evaluation, "required_windows_per_group_domain", 0)
    )
    if min(expected_groups, expected_per_class, expected_windows) <= 0:
        raise ValueError("C02 grouped support requirements must be positive")

    parameter_counts = _p01_parameter_counts(model)
    checkpoint_path = _best_checkpoint_path(context.trainer)
    seed = int(context.args_environment.seed) + int(context.iteration)
    grouped_split = getattr(context.args_task, "grouped_split", None)
    group_key = str(getattr(grouped_split, "group_key", ""))
    if not group_key:
        raise ValueError("C02 grouped evaluation requires grouped_split.group_key")

    common = {
        "run_scope": "C02_unimodal_reference_exploratory",
        "condition_id": condition,
        "run_stage": str(getattr(grouped_evaluation, "run_stage", "")),
        "dataset": "CWRU",
        "seed": seed,
        "iteration": int(context.iteration),
        "status": "completed",
        "primary_metric_name": metric_name,
        "independent_group_key": group_key,
        "aggregation": aggregation,
        "raw_label_order": "|".join(map(str, raw_label_order)),
        "training_index_order": "|".join(map(str, training_indices)),
        "label_mapping_json": json.dumps(
            _json_ready(label_identity), sort_keys=True, separators=(",", ":")
        ),
        "view_path": view_path,
        "trainable_parameters": parameter_counts["total"],
        "classifier_head_parameters": parameter_counts["head"],
        "parameter_counts_json": json.dumps(
            parameter_counts, sort_keys=True, separators=(",", ":")
        ),
        "renderer_identity_json": json.dumps(
            _json_ready(renderer_identity), sort_keys=True, separators=(",", ":")
        ),
        "checkpoint_path": checkpoint_path,
        "checkpoint_selection": str(
            getattr(grouped_evaluation, "checkpoint_selection", "")
        ),
        "training_epochs": int(context.args_trainer.num_epochs),
        "optimizer": str(context.args_task.optimizer),
        "learning_rate": float(context.args_task.lr),
        "weight_decay": float(getattr(context.args_task, "weight_decay", 0.0)),
        "pooled_window_metrics_json": json.dumps(
            _json_ready(context.result or {}), sort_keys=True, separators=(",", ":")
        ),
        "scientific_boundary": (
            "single-seed exploratory held-condition/load-domain reference; "
            "windows, files, batches, and load domains are not independent "
            "repetitions and this row cannot promote a paper claim"
        ),
    }

    rows: list[dict[str, Any]] = []
    domain_values: list[float] = []
    predictions_by_domain: dict[str, list[dict[str, Any]]] = {}
    total_windows = 0
    canonical_groups: set[str] | None = None
    support_text = "|".join(
        f"{label}:{expected_per_class}" for label in raw_label_order
    )

    for domain_id in target_domains:
        domain_groups = sorted(
            group_id
            for candidate_domain, group_id in grouped_records
            if candidate_domain == domain_id
        )
        if len(domain_groups) != expected_groups:
            raise ValueError(
                f"Target domain {domain_id} has {len(domain_groups)} groups; "
                f"expected {expected_groups}"
            )
        if canonical_groups is None:
            canonical_groups = set(domain_groups)
        elif set(domain_groups) != canonical_groups:
            raise ValueError(
                "Target domains must evaluate the same documented condition blocks"
            )

        truth: list[int] = []
        predictions: list[int] = []
        group_predictions: list[dict[str, Any]] = []
        for group_id in domain_groups:
            group = grouped_records[(domain_id, group_id)]
            if len(group) != expected_windows:
                raise ValueError(
                    f"Domain {domain_id} group {group_id!r} has {len(group)} "
                    f"windows; expected {expected_windows}"
                )
            file_ids = {str(record["file_id"]) for record in group}
            if len(file_ids) != 1:
                raise ValueError(
                    "Each C02 domain/condition-block cell must contain exactly "
                    "one verified File/run"
                )
            raw_labels = {int(record["raw_label"]) for record in group}
            if len(raw_labels) != 1:
                raise ValueError("A condition block contains mixed raw labels")
            logits = torch.as_tensor(
                [record["logits"] for record in group], dtype=torch.float64
            )
            if logits.shape != (expected_windows, len(raw_label_order)) or not bool(
                torch.isfinite(logits).all().item()
            ):
                raise ValueError(
                    "Grouped logits do not match the frozen window/class contract"
                )
            mean_probability = torch.softmax(logits, dim=1).mean(dim=0)
            predicted_index = int(torch.argmax(mean_probability).item())
            predicted_raw = raw_label_order[predicted_index]
            actual_raw = next(iter(raw_labels))
            truth.append(actual_raw)
            predictions.append(predicted_raw)
            group_predictions.append(
                {
                    "physical_group_id": group_id,
                    "file_id": next(iter(file_ids)),
                    "raw_label": actual_raw,
                    "predicted_raw_label": predicted_raw,
                    "mean_probability": [
                        round(float(value), 10) for value in mean_probability.tolist()
                    ],
                }
            )

        support = {label: truth.count(label) for label in raw_label_order}
        if set(support.values()) != {expected_per_class}:
            raise ValueError(
                f"Target domain {domain_id} class-group support is {support}; "
                f"expected {expected_per_class} for every class"
            )
        value = _macro_f1(truth, predictions, raw_label_order)
        if not math.isfinite(value):
            raise FloatingPointError("Grouped macro-F1 is not finite")
        domain_values.append(value)
        predictions_by_domain[str(domain_id)] = group_predictions
        domain_windows = sum(
            len(grouped_records[(domain_id, group_id)])
            for group_id in domain_groups
        )
        total_windows += domain_windows
        rows.append(
            {
                **common,
                "target_domain": domain_id,
                "target_environment": f"CWRU_{domain_id}_HP",
                "primary_metric_value": value,
                "group_count": expected_groups,
                "class_group_support": support_text,
                "window_count": domain_windows,
                "evaluated_domain_count": 1,
                "group_predictions_json": json.dumps(
                    group_predictions,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )

    rows.append(
        {
            **common,
            "target_domain": "mean_" + "_".join(map(str, target_domains)),
            "target_environment": "mean_across_target_load_domains",
            "primary_metric_value": float(sum(domain_values) / len(domain_values)),
            "group_count": expected_groups,
            "class_group_support": support_text,
            "window_count": total_windows,
            "evaluated_domain_count": len(target_domains),
            "group_predictions_json": json.dumps(
                predictions_by_domain,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    return rows


class _P01DataProtocolHooks(ClassificationHooks):
    def on_iteration_start(self, context: ClassificationContext) -> None:
        grouped_evaluation = getattr(
            context.args_task, "grouped_evaluation", None
        )
        if grouped_evaluation is None or not bool(
            getattr(grouped_evaluation, "enabled", False)
        ):
            return
        expected_devices = str(
            getattr(grouped_evaluation, "required_cuda_visible_devices", "")
        )
        observed_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not expected_devices or observed_devices != expected_devices:
            raise RuntimeError(
                "C02 GPU contract requires CUDA_VISIBLE_DEVICES="
                f"{expected_devices!r}, got {observed_devices!r}"
            )
        if (
            str(getattr(context.args_trainer, "device", "")) != "cuda"
            or int(getattr(context.args_trainer, "gpus", 0)) != 1
        ):
            raise RuntimeError("C02 requires trainer.device=cuda and trainer.gpus=1")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "C02 requires exactly one visible CUDA device and forbids CPU fallback"
            )

    def after_stack_built(self, context: ClassificationContext) -> None:
        if getattr(context.data_factory, "grouped_protocol", None) is None:
            return
        if str(getattr(context.args_model, "name", "")) != "P01Alignment":
            return
        path = write_p01_data_protocol_summary(
            context.path / "data_protocol_summary.json",
            context.data_factory,
            context.args_model,
            model=context.model,
        )
        print(f"[P01 DATA PROTOCOL] {path}")

    def build_result_rows(
        self, context: ClassificationContext
    ) -> list[dict[str, Any]]:
        return build_p01_grouped_result_rows(context)


def pipeline(args: Any) -> list[dict[str, Any]]:
    """Run the standard classification train/test lifecycle."""
    return run_classification_pipeline(args, hooks=_P01DataProtocolHooks())
