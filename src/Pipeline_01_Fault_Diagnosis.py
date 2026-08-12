"""Maintained fault-diagnosis Pipeline using the shared classification runtime."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any

from src.runtime import (
    ClassificationContext,
    ClassificationHooks,
    run_classification_pipeline,
)
from src.p01_g040_contract import (
    _best_checkpoint_provenance,
    _code_file_hashes,
    _validate_evidence_runtime,
    _validate_p01_trainable_parameter_count,
    _write_trainer_metrics_manifest,
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

_C2_TARGET_CONTROL_IDENTITY = {
    "mode": "seeded_sattolo_derangement_after_batching",
    "seed": 31042,
    "algorithm": "sattolo_single_cycle",
    "stage": "train_after_batching",
    "operand": "alignment_target_z2_only",
    "affected_terms": [
        "physical_energy",
        "physical_spectral",
        "semantic",
        "geometric",
    ],
    "unaffected_terms": ["classification", "physical_parseval"],
    "classification_pairing": "synchronized_original_views",
    "semantic_mask_basis": "original_label_and_index_slots",
    "seed_key": "base_seed_plus_epoch_times_1000003_plus_batch_index",
    "rng_scope": "dedicated_cpu_generator_no_global_rng_mutation",
    "fixed_point_policy": "forbidden",
}

_C3_SELECTION_IDENTITY = {
    "selected_encoder_family": "time_frequency_2d",
    "selected_representation": "frozen_log_magnitude_hann_stft",
    "selection_rule": (
        "equal_absolute_parameter_and_supported_flop_deviation_from_m5_"
        "then_direct_rendering_tie_break"
    ),
    "tie_breaker": "directly_tests_the_deterministic_rendering_explanation",
    "duplicate_1d_parameters": 38_915,
    "duplicate_1d_supported_flops": 45_646_208,
    "duplicate_1d_flops_evidence": "derived_from_measured_m1_m2_m3_cpu_profiles",
    "duplicate_2d_parameters": 55_555,
    "duplicate_2d_supported_flops": 46_336_640,
    "duplicate_2d_flops_evidence": "measured_executed_c3_cpu_profile",
    "m5_parameters": 47_235,
    "m5_supported_flops": 45_991_424,
    "absolute_parameter_difference_tie": 8_320,
    "c1_parameter_tolerance_status": "both_outside_five_percent",
}

_C05_MODEL_CONDITIONS = {
    "M1": "M1",
    "M2": "M2",
    "M3": "M3",
    "M4": "M4",
    "M5": "M5",
    "C1": "M4",
    "C2": "M5",
    "C3": "C3",
}

_C05_RUN_IDENTITIES = {
    "RUN-0019": ("M1", "matrix_cell", ""),
    "RUN-0020": ("M2", "matrix_cell", ""),
    "RUN-0021": ("M3", "matrix_cell", ""),
    "RUN-0022": ("M4", "matrix_cell", ""),
    "RUN-0023": ("M5", "matrix_cell", ""),
    "RUN-0024": ("C1", "matrix_cell", ""),
    "RUN-0025": ("C2", "matrix_cell", ""),
    "RUN-0026": ("C3", "matrix_cell", ""),
    "RUN-0027": ("M1", "fresh_process_reproduction", "RUN-0019"),
    "RUN-0028": ("C2", "fresh_process_reproduction", "RUN-0025"),
}

_C06_RUN_IDENTITIES = {
    f"RUN-{run_number:04d}": (condition_id, seed)
    for run_number, seed, condition_id in (
        (29, 42, "M1"),
        (30, 42, "M2"),
        (31, 42, "M3"),
        (32, 42, "M4"),
        (33, 42, "M5"),
        (34, 42, "C1"),
        (35, 42, "C2"),
        (36, 42, "C3"),
        (37, 123, "M1"),
        (38, 123, "M2"),
        (39, 123, "M3"),
        (40, 123, "M4"),
        (41, 123, "M5"),
        (42, 123, "C1"),
        (43, 123, "C2"),
        (44, 123, "C3"),
        (45, 456, "M1"),
        (46, 456, "M2"),
        (47, 456, "M3"),
        (48, 456, "M4"),
        (49, 456, "M5"),
        (50, 456, "C1"),
        (51, 456, "C2"),
        (52, 456, "C3"),
    )
}

_C06_PREDECLARED_SEEDS = (42, 123, 456)
_C06_PREDECLARED_TARGET_DOMAINS = (2, 3)


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
    names = [
        "encoder_1d",
        "project_1d",
        "renderer",
        "encoder_2d",
        "project_2d",
        "attention",
        "head",
    ]
    duplicate_names = ("encoder_duplicate_2d", "project_duplicate_2d")
    if any(getattr(model, name, None) is not None for name in duplicate_names):
        names[5:5] = duplicate_names
    counts = {
        name: _trainable_parameters(getattr(model, name, None))
        for name in names
    }
    counts["total"] = _trainable_parameters(model)
    if sum(counts[name] for name in names) != counts["total"]:
        raise RuntimeError("P01 parameter component counts do not sum to total")
    return counts


def _profile_p01_learned_forward_flops(
    model: Any,
    *,
    window_size: int,
    batch_size: int,
) -> dict[str, Any]:
    """Count supported learned-forward FLOPs on one fixed CPU input."""

    if window_size < 1 or batch_size < 1:
        raise ValueError("P01 forward profile dimensions must be positive")
    profile_model = deepcopy(model).cpu().eval()
    in_channels = int(getattr(profile_model, "in_channels", 0))
    if in_channels < 1:
        raise ValueError("P01 forward profile requires model.in_channels")
    sample = torch.zeros(batch_size, window_size, in_channels, dtype=torch.float32)
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        counter = FlopCounterMode(display=False)
        with torch.inference_mode(), counter:
            logits = profile_model(sample)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    if logits.shape != (batch_size, int(getattr(profile_model, "num_classes", 0))):
        raise RuntimeError(
            "P01 forward profile produced an unexpected output shape "
            f"{tuple(logits.shape)}"
        )
    raw_counts = counter.get_flop_counts().get("Global", {})
    by_operator = {
        str(operator): int(value)
        for operator, value in raw_counts.items()
        if int(value) > 0
    }
    attention_interaction_flops = 0
    attention = getattr(profile_model, "attention", None)
    if attention is not None:
        tokens = 2
        heads = int(attention.num_heads)
        head_dim = int(attention.embed_dim) // heads
        # Two matrix multiplications (QK^T and attention-weighted V), using the
        # two-FLOPs-per-MAC convention. FlopCounterMode accounts for the Q/K/V
        # and output Linear projections but not these native-MHA interactions.
        attention_interaction_flops = (
            4 * batch_size * heads * tokens * tokens * head_dim
        )
        by_operator["explicit_two_token_attention_qk_av"] = (
            attention_interaction_flops
        )
    total = int(counter.get_total_flops()) + attention_interaction_flops
    if total <= 0:
        raise RuntimeError("P01 forward FLOP counter reported no supported operations")
    renderer = getattr(profile_model, "renderer", None)
    rendered_shape: list[int] | None = None
    if renderer is not None:
        with torch.inference_mode():
            rendered_shape = list(profile_model.render_2d_view(sample).shape)
    return {
        "learned_forward_supported_flops": total,
        "by_operator": dict(sorted(by_operator.items())),
        "batch_size": batch_size,
        "window_size": window_size,
        "input_shape": [batch_size, window_size, in_channels],
        "output_shape": list(logits.shape),
        "renderer_output_shape": rendered_shape,
        "torch_version": torch.__version__,
        "method": (
            "torch.utils.flop_counter.FlopCounterMode_cpu_plus_"
            "explicit_two_token_attention_qk_av"
        ),
        "scope": (
            "one float32 eval/inference model forward; the deterministic renderer "
            "executes, while its Hann/STFT/abs/log1p and unsupported normalization, "
            "activation, pooling, and softmax operations are excluded; counted "
            "learned operators use two FLOPs per MAC"
        ),
    }


def build_p01_forward_compute_profile(
    model: Any,
    args_model: Any,
    args_data: Any,
    grouped_evaluation: Any,
    *,
    condition_id: str,
) -> dict[str, Any]:
    """Compare one admitted P01 forward with frozen M5/M4 references."""

    profile_config = getattr(grouped_evaluation, "forward_compute", None)
    goal_id = str(getattr(grouped_evaluation, "goal_id", ""))
    if goal_id not in {"C03", "C04", "C05", "C06"}:
        raise ValueError(
            "P01 forward-compute profiling is admitted only for C03/C04/C05/C06"
        )
    if profile_config is None:
        raise ValueError(f"{goal_id} requires task.grouped_evaluation.forward_compute")
    method = str(getattr(profile_config, "method", ""))
    expected_method = (
        "torch.utils.flop_counter.FlopCounterMode_cpu_plus_"
        "explicit_two_token_attention_qk_av"
    )
    if method != expected_method:
        raise ValueError(
            f"{goal_id} forward_compute.method must be {expected_method!r}"
        )
    reference_condition = str(
        getattr(profile_config, "reference_condition", "")
    )
    if reference_condition != "M5":
        raise ValueError(f"{goal_id} forward-compute reference_condition must be M5")
    batch_size = int(getattr(profile_config, "batch_size", 0))
    window_size = int(getattr(args_data, "window_size", 0))
    parameter_tolerance = float(
        getattr(profile_config, "parameter_relative_tolerance", -1.0)
    )
    flops_tolerance = float(
        getattr(
            profile_config,
            "learned_forward_supported_flops_relative_tolerance",
            -1.0,
        )
    )
    if parameter_tolerance != 0.05 or flops_tolerance != 0.10:
        raise ValueError(
            f"{goal_id} requires the frozen 5% parameter and 10% forward-FLOP tolerances"
        )

    observed = _profile_p01_learned_forward_flops(
        model, window_size=window_size, batch_size=batch_size
    )
    reference_args = deepcopy(args_model)
    reference_args.condition = reference_condition
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        reference_model = build_model(reference_args, metadata=None)
    reference = _profile_p01_learned_forward_flops(
        reference_model, window_size=window_size, batch_size=batch_size
    )

    observed_parameters = _trainable_parameters(model)
    reference_parameters = _trainable_parameters(reference_model)
    parameter_deviation = abs(observed_parameters - reference_parameters) / float(
        reference_parameters
    )
    observed_flops = int(observed["learned_forward_supported_flops"])
    reference_flops = int(reference["learned_forward_supported_flops"])
    flops_deviation = abs(observed_flops - reference_flops) / float(
        reference_flops
    )
    matched = (
        parameter_deviation <= parameter_tolerance
        and flops_deviation <= flops_tolerance
    )
    if condition_id == "C1":
        adjustment = str(getattr(profile_config, "c1_adjustment", ""))
        if adjustment != "none_required_existing_M4_within_tolerances":
            raise ValueError("C1 requires the frozen zero-adjustment identity")
        if str(getattr(args_model, "condition", "")) != "M4":
            raise ValueError("C1 must execute the unchanged M4 model condition")
        if not matched:
            raise RuntimeError(
                "C1 fails the frozen parameter/forward-FLOP matching tolerances"
            )

    result = {
        "condition_id": condition_id,
        "model_condition": str(getattr(args_model, "condition", "")),
        "observed": observed,
        "m5_reference": reference,
        "observed_trainable_parameters": observed_parameters,
        "m5_reference_trainable_parameters": reference_parameters,
        "parameter_relative_deviation": parameter_deviation,
        "learned_forward_supported_flops_relative_deviation": flops_deviation,
        "parameter_relative_tolerance": parameter_tolerance,
        "learned_forward_supported_flops_relative_tolerance": flops_tolerance,
        "within_tolerances": matched,
    }
    if goal_id in {"C04", "C05", "C06"}:
        comparison_condition = str(
            getattr(profile_config, "comparison_condition", "")
        )
        if comparison_condition != "M4":
            raise ValueError(
                f"{goal_id} forward-compute comparison_condition must be M4"
            )
        comparison_args = deepcopy(args_model)
        comparison_args.condition = comparison_condition
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            comparison_model = build_model(comparison_args, metadata=None)
        comparison = _profile_p01_learned_forward_flops(
            comparison_model,
            window_size=window_size,
            batch_size=batch_size,
        )
        comparison_parameters = _trainable_parameters(comparison_model)
        comparison_flops = int(comparison["learned_forward_supported_flops"])
        result.update(
            {
                "m4_c1_reference": comparison,
                "m4_c1_reference_trainable_parameters": comparison_parameters,
                "parameter_signed_deviation_from_m4_c1": (
                    observed_parameters - comparison_parameters
                )
                / float(comparison_parameters),
                "learned_forward_supported_flops_signed_deviation_from_m4_c1": (
                    observed_flops - comparison_flops
                )
                / float(comparison_flops),
                "parameter_relative_deviation_from_m4_c1": abs(
                    observed_parameters - comparison_parameters
                )
                / float(comparison_parameters),
                "learned_forward_supported_flops_relative_deviation_from_m4_c1": abs(
                    observed_flops - comparison_flops
                )
                / float(comparison_flops),
            }
        )
    return result


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
    *,
    forward_compute_profile: dict[str, Any] | None = None,
    training_objective_summary: dict[str, Any] | None = None,
    view_gradient_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build P01 rows from frozen-checkpoint predictions at the group boundary."""
    grouped_evaluation = getattr(
        context.args_task, "grouped_evaluation", None
    )
    if grouped_evaluation is None or not bool(
        getattr(grouped_evaluation, "enabled", False)
    ):
        if context.result is None:
            raise RuntimeError("P01 result rows require trainer.test output")
        return [dict(context.result)]

    goal_id = str(getattr(grouped_evaluation, "goal_id", "C02"))
    condition = str(getattr(context.args_model, "condition", ""))
    condition_id = str(
        getattr(grouped_evaluation, "condition_id", condition)
    )
    if goal_id == "C02":
        if condition not in {"M1", "M2"} or condition_id != condition:
            raise ValueError(
                "C02 grouped evaluation admits only matching M1/M2 identities"
            )
    elif goal_id == "C03":
        expected_model_condition = {"M3": "M3", "M4": "M4", "C1": "M4"}
        if expected_model_condition.get(condition_id) != condition:
            raise ValueError(
                "C03 condition identity/model mismatch: "
                f"condition_id={condition_id!r}, model.condition={condition!r}"
            )
    elif goal_id == "C04":
        expected_model_condition = {"M5": "M5", "C2": "M5", "C3": "C3"}
        if expected_model_condition.get(condition_id) != condition:
            raise ValueError(
                "C04 condition identity/model mismatch: "
                f"condition_id={condition_id!r}, model.condition={condition!r}"
            )
        expected_run_id = {"M5": "RUN-0016", "C2": "RUN-0017", "C3": "RUN-0018"}
        if str(getattr(grouped_evaluation, "run_id", "")) != expected_run_id[
            condition_id
        ]:
            raise ValueError(f"C04 condition {condition_id} run_id mismatch")
    elif goal_id == "C05":
        if _C05_MODEL_CONDITIONS.get(condition_id) != condition:
            raise ValueError(
                "C05 condition identity/model mismatch: "
                f"condition_id={condition_id!r}, model.condition={condition!r}"
            )
        run_id = str(getattr(grouped_evaluation, "run_id", ""))
        run_role = str(getattr(grouped_evaluation, "run_role", ""))
        reproduction_of = str(
            getattr(grouped_evaluation, "reproduction_of", "")
        )
        expected_identity = _C05_RUN_IDENTITIES.get(run_id)
        if expected_identity != (condition_id, run_role, reproduction_of):
            raise ValueError(
                "C05 run identity mismatch: "
                f"run_id={run_id!r}, condition_id={condition_id!r}, "
                f"run_role={run_role!r}, reproduction_of={reproduction_of!r}"
            )
    elif goal_id == "C06":
        if _C05_MODEL_CONDITIONS.get(condition_id) != condition:
            raise ValueError(
                "C06 condition identity/model mismatch: "
                f"condition_id={condition_id!r}, model.condition={condition!r}"
            )
        run_id = str(getattr(grouped_evaluation, "run_id", ""))
        run_role = str(getattr(grouped_evaluation, "run_role", ""))
        reproduction_of = str(getattr(grouped_evaluation, "reproduction_of", ""))
        expected_identity = _C06_RUN_IDENTITIES.get(run_id)
        observed_seed = int(context.args_environment.seed) + int(context.iteration)
        if (
            expected_identity != (condition_id, observed_seed)
            or run_role != "matrix_cell"
            or reproduction_of
        ):
            raise ValueError(
                "C06 run identity mismatch: "
                f"run_id={run_id!r}, condition_id={condition_id!r}, "
                f"seed={observed_seed}, run_role={run_role!r}, "
                f"reproduction_of={reproduction_of!r}"
            )
        predeclared_seeds = tuple(
            int(value)
            for value in getattr(grouped_evaluation, "predeclared_seeds", ())
        )
        if predeclared_seeds != _C06_PREDECLARED_SEEDS:
            raise ValueError(
                "C06 predeclared seeds must be exactly "
                f"{list(_C06_PREDECLARED_SEEDS)}"
            )
        if observed_seed not in _C06_PREDECLARED_SEEDS:
            raise ValueError(
                f"C06 observed seed {observed_seed} is outside the frozen set "
                f"{list(_C06_PREDECLARED_SEEDS)}"
            )
        predeclared_domains = tuple(
            int(value)
            for value in getattr(
                grouped_evaluation, "predeclared_target_domains", ()
            )
        )
        if predeclared_domains != _C06_PREDECLARED_TARGET_DOMAINS:
            raise ValueError(
                "C06 predeclared target domains must be exactly "
                f"{list(_C06_PREDECLARED_TARGET_DOMAINS)}"
            )
    else:
        raise ValueError(f"Unsupported P01 grouped-evaluation goal {goal_id!r}")
    model = context.model
    if condition_id == "M1":
        forbidden = ("renderer", "encoder_2d", "project_2d")
        view_path = "waveform_1d_encoder_only"
        renderer_identity: Any = {
            "status": "not_applicable",
            "reason": "M1 has no 2D renderer or 2D encoder branch",
        }
    elif condition_id == "M2":
        forbidden = ("encoder_1d", "project_1d")
        view_path = "deterministic_renderer_then_2d_encoder_only"
        renderer_identity = getattr(model, "renderer_identity")()
    elif condition_id == "M3":
        forbidden = ("attention",)
        view_path = "waveform_1d_plus_deterministic_renderer_2d_concatenation"
        renderer_identity = getattr(model, "renderer_identity")()
    elif condition_id in {"M4", "C1"}:
        forbidden = ()
        view_path = (
            "waveform_1d_plus_deterministic_renderer_2d_two_token_self_attention"
        )
        renderer_identity = getattr(model, "renderer_identity")()
    elif condition_id in {"M5", "C2"}:
        forbidden = ("attention", "encoder_duplicate_2d", "project_duplicate_2d")
        view_path = (
            "waveform_1d_plus_deterministic_renderer_2d_concatenation_"
            "with_three_level_alignment_objective"
        )
        renderer_identity = getattr(model, "renderer_identity")()
    elif condition_id == "C3":
        forbidden = ("encoder_1d", "project_1d", "attention")
        view_path = (
            "one_frozen_deterministic_renderer_output_fed_to_two_independent_"
            "2d_encoder_projection_copies"
        )
        renderer_identity = getattr(model, "renderer_identity")()
    else:
        raise ValueError(f"Unsupported P01 condition_id {condition_id!r}")
    present = [name for name in forbidden if getattr(model, name, None) is not None]
    if present:
        raise RuntimeError(
            f"Condition {condition} unexpectedly contains forbidden branch(es) {present}"
        )
    if goal_id in {"C03", "C05", "C06"} and condition_id in {"M3", "M4", "C1"}:
        required = (
            "encoder_1d",
            "project_1d",
            "renderer",
            "encoder_2d",
            "project_2d",
        )
        missing = [name for name in required if getattr(model, name, None) is None]
        if missing:
            raise RuntimeError(
                f"{goal_id} condition {condition_id} is missing paired component(s) {missing}"
            )
        expects_attention = condition_id in {"M4", "C1"}
        if (getattr(model, "attention", None) is not None) is not expects_attention:
            raise RuntimeError(
                f"{goal_id} condition {condition_id} has the wrong attention identity"
            )
        if bool(getattr(model, "uses_alignment_objective", False)):
            raise RuntimeError(
                f"{goal_id} condition {condition_id} cannot consume alignment"
            )
        alignment_identity = getattr(model, "alignment_identity", None)
        if callable(alignment_identity) and alignment_identity() is not None:
            raise RuntimeError(
                f"{goal_id} condition {condition_id} unexpectedly has alignment configuration"
            )
    if goal_id in {"C05", "C06"} and condition_id in {"M1", "M2"}:
        if bool(getattr(model, "uses_alignment_objective", False)):
            raise RuntimeError(
                f"{goal_id} condition {condition_id} cannot consume alignment"
            )
        alignment_identity = getattr(model, "alignment_identity", None)
        if callable(alignment_identity) and alignment_identity() is not None:
            raise RuntimeError(
                f"{goal_id} condition {condition_id} unexpectedly has alignment configuration"
            )
    if goal_id in {"C04", "C05", "C06"} and condition_id in {"M5", "C2", "C3"}:
        control_reader = getattr(context.task, "alignment_target_control_identity", None)
        control_identity = control_reader() if callable(control_reader) else None
        if condition_id in {"M5", "C2"}:
            required = (
                "encoder_1d",
                "project_1d",
                "renderer",
                "encoder_2d",
                "project_2d",
            )
            missing = [name for name in required if getattr(model, name, None) is None]
            if missing:
                raise RuntimeError(
                    f"{goal_id} condition {condition_id} is missing paired component(s) {missing}"
                )
            if not bool(getattr(model, "uses_alignment_objective", False)):
                raise RuntimeError(
                    f"{goal_id} condition {condition_id} must consume alignment"
                )
            if not isinstance(getattr(model, "alignment_identity")(), dict):
                raise RuntimeError(
                    f"{goal_id} condition {condition_id} requires alignment coefficients"
                )
            expected_control = (
                _C2_TARGET_CONTROL_IDENTITY if condition_id == "C2" else None
            )
            if control_identity != expected_control:
                raise RuntimeError(
                    f"{goal_id} condition {condition_id} target-control identity mismatch"
                )
            if not isinstance(training_objective_summary, dict):
                raise RuntimeError(
                    f"{goal_id} condition {condition_id} requires an observed objective summary"
                )
        else:
            required = (
                "renderer",
                "encoder_2d",
                "project_2d",
                "encoder_duplicate_2d",
                "project_duplicate_2d",
            )
            missing = [name for name in required if getattr(model, name, None) is None]
            if missing:
                raise RuntimeError(f"C3 is missing duplicate-rendering component(s) {missing}")
            if bool(getattr(model, "uses_alignment_objective", False)):
                raise RuntimeError("C3 cannot consume alignment losses")
            if control_identity is not None:
                raise RuntimeError("C3 cannot carry an alignment target control")
            if not isinstance(training_objective_summary, dict):
                raise RuntimeError("C3 requires an observed classification objective summary")
            if (
                model.encoder_2d is model.encoder_duplicate_2d
                or model.project_2d is model.project_duplicate_2d
            ):
                raise RuntimeError("C3 requires two independent 2D module copies")
    else:
        control_identity = None

    identity_reader = getattr(context.task, "label_contract_identity", None)
    label_identity = identity_reader() if callable(identity_reader) else None
    if not isinstance(label_identity, dict):
        raise RuntimeError("P01 grouped evaluation requires a label contract")
    raw_label_order = tuple(int(value) for value in label_identity["raw_labels"])
    training_indices = tuple(
        int(value) for value in label_identity["training_indices"]
    )
    if training_indices != tuple(range(len(raw_label_order))):
        raise RuntimeError("P01 training indices must be contiguous from zero")

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
            "P01 grouped evaluation primary_metric must be "
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
        raise ValueError("P01 requires exactly two distinct target domains")
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
        raise ValueError("P01 grouped support requirements must be positive")

    parameter_counts = _p01_parameter_counts(model)
    checkpoint_path = _best_checkpoint_path(context.trainer)
    seed = int(context.args_environment.seed) + int(context.iteration)
    grouped_split = getattr(context.args_task, "grouped_split", None)
    group_key = str(getattr(grouped_split, "group_key", ""))
    if not group_key:
        raise ValueError("P01 grouped evaluation requires grouped_split.group_key")

    common = {
        "run_scope": (
            "C02_unimodal_reference_exploratory"
            if goal_id == "C02"
            else (
                "C03_generic_fusion_control_exploratory"
                if goal_id == "C03"
                else (
                    "C04_alignment_and_negative_control_execution_smoke"
                    if goal_id == "C04"
                    else (
                        "C05_one_seed_minimum_admission_pilot"
                        if goal_id == "C05"
                        else "C06_three_seed_two_environment_decisive_pilot"
                    )
                )
            )
        ),
        "condition_id": condition_id,
        "model_condition": condition,
        "run_stage": str(getattr(grouped_evaluation, "run_stage", "")),
        "dataset": "CWRU",
        "seed": seed,
        "iteration": int(context.iteration),
        "status": "succeeded",
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
            (
                "single-seed exploratory held-condition/load-domain reference"
                if goal_id == "C02"
                else (
                    "single-seed C03 execution/fairness smoke, not comparative evidence"
                    if goal_id == "C03"
                    else (
                        "single-seed one-epoch C04 execution/control smoke, not mechanism evidence"
                        if goal_id == "C04"
                        else (
                            "single-seed fixed-ten-epoch C05 admission pilot, not paper evidence"
                            if goal_id == "C05"
                            else (
                                "three predeclared optimization seeds under one frozen "
                                "held-condition/two-load-domain C06 protocol; first "
                                "performance-route evidence, not mechanism evidence"
                            )
                        )
                    )
                )
            )
            + (
                "; windows, files, batches, load domains, and repeated control "
                "identities are not independent repetitions; C06 may support "
                "only its predeclared performance-route decision and is "
                "insufficient for a mechanism claim"
                if goal_id == "C06"
                else "; windows, files, batches, load domains, and repeated control "
                "identities are not independent repetitions and this row cannot "
                "promote a paper claim"
            )
        ),
    }
    if goal_id in {"C03", "C04", "C05", "C06"}:
        if not isinstance(forward_compute_profile, dict):
            raise RuntimeError(f"{goal_id} result rows require a forward-compute profile")
        if forward_compute_profile.get("condition_id") != condition_id:
            raise RuntimeError(f"{goal_id} forward-compute profile condition mismatch")
        observed_profile = forward_compute_profile["observed"]
        reference_profile = forward_compute_profile["m5_reference"]
        tuning_trials = int(
            getattr(grouped_evaluation, "source_validation_tuning_trials", -1)
        )
        if tuning_trials != 0:
            raise ValueError(f"{goal_id} freezes source-validation tuning trials at zero")
        scheduler_config = getattr(context.args_task, "scheduler", None)
        if scheduler_config is not None:
            raise ValueError(f"{goal_id} freezes the learning-rate scheduler as none")
        common.update(
            {
                "alignment_terms_consumed": (
                    "physical|semantic|geometric"
                    if goal_id in {"C04", "C05", "C06"}
                    and condition_id in {"M5", "C2"}
                    else "none"
                ),
                "data_access": "source_domains_0_1_train_val_then_target_domains_2_3_post_checkpoint",
                "source_validation_tuning_trials": tuning_trials,
                "scheduler": "none",
                "early_stopping": bool(context.args_trainer.early_stopping),
                "learned_forward_supported_flops": int(
                    observed_profile["learned_forward_supported_flops"]
                ),
                "forward_profile_method": str(observed_profile["method"]),
                "forward_profile_scope": str(observed_profile["scope"]),
                "forward_profile_input_shape_json": json.dumps(
                    observed_profile["input_shape"], separators=(",", ":")
                ),
                "forward_profile_output_shape_json": json.dumps(
                    observed_profile["output_shape"], separators=(",", ":")
                ),
                "renderer_output_shape_json": json.dumps(
                    observed_profile["renderer_output_shape"], separators=(",", ":")
                ),
                "forward_profile_torch_version": str(
                    observed_profile["torch_version"]
                ),
                "forward_profile_operators_json": json.dumps(
                    observed_profile["by_operator"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "m5_reference_trainable_parameters": int(
                    forward_compute_profile["m5_reference_trainable_parameters"]
                ),
                "m5_reference_learned_forward_supported_flops": int(
                    reference_profile["learned_forward_supported_flops"]
                ),
                "parameter_relative_deviation_from_m5": float(
                    forward_compute_profile["parameter_relative_deviation"]
                ),
                "learned_forward_supported_flops_relative_deviation_from_m5": float(
                    forward_compute_profile[
                        "learned_forward_supported_flops_relative_deviation"
                    ]
                ),
                "parameter_relative_tolerance": float(
                    forward_compute_profile["parameter_relative_tolerance"]
                ),
                "learned_forward_supported_flops_relative_tolerance": float(
                    forward_compute_profile[
                        "learned_forward_supported_flops_relative_tolerance"
                    ]
                ),
                "capacity_compute_match_status": (
                    "within_frozen_tolerances"
                    if bool(forward_compute_profile["within_tolerances"])
                    else "outside_frozen_tolerances"
                ),
            }
        )
    if goal_id in {"C04", "C05", "C06"}:
        common["run_id"] = str(grouped_evaluation.run_id)
        alignment_identity = getattr(model, "alignment_identity")()
        if not isinstance(training_objective_summary, dict):
            raise RuntimeError(
                f"{goal_id} requires a current-fit training objective summary"
            )
        if training_objective_summary.get("scope") != (
            "source_train_current_fit_not_checkpoint_persistent"
        ):
            raise RuntimeError(
                f"{goal_id} training objective summary scope mismatch"
            )
        if training_objective_summary.get("aggregation") != (
            "batch_scalar_mean_weighted_by_batch_size"
        ):
            raise RuntimeError(f"{goal_id} training objective aggregation mismatch")
        if training_objective_summary.get("alignment_coefficients") != alignment_identity:
            raise RuntimeError(
                f"{goal_id} objective summary/alignment coefficients mismatch"
            )
        means = training_objective_summary.get("means", {})
        if not isinstance(means, dict):
            raise RuntimeError(
                f"{goal_id} training objective means must be a mapping"
            )
        if condition_id in {"M5", "C2"}:
            required_means = {
                "classification",
                "physical",
                "semantic",
                "geometric",
                "weighted_physical",
                "weighted_semantic",
                "weighted_geometric",
                "total",
            }
            missing_means = sorted(required_means - set(means))
            if missing_means:
                raise RuntimeError(
                    f"{goal_id} alignment objective summary is missing {missing_means}"
                )
        else:
            missing_means = sorted({"classification", "total"} - set(means))
            if missing_means:
                raise RuntimeError(
                    f"{condition_id} classification objective summary is missing {missing_means}"
                )
        reconstruction_residual = training_objective_summary.get(
            "objective_reconstruction_residual"
        )
        if not isinstance(reconstruction_residual, (int, float)) or not math.isfinite(
            reconstruction_residual
        ):
            raise RuntimeError(
                f"{goal_id} objective reconstruction residual is not finite"
            )
        if abs(float(reconstruction_residual)) > 1.0e-6:
            raise RuntimeError(
                f"{goal_id} objective summary does not reconstruct total loss"
            )

        if condition_id == "C2":
            observation = training_objective_summary.get(
                "target_permutation_observation"
            )
            if not isinstance(observation, dict):
                raise RuntimeError("C2 objective summary lacks permutation observations")
            if (
                int(observation.get("observed_permutations", -1))
                != int(training_objective_summary.get("observed_batches", -2))
                or int(observation.get("unique_derived_seeds", -1))
                != int(training_objective_summary.get("observed_batches", -2))
                or int(observation.get("observed_fixed_points", -1)) != 0
            ):
                raise RuntimeError("C2 observed permutation schedule violates its contract")

        c3_selection = _json_ready(
            getattr(grouped_evaluation, "c3_selection", None)
        )
        duplicate_reader = getattr(model, "duplicate_control_identity", None)
        duplicate_identity = duplicate_reader() if callable(duplicate_reader) else None
        if condition_id == "C3":
            if c3_selection != _C3_SELECTION_IDENTITY:
                raise ValueError("C3 predeclared candidate-selection identity mismatch")
            if not isinstance(duplicate_identity, dict):
                raise RuntimeError("C3 executed duplicate-control identity is missing")
            if (
                int(forward_compute_profile["observed_trainable_parameters"])
                != _C3_SELECTION_IDENTITY["duplicate_2d_parameters"]
                or int(observed_profile["learned_forward_supported_flops"])
                != _C3_SELECTION_IDENTITY["duplicate_2d_supported_flops"]
                or int(forward_compute_profile["m5_reference_trainable_parameters"])
                != _C3_SELECTION_IDENTITY["m5_parameters"]
                or int(reference_profile["learned_forward_supported_flops"])
                != _C3_SELECTION_IDENTITY["m5_supported_flops"]
            ):
                raise RuntimeError("C3 executed profile differs from predeclared selection")
        else:
            if c3_selection is not None or duplicate_identity is not None:
                raise RuntimeError(f"{condition_id} cannot carry C3 control identity")

        if condition_id == "C2":
            reported_control_identity: dict[str, Any] = dict(control_identity)
        elif condition_id == "M5":
            reported_control_identity = {
                "mode": "matched_no_permutation",
                "classification_pairing": "synchronized_original_views",
                "alignment_target_pairing": "synchronized_original_views",
            }
        elif condition_id == "C3":
            reported_control_identity = {
                "mode": "not_applicable",
                "reason": "classification_only_duplicate_rendering_control",
            }
        else:
            reported_control_identity = {
                "mode": "not_applicable",
                "reason": "classification_only_non_alignment_condition",
            }
        reported_duplicate_identity = (
            duplicate_identity
            if duplicate_identity is not None
            else {
                "status": "not_applicable",
                "reason": "condition_is_not_duplicate_rendering_control",
            }
        )
        comparison_profile = forward_compute_profile.get("m4_c1_reference")
        if not isinstance(comparison_profile, dict):
            raise RuntimeError(
                f"{goal_id} profile is missing the M4/C1 comparison"
            )
        capacity_comparison = {
            "observed": {
                "trainable_parameters": int(
                    forward_compute_profile["observed_trainable_parameters"]
                ),
                "learned_forward_supported_flops": int(
                    observed_profile["learned_forward_supported_flops"]
                ),
            },
            "m5_reference": {
                "trainable_parameters": int(
                    forward_compute_profile["m5_reference_trainable_parameters"]
                ),
                "learned_forward_supported_flops": int(
                    reference_profile["learned_forward_supported_flops"]
                ),
                "parameter_relative_deviation": float(
                    forward_compute_profile["parameter_relative_deviation"]
                ),
                "learned_forward_supported_flops_relative_deviation": float(
                    forward_compute_profile[
                        "learned_forward_supported_flops_relative_deviation"
                    ]
                ),
            },
            "m4_c1_reference": {
                "trainable_parameters": int(
                    forward_compute_profile[
                        "m4_c1_reference_trainable_parameters"
                    ]
                ),
                "learned_forward_supported_flops": int(
                    comparison_profile["learned_forward_supported_flops"]
                ),
                "parameter_signed_deviation": float(
                    forward_compute_profile[
                        "parameter_signed_deviation_from_m4_c1"
                    ]
                ),
                "parameter_relative_deviation": float(
                    forward_compute_profile[
                        "parameter_relative_deviation_from_m4_c1"
                    ]
                ),
                "learned_forward_supported_flops_signed_deviation": float(
                    forward_compute_profile[
                        "learned_forward_supported_flops_signed_deviation_from_m4_c1"
                    ]
                ),
                "learned_forward_supported_flops_relative_deviation": float(
                    forward_compute_profile[
                        "learned_forward_supported_flops_relative_deviation_from_m4_c1"
                    ]
                ),
            },
            "m5_tolerance_status": (
                "within_frozen_tolerances"
                if bool(forward_compute_profile["within_tolerances"])
                else "outside_frozen_tolerances"
            ),
        }
        common.update(
            {
                "training_objective_summary_json": json.dumps(
                    _json_ready(training_objective_summary),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "alignment_target_control_identity_json": json.dumps(
                    _json_ready(reported_control_identity),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "duplicate_control_identity_json": json.dumps(
                    _json_ready(reported_duplicate_identity),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "capacity_compute_comparison_json": json.dumps(
                    capacity_comparison,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "loss_scale_retuned": False,
                "target_control_retuned": False,
                "c3_selection_json": json.dumps(
                    _json_ready(c3_selection or {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        if goal_id in {"C05", "C06"}:
            if not isinstance(view_gradient_summary, dict):
                raise RuntimeError(
                    f"{goal_id} requires a first-batch view-gradient summary"
                )
            if (
                view_gradient_summary.get("status") != "passed"
                or view_gradient_summary.get("condition_id") != condition_id
                or view_gradient_summary.get("scope")
                != "first_source_training_batch_after_backward_before_optimizer_step"
            ):
                raise RuntimeError(
                    f"{goal_id} view-gradient summary identity mismatch"
                )
            gradient_norms = view_gradient_summary.get("gradient_norms")
            if not isinstance(gradient_norms, dict) or not gradient_norms:
                raise RuntimeError(
                    f"{goal_id} view-gradient summary has no observed norms"
                )
            threshold = float(
                view_gradient_summary.get("required_gradient_norm_threshold", -1.0)
            )
            if threshold != 1.0e-12 or any(
                not math.isfinite(float(value)) or float(value) <= threshold
                for value in gradient_norms.values()
            ):
                raise RuntimeError(
                    f"{goal_id} required view-gradient group is inactive"
                )
            common.update(
                {
                    "run_role": str(grouped_evaluation.run_role),
                    "reproduction_of": str(
                        getattr(grouped_evaluation, "reproduction_of", "")
                    ),
                    "view_gradient_summary_json": json.dumps(
                        _json_ready(view_gradient_summary),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

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
    def __init__(self) -> None:
        self._forward_compute_profiles: dict[int, dict[str, Any]] = {}
        self._trained_tasks: dict[int, Any] = {}

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
        grouped_evaluation = getattr(
            context.args_task, "grouped_evaluation", None
        )
        goal_id = str(getattr(grouped_evaluation, "goal_id", "C02"))
        if goal_id in {"C04", "C05", "C06"}:
            self._trained_tasks[context.iteration] = context.task
        if goal_id in {"C03", "C04", "C05", "C06"}:
            condition_id = str(
                getattr(
                    grouped_evaluation,
                    "condition_id",
                    getattr(context.args_model, "condition", ""),
                )
            )
            self._forward_compute_profiles[context.iteration] = (
                build_p01_forward_compute_profile(
                    context.model,
                    context.args_model,
                    context.args_data,
                    grouped_evaluation,
                    condition_id=condition_id,
                )
            )

    def build_result_rows(
        self, context: ClassificationContext
    ) -> list[dict[str, Any]]:
        trained_task = self._trained_tasks.get(context.iteration)
        summary_reader = getattr(trained_task, "training_objective_summary", None)
        training_objective_summary = (
            summary_reader() if callable(summary_reader) else None
        )
        gradient_reader = getattr(trained_task, "view_gradient_summary", None)
        view_gradient_summary = (
            gradient_reader() if callable(gradient_reader) else None
        )
        try:
            return build_p01_grouped_result_rows(
                context,
                forward_compute_profile=self._forward_compute_profiles.get(
                    context.iteration
                ),
                training_objective_summary=training_objective_summary,
                view_gradient_summary=view_gradient_summary,
            )
        finally:
            self._trained_tasks.pop(context.iteration, None)
            self._forward_compute_profiles.pop(context.iteration, None)

    def build_summary_row(self, context: ClassificationContext) -> dict[str, Any]:
        grouped_evaluation = getattr(
            context.args_task, "grouped_evaluation", None
        )
        if grouped_evaluation is None or not bool(
            getattr(grouped_evaluation, "enabled", False)
        ):
            return super().build_summary_row(context)
        rows = context.result_rows
        if not isinstance(rows, list):
            raise RuntimeError("P01 grouped summary requires constructed result rows")
        summary_rows = [
            row
            for row in rows
            if row.get("target_environment") == "mean_across_target_load_domains"
        ]
        if len(summary_rows) != 1:
            raise RuntimeError(
                "P01 grouped summary requires exactly one cross-domain mean row"
            )
        return dict(summary_rows[0])


def pipeline(args: Any) -> list[dict[str, Any]]:
    """Run the standard classification train/test lifecycle."""
    return run_classification_pipeline(args, hooks=_P01DataProtocolHooks())
