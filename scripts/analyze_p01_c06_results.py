#!/usr/bin/env python3
"""Validate and summarize the frozen P01 C06 decisive matrix.

The parser treats each condition/seed process as one matrix cell, retains
missing or failed cells, and computes paired contrasts only from the two
integer target-domain rows. It intentionally performs no window-level
resampling or hypothesis test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence


CONDITIONS = ("M1", "M2", "M3", "M4", "M5", "C1", "C2", "C3")
SEEDS = (42, 123, 456)
DOMAINS = (2, 3)
EXPECTED_MODEL_CONDITIONS = {
    "M1": "M1",
    "M2": "M2",
    "M3": "M3",
    "M4": "M4",
    "M5": "M5",
    "C1": "M4",
    "C2": "M5",
    "C3": "C3",
}
EXPECTED_PARAMETERS = {
    "M1": 19_587,
    "M2": 27_907,
    "M3": 47_235,
    "M4": 49_411,
    "M5": 47_235,
    "C1": 49_411,
    "C2": 47_235,
    "C3": 55_555,
}
EXPECTED_FLOPS = {
    "M1": 22_823_296,
    "M2": 23_168_512,
    "M3": 45_991_424,
    "M4": 46_004_224,
    "M5": 45_991_424,
    "C1": 46_004_224,
    "C2": 45_991_424,
    "C3": 46_336_640,
}
EXPECTED_RUN_SCOPE = "C06_three_seed_two_environment_decisive_pilot"
CONTRAST_NAMES = (
    "alignment_gain",
    "multimodal_synergy",
    "m5_minus_c2",
    "m5_minus_c3",
    "c2_gain",
    "c3_gain",
)
NUMERIC_TIE_TOLERANCE = 1.0e-12


class C06ValidationError(RuntimeError):
    """Raised when a C06 artifact violates the frozen contract."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_int(row: Mapping[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise C06ValidationError(f"invalid integer field {key!r}: {row.get(key)!r}") from exc


def _as_float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise C06ValidationError(f"invalid float field {key!r}: {row.get(key)!r}") from exc
    if not math.isfinite(value):
        raise C06ValidationError(f"non-finite field {key!r}: {value!r}")
    return value


def validate_matrix_contract(matrix_path: Path, engine_root: Path) -> list[dict[str, str]]:
    rows = _read_csv(matrix_path)
    if len(rows) != len(CONDITIONS) * len(SEEDS):
        raise C06ValidationError(f"C06 matrix must have 24 rows, found {len(rows)}")

    expected_keys = {(condition, seed) for seed in SEEDS for condition in CONDITIONS}
    observed_keys: set[tuple[str, int]] = set()
    run_ids: set[str] = set()
    output_dirs: set[str] = set()
    for row in rows:
        condition = row.get("condition_id", "")
        seed = _as_int(row, "seed")
        key = (condition, seed)
        if key not in expected_keys or key in observed_keys:
            raise C06ValidationError(f"unexpected or duplicate matrix key {key!r}")
        observed_keys.add(key)
        if row.get("goal_id") != "C06":
            raise C06ValidationError(f"{key!r} has wrong goal_id")
        if row.get("model_condition") != EXPECTED_MODEL_CONDITIONS[condition]:
            raise C06ValidationError(f"{key!r} has wrong model condition")
        if row.get("target_domains") != "2|3":
            raise C06ValidationError(f"{key!r} has wrong target domains")
        expected_run_number = 29 + SEEDS.index(seed) * len(CONDITIONS) + CONDITIONS.index(condition)
        expected_run_id = f"RUN-{expected_run_number:04d}"
        if row.get("run_id") != expected_run_id:
            raise C06ValidationError(
                f"{key!r} expected {expected_run_id}, found {row.get('run_id')!r}"
            )
        run_ids.add(expected_run_id)
        output_dir = row.get("output_dir", "")
        if not output_dir or output_dir in output_dirs:
            raise C06ValidationError(f"{key!r} has empty or duplicate output_dir")
        output_dirs.add(output_dir)
        config_path = engine_root / row.get("config_path", "")
        if not config_path.is_file():
            raise C06ValidationError(f"{key!r} config does not exist: {config_path}")
        expected_target_seed = "31042" if condition == "C2" else "not_applicable"
        if row.get("c2_target_seed") != expected_target_seed:
            raise C06ValidationError(f"{key!r} has wrong C2 target seed declaration")
        if row.get("pre_run_status") != "planned":
            raise C06ValidationError(f"{key!r} pre-run status must be planned")

    if observed_keys != expected_keys or len(run_ids) != 24:
        raise C06ValidationError("C06 matrix is not a complete condition/seed product")
    return rows


def command_for(row: Mapping[str, str]) -> str:
    return (
        "CUDA_VISIBLE_DEVICES=3 conda run -n LQ_signal python main.py "
        f"--config {row['config_path']} "
        f"--override environment.seed={row['seed']} "
        f"--override environment.output_dir={row['output_dir']} "
        f"--override task.grouped_evaluation.run_id={row['run_id']}"
    )


def _single_path(paths: Iterable[Path], *, label: str) -> Path | None:
    candidates = sorted(paths)
    if len(candidates) > 1:
        raise C06ValidationError(
            f"ambiguous {label}: expected at most one path, found {len(candidates)}"
        )
    return candidates[0] if candidates else None


def _manifest_status(path: Path | None) -> tuple[str, str]:
    if path is None:
        return "missing", "manifest_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", f"manifest_unreadable:{exc}"
    status = str(payload.get("status", "invalid"))
    if status == "succeeded":
        return status, ""
    failure = payload.get("failure")
    return status, json.dumps(failure, sort_keys=True, separators=(",", ":"))


def _validate_success_rows(
    contract: Mapping[str, str],
    rows: list[dict[str, str]],
    *,
    engine_root: Path,
) -> None:
    condition = contract["condition_id"]
    seed = int(contract["seed"])
    if len(rows) != 3:
        raise C06ValidationError(
            f"{condition}/seed{seed} must emit three rows, found {len(rows)}"
        )
    expected_domains = {"2", "3", "mean_2_3"}
    if {row.get("target_domain", "") for row in rows} != expected_domains:
        raise C06ValidationError(f"{condition}/seed{seed} has wrong row domains")
    for row in rows:
        identity = (condition, seed, row.get("target_domain"))
        expected = {
            "run_id": contract["run_id"],
            "run_scope": EXPECTED_RUN_SCOPE,
            "condition_id": condition,
            "model_condition": EXPECTED_MODEL_CONDITIONS[condition],
            "run_stage": "three_seed_two_environment_decisive_pilot",
            "run_role": "matrix_cell",
            "reproduction_of": "",
            "status": "succeeded",
            "primary_metric_name": "condition_block_macro_f1",
            "raw_label_order": "1|2|3",
            "training_index_order": "0|1|2",
            "aggregation": "mean_softmax_windows_then_argmax_per_domain_condition_block",
            "checkpoint_selection": "lowest_validation_loss_within_fixed_budget",
            "optimizer": "adam",
            "scheduler": "none",
        }
        for key, value in expected.items():
            if row.get(key, "") != value:
                raise C06ValidationError(
                    f"{identity!r} expected {key}={value!r}, found {row.get(key)!r}"
                )
        if _as_int(row, "seed") != seed or _as_int(row, "iteration") != 0:
            raise C06ValidationError(f"{identity!r} has wrong seed/iteration")
        if _as_int(row, "training_epochs") != 10:
            raise C06ValidationError(f"{identity!r} has wrong epoch budget")
        if _as_int(row, "trainable_parameters") != EXPECTED_PARAMETERS[condition]:
            raise C06ValidationError(f"{identity!r} has wrong parameter count")
        if _as_int(row, "learned_forward_supported_flops") != EXPECTED_FLOPS[condition]:
            raise C06ValidationError(f"{identity!r} has wrong supported FLOPs")
        if _as_float(row, "learning_rate") != 0.001:
            raise C06ValidationError(f"{identity!r} has wrong learning rate")
        if _as_float(row, "weight_decay") != 0.0001:
            raise C06ValidationError(f"{identity!r} has wrong weight decay")
        if row.get("early_stopping") not in {"False", "false"}:
            raise C06ValidationError(f"{identity!r} enabled early stopping")
        if _as_int(row, "source_validation_tuning_trials") != 0:
            raise C06ValidationError(f"{identity!r} reports target or tuning trials")
        checkpoint = Path(row.get("checkpoint_path", ""))
        if not checkpoint.is_absolute():
            checkpoint = engine_root / checkpoint
        if not checkpoint.is_file():
            raise C06ValidationError(f"{identity!r} checkpoint is missing: {checkpoint}")
        metric = _as_float(row, "primary_metric_value")
        if not 0.0 <= metric <= 1.0:
            raise C06ValidationError(f"{identity!r} metric is outside [0,1]")
        if row["target_domain"] in {"2", "3"}:
            if (
                _as_int(row, "group_count") != 6
                or row.get("class_group_support") != "1:2|2:2|3:2"
                or _as_int(row, "window_count") != 384
                or _as_int(row, "evaluated_domain_count") != 1
            ):
                raise C06ValidationError(f"{identity!r} has wrong grouped support")
    if condition == "C2":
        control = json.loads(rows[0]["alignment_target_control_identity_json"])
        if control.get("seed") != 31042:
            raise C06ValidationError(f"C2/seed{seed} changed target-control seed")
        objective = json.loads(rows[0]["training_objective_summary_json"])
        observation = objective.get("target_permutation_observation", {})
        if (
            observation.get("observed_permutations") != 480
            or observation.get("unique_derived_seeds") != 480
            or observation.get("observed_fixed_points") != 0
            or observation.get("derived_seed_min") != 31042
            or observation.get("derived_seed_max") != 9031116
        ):
            raise C06ValidationError(f"C2/seed{seed} permutation schedule drifted")


def _assert_data_protocol_identity(protocol_paths: Sequence[Path]) -> None:
    if len(protocol_paths) != 24:
        raise C06ValidationError(
            f"expected 24 data-protocol summaries, found {len(protocol_paths)}"
        )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in protocol_paths]
    reference = payloads[0]
    if any(payload != reference for payload in payloads[1:]):
        raise C06ValidationError("C06 data-protocol summaries are not object-identical")


def _assert_c1_identity(domain_rows: Mapping[tuple[str, int, int], dict[str, str]]) -> None:
    for seed in SEEDS:
        for domain in DOMAINS:
            m4 = domain_rows[("M4", seed, domain)]
            c1 = domain_rows[("C1", seed, domain)]
            for field in (
                "primary_metric_value",
                "group_predictions_json",
                "trainable_parameters",
                "learned_forward_supported_flops",
            ):
                if m4[field] != c1[field]:
                    raise C06ValidationError(
                        f"C1 is not an exact M4 identity at seed={seed}, domain={domain}, "
                        f"field={field}"
                    )


def build_contrasts(
    domain_rows: Mapping[tuple[str, int, int], dict[str, str]]
) -> list[dict[str, Any]]:
    _assert_c1_identity(domain_rows)
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        for domain in DOMAINS:
            q = {
                condition: _as_float(domain_rows[(condition, seed, domain)], "primary_metric_value")
                for condition in CONDITIONS
            }
            fusion_baseline = max(q["M3"], q["M4"])
            unimodal_baseline = max(q["M1"], q["M2"])
            output.append(
                {
                    "seed": seed,
                    "target_domain": domain,
                    **{f"q_{condition.lower()}": q[condition] for condition in CONDITIONS},
                    "fusion_baseline_max_m3_m4": fusion_baseline,
                    "unimodal_baseline_max_m1_m2": unimodal_baseline,
                    "alignment_gain": q["M5"] - fusion_baseline,
                    "multimodal_synergy": q["M5"] - unimodal_baseline,
                    "m5_minus_c2": q["M5"] - q["C2"],
                    "m5_minus_c3": q["M5"] - q["C3"],
                    "c2_gain": q["C2"] - fusion_baseline,
                    "c3_gain": q["C3"] - fusion_baseline,
                    "c1_identity_checked": True,
                    "independence_boundary": (
                        "seed is an optimization repeat; domains and six documented "
                        "condition blocks are not independent populations"
                    ),
                }
            )
    return output


def _sign(value: float) -> str:
    if value > NUMERIC_TIE_TOLERANCE:
        return "positive"
    if value < -NUMERIC_TIE_TOLERANCE:
        return "negative"
    return "tie"


def stable_positive(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    by_domain = {
        domain: [float(row[field]) for row in rows if int(row["target_domain"]) == domain]
        for domain in DOMAINS
    }
    by_seed = {
        seed: [float(row[field]) for row in rows if int(row["seed"]) == seed]
        for seed in SEEDS
    }
    if any(len(values) != 3 for values in by_domain.values()):
        return False
    if any(len(values) != 2 for values in by_seed.values()):
        return False
    domain_gate = all(mean(values) > NUMERIC_TIE_TOLERANCE for values in by_domain.values())
    seed_gate = sum(
        mean(values) > NUMERIC_TIE_TOLERANCE for values in by_seed.values()
    ) >= 2
    return domain_gate and seed_gate


def summarize_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    stable = {name: stable_positive(rows, name) for name in CONTRAST_NAMES}
    for name in CONTRAST_NAMES:
        for domain in DOMAINS:
            values = [
                float(row[name])
                for row in rows
                if int(row["target_domain"]) == domain
            ]
            signs = [_sign(value) for value in values]
            summaries.append(
                {
                    "contrast": name,
                    "target_domain": domain,
                    "seed_count": len(values),
                    "mean": mean(values),
                    "sample_sd": stdev(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "positive_count": signs.count("positive"),
                    "tie_count": signs.count("tie"),
                    "negative_count": signs.count("negative"),
                    "stable_positive_across_frozen_rule": stable[name],
                }
            )
    return summaries


def route_decision(contrast_rows: Sequence[Mapping[str, Any]]) -> str:
    stable = {name: stable_positive(contrast_rows, name) for name in CONTRAST_NAMES}
    if not stable["alignment_gain"] or not stable["multimodal_synergy"]:
        return "boundary_or_bounded_stop_no_stable_alignment_gain_or_synergy"
    if (
        not stable["m5_minus_c2"]
        or not stable["m5_minus_c3"]
        or stable["c2_gain"]
        or stable["c3_gain"]
    ):
        return "boundary_or_stop_negative_control_reproduces_effect"
    return "admit_C07_C09_performance_gate_only_no_mechanism_claim"


def analyze(
    contract_rows: Sequence[Mapping[str, str]],
    *,
    engine_root: Path,
    output_dir: Path,
) -> tuple[bool, str]:
    run_status_rows: list[dict[str, Any]] = []
    successful_rows: list[dict[str, str]] = []
    domain_matrix_rows: list[dict[str, Any]] = []
    protocol_paths: list[Path] = []
    invalid_messages: list[str] = []

    for contract in contract_rows:
        condition = contract["condition_id"]
        seed = int(contract["seed"])
        root = engine_root / contract["output_dir"]
        try:
            manifest_path = _single_path(
                root.glob(".phmfactory/runs/*/run_manifest.json"),
                label=f"{condition}/seed{seed} manifest",
            )
            result_path = _single_path(
                root.glob("*/M_P01Alignment/T_DGclassification_*/all_results.csv"),
                label=f"{condition}/seed{seed} aggregate result",
            )
            protocol_path = _single_path(
                root.glob(
                    "*/M_P01Alignment/T_DGclassification_*/iter_0/data_protocol_summary.json"
                ),
                label=f"{condition}/seed{seed} data protocol",
            )
            status, failure = _manifest_status(manifest_path)
            if status == "succeeded" and result_path is None:
                status, failure = "invalid", "succeeded_manifest_but_result_missing"
            if status == "succeeded" and protocol_path is None:
                status, failure = "invalid", "succeeded_manifest_but_protocol_missing"
            rows: list[dict[str, str]] = []
            if status == "succeeded" and result_path is not None:
                rows = _read_csv(result_path)
                _validate_success_rows(contract, rows, engine_root=engine_root)
                successful_rows.extend(rows)
                assert protocol_path is not None
                protocol_paths.append(protocol_path)
            for domain in DOMAINS:
                match = next(
                    (row for row in rows if row.get("target_domain") == str(domain)),
                    None,
                )
                domain_matrix_rows.append(
                    {
                        **dict(contract),
                        "target_domain": domain,
                        "observed_status": status,
                        "primary_metric_value": (
                            match.get("primary_metric_value", "") if match else ""
                        ),
                        "trainable_parameters": (
                            match.get("trainable_parameters", "") if match else ""
                        ),
                        "learned_forward_supported_flops": (
                            match.get("learned_forward_supported_flops", "") if match else ""
                        ),
                        "checkpoint_path": match.get("checkpoint_path", "") if match else "",
                        "failure_or_invalid_reason": failure,
                    }
                )
        except (C06ValidationError, KeyError, OSError, json.JSONDecodeError) as exc:
            status = "invalid"
            failure = str(exc)
            manifest_path = None
            result_path = None
            invalid_messages.append(f"{condition}/seed{seed}: {exc}")
            for domain in DOMAINS:
                domain_matrix_rows.append(
                    {
                        **dict(contract),
                        "target_domain": domain,
                        "observed_status": status,
                        "primary_metric_value": "",
                        "trainable_parameters": "",
                        "learned_forward_supported_flops": "",
                        "checkpoint_path": "",
                        "failure_or_invalid_reason": failure,
                    }
                )
        run_status_rows.append(
            {
                **dict(contract),
                "observed_status": status,
                "manifest_path": str(manifest_path or ""),
                "result_path": str(result_path or ""),
                "failure_or_invalid_reason": failure,
            }
        )

    _write_csv(output_dir / "c06_run_status.csv", run_status_rows)
    _write_csv(output_dir / "c06_condition_domain_matrix.csv", domain_matrix_rows)
    complete = all(row["observed_status"] == "succeeded" for row in run_status_rows)
    decision = "partial_incomplete_or_invalid_matrix"
    if complete:
        try:
            _assert_data_protocol_identity(protocol_paths)
            domain_rows = {
                (row["condition_id"], int(row["seed"]), int(row["target_domain"])): row
                for row in successful_rows
                if row["target_domain"] in {"2", "3"}
            }
            if len(domain_rows) != 48:
                raise C06ValidationError(
                    f"expected 48 unique condition/seed/domain rows, found {len(domain_rows)}"
                )
            contrasts = build_contrasts(domain_rows)
            summaries = summarize_contrasts(contrasts)
            decision = route_decision(contrasts)
            _write_csv(output_dir / "c06_paired_contrasts.csv", contrasts)
            _write_csv(output_dir / "c06_contrast_summary.csv", summaries)
        except (C06ValidationError, KeyError, ValueError) as exc:
            complete = False
            decision = "partial_incomplete_or_invalid_matrix"
            invalid_messages.append(str(exc))

    summary_lines = [
        "# P01 C06 parser summary",
        "",
        f"- complete_valid_matrix: `{str(complete).lower()}`",
        f"- planned_processes: `{len(contract_rows)}`",
        f"- succeeded_processes: `{sum(row['observed_status'] == 'succeeded' for row in run_status_rows)}`",
        f"- integer-domain_rows_expected: `48`",
        f"- route_decision: `{decision}`",
        "- inference_boundary: three optimization seeds; no window, file, batch, domain, or condition block is promoted to an independent population replicate.",
        "- claim_boundary: C06 can select a performance route but cannot establish mechanism, causality, nuisance suppression, physical-frequency semantics, or independent modalities.",
    ]
    if invalid_messages:
        summary_lines.extend(["", "## Invalid or incomplete observations", ""])
        summary_lines.extend(f"- {message}" for message in invalid_messages)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "c06_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    return complete, decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/experiments/p01/p01_c06_run_matrix.csv"),
    )
    parser.add_argument("--engine-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/p01/c06_analysis"),
    )
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    engine_root = args.engine_root.resolve()
    matrix_path = args.matrix
    if not matrix_path.is_absolute():
        matrix_path = engine_root / matrix_path
    contract_rows = validate_matrix_contract(matrix_path, engine_root)
    if args.print_commands:
        for row in contract_rows:
            print(command_for(row))
    if args.validate_contract:
        print(
            f"C06 contract valid: {len(contract_rows)} processes, "
            f"seeds={list(SEEDS)}, domains={list(DOMAINS)}"
        )
        return 0
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = engine_root / output_dir
    complete, decision = analyze(
        contract_rows,
        engine_root=engine_root,
        output_dir=output_dir,
    )
    print(f"C06 complete={complete} decision={decision} output_dir={output_dir}")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
