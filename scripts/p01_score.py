"""Create a frozen offline P01 statistical summary from prediction artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import src.utils.p01_statistics as statistics_module
from src.utils.p01_statistics import (
    BOOTSTRAP_SEED,
    SCORING_DERANGEMENT_SEED,
    accuracy_metric_values,
    alignment_metric_values,
    collapse_diagnostic,
    exact_two_sided_sign_flip,
    freeze_scoring_derangement,
    load_prediction_artifact,
    load_scoring_universe,
    paired_hierarchical_bootstrap,
    seed_metric_estimates,
    single_arm_hierarchical_bootstrap,
    validate_artifact_grid,
    write_json_summary,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_mapping_sha256(mapping: dict[str, str]) -> str:
    rendered = json.dumps(
        dict(sorted(mapping.items())),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _load_group_strata(
    path: str | None, dataset_key: str
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    if path is None:
        if dataset_key == "XJTU":
            raise ValueError(
                "XJTU scoring requires --group-strata-json with frozen Domain_id bindings"
            )
        return None, {"source": "CWRU_y_true", "path": None, "file_sha256": None}
    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Group-strata JSON must be an object")
    if dataset_key in payload and isinstance(payload[dataset_key], dict):
        payload = payload[dataset_key]
    if not all(isinstance(key, str) for key in payload):
        raise ValueError("Group-strata JSON keys must be strings")
    if not all(isinstance(value, (str, int)) for value in payload.values()):
        raise ValueError("Group-strata JSON values must be strings or integers")
    mapping = {str(key): str(value) for key, value in payload.items()}
    return mapping, {
        "source": "external_frozen_group_strata_json",
        "path": str(source_path),
        "file_sha256": _sha256_file(source_path),
        "mapping_sha256": _canonical_mapping_sha256(mapping),
    }


def _analysis_code_state() -> dict[str, Any]:
    code_paths = [Path(statistics_module.__file__).resolve(), Path(__file__).resolve()]
    code_hashes = {str(path): _sha256_file(path) for path in code_paths}
    repository_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *map(str, code_paths)],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    state_payload = {
        "git_commit": commit,
        "target_files_dirty": bool(status.strip()),
        "code_file_sha256s": code_hashes,
    }
    state_sha256 = hashlib.sha256(
        json.dumps(
            state_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        **state_payload,
        "identifier": f"git:{commit};analysis_files:{state_sha256}",
        "code_state_sha256": state_sha256,
    }


def _lower_endpoint_audit(metric_summary: dict[str, Any]) -> dict[str, Any]:
    lower = float(metric_summary["interval_lower"])
    mcse = float(metric_summary["interval_lower_mcse"])
    tolerance = 2.0 * mcse
    if abs(lower) <= tolerance:
        status = "inconclusive_monte_carlo_boundary"
    elif lower > 0.0:
        status = "lower_bound_above_zero"
    else:
        status = "lower_bound_not_above_zero"
    return {
        "decision_boundary": 0.0,
        "lower_endpoint": lower,
        "lower_endpoint_mcse": mcse,
        "near_boundary_tolerance": tolerance,
        "near_boundary_rule": "absolute_distance_le_2x_endpoint_mcse",
        "status": status,
    }


def _seed_summary(
    grid, metric_values, arms: Sequence[str]  # type: ignore[no-untyped-def]
) -> dict[str, Any]:
    return {
        arm: {
            str(seed): value
            for seed, value in seed_metric_estimates(grid, metric_values, arm).items()
        }
        for arm in arms
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed P01 offline group-aware statistical scorer"
    )
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--dataset-key", choices=("CWRU", "XJTU"), required=True)
    parser.add_argument("--dataset-slug", choices=("cwru", "xjtu"), required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--analysis-scope",
        choices=("final_oof", "g050_fold0"),
        required=True,
    )
    parser.add_argument(
        "--contrast",
        nargs=2,
        metavar=("ARM_A", "ARM_B"),
        required=True,
    )
    parser.add_argument("--scoring-manifest", required=True)
    parser.add_argument(
        "--sample-universe-json",
        required=True,
        help="Independent all-fold evaluation universe bound to frozen split manifests",
    )
    parser.add_argument(
        "--group-strata-json",
        help="Required for XJTU; frozen group-to-Domain_id mapping JSON",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--scoring-seed", type=int, default=SCORING_DERANGEMENT_SEED
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        help="Must equal the confidence level frozen for the selected scope/contrast",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contrast_a, contrast_b = map(str, args.contrast)
    if args.protocol_id != "P01-G040-v1":
        raise ValueError("P01 scorer is frozen to protocol_id P01-G040-v1")
    if contrast_a not in args.arms or contrast_b not in args.arms:
        raise ValueError("Both contrast arms must be present in --arms")
    canonical_contrasts = {
        ("FULL", "B4-GATTN"): 0.95,
        ("FULL", "TRAIN-MISPAIR"): 0.95 if args.analysis_scope == "g050_fold0" else 0.975,
    }
    if (contrast_a, contrast_b) not in canonical_contrasts:
        raise ValueError(
            "P01 scorer accepts only directed FULL-minus-B4-GATTN or "
            "FULL-minus-TRAIN-MISPAIR contrasts"
        )
    confidence_level = canonical_contrasts[(contrast_a, contrast_b)]
    if args.confidence_level is not None and args.confidence_level != confidence_level:
        raise ValueError(
            f"Frozen confidence level for this scope/contrast is {confidence_level}"
        )
    if args.bootstrap_replicates != 10000:
        raise ValueError("P01 scorer requires exactly 10000 bootstrap replicates")
    if args.bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError(f"P01 bootstrap seed must be {BOOTSTRAP_SEED}")
    if args.scoring_seed != SCORING_DERANGEMENT_SEED:
        raise ValueError(
            f"P01 scoring-derangement seed must be {SCORING_DERANGEMENT_SEED}"
        )

    group_strata, group_strata_binding = _load_group_strata(
        args.group_strata_json, args.dataset_key
    )

    artifacts = [load_prediction_artifact(path) for path in args.predictions]
    grid = validate_artifact_grid(
        artifacts,
        protocol_id=args.protocol_id,
        dataset_key=args.dataset_key,
        dataset_slug=args.dataset_slug,
        expected_arms=args.arms,
        expected_seeds=args.seeds,
        expected_folds=args.folds,
        analysis_scope=args.analysis_scope,
        group_strata_by_group=group_strata,
    )
    group_strata_binding["mapping_sha256"] = _canonical_mapping_sha256(
        dict(grid.group_strata)
    )
    universe = load_scoring_universe(args.sample_universe_json)
    derangement = freeze_scoring_derangement(
        universe, args.scoring_manifest, seed=args.scoring_seed
    )
    accuracy = accuracy_metric_values(grid)
    alignment_arms = tuple(
        dict.fromkeys(
            (contrast_a, contrast_b)
            + (("FULL",) if args.analysis_scope == "final_oof" else ())
        )
    )
    alignment = alignment_metric_values(grid, derangement, arms=alignment_arms)
    bootstrap = paired_hierarchical_bootstrap(
        grid,
        {"group_class_balanced_accuracy": accuracy, "alignment_margin": alignment},
        contrast_a,
        contrast_b,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        confidence_level=confidence_level,
    )
    absolute_full_alignment = None
    if args.analysis_scope == "final_oof":
        absolute_full_alignment = single_arm_hierarchical_bootstrap(
            grid,
            {"absolute_full_alignment_margin": alignment},
            "FULL",
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
            confidence_level=0.975,
        )

    accuracy_by_seed = _seed_summary(grid, accuracy, args.arms)
    alignment_by_seed = _seed_summary(
        grid, alignment, alignment_arms
    )
    accuracy_seed_effects = [
        accuracy_by_seed[contrast_a][str(seed)]
        - accuracy_by_seed[contrast_b][str(seed)]
        for seed in grid.seeds
    ]
    alignment_seed_effects = [
        alignment_by_seed[contrast_a][str(seed)]
        - alignment_by_seed[contrast_b][str(seed)]
        for seed in grid.seeds
    ]

    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": grid.protocol_id,
        "dataset_key": grid.dataset_key,
        "dataset_slug": grid.dataset_slug,
        "dataset_id": grid.dataset_id,
        "analysis_scope": args.analysis_scope,
        "arms": list(grid.arms),
        "training_seeds": list(grid.seeds),
        "outer_folds": list(grid.folds),
        "contrast": {"arm_a": contrast_a, "arm_b": contrast_b},
        "gatekeeping_context": (
            {
                "evidence_role": "expansion_futility_screen_only",
                "supports_C1_C2_C3": False,
            }
            if args.analysis_scope == "g050_fold0"
            else {
                "stage_1": "C2",
                "stage_2": ["C1", "C3"],
                "stage_2_confirmatory_only_if_C2_passes": True,
            }
        ),
        "artifact_sha256s": {
            str(artifact.path): artifact.artifact_sha256 for artifact in artifacts
        },
        "artifact_attempt_ids": {
            str(artifact.path): artifact.attempt_id for artifact in artifacts
        },
        "ordered_split_manifest_sha256s": [
            grid.split_manifest_sha256s[fold] for fold in grid.folds
        ],
        "scoring_derangement": {
            "path": str(derangement.path),
            "file_sha256": derangement.file_sha256,
            "sample_universe_source": str(universe.path),
            "sample_universe_file_sha256": derangement.sample_universe_file_sha256,
            "sample_universe_sha256": derangement.sample_universe_sha256,
            "mapping_sha256": derangement.mapping_sha256,
            "seed": args.scoring_seed,
            "ordered_split_manifests": [
                dict(entry) for entry in universe.split_manifests
            ],
        },
        "point_estimates_by_seed": {
            "group_class_balanced_accuracy": accuracy_by_seed,
            "alignment_margin": alignment_by_seed,
        },
        "design_strata_binding": group_strata_binding,
        "analysis_code_state": _analysis_code_state(),
        "paired_hierarchical_bootstrap": bootstrap.summary(),
        "exact_seed_sign_flip_sensitivity": {
            "group_class_balanced_accuracy": exact_two_sided_sign_flip(
                accuracy_seed_effects
            ),
            "alignment_margin": exact_two_sided_sign_flip(
                alignment_seed_effects
            ),
        },
    }
    summary["paired_hierarchical_bootstrap"]["lower_endpoint_audits"] = {
        metric_name: _lower_endpoint_audit(metric_summary)
        for metric_name, metric_summary in summary[
            "paired_hierarchical_bootstrap"
        ]["metrics"].items()
    }
    if absolute_full_alignment is not None:
        absolute_summary = absolute_full_alignment.summary()
        absolute_summary["lower_endpoint_audits"] = {
            metric_name: _lower_endpoint_audit(metric_summary)
            for metric_name, metric_summary in absolute_summary["metrics"].items()
        }
        summary["C1_absolute_full_alignment_bootstrap_97_5pct"] = absolute_summary
    endpoint_statuses = [
        audit["status"]
        for audit in summary["paired_hierarchical_bootstrap"][
            "lower_endpoint_audits"
        ].values()
    ]
    if absolute_full_alignment is not None:
        endpoint_statuses.extend(
            audit["status"]
            for audit in summary[
                "C1_absolute_full_alignment_bootstrap_97_5pct"
            ]["lower_endpoint_audits"].values()
        )
    summary["monte_carlo_boundary_gate"] = {
        "status": (
            "inconclusive"
            if "inconclusive_monte_carlo_boundary" in endpoint_statuses
            else "endpoint_mcse_clear_of_zero_boundary"
        ),
        "claim_promotion_forbidden_when_inconclusive": True,
    }
    if "FULL" in grid.arms:
        collapse = collapse_diagnostic(grid)
        collapse["evidence_role"] = (
            "C1_no_collapse_component"
            if args.analysis_scope == "final_oof"
            else "fold0_local_diagnostic_not_C1_support"
        )
        summary["shared_collapse"] = collapse

    output_sha256 = write_json_summary(args.output, summary)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": output_sha256,
                "sampled_index_sha256": bootstrap.sampled_index_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
