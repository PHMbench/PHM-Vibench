from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import scripts.p01_score as score_module
from scripts.p01_score import main as score_main
from src.utils.p01_statistics import (
    TRAINING_SEEDS,
    accuracy_metric_values,
    alignment_metric_values,
    collapse_diagnostic,
    freeze_scoring_derangement,
    group_class_balanced_accuracy,
    load_prediction_artifact,
    load_scoring_universe,
    paired_hierarchical_bootstrap,
    seed_metric_estimates,
    single_arm_hierarchical_bootstrap,
    validate_artifact_grid,
)


PROTOCOL_ID = "P01-G040-v1"
FULL_FOLDS = (0, 1, 2, 3)
CORE_ARMS = ("FULL", "B4-GATTN", "TRAIN-MISPAIR")


def _split_hash(fold: int) -> str:
    payload = _split_payload_without_hash(fold)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _split_payload_without_hash(fold: int) -> dict:
    rows = _fold_rows(fold)
    return {
        "schema_version": 1,
        "split_ids": {"test": sorted(set(rows["file_id"].tolist()))},
        "split_groups": {"test": sorted(set(rows["group_id"].tolist()))},
        "cross_validation": {"outer_fold": fold},
    }


def _fold_rows(fold: int) -> dict[str, np.ndarray]:
    file_ids: list[str] = []
    window_ids: list[int] = []
    groups: list[str] = []
    labels: list[int] = []
    for label in (0, 1):
        group = f"fold{fold}_class{label}"
        for window in (0, 1):
            file_ids.append(group)
            window_ids.append(window)
            groups.append(group)
            labels.append(label)
    sample_keys = [
        f"{file_id}:{window_id}"
        for file_id, window_id in zip(file_ids, window_ids)
    ]
    return {
        "file_id": np.asarray(file_ids, dtype=str),
        "window_id": np.asarray(window_ids, dtype=np.int64),
        "group_id": np.asarray(groups, dtype=str),
        "y_true": np.asarray(labels, dtype=np.int64),
        "sample_key": np.asarray(sample_keys, dtype=str),
    }


def _representations(fold: int) -> dict[str, np.ndarray]:
    rows = _fold_rows(fold)
    window = rows["window_id"].astype(float)
    label = rows["y_true"].astype(float)
    base = np.stack(
        [1.0 + 10.0 * fold + 0.2 * window, 0.5 + 0.2 * window + 0.1 * label],
        axis=1,
    )
    paired = base + np.asarray([0.01, -0.01])
    return {
        "shared_1d": base,
        "shared_2d": paired,
        "private_1d": base + 0.3,
        "private_2d": paired + 0.4,
        "reconstructed_1d": base + 0.05,
        "reconstructed_2d": paired + 0.05,
        "encoded_1d": base + 0.1,
        "encoded_2d": paired + 0.1,
    }


def _write_prediction(
    root: Path,
    *,
    arm: str,
    seed: int,
    fold: int,
    missing_representation: str | None = None,
) -> Path:
    rows = _fold_rows(fold)
    labels = rows["y_true"]
    if arm == "FULL":
        predictions = labels.copy()
    else:
        predictions = labels.copy()
        predictions[rows["window_id"] == 1] = 1 - predictions[rows["window_id"] == 1]
    logits = np.full((len(labels), 2), -2.0, dtype=np.float64)
    logits[np.arange(len(labels)), predictions] = 2.0
    arrays: dict[str, np.ndarray] = {
        "logits": logits,
        "y_true": labels,
        "y_pred": predictions,
        "file_id": rows["file_id"],
        "window_id": rows["window_id"],
        "group_id": rows["group_id"],
        "sample_key": rows["sample_key"],
        "outer_fold": np.full(len(labels), fold, dtype=np.int64),
        "training_seed": np.full(len(labels), seed, dtype=np.int64),
    }
    representations = _representations(fold)
    names = (
        ("shared_1d", "shared_2d", "private_1d", "private_2d", "reconstructed_1d", "reconstructed_2d")
        if arm in {"FULL", "TRAIN-MISPAIR"}
        else ("encoded_1d", "encoded_2d")
    )
    for name in names:
        if name != missing_representation:
            arrays[f"repr__{name}"] = representations[name]

    target = root / arm / str(seed) / f"fold_{fold}" / "predictions.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    artifact_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()

    config_path = target.parent / "config_snapshot.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    code_state_sha = hashlib.sha256(b"synthetic-code-state").hexdigest()
    paper_identity = {
        "protocol_id": PROTOCOL_ID,
        "dataset_key": "CWRU",
        "dataset_slug": "cwru",
        "dataset_id": 1,
        "arm_id": arm,
        "attempt_id": 0,
    }
    invocation_path = target.parent / "invocation.json"
    invocation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_snapshot_sha256": config_sha,
                "code_state_sha256": code_state_sha,
                "effective_seed": seed,
                "paper": paper_identity,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    invocation_sha = hashlib.sha256(invocation_path.read_bytes()).hexdigest()
    checkpoint_path = target.parent / "best.ckpt"
    checkpoint_path.write_bytes(b"synthetic-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint_manifest_path = target.parent / "best_checkpoint.manifest.json"
    checkpoint_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_sha,
                "monitor": "val_loss",
                "mode": "min",
                "score": 0.25,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checkpoint_manifest_sha = hashlib.sha256(
        checkpoint_manifest_path.read_bytes()
    ).hexdigest()
    split_path = target.parent / "split.json"
    split_payload = _split_payload_without_hash(fold)
    split_payload["manifest_payload_sha256"] = _split_hash(fold)
    split_path.write_text(json.dumps(split_payload, sort_keys=True), encoding="utf-8")
    data_payload_sha = hashlib.sha256(
        f"synthetic-data-{fold}".encode()
    ).hexdigest()
    data_snapshot_path = target.parent / "data_snapshot.manifest.json"
    data_snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_payload_sha256": data_payload_sha,
                "config_snapshot_sha256": config_sha,
                "invocation_sha256": invocation_sha,
                "split_manifest_payload_sha256": _split_hash(fold),
                "paper": paper_identity,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    data_snapshot_sha = hashlib.sha256(data_snapshot_path.read_bytes()).hexdigest()
    metrics_path = target.parent / "logs" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("epoch,step,val_loss\n0,0,0.25\n", encoding="utf-8")
    metrics_sha = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    metrics_manifest_path = target.parent / "trainer_metrics.manifest.json"
    metrics_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "logger_type": "CSVLogger",
                "logger_name": "synthetic",
                "logger_version": 0,
                "metrics_path": str(metrics_path.resolve()),
                "metrics_sha256": metrics_sha,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metrics_manifest_sha = hashlib.sha256(
        metrics_manifest_path.read_bytes()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact": target.name,
        "artifact_sha256": artifact_sha256,
        "samples": len(labels),
        "outer_fold": fold,
        "training_seed": seed,
        "arrays": sorted(arrays),
        "provenance": {
            "protocol_id": PROTOCOL_ID,
            "dataset_key": "CWRU",
            "dataset_slug": "cwru",
            "dataset_id": 1,
            "arm_id": arm,
            "attempt_id": 0,
            "outer_fold": fold,
            "training_seed": seed,
            "config_snapshot_path": str(config_path.resolve()),
            "config_snapshot_sha256": config_sha,
            "invocation_path": str(invocation_path.resolve()),
            "invocation_sha256": invocation_sha,
            "best_checkpoint_manifest_path": str(
                checkpoint_manifest_path.resolve()
            ),
            "best_checkpoint_manifest_sha256": checkpoint_manifest_sha,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_monitor": "val_loss",
            "checkpoint_mode": "min",
            "checkpoint_score": 0.25,
            "split_manifest_path": str(split_path.resolve()),
            "split_manifest_payload_sha256": _split_hash(fold),
            "code_state_identifier": "git:synthetic;files:synthetic",
            "code_state_sha256": code_state_sha,
            "data_snapshot_manifest_path": str(data_snapshot_path.resolve()),
            "data_snapshot_manifest_sha256": data_snapshot_sha,
            "data_payload_sha256": data_payload_sha,
            "trainer_metrics_manifest_path": str(
                metrics_manifest_path.resolve()
            ),
            "trainer_metrics_manifest_sha256": metrics_manifest_sha,
            "trainer_metrics_path": str(metrics_path.resolve()),
            "trainer_metrics_sha256": metrics_sha,
        },
    }
    target.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def _write_grid(
    root: Path,
    *,
    arms: tuple[str, ...],
    folds: tuple[int, ...],
) -> list[Path]:
    return [
        _write_prediction(root, arm=arm, seed=seed, fold=fold)
        for arm in arms
        for seed in TRAINING_SEEDS
        for fold in folds
    ]


def _write_universe(path: Path, *, folds: tuple[int, ...] = FULL_FOLDS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    split_entries = []
    for fold in folds:
        rows = _fold_rows(fold)
        samples.extend(
            {
                "sample_key": str(sample_key),
                "group_id": str(group),
                "y_true": int(label),
                "outer_fold": fold,
            }
            for sample_key, group, label in zip(
                rows["sample_key"], rows["group_id"], rows["y_true"]
            )
        )
        split_payload = _split_payload_without_hash(fold)
        split_payload["manifest_payload_sha256"] = _split_hash(fold)
        split_path = path.parent / "splits" / f"fold_{fold}.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(
            json.dumps(split_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        split_entries.append(
            {
                "outer_fold": fold,
                "path": str(split_path),
                "manifest_payload_sha256": _split_hash(fold),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "dataset_key": "CWRU",
        "dataset_slug": "cwru",
        "dataset_id": 1,
        "split_manifests": split_entries,
        "samples": samples,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_grid(paths: list[Path], *, arms: tuple[str, ...], folds: tuple[int, ...], scope: str):
    return validate_artifact_grid(
        [load_prediction_artifact(path) for path in paths],
        protocol_id=PROTOCOL_ID,
        dataset_key="CWRU",
        dataset_slug="cwru",
        expected_arms=arms,
        expected_seeds=TRAINING_SEEDS,
        expected_folds=folds,
        analysis_scope=scope,
    )


def _load_test_universe(path: Path):  # type: ignore[no-untyped-def]
    return load_scoring_universe(
        path,
        expected_split_sha256s=tuple(_split_hash(fold) for fold in FULL_FOLDS),
    )


def test_group_class_balanced_accuracy_uses_equal_group_then_class_weights() -> None:
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.asarray([0, 0, 0, 0, 1, 1, 0, 1, 0])
    groups = np.asarray(["a", "a", "a", "a", "b", "c", "c", "d", "d"])
    # Class 0: mean(group a=1, group b=0)=0.5. Class 1: both groups=0.5.
    assert group_class_balanced_accuracy(y_true, y_pred, groups) == pytest.approx(0.5)


def test_full_oof_scoring_is_deterministic_and_checkpoint_local(tmp_path: Path) -> None:
    paths = _write_grid(tmp_path / "artifacts", arms=("FULL", "B4-GATTN"), folds=FULL_FOLDS)
    grid = _load_grid(paths, arms=("FULL", "B4-GATTN"), folds=FULL_FOLDS, scope="final_oof")
    universe = _load_test_universe(_write_universe(tmp_path / "universe.json"))
    mapping_path = tmp_path / "scoring_pairing" / "cwru.json"
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    first = freeze_scoring_derangement(universe, mapping_path)
    second = freeze_scoring_derangement(universe, mapping_path)
    assert first.mapping_sha256 == second.mapping_sha256
    assert first.file_sha256 == second.file_sha256
    assert all(source != partner for source, partner in first.mapping.items())
    assert len(set(first.mapping.values())) == len(first.mapping)
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    accuracy = accuracy_metric_values(grid)
    alignment = alignment_metric_values(grid, first)
    assert set(seed_metric_estimates(grid, accuracy, "FULL").values()) == {1.0}
    assert set(seed_metric_estimates(grid, accuracy, "B4-GATTN").values()) == {0.5}
    assert all(
        value > 0
        for value in seed_metric_estimates(grid, alignment, "FULL").values()
    )
    absolute = single_arm_hierarchical_bootstrap(
        grid,
        {"absolute_full_alignment_margin": alignment},
        "FULL",
        replicates=128,
        confidence_level=0.975,
    )
    absolute_metric = absolute.metrics["absolute_full_alignment_margin"]
    assert absolute_metric.interval_lower > 0
    assert absolute_metric.interval_lower_mcse >= 0

    collapse = collapse_diagnostic(grid)
    assert collapse["passes_no_collapse"] is True
    # Fold offsets are deliberately large; a cross-checkpoint concatenation would inflate this.
    assert collapse["views"]["shared_1d"]["dataset_median"] < 1.0

    drift_payload = json.loads(universe.path.read_text(encoding="utf-8"))
    drift_file_id = drift_payload["samples"][0]["sample_key"].rsplit(":", 1)[0]
    drift_payload["samples"][0]["sample_key"] = f"{drift_file_id}:9"
    drift_path = tmp_path / "universe_drift.json"
    drift_path.write_text(json.dumps(drift_payload), encoding="utf-8")
    drift_universe = _load_test_universe(drift_path)
    with pytest.raises(ValueError, match="does not match"):
        freeze_scoring_derangement(drift_universe, mapping_path)


def test_g050_fold0_cannot_create_mapping_without_full_universe(tmp_path: Path) -> None:
    paths = _write_grid(tmp_path / "g050", arms=CORE_ARMS, folds=(0,))
    grid = _load_grid(paths, arms=CORE_ARMS, folds=(0,), scope="g050_fold0")
    partial_path = _write_universe(tmp_path / "fold0_only.json", folds=(0,))
    mapping_path = tmp_path / "analysis" / "cwru.json"
    with pytest.raises(ValueError, match="every outer fold"):
        _load_test_universe(partial_path)
    assert not mapping_path.exists()

    full_universe = _load_test_universe(_write_universe(tmp_path / "all_folds.json"))
    derangement = freeze_scoring_derangement(full_universe, mapping_path)
    alignment = alignment_metric_values(grid, derangement, arms=CORE_ARMS)
    assert set(alignment) == {
        (arm, seed, 0) for arm in CORE_ARMS for seed in TRAINING_SEEDS
    }


def test_grid_rejects_missing_duplicate_and_cross_arm_drift(tmp_path: Path) -> None:
    paths = _write_grid(tmp_path / "grid", arms=("FULL", "B4-GATTN"), folds=FULL_FOLDS)
    artifacts = [load_prediction_artifact(path) for path in paths]
    with pytest.raises(ValueError, match="Incomplete OOF artifact grid"):
        validate_artifact_grid(
            [artifact for artifact in artifacts if artifact.outer_fold != 3],
            protocol_id=PROTOCOL_ID,
            dataset_key="CWRU",
            dataset_slug="cwru",
            expected_arms=("FULL", "B4-GATTN"),
            expected_seeds=TRAINING_SEEDS,
            expected_folds=FULL_FOLDS,
        )
    with pytest.raises(ValueError, match="Duplicate artifact cell"):
        validate_artifact_grid(
            artifacts + [artifacts[0]],
            protocol_id=PROTOCOL_ID,
            dataset_key="CWRU",
            dataset_slug="cwru",
            expected_arms=("FULL", "B4-GATTN"),
            expected_seeds=TRAINING_SEEDS,
            expected_folds=FULL_FOLDS,
        )

    target = next(artifact for artifact in artifacts if artifact.arm_id == "B4-GATTN")
    changed_arrays = dict(target.arrays)
    changed_groups = np.asarray(changed_arrays["group_id"]).astype(str).copy()
    changed_groups[0] = "unexpected-group"
    changed_arrays["group_id"] = changed_groups
    changed = replace(target, arrays=changed_arrays)
    drifted = [changed if artifact is target else artifact for artifact in artifacts]
    with pytest.raises(ValueError, match="Cross-arm/seed test rows differ"):
        validate_artifact_grid(
            drifted,
            protocol_id=PROTOCOL_ID,
            dataset_key="CWRU",
            dataset_slug="cwru",
            expected_arms=("FULL", "B4-GATTN"),
            expected_seeds=TRAINING_SEEDS,
            expected_folds=FULL_FOLDS,
        )


def test_grid_rejects_missing_arm_specific_representation(tmp_path: Path) -> None:
    paths = _write_grid(tmp_path / "grid", arms=("FULL", "B4-GATTN"), folds=FULL_FOLDS)
    artifacts = [load_prediction_artifact(path) for path in paths]
    target = next(artifact for artifact in artifacts if artifact.arm_id == "B4-GATTN")
    changed_arrays = dict(target.arrays)
    del changed_arrays["repr__encoded_2d"]
    changed = replace(target, arrays=changed_arrays)
    with pytest.raises(ValueError, match="repr__encoded_2d"):
        validate_artifact_grid(
            [changed if artifact is target else artifact for artifact in artifacts],
            protocol_id=PROTOCOL_ID,
            dataset_key="CWRU",
            dataset_slug="cwru",
            expected_arms=("FULL", "B4-GATTN"),
            expected_seeds=TRAINING_SEEDS,
            expected_folds=FULL_FOLDS,
        )


def test_bootstrap_uses_one_paired_draw_for_all_arms_and_is_deterministic(tmp_path: Path) -> None:
    paths = _write_grid(tmp_path / "grid", arms=("FULL", "B4-GATTN"), folds=FULL_FOLDS)
    grid = _load_grid(paths, arms=("FULL", "B4-GATTN"), folds=FULL_FOLDS, scope="final_oof")
    identical = {}
    for seed in TRAINING_SEEDS:
        for fold in FULL_FOLDS:
            pattern = np.asarray([0.0, 1.0, 0.25, 0.75]) + seed / 100000.0
            identical[("FULL", seed, fold)] = pattern
            identical[("B4-GATTN", seed, fold)] = pattern.copy()
    first = paired_hierarchical_bootstrap(
        grid,
        {"paired_fixture": identical},
        "FULL",
        "B4-GATTN",
        replicates=128,
        seed=20260801,
    )
    second = paired_hierarchical_bootstrap(
        grid,
        {"paired_fixture": identical},
        "FULL",
        "B4-GATTN",
        replicates=128,
        seed=20260801,
    )
    assert np.array_equal(first.replicate_effects["paired_fixture"], np.zeros(128))
    assert np.array_equal(
        first.replicate_effects["paired_fixture"],
        second.replicate_effects["paired_fixture"],
    )
    assert first.sampled_index_sha256 == second.sampled_index_sha256
    assert first.metrics["paired_fixture"].interval_lower_mcse == 0.0


def test_loader_rejects_missing_fold_seed_provenance(tmp_path: Path) -> None:
    target = _write_prediction(tmp_path, arm="FULL", seed=42, fold=0)
    manifest_path = target.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["provenance"]["outer_fold"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outer_fold and training_seed"):
        load_prediction_artifact(target)


def test_loader_rejects_incomplete_full_provenance_contract(tmp_path: Path) -> None:
    target = _write_prediction(tmp_path, arm="FULL", seed=42, fold=0)
    manifest_path = target.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["provenance"]["data_payload_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="data_payload_sha256"):
        load_prediction_artifact(target)


def test_loader_rejects_trainer_metrics_csv_drift(tmp_path: Path) -> None:
    target = _write_prediction(tmp_path, arm="FULL", seed=42, fold=0)
    manifest = json.loads(
        target.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    metrics_path = Path(manifest["provenance"]["trainer_metrics_path"])
    metrics_path.write_text("epoch,step,val_loss\n0,0,9.99\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_prediction_artifact(target)


def test_loader_rejects_cross_cell_invocation_identity_rebinding(
    tmp_path: Path,
) -> None:
    target = _write_prediction(tmp_path, arm="FULL", seed=42, fold=0)
    manifest_path = target.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    invocation_path = Path(provenance["invocation_path"])
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    invocation["paper"]["arm_id"] = "B4-GATTN"
    invocation_path.write_text(json.dumps(invocation), encoding="utf-8")
    provenance["invocation_sha256"] = hashlib.sha256(
        invocation_path.read_bytes()
    ).hexdigest()
    data_snapshot_path = Path(provenance["data_snapshot_manifest_path"])
    data_snapshot = json.loads(data_snapshot_path.read_text(encoding="utf-8"))
    data_snapshot["invocation_sha256"] = provenance["invocation_sha256"]
    data_snapshot_path.write_text(json.dumps(data_snapshot), encoding="utf-8")
    provenance["data_snapshot_manifest_sha256"] = hashlib.sha256(
        data_snapshot_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="paper identity mismatch for arm_id"):
        load_prediction_artifact(target)


def test_cli_writes_g050_json_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        score_module.statistics_module.FROZEN_SPLIT_MANIFEST_SHA256S,
        "CWRU",
        tuple(_split_hash(fold) for fold in FULL_FOLDS),
    )
    paths = _write_grid(tmp_path / "g050_cli", arms=CORE_ARMS, folds=(0,))
    universe_path = _write_universe(tmp_path / "universe.json")
    mapping_path = tmp_path / "analysis" / "scoring_pairing" / "cwru.json"
    output_path = tmp_path / "analysis" / "g050_summary.json"
    argv = [
        "--predictions",
        *map(str, paths),
        "--protocol-id",
        PROTOCOL_ID,
        "--dataset-key",
        "CWRU",
        "--dataset-slug",
        "cwru",
        "--arms",
        *CORE_ARMS,
        "--seeds",
        *map(str, TRAINING_SEEDS),
        "--folds",
        "0",
        "--analysis-scope",
        "g050_fold0",
        "--contrast",
        "FULL",
        "B4-GATTN",
        "--sample-universe-json",
        str(universe_path),
        "--scoring-manifest",
        str(mapping_path),
        "--output",
        str(output_path),
    ]
    with pytest.raises(ValueError, match="exactly 10000"):
        score_main([*argv[:-2], "--bootstrap-replicates", "32", *argv[-2:]])
    assert score_main(argv) == 0
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["dataset_key"] == "CWRU"
    assert summary["analysis_scope"] == "g050_fold0"
    assert summary["outer_folds"] == [0]
    assert summary["paired_hierarchical_bootstrap"]["replicates"] == 10000
    assert summary["scoring_derangement"]["path"] == str(mapping_path)
    assert summary["analysis_code_state"]["code_state_sha256"]
    assert summary["design_strata_binding"]["mapping_sha256"]
    assert "lower_endpoint_audits" in summary["paired_hierarchical_bootstrap"]
    assert summary["shared_collapse"]["evidence_role"] == (
        "fold0_local_diagnostic_not_C1_support"
    )


def test_final_cli_emits_c1_absolute_97_5pct_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        score_module.statistics_module.FROZEN_SPLIT_MANIFEST_SHA256S,
        "CWRU",
        tuple(_split_hash(fold) for fold in FULL_FOLDS),
    )
    paths = _write_grid(
        tmp_path / "final_cli", arms=("FULL", "B4-GATTN"), folds=FULL_FOLDS
    )
    universe_path = _write_universe(tmp_path / "universe.json")
    output_path = tmp_path / "analysis" / "final_summary.json"
    mapping_path = tmp_path / "analysis" / "scoring_pairing" / "cwru.json"
    original_paired = paired_hierarchical_bootstrap
    original_single = single_arm_hierarchical_bootstrap

    def _fast_paired(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["replicates"] = 32
        return original_paired(*args, **kwargs)

    def _fast_single(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["replicates"] = 32
        return original_single(*args, **kwargs)

    monkeypatch.setattr(score_module, "paired_hierarchical_bootstrap", _fast_paired)
    monkeypatch.setattr(score_module, "single_arm_hierarchical_bootstrap", _fast_single)
    argv = [
        "--predictions",
        *map(str, paths),
        "--protocol-id",
        PROTOCOL_ID,
        "--dataset-key",
        "CWRU",
        "--dataset-slug",
        "cwru",
        "--arms",
        "FULL",
        "B4-GATTN",
        "--seeds",
        *map(str, TRAINING_SEEDS),
        "--folds",
        *map(str, FULL_FOLDS),
        "--analysis-scope",
        "final_oof",
        "--contrast",
        "FULL",
        "B4-GATTN",
        "--sample-universe-json",
        str(universe_path),
        "--scoring-manifest",
        str(mapping_path),
        "--output",
        str(output_path),
    ]
    assert score_main(argv) == 0
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    c1 = summary["C1_absolute_full_alignment_bootstrap_97_5pct"]
    assert c1["metrics"]["absolute_full_alignment_margin"][
        "confidence_level"
    ] == 0.975
    assert "lower_endpoint_audits" in c1
    assert summary["shared_collapse"]["evidence_role"] == (
        "C1_no_collapse_component"
    )


def test_xjtu_group_strata_binding_records_path_file_and_mapping_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "xjtu_domain_strata.json"
    path.write_text(
        json.dumps({"XJTU": {"bearing-a": 0, "bearing-b": 1}}),
        encoding="utf-8",
    )
    mapping, binding = score_module._load_group_strata(str(path), "XJTU")
    assert mapping == {"bearing-a": "0", "bearing-b": "1"}
    assert binding["path"] == str(path.resolve())
    assert binding["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert binding["mapping_sha256"]
