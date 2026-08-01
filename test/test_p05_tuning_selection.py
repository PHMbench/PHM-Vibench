from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.utils.p05_tuning_selection as tuning_selection
from src.utils.p05_tuning_selection import export_p05_tuning_selection


MATRIX_HASH = "10" * 32
ARMS = ("P05-M", "P05-B0", "P05-B1", "P05-B3")
DATASETS = (("CWRU", 1), ("XJTU", 2))
RATES = ((1.0e-3, "LR1E3"), (3.0e-4, "LR3E4"))


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _write_semantic(path: Path, semantic: dict) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    semantic_hash = _hash_bytes(_canonical(semantic))
    manifest = {**semantic, "content": {"semantic_sha256": semantic_hash}}
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return semantic_hash, _hash_file(path)


def _rewrite_semantic(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    semantic = {key: item for key, item in value.items() if key != "content"}
    mutate(semantic)
    _write_semantic(path, semantic)


def _provenance(dataset_id: int, *, variant: int = 0) -> dict[str, str]:
    base = 30 + dataset_id * 10 + variant
    return {
        "source_metadata_sha256": "31" * 32,
        "derived_metadata_sha256": "32" * 32,
        "signal_cache_manifest_sha256": "33" * 32,
        "split_manifest_sha256": f"{base:02d}" * 32,
        "normalization_sha256": f"{base + 1:02d}" * 32,
        "train_weight_plan_sha256": f"{base + 2:02d}" * 32,
        "validation_weight_plan_sha256": f"{base + 3:02d}" * 32,
    }


def _metrics(arm: str, dataset: str, rate: float) -> tuple[float, float]:
    if dataset == "CWRU" and arm == "P05-M":
        return (0.80005, 0.40) if rate == 1.0e-3 else (0.80000, 0.30)
    if dataset == "CWRU" and arm == "P05-B0":
        return (0.70, 0.20)
    if dataset == "CWRU" and arm == "P05-B1":
        return (0.70011, 0.50) if rate == 1.0e-3 else (0.70000, 0.10)
    if dataset == "CWRU" and arm == "P05-B3":
        return (0.70010, 0.50) if rate == 1.0e-3 else (0.70000, 0.10)
    return (0.75, 0.40) if rate == 1.0e-3 else (0.65, 0.30)


def _make_grid(tmp_path: Path) -> list[Path]:
    shared = tmp_path / "shared"
    code_manifest = shared / "code_snapshot.json"
    code_semantic = {
        "schema_name": "p05.code_snapshot",
        "schema_version": 1,
        "paper_id": "P05",
        "identity": "test-fixture",
    }
    observed_code_hash, _ = _write_semantic(code_manifest, code_semantic)
    # The selector binds the semantic hash actually recorded by the source
    # manifest; this local alias keeps all sixteen jobs on exactly one snapshot.
    code_hash = observed_code_hash

    candidates = []
    for dataset, dataset_id in DATASETS:
        provenance = _provenance(dataset_id)
        for arm in ARMS:
            arm_short = arm[4:]
            for rate, rate_token in RATES:
                job_id = f"P05-TUNE-{arm_short}-{dataset}-{rate_token}"
                run_dir = tmp_path / "runs" / job_id
                run_dir.mkdir(parents=True)
                config = run_dir / "config_snapshot.yaml"
                config.write_text(
                    f"job_id: {job_id}\nlearning_rate: {rate}\n",
                    encoding="utf-8",
                )
                checkpoint = run_dir / "best.ckpt"
                checkpoint.write_bytes(f"checkpoint:{job_id}".encode("ascii"))
                config_hash = _hash_file(config)
                checkpoint_hash = _hash_file(checkpoint)

                run_contract = run_dir / "run_contract.json"
                run_contract_semantic = {
                    "schema_name": "p05.run_artifact_bundle",
                    "schema_version": 1,
                    "paper_id": "P05",
                    "dataset_id": dataset_id,
                    "normalization_plan": {
                        "sha256": provenance["normalization_sha256"]
                    },
                    "weight_plans": {
                        "train": {
                            "sha256": provenance["train_weight_plan_sha256"]
                        },
                        "validation": {
                            "sha256": provenance["validation_weight_plan_sha256"]
                        },
                    },
                    "provenance": {
                        "config_sha256": config_hash,
                        "code_sha256": code_hash,
                        "checkpoint_sha256": checkpoint_hash,
                        "model_sha256": "99" * 32,
                    },
                }
                run_contract_hash, _ = _write_semantic(
                    run_contract,
                    run_contract_semantic,
                )
                f1, loss = _metrics(arm, dataset, rate)
                candidate_path = run_dir / "tuning_validation_candidate.json"
                candidate_semantic = {
                    "schema_name": "p05.tuning_validation_candidate",
                    "schema_version": 1,
                    "paper_id": "P05",
                    "protocol_bundle_sha256": tuning_selection.PROTOCOL_BUNDLE_SHA256,
                    "source_matrix_sha256": MATRIX_HASH,
                    "job": {
                        "job_id": job_id,
                        "phase": "tuning",
                        "arm_id": arm,
                        "dataset": dataset,
                        "dataset_id": dataset_id,
                        "seed": 20260801,
                        "learning_rate": rate,
                    },
                    "execution": {
                        "status": "completed",
                        "stage": "fit_validate_only",
                        "evidence_eligible": False,
                        "claim_decision": "not_performed",
                        "data_roles_constructed": ["train", "validation"],
                        "test_access_count": 0,
                        "max_epochs": 60,
                        "patience": 10,
                        "epochs_completed": 17,
                        "checkpoint_monitor": "val_loss",
                        "checkpoint_mode": "min",
                        "save_top_k": 1,
                        "selected_checkpoint_count": 1,
                    },
                    "validation": {
                        "partition": "validation",
                        "checkpoint_epoch": 8,
                        "val_loss": loss,
                        "val_f1_macro": f1,
                        "loss_definition": "group_equal_weighted_cross_entropy",
                        "macro_f1_construction": (
                            "one_epoch_level_weighted_confusion_matrix"
                        ),
                        "weighting": "equal_group_then_equal_window",
                        "zero_division": 0,
                    },
                    "artifacts": {
                        "config_snapshot": {
                            "path": str(config),
                            "sha256": config_hash,
                        },
                        "code_snapshot": {
                            "path": str(code_manifest),
                            "semantic_sha256": code_hash,
                        },
                        "run_contract": {
                            "path": str(run_contract),
                            "semantic_sha256": run_contract_hash,
                        },
                        "checkpoint": {
                            "path": str(checkpoint),
                            "sha256": checkpoint_hash,
                        },
                    },
                    "provenance": provenance,
                }
                _write_semantic(candidate_path, candidate_semantic)
                candidates.append(candidate_path)
    return candidates


def test_selects_exact_grid_with_frozen_ties_and_downstream_index(tmp_path) -> None:
    candidates = _make_grid(tmp_path)
    result = export_p05_tuning_selection(
        tmp_path / "selection",
        candidate_manifest_paths=list(reversed(candidates)),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "created"
    assert manifest["schema_name"] == "p05.tuning_selection"
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "computed_unadjudicated"
    assert manifest["claim_decision"] == "not_performed"
    assert manifest["evidence_eligible"] is False
    assert manifest["test_access"] == "forbidden_and_not_performed"
    assert manifest["source_matrix_sha256"] == MATRIX_HASH
    assert len(manifest["candidates"]) == 16
    assert len(manifest["selections"]) == 8
    assert len(manifest["selection_index"]) == 8

    assert manifest["selection_index"]["CWRU/P05-M"]["selected_learning_rate"] == 3e-4
    assert manifest["selection_index"]["CWRU/P05-B0"]["selected_learning_rate"] == 3e-4
    assert manifest["selection_index"]["CWRU/P05-B1"]["selected_learning_rate"] == 1e-3
    assert manifest["selection_index"]["CWRU/P05-B3"]["selected_learning_rate"] == 3e-4
    for key, indexed in manifest["selection_index"].items():
        row = manifest["selections"][indexed["row_index"]]
        dataset, arm = key.split("/", maxsplit=1)
        assert (row["dataset"], row["arm_id"]) == (dataset, arm)
        assert indexed["selection_id"] == row["selection_id"]
        assert indexed["selected_learning_rate"] == row["selected_learning_rate"]
        assert indexed["selected_job_id"] == row["selected_job_id"]
        assert indexed["selected_checkpoint_sha256"] == row[
            "selected_checkpoint_sha256"
        ]
        assert indexed["selected_run_contract_sha256"] == row[
            "selected_run_contract_sha256"
        ]
    semantic = {key: value for key, value in manifest.items() if key != "content"}
    assert manifest["content"]["semantic_sha256"] == _hash_bytes(_canonical(semantic))


def test_selection_is_create_only_reusable_and_conflict_preserving(tmp_path) -> None:
    candidates = _make_grid(tmp_path)
    package = tmp_path / "selection"
    created = export_p05_tuning_selection(package, candidate_manifest_paths=candidates)
    before = created.manifest_path.read_bytes()
    reused = export_p05_tuning_selection(
        package,
        candidate_manifest_paths=list(reversed(candidates)),
    )
    assert reused.status == "reused"
    assert reused.manifest_path.read_bytes() == before

    _rewrite_semantic(
        candidates[0],
        lambda value: value["validation"].update(val_f1_macro=0.91),
    )
    with pytest.raises(FileExistsError, match="content conflicts"):
        export_p05_tuning_selection(package, candidate_manifest_paths=candidates)
    assert created.manifest_path.read_bytes() == before


def test_rejects_incomplete_duplicate_and_test_touched_candidates(tmp_path) -> None:
    candidates = _make_grid(tmp_path)
    with pytest.raises(ValueError, match="exactly 16"):
        export_p05_tuning_selection(
            tmp_path / "missing",
            candidate_manifest_paths=candidates[:-1],
        )
    duplicated = [*candidates[:-1], candidates[0]]
    with pytest.raises(ValueError, match="job IDs must be unique"):
        export_p05_tuning_selection(
            tmp_path / "duplicate",
            candidate_manifest_paths=duplicated,
        )

    _rewrite_semantic(
        candidates[0],
        lambda value: value["execution"].update(test_access_count=1),
    )
    with pytest.raises(ValueError, match="test_access_count"):
        export_p05_tuning_selection(
            tmp_path / "test-touched",
            candidate_manifest_paths=candidates,
        )
    assert not (tmp_path / "test-touched").exists()


def test_rejects_incomplete_status_tampering_and_nonfinite_metric(tmp_path) -> None:
    candidates = _make_grid(tmp_path)
    _rewrite_semantic(
        candidates[0],
        lambda value: value["execution"].update(status="failed"),
    )
    with pytest.raises(ValueError, match="status"):
        export_p05_tuning_selection(
            tmp_path / "failed-candidate",
            candidate_manifest_paths=candidates,
        )

    candidates = _make_grid(tmp_path / "bad-id")
    _rewrite_semantic(
        candidates[0],
        lambda value: value["job"].update(job_id="P05-TUNE-B0-CWRU-LR1E3"),
    )
    with pytest.raises(ValueError, match="job_id conflicts"):
        export_p05_tuning_selection(
            tmp_path / "misbound-job",
            candidate_manifest_paths=candidates,
        )

    candidates = _make_grid(tmp_path / "tamper")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    payload["validation"]["val_loss"] = 99.0
    candidates[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        export_p05_tuning_selection(
            tmp_path / "tampered",
            candidate_manifest_paths=candidates,
        )

    candidates = _make_grid(tmp_path / "nan")
    # json.dumps would emit NaN, while the production reader explicitly rejects
    # all non-standard JSON constants before any output directory is created.
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    payload["validation"]["val_loss"] = float("nan")
    candidates[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid P05 tuning validation candidate"):
        export_p05_tuning_selection(
            tmp_path / "nonfinite",
            candidate_manifest_paths=candidates,
        )


def test_rejects_artifact_and_cross_candidate_provenance_drift(tmp_path) -> None:
    candidates = _make_grid(tmp_path)
    first = json.loads(candidates[0].read_text(encoding="utf-8"))
    Path(first["artifacts"]["checkpoint"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file hash mismatch"):
        export_p05_tuning_selection(
            tmp_path / "bad-checkpoint",
            candidate_manifest_paths=candidates,
        )

    candidates = _make_grid(tmp_path / "matrix-drift")
    _rewrite_semantic(
        candidates[0],
        lambda value: value.update(source_matrix_sha256="88" * 32),
    )
    with pytest.raises(ValueError, match="same source matrix"):
        export_p05_tuning_selection(
            tmp_path / "source-matrix-drift",
            candidate_manifest_paths=candidates,
        )

    candidates = _make_grid(tmp_path / "drift")
    candidate_path = candidates[1]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    run_contract_path = Path(candidate["artifacts"]["run_contract"]["path"])
    new_hash = "77" * 32
    _rewrite_semantic(
        run_contract_path,
        lambda value: value["normalization_plan"].update(sha256=new_hash),
    )
    run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))

    def update_candidate(value):
        value["provenance"]["normalization_sha256"] = new_hash
        value["artifacts"]["run_contract"]["semantic_sha256"] = run_contract[
            "content"
        ]["semantic_sha256"]

    _rewrite_semantic(candidate_path, update_candidate)
    with pytest.raises(ValueError, match="all CWRU.*normalization"):
        export_p05_tuning_selection(
            tmp_path / "provenance-drift",
            candidate_manifest_paths=candidates,
        )


def test_atomic_failure_and_symlink_target_leave_no_partial_package(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = _make_grid(tmp_path)

    def fail_install(source, target):
        del source, target
        raise RuntimeError("synthetic install failure")

    monkeypatch.setattr(tuning_selection, "_rename_directory_noreplace", fail_install)
    package = tmp_path / "selection"
    with pytest.raises(RuntimeError, match="synthetic install failure"):
        export_p05_tuning_selection(package, candidate_manifest_paths=candidates)
    assert not package.exists()
    assert not list(tmp_path.glob(".selection.*.tmp"))

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symlink"):
        export_p05_tuning_selection(linked, candidate_manifest_paths=candidates)
    assert not list(real.iterdir())
