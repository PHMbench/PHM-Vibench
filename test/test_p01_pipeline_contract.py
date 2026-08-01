from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.Pipeline_01_Fault_Diagnosis import (
    _best_checkpoint_provenance,
    _code_file_hashes,
    _validate_evidence_runtime,
    _validate_p01_trainable_parameter_count,
    _write_trainer_metrics_manifest,
)
from src.configs.config_utils import merge_with_local_override
from src.utils.config_utils import apply_overrides_to_config, parse_overrides


EMPTY_LOCAL = "configs/experiments/p01/no_local_override.yaml"


def _evidence_config():  # type: ignore[no-untyped-def]
    config = merge_with_local_override(
        "configs/experiments/p01/p01_shared_private_cwru.yaml",
        EMPTY_LOCAL,
    )
    config.paper.evidence_status = "candidate_unreviewed"
    config.environment.output_dir = (
        "results/p01/P01-G040-v1/cwru/FULL/fold_0/seed_42/attempt_0"
    )
    return config


def test_evidence_runtime_accepts_only_the_bound_single_gpu_contract(
    monkeypatch,
) -> None:
    config = _evidence_config()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    _validate_evidence_runtime(
        config,
        SimpleNamespace(local_config=EMPTY_LOCAL),
    )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError, match="GPU index 2"):
        _validate_evidence_runtime(
            config,
            SimpleNamespace(local_config=EMPTY_LOCAL),
        )


def test_evidence_runtime_rejects_noncanonical_local_override(
    tmp_path, monkeypatch
) -> None:
    local = tmp_path / "override.yaml"
    local.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="canonical empty"):
        _validate_evidence_runtime(
            _evidence_config(),
            SimpleNamespace(local_config=str(local)),
        )


@pytest.mark.parametrize("attempt_id", [-1, 2, True, "0"])
def test_evidence_runtime_rejects_invalid_attempt_id(
    attempt_id, monkeypatch
) -> None:
    config = _evidence_config()
    config.paper.attempt_id = attempt_id
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="attempt_id"):
        _validate_evidence_runtime(
            config,
            SimpleNamespace(local_config=EMPTY_LOCAL),
        )


def test_evidence_runtime_binds_attempt_to_output_identity(monkeypatch) -> None:
    config = _evidence_config()
    config.paper.attempt_id = 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="dataset/arm/fold/seed/attempt"):
        _validate_evidence_runtime(
            config,
            SimpleNamespace(local_config=EMPTY_LOCAL),
        )

    config.environment.output_dir = (
        "results/p01/P01-G040-v1/cwru/FULL/fold_0/seed_42/attempt_1"
    )
    _validate_evidence_runtime(
        config,
        SimpleNamespace(local_config=EMPTY_LOCAL),
    )


def test_evidence_runtime_rejects_unregistered_arm(monkeypatch) -> None:
    config = _evidence_config()
    config.paper.arm_id = "UNREGISTERED"
    config.environment.output_dir = (
        "results/p01/P01-G040-v1/cwru/UNREGISTERED/"
        "fold_0/seed_42/attempt_0"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="unregistered paper.arm_id"):
        _validate_evidence_runtime(
            config,
            SimpleNamespace(local_config=EMPTY_LOCAL),
        )


def test_evidence_runtime_rejects_registered_arm_model_drift(monkeypatch) -> None:
    config = _evidence_config()
    config.model.encoder_dim = 65
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="model contract drift"):
        _validate_evidence_runtime(
            config,
            SimpleNamespace(local_config=EMPTY_LOCAL),
        )


def test_evidence_runtime_rejects_objective_and_pairing_drift(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    objective_drift = _evidence_config()
    objective_drift.task.auxiliary_loss_weights.alignment = 0.2
    with pytest.raises(RuntimeError, match="auxiliary-loss contract drift"):
        _validate_evidence_runtime(
            objective_drift,
            SimpleNamespace(local_config=EMPTY_LOCAL),
        )

    pairing_drift = _evidence_config()
    pairing_drift.data.pairing.mode = "batch_shuffle"
    with pytest.raises(RuntimeError, match="pairing contract drift"):
        _validate_evidence_runtime(
            pairing_drift,
            SimpleNamespace(local_config=EMPTY_LOCAL),
        )


def test_instantiated_model_parameter_count_is_frozen() -> None:
    config = _evidence_config()

    class CountModel(torch.nn.Module):
        def __init__(self, count: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(count))

    _validate_p01_trainable_parameter_count(config, CountModel(78596))
    with pytest.raises(RuntimeError, match="trainable-parameter count drift"):
        _validate_p01_trainable_parameter_count(config, CountModel(78595))


def test_code_state_hashes_cover_p01_execution_modules() -> None:
    hashes = _code_file_hashes(_evidence_config())
    required = {
        "src/Pipeline_01_Fault_Diagnosis.py",
        "src/data_factory/data_factory.py",
        "src/data_factory/dataset_task/Dataset_cluster.py",
        "src/task_factory/Default_task.py",
        "src/trainer_factory/Default_trainer.py",
        "src/model_factory/X_model/P01SharedPrivate.py",
        "src/utils/evaluation_artifacts.py",
    }
    assert required <= set(hashes)
    assert all(len(digest) == 64 for digest in hashes.values())


def test_best_checkpoint_provenance_binds_content(tmp_path) -> None:
    checkpoint_path = tmp_path / "best.ckpt"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint_path)
    callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=tmp_path)
    callback.best_model_path = str(checkpoint_path)
    callback.best_model_score = torch.tensor(0.25)
    provenance = _best_checkpoint_provenance(
        SimpleNamespace(callbacks=[callback])
    )
    assert provenance["path"] == str(checkpoint_path.resolve())
    assert provenance["monitor"] == "val_loss"
    assert provenance["mode"] == "min"
    assert provenance["score"] == pytest.approx(0.25)
    assert len(provenance["sha256"]) == 64


def test_trainer_metrics_manifest_is_hash_bound_and_write_once(tmp_path) -> None:
    run_path = tmp_path / "run"
    logger = CSVLogger(str(run_path), name="logs", version=0)
    logger.log_metrics({"val_loss": 0.25}, step=1)
    logger.save()

    provenance = _write_trainer_metrics_manifest(
        SimpleNamespace(loggers=[logger]),
        run_path,
    )
    manifest_path = run_path / "artifacts" / "trainer_metrics.manifest.json"
    metrics_path = run_path / "logs" / "version_0" / "metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert provenance["trainer_metrics_manifest_path"] == str(
        manifest_path.resolve()
    )
    assert provenance["trainer_metrics_path"] == str(metrics_path.resolve())
    assert provenance["trainer_metrics_sha256"] == manifest["metrics_sha256"]
    assert len(provenance["trainer_metrics_manifest_sha256"]) == 64
    assert manifest["logger_version"] == 0
    assert manifest["metrics_path"] == str(metrics_path.resolve())

    with pytest.raises(FileExistsError):
        _write_trainer_metrics_manifest(
            SimpleNamespace(loggers=[logger]),
            run_path,
        )


@pytest.mark.parametrize(
    ("config_path", "arm_id", "arm_overrides"),
    [
        (
            "configs/experiments/p01/p01_shared_private_cwru.yaml",
            "FULL",
            [],
        ),
        (
            "configs/experiments/p01/p01_generic_attention_cwru.yaml",
            "B4-GATTN",
            ["paper.baseline_id=B4-GATTN"],
        ),
        (
            "configs/experiments/p01/p01_shared_private_cwru.yaml",
            "TRAIN-MISPAIR",
            [
                "paper.ablation_id=TRAIN-MISPAIR",
                "paper.supports_claim_ids=[C3]",
                "data.pairing.mode=frozen_within_group_class_derangement",
                "data.pairing.seed=20260801",
                "data.pairing.splits=[train]",
                "data.pairing.protocol_id=P01-G040-v1",
                "data.pairing.group_key=File",
                "data.pairing.manifest_dir=results/p01/P01-G040-v1/protocol/pairing/cwru/fold_0",
            ],
        ),
    ],
)
def test_exact_g050_identity_overrides_pass_runtime_admission(
    config_path: str,
    arm_id: str,
    arm_overrides: list[str],
    monkeypatch,
) -> None:
    config = merge_with_local_override(config_path, EMPTY_LOCAL)
    common = [
        "environment.seed=42",
        f"environment.output_dir=results/p01/P01-G040-v1/cwru/{arm_id}/fold_0/seed_42/attempt_0",
        "data.split.outer_fold=0",
        "data.split.manifest_path=results/p01/P01-G040-v1/protocol/splits/cwru/fold_0.json",
        "data.split.expected_manifest_payload_sha256=ed14b16912d91fd7d92d81bfb6d4e0fcdabe9fdc9fe5c56d613dc9143f8cc202",
        f"paper.arm_id={arm_id}",
        "paper.dataset_key=CWRU",
        "paper.dataset_slug=cwru",
        "paper.dataset_id=1",
        "paper.attempt_id=0",
        "paper.evidence_status=candidate_unreviewed",
    ]
    config = apply_overrides_to_config(
        config,
        parse_overrides([*common, *arm_overrides]),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    _validate_evidence_runtime(
        config,
        SimpleNamespace(local_config=EMPTY_LOCAL),
    )
