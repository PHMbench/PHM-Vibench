import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import src.Pipeline_05_Explainable_Fault_Diagnosis as pipeline_module
from src.configs.p05_contract import P05ExperimentContract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(*, phase="tuning", arm_id="P05-B0", dataset_id=1, seed=20260801):
    return P05ExperimentContract(
        arm_id=arm_id,
        dataset="CWRU" if dataset_id == 1 else "XJTU",
        dataset_id=dataset_id,
        phase=phase,
        seed=seed,
        trace_export=arm_id == "P05-M" and phase != "tuning",
    )


def test_p05_config_loader_uses_only_materialized_source(monkeypatch) -> None:
    base = SimpleNamespace(trainer=SimpleNamespace(p05_evidence_mode=True))
    args = SimpleNamespace(
        config_path="/tmp/materialized/config.yaml",
        local_config=None,
        override=[],
    )
    monkeypatch.setattr(pipeline_module, "load_config", lambda path: base)
    monkeypatch.setattr(
        pipeline_module,
        "merge_with_local_override",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("P05 must not consult implicit local config")
        ),
    )

    assert pipeline_module._load_pipeline_config(args) is base


@pytest.mark.parametrize(
    ("local_config", "override", "message"),
    [
        ("configs/local/local.yaml", [], "local config"),
        (None, ["task.lr=0.1"], "CLI config"),
    ],
)
def test_p05_config_loader_rejects_explicit_mutation_before_merge(
    monkeypatch,
    local_config,
    override,
    message,
) -> None:
    base = SimpleNamespace(trainer=SimpleNamespace(p05_evidence_mode=True))
    args = SimpleNamespace(
        config_path="/tmp/materialized/config.yaml",
        local_config=local_config,
        override=override,
    )
    monkeypatch.setattr(pipeline_module, "load_config", lambda path: base)
    monkeypatch.setattr(
        pipeline_module,
        "merge_with_local_override",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rejected P05 mutation must not be read or merged")
        ),
    )

    with pytest.raises(ValueError, match=message):
        pipeline_module._load_pipeline_config(args)


def test_legacy_config_loader_preserves_local_merge(monkeypatch) -> None:
    base = SimpleNamespace(trainer=SimpleNamespace(p05_evidence_mode=False))
    merged = object()
    args = SimpleNamespace(
        config_path="legacy.yaml",
        local_config="local.yaml",
        override=[],
    )
    monkeypatch.setattr(pipeline_module, "load_config", lambda path: base)
    monkeypatch.setattr(
        pipeline_module,
        "merge_with_local_override",
        lambda path, local: merged,
    )

    assert pipeline_module._load_pipeline_config(args) is merged


def _complete_provenance() -> dict[str, str]:
    return {
        name: f"{index:02x}" * 32
        for index, name in enumerate(
            pipeline_module._P05_ATTEMPT_PROVENANCE_FIELDS,
            start=1,
        )
    }


def test_static_provenance_binds_registered_files_without_reading_source_workbook(
    tmp_path,
):
    metadata = tmp_path / "metadata_p05_v2.csv"
    metadata.write_text("Id,Dataset_id\n1,1\n", encoding="utf-8")
    split = tmp_path / "cwru_split.json"
    split.write_text('{"split":"frozen"}\n', encoding="utf-8")
    cache_manifest = tmp_path / "cache_manifest.json"
    cache_manifest.write_text('{"cache":"verified"}\n', encoding="utf-8")
    config_snapshot = tmp_path / "config_snapshot.yaml"
    config_snapshot.write_text("paper_id: P05\n", encoding="utf-8")
    metadata_manifest = {
        "paper_id": "P05",
        "protocol_id": "P05-G040-v3.2",
        "source_workbook": {"sha256": "11" * 32},
        "derived_metadata": {
            "file": metadata.name,
            "csv_sha256": _sha256(metadata),
            "semantic_serialization": {"sha256": "22" * 32},
        },
        "split_manifests": {
            "CWRU": {"file": split.name, "sha256": _sha256(split)},
        },
    }
    metadata.with_suffix(".manifest.json").write_text(
        json.dumps(metadata_manifest),
        encoding="utf-8",
    )

    observed = pipeline_module._resolve_p05_static_provenance(
        args_data=SimpleNamespace(
            metadata_path=str(metadata),
            cache_manifest_path=str(cache_manifest),
        ),
        experiment_contract=_contract(),
        config_snapshot_path=config_snapshot,
        code_snapshot_sha256="33" * 32,
    )

    assert observed == {
        "source_metadata_sha256": "11" * 32,
        "derived_metadata_sha256": "22" * 32,
        "signal_cache_manifest_sha256": _sha256(cache_manifest),
        "split_manifest_sha256": _sha256(split),
        "config_snapshot_sha256": _sha256(config_snapshot),
        "code_snapshot_sha256": "33" * 32,
    }


def test_active_attempt_retains_partial_outputs_and_classifies_cuda_failure(
    tmp_path,
    monkeypatch,
):
    args = SimpleNamespace(
        config_path="job/config.yaml",
        requested_config="job/config.yaml",
        override=[],
        notes="",
        local_config=None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_p05_package_versions",
        lambda: {"python": "3.10.0", "torch": "2.0.0"},
    )
    provenance = _complete_provenance()
    started = pipeline_module._begin_pipeline_p05_attempt(
        args,
        run_path=tmp_path / "run",
        experiment_contract=_contract(),
        runtime_contract=SimpleNamespace(
            runtime_identity={
                "cuda_visible_devices": "0",
                "gpu_uuid": "GPU-test",
                "physical_gpu_index": 0,
            }
        ),
        provenance=provenance,
        started_at_utc="2026-08-01T00:00:00+00:00",
    )
    pipeline_module._record_p05_attempt_output(
        args,
        "checkpoint",
        "aa" * 32,
    )
    pipeline_module._finish_active_p05_attempt_failure(
        args,
        RuntimeError("CUDA worker exited"),
    )

    terminal = json.loads(
        (started.package_dir / "terminal.json").read_text(encoding="utf-8")
    )
    assert terminal["terminal"]["status"] == "failed"
    assert terminal["terminal"]["claim_decision"] == "not_performed"
    assert terminal["failure"]["category"] == "infrastructure"
    assert terminal["outputs"]["checkpoint"] == "aa" * 32
    assert set(terminal["missing_outputs"]) == {
        "all_results",
        "materialized_job",
        "result",
        "run_contract",
        "tuning_candidate",
    }
    assert args._p05_active_attempt_package is None


def test_tuning_attempt_completes_only_after_every_registered_output(tmp_path, monkeypatch):
    args = SimpleNamespace(
        config_path="job/config.yaml",
        requested_config="job/config.yaml",
        override=[],
        notes="",
        local_config=None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_p05_package_versions",
        lambda: {"python": "3.10.0", "torch": "2.0.0"},
    )
    started = pipeline_module._begin_pipeline_p05_attempt(
        args,
        run_path=tmp_path / "run",
        experiment_contract=_contract(),
        runtime_contract=SimpleNamespace(runtime_identity={"gpu_uuid": "GPU-test"}),
        provenance=_complete_provenance(),
        started_at_utc="2026-08-01T00:00:00+00:00",
    )
    for name, value in (
        ("checkpoint", "aa" * 32),
        ("materialized_job", "ff" * 32),
        ("run_contract", "bb" * 32),
        ("tuning_candidate", "ee" * 32),
        ("result", "cc" * 32),
        ("all_results", "dd" * 32),
    ):
        pipeline_module._record_p05_attempt_output(args, name, value)
    pipeline_module._finish_active_p05_attempt_success(args)

    terminal = json.loads(
        (started.package_dir / "terminal.json").read_text(encoding="utf-8")
    )
    assert terminal["terminal"]["status"] == "completed"
    assert terminal["missing_outputs"] == {}
    assert terminal["failure"] is None


class _Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def test_prediction_pipeline_route_is_decisive_only(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.ckpt"
    config.write_text("paper_id: P05\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    requests = []

    class _DataFactory:
        split_result = SimpleNamespace(
            train_ids=(2, 1),
            val_ids=(3,),
            test_ids=(4,),
        )

        def get_dataloader(self, split):
            requests.append(split)
            return [split]

    captured = {}

    def fake_export(package_dir, **kwargs):
        captured["package_dir"] = package_dir
        captured.update(kwargs)
        return SimpleNamespace(
            arrays_path=Path(package_dir) / "prediction_arrays.npz",
            arrays_sha256="10" * 32,
            manifest_path=Path(package_dir) / "manifest.json",
            manifest_sha256="20" * 32,
            semantic_sha256="30" * 32,
            status="created",
        )

    monkeypatch.setattr(
        pipeline_module,
        "export_p05_window_predictions",
        fake_export,
    )
    monkeypatch.setattr(pipeline_module, "model_state_sha256", lambda _: "40" * 32)
    task = SimpleNamespace(network=_Network())
    run_contract = {
        "code_semantic_sha256": "50" * 32,
        "semantic_sha256": "60" * 32,
    }

    skipped = pipeline_module._export_registered_p05_predictions(
        task=task,
        data_factory=_DataFactory(),
        run_path=tmp_path / "run",
        config_snapshot_path=config,
        checkpoint_path=checkpoint,
        run_contract_record=run_contract,
        expected_window_size=4096,
        experiment_contract=_contract(),
    )
    assert skipped == {}
    assert requests == []

    result = pipeline_module._export_registered_p05_predictions(
        task=task,
        data_factory=_DataFactory(),
        run_path=tmp_path / "run",
        config_snapshot_path=config,
        checkpoint_path=checkpoint,
        run_contract_record=run_contract,
        expected_window_size=4096,
        experiment_contract=_contract(
            phase="decisive",
            arm_id="P05-B1",
            dataset_id=1,
            seed=42,
        ),
    )
    assert requests == ["train", "val", "test"]
    assert captured["expected_record_ids_by_split"] == {
        "train": ["2", "1"],
        "val": ["3"],
        "test": ["4"],
    }
    assert captured["expected_windows_per_record"] == 16
    assert captured["require_cuda"] is True
    assert result["scientific_status"] == "computed_unadjudicated"


def test_d01_d02_pipeline_route_requires_both_decisive_method_traces(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.ckpt"
    config.write_text("paper_id: P05\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fake_diagnostics(artifact_dir, **kwargs):
        calls.append((Path(artifact_dir), kwargs))
        return SimpleNamespace(
            arrays_path=Path(artifact_dir) / "diagnostic_arrays.npz",
            arrays_sha256="10" * 32,
            manifest_path=Path(artifact_dir) / "manifest.json",
            manifest_sha256="20" * 32,
            semantic_sha256=("30" if str(artifact_dir).endswith("val") else "31")
            * 32,
            status="created",
        )

    monkeypatch.setattr(
        pipeline_module,
        "create_p05_d01_d02_trace_diagnostics",
        fake_diagnostics,
    )
    monkeypatch.setattr(pipeline_module, "model_state_sha256", lambda _: "40" * 32)
    task = SimpleNamespace(network=_Network())
    traces = {
        "val": {"package_dir": "/trace/val", "semantic_sha256": "50" * 32},
        "test": {"package_dir": "/trace/test", "semantic_sha256": "60" * 32},
    }

    assert pipeline_module._export_registered_p05_trace_diagnostics(
        task=task,
        run_path=tmp_path / "run",
        config_snapshot_path=config,
        checkpoint_path=checkpoint,
        trace_records={},
        experiment_contract=_contract(
            phase="pilot",
            arm_id="P05-M",
            dataset_id=1,
        ),
    ) == {}
    assert calls == []

    observed = pipeline_module._export_registered_p05_trace_diagnostics(
        task=task,
        run_path=tmp_path / "run",
        config_snapshot_path=config,
        checkpoint_path=checkpoint,
        trace_records=traces,
        experiment_contract=_contract(
            phase="decisive",
            arm_id="P05-M",
            dataset_id=1,
            seed=42,
        ),
    )
    assert [path.name for path, _ in calls] == ["val", "test"]
    assert calls[0][1]["trace_package"] == "/trace/val"
    assert calls[1][1]["expected_trace_semantic_sha256"] == "60" * 32
    assert set(observed) == {"val", "test"}
    assert all(
        record["scientific_status"] == "computed_unadjudicated"
        for record in observed.values()
    )


def test_tuning_checkpoint_is_revalidated_on_validation_only(tmp_path):
    checkpoint = tmp_path / "model-epoch=07-val_loss=0.2500.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    callback = ModelCheckpoint()
    callback.best_model_path = str(checkpoint)
    callback.best_model_score = torch.tensor(0.25, dtype=torch.float64)
    requests = []

    class _Trainer:
        callbacks = [callback]
        fit_loop = SimpleNamespace(
            epoch_progress=SimpleNamespace(
                current=SimpleNamespace(completed=12),
            )
        )

        def validate(self, task, loader, verbose):
            del task
            assert loader == ["val"]
            assert verbose is False
            return [
                {
                    "val_loss": torch.tensor(0.25, dtype=torch.float64),
                    "val_acc": torch.tensor(0.75, dtype=torch.float64),
                    "val_f1_macro": torch.tensor(0.625, dtype=torch.float64),
                }
            ]

    class _DataFactory:
        def get_dataloader(self, split):
            requests.append(split)
            if split == "test":
                raise AssertionError("tuning revalidation accessed test")
            return [split]

    observed = pipeline_module._validate_selected_p05_tuning_checkpoint(
        task=SimpleNamespace(network=_Network()),
        trainer=_Trainer(),
        data_factory=_DataFactory(),
        checkpoint_path=checkpoint,
        experiment_contract=_contract(),
    )

    assert requests == ["val"]
    assert observed == {
        "checkpoint_epoch": 7,
        "epochs_completed": 12,
        "val_acc": 0.75,
        "val_f1_macro": 0.625,
        "val_loss": 0.25,
    }


def test_tuning_candidate_route_binds_materialized_source_and_zero_test_access(
    tmp_path,
    monkeypatch,
):
    source_package = tmp_path / "materialized"
    source_package.mkdir()
    source_config = source_package / "config.yaml"
    source_config.write_text("paper_id: P05\n", encoding="utf-8")
    (source_package / "manifest.json").write_text("{}\n", encoding="utf-8")
    runtime_config = tmp_path / "run" / "config_snapshot.yaml"
    runtime_config.parent.mkdir()
    runtime_config.write_text("paper_id: P05\n", encoding="utf-8")
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    code_manifest = tmp_path / "code_manifest.json"
    code_manifest.write_text("{}\n", encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text("{}\n", encoding="utf-8")
    captured = {}

    def fake_export(package_dir, **kwargs):
        captured["package_dir"] = Path(package_dir)
        captured.update(kwargs)
        return SimpleNamespace(
            manifest_path=Path(package_dir) / "manifest.json",
            manifest_sha256="a1" * 32,
            semantic_sha256="b2" * 32,
            status="created",
        )

    monkeypatch.setattr(
        pipeline_module,
        "export_p05_tuning_validation_candidate",
        fake_export,
    )
    provenance = _complete_provenance()
    observed = pipeline_module._export_registered_p05_tuning_candidate(
        config_source_path=source_config,
        config_snapshot_path=runtime_config,
        code_snapshot=SimpleNamespace(manifest_path=code_manifest),
        run_contract_record={"manifest_path": str(run_manifest)},
        checkpoint_path=checkpoint,
        data_factory=SimpleNamespace(
            execution_stage="fit_validate_only",
            test_dataset=None,
            test_loader=None,
        ),
        execution_stage="fit_validate_only",
        experiment_contract=_contract(),
        tuning_validation_record={
            "checkpoint_epoch": 7,
            "epochs_completed": 12,
            "val_f1_macro": 0.625,
            "val_loss": 0.25,
        },
        attempt_provenance=provenance,
        run_path=tmp_path / "run",
    )

    assert captured["materialized_job_manifest_path"] == (
        source_package / "manifest.json"
    )
    assert captured["source_matrix_path"] == pipeline_module._P05_TUNING_MATRIX_PATH
    assert captured["data_roles_constructed"] == ["train", "validation"]
    assert captured["test_access_count"] == 0
    assert set(captured["provenance"]) == set(
        pipeline_module._P05_TUNING_CANDIDATE_PROVENANCE_FIELDS
    )
    assert observed["semantic_sha256"] == "b2" * 32
    assert observed["scientific_status"] == "computed_unadjudicated"


def test_tuning_candidate_route_rejects_constructed_test_and_is_tuning_only(
    tmp_path,
    monkeypatch,
):
    called = []
    monkeypatch.setattr(
        pipeline_module,
        "export_p05_tuning_validation_candidate",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )
    common = {
        "config_source_path": tmp_path / "config.yaml",
        "config_snapshot_path": tmp_path / "snapshot.yaml",
        "code_snapshot": SimpleNamespace(manifest_path=tmp_path / "code.json"),
        "run_contract_record": {"manifest_path": str(tmp_path / "run.json")},
        "checkpoint_path": tmp_path / "best.ckpt",
        "execution_stage": "fit_validate_only",
        "tuning_validation_record": {
            "checkpoint_epoch": 0,
            "epochs_completed": 1,
            "val_f1_macro": 0.5,
            "val_loss": 1.0,
        },
        "attempt_provenance": _complete_provenance(),
        "run_path": tmp_path / "run",
    }
    assert pipeline_module._export_registered_p05_tuning_candidate(
        **common,
        data_factory=SimpleNamespace(
            execution_stage="fit_validate_only",
            test_dataset=object(),
            test_loader=object(),
        ),
        experiment_contract=_contract(phase="pilot", arm_id="P05-M"),
    ) == {}
    assert called == []

    try:
        pipeline_module._export_registered_p05_tuning_candidate(
            **common,
            data_factory=SimpleNamespace(
                execution_stage="fit_validate_only",
                test_dataset=object(),
                test_loader=None,
            ),
            experiment_contract=_contract(),
        )
    except RuntimeError as exc:
        assert "constructed test data" in str(exc)
    else:
        raise AssertionError("constructed tuning test data was accepted")


def test_tuning_attempt_requires_candidate_output():
    outputs = pipeline_module._expected_p05_attempt_outputs(_contract())
    assert "tuning_candidate" in outputs


def test_pilot_evaluator_route_binds_full_validation_and_runtime_provenance(
    tmp_path,
    monkeypatch,
):
    count = 256
    record_ids = [f"record-{index:03d}" for index in range(count)]
    starts = torch.arange(count, dtype=torch.int64) * 4
    ends = starts + 4
    batch = {
        "x": torch.ones((count, 4, 2), dtype=torch.float32),
        "y": torch.arange(count, dtype=torch.int64).remainder(2),
        "sample_id": [
            f"{record_ids[index]}:{int(starts[index])}:{int(ends[index])}"
            for index in range(count)
        ],
        "record_id": record_ids,
        "group_id": [f"bearing-{index % 4}" for index in range(count)],
        "window_start": starts,
        "window_end": ends,
    }

    class _DataFactory:
        def get_dataloader(self, split):
            assert split == "val"
            return [batch]

    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "best.ckpt"
    config.write_text("paper_id: P05\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    captured = {}

    def fake_central(**kwargs):
        captured["central"] = kwargs
        return SimpleNamespace(semantic_sha256="91" * 32)

    def fake_d03(artifact_dir, **kwargs):
        captured["d03_dir"] = Path(artifact_dir)
        captured["d03"] = kwargs
        return SimpleNamespace(
            manifest_path=Path(artifact_dir) / "manifest.json",
            manifest_sha256="92" * 32,
            semantic_sha256="93" * 32,
            status="created",
        )

    def fake_summary(package_dir, **kwargs):
        captured["summary_dir"] = Path(package_dir)
        captured["summary"] = kwargs
        return SimpleNamespace(
            manifest_path=Path(package_dir) / "manifest.json",
            manifest_sha256="94" * 32,
            semantic_sha256="95" * 32,
            status="created",
        )

    monkeypatch.setattr(
        pipeline_module,
        "run_p05_pilot_interventions_from_loader",
        fake_central,
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_p05_d03_noise_interventions_from_loader",
        fake_d03,
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_p05_pilot_evaluator_benchmark",
        fake_summary,
    )
    provenance = _complete_provenance()
    observed = pipeline_module._export_registered_p05_pilot_evaluator_benchmark(
        task=SimpleNamespace(network=_Network()),
        data_factory=_DataFactory(),
        runtime_contract=SimpleNamespace(
            runtime_identity={
                "physical_gpu_index": 1,
                "gpu_uuid": "GPU-pilot-test",
            }
        ),
        run_path=tmp_path / "run",
        config_snapshot_path=config,
        checkpoint_path=checkpoint,
        run_contract_record={
            "code_semantic_sha256": "81" * 32,
            "semantic_sha256": "82" * 32,
        },
        attempt_provenance=provenance,
        expected_window_size=4,
        experiment_contract=_contract(
            phase="pilot",
            arm_id="P05-M",
            seed=20260801,
        ),
    )

    assert captured["central"]["expected_sample_ids"] == tuple(
        sorted(batch["sample_id"])
    )
    assert captured["central"]["require_cuda"] is True
    d03_provenance = captured["d03"]["provenance"]
    assert d03_provenance.physical_gpu_index == 1
    assert d03_provenance.device_uuid == "GPU-pilot-test"
    assert d03_provenance.cache_manifest_sha256 == provenance[
        "signal_cache_manifest_sha256"
    ]
    assert captured["d03"]["phase"] == "pilot_benchmark"
    assert captured["d03"]["chunk_size"] == 256
    assert observed["d03"]["semantic_sha256"] == "93" * 32
    assert observed["summary"]["semantic_sha256"] == "95" * 32
    outputs = pipeline_module._expected_p05_attempt_outputs(
        _contract(phase="pilot", arm_id="P05-M")
    )
    assert {"pilot_d03", "pilot_evaluator_benchmark"}.issubset(outputs)
