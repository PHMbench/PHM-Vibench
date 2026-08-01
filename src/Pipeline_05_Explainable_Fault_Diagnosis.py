import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.configs.config_utils import (
    load_config,
    merge_with_local_override,
    path_name,
    save_config,
    transfer_namespace,
)
from src.configs.p05_contract import validate_p05_experiment_contract
from src.explain_factory.eligibility import explain_ready, write_eligibility
from src.explain_factory.metadata_reader import (
    read_meta_from_batch,
    snapshot_metadata,
    write_metadata_snapshot,
)
from src.explain_factory.p05_evaluation_bundle import (
    P05EvaluationFrozenParameters,
    create_p05_c2_c3_evaluation_bundle,
)
from src.explain_factory.p05_intervention_runner import (
    P05InterventionProvenance,
    run_p05_pilot_interventions_from_loader,
    run_p05_same_checkpoint_interventions,
)
from src.explain_factory.p05_d03_noise_intervention import (
    P05D03Provenance,
    run_p05_d03_noise_interventions_from_loader,
)
from src.explain_factory.p05_pilot_evaluator_benchmark import (
    create_p05_pilot_evaluator_benchmark,
)
from src.explain_factory.p05_prediction_runner import (
    export_p05_window_predictions,
)
from src.explain_factory.p05_trace_runner import (
    export_p05_loader_trace,
    model_state_sha256,
    resolve_best_checkpoint_path,
    sha256_file,
)
from src.explain_factory.p05_trace_diagnostics import (
    create_p05_d01_d02_trace_diagnostics,
)
from src.utils.config_utils import parse_overrides, apply_overrides_to_config
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab, get_num_classes
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer
from src.trainer_factory.p05_runtime import prepare_p05_runtime
from src.utils.p05_attempt_record import begin_p05_attempt, finish_p05_attempt
from src.utils.p05_code_snapshot import export_p05_code_snapshot
from src.utils.p05_materialized_job_binding import (
    verify_p05_materialized_job_binding,
)
from src.utils.p05_run_artifacts import export_p05_run_artifact_bundle
from src.utils.p05_tuning_candidate import (
    export_p05_tuning_validation_candidate,
)


EVIDENCE_EXECUTION_STAGES = {"fit_validate_only", "fit_validate_test"}
_P05_ATTEMPT_PROVENANCE_FIELDS = (
    "source_metadata_sha256",
    "derived_metadata_sha256",
    "signal_cache_manifest_sha256",
    "split_manifest_sha256",
    "config_snapshot_sha256",
    "code_snapshot_sha256",
    "normalization_sha256",
    "train_weight_plan_sha256",
    "validation_weight_plan_sha256",
)
_P05_TUNING_CANDIDATE_PROVENANCE_FIELDS = (
    "source_metadata_sha256",
    "derived_metadata_sha256",
    "signal_cache_manifest_sha256",
    "split_manifest_sha256",
    "normalization_sha256",
    "train_weight_plan_sha256",
    "validation_weight_plan_sha256",
)
_P05_TUNING_MATRIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "p05"
    / "protocol"
    / "neural_tuning_matrix_p05_v1.yaml"
)


def _resolve_execution_stage(args_environment):
    stage = str(getattr(args_environment, "stage", "fit_validate_test"))
    if stage not in EVIDENCE_EXECUTION_STAGES:
        raise ValueError(
            "environment.stage must be one of "
            f"{sorted(EVIDENCE_EXECUTION_STAGES)}, got {stage!r}"
        )
    return stage


def _load_pipeline_config(args):
    """Load an exact P05 evidence config without implicit local mutation."""

    base = load_config(args.config_path)
    trainer = getattr(base, "trainer", None)
    p05_evidence_mode = getattr(trainer, "p05_evidence_mode", False) is True
    if not p05_evidence_mode:
        return merge_with_local_override(
            args.config_path,
            getattr(args, "local_config", None),
        )
    if getattr(args, "local_config", None) is not None:
        raise ValueError("P05 evidence mode forbids local config overrides")
    if getattr(args, "override", None):
        raise ValueError("P05 evidence mode forbids CLI config overrides")
    return base


def _validate_p05_process_contract(args_environment, runtime_contract):
    if runtime_contract is None:
        return
    iterations = getattr(args_environment, "iterations", None)
    if type(iterations) is not int or iterations != 1:
        raise ValueError(
            "P05 evidence mode requires environment.iterations=1 and one seed per process"
        )
    seed = getattr(args_environment, "seed", None)
    if type(seed) is not int or seed < 0:
        raise ValueError("P05 evidence mode requires a non-negative integer seed")


def _validate_p05_evaluation_contract(args_task, runtime_contract):
    """Bind trace export explicitly to the registered P05-M evidence arm."""

    trace_export = getattr(args_task, "p05_trace_export", False)
    if type(trace_export) is not bool:
        raise TypeError("task.p05_trace_export must be a literal boolean")
    if runtime_contract is None:
        if trace_export:
            raise ValueError("P05 trace export is available only in evidence mode")
        return False

    evidence_mode = getattr(args_task, "p05_evidence_mode", None)
    if evidence_mode is not True:
        raise ValueError(
            "trainer P05 evidence mode requires task.p05_evidence_mode=true"
        )
    arm_id = getattr(args_task, "p05_arm_id", None)
    if not isinstance(arm_id, str) or not arm_id.startswith("P05-"):
        raise ValueError("P05 evidence runs require a registered task.p05_arm_id")
    if arm_id == "P05-M" and not trace_export:
        raise ValueError("P05-M evidence runs require task.p05_trace_export=true")
    if arm_id != "P05-M" and trace_export:
        raise ValueError("complete fuzzy trace export is registered only for P05-M")
    return trace_export


def _required_p05_sha256(value, *, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _empty_p05_attempt_provenance():
    return {name: None for name in _P05_ATTEMPT_PROVENANCE_FIELDS}


def _read_p05_json_object(path, *, name):
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be a real non-symlink file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object: {source}")
    return payload


def _resolve_p05_static_provenance(
    *,
    args_data,
    experiment_contract,
    config_snapshot_path,
    code_snapshot_sha256,
):
    """Resolve registered source hashes without reading the raw workbook."""

    metadata_value = getattr(args_data, "metadata_path", None)
    cache_manifest_value = getattr(args_data, "cache_manifest_path", None)
    if not isinstance(metadata_value, str) or not metadata_value.strip():
        raise ValueError("P05 provenance requires data.metadata_path")
    if not isinstance(cache_manifest_value, str) or not cache_manifest_value.strip():
        raise ValueError("P05 provenance requires data.cache_manifest_path")

    metadata_path = Path(metadata_value)
    metadata_manifest_path = metadata_path.with_suffix(".manifest.json")
    manifest = _read_p05_json_object(
        metadata_manifest_path,
        name="P05 metadata manifest",
    )
    if manifest.get("paper_id") != "P05" or manifest.get("protocol_id") != "P05-G040-v3.2":
        raise ValueError("P05 metadata manifest is not bound to frozen P05-G040-v3.2")

    derived = manifest.get("derived_metadata")
    source = manifest.get("source_workbook")
    split_manifests = manifest.get("split_manifests")
    if not isinstance(derived, dict) or not isinstance(source, dict):
        raise ValueError("P05 metadata manifest is missing source/derived provenance")
    if not isinstance(split_manifests, dict):
        raise ValueError("P05 metadata manifest is missing split provenance")

    derived_file = derived.get("file")
    if not isinstance(derived_file, str) or not derived_file:
        raise ValueError("P05 metadata manifest has no derived metadata file")
    recorded_metadata_path = Path(derived_file)
    if not recorded_metadata_path.is_absolute():
        recorded_metadata_path = metadata_manifest_path.parent / recorded_metadata_path
    if recorded_metadata_path.resolve(strict=True) != metadata_path.resolve(strict=True):
        raise ValueError("configured P05 metadata differs from its registered manifest")
    recorded_csv_sha256 = _required_p05_sha256(
        derived.get("csv_sha256"),
        name="derived_metadata.csv_sha256",
    )
    if sha256_file(metadata_path) != recorded_csv_sha256:
        raise ValueError("P05 derived metadata file hash differs from its manifest")
    semantic = derived.get("semantic_serialization")
    if not isinstance(semantic, dict):
        raise ValueError("P05 metadata manifest lacks semantic serialization")

    split_name = {1: "CWRU", 2: "XJTU"}[experiment_contract.dataset_id]
    split_record = split_manifests.get(split_name)
    if not isinstance(split_record, dict):
        raise ValueError(f"P05 metadata manifest lacks the {split_name} split record")
    split_file = split_record.get("file")
    if not isinstance(split_file, str) or not split_file:
        raise ValueError(f"P05 {split_name} split manifest has no file")
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = metadata_manifest_path.parent / split_path
    recorded_split_sha256 = _required_p05_sha256(
        split_record.get("sha256"),
        name=f"split_manifests.{split_name}.sha256",
    )
    if sha256_file(split_path) != recorded_split_sha256:
        raise ValueError(f"P05 {split_name} split manifest hash mismatch")

    return {
        "source_metadata_sha256": _required_p05_sha256(
            source.get("sha256"),
            name="source_workbook.sha256",
        ),
        "derived_metadata_sha256": _required_p05_sha256(
            semantic.get("sha256"),
            name="derived_metadata.semantic_serialization.sha256",
        ),
        "signal_cache_manifest_sha256": sha256_file(cache_manifest_value),
        "split_manifest_sha256": recorded_split_sha256,
        "config_snapshot_sha256": sha256_file(config_snapshot_path),
        "code_snapshot_sha256": _required_p05_sha256(
            code_snapshot_sha256,
            name="code_snapshot_sha256",
        ),
    }


def _p05_package_versions():
    versions = {
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    try:
        versions["pytorch_lightning"] = importlib.metadata.version(
            "pytorch-lightning"
        )
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("P05 could not resolve the pytorch-lightning version") from exc
    if torch.version.cuda is not None:
        versions["torch_cuda"] = str(torch.version.cuda)
    return versions


def _p05_command_argv(args):
    requested = getattr(args, "requested_config", None) or args.config_path
    argv = [
        "conda",
        "run",
        "-n",
        "LQ_signal",
        "python",
        "main.py",
        "--config",
        str(requested),
    ]
    for override in getattr(args, "override", None) or []:
        argv.extend(["--override", str(override)])
    notes = getattr(args, "notes", "")
    if isinstance(notes, str) and notes:
        argv.extend(["--notes", notes])
    local_config = getattr(args, "local_config", None)
    if local_config is not None:
        argv.extend(["--local_config", str(local_config)])
    return argv


def _expected_p05_attempt_outputs(experiment_contract):
    outputs = {
        "all_results",
        "checkpoint",
        "code_snapshot",
        "config_snapshot",
        "materialized_job",
        "result",
        "run_contract",
    }
    if experiment_contract.phase == "pilot":
        outputs.add("pilot_timing")
        if experiment_contract.arm_id == "P05-M":
            outputs.update(
                {
                    "pilot_d03",
                    "pilot_evaluator_benchmark",
                    "trace_val",
                }
            )
    elif experiment_contract.phase == "tuning":
        outputs.add("tuning_candidate")
    elif experiment_contract.phase == "decisive":
        if experiment_contract.arm_id == "P05-M":
            outputs.update(
                {
                    "diagnostics_test",
                    "diagnostics_val",
                    "evaluation",
                    "trace_test",
                    "trace_val",
                }
            )
        else:
            outputs.add("predictions")
    return outputs


def _begin_pipeline_p05_attempt(
    args,
    *,
    run_path,
    experiment_contract,
    runtime_contract,
    provenance,
    started_at_utc,
):
    missing_reasons = {
        name: "not available when the attempt start record was committed"
        for name, value in provenance.items()
        if value is None
    }
    config_hash = provenance.get("config_snapshot_sha256") or "unresolved"
    attempt_id = (
        f"{experiment_contract.phase}.{experiment_contract.arm_id}."
        f"D{experiment_contract.dataset_id}.S{experiment_contract.seed}."
        f"{config_hash[:12]}"
    )
    package = Path(run_path) / "artifacts" / "p05" / "attempt"
    result = begin_p05_attempt(
        package,
        attempt_id=attempt_id,
        arm_id=experiment_contract.arm_id,
        phase=experiment_contract.phase,
        dataset_id=experiment_contract.dataset_id,
        seed=experiment_contract.seed,
        command_argv=_p05_command_argv(args),
        working_directory=Path(__file__).resolve().parents[1],
        package_versions=_p05_package_versions(),
        device_identity=runtime_contract.runtime_identity,
        provenance=provenance,
        unavailable_reasons=missing_reasons,
        started_at_utc=started_at_utc,
    )
    args._p05_active_attempt_package = str(result.package_dir)
    args._p05_attempt_outputs = {
        name: value
        for name, value in {
            "code_snapshot": provenance.get("code_snapshot_sha256"),
            "config_snapshot": provenance.get("config_snapshot_sha256"),
            "materialized_job": getattr(
                args,
                "_p05_materialized_binding_sha256",
                None,
            ),
        }.items()
        if value is not None
    }
    args._p05_expected_attempt_outputs = sorted(
        _expected_p05_attempt_outputs(experiment_contract)
    )
    return result


def _record_p05_attempt_output(args, name, sha256):
    package = getattr(args, "_p05_active_attempt_package", None)
    if package is None:
        return
    outputs = getattr(args, "_p05_attempt_outputs", None)
    if not isinstance(outputs, dict):
        raise RuntimeError("P05 active attempt has no output registry")
    value = _required_p05_sha256(sha256, name=f"attempt output {name}")
    existing = outputs.get(name)
    if existing is not None and existing != value:
        raise RuntimeError(f"P05 attempt output {name!r} changed after registration")
    outputs[name] = value


def _classify_p05_attempt_failure(exc):
    message = str(exc).lower()
    if isinstance(exc, FloatingPointError):
        return "scientific"
    if isinstance(exc, (FileNotFoundError, json.JSONDecodeError)):
        return "provenance"
    if isinstance(exc, (MemoryError, OSError, TimeoutError)) or any(
        token in message
        for token in ("cuda", "gpu", "nvidia-smi", "out of memory", "worker exited")
    ):
        return "infrastructure"
    if any(
        token in message
        for token in ("sha256", "hash mismatch", "metadata", "split manifest", "provenance")
    ):
        return "provenance"
    if "preflight" in message or "contract" in message:
        return "preflight"
    return "implementation"


def _finish_active_p05_attempt_failure(args, exc):
    package = getattr(args, "_p05_active_attempt_package", None)
    if package is None:
        return
    outputs = dict(getattr(args, "_p05_attempt_outputs", {}) or {})
    expected = set(getattr(args, "_p05_expected_attempt_outputs", []) or [])
    missing = {
        name: "attempt terminated before this required output was committed"
        for name in sorted(expected - set(outputs))
    }
    message = str(exc).replace("\x00", " ").strip() or type(exc).__name__
    finish_p05_attempt(
        package,
        status="failed",
        output_artifact_sha256=outputs,
        missing_outputs=missing,
        failure_category=_classify_p05_attempt_failure(exc),
        failure_type=type(exc).__name__[:256],
        failure_message=message[:4096],
    )
    args._p05_active_attempt_package = None


def _finish_active_p05_attempt_success(args):
    package = getattr(args, "_p05_active_attempt_package", None)
    if package is None:
        return
    outputs = dict(getattr(args, "_p05_attempt_outputs", {}) or {})
    expected = set(getattr(args, "_p05_expected_attempt_outputs", []) or [])
    missing = sorted(expected - set(outputs))
    if missing:
        raise RuntimeError(
            f"P05 attempt cannot complete with missing registered outputs: {missing}"
        )
    finish_p05_attempt(
        package,
        status="completed",
        output_artifact_sha256=outputs,
    )
    args._p05_active_attempt_package = None


def _finite_p05_metric(value, *, name, minimum=None, maximum=None):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        value = value.detach().to(device="cpu").item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _validate_selected_p05_tuning_checkpoint(
    *,
    task,
    trainer,
    data_factory,
    checkpoint_path,
    experiment_contract,
):
    """Re-evaluate the selected tuning checkpoint on validation only."""

    if experiment_contract.phase != "tuning":
        return {}
    callbacks = getattr(trainer, "callbacks", None)
    checkpoint_callbacks = [
        callback
        for callback in callbacks or []
        if isinstance(callback, ModelCheckpoint)
    ]
    if len(checkpoint_callbacks) != 1:
        raise RuntimeError("P05 tuning requires exactly one checkpoint callback")
    callback = checkpoint_callbacks[0]
    selected_path = Path(checkpoint_path).resolve(strict=True)
    if Path(callback.best_model_path).resolve(strict=True) != selected_path:
        raise RuntimeError("P05 tuning checkpoint differs from callback.best_model_path")
    match = re.search(r"(?:^|-)epoch=(\d+)(?:-|\.ckpt$)", selected_path.name)
    if match is None:
        raise RuntimeError("P05 tuning checkpoint filename does not bind its epoch")
    checkpoint_epoch = int(match.group(1))

    progress = getattr(getattr(trainer, "fit_loop", None), "epoch_progress", None)
    current = getattr(progress, "current", None)
    epochs_completed = getattr(current, "completed", None)
    if type(epochs_completed) is not int or not 1 <= epochs_completed <= 60:
        raise RuntimeError("P05 tuning completed-epoch count is unavailable or invalid")
    if checkpoint_epoch >= epochs_completed:
        raise RuntimeError("P05 tuning checkpoint epoch was not completed")

    network = getattr(task, "network", None)
    if network is None:
        raise RuntimeError("P05 tuning task has no checkpoint-loaded network")
    state_before = model_state_sha256(network)
    validation_results = trainer.validate(
        task,
        data_factory.get_dataloader("val"),
        verbose=False,
    )
    if not isinstance(validation_results, list) or len(validation_results) != 1:
        raise RuntimeError("P05 tuning validation must return exactly one result")
    metrics = validation_results[0]
    if not isinstance(metrics, Mapping):
        raise TypeError("P05 tuning validation result must be a mapping")
    required = {"val_loss", "val_acc", "val_f1_macro"}
    missing = sorted(required - set(metrics))
    if missing:
        raise KeyError(f"P05 tuning validation result is missing {missing}")
    val_loss = _finite_p05_metric(metrics["val_loss"], name="val_loss", minimum=0.0)
    val_acc = _finite_p05_metric(
        metrics["val_acc"],
        name="val_acc",
        minimum=0.0,
        maximum=1.0,
    )
    val_f1_macro = _finite_p05_metric(
        metrics["val_f1_macro"],
        name="val_f1_macro",
        minimum=0.0,
        maximum=1.0,
    )
    selected_loss = _finite_p05_metric(
        callback.best_model_score,
        name="checkpoint best_model_score",
        minimum=0.0,
    )
    if not math.isclose(
        val_loss,
        selected_loss,
        rel_tol=1.0e-7,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError(
            "P05 selected checkpoint validation loss does not reproduce its monitor"
        )
    if model_state_sha256(network) != state_before:
        raise RuntimeError("P05 tuning validation mutated the selected model state")
    return {
        "checkpoint_epoch": checkpoint_epoch,
        "epochs_completed": epochs_completed,
        "val_acc": val_acc,
        "val_f1_macro": val_f1_macro,
        "val_loss": val_loss,
    }


def _export_registered_p05_tuning_candidate(
    *,
    config_source_path,
    config_snapshot_path,
    code_snapshot,
    run_contract_record,
    checkpoint_path,
    data_factory,
    execution_stage,
    experiment_contract,
    tuning_validation_record,
    attempt_provenance,
    run_path,
):
    """Emit the exact validation-only candidate consumed by LR selection."""

    if experiment_contract.phase != "tuning":
        return {}
    if execution_stage != "fit_validate_only":
        raise RuntimeError("P05 tuning candidate export requires fit_validate_only")
    if getattr(data_factory, "execution_stage", None) != "fit_validate_only":
        raise RuntimeError("P05 tuning data factory is not validation-only")
    if (
        getattr(data_factory, "test_dataset", None) is not None
        or getattr(data_factory, "test_loader", None) is not None
    ):
        raise RuntimeError("P05 tuning constructed test data before candidate export")
    required_metrics = {
        "checkpoint_epoch",
        "epochs_completed",
        "val_f1_macro",
        "val_loss",
    }
    if not isinstance(tuning_validation_record, Mapping) or not required_metrics.issubset(
        tuning_validation_record
    ):
        raise RuntimeError("P05 tuning candidate lacks checkpoint validation metrics")
    if code_snapshot is None or not run_contract_record:
        raise RuntimeError("P05 tuning candidate lacks code or run-contract provenance")

    source_config = Path(config_source_path)
    if source_config.is_symlink():
        raise ValueError("P05 tuning source config must not be a symlink")
    source_config = source_config.resolve(strict=True)
    if source_config.name != "config.yaml":
        raise ValueError("P05 tuning must execute a materialized config.yaml")
    materialized_manifest = source_config.with_name("manifest.json")
    provenance = {
        name: attempt_provenance.get(name)
        for name in _P05_TUNING_CANDIDATE_PROVENANCE_FIELDS
    }
    result = export_p05_tuning_validation_candidate(
        Path(run_path) / "artifacts" / "p05" / "tuning_validation_candidate",
        materialized_job_manifest_path=materialized_manifest,
        source_matrix_path=_P05_TUNING_MATRIX_PATH,
        val_loss=tuning_validation_record["val_loss"],
        val_f1_macro=tuning_validation_record["val_f1_macro"],
        checkpoint_epoch=tuning_validation_record["checkpoint_epoch"],
        epochs_completed=tuning_validation_record["epochs_completed"],
        data_roles_constructed=["train", "validation"],
        test_access_count=0,
        config_snapshot_path=config_snapshot_path,
        code_snapshot_manifest_path=code_snapshot.manifest_path,
        run_contract_manifest_path=run_contract_record["manifest_path"],
        checkpoint_path=checkpoint_path,
        provenance=provenance,
    )
    return {
        "manifest_path": str(result.manifest_path),
        "manifest_sha256": result.manifest_sha256,
        "semantic_sha256": result.semantic_sha256,
        "scientific_status": "computed_unadjudicated",
        "status": result.status,
    }


def _p05_complete_loader_sample_ids(dataloader, *, expected_window_size):
    """Collect and validate only stable IDs for one complete loader pass."""

    required = {
        "x",
        "sample_id",
        "record_id",
        "window_start",
        "window_end",
    }
    observed = []
    seen = set()
    for batch_index, batch in enumerate(dataloader):
        if not isinstance(batch, Mapping):
            raise TypeError(f"P05 pilot ID batch {batch_index} must be a mapping")
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"P05 pilot ID batch {batch_index} is missing {missing}")
        x = batch["x"]
        if (
            not torch.is_tensor(x)
            or x.dtype != torch.float32
            or x.ndim != 3
            or tuple(x.shape[1:]) != (expected_window_size, 2)
        ):
            raise ValueError(f"P05 pilot ID batch {batch_index} has invalid x")
        count = int(x.shape[0])
        vectors = {}
        for name in ("sample_id", "record_id", "window_start", "window_end"):
            value = batch[name]
            if torch.is_tensor(value):
                value = value.detach().to(device="cpu").numpy()
            array = np.asarray(value)
            if array.shape != (count,):
                raise ValueError(
                    f"P05 pilot ID batch {batch_index}.{name} has invalid shape"
                )
            vectors[name] = array.tolist()
        for index in range(count):
            sample_id = vectors["sample_id"][index]
            record_id = vectors["record_id"][index]
            start = vectors["window_start"][index]
            end = vectors["window_end"][index]
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or not isinstance(record_id, str)
                or not record_id
                or isinstance(start, bool)
                or not isinstance(start, (int, np.integer))
                or isinstance(end, bool)
                or not isinstance(end, (int, np.integer))
            ):
                raise ValueError("P05 pilot stable-ID components are invalid")
            expected_id = f"{record_id}:{int(start)}:{int(end)}"
            if sample_id != expected_id or int(end) - int(start) != expected_window_size:
                raise ValueError("P05 pilot stable-ID/window binding is invalid")
            if sample_id in seen:
                raise ValueError(f"P05 pilot loader duplicated sample_id {sample_id!r}")
            seen.add(sample_id)
            observed.append(sample_id)
    if len(observed) < 256:
        raise ValueError("P05 pilot validation partition has fewer than 256 windows")
    return tuple(sorted(observed))


def _export_registered_p05_pilot_evaluator_benchmark(
    *,
    task,
    data_factory,
    runtime_contract,
    run_path,
    config_snapshot_path,
    checkpoint_path,
    run_contract_record,
    attempt_provenance,
    expected_window_size,
    experiment_contract,
):
    """Run the frozen non-evidence first-256 pilot evaluator benchmark."""

    if not (
        experiment_contract.phase == "pilot"
        and experiment_contract.arm_id == "P05-M"
    ):
        return {}
    network = getattr(task, "network", None)
    if network is None:
        raise RuntimeError("P05 pilot evaluator task has no checkpoint-loaded network")
    expected_ids = _p05_complete_loader_sample_ids(
        data_factory.get_dataloader("val"),
        expected_window_size=expected_window_size,
    )
    config_hash = sha256_file(config_snapshot_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    model_hash = model_state_sha256(network)
    central = run_p05_pilot_interventions_from_loader(
        network=network,
        batches=data_factory.get_dataloader("val"),
        provenance=P05InterventionProvenance(
            dataset=experiment_contract.dataset,
            split="validation",
            model_seed=experiment_contract.seed,
            config_sha256=config_hash,
            checkpoint_sha256=checkpoint_hash,
            model_sha256=model_hash,
        ),
        expected_sample_ids=expected_ids,
        expected_window_size=expected_window_size,
        require_cuda=True,
    )
    identity = runtime_contract.runtime_identity
    d03 = run_p05_d03_noise_interventions_from_loader(
        Path(run_path) / "artifacts" / "p05" / "pilot_d03",
        network=network,
        batches=data_factory.get_dataloader("val"),
        provenance=P05D03Provenance(
            dataset=experiment_contract.dataset,
            split="validation",
            model_seed=experiment_contract.seed,
            config_sha256=config_hash,
            code_sha256=run_contract_record["code_semantic_sha256"],
            checkpoint_sha256=checkpoint_hash,
            model_sha256=model_hash,
            run_contract_sha256=run_contract_record["semantic_sha256"],
            source_metadata_sha256=attempt_provenance["source_metadata_sha256"],
            derived_metadata_sha256=attempt_provenance["derived_metadata_sha256"],
            cache_manifest_sha256=attempt_provenance[
                "signal_cache_manifest_sha256"
            ],
            split_manifest_sha256=attempt_provenance["split_manifest_sha256"],
            normalization_sha256=attempt_provenance["normalization_sha256"],
            physical_gpu_index=identity["physical_gpu_index"],
            device_uuid=identity["gpu_uuid"],
        ),
        expected_sample_ids=expected_ids,
        phase="pilot_benchmark",
        budget_retained=None,
        expected_window_size=expected_window_size,
        require_cuda=True,
        chunk_size=256,
    )
    summary = create_p05_pilot_evaluator_benchmark(
        Path(run_path) / "artifacts" / "p05" / "pilot_evaluator_benchmark",
        central_result=central,
        d03_result=d03,
    )
    return {
        "central_semantic_sha256": central.semantic_sha256,
        "d03": {
            "manifest_path": str(d03.manifest_path),
            "manifest_sha256": d03.manifest_sha256,
            "semantic_sha256": d03.semantic_sha256,
            "status": d03.status,
        },
        "summary": {
            "manifest_path": str(summary.manifest_path),
            "manifest_sha256": summary.manifest_sha256,
            "semantic_sha256": summary.semantic_sha256,
            "scientific_status": "engineering_non_evidence",
            "status": summary.status,
        },
    }


def _export_registered_p05_traces(
    *,
    task,
    data_factory,
    run_path,
    config_snapshot_path,
    checkpoint_path,
    execution_stage,
    expected_window_size,
    experiment_contract,
):
    """Export traces and the actual-forward-bound unadjudicated evaluation."""

    network = getattr(task, "network", None)
    if network is None:
        raise RuntimeError("P05 task has no bound network for trace export")
    config_hash = sha256_file(config_snapshot_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    model_hash = model_state_sha256(network)
    partitions = ["val"]
    if execution_stage == "fit_validate_test":
        partitions.append("test")
    records = {}
    for partition in partitions:
        result = export_p05_loader_trace(
            Path(run_path) / "artifacts" / "p05" / "traces" / partition,
            network=network,
            dataloader=data_factory.get_dataloader(partition),
            config_sha256=config_hash,
            checkpoint_sha256=checkpoint_hash,
            model_sha256=model_hash,
            expected_window_size=expected_window_size,
            require_cuda=True,
        )
        records[partition] = {
            "package_dir": str(result.package_dir),
            "manifest_path": str(result.manifest_path),
            "manifest_sha256": result.manifest_sha256,
            "semantic_sha256": result.semantic_sha256,
            "status": result.status,
        }
    evaluation_record = {}
    if (
        execution_stage == "fit_validate_test"
        and experiment_contract.phase == "decisive"
        and experiment_contract.arm_id == "P05-M"
    ):
        required_batch_fields = {
            "x",
            "y",
            "sample_id",
            "record_id",
            "group_id",
            "window_start",
            "window_end",
        }
        actual_results = []
        for batch_index, batch in enumerate(data_factory.get_dataloader("test")):
            if not isinstance(batch, Mapping):
                raise TypeError(
                    f"P05 actual intervention batch {batch_index} must be a mapping"
                )
            missing = sorted(required_batch_fields - set(batch))
            if missing:
                raise KeyError(
                    f"P05 actual intervention batch {batch_index} is missing {missing}"
                )
            stable_batch = {name: batch[name] for name in required_batch_fields}
            actual_results.append(
                run_p05_same_checkpoint_interventions(
                    network=network,
                    batch=stable_batch,
                    provenance=P05InterventionProvenance(
                        dataset=experiment_contract.dataset,
                        split="test",
                        model_seed=experiment_contract.seed,
                        config_sha256=config_hash,
                        checkpoint_sha256=checkpoint_hash,
                        model_sha256=model_hash,
                    ),
                    expected_window_size=expected_window_size,
                    require_cuda=True,
                )
            )
        evaluation = create_p05_c2_c3_evaluation_bundle(
            Path(run_path) / "artifacts" / "p05" / "evaluation_bundle",
            validation_trace_package=records["val"]["package_dir"],
            evaluation_trace_package=records["test"]["package_dir"],
            actual_intervention_results=actual_results,
            frozen=P05EvaluationFrozenParameters(
                dataset=experiment_contract.dataset,
                model_seed=experiment_contract.seed,
                validation_trace_semantic_sha256=records["val"][
                    "semantic_sha256"
                ],
                evaluation_trace_semantic_sha256=records["test"][
                    "semantic_sha256"
                ],
            ),
        )
        evaluation_record = {
            "arrays_sha256": evaluation.arrays_sha256,
            "manifest_path": str(evaluation.manifest_path),
            "manifest_sha256": evaluation.manifest_sha256,
            "semantic_sha256": evaluation.semantic_sha256,
            "scientific_status": "computed_unadjudicated",
            "status": evaluation.status,
        }
    return records, evaluation_record


def _export_registered_p05_trace_diagnostics(
    *,
    task,
    run_path,
    config_snapshot_path,
    checkpoint_path,
    trace_records,
    experiment_contract,
):
    """Create mandatory D01/D02 artifacts for decisive P05-M trace splits."""

    if (
        experiment_contract.phase != "decisive"
        or experiment_contract.arm_id != "P05-M"
    ):
        return {}
    if set(trace_records) != {"val", "test"}:
        raise RuntimeError("decisive P05-M diagnostics require val and test traces")
    network = getattr(task, "network", None)
    if network is None:
        raise RuntimeError("P05 task has no bound network for D01/D02 diagnostics")
    config_hash = sha256_file(config_snapshot_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    model_hash = model_state_sha256(network)
    diagnostics = {}
    for partition in ("val", "test"):
        trace_record = trace_records[partition]
        result = create_p05_d01_d02_trace_diagnostics(
            Path(run_path)
            / "artifacts"
            / "p05"
            / "diagnostics"
            / "d01_d02"
            / partition,
            trace_package=trace_record["package_dir"],
            expected_trace_semantic_sha256=trace_record["semantic_sha256"],
            expected_config_sha256=config_hash,
            expected_checkpoint_sha256=checkpoint_hash,
            expected_model_sha256=model_hash,
        )
        diagnostics[partition] = {
            "arrays_path": str(result.arrays_path),
            "arrays_sha256": result.arrays_sha256,
            "manifest_path": str(result.manifest_path),
            "manifest_sha256": result.manifest_sha256,
            "semantic_sha256": result.semantic_sha256,
            "scientific_status": "computed_unadjudicated",
            "status": result.status,
        }
    return diagnostics


def _export_registered_p05_run_contract(
    *,
    task,
    data_factory,
    trainer,
    runtime_contract,
    run_path,
    config_snapshot_path,
    checkpoint_path,
    code_snapshot,
):
    """Create the immutable preprocessing/runtime/checkpoint provenance bundle."""

    trainer_identity = getattr(trainer, "p05_runtime_identity", None)
    expected_identity = getattr(runtime_contract, "runtime_identity", None)
    if not isinstance(trainer_identity, dict) or trainer_identity != expected_identity:
        raise RuntimeError(
            "P05 trainer runtime identity differs from the accepted preflight"
        )
    protocol = data_factory.get_protocol_artifacts()
    if not isinstance(protocol, dict):
        raise TypeError("P05 data factory returned invalid protocol artifacts")
    network = getattr(task, "network", None)
    if network is None:
        raise RuntimeError("P05 task has no checkpoint-loaded network")

    result = export_p05_run_artifact_bundle(
        Path(run_path) / "artifacts" / "p05" / "run_contract",
        normalization_plan=protocol["normalization_plan"],
        weight_plans={
            "train": protocol["weight_plans"]["train"],
            "val": protocol["weight_plans"]["val"],
        },
        runtime_identity=trainer_identity,
        config_sha256=sha256_file(config_snapshot_path),
        model_sha256=model_state_sha256(network),
        checkpoint_sha256=sha256_file(checkpoint_path),
        code_sha256=code_snapshot.semantic_sha256,
    )
    return {
        "code_manifest_path": str(code_snapshot.manifest_path),
        "code_manifest_sha256": code_snapshot.manifest_sha256,
        "code_semantic_sha256": code_snapshot.semantic_sha256,
        "manifest_path": str(result.manifest_path),
        "manifest_sha256": result.manifest_sha256,
        "semantic_sha256": result.semantic_sha256,
        "status": result.status,
    }


def _export_registered_p05_predictions(
    *,
    task,
    data_factory,
    run_path,
    config_snapshot_path,
    checkpoint_path,
    run_contract_record,
    expected_window_size,
    experiment_contract,
):
    """Export decisive non-fuzzy-arm predictions from the selected checkpoint."""

    if experiment_contract.phase != "decisive":
        return {}
    if experiment_contract.arm_id not in {"P05-B0", "P05-B1", "P05-B3"}:
        return {}
    network = getattr(task, "network", None)
    if network is None:
        raise RuntimeError("P05 task has no bound network for prediction export")
    split_result = getattr(data_factory, "split_result", None)
    if split_result is None:
        raise RuntimeError("P05 data factory has no registered split result")
    split_ids = {
        "train": [str(value) for value in split_result.train_ids],
        "val": [str(value) for value in split_result.val_ids],
        "test": [str(value) for value in split_result.test_ids],
    }
    result = export_p05_window_predictions(
        Path(run_path) / "artifacts" / "p05" / "predictions",
        network=network,
        split_dataloaders={
            split: data_factory.get_dataloader(split)
            for split in ("train", "val", "test")
        },
        expected_record_ids_by_split=split_ids,
        expected_windows_per_record=(
            16 if experiment_contract.dataset_id == 1 else 4
        ),
        config_sha256=sha256_file(config_snapshot_path),
        code_sha256=run_contract_record["code_semantic_sha256"],
        checkpoint_sha256=sha256_file(checkpoint_path),
        model_sha256=model_state_sha256(network),
        run_contract_sha256=run_contract_record["semantic_sha256"],
        expected_window_size=expected_window_size,
        require_cuda=True,
    )
    return {
        "arrays_path": str(result.arrays_path),
        "arrays_sha256": result.arrays_sha256,
        "manifest_path": str(result.manifest_path),
        "manifest_sha256": result.manifest_sha256,
        "semantic_sha256": result.semantic_sha256,
        "scientific_status": "computed_unadjudicated",
        "status": result.status,
    }



def _pipeline_impl(args):
    """领域泛化(Domain Generalization)任务的流水线（带 explain/report 产物写出）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        所有迭代的实验结果列表
    """
    # -----------------------
    # 1. 加载配置文件
    # -----------------------
    config_path = args.config_path
    print(f"[INFO] 加载配置文件: {config_path}")
    configs = _load_pipeline_config(args)

    # 应用CLI override参数（最高优先级）
    if hasattr(args, 'override') and args.override:
        print(f"[INFO] 应用CLI override参数: {args.override}")
        overrides = parse_overrides(args.override)
        configs = apply_overrides_to_config(configs, overrides)
        print(f"[INFO] 已应用 {len(overrides)} 个override参数")

    # 确保配置中包含必要的部分
    required_sections = ['data', 'model', 'task', 'trainer', 'environment']
    for section in required_sections:
        if not hasattr(configs, section):
            print(f"[ERROR] 配置文件中缺少 {section} 部分")
            return
    
    # 设置环境变量和命名空间
    args_environment = transfer_namespace(configs.environment if hasattr(configs, 'environment') else {})

    args_data = transfer_namespace(configs.data if hasattr(configs, 'data') else {})

    execution_stage = _resolve_execution_stage(args_environment)
    # The data factory owns signal-level isolation and therefore receives an
    # explicit copy of the environment stage instead of inferring it globally.
    args_data.execution_stage = execution_stage

    args_model = transfer_namespace(configs.model if hasattr(configs, 'model') else {})

    args_task = transfer_namespace(configs.task if hasattr(configs, 'task') else {})

    args_trainer = transfer_namespace(configs.trainer if hasattr(configs, 'trainer') else {})
    # Evidence mode must validate the physical single-GPU runtime before any
    # data, model, task, logger, or callback is constructed. Legacy mode is a
    # no-op and preserves the historical pipeline order.
    runtime_contract = prepare_p05_runtime(args_trainer)
    _validate_p05_process_contract(args_environment, runtime_contract)
    experiment_contract = validate_p05_experiment_contract(
        args_environment,
        args_data,
        args_model,
        args_task,
        args_trainer,
        runtime_contract,
    )
    materialized_job_binding = None
    if runtime_contract is not None:
        materialized_job_binding = verify_p05_materialized_job_binding(
            config_path=args.config_path,
            experiment_contract=experiment_contract,
            runtime_identity=runtime_contract.runtime_identity,
            cli_overrides=getattr(args, "override", None),
            local_config=getattr(args, "local_config", None),
        )
        args._p05_materialized_binding_sha256 = (
            materialized_job_binding.materialized_manifest_semantic_sha256
        )
    trace_export_enabled = (
        experiment_contract.trace_export if experiment_contract is not None else False
    )
    if args_task.name == 'Multitask':
        args_data.task_list = args_task.task_list
        args_model.task_list = args_task.task_list    
    for key, value in configs.environment.__dict__.items():
        if key.isupper():
            os.environ[key] = str(value)
            print(f"[INFO] 设置环境变量: {key}={value}")

    # 创建实验目录（依赖 environment.output_dir / path_name，不再强制依赖 VBENCH_* 变量）
    print("[INFO] 创建实验目录...")
    
    # -----------------------
    # 2. 多次迭代训练与测试
    # -----------------------
    all_results = []
    
    for it in range(args_environment.iterations):
        print(f"\n{'='*50}\n[INFO] 开始实验迭代 {it+1}/{args_environment.iterations}\n{'='*50}")
        
        # 设置路径和名称
        path, name = path_name(configs, it)
        Path(path).mkdir(parents=True, exist_ok=True)
        # 把name 加到args_trainer中
        args_trainer.logger_name = name

        # UXFD merge: always snapshot the fully-resolved config for auditability.
        config_snapshot_path = Path(path) / "config_snapshot.yaml"
        try:
            save_config(configs, config_snapshot_path)
        except Exception as e:
            if runtime_contract is not None:
                raise RuntimeError(
                    "P05 evidence run could not write its resolved config snapshot"
                ) from e
            print(f"[WARN] 保存 config_snapshot.yaml 失败: {e}")

        # 设置随机种子
        current_seed = args_environment.seed + it
        seed_everything(current_seed, workers=runtime_contract is not None)
        print(f"[INFO] 设置随机种子: {current_seed}")
        args._p05_lab_active = True
        init_lab(args_environment, args, name)

        code_snapshot = None
        if runtime_contract is not None:
            attempt_started_at = datetime.now(timezone.utc).isoformat(
                timespec="microseconds"
            )
            attempt_provenance = _empty_p05_attempt_provenance()
            try:
                attempt_provenance["config_snapshot_sha256"] = sha256_file(
                    config_snapshot_path
                )
                code_snapshot = export_p05_code_snapshot(
                    Path(path) / "artifacts" / "p05" / "code_snapshot",
                    source_root=Path(__file__).resolve().parents[1],
                )
                attempt_provenance[
                    "code_snapshot_sha256"
                ] = code_snapshot.semantic_sha256
                attempt_provenance.update(
                    _resolve_p05_static_provenance(
                        args_data=args_data,
                        experiment_contract=experiment_contract,
                        config_snapshot_path=config_snapshot_path,
                        code_snapshot_sha256=code_snapshot.semantic_sha256,
                    )
                )
            except Exception:
                _begin_pipeline_p05_attempt(
                    args,
                    run_path=path,
                    experiment_contract=experiment_contract,
                    runtime_contract=runtime_contract,
                    provenance=attempt_provenance,
                    started_at_utc=attempt_started_at,
                )
                raise

        # 构建数据工厂
        print("[INFO] 构建数据工厂...")
        try:
            data_factory = build_data(args_data, args_task)
        except Exception:
            if runtime_contract is not None:
                _begin_pipeline_p05_attempt(
                    args,
                    run_path=path,
                    experiment_contract=experiment_contract,
                    runtime_contract=runtime_contract,
                    provenance=attempt_provenance,
                    started_at_utc=attempt_started_at,
                )
            raise
        args._p05_active_data_factory = data_factory
        if runtime_contract is not None:
            try:
                protocol = data_factory.get_protocol_artifacts()
                normalization = protocol["normalization_plan"]
                weight_plans = protocol["weight_plans"]
                attempt_provenance.update(
                    {
                        "normalization_sha256": normalization.sha256,
                        "train_weight_plan_sha256": weight_plans["train"].sha256,
                        "validation_weight_plan_sha256": weight_plans["val"].sha256,
                    }
                )
            except Exception:
                _begin_pipeline_p05_attempt(
                    args,
                    run_path=path,
                    experiment_contract=experiment_contract,
                    runtime_contract=runtime_contract,
                    provenance=attempt_provenance,
                    started_at_utc=attempt_started_at,
                )
                raise
            _begin_pipeline_p05_attempt(
                args,
                run_path=path,
                experiment_contract=experiment_contract,
                runtime_contract=runtime_contract,
                provenance=attempt_provenance,
                started_at_utc=attempt_started_at,
            )
        # 构建模型
        print("[INFO] 构建模型...")
        model = build_model(args_model,metadata=data_factory.get_metadata())
        
        # 构建任务
        print("[INFO] 构建任务...")
        task = build_task(
            args_task=args_task,
            network=model,
            args_data=args_data,
            args_model=args_model,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=data_factory.get_metadata()
        )
        
        # 构建训练器
        print("[INFO] 构建训练器...")
        trainer = build_trainer(
            args_environment,
            args_trainer,
            args_data,
            path
        )

        # UXFD merge: always write a data metadata snapshot from the test dataloader (best-effort).
        artifacts_dir = Path(path) / "artifacts"
        meta_snapshot_path = artifacts_dir / "data_metadata_snapshot.json"
        batch_meta: dict = {}
        meta_source = "default"
        degraded = True
        try:
            snapshot_mode = (
                "val" if execution_stage == "fit_validate_only" else "test"
            )
            snapshot_loader = data_factory.get_dataloader(snapshot_mode)
            batch = next(iter(snapshot_loader))
            x0, y0, meta0, meta_source = read_meta_from_batch(batch)
            if isinstance(meta0, dict):
                batch_meta.update(meta0)
            # attach shapes for traceability (safe even if meta is empty)
            if hasattr(x0, "shape"):
                batch_meta.setdefault("x_shape", [int(v) for v in x0.shape])
            if hasattr(y0, "shape"):
                batch_meta.setdefault("y_shape", [int(v) for v in y0.shape])

            snapshot = snapshot_metadata(meta=batch_meta, meta_source=meta_source)
            degraded = snapshot.degraded
            write_metadata_snapshot(meta_snapshot_path, snapshot)
        except Exception as e:
            print(f"[WARN] 写入 data_metadata_snapshot.json 失败: {e}")
            try:
                snapshot = snapshot_metadata(meta={}, meta_source="default")
                write_metadata_snapshot(meta_snapshot_path, snapshot)
            except Exception:
                pass

        # UXFD merge: if explain enabled, write eligibility.json (never crash).
        try:
            extensions = getattr(args_trainer, "extensions", None)
            explain_cfg = getattr(extensions, "explain", None) if extensions is not None else None
            explain_enable = bool(getattr(explain_cfg, "enable", False)) if explain_cfg is not None else False
            if explain_enable:
                explainer_id = str(getattr(explain_cfg, "explainer", "") or "unknown")
                eligibility_path = artifacts_dir / "explain" / "eligibility.json"
                required_meta_keys = []
                if explainer_id in {"timefreq", "time_freq"}:
                    required_meta_keys = ["sampling_rate"]
                ready = explain_ready(
                    explainer_id=explainer_id,
                    meta=batch_meta,
                    required_meta_keys=required_meta_keys,
                    meta_source=str(meta_source),
                    degraded=bool(degraded),
                )
                write_eligibility(eligibility_path, ready)
        except Exception as e:
            print(f"[WARN] 写入 explain eligibility 失败: {e}")
        
        # 执行训练
        print("[INFO] 开始训练...")
        trainer.fit(
            task,
            data_factory.get_dataloader('train'),
            data_factory.get_dataloader('val')
        )
        pilot_timing = getattr(trainer, "p05_pilot_timing_result", None)
        if pilot_timing is not None:
            _record_p05_attempt_output(
                args,
                "pilot_timing",
                pilot_timing.semantic_sha256,
            )

        trace_records = {}
        diagnostic_records = {}
        evaluation_record = {}
        prediction_record = {}
        pilot_evaluator_record = {}
        tuning_validation_record = {}
        tuning_candidate_record = {}
        run_contract_record = {}
        if (
            execution_stage == "fit_validate_test"
            or trace_export_enabled
            or runtime_contract is not None
        ):
            checkpoint_path = resolve_best_checkpoint_path(trainer)
            task = load_best_model_checkpoint(task, trainer)
            if runtime_contract is not None:
                run_contract_record = _export_registered_p05_run_contract(
                    task=task,
                    data_factory=data_factory,
                    trainer=trainer,
                    runtime_contract=runtime_contract,
                    run_path=path,
                    config_snapshot_path=config_snapshot_path,
                    checkpoint_path=checkpoint_path,
                    code_snapshot=code_snapshot,
                )
                _record_p05_attempt_output(
                    args,
                    "checkpoint",
                    sha256_file(checkpoint_path),
                )
                _record_p05_attempt_output(
                    args,
                    "run_contract",
                    run_contract_record["semantic_sha256"],
                )
                tuning_validation_record = (
                    _validate_selected_p05_tuning_checkpoint(
                        task=task,
                        trainer=trainer,
                        data_factory=data_factory,
                        checkpoint_path=checkpoint_path,
                        experiment_contract=experiment_contract,
                    )
                )
                tuning_candidate_record = _export_registered_p05_tuning_candidate(
                    config_source_path=args.config_path,
                    config_snapshot_path=config_snapshot_path,
                    code_snapshot=code_snapshot,
                    run_contract_record=run_contract_record,
                    checkpoint_path=checkpoint_path,
                    data_factory=data_factory,
                    execution_stage=execution_stage,
                    experiment_contract=experiment_contract,
                    tuning_validation_record=tuning_validation_record,
                    attempt_provenance=attempt_provenance,
                    run_path=path,
                )
                if tuning_candidate_record:
                    _record_p05_attempt_output(
                        args,
                        "tuning_candidate",
                        tuning_candidate_record["semantic_sha256"],
                    )
            if trace_export_enabled:
                trace_records, evaluation_record = _export_registered_p05_traces(
                    task=task,
                    data_factory=data_factory,
                    run_path=path,
                    config_snapshot_path=config_snapshot_path,
                    checkpoint_path=checkpoint_path,
                    execution_stage=execution_stage,
                    expected_window_size=int(args_data.window_size),
                    experiment_contract=experiment_contract,
                )
                for partition, trace_record in trace_records.items():
                    _record_p05_attempt_output(
                        args,
                        f"trace_{partition}",
                        trace_record["semantic_sha256"],
                    )
                if evaluation_record:
                    _record_p05_attempt_output(
                        args,
                        "evaluation",
                        evaluation_record["semantic_sha256"],
                    )
                diagnostic_records = _export_registered_p05_trace_diagnostics(
                    task=task,
                    run_path=path,
                    config_snapshot_path=config_snapshot_path,
                    checkpoint_path=checkpoint_path,
                    trace_records=trace_records,
                    experiment_contract=experiment_contract,
                )
                for partition, diagnostic_record in diagnostic_records.items():
                    _record_p05_attempt_output(
                        args,
                        f"diagnostics_{partition}",
                        diagnostic_record["semantic_sha256"],
                    )
                pilot_evaluator_record = (
                    _export_registered_p05_pilot_evaluator_benchmark(
                        task=task,
                        data_factory=data_factory,
                        runtime_contract=runtime_contract,
                        run_path=path,
                        config_snapshot_path=config_snapshot_path,
                        checkpoint_path=checkpoint_path,
                        run_contract_record=run_contract_record,
                        attempt_provenance=attempt_provenance,
                        expected_window_size=int(args_data.window_size),
                        experiment_contract=experiment_contract,
                    )
                )
                if pilot_evaluator_record:
                    _record_p05_attempt_output(
                        args,
                        "pilot_d03",
                        pilot_evaluator_record["d03"]["semantic_sha256"],
                    )
                    _record_p05_attempt_output(
                        args,
                        "pilot_evaluator_benchmark",
                        pilot_evaluator_record["summary"]["semantic_sha256"],
                    )
            if runtime_contract is not None:
                prediction_record = _export_registered_p05_predictions(
                    task=task,
                    data_factory=data_factory,
                    run_path=path,
                    config_snapshot_path=config_snapshot_path,
                    checkpoint_path=checkpoint_path,
                    run_contract_record=run_contract_record,
                    expected_window_size=int(args_data.window_size),
                    experiment_contract=experiment_contract,
                )
                if prediction_record:
                    _record_p05_attempt_output(
                        args,
                        "predictions",
                        prediction_record["semantic_sha256"],
                    )
        
        if execution_stage == "fit_validate_only":
            # Pilot/tuning processes terminate after fit+validation.  Deliberately
            # do not load a test loader, checkpoint for test, emit test_result, or
            # invoke any test-end reporter.
            stage_result = {
                "stage": execution_stage,
                "status": "fit_validate_complete",
                "seed": current_seed,
            }
            if "val" in trace_records:
                stage_result["trace_validation_semantic_sha256"] = trace_records[
                    "val"
                ]["semantic_sha256"]
            if run_contract_record:
                stage_result["run_contract_semantic_sha256"] = run_contract_record[
                    "semantic_sha256"
                ]
            if materialized_job_binding is not None:
                stage_result["materialized_job_id"] = materialized_job_binding.job_id
                stage_result["materialized_job_semantic_sha256"] = (
                    materialized_job_binding.materialized_manifest_semantic_sha256
                )
            if tuning_validation_record:
                stage_result.update(tuning_validation_record)
            if tuning_candidate_record:
                stage_result["tuning_candidate_semantic_sha256"] = (
                    tuning_candidate_record["semantic_sha256"]
                )
            if pilot_evaluator_record:
                stage_result["pilot_d03_semantic_sha256"] = (
                    pilot_evaluator_record["d03"]["semantic_sha256"]
                )
                stage_result["pilot_evaluator_benchmark_semantic_sha256"] = (
                    pilot_evaluator_record["summary"]["semantic_sha256"]
                )
            all_results.append(stage_result)
            result_path = Path(path) / f"fit_validate_result_{it}.csv"
            pd.DataFrame([stage_result]).to_csv(
                result_path,
                index=False,
            )
            _record_p05_attempt_output(args, "result", sha256_file(result_path))
        else:
            # The validation-selected checkpoint was loaded once above.
            print("[INFO] 使用已加载的最佳模型进行测试...")
            result = trainer.test(task, data_factory.get_dataloader('test'))
            result_record = dict(result[0])
            for partition, trace_record in trace_records.items():
                result_record[f"trace_{partition}_semantic_sha256"] = trace_record[
                    "semantic_sha256"
                ]
            for partition, diagnostic_record in diagnostic_records.items():
                result_record[
                    f"p05_d01_d02_{partition}_semantic_sha256"
                ] = diagnostic_record["semantic_sha256"]
            if run_contract_record:
                result_record["run_contract_semantic_sha256"] = run_contract_record[
                    "semantic_sha256"
                ]
            if materialized_job_binding is not None:
                result_record["materialized_job_id"] = materialized_job_binding.job_id
                result_record["materialized_job_semantic_sha256"] = (
                    materialized_job_binding.materialized_manifest_semantic_sha256
                )
            if evaluation_record:
                result_record["p05_evaluation_semantic_sha256"] = evaluation_record[
                    "semantic_sha256"
                ]
            if prediction_record:
                result_record["p05_prediction_semantic_sha256"] = prediction_record[
                    "semantic_sha256"
                ]
            all_results.append(result_record)

            # 保存结果
            print("[INFO] 保存测试结果...")
            result_df = pd.DataFrame([result_record])
            result_path = Path(path) / f"test_result_{it}.csv"
            result_df.to_csv(result_path, index=False)
            _record_p05_attempt_output(args, "result", sha256_file(result_path))

            # UXFD merge: rewrite manifest after test_result exists (callback runs earlier).
            try:
                from src.trainer_factory.extensions import ManifestWriterCallback

                # Determine main process similar to Default_trainer.py logic.
                is_main_process = True
                if "LOCAL_RANK" in os.environ:
                    is_main_process = int(os.environ["LOCAL_RANK"]) == 0

                extensions = getattr(args_trainer, "extensions", None)
                report_cfg = getattr(extensions, "report", None) if extensions is not None else None
                report_enable = getattr(report_cfg, "enable", True) if report_cfg is not None else True
                manifest_enable = getattr(report_cfg, "manifest", True) if report_cfg is not None else True
                enabled = bool(report_enable) and bool(manifest_enable)

                ManifestWriterCallback(
                    run_dir=path,
                    paper_id=str(getattr(args_trainer, "paper_id", "") or ""),
                    preset_version=str(getattr(args_trainer, "preset_version", "") or ""),
                    run_id=str(getattr(args_trainer, "logger_name", "") or ""),
                    enabled=enabled,
                    is_main_process=is_main_process,
                ).on_test_end(trainer, task)
            except Exception as e:
                print(f"[WARN] 更新 artifacts/manifest.json 失败: {e}")

        data_factory.data.close()  # 关闭数据工厂，释放资源
        args._p05_active_data_factory = None

        # 关闭wandb和swanlab
        close_lab()
        args._p05_lab_active = False

    print(f"\n{'='*50}\n[INFO] 所有实验已完成\n{'='*50}")
    all_results_path = Path(path) / "all_results.csv"
    pd.DataFrame(all_results).to_csv(all_results_path, index=False)
    _record_p05_attempt_output(args, "all_results", sha256_file(all_results_path))
    _finish_active_p05_attempt_success(args)
    return all_results


def pipeline(args):
    """Run P05 while retaining terminal records and closing active resources."""

    args._p05_active_attempt_package = None
    args._p05_attempt_outputs = {}
    args._p05_expected_attempt_outputs = []
    args._p05_active_data_factory = None
    args._p05_lab_active = False
    args._p05_materialized_binding_sha256 = None
    try:
        return _pipeline_impl(args)
    except Exception as exc:
        try:
            _finish_active_p05_attempt_failure(args, exc)
        except Exception as record_exc:
            raise RuntimeError(
                "P05 failed and its immutable terminal record could not be committed"
            ) from record_exc
        raise
    finally:
        active_data_factory = getattr(args, "_p05_active_data_factory", None)
        if active_data_factory is not None:
            try:
                active_data_factory.data.close()
            except Exception as cleanup_exc:
                print(f"[WARN] P05 data cleanup after failure did not complete: {cleanup_exc}")
            args._p05_active_data_factory = None
        if getattr(args, "_p05_lab_active", False):
            try:
                close_lab()
            except Exception as cleanup_exc:
                print(f"[WARN] P05 logger cleanup after failure did not complete: {cleanup_exc}")
            args._p05_lab_active = False


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="领域泛化(DG)任务流水线")
    
    parser.add_argument('--config_path', 
                        type=str, 
                        default='/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/Single_DG/CWRU.yaml',
                        help='配置文件路径')
    parser.add_argument('--notes', 
                        type=str, 
                        default='',
                        help='实验备注')
    parser.add_argument('--local_config',
                        type=str,
                        default=None,
                        help='本机覆盖配置路径（可选）')

    
    args = parser.parse_args()
    
    # 执行DG流水线
    results = pipeline(args)
    print(f"完成所有实验！")
