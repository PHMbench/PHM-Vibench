"""P19 source training and frozen exports through the native PHMFactory spine.

Readers, channel selection, windowing, normalization, tasks, optimizers, trainers
and checkpoint selection remain owned by their existing factories. Source phases
require a separate source-only metadata population because the Data Factory reads
its selected records before it resolves dataset partitions.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from copy import copy, deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .features import ProgramSpec, program_features


SOURCE_ROLES = ("fit", "tune", "grow", "select")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _device(native_config: Mapping[str, Any], requested: str | None = None) -> str:
    from phmfactory.device import resolve_device_request
    from src.configs.config_utils import dict_to_namespace

    trainer = native_config["trainer"]
    if trainer.get("device") not in {"cpu", "cuda"} or trainer.get("devices") != 1:
        raise ValueError("P19 requires explicit trainer.device=cpu/cuda and trainer.devices=1")
    expected = "cpu" if trainer["device"] == "cpu" else "cuda:0"
    if requested is not None and requested != expected:
        raise ValueError(f"Requested export device {requested!r} differs from native device {expected!r}")
    if native_config["model"].get("device") not in {trainer["device"], expected}:
        raise ValueError("Native model.device must agree with the explicit trainer device")
    if expected == "cuda:0":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.startswith("GPU-") or "," in visible:
            raise RuntimeError("P19 requires CUDA_VISIBLE_DEVICES equal to the physical GPU 0 UUID; numeric indices are ambiguous")
        try:
            physical_uuid = subprocess.check_output(
                ["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"],
                text=True).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("Cannot verify physical GPU 0 UUID; CUDA initialization is prohibited") from error
        if not physical_uuid.startswith("GPU-") or "\n" in physical_uuid or visible != physical_uuid:
            raise RuntimeError("CUDA_VISIBLE_DEVICES differs from the verified physical GPU 0 UUID; CUDA initialization is prohibited")
        configured_visible = native_config["environment"].get("CUDA_VISIBLE_DEVICES", visible)
        if configured_visible != physical_uuid:
            raise RuntimeError("environment.CUDA_VISIBLE_DEVICES would replace the verified physical GPU 0 UUID")
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Requested physical GPU 0 is unavailable; no device fallback")
    resolve_device_request(dict_to_namespace(trainer))
    return expected


def _population(native_config: dict[str, Any], *, source_only: bool) -> tuple[Any, dict[str, Any]]:
    """Reject forbidden populations before the native factory can read windows."""
    from phmfactory.config import validate_complete_experiment
    from src.configs.config_utils import dict_to_namespace
    from src.data_factory.data_utils import MetadataAccessor, read_metadata_table
    from src.data_factory.ID.Id_searcher import search_ids_for_task, search_target_dataset_metadata
    from src.data_factory.splitting import resolve_data_splits

    validate_complete_experiment(native_config)
    task = native_config["task"]
    if (task.get("type"), task.get("name")) != ("Default_task", "Default_task"):
        raise ValueError("P19 source-role binding uses native Default_task/Default_task; validate DG before binding")
    if task.get("loss") != "CE":
        raise ValueError("P19 requires the native multiclass CE objective")
    if native_config["environment"].get("iterations") != 1:
        raise ValueError("One native invocation must have one explicitly tracked training seed")
    if any(native_config["environment"].get(service, False) for service in ("wandb", "swanlab")):
        raise ValueError("P19 local execution requires disabled external experiment logging")
    if native_config["trainer"].get("test_after_fit") is not False:
        raise ValueError("P19 source training must explicitly disable trainer.test_after_fit")
    data = native_config["data"]
    if data.get("factory_name", "default") != "default":
        raise ValueError("P19 requires the maintained default Data Factory")
    split = data.get("split", {})
    expected_map = {"train": "fit", "val": "tune", "test": "grow" if source_only else "test"}
    if (split.get("strategy") != "grouped_metadata" or split.get("test_policy") != "partition"
            or split.get("manifest_mode") != "read_only" or split.get("partition_map") != expected_map):
        raise ValueError(f"P19 requires read-only native role binding {expected_map}")
    manifest = json.loads(Path(split["manifest_path"]).read_text(encoding="utf-8"))
    required_roles = set(SOURCE_ROLES) if source_only else {*SOURCE_ROLES, "test"}
    if set(manifest.get("partitions", {})) != required_roles:
        raise ValueError(f"P19 {'source-only' if source_only else 'evaluation'} manifest must contain exactly {sorted(required_roles)}")
    frame = read_metadata_table(Path(data["data_dir"]) / data["metadata_file"],
                                encoding=data.get("metadata_encoding"))
    if set(frame["Name"]) != {"Synthetic"}:
        cache_dir = data.get("cache_dir")
        source_root = Path(data["data_dir"]).expanduser().resolve()
        if not isinstance(cache_dir, str) or not cache_dir.strip():
            raise ValueError("Real P19 data require explicit data.cache_dir outside the immutable source root")
        cache_root = Path(cache_dir).expanduser().resolve()
        if cache_root == source_root or cache_root.is_relative_to(source_root):
            raise ValueError("Real P19 data.cache_dir must be outside the immutable source root")
    metadata = MetadataAccessor(frame)
    declared_ids = {str(record_id) for role in required_roles for record_id in manifest["partitions"][role]["ids"]}
    if {str(record_id) for record_id in metadata.keys()} != declared_ids:
        raise ValueError("P19 metadata must contain exactly the manifest population; target records cannot enter source-only metadata")
    args = dict_to_namespace(native_config)
    selected = search_target_dataset_metadata(metadata, args.task)
    if {str(record_id) for record_id in selected.keys()} != declared_ids:
        raise ValueError("task.target_system_id must select the entire declared P19 population")
    source_ids, task_test_ids = search_ids_for_task(selected, args.task)
    resolved = resolve_data_splits(selected, args.data, args.task, source_ids, task_test_ids)
    for mode, role in expected_map.items():
        if [str(value) for value in getattr(resolved, f"{mode}_ids")] != [str(value) for value in manifest["partitions"][role]["ids"]]:
            raise ValueError(f"Native resolved {mode} IDs differ from frozen {role} IDs")
    return selected, manifest


def _coordinates(dataset: Any, indices: list[int], unit_by_record: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        position = dataset.file_windows_list[index]
        file_id, window_id = position["file_id"], int(position["window_id"])
        start, end = dataset.dataset_dict[file_id].window_intervals[window_id]
        meta = dataset.metadata[file_id]
        rows.append(dict(item=f"{meta['Name']}:{file_id}:{start}:{end}", record_id=str(file_id),
                         parent_record_id=str(meta["File"]), unit=unit_by_record[str(file_id)],
                         window_start=int(start), window_end=int(end)))
    return rows


def _role_loader(factory: Any, config: Any, record_ids: list[Any]) -> Any:
    """Use native adapters and samplers for exactly one declared source role."""
    from torch.utils.data import DataLoader
    from src.data_factory.dataset_task.adapters import resolve_dataset_adapter
    from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
    from src.data_factory.samplers.Get_sampler import Get_sampler

    dataset_class = resolve_dataset_adapter(config.task.type, config.task.name)
    datasets = {record_id: dataset_class({record_id: factory.data[record_id]}, factory.get_metadata(),
                                        config.data, config.task, "test") for record_id in record_ids}
    dataset = IdIncludedDataset(datasets, factory.get_metadata())
    evaluation_data = deepcopy(config.data)
    evaluation_data.batch_size = getattr(config.data, "eval_batch_size", config.data.batch_size)
    if type(evaluation_data.batch_size) is not int or evaluation_data.batch_size <= 0:
        raise ValueError("data.eval_batch_size must be an explicit positive integer")
    sampler = Get_sampler(config.task, evaluation_data, dataset, "test")
    return DataLoader(dataset, batch_sampler=sampler, num_workers=config.data.num_workers)


def _collect(task: Any, loader: Any, device: str, units: Mapping[str, str],
             program: ProgramSpec | None = None) -> dict[str, np.ndarray]:
    import torch
    from src.task_factory.Components.loss import prepare_loss_inputs

    task.to(device).eval().requires_grad_(False)
    before = {name: value.detach().cpu().clone() for name, value in task.network.state_dict().items()}
    captured: list[np.ndarray] = []
    extractor = getattr(task.network, "feature_extractor_layers", None)
    handle = extractor.register_forward_hook(
        lambda _module, _inputs, value: captured.append(value.detach().cpu().numpy().copy())) if extractor is not None else None
    rows: list[dict[str, Any]] = []
    logits, labels, native_values, signal_values = [], [], [], []
    names: list[str] | None = None
    try:
        if loader.batch_sampler.shuffle or loader.batch_sampler.drop_last:
            raise ValueError("Frozen exports require a deterministic native sampler retaining every window")
        with torch.inference_mode():
            for indices, batch in zip(loader.batch_sampler, loader, strict=True):
                coordinates = _coordinates(loader.dataset, indices, units)
                batch_ids = batch["file_id"].tolist() if hasattr(batch["file_id"], "tolist") else batch["file_id"]
                if [str(value) for value in batch_ids] != [row["record_id"] for row in coordinates]:
                    raise ValueError("Native batch record order differs from window coordinates")
                captured.clear()
                native_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                                for key, value in batch.items()}
                predicted = task(native_batch)
                _, target = prepare_loss_inputs(task.loss_name, predicted, native_batch["y"])
                if extractor is not None and (len(captured) != 1 or captured[0].ndim != 2 or len(captured[0]) != len(coordinates)):
                    raise ValueError("Native feature extractor must emit one [N,F] tensor in the same forward pass")
                features = captured[0] if extractor is not None else np.empty((len(coordinates), 0))
                if not np.isfinite(features).all():
                    raise ValueError("TSPN native features contain nonfinite values")
                if program is not None:
                    values, current_names = program_features(batch["x"].detach().cpu().numpy(), program)
                    if names is not None and list(current_names) != names:
                        raise ValueError("Signal feature coordinates changed between native batches")
                    names = list(current_names)
                    signal_values.append(values)
                rows.extend(coordinates)
                logits.append(predicted.detach().cpu().numpy())
                labels.append(target.detach().cpu().numpy())
                native_values.append(features)
        if not rows:
            raise ValueError("Declared export role produced no native windows")
        for name, value in task.network.state_dict().items():
            if not torch.equal(before[name], value.detach().cpu()):
                raise RuntimeError(f"Frozen TSPN parameter or running buffer changed during export: {name}")
        if any(parameter.requires_grad for parameter in task.parameters()):
            raise RuntimeError("Frozen native task retained trainable parameters")
    finally:
        if handle is not None:
            handle.remove()
    exported = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    exported.update(logits=np.concatenate(logits), y=np.concatenate(labels), native_features=np.concatenate(native_values))
    if program is not None:
        exported.update(program_features=np.concatenate(signal_values), feature_names=np.asarray(names))
    if len(np.unique(exported["item"])) != len(rows):
        raise ValueError("Duplicate native window identity in export")
    return exported


def _units(metadata: Any, manifest: Mapping[str, Any], role: str) -> dict[str, str]:
    return {str(record_id): str(metadata[record_id][manifest["group_key"]])
            for record_id in manifest["partitions"][role]["ids"]}


def _fit_reference_loader(loader: Any) -> Any:
    """Retain already materialized fit windows and native batch coordinates."""
    from torch.utils.data import DataLoader

    sampler = copy(loader.batch_sampler)
    if sampler.drop_last:
        raise ValueError("Native source training must retain every fit window before fit-reference export")
    sampler.shuffle = False
    sampler.drop_last = False
    return DataLoader(loader.dataset, batch_sampler=sampler, num_workers=loader.num_workers,
                      collate_fn=loader.collate_fn, pin_memory=loader.pin_memory)


def train_native(native_config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Fit on source fit and save fit/tune references from the selected checkpoint."""
    config = deepcopy(native_config)
    device = _device(config)
    from phmfactory.config import analyze_config
    from phmfactory.runtime import CompiledRunSpec
    from src.runtime.classification import ClassificationHooks, run_classification_pipeline
    from scipy.special import softmax
    from .metrics import aggregate_metrics
    import yaml

    metadata, manifest = _population(config, source_only=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config["environment"]["output_dir"] = str(output / "native_runs")
    config_path = output / "native_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    analysis = analyze_config(config_path)
    compiled = CompiledRunSpec.compile(analysis.to_resolved_config())
    effective_path = output / "effective_config.json"
    _write_json(effective_path, analysis.runtime_config())

    class Capture(ClassificationHooks):
        context: Any = None

        def after_stack_built(self, context: Any) -> None:
            self.context = context
            if "eval_batch_size" in config["data"]:
                context.data_factory.val_loader = _role_loader(
                    context.data_factory, context.configs, manifest["partitions"]["tune"]["ids"])

    capture = Capture()
    args = Namespace(config_path=str(config_path), compiled_run_spec=compiled,
                     resolved_pipeline=compiled.pipeline, notes="")
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr, redirect_stdout(stdout), redirect_stderr(stderr):
        result = run_classification_pipeline(args, hooks=capture)
        if result.get("test_metrics") is not None:
            raise RuntimeError("Source native training unexpectedly scored a test population")
        context = capture.context
        if context is None:
            raise RuntimeError("Native training did not build its task stack")
        reference = _collect(context.task, context.data_factory.get_dataloader("val"), device,
                             _units(metadata, manifest, "tune"))
        fit_reference = _collect(context.task,
                                 _fit_reference_loader(context.data_factory.get_dataloader("train")),
                                 device, _units(metadata, manifest, "fit"))
    reference_path = output / "tune_reference.npz"
    np.savez_compressed(reference_path, **reference)
    fit_reference_path = output / "fit_reference.npz"
    np.savez_compressed(fit_reference_path, **fit_reference)
    fit_metrics = aggregate_metrics(softmax(fit_reference["logits"], axis=1),
                                    fit_reference["y"], fit_reference["unit"])
    fit_metrics_path = output / "fit_metrics.json"
    _write_json(fit_metrics_path, fit_metrics)
    record = dict(checkpoint=result["best_checkpoint"], effective_config=str(effective_path),
                  config=str(config_path), native_config=analysis.runtime_config(),
                  tune_reference=str(reference_path), fit_reference=str(fit_reference_path),
                  fit_metrics=fit_metrics, fit_metrics_path=str(fit_metrics_path),
                  fit_accuracy=fit_metrics["accuracy"],
                  reference_scope="best validation-selected checkpoint restored by the native runtime",
                  device=device, physical_gpu=0 if device == "cuda:0" else None,
                  physical_gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES") if device == "cuda:0" else None,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  stdout=str(output / "stdout.log"),
                  stderr=str(output / "stderr.log"), result=result)
    _write_json(output / "training.json", record)
    return record


def _export(native_config: dict[str, Any], checkpoint: Path, roles: Mapping[str, Any],
            program_spec: Mapping[str, Any], output: Path, device: str, *,
            source_only: bool, reference: Path | None = None) -> dict[str, Any]:
    _device(native_config, device)
    from pytorch_lightning import seed_everything
    from pytorch_lightning.trainer.connectors.accelerator_connector import _set_torch_flags
    from src.configs.config_utils import dict_to_namespace
    from src.data_factory import build_data
    from src.model_factory import build_model
    from src.task_factory import build_task
    from src.runtime.classification import _close_data_factory

    metadata, manifest = _population(native_config, source_only=source_only)
    allowed = set(SOURCE_ROLES) if source_only else {"test"}
    if not roles or not set(roles) <= allowed:
        raise ValueError(f"{'Source' if source_only else 'Target'} export permits only roles {sorted(allowed)}")
    program = ProgramSpec.from_dict(dict(program_spec["program"]))
    class_names = list(program_spec["class_names"])
    if (not class_names or len(set(class_names)) != len(class_names)
            or any(not isinstance(value, str) or not value for value in class_names)
            or native_config["model"].get("num_classes") != len(class_names)):
        raise ValueError("Explicit class_names must match native model.num_classes")
    if set(metadata.df["Label"]) != set(range(len(class_names))):
        raise ValueError("Native metadata labels differ from the frozen class coordinates")
    if "Sample_rate" not in metadata.df or not np.all(metadata.df["Sample_rate"].to_numpy() == program.sample_rate_hz):
        raise ValueError("Native metadata sampling rates differ from the frozen signal program")
    for role, requested in roles.items():
        expected_ids = manifest["partitions"][role]["ids"]
        if [str(value) for value in requested["record_ids"]] != [str(value) for value in expected_ids]:
            raise ValueError(f"Requested {role} record IDs differ from frozen native manifest")
        supplied_units = {str(key): str(value) for key, value in requested["unit_by_record"].items()}
        if supplied_units != _units(metadata, manifest, role):
            raise ValueError(f"Requested {role} physical identities differ from native metadata")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = dict_to_namespace(deepcopy(native_config))
    config.model.weights_path, config.model.weights_strict = str(checkpoint), True
    seed_everything(int(config.environment.seed))
    _set_torch_flags(deterministic=getattr(config.trainer, "deterministic", None), benchmark=None)
    factory = build_data(config.data, config.task)
    try:
        model = build_model(config.model, metadata=factory.get_metadata())
        task = build_task(args_task=config.task, network=model, args_data=config.data,
                          args_model=config.model, args_trainer=config.trainer,
                          args_environment=config.environment, metadata=factory.get_metadata())
        exported = {}
        for role in roles:
            loader = _role_loader(factory, config, manifest["partitions"][role]["ids"])
            values = _collect(task, loader, device, _units(metadata, manifest, role), program)
            if values["logits"].shape[1] != len(class_names):
                raise ValueError("Native class-logit width differs from frozen class_names")
            values.update(role=np.full(len(values["y"]), role), split=np.full(len(values["y"]), role),
                          class_names=np.asarray(class_names), training_seed=np.asarray(config.environment.seed))
            exported[role] = values
    finally:
        _close_data_factory(factory)
    reference_error = None
    if reference is not None:
        if "tune" not in exported:
            raise ValueError("Native checkpoint replay requires a tune export")
        with np.load(reference, allow_pickle=False) as saved:
            actual = exported["tune"]
            if (not np.array_equal(saved["item"], actual["item"])
                    or saved["logits"].shape != actual["logits"].shape
                    or saved["native_features"].shape != actual["native_features"].shape):
                raise ValueError("Native best-task/checkpoint replay coordinates or shapes differ")
            reference_error = max(float(np.max(np.abs(saved[key] - actual[key])))
                                  for key in ("logits", "native_features") if saved[key].size)
            if not np.isfinite(reference_error) or reference_error > 1e-6:
                raise ValueError(f"Native best-task/checkpoint replay discrepancy: {reference_error} > 1e-6")
    fixed = {"feature_names", "class_names", "training_seed"}
    first = next(iter(exported.values()))
    combined = {key: first[key] if key in fixed else np.concatenate([value[key] for value in exported.values()])
                for key in first}
    if len(np.unique(combined["item"])) != len(combined["item"]):
        raise ValueError("Native observation appears in more than one export role")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for role, values in exported.items():
        paths[role] = str(output / f"{role}.npz")
        np.savez_compressed(paths[role], **values)
    bank = output / "operator_inputs.npz"
    np.savez_compressed(bank, **combined)
    config_path = output / "effective_config.json"
    _write_json(config_path, native_config)
    report = dict(operator_inputs=str(bank), roles=paths, checkpoint=str(checkpoint),
                  effective_config=str(config_path), program=program_spec,
                  frozen_parameters=True, frozen_buffers=True,
                  device=device, physical_gpu=0 if device == "cuda:0" else None,
                  physical_gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES") if device == "cuda:0" else None,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  native_feature_kind="feature_extractor_layers" if first["native_features"].shape[1] else "unavailable",
                  native_reference_max_abs=reference_error, target_accessed=not source_only)
    _write_json(output / "sources.json", report)
    return report


def export_native(native_config: dict[str, Any], checkpoint: Path, roles: Mapping[str, Any],
                  program_spec: Mapping[str, Any], output: Path, device: str, *,
                  reference: Path | None = None) -> dict[str, Any]:
    """Export source fit/tune/grow/select only; target records are rejected."""
    return _export(native_config, checkpoint, roles, program_spec, output, device,
                   source_only=True, reference=reference)


def export_target_native(native_config: dict[str, Any], checkpoint: Path,
                         target_role: Mapping[str, Any], program_spec: Mapping[str, Any],
                         output: Path, device: str, frozen_state: Path) -> dict[str, Any]:
    """Expose target windows only after the source-calibrated readout is frozen."""
    frozen = json.loads(Path(frozen_state).read_text(encoding="utf-8"))
    if (frozen.get("schema_version") != 1 or not isinstance(frozen.get("states"), Mapping)
            or not frozen["states"] or not np.isfinite(frozen.get("temperature", np.nan))
            or frozen["temperature"] <= 0):
        raise ValueError("Target export requires an existing frozen source readout and calibration")
    if Path(frozen.get("checkpoint", "")).resolve() != Path(checkpoint).resolve():
        raise ValueError("Target checkpoint differs from the frozen source checkpoint")
    if frozen.get("class_names") != list(program_spec["class_names"]):
        raise ValueError("Target class coordinates differ from the frozen source readout")
    source_config = frozen.get("native_config")
    if not isinstance(source_config, Mapping):
        raise ValueError("Target export requires the source effective native_config in the freeze artifact")
    population_fields = {"data_dir", "metadata_file", "cache_dir", "split"}
    source_processing = {key: value for key, value in source_config["data"].items() if key not in population_fields}
    target_processing = {key: value for key, value in native_config["data"].items() if key not in population_fields}
    if (source_processing != target_processing or source_config["model"] != native_config["model"]
            or source_config["environment"]["seed"] != native_config["environment"]["seed"]):
        raise ValueError("Target preprocessing, model or seed differs from the source-frozen native configuration")
    if frozen["states"] != {"native_reference": True}:
        from .features import feature_names
        from .readouts import ReadoutState

        state = ReadoutState.from_dict(dict(frozen["states"]))
        program = ProgramSpec.from_dict(dict(program_spec["program"]))
        channels = native_config["model"].get("in_channels")
        if (frozen["states"].get("source_frozen") is not True
                or state.feature_names != feature_names(program, channels)
                or any(len(saved["readout"]["intercept"]) != len(frozen["class_names"])
                       for saved in state.states.values())):
            raise ValueError("Target signal/class coordinates differ from the frozen source state")
    return _export(native_config, checkpoint, {"test": target_role}, program_spec, output, device, source_only=False)
