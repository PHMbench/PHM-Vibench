"""Guarded runtime shell for PHM generative pipelines.

The public entrypoint remains::

    python main.py --config <yaml> [--override key=value ...]

This first migration slice deliberately implements configuration loading,
preflight validation, stage selection, and iteration dispatch only. Concrete
train/sample/eval implementations are added by later focused PRs so the
unrelated-history source snapshot is never merged wholesale.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STAGE_NAMES = frozenset({"train", "sample", "eval"})
REQUIRED_CONFIG_SECTIONS = ("environment", "data", "model", "task", "trainer")


def _get_attr(value: Any, key: str, default: Any = None) -> Any:
    """Read one config field from mapping- or namespace-style objects."""

    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _validate_required_sections(configs: Any) -> None:
    """Reject incomplete five-block configurations before factory dispatch."""

    missing = [
        section
        for section in REQUIRED_CONFIG_SECTIONS
        if _get_attr(configs, section, None) is None
    ]
    if missing:
        raise ValueError(
            "generative config is missing required section(s): "
            + ", ".join(missing)
        )


def _load_configs(args: Any) -> Any:
    """Load the public config path and apply normal local/CLI overrides."""

    from src.configs.config_utils import merge_with_local_override
    from src.utils.config_utils import apply_overrides_to_config, parse_overrides

    config_path = getattr(args, "config_path", None)
    if not isinstance(config_path, str) or not config_path.strip():
        raise ValueError("Pipeline_06_generative requires args.config_path")

    configs = merge_with_local_override(
        config_path,
        getattr(args, "local_config", None),
    )
    overrides = getattr(args, "override", None)
    if overrides:
        configs = apply_overrides_to_config(configs, parse_overrides(overrides))

    _validate_required_sections(configs)
    return configs


def _generative_cfg(configs: Any) -> Any:
    """Return ``task.generative`` and reject non-generative task configs."""

    task_cfg = _get_attr(configs, "task", None)
    generative_cfg = _get_attr(task_cfg, "generative", None)
    if generative_cfg is None:
        raise ValueError(
            "Pipeline_06_generative requires task.generative configuration"
        )
    return generative_cfg


def _resolve_mode(configs: Any) -> str:
    """Resolve and validate the explicit Pipeline 06 stage."""

    mode = str(_get_attr(_generative_cfg(configs), "mode", "train")).strip().lower()
    if mode not in STAGE_NAMES:
        supported = ", ".join(sorted(STAGE_NAMES))
        raise ValueError(
            f"unsupported generative mode {mode!r}; expected one of: {supported}"
        )
    return mode


def _resolve_iterations(configs: Any) -> int:
    """Return the positive number of independently recorded iterations."""

    environment_cfg = _get_attr(configs, "environment", None)
    iterations = int(_get_attr(environment_cfg, "iterations", 1))
    if iterations <= 0:
        raise ValueError(
            f"environment.iterations must be positive, got {iterations}"
        )
    return iterations


def _validate_stage_inputs(mode: str, generative_cfg: Any) -> None:
    """Fail before deep runtime code when required stage artifacts are absent."""

    if mode == "sample":
        checkpoint_path = _get_attr(generative_cfg, "checkpoint_path", None)
        allow_untrained_smoke = bool(
            _get_attr(generative_cfg, "allow_untrained_smoke", False)
        )
        if not checkpoint_path and not allow_untrained_smoke:
            raise ValueError(
                "generative sample mode requires "
                "task.generative.checkpoint_path; set "
                "allow_untrained_smoke=true only for an explicitly untrained smoke"
            )
        if checkpoint_path and not _get_attr(
            generative_cfg,
            "normalization_path",
            None,
        ):
            raise ValueError(
                "trained generative sample mode requires "
                "task.generative.normalization_path"
            )
        if checkpoint_path and not _get_attr(
            generative_cfg,
            "normalization_sha256",
            None,
        ):
            raise ValueError(
                "trained generative sample mode requires "
                "task.generative.normalization_sha256"
            )

    if mode == "eval" and not _get_attr(
        generative_cfg,
        "generated_path",
        None,
    ):
        raise ValueError(
            "generative eval mode requires task.generative.generated_path"
        )


def _namespaces(configs: Any) -> tuple[Any, Any, Any, Any, Any]:
    from src.configs.config_utils import transfer_namespace

    return tuple(
        transfer_namespace(_get_attr(configs, section))
        for section in REQUIRED_CONFIG_SECTIONS
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _run_path(configs: Any, iteration: int) -> tuple[Path, str]:
    from src.configs.config_utils import path_name

    path, name = path_name(configs, iteration)
    return Path(path), str(name)


def _stage_ledger_path(configs: Any) -> Path:
    configured = _get_attr(_generative_cfg(configs), "stage_ledger_path", None)
    if configured:
        return Path(str(configured))
    output_dir = _get_attr(_get_attr(configs, "environment"), "output_dir", None)
    if not output_dir:
        raise ValueError("environment.output_dir is required for Pipeline 06 evidence")
    return Path(str(output_dir)) / "stage_ledger.json"


def _record_stage(configs: Any, stage: str, **values: Any) -> dict[str, Any]:
    from src.utils.generative_evidence import update_stage_ledger

    return update_stage_ledger(
        _stage_ledger_path(configs),
        stage=stage,
        values=values,
    )


def _build_stack(
    args_data: Any,
    args_model: Any,
    args_task: Any,
    args_trainer: Any,
    args_environment: Any,
) -> tuple[Any, Any, Any]:
    from src.data_factory import build_data
    from src.model_factory import build_model
    from src.task_factory import build_task

    data_factory = build_data(args_data, args_task)
    if data_factory is None:
        raise RuntimeError("data factory returned None")
    metadata = data_factory.get_metadata()
    model = build_model(args_model, metadata=metadata)
    if model is None:
        raise RuntimeError("model factory returned None")
    task = build_task(
        args_task=args_task,
        network=model,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=metadata,
    )
    if task is None:
        raise RuntimeError("task factory returned None")
    return data_factory, model, task


def _close_data_factory(data_factory: Any) -> None:
    data = getattr(data_factory, "data", None)
    close = getattr(data, "close", None)
    if callable(close):
        close()


def _to_ncl(value: Any, channels: int) -> Any:
    import torch

    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError(f"windows must be rank 3, got {tuple(tensor.shape)}")
    if tensor.shape[1] == channels:
        result = tensor.contiguous()
    elif tensor.shape[2] == channels:
        result = tensor.transpose(1, 2).contiguous()
    else:
        raise ValueError(
            f"cannot find channel axis={channels} in window shape {tuple(tensor.shape)}"
        )
    if not torch.isfinite(result).all():
        raise ValueError("windows contain NaN/Inf")
    return result


def _flatten(value: Any) -> list[Any]:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().reshape(-1).tolist()
    except ImportError:  # pragma: no cover - runtime requires torch.
        pass
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [value]


def _metadata_row(metadata: Any, file_id: Any) -> Mapping[str, Any]:
    candidates = [file_id, str(file_id)]
    try:
        candidates.insert(1, int(file_id))
    except (TypeError, ValueError):
        pass
    for key in candidates:
        try:
            row = metadata[key]
        except (KeyError, TypeError, IndexError):
            continue
        if isinstance(row, Mapping):
            return row
    raise ValueError(f"file_id={file_id!r} is absent from metadata")


def _batch_conditions(batch: Mapping[str, Any], metadata: Any) -> tuple[Any, Any]:
    import torch

    file_ids = _flatten(batch.get("file_id", []))
    labels = _flatten(batch["fault_label"]) if "fault_label" in batch else None
    if labels is None and "y" in batch:
        labels = _flatten(batch["y"])
    domains = _flatten(batch["domain_id"]) if "domain_id" in batch else None
    if labels is None and file_ids:
        labels = [int(_metadata_row(metadata, item)["Label"]) for item in file_ids]
    if domains is None and file_ids:
        domains = [int(_metadata_row(metadata, item)["Domain_id"]) for item in file_ids]
    label_tensor = torch.as_tensor(labels).long().reshape(-1) if labels is not None else None
    domain_tensor = (
        torch.as_tensor(domains).long().reshape(-1) if domains is not None else None
    )
    return label_tensor, domain_tensor


def _train_reference(
    data_factory: Any,
    channels: int,
    *,
    max_batches: int = 32,
    max_samples: int | None = None,
) -> tuple[Any, Any, Any]:
    import torch

    windows = []
    labels = []
    domains = []
    metadata = data_factory.get_metadata()
    for index, batch in enumerate(data_factory.get_dataloader("train")):
        if "x" not in batch:
            raise ValueError("train batch is missing x")
        current = _to_ncl(batch["x"], channels)
        current_labels, current_domains = _batch_conditions(batch, metadata)
        windows.append(current)
        if current_labels is not None:
            labels.append(current_labels)
        if current_domains is not None:
            domains.append(current_domains)
        if index + 1 >= max_batches:
            break
        if max_samples is not None and sum(item.shape[0] for item in windows) >= max_samples:
            break
    if not windows:
        raise ValueError("train dataloader produced no windows")
    real = torch.cat(windows, dim=0)
    real_labels = torch.cat(labels) if labels else None
    real_domains = torch.cat(domains) if domains else None
    if max_samples is not None:
        real = real[:max_samples]
        real_labels = real_labels[:max_samples] if real_labels is not None else None
        real_domains = real_domains[:max_samples] if real_domains is not None else None
    return real, real_labels, real_domains


def _write_run_contracts(
    run_path: Path,
    configs: Any,
    args: Any,
    args_task: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    from src.utils.generative_evidence import write_hashed_json

    config_path, config_hash, _ = write_hashed_json(
        run_path / "resolved_config.json",
        _plain(configs),
    )
    protocol = {
        "schema_version": "0.2.1",
        "public_entry": "python main.py --config <yaml> [--override key=value ...]",
        "config_source": str(getattr(args, "config_path", "")),
        "task": {
            "type": str(getattr(args_task, "type", "generative")),
            "name": str(getattr(args_task, "name", "")),
        },
        "conditions": ["fault_label", "domain_id"],
        "stages": ["train", "sample", "eval"],
        "reference_split": "train",
    }
    protocol_path, protocol_hash, _ = write_hashed_json(
        run_path / "generative_protocol.json",
        protocol,
    )
    return (
        {"path": str(config_path), "sha256": config_hash},
        {"path": str(protocol_path), "sha256": protocol_hash},
    )


def _write_domain_map(run_path: Path, metadata: Any) -> dict[str, str]:
    from src.utils.generative_evidence import write_hashed_json

    rows = []
    for key in metadata.keys():
        row = _metadata_row(metadata, key)
        rows.append(
            {
                "file_id": str(key),
                "fault_label": int(row["Label"]),
                "domain_id": int(row["Domain_id"]),
                "dataset_id": str(row.get("Dataset_id", "")),
                "dataset_name": str(row.get("Name", "")),
            }
        )
    path, digest, _ = write_hashed_json(run_path / "domain_map.json", {"rows": rows})
    return {"path": str(path), "sha256": digest}


def _checkpoint_path(trainer: Any, run_path: Path) -> Path:
    candidates = []
    for callback in getattr(trainer, "callbacks", []):
        best = getattr(callback, "best_model_path", "")
        if best:
            candidates.append(Path(best))
    candidates.extend(sorted(run_path.rglob("*.ckpt")))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        raise RuntimeError("training completed without an exact checkpoint artifact")
    return existing[0]


def _condition_counts(condition: dict[str, Any]) -> dict[str, int]:
    labels = condition["fault_label"].detach().cpu().reshape(-1).tolist()
    domains = condition["domain_id"].detach().cpu().reshape(-1).tolist()
    counts: dict[str, int] = {}
    for label, domain in zip(labels, domains):
        key = f"fault={int(label)},domain={int(domain)}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _sample_conditions(gen_cfg: Any, metadata: Any, num_samples: int, device: Any) -> dict:
    import torch

    policy = str(_get_attr(gen_cfg, "condition_sampling_policy", "first_metadata_repeated"))
    pairs: list[tuple[int, int]] = []
    if policy == "first_metadata_repeated":
        keys = list(metadata.keys())
        if not keys:
            raise ValueError("metadata contains no condition rows")
        row = _metadata_row(metadata, keys[0])
        pairs = [(int(row["Label"]), int(row["Domain_id"]))] * int(num_samples)
    elif policy == "grid":
        grid = _get_attr(gen_cfg, "condition_grid", None)
        if grid is None:
            raise ValueError("condition_sampling_policy=grid requires condition_grid")
        labels = _flatten(_get_attr(grid, "fault_label", []))
        domains = _flatten(_get_attr(grid, "domain_id", []))
        count = int(_get_attr(grid, "samples_per_condition", 1))
        if not labels or not domains or count <= 0:
            raise ValueError("condition_grid values and samples_per_condition are required")
        pairs = [
            (int(label), int(domain))
            for label in labels
            for domain in domains
            for _ in range(count)
        ]
    elif policy == "explicit":
        rows = _get_attr(gen_cfg, "explicit_conditions", None)
        if not rows:
            raise ValueError("condition_sampling_policy=explicit requires rows")
        for row in rows:
            count = int(_get_attr(row, "count", 1))
            if count <= 0:
                raise ValueError("explicit condition count must be positive")
            pairs.extend(
                (
                    int(_get_attr(row, "fault_label")),
                    int(_get_attr(row, "domain_id")),
                )
                for _ in range(count)
            )
    else:
        raise ValueError(f"unsupported condition sampling policy: {policy}")
    return {
        "fault_label": torch.tensor(
            [label for label, _ in pairs], dtype=torch.long, device=device
        ),
        "domain_id": torch.tensor(
            [domain for _, domain in pairs], dtype=torch.long, device=device
        ),
    }


def _sample_device(args_trainer: Any) -> Any:
    import torch

    requested = str(getattr(args_trainer, "device", "cpu")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"trainer.device={requested!r} requires available CUDA")
    return torch.device("cuda")


def _load_samples(path: str | Path) -> tuple[Any, Any, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "samples" not in payload:
        raise ValueError("sample artifact must be a mapping containing samples")
    samples = torch.as_tensor(payload["samples"]).float()
    labels = torch.as_tensor(payload.get("fault_label")).long().reshape(-1)
    domains = torch.as_tensor(payload.get("domain_id")).long().reshape(-1)
    if not torch.isfinite(samples).all():
        raise ValueError("sample artifact contains NaN/Inf")
    return samples, labels, domains


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> str:
    from src.task_factory.Components.generative import REQUIRED_METRICS
    from src.utils.generative_evidence import sha256_file

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "status", "reason"])
        writer.writeheader()
        for name in REQUIRED_METRICS:
            result = metrics[name]
            writer.writerow(
                {
                    "metric": name,
                    "value": "" if result["value"] is None else result["value"],
                    "status": result["status"],
                    "reason": result["reason"],
                }
            )
    return sha256_file(path)


def _run_train_stage(args: Any, configs: Any, iteration: int) -> Any:
    import torch
    from pytorch_lightning import seed_everything

    from src.task_factory.Components.generative import (
        build_normalization_evidence,
        write_normalization_evidence,
    )
    from src.trainer_factory import build_trainer
    from src.utils.generative_evidence import strict_load_lightning_checkpoint

    args_environment, args_data, args_model, args_task, args_trainer = _namespaces(configs)
    run_path, name = _run_path(configs, iteration)
    seed = int(getattr(args_environment, "seed", 0)) + iteration
    seed_everything(seed, workers=True)
    args_trainer.logger_name = name
    _record_stage(configs, "train", status="running", run_dir=str(run_path), seed=seed)
    data_factory = None
    try:
        data_factory, _, task = _build_stack(
            args_data, args_model, args_task, args_trainer, args_environment
        )
        channels = int(getattr(args_model, "in_channels", 1))
        train_windows, _, _ = _train_reference(data_factory, channels)
        normalization = build_normalization_evidence(
            train_windows,
            method=str(getattr(args_data, "normalization", "standardization")),
        )
        normalization_path, normalization_hash, normalization_hash_path = (
            write_normalization_evidence(
                str(run_path / "normalization_params.json"),
                normalization,
            )
        )
        config_evidence, protocol_evidence = _write_run_contracts(
            run_path, configs, args, args_task
        )
        trainer = build_trainer(args_environment, args_trainer, args_data, str(run_path))
        if trainer is None:
            raise RuntimeError("trainer factory returned None")
        started = time.perf_counter()
        trainer.fit(
            task,
            data_factory.get_dataloader("train"),
            data_factory.get_dataloader("val"),
        )
        wall_clock = time.perf_counter() - started
        if not math.isfinite(wall_clock):
            raise ValueError("training wall-clock value is not finite")
        checkpoint = _checkpoint_path(trainer, run_path)
        checkpoint_evidence = strict_load_lightning_checkpoint(task, checkpoint)
        if any(not torch.isfinite(parameter).all() for parameter in task.parameters()):
            raise ValueError("trained checkpoint contains NaN/Inf parameters")
        result = {
            "stage": "train",
            "status": "completed",
            "run_dir": str(run_path),
            "seed": seed,
            "training_wall_clock_seconds": wall_clock,
            "checkpoint": checkpoint_evidence,
            "normalization": {
                "path": normalization_path,
                "sha256": normalization_hash,
                "sha256_path": normalization_hash_path,
                "source_split": "train",
                "scope": "per_channel",
            },
            "config": config_evidence,
            "protocol": protocol_evidence,
        }
        _record_stage(
            configs,
            "train",
            **{key: value for key, value in result.items() if key != "stage"},
        )
        return result
    finally:
        if data_factory is not None:
            _close_data_factory(data_factory)


def _run_sample_stage(args: Any, configs: Any, iteration: int) -> Any:
    import torch
    from pytorch_lightning import seed_everything

    from src.task_factory.Components.generative import (
        build_synthetic_manifest,
        evaluate_smoke_metrics,
        load_normalization_evidence,
    )
    from src.utils.generative_evidence import (
        dependency_lock_evidence,
        git_commit_sha,
        load_hashed_json,
        sha256_file,
        strict_load_lightning_checkpoint,
        write_hashed_json,
    )

    args_environment, args_data, args_model, args_task, args_trainer = _namespaces(configs)
    gen_cfg = _get_attr(args_task, "generative")
    run_path, name = _run_path(configs, iteration)
    seed = int(getattr(args_environment, "seed", 0)) + iteration
    seed_everything(seed, workers=True)
    _record_stage(configs, "sample", status="running", run_dir=str(run_path), seed=seed)
    data_factory = None
    try:
        data_factory, _, task = _build_stack(
            args_data, args_model, args_task, args_trainer, args_environment
        )
        checkpoint_path = str(_get_attr(gen_cfg, "checkpoint_path", "") or "")
        checkpoint_evidence: dict[str, Any] = {}
        normalization_evidence: dict[str, Any] = {}
        if checkpoint_path:
            checkpoint_evidence = strict_load_lightning_checkpoint(task, checkpoint_path)
            normalization_path = str(_get_attr(gen_cfg, "normalization_path"))
            normalization_hash = str(_get_attr(gen_cfg, "normalization_sha256"))
            normalization_evidence = load_normalization_evidence(
                normalization_path,
                expected_hash=normalization_hash,
            )
            normalization_evidence["path"] = normalization_path
        device = _sample_device(args_trainer)
        task.to(device)
        num_samples = int(_get_attr(gen_cfg, "num_samples", 2))
        condition = _sample_conditions(
            gen_cfg,
            data_factory.get_metadata(),
            num_samples,
            device,
        )
        num_samples = int(condition["fault_label"].numel())
        num_steps = int(_get_attr(gen_cfg, "num_steps", 8))
        length = int(_get_attr(gen_cfg, "length", getattr(args_data, "window_size", 128)))
        channels = int(getattr(args_model, "in_channels", 1))
        started = time.perf_counter()
        samples = task.sample(
            condition,
            num_samples=num_samples,
            length=length,
            channels=channels,
            num_steps=num_steps,
            device=device,
        ).detach().cpu()
        wall_clock = time.perf_counter() - started
        if not torch.isfinite(samples).all() or not math.isfinite(wall_clock):
            raise ValueError("sampling produced non-finite samples or timing")
        output_dir = run_path / "synthetic"
        output_dir.mkdir(parents=True, exist_ok=True)
        samples_path = output_dir / "samples.pt"
        torch.save(
            {
                "samples": samples,
                "fault_label": condition["fault_label"].detach().cpu(),
                "domain_id": condition["domain_id"].detach().cpu(),
                "condition_sampling_policy": str(
                    _get_attr(gen_cfg, "condition_sampling_policy", "first_metadata_repeated")
                ),
            },
            samples_path,
        )
        samples_hash = sha256_file(samples_path)
        real, real_labels, real_domains = _train_reference(
            data_factory,
            channels,
            max_samples=num_samples,
        )
        leakage_bundle = evaluate_smoke_metrics(
            real,
            samples,
            real_labels=real_labels,
            fake_labels=condition["fault_label"].detach().cpu(),
            real_domains=real_domains,
            fake_domains=condition["domain_id"].detach().cpu(),
        )
        config_evidence, generated_protocol = _write_run_contracts(
            run_path, configs, args, args_task
        )
        protocol_path = str(
            _get_attr(gen_cfg, "protocol_path", "")
            or Path(str(_get_attr(gen_cfg, "normalization_path", run_path)))
            .parent.joinpath("generative_protocol.json")
        )
        if checkpoint_path:
            _, protocol_hash = load_hashed_json(protocol_path)
            protocol_evidence = {"path": protocol_path, "sha256": protocol_hash}
        else:
            protocol_evidence = generated_protocol
        metadata_path = Path(str(args_data.data_dir)) / str(args_data.metadata_file)
        domain_map = _write_domain_map(run_path, data_factory.get_metadata())
        source_split = str(_get_attr(gen_cfg, "source_split", "train"))
        manifest = build_synthetic_manifest(
            synthetic_dataset_id=str(
                _get_attr(gen_cfg, "synthetic_dataset_id", f"{name}-iter-{iteration}")
            ),
            method_id=str(
                getattr(task, "method_id", "conditional_flow_matching")
            ),
            model_type=str(args_model.type),
            model_name=str(args_model.name),
            loss_id=str(getattr(task, "loss_id", "conditional_flow_matching")),
            sampler_id=str(getattr(task, "sampler_id", "euler_ode")),
            source_split=source_split,
            seed=seed,
            num_steps=num_steps,
            num_samples=num_samples,
            shape=list(samples.shape),
            condition_sampling_policy=str(
                _get_attr(gen_cfg, "condition_sampling_policy", "first_metadata_repeated")
            ),
            condition_counts=_condition_counts(condition),
            checkpoint_evidence=checkpoint_evidence,
            normalization_evidence=normalization_evidence,
            config_evidence=config_evidence,
            protocol_evidence=protocol_evidence,
            code_evidence={"commit": git_commit_sha()},
            dependency_evidence=dependency_lock_evidence(),
            data_evidence={
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "domain_map_path": domain_map["path"],
                "domain_map_sha256": domain_map["sha256"],
            },
            generated_evidence={"path": str(samples_path), "sha256": samples_hash},
            leakage_metrics={
                name: leakage_bundle[name]
                for name in ("nearest_neighbor_leakage_l2", "duplicate_rate")
            },
            population_metrics={
                "population_dependency_mmd": leakage_bundle[
                    "population_dependency_mmd"
                ]
            },
            sampler_metadata=dict(getattr(task, "sampler_metadata", lambda: {})()),
            scientific_status=str(_get_attr(gen_cfg, "validity_status", "exploratory")),
        )
        manifest_path, manifest_hash, _ = write_hashed_json(
            output_dir / "synthetic_data_manifest.json",
            manifest,
        )
        result = {
            "stage": "sample",
            "status": "completed",
            "run_dir": str(run_path),
            "sampling_wall_clock_seconds": wall_clock,
            "samples": {"path": str(samples_path), "sha256": samples_hash},
            "synthetic_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_hash,
            },
            "checkpoint": checkpoint_evidence,
            "normalization": normalization_evidence,
        }
        _record_stage(
            configs,
            "sample",
            **{key: value for key, value in result.items() if key != "stage"},
        )
        return result
    finally:
        if data_factory is not None:
            _close_data_factory(data_factory)


def _run_eval_stage(args: Any, configs: Any, iteration: int) -> Any:
    from pytorch_lightning import seed_everything

    from src.task_factory.Components.generative import (
        build_evaluation_manifest,
        evaluate_smoke_metrics,
        load_normalization_evidence,
    )
    from src.utils.generative_evidence import (
        load_hashed_json,
        sha256_file,
        write_hashed_json,
    )

    args_environment, args_data, args_model, args_task, args_trainer = _namespaces(configs)
    gen_cfg = _get_attr(args_task, "generative")
    run_path, _ = _run_path(configs, iteration)
    seed = int(getattr(args_environment, "seed", 0)) + iteration
    seed_everything(seed, workers=True)
    _record_stage(configs, "eval", status="running", run_dir=str(run_path), seed=seed)
    data_factory = None
    try:
        data_factory, _, _ = _build_stack(
            args_data, args_model, args_task, args_trainer, args_environment
        )
        generated_path = Path(str(_get_attr(gen_cfg, "generated_path")))
        samples, fake_labels, fake_domains = _load_samples(generated_path)
        manifest_path = Path(
            str(
                _get_attr(gen_cfg, "synthetic_manifest_path", "")
                or generated_path.with_name("synthetic_data_manifest.json")
            )
        )
        synthetic_manifest, synthetic_manifest_hash = load_hashed_json(manifest_path)
        expected_samples_hash = _get_attr(
            _get_attr(synthetic_manifest, "generated_artifact", {}),
            "sha256",
            None,
        )
        actual_samples_hash = sha256_file(generated_path)
        if expected_samples_hash != actual_samples_hash:
            raise ValueError(
                "generated sample hash does not match the synthetic manifest: "
                f"expected {expected_samples_hash}, got {actual_samples_hash}"
            )
        normalization = _get_attr(synthetic_manifest, "normalization", {})
        normalization_path = str(_get_attr(normalization, "path", ""))
        normalization_hash = str(_get_attr(normalization, "sha256", ""))
        load_normalization_evidence(
            normalization_path,
            expected_hash=normalization_hash,
        )
        reference_split = str(_get_attr(gen_cfg, "eval_split", "train")).lower()
        if reference_split != "train":
            raise ValueError(
                "G3 deterministic evaluation requires task.generative.eval_split=train"
            )
        channels = int(getattr(args_model, "in_channels", samples.shape[1]))
        samples = _to_ncl(samples, channels)
        real, real_labels, real_domains = _train_reference(
            data_factory,
            channels,
            max_samples=samples.shape[0],
        )
        ledger, _ = load_hashed_json(_stage_ledger_path(configs))
        train_stage = _get_attr(_get_attr(ledger, "stages", {}), "train", {})
        sample_stage = _get_attr(_get_attr(ledger, "stages", {}), "sample", {})
        training_wall_clock = _get_attr(
            train_stage,
            "training_wall_clock_seconds",
            None,
        )
        sampling_wall_clock = _get_attr(
            sample_stage,
            "sampling_wall_clock_seconds",
            None,
        )
        metrics = evaluate_smoke_metrics(
            real,
            samples,
            real_labels=real_labels,
            fake_labels=fake_labels,
            real_domains=real_domains,
            fake_domains=fake_domains,
            duplicate_threshold=float(
                _get_attr(gen_cfg, "leakage_duplicate_threshold", 1e-6)
            ),
            training_wall_clock_seconds=(
                float(training_wall_clock) if training_wall_clock is not None else None
            ),
        )
        metrics_path = run_path / "generative_eval_metrics.csv"
        metrics_hash = _write_metrics_csv(metrics_path, metrics)
        evaluation_manifest = build_evaluation_manifest(
            generated_path=str(generated_path),
            generated_sha256=actual_samples_hash,
            synthetic_manifest_path=str(manifest_path),
            synthetic_manifest_sha256=synthetic_manifest_hash,
            metrics_path=str(metrics_path),
            metrics_sha256=metrics_hash,
            reference_split=reference_split,
            metrics=metrics,
            training_wall_clock_seconds=(
                float(training_wall_clock) if training_wall_clock is not None else None
            ),
            sampling_wall_clock_seconds=(
                float(sampling_wall_clock) if sampling_wall_clock is not None else None
            ),
        )
        evaluation_path, evaluation_hash, _ = write_hashed_json(
            run_path / "evaluation_evidence_manifest.json",
            evaluation_manifest,
        )
        result = {
            "stage": "eval",
            "status": "completed",
            "run_dir": str(run_path),
            "metrics": {"path": str(metrics_path), "sha256": metrics_hash},
            "evaluation_manifest": {
                "path": str(evaluation_path),
                "sha256": evaluation_hash,
            },
            "metric_summary": metrics["summary"],
        }
        _record_stage(
            configs,
            "eval",
            **{key: value for key, value in result.items() if key != "stage"},
        )
        return result
    finally:
        if data_factory is not None:
            _close_data_factory(data_factory)


def pipeline(args: Any) -> list[Any]:
    """Load, preflight, and dispatch one explicit generative stage.

    Train, sample, and eval remain separate invocations. A future optional
    orchestrator may invoke ``main.py`` three times, but it must not bypass this
    public entrypoint or hide intermediate checkpoint/sample paths.
    """

    configs = _load_configs(args)
    mode = _resolve_mode(configs)
    generative_cfg = _generative_cfg(configs)
    _validate_stage_inputs(mode, generative_cfg)

    handlers = {
        "train": _run_train_stage,
        "sample": _run_sample_stage,
        "eval": _run_eval_stage,
    }
    handler = handlers[mode]

    results: list[Any] = []
    for iteration in range(_resolve_iterations(configs)):
        try:
            results.append(handler(args, configs, iteration))
        except Exception as exc:
            try:
                _record_stage(
                    configs,
                    mode,
                    status="failed",
                    iteration=iteration,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            except Exception as ledger_exc:
                raise exc from ledger_exc
            raise
    return results
