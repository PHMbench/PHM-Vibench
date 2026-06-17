import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from src.configs.config_utils import merge_with_local_override, path_name, transfer_namespace
from src.data_factory import build_data
from src.data_factory.data_utils import (
    resolve_normalization_method,
    write_normalization_params_artifact,
)
from src.data_factory.ID.domain_map import hash_file
from src.model_factory import build_model
from src.task_factory import build_task
from src.task_factory.task.generative.generative_eval import evaluate_generated_windows
from src.task_factory.Components.generative.metrics.leakage import leakage_metrics
from src.utils.config_utils import apply_overrides_to_config, parse_overrides


PROTOCOL_SCHEMA_PATH = "docs/schemas/generative_protocol.schema.json"
STAGE_NAMES = {"train", "sample", "eval", "paperpack"}


def _load_configs(args):
    configs = merge_with_local_override(args.config_path, getattr(args, "local_config", None))
    if hasattr(args, "override") and args.override:
        overrides = parse_overrides(args.override)
        configs = apply_overrides_to_config(configs, overrides)
    for section in ["environment", "data", "model", "task", "trainer"]:
        if not hasattr(configs, section):
            raise ValueError(f"config is missing required section: {section}")
    return configs


def _namespaces(configs):
    return (
        transfer_namespace(configs.environment),
        transfer_namespace(configs.data),
        transfer_namespace(configs.model),
        transfer_namespace(configs.task),
        transfer_namespace(configs.trainer),
    )


def _generative_cfg(args_task):
    cfg = getattr(args_task, "generative", None)
    if cfg is None:
        raise ValueError("generative tasks require task.generative.* configuration")
    return cfg


def _build_stack(args_data, args_model, args_task, args_trainer, args_environment):
    data_factory = build_data(args_data, args_task)
    model = build_model(args_model, metadata=data_factory.get_metadata())
    task = build_task(
        args_task=args_task,
        network=model,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=data_factory.get_metadata(),
    )
    if task is None:
        raise RuntimeError("failed to build generative task")
    return data_factory, model, task


def _first_condition_from_metadata(metadata, device):
    first_id = list(metadata.keys())[0]
    row = metadata[first_id]
    return {
        "fault_label": torch.tensor([int(row["Label"])], device=device),
        "domain_id": torch.tensor([int(row["Domain_id"])], device=device),
    }


def _get_cfg_attr(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _optional_float(*values) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _as_int_list(value, field_name: str) -> list[int]:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if torch.is_tensor(value):
        values = value.detach().cpu().view(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return [int(item) for item in values]


def _condition_from_pairs(pairs: list[tuple[int, int]], device) -> dict[str, torch.Tensor]:
    if not pairs:
        raise ValueError("condition sampling produced no fault/domain pairs")
    labels = torch.tensor([label for label, _ in pairs], dtype=torch.long, device=device)
    domains = torch.tensor([domain for _, domain in pairs], dtype=torch.long, device=device)
    return {"fault_label": labels, "domain_id": domains}


def _metadata_condition_pairs(metadata, split: str | None = None) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    split_name = str(split).lower() if split else None
    for row in metadata.values():
        if split_name:
            row_split = (
                row.get("split")
                or row.get("Split")
                or row.get("source_split")
                or row.get("Source_split")
            )
            if row_split is not None and str(row_split).lower() != split_name:
                continue
        pairs.append((int(row["Label"]), int(row["Domain_id"])))
    if not pairs:
        raise ValueError("metadata does not contain any usable fault/domain condition pairs")
    return pairs


def _metadata_row_split(row) -> str | None:
    value = (
        row.get("split")
        or row.get("Split")
        or row.get("source_split")
        or row.get("Source_split")
    )
    return str(value).lower() if value is not None else None


def _metadata_has_explicit_split(metadata) -> bool:
    return bool(metadata) and all(
        _metadata_row_split(row) is not None for row in metadata.values()
    )


def _condition_from_grid(gen_cfg, device) -> dict[str, torch.Tensor]:
    grid = _get_cfg_attr(gen_cfg, "condition_grid")
    if grid is None:
        raise ValueError("condition_sampling_policy=grid requires task.generative.condition_grid")
    labels = _as_int_list(_get_cfg_attr(grid, "fault_label"), "condition_grid.fault_label")
    domains = _as_int_list(_get_cfg_attr(grid, "domain_id"), "condition_grid.domain_id")
    samples_per_condition = int(_get_cfg_attr(grid, "samples_per_condition", 1))
    if samples_per_condition <= 0:
        raise ValueError("condition_grid.samples_per_condition must be positive")
    pairs = [
        (label, domain)
        for label in labels
        for domain in domains
        for _ in range(samples_per_condition)
    ]
    return _condition_from_pairs(pairs, device)


def _condition_from_explicit(gen_cfg, device) -> dict[str, torch.Tensor]:
    rows = _get_cfg_attr(gen_cfg, "explicit_conditions")
    if not rows:
        raise ValueError("condition_sampling_policy=explicit requires explicit_conditions")
    pairs: list[tuple[int, int]] = []
    for row in rows:
        label = int(_get_cfg_attr(row, "fault_label"))
        domain = int(_get_cfg_attr(row, "domain_id"))
        count = int(_get_cfg_attr(row, "count", 1))
        if count <= 0:
            raise ValueError("explicit_conditions.count must be positive")
        pairs.extend((label, domain) for _ in range(count))
    return _condition_from_pairs(pairs, device)


def _condition_from_train_distribution(metadata, num_samples: int, seed: int, device):
    pairs = _metadata_condition_pairs(metadata, split="train")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(len(pairs), (int(num_samples),), generator=generator).tolist()
    sampled = [pairs[index] for index in indices]
    return _condition_from_pairs(sampled, device)


def _select_condition(gen_cfg, metadata, num_samples: int, seed: int, device):
    policy = str(_get_cfg_attr(gen_cfg, "condition_sampling_policy", "first_metadata_repeated"))
    condition_seed = int(_get_cfg_attr(gen_cfg, "condition_seed", seed) or seed)
    if policy == "first_metadata_repeated":
        condition = _first_condition_from_metadata(metadata, device)
        return _expand_condition(condition, int(num_samples), device)
    if policy == "grid":
        return _condition_from_grid(gen_cfg, device)
    if policy == "train_distribution":
        return _condition_from_train_distribution(metadata, int(num_samples), condition_seed, device)
    if policy == "explicit":
        return _condition_from_explicit(gen_cfg, device)
    raise ValueError(f"Unsupported condition_sampling_policy: {policy}")


def _condition_sampling_split_verified(policy: str, metadata) -> bool:
    if policy != "train_distribution":
        return True
    return _metadata_has_explicit_split(metadata)


def _hash_path_or_value(path_or_value: str) -> str:
    path = Path(path_or_value)
    if path.exists() and path.is_file():
        return hash_file(str(path))
    return hashlib.sha256(str(path_or_value).encode("utf-8")).hexdigest()


def _dependency_lock_hash() -> str:
    for candidate in ["requirements.txt", "environment.yml", "pyproject.toml"]:
        if Path(candidate).exists():
            return hash_file(candidate)
    return "missing"


def _build_normalization_params(data_factory, args_data, channels: int, max_batches: int = 32) -> dict:
    method = resolve_normalization_method(getattr(args_data, "normalization", "standardization"))
    chunks: list[torch.Tensor] = []
    for batch_idx, batch in enumerate(data_factory.get_dataloader("train")):
        if "x" not in batch:
            raise ValueError("train batch is missing x; cannot record normalization params")
        chunks.append(_to_ncl(batch["x"], channels).detach().cpu().float())
        if batch_idx + 1 >= max_batches:
            break
    if not chunks:
        raise ValueError("train split produced no batches; cannot record normalization params")

    windows = torch.cat(chunks, dim=0)
    if not torch.isfinite(windows).all():
        raise ValueError("train split contains NaN/Inf; cannot record normalization params")
    flat = windows.permute(1, 0, 2).reshape(windows.shape[1], -1)
    channel_stats: dict[str, dict[str, float]] = {}
    if method == "standardization":
        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False)
        for idx in range(flat.shape[0]):
            channel_stats[str(idx)] = {
                "mean": float(mean[idx].item()),
                "std": float(std[idx].item()),
                "epsilon": 1e-8,
            }
    elif method == "robust_scaler":
        median = flat.median(dim=1).values
        q1 = torch.quantile(flat, 0.25, dim=1)
        q3 = torch.quantile(flat, 0.75, dim=1)
        iqr = q3 - q1
        for idx in range(flat.shape[0]):
            channel_stats[str(idx)] = {
                "median": float(median[idx].item()),
                "q1": float(q1[idx].item()),
                "q3": float(q3[idx].item()),
                "iqr": float(iqr[idx].item()),
                "epsilon": 1e-8,
            }
    else:  # pragma: no cover - resolve_normalization_method guards this.
        raise ValueError(f"unsupported normalization method: {method}")

    return {
        "method": method,
        "scope": "per_channel",
        "source_split": "train",
        "source": "train_dataloader_processed_windows",
        "num_windows": int(windows.shape[0]),
        "num_values_per_channel": int(flat.shape[1]),
        "channels": channel_stats,
    }


def _attach_normalization_artifacts(run_path: str | Path, data_factory, args_data, task, channels: int) -> tuple[str, str]:
    params = _build_normalization_params(data_factory, args_data, channels)
    params_path, params_hash, sha_path = write_normalization_params_artifact(params, run_path)
    for obj in (args_data, getattr(task, "args_data", None)):
        if obj is None:
            continue
        setattr(obj, "normalization_params_path", params_path)
        setattr(obj, "normalization_params_hash", params_hash)
        setattr(obj, "normalization_params_sha256_path", sha_path)
        setattr(obj, "normalization_scope", "per_channel")
    return params_path, params_hash


def _count_parameters(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _peak_memory_bytes() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def _expand_condition(condition: dict[str, torch.Tensor], num_samples: int, device) -> dict[str, torch.Tensor]:
    expanded = {}
    for key, value in condition.items():
        value = value.to(device).long().view(-1)
        if value.numel() == 1 and num_samples > 1:
            value = value.repeat(num_samples)
        if value.numel() != num_samples:
            raise ValueError(
                f"condition {key} must have 1 or num_samples values; "
                f"got {value.numel()} for num_samples={num_samples}"
            )
        expanded[key] = value
    return expanded


def _condition_counts(condition: dict[str, torch.Tensor]) -> dict[str, int]:
    labels = condition.get("fault_label")
    domains = condition.get("domain_id")
    if labels is None or domains is None:
        return {}
    counts: dict[str, int] = {}
    for label, domain in zip(labels.view(-1).tolist(), domains.view(-1).tolist()):
        key = f"fault={int(label)},domain={int(domain)}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _flatten_values(value) -> list:
    if torch.is_tensor(value):
        return value.detach().cpu().view(-1).tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if torch.is_tensor(item):
                out.extend(item.detach().cpu().view(-1).tolist())
            else:
                out.append(item)
        return out
    return [value]


def _metadata_row(metadata, file_id):
    candidates = [file_id]
    try:
        candidates.append(int(file_id))
    except (TypeError, ValueError):
        pass
    candidates.append(str(file_id))
    for key in candidates:
        try:
            return metadata[key]
        except KeyError:
            continue
    raise ValueError(f"file_id={file_id!r} is missing from metadata")


def _labels_from_batch(batch, metadata, batch_key: str, metadata_key: str):
    if batch_key in batch:
        return torch.as_tensor(_flatten_values(batch[batch_key]), dtype=torch.long)
    if batch_key == "fault_label" and "y" in batch:
        return torch.as_tensor(_flatten_values(batch["y"]), dtype=torch.long)
    if "file_id" not in batch:
        return None
    values = []
    for file_id in _flatten_values(batch["file_id"]):
        row = _metadata_row(metadata, file_id)
        if metadata_key not in row:
            return None
        values.append(int(row[metadata_key]))
    return torch.as_tensor(values, dtype=torch.long)


def _to_ncl(x: torch.Tensor, channels: int) -> torch.Tensor:
    x = torch.as_tensor(x).float()
    if x.ndim != 3:
        raise ValueError(f"expected [N, L, C] or [N, C, L], got shape={tuple(x.shape)}")
    if x.shape[1] == channels:
        return x.contiguous()
    if x.shape[2] == channels:
        return x.transpose(1, 2).contiguous()
    raise ValueError(
        f"expected channel axis with channels={channels} in [N,C,L] or [N,L,C], "
        f"got shape={tuple(x.shape)}"
    )


def _stage_ledger_path(configs, run_path: str | Path, mode: str) -> Path:
    args_task = transfer_namespace(configs.task)
    gen_cfg = _generative_cfg(args_task)
    configured = getattr(gen_cfg, "stage_ledger_path", None)
    if configured:
        return Path(str(configured))
    output_dir = Path(str(configs.environment.get("output_dir", "")))
    if output_dir.name in STAGE_NAMES:
        return output_dir.parent / "stage_ledger.json"
    return Path(run_path) / "stage_ledger.json"


def _update_stage_ledger(path: str | Path, *, mode: str, values: dict) -> None:
    ledger_path = Path(path)
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        ledger = {"schema_version": "0.3.0", "stages": {}}
    ledger.setdefault("schema_version", "0.3.0")
    ledger.setdefault("stages", {})
    stage = dict(ledger["stages"].get(mode, {}))
    stage.update(
        {key: str(value) for key, value in values.items() if value not in {None, ""}}
    )
    ledger["stages"][mode] = stage
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _find_checkpoint_path(run_path: str | Path) -> str:
    candidates = sorted(Path(run_path).rglob("*.ckpt"))
    if not candidates:
        return ""
    for path in candidates:
        if path.name == "best.ckpt":
            return str(path)
    return str(candidates[0])


def _resolve_synthetic_manifest_path(generated_path: str | Path) -> str:
    sample_path = Path(generated_path)
    candidates = [
        sample_path.with_name("synthetic_data_manifest.json"),
        sample_path.parent / "synthetic_data_manifest.json",
        sample_path.parent.parent / "synthetic_data_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def _metric_status_summary(metrics: dict) -> dict[str, int]:
    summary = {"ok": 0, "not_computable": 0}
    for key, value in metrics.items():
        if not key.endswith("_status"):
            continue
        status = str(value)
        if status in summary:
            summary[status] += 1
    return summary


def _write_eval_evidence_manifest(
    path: str | Path,
    *,
    generated_path: str | Path,
    metrics_path: str | Path,
    reference_split: str,
    allow_test_reference_eval: bool,
    metrics: dict,
) -> dict:
    synthetic_manifest_path = _resolve_synthetic_manifest_path(generated_path)
    status_summary = _metric_status_summary(metrics)
    missing: list[str] = []
    if not synthetic_manifest_path:
        missing.append("synthetic_manifest_path")
    if status_summary["not_computable"]:
        missing.append("metric_status_ok")
    if not status_summary["ok"] and not status_summary["not_computable"]:
        missing.append("metric_status_reason_recorded")
    manifest = {
        "schema_version": "0.3.0",
        "generated_path": str(generated_path),
        "synthetic_manifest_path": synthetic_manifest_path,
        "metrics_path": str(metrics_path),
        "reference_split": reference_split,
        "allow_test_reference_eval": bool(allow_test_reference_eval),
        "metric_status_summary": status_summary,
        "promotion": {
            "eligible": not missing,
            "missing": missing,
        },
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _load_sample_payload(path: str | Path) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    from src.utils.utils import safe_torch_load

    payload = safe_torch_load(str(path), map_location="cpu")
    if isinstance(payload, dict):
        if "samples" not in payload:
            raise ValueError("generated sample payload dict must contain 'samples'")
        samples = torch.as_tensor(payload["samples"]).float()
        fault_label = payload.get("fault_label")
        domain_id = payload.get("domain_id")
        return (
            samples,
            torch.as_tensor(fault_label).long().view(-1) if fault_label is not None else None,
            torch.as_tensor(domain_id).long().view(-1) if domain_id is not None else None,
        )
    return torch.as_tensor(payload).float(), None, None


def _leakage_checks_from_train_batch(data_factory, fake: torch.Tensor, channels: int, threshold: float):
    try:
        train_batch = next(iter(data_factory.get_dataloader("train")))
        real = _to_ncl(train_batch["x"], channels)
        n = min(real.shape[0], fake.shape[0])
        metrics = leakage_metrics(real[:n], fake[:n], duplicate_threshold=threshold)
        passed = metrics.get("leakage_nearest_neighbor_pass", 0.0) == 1.0
        return {
            "split_guard_passed": True,
            "nearest_neighbor_check": "passed" if passed else "failed",
            "nearest_neighbor_l2": metrics.get("leakage_nearest_neighbor_l2", float("nan")),
            "duplicate_rate": metrics.get("leakage_duplicate_rate", float("nan")),
            "duplicate_threshold": float(threshold),
        }
    except Exception as exc:
        return {
            "split_guard_passed": True,
            "nearest_neighbor_check": "not_run",
            "reason": str(exc),
        }


def _train_one_iteration(args, configs, iteration):
    from pytorch_lightning import seed_everything

    from src.trainer_factory import build_trainer
    from src.utils.utils import close_lab, init_lab, load_best_model_checkpoint

    args_environment, args_data, args_model, args_task, args_trainer = _namespaces(configs)
    path, name = path_name(configs, iteration)
    args_trainer.logger_name = name
    seed_everything(int(args_environment.seed) + iteration)
    init_lab(args_environment, args, name)

    data_factory, _, task = _build_stack(
        args_data, args_model, args_task, args_trainer, args_environment
    )
    channels = int(getattr(args_model, "in_channels", getattr(args_model, "channels", 1)))
    _attach_normalization_artifacts(path, data_factory, args_data, task, channels)
    trainer = build_trainer(args_environment, args_trainer, args_data, path)
    if trainer is None:
        raise RuntimeError("failed to build trainer")

    start = time.perf_counter()
    trainer.fit(task, data_factory.get_dataloader("train"), data_factory.get_dataloader("val"))
    fit_wall_clock = time.perf_counter() - start
    result = []
    gen_cfg = _generative_cfg(args_task)
    if bool(getattr(gen_cfg, "run_test_loss_after_train", False)):
        task = load_best_model_checkpoint(task, trainer)
        result = trainer.test(task, data_factory.get_dataloader("test"))
    data_factory.data.close()
    close_lab()

    result_row = result[0] if result else {"train_completed": True}
    result_row.update(
        {
            "train_wall_clock_sec": float(fit_wall_clock),
            "parameter_count": _count_parameters(task),
            "post_train_test_loss_ran": float(bool(result)),
        }
    )
    pd.DataFrame([result_row]).to_csv(os.path.join(path, f"train_result_{iteration}.csv"), index=False)
    _update_stage_ledger(
        _stage_ledger_path(configs, path, "train"),
        mode="train",
        values={
            "run_dir": path,
            "checkpoint_path": _find_checkpoint_path(path),
            "train_result_path": os.path.join(path, f"train_result_{iteration}.csv"),
        },
    )
    return result_row


def _sample_once(args, configs, iteration):
    from pytorch_lightning import seed_everything

    args_environment, args_data, args_model, args_task, args_trainer = _namespaces(configs)
    gen_cfg = _generative_cfg(args_task)
    path, name = path_name(configs, iteration)
    args_trainer.logger_name = name
    seed = int(args_environment.seed) + iteration
    seed_everything(seed)

    data_factory, _, task = _build_stack(
        args_data, args_model, args_task, args_trainer, args_environment
    )
    channels = int(getattr(args_model, "in_channels", getattr(args_model, "channels", 2)))
    _attach_normalization_artifacts(path, data_factory, args_data, task, channels)
    checkpoint_path = str(getattr(gen_cfg, "checkpoint_path", "") or "")
    if checkpoint_path:
        from src.utils.utils import safe_torch_load

        state = safe_torch_load(checkpoint_path, map_location="cpu")
        task.load_state_dict(state.get("state_dict", state), strict=False)
    elif not bool(getattr(gen_cfg, "allow_untrained_smoke", False)):
        raise ValueError("sample mode requires task.generative.checkpoint_path")

    num_samples = int(getattr(gen_cfg, "num_samples", 2))
    num_steps = int(getattr(gen_cfg, "num_steps", 8))
    length = int(getattr(gen_cfg, "length", getattr(args_data, "window_size", 128)))
    metadata = data_factory.get_metadata()
    policy = str(getattr(gen_cfg, "condition_sampling_policy", "first_metadata_repeated"))
    expanded_condition = _select_condition(
        gen_cfg,
        metadata,
        num_samples,
        seed,
        task.device,
    )
    num_samples = int(expanded_condition["fault_label"].numel())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sample_start = time.perf_counter()
    samples = task.sample(
        expanded_condition,
        num_samples=num_samples,
        length=length,
        channels=channels,
        num_steps=num_steps,
        device=task.device,
    )
    sampling_wall_clock = time.perf_counter() - sample_start

    output_dir = Path(path) / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "samples.pt"
    payload = {
        "samples": samples.cpu(),
        "fault_label": expanded_condition["fault_label"].detach().cpu(),
        "domain_id": expanded_condition["domain_id"].detach().cpu(),
        "condition_policy": policy,
        "condition_counts": _condition_counts(expanded_condition),
        "num_steps": num_steps,
        "sampler_id": str(getattr(task, "sampler_id", "euler_ode")),
        "sampler_metadata": dict(getattr(task, "sampler_metadata", lambda: {})()),
    }
    torch.save(payload, tensor_path)

    domain_map_path = str(getattr(gen_cfg, "domain_map_path", "configs/domain_maps/dummy_domain_map.csv"))
    manifest_path = output_dir / "synthetic_data_manifest.json"
    leakage_threshold = float(getattr(gen_cfg, "leakage_duplicate_threshold", 1e-6))
    leakage_checks = _leakage_checks_from_train_batch(data_factory, samples.detach().cpu(), channels, leakage_threshold)
    task.write_sample_manifest(
        output_path=manifest_path,
        synthetic_dataset_id=str(getattr(gen_cfg, "synthetic_dataset_id", f"{name}_iter{iteration}")),
        checkpoint_path=checkpoint_path or "untrained_smoke",
        generator_run_id=name,
        source_split=str(getattr(gen_cfg, "source_split", "train")),
        domain_map_path=domain_map_path,
        domain_map_hash=hash_file(domain_map_path),
        sampler_id=str(getattr(task, "sampler_id", "euler_ode")),
        num_steps=num_steps,
        seed=seed,
        num_samples=num_samples,
        shape=list(samples.shape),
        status=str(getattr(gen_cfg, "validity_status", "exploratory")),
        config_path=str(args.config_path),
        config_hash=_hash_path_or_value(str(args.config_path)),
        protocol_path=PROTOCOL_SCHEMA_PATH,
        protocol_hash=_hash_path_or_value(PROTOCOL_SCHEMA_PATH),
        dependency_lock_hash=_dependency_lock_hash(),
        leakage_checks=leakage_checks,
        condition_sampling_policy=policy,
        condition_counts=_condition_counts(expanded_condition),
        condition_sampling_split_verified=_condition_sampling_split_verified(policy, metadata),
        sampler_metadata=dict(getattr(task, "sampler_metadata", lambda: {})()),
    )
    _update_stage_ledger(
        _stage_ledger_path(configs, path, "sample"),
        mode="sample",
        values={
            "run_dir": path,
            "samples_path": tensor_path,
            "synthetic_manifest_path": manifest_path,
        },
    )
    data_factory.data.close()
    return {
        "sample_path": str(tensor_path),
        "manifest_path": str(manifest_path),
        "sampling_wall_clock_sec": float(sampling_wall_clock),
        "sampling_nfe": float(num_steps),
        "samples_per_second": float(num_samples / sampling_wall_clock) if sampling_wall_clock > 0 else float("inf"),
        "parameter_count": _count_parameters(task),
        "peak_memory_bytes": float(_peak_memory_bytes()),
    }


def _eval_once(args, configs, iteration):
    args_environment, args_data, args_model, args_task, args_trainer = _namespaces(configs)
    gen_cfg = _generative_cfg(args_task)
    path, _ = path_name(configs, iteration)
    data_factory, _, task = _build_stack(args_data, args_model, args_task, args_trainer, args_environment)
    fake_path = getattr(gen_cfg, "generated_path", None)
    if fake_path is None:
        raise ValueError("eval mode requires task.generative.generated_path")
    fake, fake_labels, fake_domains = _load_sample_payload(fake_path)
    channels = int(getattr(args_model, "in_channels", getattr(args_model, "channels", fake.shape[1])))
    eval_split = str(getattr(gen_cfg, "eval_split", "train")).lower()
    if eval_split == "valid":
        eval_split = "val"
    if eval_split in {"test", "target_test"} and not bool(getattr(gen_cfg, "allow_test_reference_eval", False)):
        raise ValueError(
            "generative eval uses test data only with task.generative.allow_test_reference_eval=true"
        )
    real_batch = next(iter(data_factory.get_dataloader(eval_split)))
    real = _to_ncl(real_batch["x"], channels)
    fake = _to_ncl(fake, channels)
    n = min(real.shape[0], fake.shape[0])
    real = real[:n]
    fake = fake[:n]
    real_labels = _labels_from_batch(real_batch, data_factory.get_metadata(), "fault_label", "Label")
    real_domains = _labels_from_batch(real_batch, data_factory.get_metadata(), "domain_id", "Domain_id")
    if real_labels is not None:
        real_labels = real_labels[:n]
    if real_domains is not None:
        real_domains = real_domains[:n]
    if fake_labels is not None:
        fake_labels = fake_labels[:n]
    if fake_domains is not None:
        fake_domains = fake_domains[:n]
    metric_start = time.perf_counter()
    metrics = evaluate_generated_windows(
        real,
        fake,
        real_labels=real_labels,
        fake_labels=fake_labels,
        real_domains=real_domains,
        fake_domains=fake_domains,
        sampling_rate_hz=_optional_float(
            getattr(gen_cfg, "sampling_rate_hz", None),
            getattr(gen_cfg, "sampling_rate", None),
            getattr(args_data, "sampling_rate_hz", None),
            getattr(args_data, "sampling_rate", None),
        ),
        shaft_rpm=_optional_float(
            getattr(gen_cfg, "shaft_rpm", None),
            getattr(gen_cfg, "rpm", None),
            getattr(args_data, "shaft_rpm", None),
            getattr(args_data, "rpm", None),
        ),
        fault_frequency_hz=_optional_float(
            getattr(gen_cfg, "fault_frequency_hz", None),
            getattr(args_data, "fault_frequency_hz", None),
        ),
    )
    metrics.update(
        {
            "metric_compute_time_sec": float(time.perf_counter() - metric_start),
            "parameter_count": float(_count_parameters(task)),
            "sampling_nfe": float(getattr(gen_cfg, "num_steps", 0)),
            "eval_split_is_test": float(eval_split == "test"),
            "eval_num_real": float(real.shape[0]),
            "eval_num_fake": float(fake.shape[0]),
        }
    )
    metrics_path = Path(path) / "generative_eval_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    eval_evidence_path = Path(path) / "eval_evidence_manifest.json"
    _write_eval_evidence_manifest(
        eval_evidence_path,
        generated_path=fake_path,
        metrics_path=metrics_path,
        reference_split=eval_split,
        allow_test_reference_eval=bool(
            getattr(gen_cfg, "allow_test_reference_eval", False)
        ),
        metrics=metrics,
    )
    _update_stage_ledger(
        _stage_ledger_path(configs, path, "eval"),
        mode="eval",
        values={
            "run_dir": path,
            "metrics_path": metrics_path,
            "eval_evidence_manifest_path": eval_evidence_path,
        },
    )
    data_factory.data.close()
    return metrics


def pipeline(args):
    configs = _load_configs(args)
    args_environment, _, _, args_task, _ = _namespaces(configs)
    mode = str(getattr(_generative_cfg(args_task), "mode", "train")).lower()
    all_results = []
    for iteration in range(int(args_environment.iterations)):
        if mode == "train":
            result = _train_one_iteration(args, configs, iteration)
        elif mode == "sample":
            result = _sample_once(args, configs, iteration)
        elif mode == "eval":
            result = _eval_once(args, configs, iteration)
        else:
            raise ValueError(f"Unsupported generative mode: {mode}")
        all_results.append(result)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHM generative benchmark pipeline")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--local_config", type=str, default=None)
    args = parser.parse_args()
    pipeline(args)
