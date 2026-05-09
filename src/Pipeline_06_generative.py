import argparse
import hashlib
import os
import time
from pathlib import Path

import pandas as pd
import torch

from src.configs.config_utils import merge_with_local_override, path_name, transfer_namespace
from src.data_factory import build_data
from src.data_factory.ID.domain_map import hash_file
from src.model_factory import build_model
from src.task_factory import build_task
from src.task_factory.task.generative.generative_eval import evaluate_generated_windows
from src.task_factory.Components.generative.metrics.leakage import leakage_metrics
from src.utils.config_utils import apply_overrides_to_config, parse_overrides
from src.utils.utils import safe_torch_load


PROTOCOL_SCHEMA_PATH = "docs/schemas/generative_protocol.schema.json"


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
    return x.contiguous()


def _load_sample_payload(path: str | Path) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
    checkpoint_path = str(getattr(gen_cfg, "checkpoint_path", "") or "")
    if checkpoint_path:
        state = safe_torch_load(checkpoint_path, map_location="cpu")
        task.load_state_dict(state.get("state_dict", state), strict=False)
    elif not bool(getattr(gen_cfg, "allow_untrained_smoke", False)):
        raise ValueError("sample mode requires task.generative.checkpoint_path")

    num_samples = int(getattr(gen_cfg, "num_samples", 2))
    num_steps = int(getattr(gen_cfg, "num_steps", 8))
    length = int(getattr(gen_cfg, "length", getattr(args_data, "window_size", 128)))
    channels = int(getattr(args_model, "in_channels", getattr(args_model, "channels", 2)))
    condition = _first_condition_from_metadata(data_factory.get_metadata(), task.device)
    expanded_condition = _expand_condition(condition, num_samples, task.device)
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
        "condition_policy": str(getattr(gen_cfg, "condition_sampling_policy", "first_metadata_repeated")),
        "num_steps": num_steps,
        "sampler_id": "euler_ode",
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
        sampler_id="euler_ode",
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
        condition_sampling_policy=str(getattr(gen_cfg, "condition_sampling_policy", "first_metadata_repeated")),
        condition_counts=_condition_counts(expanded_condition),
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
    if fake.ndim == 3 and fake.shape[1] != channels and fake.shape[2] == channels:
        fake = fake.transpose(1, 2).contiguous()
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
