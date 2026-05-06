import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from src.configs.config_utils import merge_with_local_override, path_name, transfer_namespace
from src.data_factory import build_data
from src.data_factory.ID.domain_map import hash_file
from src.model_factory import build_model
from src.task_factory import build_task
from src.task_factory.task.generative.generative_eval import evaluate_generated_windows
from src.utils.config_utils import apply_overrides_to_config, parse_overrides
from src.utils.utils import safe_torch_load


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

    trainer.fit(task, data_factory.get_dataloader("train"), data_factory.get_dataloader("val"))
    task = load_best_model_checkpoint(task, trainer)
    result = trainer.test(task, data_factory.get_dataloader("test"))
    data_factory.data.close()
    close_lab()

    result_row = result[0] if result else {}
    pd.DataFrame([result_row]).to_csv(os.path.join(path, f"test_result_{iteration}.csv"), index=False)
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
    samples = task.sample(
        condition,
        num_samples=num_samples,
        length=length,
        channels=channels,
        num_steps=num_steps,
        device=task.device,
    )

    output_dir = Path(path) / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "samples.pt"
    torch.save(samples.cpu(), tensor_path)

    domain_map_path = str(getattr(gen_cfg, "domain_map_path", "configs/domain_maps/dummy_domain_map.csv"))
    manifest_path = output_dir / "synthetic_data_manifest.json"
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
    )
    data_factory.data.close()
    return {"sample_path": str(tensor_path), "manifest_path": str(manifest_path)}


def _eval_once(args, configs, iteration):
    args_environment, args_data, args_model, args_task, args_trainer = _namespaces(configs)
    gen_cfg = _generative_cfg(args_task)
    path, _ = path_name(configs, iteration)
    data_factory, _, _ = _build_stack(args_data, args_model, args_task, args_trainer, args_environment)
    fake_path = getattr(gen_cfg, "generated_path", None)
    if fake_path is None:
        raise ValueError("eval mode requires task.generative.generated_path")
    fake = safe_torch_load(str(fake_path), map_location="cpu")
    real_batch = next(iter(data_factory.get_dataloader("test")))
    real = torch.as_tensor(real_batch["x"]).float()
    if real.ndim == 3 and real.shape[1] != fake.shape[1]:
        real = real.transpose(1, 2).contiguous()
    real = real[: fake.shape[0]]
    metrics = evaluate_generated_windows(real, fake[: real.shape[0]])
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
