import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.configs.config_utils import load_config, path_name, transfer_namespace, merge_with_local_override, save_config
from src.utils.config_utils import parse_overrides, apply_overrides_to_config
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab, get_num_classes
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer
from src.utils.evaluation_artifacts import export_classification_artifacts
from src.utils.generative_evidence import (
    dependency_lock_evidence,
    git_commit_sha,
    runtime_environment,
    sha256_file,
)

P01_PROTOCOL_ID = "P01-G040-v1"
P01_REGISTERED_ARMS = {
    "FULL",
    "B1-1D",
    "B2-2D",
    "B3-CONCAT",
    "B4-GATTN",
    "B5-NCE",
    "TRAIN-MISPAIR",
    "A-NO-ALIGN",
    "A-NO-PRIVATE-IND",
    "A-NO-REC",
    "A-NO-VAR",
    "A-SHARED-ONLY",
    "S-SHARED-ONLY-CAPACITY",
}
P01_BASELINE_MODEL_CONTRACTS = {
    "B1-1D": ("one_d", 112, 384),
    "B2-2D": ("two_d", 128, 32),
    "B3-CONCAT": ("concat", 80, 192),
    "B4-GATTN": ("generic_attention", 72, 256),
    "B5-NCE": ("contrastive", 56, 384),
}
P01_TRAINABLE_PARAMETER_COUNTS = {
    "CWRU": {
        "FULL": 78596,
        "B1-1D": 77580,
        "B2-2D": 79716,
        "B3-CONCAT": 78484,
        "B4-GATTN": 78828,
        "B5-NCE": 78444,
        "TRAIN-MISPAIR": 78596,
        "A-NO-ALIGN": 78596,
        "A-NO-PRIVATE-IND": 78596,
        "A-NO-REC": 78596,
        "A-NO-VAR": 78596,
        "A-SHARED-ONLY": 45252,
        "S-SHARED-ONLY-CAPACITY": 78589,
    },
    "XJTU": {
        "FULL": 78466,
        "B1-1D": 76810,
        "B2-2D": 79650,
        "B3-CONCAT": 78098,
        "B4-GATTN": 78314,
        "B5-NCE": 77674,
        "TRAIN-MISPAIR": 78466,
        "A-NO-ALIGN": 78466,
        "A-NO-PRIVATE-IND": 78466,
        "A-NO-REC": 78466,
        "A-NO-VAR": 78466,
        "A-SHARED-ONLY": 45122,
        "S-SHARED-ONLY-CAPACITY": 78477,
    },
}
P01_EMPTY_LOCAL_CONFIG = Path("configs/experiments/p01/no_local_override.yaml")
P01_EMPTY_LOCAL_CONFIG_SHA256 = (
    "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
)


def _git_state(repository_root: str | Path = ".") -> dict:
    """Return hashes of the local Git state without embedding patch contents."""

    def capture(arguments):
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=str(repository_root),
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout if result.returncode == 0 else None

    status = capture(["status", "--porcelain=v1", "--untracked-files=normal"])
    patch = capture(["diff", "--binary", "HEAD"])
    return {
        "commit": git_commit_sha(repository_root),
        "dirty": None if status is None else bool(status.strip()),
        "status_sha256": (
            "unavailable" if status is None else hashlib.sha256(status).hexdigest()
        ),
        "tracked_patch_sha256": (
            "unavailable" if patch is None else hashlib.sha256(patch).hexdigest()
        ),
    }


def _code_file_hashes(configs) -> dict:
    repository_root = Path.cwd().resolve()
    candidates = [
        Path(__file__),
        Path("phmfactory/cli.py"),
        Path("src/configs/config_utils.py"),
        Path("src/utils/evaluation_artifacts.py"),
        Path("src/utils/utils.py"),
        Path("src/data_factory/data_factory.py"),
        Path("src/data_factory/grouped_split.py"),
        Path("src/data_factory/ID/Id_searcher.py"),
        Path("src/data_factory/dataset_task/Default_dataset.py"),
        Path("src/data_factory/dataset_task/Dataset_cluster.py"),
        Path("src/data_factory/samplers/Get_sampler.py"),
        Path("src/data_factory/samplers/Sampler.py"),
        Path("src/task_factory/Default_task.py"),
        Path("src/trainer_factory/Default_trainer.py"),
        Path("src/model_factory/X_model/UXFD/signal_processing_2d/stft_tfr.py"),
        Path("src/data_factory/reader/RM_001_CWRU.py"),
        Path("src/data_factory/reader/RM_002_XJTU.py"),
    ]
    model_type = str(getattr(configs.model, "type", ""))
    model_name = str(getattr(configs.model, "name", ""))
    if model_type and model_name:
        candidates.append(Path("src/model_factory") / model_type / f"{model_name}.py")
    hashes = {}
    for path in candidates:
        resolved = path.resolve()
        if not resolved.is_file():
            continue
        try:
            key = str(resolved.relative_to(repository_root))
        except ValueError:
            key = str(resolved)
        hashes[key] = sha256_file(resolved)
    return dict(sorted(hashes.items()))


def _paper_fields(configs) -> dict:
    if not hasattr(configs, "paper"):
        return {}
    names = (
        "paper_id", "protocol_id", "dataset_key", "dataset_slug", "dataset_id",
        "arm_id", "attempt_id", "method_id",
        "baseline_id", "ablation_id", "supports_claim_ids", "evidence_status",
    )
    return {
        name: getattr(configs.paper, name)
        for name in names
        if hasattr(configs.paper, name)
    }


def _validate_p01_model_configuration(configs, arm_id: str, dataset_key: str) -> None:
    """Reject arm/model substitutions before an evidence model is constructed."""

    model = configs.model
    common_checks = {
        "model.type": (getattr(model, "type", None), "X_model"),
        "model.in_channels": (getattr(model, "in_channels", None), 2),
        "model.dropout": (getattr(model, "dropout", None), 0.1),
        "model.time_frequency.n_fft": (
            getattr(getattr(model, "time_frequency", None), "n_fft", None),
            128,
        ),
        "model.time_frequency.hop_length": (
            getattr(getattr(model, "time_frequency", None), "hop_length", None),
            32,
        ),
        "model.time_frequency.center": (
            getattr(getattr(model, "time_frequency", None), "center", None),
            True,
        ),
        "model.time_frequency.normalized": (
            getattr(getattr(model, "time_frequency", None), "normalized", None),
            False,
        ),
    }
    if arm_id in P01_BASELINE_MODEL_CONTRACTS:
        variant, encoder_dim, head_hidden = P01_BASELINE_MODEL_CONTRACTS[arm_id]
        arm_checks = {
            "model.name": (getattr(model, "name", None), "P01Baselines"),
            "model.variant": (getattr(model, "variant", None), variant),
            "model.encoder_dim": (getattr(model, "encoder_dim", None), encoder_dim),
            "model.head_hidden": (getattr(model, "head_hidden", None), head_hidden),
            "model.projection_dim": (getattr(model, "projection_dim", None), 32),
            "model.contrastive_temperature": (
                getattr(model, "contrastive_temperature", None),
                0.1,
            ),
        }
    else:
        arm_checks = {
            "model.name": (getattr(model, "name", None), "P01SharedPrivate"),
            "model.encoder_dim": (getattr(model, "encoder_dim", None), 64),
            "model.latent_dim": (getattr(model, "latent_dim", None), 32),
            "model.objective.variance_floor": (
                getattr(getattr(model, "objective", None), "variance_floor", None),
                0.1,
            ),
            "model.pairing.mode": (
                getattr(getattr(model, "pairing", None), "mode", None),
                "paired",
            ),
        }
        shared_only = arm_id in {"A-SHARED-ONLY", "S-SHARED-ONLY-CAPACITY"}
        ablation = getattr(model, "ablation", None)
        if shared_only:
            expected_hidden = (
                64
                if arm_id == "A-SHARED-ONLY"
                else {"CWRU": 965, "XJTU": 1017}[dataset_key]
            )
            arm_checks.update(
                {
                    "model.ablation.private_branch_enabled": (
                        getattr(ablation, "private_branch_enabled", None),
                        False,
                    ),
                    "model.ablation.shared_only_head_hidden": (
                        getattr(ablation, "shared_only_head_hidden", None),
                        expected_hidden,
                    ),
                }
            )
        elif ablation is not None and getattr(
            ablation, "private_branch_enabled", True
        ) is not True:
            arm_checks["model.ablation.private_branch_enabled"] = (
                getattr(ablation, "private_branch_enabled", None),
                True,
            )

    mismatches = [
        f"{name}={observed!r} (expected {expected!r})"
        for name, (observed, expected) in {**common_checks, **arm_checks}.items()
        if observed != expected
    ]
    if mismatches:
        raise RuntimeError("P01 evidence model contract drift: " + "; ".join(mismatches))

    weights = getattr(configs.task, "auxiliary_loss_weights", None)
    observed_weights = {} if weights is None else dict(vars(weights))
    if arm_id == "B5-NCE":
        expected_weights = {"contrastive_alignment": 0.1}
    elif arm_id in P01_BASELINE_MODEL_CONTRACTS:
        expected_weights = {}
    elif arm_id in {"A-SHARED-ONLY", "S-SHARED-ONLY-CAPACITY"}:
        expected_weights = {"alignment": 0.1, "shared_variance": 0.1}
    else:
        expected_weights = {
            "alignment": 0.1,
            "private_independence": 0.01,
            "reconstruction": 0.1,
            "shared_variance": 0.1,
        }
        zeroed_weight_by_arm = {
            "A-NO-ALIGN": "alignment",
            "A-NO-PRIVATE-IND": "private_independence",
            "A-NO-REC": "reconstruction",
            "A-NO-VAR": "shared_variance",
        }
        if arm_id in zeroed_weight_by_arm:
            expected_weights[zeroed_weight_by_arm[arm_id]] = 0.0
    if observed_weights != expected_weights:
        raise RuntimeError(
            "P01 evidence auxiliary-loss contract drift: "
            f"observed={observed_weights!r}, expected={expected_weights!r}"
        )

    pairing = getattr(configs.data, "pairing", None)
    observed_pairing = {} if pairing is None else dict(vars(pairing))
    if arm_id == "TRAIN-MISPAIR":
        fold = int(getattr(configs.data.split, "outer_fold"))
        dataset_slug = {"CWRU": "cwru", "XJTU": "xjtu"}[dataset_key]
        group_key = {"CWRU": "File", "XJTU": "FileParent"}[dataset_key]
        expected_pairing = {
            "mode": "frozen_within_group_class_derangement",
            "seed": 20260801,
            "splits": ["train"],
            "protocol_id": P01_PROTOCOL_ID,
            "group_key": group_key,
            "manifest_dir": (
                f"results/p01/{P01_PROTOCOL_ID}/protocol/pairing/"
                f"{dataset_slug}/fold_{fold}"
            ),
        }
    else:
        expected_pairing = {"mode": "paired"}
    if observed_pairing != expected_pairing:
        raise RuntimeError(
            "P01 evidence pairing contract drift: "
            f"observed={observed_pairing!r}, expected={expected_pairing!r}"
        )


def _validate_p01_trainable_parameter_count(configs, model) -> None:
    """Bind the instantiated model to the parameter count frozen per arm/dataset."""

    paper = _paper_fields(configs)
    if paper.get("paper_id") != "P01" or paper.get("evidence_status") != "candidate_unreviewed":
        return
    dataset_key = str(paper["dataset_key"])
    arm_id = str(paper["arm_id"])
    expected = P01_TRAINABLE_PARAMETER_COUNTS[dataset_key][arm_id]
    observed = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    if observed != expected:
        raise RuntimeError(
            "P01 evidence trainable-parameter count drift: "
            f"arm={arm_id}, dataset={dataset_key}, observed={observed}, expected={expected}"
        )


def _validate_evidence_runtime(configs, args) -> None:
    """Fail closed on the physical-GPU and local-override protocol contract."""

    paper = _paper_fields(configs)
    if paper.get("paper_id") != "P01" or paper.get("evidence_status") != "candidate_unreviewed":
        return
    required_paper_fields = (
        "protocol_id", "dataset_key", "dataset_slug", "dataset_id", "arm_id",
        "attempt_id",
    )
    missing_paper_fields = [
        name for name in required_paper_fields if not str(paper.get(name, "")).strip()
    ]
    if missing_paper_fields:
        raise RuntimeError(
            "P01 evidence requires paper fields: " + ", ".join(missing_paper_fields)
        )
    if paper["protocol_id"] != P01_PROTOCOL_ID:
        raise RuntimeError(
            f"P01 evidence protocol must be {P01_PROTOCOL_ID}, got {paper['protocol_id']}"
        )
    dataset_key = str(paper["dataset_key"])
    dataset_slug = str(paper["dataset_slug"])
    dataset_id = int(paper["dataset_id"])
    arm_id = str(paper["arm_id"])
    attempt_id = paper["attempt_id"]
    if (
        isinstance(attempt_id, bool)
        or not isinstance(attempt_id, int)
        or attempt_id not in {0, 1}
    ):
        raise RuntimeError("P01 evidence paper.attempt_id must be integer 0 or 1")
    expected_dataset_binding = {
        "CWRU": ("cwru", 1),
        "XJTU": ("xjtu", 2),
    }
    if (
        dataset_key not in expected_dataset_binding
        or expected_dataset_binding[dataset_key] != (dataset_slug, dataset_id)
        or arm_id not in P01_REGISTERED_ARMS
    ):
        raise RuntimeError(
            "P01 evidence has an invalid dataset binding or unregistered paper.arm_id"
        )
    _validate_p01_model_configuration(configs, arm_id, dataset_key)
    expected_target = [dataset_id]
    if list(getattr(configs.task, "target_system_id", [])) != expected_target:
        raise RuntimeError(
            "paper.dataset_id does not match task.target_system_id"
        )
    if arm_id.startswith("B") and str(paper.get("baseline_id", "")) != arm_id:
        raise RuntimeError("Baseline evidence requires paper.baseline_id=paper.arm_id")
    if (
        arm_id.startswith(("A-", "S-")) or arm_id == "TRAIN-MISPAIR"
    ) and str(paper.get("ablation_id", "")) != arm_id:
        raise RuntimeError("Ablation evidence requires paper.ablation_id=paper.arm_id")

    split = getattr(configs.data, "split", None)
    expected_dataset = {
        "CWRU": {
            "label_policy": "native",
            "group_key": "File",
            "stratify_key": "Label",
            "outer_folds": 4,
        },
        "XJTU": {
            "label_policy": "binary_fault",
            "group_key": "FileParent",
            "stratify_key": "Domain_id",
            "outer_folds": 5,
        },
    }[dataset_key]
    checks = {
        "environment.iterations": (getattr(configs.environment, "iterations", None), 1),
        "data.batch_size": (getattr(configs.data, "batch_size", None), 64),
        "data.window_size": (getattr(configs.data, "window_size", None), 4096),
        "data.num_window": (getattr(configs.data, "num_window", None), 64),
        "data.window_sampling_strategy": (
            getattr(configs.data, "window_sampling_strategy", None),
            "evenly_spaced",
        ),
        "data.normalization": (
            getattr(configs.data, "normalization", None),
            "standardization",
        ),
        "data.dtype": (getattr(configs.data, "dtype", None), "float32"),
        "data.read_only_cache_required": (
            getattr(configs.data, "read_only_cache_required", None),
            True,
        ),
        "data.split.strategy": (getattr(split, "strategy", None), "grouped_kfold"),
        "data.split.group_key": (
            getattr(split, "group_key", None),
            expected_dataset["group_key"],
        ),
        "data.split.stratify_key": (
            getattr(split, "stratify_key", None),
            expected_dataset["stratify_key"],
        ),
        "data.split.seed": (getattr(split, "seed", None), 20260801),
        "data.split.outer_folds": (
            getattr(split, "outer_folds", None),
            expected_dataset["outer_folds"],
        ),
        "data.split.validation_offset": (
            getattr(split, "validation_offset", None),
            1,
        ),
        "task.label_policy": (
            getattr(configs.task, "label_policy", None),
            expected_dataset["label_policy"],
        ),
        "task.batch_size": (getattr(configs.task, "batch_size", None), 64),
        "task.optimizer": (getattr(configs.task, "optimizer", None), "adam"),
        "task.lr": (getattr(configs.task, "lr", None), 0.001),
        "task.weight_decay": (getattr(configs.task, "weight_decay", None), 0.0001),
        "task.export_predictions": (
            getattr(configs.task, "export_predictions", None),
            True,
        ),
        "trainer.monitor": (getattr(configs.trainer, "monitor", None), "val_loss"),
        "trainer.num_epochs": (getattr(configs.trainer, "num_epochs", None), 50),
        "trainer.early_stopping": (
            getattr(configs.trainer, "early_stopping", None),
            True,
        ),
        "trainer.patience": (getattr(configs.trainer, "patience", None), 10),
        "model.in_channels": (getattr(configs.model, "in_channels", None), 2),
    }
    mismatches = [
        f"{name}={observed!r} (expected {expected!r})"
        for name, (observed, expected) in checks.items()
        if observed != expected
    ]
    if mismatches:
        raise RuntimeError("P01 evidence protocol drift: " + "; ".join(mismatches))
    data_root = Path(str(getattr(configs.data, "data_dir", "")))
    metadata_path = data_root / str(getattr(configs.data, "metadata_file", ""))
    cache_path = data_root / "cache.h5"
    if not metadata_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError(
            "P01 evidence requires pre-existing metadata and read-only cache; "
            "automatic download/cache creation is forbidden"
        )

    full_representations = {
        "shared_1d", "shared_2d", "private_1d", "private_2d",
        "reconstructed_1d", "reconstructed_2d",
    }
    required_by_arm = {
        "B1-1D": {"encoded_1d"},
        "B2-2D": {"encoded_2d"},
        "B3-CONCAT": {"encoded_1d", "encoded_2d"},
        "B4-GATTN": {"encoded_1d", "encoded_2d"},
        "B5-NCE": {"encoded_1d", "encoded_2d"},
        "A-SHARED-ONLY": {"shared_1d", "shared_2d"},
        "S-SHARED-ONLY-CAPACITY": {"shared_1d", "shared_2d"},
    }
    expected_representations = required_by_arm.get(arm_id, full_representations)
    observed_representations = set(
        getattr(configs.task, "required_representation_arrays", [])
    )
    if observed_representations != expected_representations:
        raise RuntimeError(
            "P01 evidence representation contract drift: "
            f"observed={sorted(observed_representations)}, "
            f"expected={sorted(expected_representations)}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    if len(devices) != 1 or not devices[0].isdigit():
        raise RuntimeError(
            "P01 evidence requires one explicit numeric CUDA_VISIBLE_DEVICES index"
        )
    if int(devices[0]) == 2:
        raise RuntimeError("Physical GPU index 2 is forbidden for P01 evidence")
    if int(getattr(configs.trainer, "gpus", 0)) != 1:
        raise RuntimeError("P01 evidence requires trainer.gpus=1")
    if str(getattr(configs.trainer, "device", "")) != "cuda":
        raise RuntimeError("P01 evidence requires trainer.device=cuda")
    local_config = getattr(args, "local_config", None)
    canonical_local = P01_EMPTY_LOCAL_CONFIG.resolve()
    candidate_local = Path(str(local_config)).resolve() if local_config else None
    if candidate_local != canonical_local or not canonical_local.is_file():
        raise RuntimeError(
            "P01 evidence requires the canonical empty no_local_override.yaml"
        )
    if sha256_file(canonical_local) != P01_EMPTY_LOCAL_CONFIG_SHA256:
        raise RuntimeError("Canonical P01 local override hash has drifted")
    if yaml.safe_load(canonical_local.read_text(encoding="utf-8")) != {}:
        raise RuntimeError("Canonical P01 local override must resolve to an empty mapping")

    fold = getattr(split, "outer_fold", None)
    seed = getattr(configs.environment, "seed", None)
    output_dir = str(getattr(configs.environment, "output_dir", ""))
    expected_suffix = (
        f"/{dataset_slug}/{arm_id}/fold_{fold}/seed_{seed}/attempt_{attempt_id}"
    ).lower()
    normalized_output = "/" + output_dir.replace("\\", "/").strip("/")
    if not normalized_output.lower().endswith(expected_suffix):
        raise RuntimeError(
            "P01 evidence output_dir must bind dataset/arm/fold/seed/attempt: "
            + expected_suffix
        )


def _best_checkpoint_provenance(trainer) -> dict:
    checkpoints = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint)
    ]
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one ModelCheckpoint callback, found {len(checkpoints)}"
        )
    callback = checkpoints[0]
    checkpoint_path = Path(str(callback.best_model_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint is missing after training: {checkpoint_path}"
        )
    score = callback.best_model_score
    if isinstance(score, torch.Tensor):
        score_value = float(score.detach().cpu().item())
    elif score is None:
        score_value = None
    else:
        score_value = float(score)
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "monitor": str(callback.monitor),
        "mode": str(callback.mode),
        "score": score_value,
    }


def _write_trainer_metrics_manifest(trainer, run_path: str | Path) -> dict:
    """Freeze the one canonical CSV trainer log and return provenance fields."""

    loggers = list(getattr(trainer, "loggers", []) or [])
    if not loggers:
        primary_logger = getattr(trainer, "logger", None)
        if primary_logger is not None:
            loggers = [primary_logger]
    csv_loggers = [logger for logger in loggers if isinstance(logger, CSVLogger)]
    if len(csv_loggers) != 1:
        raise RuntimeError(
            f"Expected exactly one CSVLogger for trainer metrics, found {len(csv_loggers)}"
        )
    logger = csv_loggers[0]
    if isinstance(logger.version, bool) or int(logger.version) != 0:
        raise RuntimeError(
            f"P01 trainer CSVLogger version must be 0, got {logger.version!r}"
        )
    logger.save()

    expected_log_dir = (Path(run_path) / "logs" / "version_0").resolve()
    observed_log_dir = Path(str(logger.log_dir)).resolve()
    if observed_log_dir != expected_log_dir:
        raise RuntimeError(
            "P01 trainer metrics log directory drift: "
            f"observed={observed_log_dir}, expected={expected_log_dir}"
        )
    metrics_path = observed_log_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Trainer metrics CSV is missing: {metrics_path}")

    payload = {
        "schema_version": 1,
        "logger_type": "CSVLogger",
        "logger_name": str(logger.name),
        "logger_version": 0,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    hparams_path = observed_log_dir / "hparams.yaml"
    if hparams_path.is_file():
        payload.update(
            {
                "hparams_path": str(hparams_path),
                "hparams_sha256": sha256_file(hparams_path),
            }
        )

    manifest_path = Path(run_path) / "artifacts" / "trainer_metrics.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return {
        "trainer_metrics_manifest_path": str(manifest_path.resolve()),
        "trainer_metrics_manifest_sha256": sha256_file(manifest_path),
        "trainer_metrics_path": str(metrics_path),
        "trainer_metrics_sha256": payload["metrics_sha256"],
    }



def pipeline(args):
    """领域泛化(Domain Generalization)任务的流水线
    
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
    # 支持机器特定的本地覆盖 YAML（方案B）
    # 优先顺序：命令行 --local_config > configs/local/{hostname}.yaml > configs/local/local.yaml > configs/local/default.yaml
    configs = merge_with_local_override(config_path, getattr(args, 'local_config', None))

    # 应用CLI override参数（最高优先级）
    if hasattr(args, 'override') and args.override:
        print(f"[INFO] 应用CLI override参数: {args.override}")
        overrides = parse_overrides(args.override)
        configs = apply_overrides_to_config(configs, overrides)
        print(f"[INFO] 已应用 {len(overrides)} 个override参数")

    _validate_evidence_runtime(configs, args)

    # 确保配置中包含必要的部分
    required_sections = ['data', 'model', 'task', 'trainer', 'environment']
    for section in required_sections:
        if not hasattr(configs, section):
            print(f"[ERROR] 配置文件中缺少 {section} 部分")
            return
    
    # 设置环境变量和命名空间
    args_environment = transfer_namespace(configs.environment if hasattr(configs, 'environment') else {})

    args_data = transfer_namespace(configs.data if hasattr(configs, 'data') else {})

    args_model = transfer_namespace(configs.model if hasattr(configs, 'model') else {})

    args_task = transfer_namespace(configs.task if hasattr(configs, 'task') else {})

    args_trainer = transfer_namespace(configs.trainer if hasattr(configs, 'trainer') else {})
    if hasattr(configs, 'paper'):
        for paper_field in (
            'paper_id', 'protocol_id', 'dataset_key', 'dataset_slug',
            'dataset_id', 'arm_id', 'attempt_id',
            'method_id', 'baseline_id', 'ablation_id',
            'supports_claim_ids', 'evidence_status'
        ):
            if hasattr(configs.paper, paper_field):
                setattr(args_trainer, paper_field, getattr(configs.paper, paper_field))
    if args_task.name == 'Multitask':
        args_data.task_list = args_task.task_list
        args_model.task_list = args_task.task_list    
    for key, value in configs.environment.__dict__.items():
        if key.isupper():
            os.environ[key] = str(value)
            print(f"[INFO] 设置环境变量: {key}={value}")

    # Capture source state before this run creates any result directories.
    launch_git_state = _git_state()
    launch_code_files = _code_file_hashes(configs)
    launch_code_state_sha256 = hashlib.sha256(
        json.dumps(
            launch_code_files,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
        ).encode('utf-8')
    ).hexdigest()

    # 创建实验目录（依赖 environment.output_dir / path_name，不再强制依赖 VBENCH_* 变量）
    print("[INFO] 创建实验目录...")
    
    # -----------------------
    # 2. 多次迭代训练与测试
    # -----------------------
    all_results = []
    
    for it in range(args_environment.iterations):
        print(f"\n{'='*50}\n[INFO] 开始实验迭代 {it+1}/{args_environment.iterations}\n{'='*50}")
        
        # 设置路径、名称和随机种子
        current_seed = args_environment.seed + it
        path, name = path_name(configs, it)
        paper_fields_for_run = _paper_fields(configs)
        is_p01_evidence = (
            paper_fields_for_run.get('paper_id') == 'P01'
            and paper_fields_for_run.get('evidence_status') == 'candidate_unreviewed'
        )
        if is_p01_evidence:
            # The approved P01 path is already unique by dataset/arm/fold/seed;
            # do not append legacy metadata/model/task path components.
            path = str(Path(str(args_environment.output_dir)))
            name = (
                f"{paper_fields_for_run['dataset_slug']}_"
                f"{paper_fields_for_run['arm_id']}_"
                f"f{getattr(args_data.split, 'outer_fold')}_s{current_seed}_"
                f"a{paper_fields_for_run['attempt_id']}"
            )
            if Path(path).exists():
                raise FileExistsError(
                    f"Refusing to reuse P01 evidence output directory: {path}"
                )
        # 把name 加到args_trainer中
        args_trainer.logger_name = name
        # 设置随机种子
        seed_everything(current_seed)
        print(f"[INFO] 设置随机种子: {current_seed}")
        init_lab(args_environment, args, name)
        os.makedirs(path, exist_ok=True)
        config_snapshot_path = Path(path) / 'config_snapshot.yaml'
        save_config(configs, config_snapshot_path)
        config_snapshot_sha256 = sha256_file(config_snapshot_path)
        invocation = {
            'schema_version': 1,
            'argv': list(sys.argv),
            'config_path': str(args.config_path),
            'config_source_sha256': sha256_file(args.config_path),
            'requested_config': str(getattr(args, 'requested_config', args.config_path)),
            'resolved_config_path': str(getattr(args, 'resolved_config_path', args.config_path)),
            'local_config': str(getattr(args, 'local_config', None)),
            'local_config_sha256': (
                None
                if not getattr(args, 'local_config', None)
                else sha256_file(getattr(args, 'local_config'))
            ),
            'overrides': list(getattr(args, 'override', []) or []),
            'iteration': it,
            'effective_seed': current_seed,
            'config_snapshot_sha256': config_snapshot_sha256,
            'paper': _paper_fields(configs),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'conda_environment': os.environ.get('CONDA_DEFAULT_ENV'),
            'working_directory': str(Path.cwd()),
            'runtime': runtime_environment(),
            'dependency_lock': dependency_lock_evidence(),
            'git': launch_git_state,
            'code_files': launch_code_files,
            'code_state_sha256': launch_code_state_sha256,
        }
        invocation_path = Path(path) / 'invocation.json'
        with invocation_path.open('x', encoding='utf-8') as handle:
            json.dump(invocation, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write('\n')
        invocation_sha256 = sha256_file(invocation_path)


        # 构建数据工厂
        print("[INFO] 构建数据工厂...")
        data_factory = build_data(args_data, args_task)
        split_manifest = None
        data_snapshot_path = None
        data_snapshot_sha256 = None
        data_payload_sha256 = None
        if bool(getattr(args_task, 'export_predictions', False)):
            split_manifest = data_factory.get_split_manifest()
            data_snapshot = {
                'schema_version': 1,
                'paper': _paper_fields(configs),
                'config_snapshot_sha256': config_snapshot_sha256,
                'invocation_sha256': invocation_sha256,
                'split_manifest_payload_sha256': split_manifest[
                    'manifest_payload_sha256'
                ],
                'metadata_source': str(
                    (Path(str(args_data.data_dir)) / str(args_data.metadata_file)).resolve()
                ),
                'metadata_source_sha256': sha256_file(
                    Path(str(args_data.data_dir)) / str(args_data.metadata_file)
                ),
                **data_factory.get_data_fingerprint(),
            }
            data_payload_sha256 = data_snapshot['data_payload_sha256']
            data_snapshot_path = Path(path) / 'artifacts' / 'data_snapshot.manifest.json'
            data_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with data_snapshot_path.open('x', encoding='utf-8') as handle:
                json.dump(
                    data_snapshot,
                    handle,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                handle.write('\n')
            data_snapshot_sha256 = sha256_file(data_snapshot_path)
        # 构建模型
        print("[INFO] 构建模型...")
        model = build_model(args_model,metadata=data_factory.get_metadata())
        _validate_p01_trainable_parameter_count(configs, model)
        
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
        
        # 执行训练
        print("[INFO] 开始训练...")
        trainer.fit(
            task,
            data_factory.get_dataloader('train'),
            data_factory.get_dataloader('val')
        )
        
        # 加载最佳模型并测试
        print("[INFO] 加载最佳模型并测试...")
        best_checkpoint = _best_checkpoint_provenance(trainer)
        checkpoint_manifest_path = Path(path) / 'artifacts' / 'best_checkpoint.manifest.json'
        checkpoint_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_manifest_path.open('x', encoding='utf-8') as handle:
            json.dump(
                {"schema_version": 1, **best_checkpoint},
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write('\n')
        best_checkpoint_manifest_sha256 = sha256_file(checkpoint_manifest_path)
        task = load_best_model_checkpoint(task, trainer)
        result = trainer.test(task, data_factory.get_dataloader('test'))
        trainer_metrics_provenance = {}
        if is_p01_evidence:
            trainer_metrics_provenance = _write_trainer_metrics_manifest(
                trainer, path
            )
        if bool(getattr(args_task, 'export_predictions', False)):
            split_config = getattr(args_data, 'split', None)
            group_key = getattr(split_config, 'group_key', None)
            if not group_key:
                raise ValueError("task.export_predictions requires data.split.group_key")
            if split_manifest is None or data_snapshot_path is None:
                raise RuntimeError("Prediction export requires frozen data provenance")
            paper_fields = _paper_fields(configs)
            export_classification_artifacts(
                task,
                data_factory.get_dataloader('test'),
                os.path.join(path, 'artifacts', 'predictions.npz'),
                metadata=data_factory.get_metadata(),
                group_key=str(group_key),
                outer_fold=int(getattr(split_config, 'outer_fold')),
                training_seed=int(current_seed),
                expected_file_ids=split_manifest['split_ids']['test'],
                expected_group_ids=split_manifest['split_groups']['test'],
                required_representation_names=getattr(
                    args_task, 'required_representation_arrays', []
                ),
                provenance={
                    **paper_fields,
                    **trainer_metrics_provenance,
                    'outer_fold': int(getattr(split_config, 'outer_fold')),
                    'training_seed': int(current_seed),
                    'config_snapshot_path': str(config_snapshot_path.resolve()),
                    'config_snapshot_sha256': config_snapshot_sha256,
                    'invocation_path': str(invocation_path.resolve()),
                    'invocation_sha256': invocation_sha256,
                    'best_checkpoint_manifest_path': str(
                        checkpoint_manifest_path.resolve()
                    ),
                    'best_checkpoint_manifest_sha256': (
                        best_checkpoint_manifest_sha256
                    ),
                    'checkpoint_path': best_checkpoint['path'],
                    'checkpoint_sha256': best_checkpoint['sha256'],
                    'checkpoint_monitor': best_checkpoint['monitor'],
                    'checkpoint_mode': best_checkpoint['mode'],
                    'checkpoint_score': best_checkpoint['score'],
                    'split_manifest_path': str(
                        Path(str(getattr(split_config, 'manifest_path'))).resolve()
                    ),
                    'code_state_identifier': (
                        f"git:{launch_git_state['commit']};"
                        f"files:{launch_code_state_sha256}"
                    ),
                    'code_state_sha256': launch_code_state_sha256,
                    'data_snapshot_manifest_path': str(data_snapshot_path.resolve()),
                    'data_snapshot_manifest_sha256': data_snapshot_sha256,
                    'data_payload_sha256': data_payload_sha256,
                    'split_manifest_payload_sha256': split_manifest[
                        'manifest_payload_sha256'
                    ],
                },
            )
        data_factory.data.close()  # 关闭数据工厂，释放资源
        all_results.append(result[0])  # Lightning返回的是包含字典的列表
        
        # 保存结果
        print("[INFO] 保存测试结果...")
        result_df = pd.DataFrame([result[0]])
        result_df.to_csv(os.path.join(path, f'test_result_{it}.csv'), index=False)

        # 关闭wandb和swanlab
        close_lab()

    print(f"\n{'='*50}\n[INFO] 所有实验已完成\n{'='*50}")
    pd.DataFrame(all_results).to_csv(os.path.join(path, 'all_results.csv'), index=False)
    return all_results


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
