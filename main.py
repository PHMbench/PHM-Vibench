import argparse
import importlib
from pathlib import Path
from typing import Any

import yaml

from pydantic import ValidationError

from src.config_schema import ExperimentConfig
from src.configs.config_utils import load_config, merge_with_local_override
from src.utils.config_utils import apply_overrides_to_config, parse_overrides


ALLOWED_PIPELINES = {
    "Pipeline_01_default",
    "Pipeline_02_pretrain_fewshot",
    "Pipeline_03_multitask_pretrain_finetune",
    "Pipeline_04_unified_metric",
    "Pipeline_05_default_w_explain",
    "Pipeline_06_generative",
    "Pipeline_ID",
}


def _namespace_to_dict(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.__dict__.items()}
    if isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(v) for v in value]
    return value


def validate_pipeline_name(pipeline_name: str) -> str:
    """Return a whitelisted pipeline module name or raise a clear error."""
    name = str(pipeline_name or "").strip()
    if name not in ALLOWED_PIPELINES:
        allowed = ", ".join(sorted(ALLOWED_PIPELINES))
        raise ValueError(f"Unsupported pipeline '{name}'. Allowed pipelines: {allowed}")
    return name


def _load_yaml_probe(path: Path) -> dict[str, Any]:
    """Load a config file only far enough to discover top-level fields."""
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML config '{path}': {exc}") from exc
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Config '{path}' must be a YAML mapping")
    return cfg_dict


def resolve_pipeline_name(config_path: str) -> str:
    """Resolve the top-level pipeline field without allowing arbitrary imports."""
    pipeline_name = "Pipeline_01_default"
    cfg_path = Path(config_path)

    if cfg_path.exists():
        cfg_dict = _load_yaml_probe(cfg_path)
        yaml_pipeline = cfg_dict.get("pipeline")
        if isinstance(yaml_pipeline, str) and yaml_pipeline.strip():
            pipeline_name = yaml_pipeline.strip()
    else:
        # Preset names are resolved by the config loader, not by pathlib.
        cfg = load_config(config_path)
        yaml_pipeline = getattr(cfg, "pipeline", None)
        if isinstance(yaml_pipeline, str) and yaml_pipeline.strip():
            pipeline_name = yaml_pipeline.strip()

    return validate_pipeline_name(pipeline_name)


def load_preflight_config(args) -> tuple[Any, dict[str, Any]]:
    """Load config exactly for validation without importing or executing a pipeline."""
    config_path = str(args.config_path)
    cfg_path = Path(config_path)
    if cfg_path.exists():
        _load_yaml_probe(cfg_path)

    configs = merge_with_local_override(config_path, getattr(args, "local_config", None))
    if getattr(args, "override", None):
        configs = apply_overrides_to_config(configs, parse_overrides(args.override))
    resolved = _namespace_to_dict(configs)
    return configs, resolved


def preflight(args) -> dict[str, Any]:
    """Validate config/pipeline contracts and return the resolved config dict."""
    configs, resolved = load_preflight_config(args)
    pipeline_name = validate_pipeline_name(str(resolved.get("pipeline", "")))
    required_sections = ["environment", "data", "model", "task", "trainer"]
    missing = [section for section in required_sections if section not in resolved]
    if missing:
        raise ValueError(f"config is missing required section(s): {', '.join(missing)}")
    try:
        ExperimentConfig.model_validate(resolved)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    resolved["pipeline"] = pipeline_name
    return resolved


def main():
    """
    Vbench 主入口，配置环境变量并调用实验流水线
    """
    parser = argparse.ArgumentParser(description="任务流水线")

    # 推荐入口：--config（支持 YAML 路径或预设名称）
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径或预设名称（推荐使用）",
    )

    # 兼容旧参数：--config_path（若未提供 --config，则仍可使用）
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="[兼容] 配置文件路径（将逐步被 --config 替代）",
    )

    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="实验备注",
    )

    parser.add_argument(
        "--override",
        action="append",
        help="覆盖配置参数 (格式: key=value)，可多次使用",
    )

    parser.add_argument(
        "--local_config",
        type=str,
        default=None,
        help="可选的本机覆盖配置 YAML",
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只验证配置、pipeline 和 schema，不启动训练/采样/评估",
    )

    args, unknown_args = parser.parse_known_args()
    hydra_style_overrides = []
    for item in unknown_args:
        if item.startswith("--") or "=" not in item:
            parser.error(f"unrecognized argument: {item}")
        hydra_style_overrides.append(item)

    if hydra_style_overrides:
        args.override = (args.override or []) + hydra_style_overrides

    # 统一解析最终配置路径：优先使用 --config，其次回退到 --config_path，最后使用默认 demo
    if args.config is not None:
        config_path = args.config
    elif args.config_path is not None:
        config_path = args.config_path
    else:
        # 默认使用 v0.1.0 的跨域 DG demo
        config_path = "configs/demo/01_cross_domain/cwru_dg.yaml"

    # 为下游 Pipeline 保持向后兼容：填充 config_path 属性
    args.config_path = config_path

    if args.preflight_only:
        resolved = preflight(args)
        print(f"[OK] preflight passed: {config_path} ({resolved['pipeline']})")
        return resolved

    # 从 YAML/预设中读取 pipeline 名称，并做白名单验证，避免任意 import。
    pipeline_name = resolve_pipeline_name(config_path)

    pipeline_module = importlib.import_module(f"src.{pipeline_name}")
    results = pipeline_module.pipeline(args)
    print("完成所有实验！")
    return results


if __name__ == "__main__":
    main()
