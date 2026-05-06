import argparse
import importlib
from pathlib import Path

import yaml


ALLOWED_PIPELINES = {
    "Pipeline_01_default",
    "Pipeline_02_pretrain_fewshot",
    "Pipeline_03_multitask_pretrain_finetune",
    "Pipeline_04_unified_metric",
    "Pipeline_05_default_w_explain",
    "Pipeline_06_generative",
    "Pipeline_ID",
}


def validate_pipeline_name(pipeline_name: str) -> str:
    """Return a whitelisted pipeline module name or raise a clear error."""
    name = str(pipeline_name or "").strip()
    if name not in ALLOWED_PIPELINES:
        allowed = ", ".join(sorted(ALLOWED_PIPELINES))
        raise ValueError(f"Unsupported pipeline '{name}'. Allowed pipelines: {allowed}")
    return name


def resolve_pipeline_name(config_path: str) -> str:
    """Resolve the top-level pipeline field without allowing arbitrary imports."""
    pipeline_name = "Pipeline_01_default"
    cfg_path = Path(config_path)

    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f) or {}
            if isinstance(cfg_dict, dict):
                yaml_pipeline = cfg_dict.get("pipeline")
                if isinstance(yaml_pipeline, str) and yaml_pipeline.strip():
                    pipeline_name = yaml_pipeline.strip()
        except Exception:
            # If the light YAML probe fails, let the selected pipeline do full validation.
            pass
    else:
        # Preset names are resolved by the config loader, not by pathlib.
        try:
            from src.configs.config_utils import load_config

            cfg = load_config(config_path)
            yaml_pipeline = getattr(cfg, "pipeline", None)
            if isinstance(yaml_pipeline, str) and yaml_pipeline.strip():
                pipeline_name = yaml_pipeline.strip()
        except Exception:
            pass

    return validate_pipeline_name(pipeline_name)


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

    # 从 YAML/预设中读取 pipeline 名称，并做白名单验证，避免任意 import。
    pipeline_name = resolve_pipeline_name(config_path)

    pipeline_module = importlib.import_module(f"src.{pipeline_name}")
    results = pipeline_module.pipeline(args)
    print("完成所有实验！")
    return results


if __name__ == "__main__":
    main()
