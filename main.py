import argparse
import importlib
import sys
import warnings
from pathlib import Path

import yaml

from src.configs.config_utils import merge_with_local_override
from src.configs.preflight import PreflightError, run_preflight
from src.utils.config_utils import apply_overrides_to_config, parse_overrides


def _die(message: str, code: int = 2) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(code)


def _resolve_config_arg(args: argparse.Namespace) -> str:
    if args.config is not None and str(args.config).strip():
        return str(args.config)

    if args.config_path is not None and str(args.config_path).strip():
        warnings.warn(
            "--config_path is deprecated; use --config instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return str(args.config_path)

    _die("Missing required --config <yaml>. No default demo is selected implicitly.")


def _load_pipeline_name(config_path: str) -> str:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        _die(f"Config file does not exist: {config_path}")
    if not cfg_path.is_file():
        _die(f"Config path is not a file: {config_path}")

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        _die(f"Invalid YAML in config {config_path}: {exc}")
    except OSError as exc:
        _die(f"Cannot read config {config_path}: {exc}")

    if not isinstance(cfg_dict, dict):
        _die(f"Config YAML must be a mapping: {config_path}")

    pipeline_name = cfg_dict.get("pipeline")
    if not isinstance(pipeline_name, str) or not pipeline_name.strip():
        _die(f"Config must define a non-empty top-level 'pipeline': {config_path}")

    return pipeline_name.strip()


def main(argv=None):
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
        "--local_config",
        type=str,
        default=None,
        help="本机覆盖配置路径（可选）",
    )

    parser.add_argument(
        "--fs_config_path",
        type=str,
        default=None,
        help="[Pipeline_02 legacy] few-shot config path",
    )

    parser.add_argument(
        "--override",
        action="append",
        help="覆盖配置参数 (格式: key=value)，可多次使用",
    )

    args = parser.parse_args(argv)

    # 统一解析最终配置路径：优先使用 --config，其次兼容 --config_path；不再隐式选择默认 demo。
    config_path = _resolve_config_arg(args)
    args.config_path = config_path

    # 从 YAML 中读取 pipeline 名称；缺失或无效时 fail-fast。
    pipeline_name = _load_pipeline_name(config_path)

    try:
        resolved_config = merge_with_local_override(config_path, getattr(args, "local_config", None))
        if args.override:
            resolved_config = apply_overrides_to_config(
                resolved_config,
                parse_overrides(args.override),
            )
        run_preflight(
            resolved_config,
            config_path=config_path,
            args=args,
            strict=True,
            require_data=True,
            create_output_dir=True,
        )
    except PreflightError as exc:
        _die(str(exc))
    except Exception as exc:
        _die(f"Config preflight could not resolve {config_path}: {exc}")

    try:
        pipeline_module = importlib.import_module(f"src.{pipeline_name}")
    except ModuleNotFoundError as exc:
        expected_module = f"src.{pipeline_name}"
        if exc.name == expected_module:
            _die(f"Pipeline module not found: {expected_module}")
        raise

    if not hasattr(pipeline_module, "pipeline"):
        _die(f"Pipeline module src.{pipeline_name} does not expose pipeline(args)")

    results = pipeline_module.pipeline(args)
    print("完成所有实验！")
    return results


if __name__ == "__main__":
    main()
