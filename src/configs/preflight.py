from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_SECTIONS = ("environment", "data", "model", "task", "trainer")
P02_MODULE = "Pipeline_02_pretrain_fewshot"
P02_MODES = {"single", "staged", "legacy"}


@dataclass(frozen=True)
class PreflightCheck:
    check: str
    ok: bool
    message: str
    fix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "ok": self.ok,
            "message": self.message,
            "fix": self.fix,
        }


class PreflightError(RuntimeError):
    """Raised when a resolved experiment config cannot safely enter training."""

    def __init__(self, failures: Iterable[PreflightCheck]) -> None:
        self.failures = list(failures)
        detail = "\n".join(f"- {f.check}: {f.message}" for f in self.failures)
        super().__init__(f"Preflight failed:\n{detail}")


def namespace_to_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: namespace_to_dict(v) for k, v in value.items()}
    if isinstance(value, (SimpleNamespace,)) or (
        hasattr(value, "__dict__") and not isinstance(value, (str, bytes))
    ):
        return {k: namespace_to_dict(v) for k, v in value.__dict__.items()}
    if isinstance(value, list):
        return [namespace_to_dict(v) for v in value]
    return value


def _add(checks: List[PreflightCheck], name: str, ok: bool, message: str, fix: str = "") -> None:
    checks.append(PreflightCheck(name, bool(ok), message, fix))


def _block(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _path_from_data_dir(data_dir: Any, child: Any = "") -> Optional[Path]:
    if not isinstance(data_dir, str) or not data_dir.strip():
        return None
    base = Path(data_dir)
    if not isinstance(child, str) or not child.strip():
        return base
    return base / child


def _check_required_sections(config: Dict[str, Any], checks: List[PreflightCheck]) -> None:
    for section in REQUIRED_SECTIONS:
        value = config.get(section)
        _add(
            checks,
            f"preflight.has_{section}",
            isinstance(value, dict),
            f"{section} present: {isinstance(value, dict)}",
            fix=f"Ensure resolved config contains `{section}: ...`.",
        )


def _check_pipeline(config: Dict[str, Any], checks: List[PreflightCheck]) -> None:
    pipeline = config.get("pipeline")
    ok_name = isinstance(pipeline, str) and bool(pipeline.strip())
    _add(
        checks,
        "preflight.pipeline_declared",
        ok_name,
        f"pipeline={pipeline!r}",
        fix="Set top-level `pipeline:` to an existing src/Pipeline_*.py module.",
    )
    if not ok_name:
        return

    module_name = f"src.{str(pipeline).strip()}"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        _add(
            checks,
            "preflight.pipeline_import",
            False,
            f"{module_name} import failed: {exc!r}",
            fix="Fix the pipeline name or missing import dependency before training.",
        )
        return

    _add(
        checks,
        "preflight.pipeline_import",
        hasattr(module, "pipeline"),
        f"{module_name}.pipeline exists: {hasattr(module, 'pipeline')}",
        fix="Expose a callable `pipeline(args)` from the selected module.",
    )


def _check_p02_mode(
    config: Dict[str, Any],
    checks: List[PreflightCheck],
    args: Optional[Any],
) -> None:
    if str(config.get("pipeline") or "").strip() != P02_MODULE:
        return

    mode = config.get("pipeline_mode")
    mode_ok = mode in P02_MODES
    _add(
        checks,
        "preflight.p02_pipeline_mode",
        mode_ok,
        f"pipeline_mode={mode!r}",
        fix="Set `pipeline_mode: single`, `staged`, or `legacy`.",
    )
    if not mode_ok:
        return

    stages = config.get("stages")
    fs_config_path = getattr(args, "fs_config_path", None) if args is not None else None
    has_fs_config = isinstance(fs_config_path, str) and bool(fs_config_path.strip())

    if mode == "staged":
        _add(
            checks,
            "preflight.p02_stages",
            isinstance(stages, list) and bool(stages),
            "pipeline_mode=staged requires non-empty top-level stages.",
            fix="Add top-level `stages:` or switch `pipeline_mode`.",
        )
        _add(
            checks,
            "preflight.p02_no_fs_config_conflict",
            not has_fs_config,
            "pipeline_mode=staged conflicts with --fs_config_path.",
            fix="Remove --fs_config_path for staged mode.",
        )
    elif mode == "legacy":
        _add(
            checks,
            "preflight.p02_legacy_fs_config",
            has_fs_config,
            "pipeline_mode=legacy requires --fs_config_path.",
            fix="Pass --fs_config_path <fewshot.yaml> or use single/staged mode.",
        )
    elif mode == "single":
        _add(
            checks,
            "preflight.p02_single_no_stages",
            "stages" not in config,
            "pipeline_mode=single conflicts with top-level stages.",
            fix="Remove `stages:` or switch to `pipeline_mode: staged`.",
        )
        _add(
            checks,
            "preflight.p02_single_no_fs_config",
            not has_fs_config,
            "pipeline_mode=single conflicts with --fs_config_path.",
            fix="Remove --fs_config_path or switch to legacy mode.",
        )


def _check_environment(
    config: Dict[str, Any],
    checks: List[PreflightCheck],
    create_output_dir: bool,
) -> None:
    env = _block(config, "environment")
    output_dir = env.get("output_dir")
    output_ok = isinstance(output_dir, str) and bool(output_dir.strip())
    _add(
        checks,
        "preflight.output_dir_set",
        output_ok,
        f"environment.output_dir={output_dir!r}",
        fix="Set `environment.output_dir` to a run output directory.",
    )
    if not output_ok:
        return

    path = Path(str(output_dir))
    if create_output_dir:
        try:
            path.mkdir(parents=True, exist_ok=True)
            _add(
                checks,
                "preflight.output_dir_writable",
                path.is_dir(),
                f"output_dir ready: {path}",
                fix="Choose a writable output_dir.",
            )
        except OSError as exc:
            _add(
                checks,
                "preflight.output_dir_writable",
                False,
                f"cannot create output_dir {path}: {exc}",
                fix="Choose a writable output_dir.",
            )
    else:
        _add(
            checks,
            "preflight.output_dir_parent_exists",
            path.parent.exists() or not path.is_absolute(),
            f"output_dir parent={path.parent}",
            fix="Create the parent directory or use a repo-relative output_dir.",
        )


def _check_data_paths(config: Dict[str, Any], checks: List[PreflightCheck]) -> None:
    data = _block(config, "data")
    data_dir = _path_from_data_dir(data.get("data_dir"))
    metadata = _path_from_data_dir(data.get("data_dir"), data.get("metadata_file"))

    _add(
        checks,
        "preflight.data_dir_set",
        data_dir is not None,
        f"data.data_dir={data.get('data_dir')!r}",
        fix="Set `data.data_dir` to a repo-relative path or environment-expanded path.",
    )
    if data_dir is not None:
        _add(
            checks,
            "preflight.data_dir_exists",
            data_dir.exists() and data_dir.is_dir(),
            f"data_dir={data_dir}",
            fix="Provide the dataset directory or override `data.data_dir`.",
        )

    _add(
        checks,
        "preflight.metadata_file_set",
        isinstance(data.get("metadata_file"), str) and bool(str(data.get("metadata_file")).strip()),
        f"data.metadata_file={data.get('metadata_file')!r}",
        fix="Set `data.metadata_file` relative to `data.data_dir`.",
    )
    if metadata is not None:
        _add(
            checks,
            "preflight.metadata_file_exists",
            metadata.exists() and metadata.is_file(),
            f"metadata_file={metadata}",
            fix="Provide metadata file or override `data.metadata_file`.",
        )


def build_preflight_report(
    config: Any,
    *,
    config_path: Optional[str] = None,
    args: Optional[Any] = None,
    require_data: bool = True,
    create_output_dir: bool = False,
) -> List[Dict[str, Any]]:
    checks: List[PreflightCheck] = []
    resolved = namespace_to_dict(config)
    if not isinstance(resolved, dict):
        _add(
            checks,
            "preflight.config_mapping",
            False,
            f"resolved config must be a mapping, got {type(resolved).__name__}",
            fix="Load a YAML mapping config.",
        )
        return [c.to_dict() for c in checks]

    if config_path:
        cfg_path = Path(config_path)
        _add(
            checks,
            "preflight.config_file_exists",
            cfg_path.exists() and cfg_path.is_file(),
            f"config_path={cfg_path}",
            fix="Pass an existing YAML file to --config.",
        )

    _check_required_sections(resolved, checks)
    _check_pipeline(resolved, checks)
    _check_p02_mode(resolved, checks, args)
    _check_environment(resolved, checks, create_output_dir=create_output_dir)
    if require_data:
        _check_data_paths(resolved, checks)

    return [c.to_dict() for c in checks]


def run_preflight(
    config: Any,
    *,
    config_path: Optional[str] = None,
    args: Optional[Any] = None,
    strict: bool = True,
    require_data: bool = True,
    create_output_dir: bool = True,
) -> List[Dict[str, Any]]:
    report = build_preflight_report(
        config,
        config_path=config_path,
        args=args,
        require_data=require_data,
        create_output_dir=create_output_dir,
    )
    failures = [PreflightCheck(**item) for item in report if not item["ok"]]
    if strict and failures:
        raise PreflightError(failures)
    return report
