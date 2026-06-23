from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.configs.config_utils import save_config

from .eligibility import explain_ready, write_eligibility
from .metadata_reader import read_meta_from_batch, snapshot_metadata, write_metadata_snapshot


def _is_main_process() -> bool:
    return "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0


def _manifest_enabled(args_trainer: Any) -> bool:
    extensions = getattr(args_trainer, "extensions", None)
    report_cfg = getattr(extensions, "report", None) if extensions is not None else None
    report_enable = getattr(report_cfg, "enable", True) if report_cfg is not None else True
    manifest_enable = getattr(report_cfg, "manifest", True) if report_cfg is not None else True
    return bool(report_enable) and bool(manifest_enable)


def write_run_artifact_sidecars(
    run_dir: str | Path,
    cfg: Any,
    args_trainer: Any,
    data_factory: Any,
) -> Tuple[Dict[str, Any], str, bool]:
    run_dir = Path(run_dir)
    save_config(cfg, run_dir / "config_snapshot.yaml")

    batch_meta, meta_source, degraded = write_data_metadata_snapshot_from_data_factory(
        run_dir=run_dir,
        data_factory=data_factory,
    )

    extensions = getattr(args_trainer, "extensions", None)
    explain_cfg = getattr(extensions, "explain", None) if extensions is not None else None
    explain_enable = bool(getattr(explain_cfg, "enable", False)) if explain_cfg is not None else False
    if not explain_enable:
        return batch_meta, str(meta_source), bool(degraded)

    explainer_id = str(getattr(explain_cfg, "explainer", "") or "unknown")
    required_meta_keys = ["sampling_rate"] if explainer_id in {"timefreq", "time_freq"} else []
    write_explain_eligibility(
        run_dir=run_dir,
        explainer_id=explainer_id,
        meta=batch_meta,
        meta_source=str(meta_source),
        degraded=bool(degraded),
        required_meta_keys=required_meta_keys,
    )
    return batch_meta, str(meta_source), bool(degraded)


def rewrite_manifest_after_test_result(
    run_dir: str | Path,
    args_trainer: Any,
    trainer: Any,
    *,
    seed: Optional[int],
    stage: str = "test",
    required: bool = True,
) -> Optional[Path]:
    if not _manifest_enabled(args_trainer) or not _is_main_process():
        return None

    from src.trainer_factory.extensions import write_run_manifest

    return write_run_manifest(
        run_dir=run_dir,
        stage=stage,
        paper_id=str(getattr(args_trainer, "paper_id", "") or ""),
        preset_version=str(getattr(args_trainer, "preset_version", "") or ""),
        run_id=str(getattr(args_trainer, "logger_name", "") or ""),
        seed=seed,
        trainer=trainer,
        required=required,
    )


def write_data_metadata_snapshot_from_data_factory(
    run_dir: Path,
    data_factory: Any,
) -> Tuple[Dict[str, Any], str, bool]:
    """Best-effort snapshot of batch metadata for auditability.

    Writes: <run_dir>/artifacts/data_metadata_snapshot.json
    Returns: (meta, meta_source, degraded)
    """
    artifacts_dir = run_dir / "artifacts"
    meta_snapshot_path = artifacts_dir / "data_metadata_snapshot.json"
    batch_meta: Dict[str, Any] = {}
    meta_source = "default"

    try:
        test_loader = data_factory.get_dataloader("test")
        batch = next(iter(test_loader))
        x0, y0, meta0, meta_source = read_meta_from_batch(batch)
        if isinstance(meta0, dict):
            batch_meta.update(meta0)
        if hasattr(x0, "shape"):
            batch_meta.setdefault("x_shape", [int(v) for v in x0.shape])
        if hasattr(y0, "shape"):
            batch_meta.setdefault("y_shape", [int(v) for v in y0.shape])

        snapshot = snapshot_metadata(meta=batch_meta, meta_source=str(meta_source))
        write_metadata_snapshot(meta_snapshot_path, snapshot)
        return batch_meta, str(meta_source), bool(snapshot.degraded)
    except Exception:
        try:
            snapshot = snapshot_metadata(meta={}, meta_source="default")
            write_metadata_snapshot(meta_snapshot_path, snapshot)
        except Exception:
            pass
        return batch_meta, str(meta_source), True


def write_explain_eligibility(
    run_dir: Path,
    explainer_id: str,
    meta: Optional[Dict[str, Any]],
    meta_source: str,
    degraded: bool,
    required_meta_keys: Optional[List[str]] = None,
) -> None:
    eligibility_path = Path(run_dir) / "artifacts" / "explain" / "eligibility.json"
    ready = explain_ready(
        explainer_id=str(explainer_id or "unknown"),
        meta=meta or {},
        required_meta_keys=required_meta_keys or [],
        meta_source=str(meta_source),
        degraded=bool(degraded),
    )
    write_eligibility(eligibility_path, ready)
