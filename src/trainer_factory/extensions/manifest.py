from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl


def _list_existing(paths: List[Path]) -> List[str]:
    return [str(p) for p in paths if p.exists()]


def _find_first(glob_root: Path, pattern: str) -> str:
    for p in sorted(glob_root.glob(pattern)):
        if p.exists():
            return str(p)
    return ""


def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _metrics_inline(trainer: Optional["pl.Trainer"]) -> Dict[str, Any]:
    if trainer is None:
        return {}

    metrics: Dict[str, Any] = {}
    for k, v in getattr(trainer, "callback_metrics", {}).items():
        try:
            if hasattr(v, "item"):
                metrics[str(k)] = float(v.item())
            elif isinstance(v, (int, float)):
                metrics[str(k)] = v
        except Exception:
            continue
    return metrics


def build_run_manifest(
    run_dir: str | Path,
    *,
    stage: str,
    run_id: str = "",
    seed: Optional[int] = None,
    paper_id: str = "",
    preset_version: str = "",
    trainer: Optional["pl.Trainer"] = None,
) -> Dict[str, Any]:
    """Build the stable parent-consumable run manifest payload."""

    run_dir = Path(run_dir)
    artifacts_dir = run_dir / "artifacts"

    config_snapshot = str(run_dir / "config_snapshot.yaml") if (run_dir / "config_snapshot.yaml").exists() else ""
    logs_metrics = _find_first(run_dir / "logs", "**/metrics.csv")
    test_results = _find_first(run_dir, "test_result_*.csv") or _find_first(run_dir, "test_result.csv")
    predictions = (
        _find_first(artifacts_dir, "predictions.npz")
        or _find_first(run_dir, "predictions.npz")
        or _find_first(artifacts_dir, "predictions.*")
        or _find_first(run_dir, "predictions.*")
    )

    figures_dir = str(run_dir / "figures") if (run_dir / "figures").exists() else ""
    explain_dir = str(artifacts_dir / "explain") if (artifacts_dir / "explain").exists() else ""
    distilled_dir = str(artifacts_dir / "distilled") if (artifacts_dir / "distilled").exists() else ""
    explain_summary_path = artifacts_dir / "explain" / "summary.json"
    explain_summary = str(explain_summary_path) if explain_summary_path.exists() else ""

    data_metadata_snapshot_path = artifacts_dir / "data_metadata_snapshot.json"
    data_metadata_snapshot = (
        str(data_metadata_snapshot_path) if data_metadata_snapshot_path.exists() else ""
    )
    eligibility_path = artifacts_dir / "explain" / "eligibility.json"
    eligibility = str(eligibility_path) if eligibility_path.exists() else ""

    manifest: Dict[str, Any] = {
        "paper_id": paper_id,
        "preset_version": preset_version,
        "run_id": run_id or os.path.basename(str(run_dir)),
        "run_dir": str(run_dir),
        "stage": stage,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed if seed is not None else "",
        "git_sha": _git_sha(),
        "config_snapshot": config_snapshot,
        "metrics_path": test_results or logs_metrics,
        "metrics_csv_logger": logs_metrics,
        "figures_dir": figures_dir,
        "predictions_path": str(predictions) if predictions else "",
        "data_metadata_snapshot": data_metadata_snapshot,
        "eligibility": eligibility,
        "explain_dir": explain_dir,
        "explain_summary": explain_summary,
        "distilled_dir": distilled_dir,
    }

    metrics_inline = _metrics_inline(trainer)
    if metrics_inline:
        manifest["metrics_inline"] = metrics_inline

    return manifest


def _validate_required_manifest(manifest: Dict[str, Any]) -> None:
    missing = []
    for key in [
        "run_id",
        "stage",
        "run_dir",
        "timestamp",
        "seed",
        "git_sha",
        "config_snapshot",
        "metrics_path",
        "data_metadata_snapshot",
    ]:
        if manifest.get(key) in (None, ""):
            missing.append(key)

    for key in ["config_snapshot", "metrics_path", "data_metadata_snapshot"]:
        value = manifest.get(key)
        if value and not Path(str(value)).exists():
            missing.append(f"{key}:missing_file")

    if missing:
        raise RuntimeError(f"Run manifest missing required fields: {', '.join(missing)}")


def write_run_manifest(
    run_dir: str | Path,
    *,
    stage: str,
    run_id: str = "",
    seed: Optional[int] = None,
    paper_id: str = "",
    preset_version: str = "",
    trainer: Optional["pl.Trainer"] = None,
    required: bool = True,
) -> Path:
    """Write `artifacts/manifest.json`; raise when required contract fields are absent."""

    run_dir = Path(run_dir)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        run_dir,
        stage=stage,
        run_id=run_id,
        seed=seed,
        paper_id=paper_id,
        preset_version=preset_version,
        trainer=trainer,
    )
    if required:
        _validate_required_manifest(manifest)

    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


@dataclass
class ManifestContext:
    run_dir: Path
    paper_id: str = ""
    preset_version: str = ""
    run_id: str = ""


class ManifestWriterCallback(pl.Callback):
    """Write an auditable `artifacts/manifest.json` for each run directory.

    This callback is intentionally best-effort and must never crash training.
    """

    def __init__(
        self,
        run_dir: str,
        paper_id: str = "",
        preset_version: str = "",
        run_id: str = "",
        enabled: bool = True,
        is_main_process: bool = True,
    ) -> None:
        super().__init__()
        self.ctx = ManifestContext(run_dir=Path(run_dir), paper_id=paper_id, preset_version=preset_version, run_id=run_id)
        self.enabled = enabled
        self.is_main_process = is_main_process

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # noqa: D401
        self._write(trainer, stage="fit")

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # noqa: D401
        self._write(trainer, stage="test")

    def _write(self, trainer: "pl.Trainer", stage: str) -> None:
        if not self.enabled or not self.is_main_process:
            return

        try:
            write_run_manifest(
                self.ctx.run_dir,
                stage=stage,
                paper_id=self.ctx.paper_id,
                preset_version=self.ctx.preset_version,
                run_id=self.ctx.run_id,
                trainer=trainer,
                required=False,
            )
        except Exception as exc:
            print(f"[WARN] failed to write artifacts/manifest.json for {self.ctx.run_dir}: {exc}")
            return
