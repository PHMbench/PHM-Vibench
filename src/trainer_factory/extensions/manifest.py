from __future__ import annotations

import json
import os
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
            run_dir = self.ctx.run_dir
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            config_snapshot = str((run_dir / "config_snapshot.yaml")) if (run_dir / "config_snapshot.yaml").exists() else ""
            logs_metrics = _find_first(run_dir / "logs", "**/metrics.csv")
            test_results = _find_first(run_dir, "test_result_*.csv")

            figures_dir = str(run_dir / "figures") if (run_dir / "figures").exists() else ""
            explain_dir = str(artifacts_dir / "explain") if (artifacts_dir / "explain").exists() else ""
            distilled_dir = str(artifacts_dir / "distilled") if (artifacts_dir / "distilled").exists() else ""
            explain_summary = str(artifacts_dir / "explain" / "summary.json") if (artifacts_dir / "explain" / "summary.json").exists() else ""

            data_metadata_snapshot = str(artifacts_dir / "data_metadata_snapshot.json") if (artifacts_dir / "data_metadata_snapshot.json").exists() else ""
            eligibility = str(artifacts_dir / "explain" / "eligibility.json") if (artifacts_dir / "explain" / "eligibility.json").exists() else ""

            manifest: Dict[str, Any] = {
                "paper_id": self.ctx.paper_id,
                "preset_version": self.ctx.preset_version,
                "run_id": self.ctx.run_id or os.path.basename(str(run_dir)),
                "run_dir": str(run_dir),
                "stage": stage,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "config_snapshot": config_snapshot,
                "metrics_path": test_results or logs_metrics,
                "metrics_csv_logger": logs_metrics,
                "figures_dir": figures_dir,
                "data_metadata_snapshot": data_metadata_snapshot,
                "eligibility": eligibility,
                "explain_dir": explain_dir,
                "explain_summary": explain_summary,
                "distilled_dir": distilled_dir,
            }

            # Best-effort: embed scalar callback_metrics into metrics_inline for CSV flattening.
            metrics_inline: Dict[str, Any] = {}
            for k, v in trainer.callback_metrics.items():
                try:
                    if hasattr(v, "item"):
                        metrics_inline[str(k)] = float(v.item())
                    else:
                        # only keep scalars
                        if isinstance(v, (int, float)):
                            metrics_inline[str(k)] = v
                except Exception:
                    continue
            if metrics_inline:
                manifest["metrics_inline"] = metrics_inline

            (artifacts_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            # must never crash training
            return
