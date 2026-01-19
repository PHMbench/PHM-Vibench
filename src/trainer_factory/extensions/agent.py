from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl


@dataclass
class AgentContext:
    run_dir: Path
    paper_id: str = ""
    preset_version: str = ""
    run_id: str = ""


class DistillWriterCallback(pl.Callback):
    """Write TODO-only distilled artifacts (LLM-free).

    Output directory: <run_dir>/artifacts/distilled/
    This callback is best-effort and must never crash training.
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
        self.ctx = AgentContext(run_dir=Path(run_dir), paper_id=paper_id, preset_version=preset_version, run_id=run_id)
        self.enabled = enabled
        self.is_main_process = is_main_process

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # noqa: D401
        self._write(trainer, pl_module, stage="test")

    def _write(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if not self.enabled or not self.is_main_process:
            return

        try:
            run_dir = self.ctx.run_dir
            artifacts_dir = run_dir / "artifacts"
            distilled_dir = artifacts_dir / "distilled"
            distilled_dir.mkdir(parents=True, exist_ok=True)

            config_snapshot = str((run_dir / "config_snapshot.yaml")) if (run_dir / "config_snapshot.yaml").exists() else ""

            metrics_inline: Dict[str, Any] = {}
            for k, v in trainer.callback_metrics.items():
                try:
                    if hasattr(v, "item"):
                        metrics_inline[str(k)] = float(v.item())
                    elif isinstance(v, (int, float, bool, str)):
                        metrics_inline[str(k)] = v
                except Exception:
                    continue

            uxfd_debug: Optional[Dict[str, Any]] = None
            try:
                network = getattr(pl_module, "network", None)
                if network is not None and hasattr(network, "get_uxfd_debug_state"):
                    uxfd_debug = network.get_uxfd_debug_state()  # type: ignore[assignment]
            except Exception:
                uxfd_debug = None

            payload: Dict[str, Any] = {
                "paper_id": self.ctx.paper_id,
                "preset_version": self.ctx.preset_version,
                "run_id": self.ctx.run_id or os.path.basename(str(run_dir)),
                "stage": stage,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "config_snapshot": config_snapshot,
                "metrics_inline": metrics_inline,
                "uxfd_debug": uxfd_debug or {},
                "notes": "LLM-free distilled artifact (placeholder for future agent pipeline).",
            }

            (distilled_dir / "summary.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            return

