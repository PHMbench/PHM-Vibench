"""Explainability extensions for the shared classification runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.configs.config_utils import save_config
from src.explain_factory.eligibility import explain_ready, write_eligibility
from src.explain_factory.metadata_reader import (
    read_meta_from_batch,
    snapshot_metadata,
    write_metadata_snapshot,
)
from src.runtime.classification import ClassificationContext, ClassificationHooks


def _register_if_present(
    context: ClassificationContext,
    *,
    role: str,
    path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    attestation = getattr(context.args, "run_attestation", None)
    if attestation is None or not path.is_file():
        return
    try:
        attestation.register_artifact(
            role=role,
            path=path,
            metadata={"iteration": context.iteration, **dict(metadata or {})},
        )
    except Exception as exc:
        print(f"[WARN] 登记 {role} 到 run manifest 失败: {exc}")


class ExplainabilityHooks(ClassificationHooks):
    """Add UXFD evidence while preserving one invocation-level run identity."""

    def on_iteration_start(self, context: ClassificationContext) -> None:
        snapshot_path = context.path / "config_snapshot.yaml"
        try:
            save_config(context.configs, snapshot_path)
        except Exception as exc:
            print(f"[WARN] 保存 config_snapshot.yaml 失败: {exc}")
        _register_if_present(
            context,
            role="explainability_config_snapshot",
            path=snapshot_path,
        )

    def after_stack_built(self, context: ClassificationContext) -> None:
        artifacts_dir = context.path / "artifacts"
        snapshot_path = artifacts_dir / "data_metadata_snapshot.json"
        batch_meta: dict = {}
        meta_source = "default"
        degraded = True
        try:
            batch = next(iter(context.data_factory.get_dataloader("test")))
            x0, y0, meta0, meta_source = read_meta_from_batch(batch)
            if isinstance(meta0, dict):
                batch_meta.update(meta0)
            if hasattr(x0, "shape"):
                batch_meta.setdefault("x_shape", [int(value) for value in x0.shape])
            if hasattr(y0, "shape"):
                batch_meta.setdefault("y_shape", [int(value) for value in y0.shape])
            snapshot = snapshot_metadata(meta=batch_meta, meta_source=meta_source)
            degraded = snapshot.degraded
            write_metadata_snapshot(snapshot_path, snapshot)
        except Exception as exc:
            print(f"[WARN] 写入 data_metadata_snapshot.json 失败: {exc}")
            try:
                write_metadata_snapshot(
                    snapshot_path,
                    snapshot_metadata(meta={}, meta_source="default"),
                )
            except Exception:
                pass
        _register_if_present(
            context,
            role="explainability_data_metadata",
            path=snapshot_path,
            metadata={"meta_source": str(meta_source), "degraded": bool(degraded)},
        )

        extensions = getattr(context.args_trainer, "extensions", None)
        explain_cfg = getattr(extensions, "explain", None) if extensions else None
        if not bool(getattr(explain_cfg, "enable", False)):
            return
        eligibility_path = artifacts_dir / "explain" / "eligibility.json"
        explainer_id = str(getattr(explain_cfg, "explainer", "") or "unknown")
        try:
            required_meta_keys = (
                ["sampling_rate"]
                if explainer_id in {"timefreq", "time_freq"}
                else []
            )
            write_eligibility(
                eligibility_path,
                explain_ready(
                    explainer_id=explainer_id,
                    meta=batch_meta,
                    required_meta_keys=required_meta_keys,
                    meta_source=str(meta_source),
                    degraded=bool(degraded),
                ),
            )
        except Exception as exc:
            print(f"[WARN] 写入 explain eligibility 失败: {exc}")
        _register_if_present(
            context,
            role="explainability_eligibility",
            path=eligibility_path,
            metadata={"explainer": explainer_id},
        )

    def after_test(self, context: ClassificationContext) -> None:
        legacy_manifest = context.path / "artifacts" / "manifest.json"
        try:
            from src.trainer_factory.extensions import ManifestWriterCallback

            is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
            extensions = getattr(context.args_trainer, "extensions", None)
            report_cfg = getattr(extensions, "report", None) if extensions else None
            enabled = bool(getattr(report_cfg, "enable", True)) and bool(
                getattr(report_cfg, "manifest", True)
            )
            ManifestWriterCallback(
                run_dir=str(context.path),
                paper_id=str(getattr(context.args_trainer, "paper_id", "") or ""),
                preset_version=str(
                    getattr(context.args_trainer, "preset_version", "") or ""
                ),
                run_id=str(getattr(context.args_trainer, "logger_name", "") or ""),
                enabled=enabled,
                is_main_process=is_main_process,
            ).on_test_end(context.trainer, context.task)
        except Exception as exc:
            print(f"[WARN] 更新 artifacts/manifest.json 失败: {exc}")
        _register_if_present(
            context,
            role="explainability_legacy_manifest",
            path=legacy_manifest,
            metadata={"compatibility_only": True},
        )
