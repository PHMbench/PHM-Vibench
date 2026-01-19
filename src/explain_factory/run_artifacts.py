from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .eligibility import explain_ready, write_eligibility
from .metadata_reader import read_meta_from_batch, snapshot_metadata, write_metadata_snapshot


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
    """Best-effort eligibility writer; never raises."""
    try:
        eligibility_path = run_dir / "artifacts" / "explain" / "eligibility.json"
        ready = explain_ready(
            explainer_id=str(explainer_id or "unknown"),
            meta=meta or {},
            required_meta_keys=required_meta_keys or [],
            meta_source=str(meta_source),
            degraded=bool(degraded),
        )
        write_eligibility(eligibility_path, ready)
    except Exception:
        return

