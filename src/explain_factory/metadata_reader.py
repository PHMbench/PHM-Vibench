from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class MetadataSnapshot:
    meta_source: str
    degraded: bool
    missing_keys: Tuple[str, ...]
    meta: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["missing_keys"] = list(self.missing_keys)
        return data


def read_meta_from_batch(batch: Any) -> Tuple[Any, Any, Dict[str, Any], str]:
    """Best-effort batch unpacking: (x,y), (x,y,meta), dict with keys."""
    if isinstance(batch, dict):
        x = batch.get("x")
        y = batch.get("y")
        meta = batch.get("meta") or {}
        return x, y, meta, "batch"

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            x, y = batch
            return x, y, {}, "default"
        if len(batch) >= 3:
            x, y, meta = batch[0], batch[1], batch[2]
            return x, y, (meta or {}), "batch"

    return None, None, {}, "default"


def snapshot_metadata(
    meta: Dict[str, Any],
    meta_source: str,
    required_keys: Tuple[str, ...] = (),
) -> MetadataSnapshot:
    missing = tuple(k for k in required_keys if k not in meta or meta.get(k) in (None, ""))
    degraded = meta_source == "default" or bool(missing)
    return MetadataSnapshot(meta_source=meta_source, degraded=degraded, missing_keys=missing, meta=meta)


def write_metadata_snapshot(path: Path, snapshot: MetadataSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

