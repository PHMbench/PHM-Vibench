from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


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
    """Unpack one explainable batch without inventing missing metadata."""
    if isinstance(batch, Mapping):
        missing = [key for key in ("x", "y", "meta") if key not in batch]
        if missing:
            raise ValueError(f"Explain batch is missing keys: {missing}")
        x, y, meta = batch["x"], batch["y"], batch["meta"]
    elif isinstance(batch, (tuple, list)):
        if len(batch) != 3:
            raise ValueError("Explain batch must be exactly (x, y, meta)")
        x, y, meta = batch
    else:
        raise TypeError(
            "Explain batch must be a mapping or an exact (x, y, meta) sequence"
        )

    if x is None or y is None:
        raise ValueError("Explain batch requires non-null x and y")
    if not isinstance(meta, Mapping) or not meta:
        raise ValueError("Explain batch requires a non-empty metadata mapping")
    return x, y, dict(meta), "batch"


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
