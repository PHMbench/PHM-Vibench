"""Create and verify the read-only P05 signal-cache manifest.

Only metadata-selected root datasets are inspected.  HDF5 files are always
opened ``r`` with SWMR, content is hashed in axis-0 chunks, and an existing
manifest is reusable only when its bytes are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import h5py
import numpy as np
import pandas as pd


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "p05_verified_signal_cache"
REQUIRED_METADATA_COLUMNS = ("Id", "Dataset_id", "Name", "File")
CHANNEL_ORDER_BY_NAME = {
    "RM_001_CWRU": ("DE_time", "FE_time"),
    "RM_002_XJTU": (
        "Horizontal_vibration_signals",
        "Vertical_vibration_signals",
    ),
}


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing_file(value: Any, *, key: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{key} is required")
    candidate = Path(str(value)).expanduser().resolve(strict=False)
    if not candidate.is_file():
        raise FileNotFoundError(f"{key} does not resolve to a file: {candidate}")
    return candidate.resolve(strict=True)


def expected_channel_order(name: Any) -> tuple[str, str]:
    normalized = str(name)
    try:
        return CHANNEL_ORDER_BY_NAME[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported P05 metadata Name for channel binding: {normalized!r}") from exc


def _integer(value: Any, *, column: str, row: Any) -> int:
    if value is None or pd.isna(value) or isinstance(value, bool):
        raise ValueError(f"{column} must be an integer at metadata row {row!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be an integer at metadata row {row!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{column} must be an integer at metadata row {row!r}")
    return int(number)


def _text(value: Any, *, column: str, row: Any) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{column} must be a non-empty string at metadata row {row!r}")
    return str(value)


def _read_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise RuntimeError("openpyxl is required to read metadata workbooks") from exc
    raise ValueError(f"unsupported metadata format: {path}")


def normalize_manifest_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"cache manifest metadata is missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        sample_id = _integer(row["Id"], column="Id", row=index)
        name = _text(row["Name"], column="Name", row=sample_id)
        rows.append(
            {
                "Id": sample_id,
                "Dataset_id": _integer(
                    row["Dataset_id"], column="Dataset_id", row=sample_id
                ),
                "Name": name,
                "File": _text(row["File"], column="File", row=sample_id),
                "channel_order": list(expected_channel_order(name)),
            }
        )
    normalized = pd.DataFrame(rows)
    duplicate_ids = normalized.loc[normalized["Id"].duplicated(keep=False), "Id"].tolist()
    if duplicate_ids:
        raise ValueError(f"duplicate metadata Id values: {sorted(set(duplicate_ids))[:10]}")
    return normalized.sort_values("Id", kind="mergesort").reset_index(drop=True)


def hash_h5_dataset(dataset: h5py.Dataset, *, chunk_rows: int) -> str:
    """Hash one dataset in C-order without materializing it in full."""

    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if dataset.ndim < 1 or dataset.shape[0] <= 0:
        raise ValueError(f"cache dataset must have a non-empty axis 0, got {dataset.shape}")
    digest = hashlib.sha256()
    for start in range(0, int(dataset.shape[0]), chunk_rows):
        stop = min(start + chunk_rows, int(dataset.shape[0]))
        block = np.ascontiguousarray(dataset[start:stop, ...])
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def _dataset_contract(dataset: h5py.Dataset, *, sample_id: int) -> tuple[list[int], str, str]:
    shape = [int(value) for value in dataset.shape]
    dtype = np.dtype(dataset.dtype)
    if len(shape) != 3 or shape[1:] != [2, 1]:
        raise ValueError(
            f"cache Id {sample_id} must have shape (L,2,1), got {tuple(shape)}"
        )
    if dtype != np.dtype(np.float64):
        raise ValueError(f"cache Id {sample_id} must use float64, got {dtype}")
    return shape, str(dtype), dtype.str


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _create_or_reuse_identical(path: Path, content: bytes) -> str:
    if path.is_symlink():
        raise ValueError(f"refusing symlink manifest target: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(
                f"refusing to overwrite non-identical cache manifest: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            return "created"
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise FileExistsError(
                    f"refusing non-identical concurrently created cache manifest: {path}"
                )
            return "reused_identical"
    finally:
        if temporary.exists():
            temporary.unlink()


def build_cache_manifest(
    *,
    cache_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    chunk_rows: int = 65536,
    progress_every: int = 100,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    """Build a create-only manifest for metadata-selected cache datasets."""

    cache = resolve_existing_file(cache_path, key="cache_path")
    metadata_file = resolve_existing_file(metadata_path, key="metadata_path")
    output = Path(output_path).expanduser().resolve(strict=False)
    if output == cache or output == metadata_file:
        raise ValueError("cache manifest output must differ from both inputs")
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if not isinstance(progress_every, int) or progress_every <= 0:
        raise ValueError("progress_every must be a positive integer")
    stream = progress_stream if progress_stream is not None else sys.stderr

    metadata = normalize_manifest_metadata(_read_metadata(metadata_file))
    entries: list[dict[str, Any]] = []
    print(
        f"[p05-cache] hashing {len(metadata)} selected datasets from {cache}",
        file=stream,
        flush=True,
    )
    with h5py.File(cache, "r", libver="latest", swmr=True) as handle:
        for position, row in enumerate(metadata.to_dict(orient="records"), start=1):
            sample_id = int(row["Id"])
            key = str(sample_id)
            if key not in handle:
                raise KeyError(f"cache is missing metadata Id {sample_id}")
            dataset = handle[key]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"cache root key {key!r} is not an HDF5 dataset")
            shape, dtype, dtype_str = _dataset_contract(dataset, sample_id=sample_id)
            entries.append(
                {
                    "Id": sample_id,
                    "Dataset_id": int(row["Dataset_id"]),
                    "Name": str(row["Name"]),
                    "File": str(row["File"]),
                    "shape": shape,
                    "dtype": dtype,
                    "dtype_str": dtype_str,
                    "channel_order": list(row["channel_order"]),
                    "content_sha256": hash_h5_dataset(dataset, chunk_rows=chunk_rows),
                }
            )
            if position % progress_every == 0 or position == len(metadata):
                print(
                    f"[p05-cache] hashed {position}/{len(metadata)} datasets",
                    file=stream,
                    flush=True,
                )
        root_key_count = len(handle)

    value = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "paper_id": "P05",
        "protocol_id": "P05-G040-v3.2",
        "cache": {
            "path": str(cache),
            "format": "HDF5",
            "open_mode": "r",
            "libver": "latest",
            "swmr": True,
            "size_bytes": int(cache.stat().st_size),
            "root_key_count": int(root_key_count),
            "selected_entry_count": len(entries),
        },
        "metadata": {
            "path": str(metadata_file),
            "sha256": sha256_file(metadata_file),
            "selected_row_count": len(metadata),
        },
        "hashing": {
            "algorithm": "sha256",
            "layout": "axis0_chunks_numpy_ascontiguousarray_C_order",
            "chunk_rows": chunk_rows,
        },
        "entries": entries,
    }
    payload = _json_bytes(value)
    status = _create_or_reuse_identical(output, payload)
    print(f"[p05-cache] manifest {status}: {output}", file=stream, flush=True)
    return {
        "path": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "entry_count": len(entries),
        "status": status,
    }


def load_cache_manifest(
    manifest_path: str | Path, *, expected_cache_path: str | Path
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest_file = resolve_existing_file(manifest_path, key="cache_manifest_path")
    cache = resolve_existing_file(expected_cache_path, key="cache_path")
    value = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cache manifest must be a JSON object")
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported cache manifest schema_version")
    if value.get("kind") != MANIFEST_KIND:
        raise ValueError("unexpected cache manifest kind")
    cache_value = value.get("cache")
    if not isinstance(cache_value, dict):
        raise ValueError("cache manifest is missing cache binding")
    if Path(str(cache_value.get("path", ""))).resolve(strict=False) != cache:
        raise ValueError("cache manifest path does not bind the requested cache_path")
    if cache_value.get("open_mode") != "r" or cache_value.get("swmr") is not True:
        raise ValueError("cache manifest must require read-only SWMR access")
    hashing = value.get("hashing")
    if not isinstance(hashing, dict) or hashing.get("algorithm") != "sha256":
        raise ValueError("cache manifest has an invalid hashing contract")
    chunk_rows = hashing.get("chunk_rows")
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("cache manifest chunk_rows must be a positive integer")
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("cache manifest entries must be a list")
    entries: dict[int, dict[str, Any]] = {}
    required = {
        "Id",
        "Dataset_id",
        "Name",
        "File",
        "shape",
        "dtype",
        "dtype_str",
        "channel_order",
        "content_sha256",
    }
    for raw in raw_entries:
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError("cache manifest entry is incomplete")
        sample_id = _integer(raw["Id"], column="Id", row="manifest")
        if sample_id in entries:
            raise ValueError(f"duplicate cache manifest Id {sample_id}")
        if len(str(raw["content_sha256"])) != 64:
            raise ValueError(f"invalid content_sha256 for cache Id {sample_id}")
        entries[sample_id] = raw
    if cache_value.get("selected_entry_count") != len(entries):
        raise ValueError("cache manifest selected_entry_count mismatch")
    return value, entries


def validate_entry_metadata(entry: Mapping[str, Any], metadata_row: Mapping[str, Any]) -> None:
    sample_id = _integer(metadata_row.get("Id"), column="Id", row="active metadata")
    expected = {
        "Id": sample_id,
        "Dataset_id": _integer(
            metadata_row.get("Dataset_id"), column="Dataset_id", row=sample_id
        ),
        "Name": _text(metadata_row.get("Name"), column="Name", row=sample_id),
        "File": _text(metadata_row.get("File"), column="File", row=sample_id),
    }
    for field, expected_value in expected.items():
        actual = entry.get(field)
        if field in {"Id", "Dataset_id"}:
            actual = _integer(actual, column=field, row=f"manifest Id {sample_id}")
        else:
            actual = str(actual)
        if actual != expected_value:
            raise ValueError(
                f"cache manifest metadata mismatch for Id {sample_id}, field {field}: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    expected_channels = list(expected_channel_order(expected["Name"]))
    if entry.get("channel_order") != expected_channels:
        raise ValueError(f"cache manifest channel_order mismatch for Id {sample_id}")


def read_verified_dataset(
    dataset: h5py.Dataset, *, entry: Mapping[str, Any], chunk_rows: int
) -> np.ndarray:
    sample_id = _integer(entry.get("Id"), column="Id", row="manifest")
    actual_shape = [int(value) for value in dataset.shape]
    expected_shape = entry.get("shape")
    if actual_shape != expected_shape:
        raise ValueError(
            f"cache shape mismatch for Id {sample_id}: expected {expected_shape}, "
            f"got {actual_shape}"
        )
    actual_dtype = np.dtype(dataset.dtype)
    if str(actual_dtype) != entry.get("dtype") or actual_dtype.str != entry.get("dtype_str"):
        raise ValueError(f"cache dtype mismatch for Id {sample_id}")
    output = np.empty(tuple(actual_shape), dtype=actual_dtype)
    digest = hashlib.sha256()
    for start in range(0, actual_shape[0], chunk_rows):
        stop = min(start + chunk_rows, actual_shape[0])
        block = np.ascontiguousarray(dataset[start:stop, ...])
        digest.update(block.tobytes(order="C"))
        output[start:stop, ...] = block
    actual_digest = digest.hexdigest()
    if actual_digest != entry.get("content_sha256"):
        raise ValueError(
            f"cache content SHA-256 mismatch for Id {sample_id}: "
            f"expected {entry.get('content_sha256')}, got {actual_digest}"
        )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_cache_manifest(
        cache_path=args.cache,
        metadata_path=args.metadata,
        output_path=args.output,
        chunk_rows=args.chunk_rows,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
