#!/usr/bin/env python3
"""Build the immutable P05 protocol metadata package from a local workbook.

The command is deliberately fail-closed: it verifies the caller-supplied
workbook SHA-256, derives the approved CWRU/XJTU rows, validates the frozen
semantic contract, and only then creates missing outputs.  Existing outputs
are reusable only when their bytes are already identical; this module never
overwrites a differing file and never writes to the source workbook.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


PROTOCOL_COLUMNS = (
    "Id",
    "Dataset_id",
    "Name",
    "File",
    "Original_Label",
    "Protocol_Label",
    "Label",
    "Domain_id",
    "Sample_rate",
    "Protocol_Group",
    "Protocol_Fold",
    "Protocol_Split",
)
INTEGER_COLUMNS = frozenset(
    {
        "Id",
        "Dataset_id",
        "Original_Label",
        "Protocol_Label",
        "Label",
        "Domain_id",
        "Sample_rate",
        "Protocol_Fold",
    }
)
REQUIRED_SOURCE_COLUMNS = frozenset(
    {"Id", "Dataset_id", "Name", "File", "Label", "Domain_id", "Sample_rate"}
)
ROLE_ORDER = ("train", "validation", "test")

CWRU_DATASET_ID = 1
XJTU_DATASET_ID = 2
CWRU_NAME = "RM_001_CWRU"
XJTU_NAME = "RM_002_XJTU"
CWRU_SEED = 20260801
SKLEARN_VERSION = "1.2.2"


@dataclass(frozen=True)
class ExpectedContract:
    """Frozen values that a derived package must match before any write."""

    row_count: int
    payload_bytes: int
    semantic_sha256: str
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class DerivationResult:
    metadata: pd.DataFrame
    summary: Mapping[str, Any]


PRODUCTION_CONTRACT = ExpectedContract(
    row_count=8471,
    payload_bytes=2163841,
    semantic_sha256="87392b6517b6bde753c63a982d998ee5b090ab9ed106f36b294b3ddfdcb3e381",
    summary={
        "CWRU": {
            "source_rows": 155,
            "valid_labeled_rows": 150,
            "ignored_unlabeled_or_minus_one_rows": 5,
            "included_rows": 98,
            "excluded_48000_hz_rows": 52,
            "splits": {
                "train": {
                    "rows": 56,
                    "groups": 56,
                    "class_counts": {"0": 2, "1": 13, "2": 16, "3": 25},
                },
                "validation": {
                    "rows": 19,
                    "groups": 19,
                    "class_counts": {"0": 1, "1": 5, "2": 5, "3": 8},
                },
                "test": {
                    "rows": 23,
                    "groups": 23,
                    "class_counts": {"0": 1, "1": 6, "2": 7, "3": 9},
                },
            },
        },
        "XJTU": {
            "source_rows": 9215,
            "included_rows": 8373,
            "excluded_minus_one_rows": 842,
            "splits": {
                "train": {
                    "rows": 409,
                    "groups": 5,
                    "class_counts": {"0": 337, "1": 72},
                },
                "validation": {
                    "rows": 1317,
                    "groups": 5,
                    "class_counts": {"0": 1071, "1": 246},
                },
                "test": {
                    "rows": 6647,
                    "groups": 5,
                    "class_counts": {"0": 6317, "1": 330},
                },
            },
        },
        "combined_rows": 8471,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, *, column: str, row: Any) -> int:
    if value is None or pd.isna(value) or isinstance(value, bool):
        raise ValueError(f"{column} must be an integer at source row {row!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be an integer at source row {row!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{column} must be an integer at source row {row!r}")
    return int(numeric)


def _optional_integer(value: Any, *, column: str, row: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return _integer(value, column=column, row=row)


def _text(value: Any, *, column: str, row: Any) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{column} must be a non-empty string at source row {row!r}")
    result = str(value)
    if not result.strip():
        raise ValueError(f"{column} must be a non-empty string at source row {row!r}")
    return result


def _xjtu_path_parts(file_value: str, *, row: Any) -> tuple[str, ...]:
    normalized = file_value.replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part)
    if len(parts) < 3 or any(part in {".", ".."} for part in parts):
        raise ValueError(
            f"XJTU File must contain condition/bearing/file at source row {row!r}"
        )
    return parts


def _split_summary(frame: pd.DataFrame, labels: Sequence[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role in ROLE_ORDER:
        selected = frame.loc[frame["Protocol_Split"] == role]
        counts = selected["Protocol_Label"].value_counts().to_dict()
        output[role] = {
            "rows": int(len(selected)),
            "groups": int(selected["Protocol_Group"].nunique()),
            "class_counts": {str(label): int(counts.get(label, 0)) for label in labels},
        }
    return output


def _validate_source_frame(source: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(source.columns))
    if missing:
        raise ValueError(f"source workbook is missing required columns: {missing}")
    if source.empty:
        raise ValueError("source workbook contains no rows")

    normalized = source.copy()
    normalized["Id"] = [
        _integer(value, column="Id", row=index)
        for index, value in normalized["Id"].items()
    ]
    normalized["Dataset_id"] = [
        _integer(value, column="Dataset_id", row=index)
        for index, value in normalized["Dataset_id"].items()
    ]
    duplicates = normalized.loc[normalized["Id"].duplicated(keep=False), "Id"].tolist()
    if duplicates:
        raise ValueError(f"duplicate Id records in source workbook: {sorted(set(duplicates))[:10]}")
    target_rows = normalized.loc[normalized["Dataset_id"].isin([CWRU_DATASET_ID, XJTU_DATASET_ID])]
    duplicate_files = target_rows.duplicated(
        subset=["Dataset_id", "Name", "File"], keep=False
    )
    if duplicate_files.any():
        examples = target_rows.loc[
            duplicate_files, ["Dataset_id", "Name", "File"]
        ].head(10)
        raise ValueError(
            "duplicate target source records: "
            f"{examples.to_dict(orient='records')}"
        )
    return normalized


def _derive_cwru(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = source.loc[source["Dataset_id"] == CWRU_DATASET_ID].copy()
    rows: list[dict[str, Any]] = []
    valid_labeled = 0
    ignored = 0
    excluded_48000 = 0

    for index, row in raw.iterrows():
        row_id = int(row["Id"])
        name = _text(row["Name"], column="Name", row=row_id)
        if name != CWRU_NAME:
            raise ValueError(f"unexpected CWRU Name {name!r} for Id {row_id}")
        file_value = _text(row["File"], column="File", row=row_id)
        label = _optional_integer(row["Label"], column="Label", row=row_id)
        if label is None or label == -1:
            ignored += 1
            continue
        if label not in {0, 1, 2, 3}:
            raise ValueError(f"unknown CWRU label {label} for Id {row_id}")
        valid_labeled += 1
        sample_rate = _integer(row["Sample_rate"], column="Sample_rate", row=row_id)
        if sample_rate not in {12000, 48000}:
            raise ValueError(f"unexpected CWRU Sample_rate {sample_rate} for Id {row_id}")
        if sample_rate == 48000:
            excluded_48000 += 1
            continue
        domain_id = _integer(row["Domain_id"], column="Domain_id", row=row_id)
        if domain_id not in {0, 1, 2, 3}:
            raise ValueError(f"unexpected CWRU Domain_id {domain_id} for Id {row_id}")
        rows.append(
            {
                "Id": row_id,
                "Dataset_id": CWRU_DATASET_ID,
                "Name": name,
                "File": file_value,
                "Original_Label": label,
                "Protocol_Label": label,
                "Label": label,
                "Domain_id": domain_id,
                "Sample_rate": sample_rate,
                "Protocol_Group": f"CWRU/{file_value}",
                "Protocol_Fold": -1,
                "Protocol_Split": "",
            }
        )

    derived = pd.DataFrame(rows, columns=PROTOCOL_COLUMNS)
    if derived.empty:
        raise ValueError("CWRU protocol selection is empty")
    if derived["Protocol_Group"].duplicated().any():
        groups = derived.loc[derived["Protocol_Group"].duplicated(keep=False), "Protocol_Group"]
        raise ValueError(f"duplicate CWRU File groups: {sorted(set(groups))[:10]}")

    source_rows = derived.loc[derived["Domain_id"].isin([0, 1, 2])].copy()
    test_rows = derived.loc[derived["Domain_id"] == 3].copy()
    if len(source_rows) + len(test_rows) != len(derived):
        raise ValueError("CWRU rows contain an unassigned protocol domain")
    source_rows = source_rows.sort_values("Protocol_Group", kind="mergesort")
    train_groups, validation_groups = train_test_split(
        source_rows["Protocol_Group"].tolist(),
        test_size=0.25,
        shuffle=True,
        random_state=CWRU_SEED,
        stratify=source_rows["Protocol_Label"].tolist(),
    )
    train_set = set(train_groups)
    validation_set = set(validation_groups)
    if train_set & validation_set:
        raise ValueError("CWRU train/validation group overlap")
    derived.loc[derived["Protocol_Group"].isin(train_set), "Protocol_Split"] = "train"
    derived.loc[
        derived["Protocol_Group"].isin(validation_set), "Protocol_Split"
    ] = "validation"
    derived.loc[derived["Domain_id"] == 3, "Protocol_Split"] = "test"
    if set(derived["Protocol_Split"]) != set(ROLE_ORDER):
        raise ValueError("CWRU split assignment is incomplete")

    summary = {
        "source_rows": int(len(raw)),
        "valid_labeled_rows": valid_labeled,
        "ignored_unlabeled_or_minus_one_rows": ignored,
        "included_rows": int(len(derived)),
        "excluded_48000_hz_rows": excluded_48000,
        "splits": _split_summary(derived, [0, 1, 2, 3]),
    }
    return derived, summary


def _derive_xjtu(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = source.loc[source["Dataset_id"] == XJTU_DATASET_ID].copy()
    condition_by_domain = {0: "35Hz12kN", 1: "37.5Hz11kN", 2: "40Hz10kN"}
    role_by_domain = {0: "train", 1: "validation", 2: "test"}
    rows: list[dict[str, Any]] = []
    excluded_minus_one = 0

    for index, row in raw.iterrows():
        row_id = int(row["Id"])
        name = _text(row["Name"], column="Name", row=row_id)
        if name != XJTU_NAME:
            raise ValueError(f"unexpected XJTU Name {name!r} for Id {row_id}")
        file_value = _text(row["File"], column="File", row=row_id)
        label = _optional_integer(row["Label"], column="Label", row=row_id)
        if label is None:
            raise ValueError(f"XJTU Label is missing for Id {row_id}")
        if label == -1:
            excluded_minus_one += 1
            continue
        if label < 0 or label > 15:
            raise ValueError(f"unknown XJTU label {label} for Id {row_id}")
        domain_id = _integer(row["Domain_id"], column="Domain_id", row=row_id)
        if domain_id not in role_by_domain:
            raise ValueError(f"unexpected XJTU Domain_id {domain_id} for Id {row_id}")
        sample_rate = _integer(row["Sample_rate"], column="Sample_rate", row=row_id)
        parts = _xjtu_path_parts(file_value, row=row_id)
        if parts[0] != condition_by_domain[domain_id]:
            raise ValueError(
                f"XJTU File condition {parts[0]!r} disagrees with Domain_id {domain_id} "
                f"for Id {row_id}"
            )
        protocol_label = 0 if label == 0 else 1
        rows.append(
            {
                "Id": row_id,
                "Dataset_id": XJTU_DATASET_ID,
                "Name": name,
                "File": file_value,
                "Original_Label": label,
                "Protocol_Label": protocol_label,
                "Label": protocol_label,
                "Domain_id": domain_id,
                "Sample_rate": sample_rate,
                "Protocol_Group": f"XJTU/{parts[0]}/{parts[1]}",
                "Protocol_Fold": -1,
                "Protocol_Split": role_by_domain[domain_id],
            }
        )

    derived = pd.DataFrame(rows, columns=PROTOCOL_COLUMNS)
    if derived.empty:
        raise ValueError("XJTU protocol selection is empty")
    group_domains = derived.groupby("Protocol_Group", sort=False)["Domain_id"].nunique()
    if (group_domains != 1).any():
        raise ValueError("an XJTU bearing group spans multiple Domain_id values")
    if derived["Protocol_Group"].nunique() != 15:
        raise ValueError(
            f"XJTU protocol requires exactly 15 bearing groups, got "
            f"{derived['Protocol_Group'].nunique()}"
        )
    positive_label_by_group: dict[str, int] = {}
    for group, group_frame in derived.groupby("Protocol_Group", sort=False):
        original = set(int(value) for value in group_frame["Original_Label"])
        positives = sorted(value for value in original if value > 0)
        if 0 not in original or len(positives) != 1:
            raise ValueError(
                f"XJTU group {group!r} must contain label 0 and exactly one positive label"
            )
        positive_label_by_group[str(group)] = positives[0]
    if len(set(positive_label_by_group.values())) != 15:
        raise ValueError("XJTU positive bearing labels must be unique across 15 groups")

    summary = {
        "source_rows": int(len(raw)),
        "included_rows": int(len(derived)),
        "excluded_minus_one_rows": excluded_minus_one,
        "splits": _split_summary(derived, [0, 1]),
    }
    return derived, summary


def derive_protocol_metadata(source: pd.DataFrame) -> DerivationResult:
    """Derive the approved P05 metadata table without writing any files."""

    if sklearn.__version__ != SKLEARN_VERSION:
        raise RuntimeError(
            f"P05 metadata split requires scikit-learn {SKLEARN_VERSION}, "
            f"got {sklearn.__version__}"
        )
    normalized = _validate_source_frame(source)
    cwru, cwru_summary = _derive_cwru(normalized)
    xjtu, xjtu_summary = _derive_xjtu(normalized)
    combined = pd.concat([cwru, xjtu], ignore_index=True)
    combined = combined.sort_values(["Dataset_id", "Id"], kind="mergesort").reset_index(
        drop=True
    )
    if combined["Id"].duplicated().any():
        raise ValueError("derived metadata contains duplicate Id records")
    if combined[list(PROTOCOL_COLUMNS)].duplicated().any():
        raise ValueError("derived metadata contains duplicate canonical records")
    return DerivationResult(
        metadata=combined.loc[:, PROTOCOL_COLUMNS],
        summary={
            "CWRU": cwru_summary,
            "XJTU": xjtu_summary,
            "combined_rows": int(len(combined)),
        },
    )


def semantic_metadata_bytes(metadata: pd.DataFrame) -> bytes:
    """Serialize the frozen semantic row list exactly as preregistered."""

    if tuple(metadata.columns) != PROTOCOL_COLUMNS:
        raise ValueError(
            f"canonical columns differ: expected {list(PROTOCOL_COLUMNS)}, "
            f"got {list(metadata.columns)}"
        )
    records: list[dict[str, Any]] = []
    for values in metadata.itertuples(index=False, name=None):
        record: dict[str, Any] = {}
        for column, value in zip(PROTOCOL_COLUMNS, values):
            record[column] = int(value) if column in INTEGER_COLUMNS else str(value)
        records.append(record)
    return json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_csv_bytes(metadata: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(PROTOCOL_COLUMNS),
        extrasaction="raise",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for values in metadata.itertuples(index=False, name=None):
        writer.writerow(
            {
                column: int(value) if column in INTEGER_COLUMNS else str(value)
                for column, value in zip(PROTOCOL_COLUMNS, values)
            }
        )
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _split_manifest(
    metadata: pd.DataFrame,
    *,
    dataset_id: int,
    dataset_name: str,
    metadata_semantic_sha256: str,
) -> dict[str, Any]:
    selected = metadata.loc[metadata["Dataset_id"] == dataset_id]
    roles: dict[str, Any] = {}
    group_sets: dict[str, set[str]] = {}
    for role in ROLE_ORDER:
        role_frame = selected.loc[selected["Protocol_Split"] == role].sort_values(
            "Id", kind="mergesort"
        )
        rows: list[dict[str, Any]] = []
        for values in role_frame.itertuples(index=False, name=None):
            rows.append(
                {
                    column: int(value) if column in INTEGER_COLUMNS else str(value)
                    for column, value in zip(PROTOCOL_COLUMNS, values)
                }
            )
        groups = sorted(set(role_frame["Protocol_Group"].astype(str)), key=str)
        group_sets[role] = set(groups)
        class_counts = role_frame["Protocol_Label"].value_counts().sort_index().to_dict()
        roles[role] = {
            "row_count": int(len(role_frame)),
            "ids": [int(value) for value in role_frame["Id"]],
            "groups": groups,
            "class_counts": {str(int(key)): int(value) for key, value in class_counts.items()},
            "rows": rows,
        }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(f"group leakage between {left} and {right}: {sorted(overlap)}")
    return {
        "schema_version": 1,
        "paper_id": "P05",
        "protocol_id": "P05-G040-v3.2",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "metadata_semantic_sha256": metadata_semantic_sha256,
        "protocol_fold": -1,
        "role_key": "Protocol_Split",
        "group_key": "Protocol_Group",
        "label_key": "Label",
        "roles": roles,
    }


def _validate_expected(
    result: DerivationResult,
    semantic_payload: bytes,
    expected: ExpectedContract,
) -> str:
    actual_digest = hashlib.sha256(semantic_payload).hexdigest()
    if len(result.metadata) != expected.row_count:
        raise ValueError(
            f"derived row count mismatch: expected {expected.row_count}, "
            f"got {len(result.metadata)}"
        )
    if len(semantic_payload) != expected.payload_bytes:
        raise ValueError(
            f"semantic payload byte count mismatch: expected {expected.payload_bytes}, "
            f"got {len(semantic_payload)}"
        )
    if actual_digest != expected.semantic_sha256:
        raise ValueError(
            f"semantic SHA-256 mismatch: expected {expected.semantic_sha256}, "
            f"got {actual_digest}"
        )
    if result.summary != expected.summary:
        raise ValueError(
            "protocol count summary mismatch: expected "
            f"{json.dumps(expected.summary, sort_keys=True)}, got "
            f"{json.dumps(result.summary, sort_keys=True)}"
        )
    return actual_digest


def _write_missing_outputs(
    outputs: Mapping[Path, bytes], *, source_workbook: Path
) -> dict[str, str]:
    source_resolved = source_workbook.resolve()
    resolved_targets: dict[Path, Path] = {}
    for target in outputs:
        resolved = target.resolve(strict=False)
        if resolved == source_resolved:
            raise ValueError(f"output path must not be the source workbook: {target}")
        if resolved in resolved_targets.values():
            raise ValueError(f"output paths must be distinct: {target}")
        resolved_targets[target] = resolved

    statuses: dict[str, str] = {}
    missing: list[tuple[Path, bytes]] = []
    for target, content in outputs.items():
        if target.is_symlink():
            raise ValueError(f"refusing symlink output target: {target}")
        if target.exists():
            if not target.is_file() or target.read_bytes() != content:
                raise FileExistsError(
                    f"refusing to overwrite non-identical existing output: {target}"
                )
            statuses[str(target)] = "reused_identical"
        else:
            missing.append((target, content))

    staged: list[tuple[Path, Path]] = []
    try:
        for target, content in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, target))
        for temporary, target in staged:
            try:
                # A hard link in the same directory is an atomic create that
                # refuses to replace a path another process created after the
                # conflict preflight.  ``os.replace`` would violate the
                # no-overwrite contract under that race.
                os.link(temporary, target)
                statuses[str(target)] = "created"
            except FileExistsError:
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or target.read_bytes() != outputs[target]
                ):
                    raise FileExistsError(
                        "refusing to overwrite non-identical concurrently created "
                        f"output: {target}"
                    )
                statuses[str(target)] = "reused_identical"
            finally:
                if temporary.exists():
                    temporary.unlink()
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()
    return statuses


def build_p05_metadata_package(
    *,
    workbook_path: str | Path,
    expected_workbook_sha256: str,
    output_csv_path: str | Path,
    metadata_manifest_path: str | Path,
    cwru_split_manifest_path: str | Path,
    xjtu_split_manifest_path: str | Path,
    expected_contract: ExpectedContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    """Build or byte-identically reuse the four-file protocol package."""

    workbook = Path(workbook_path)
    if not workbook.is_file():
        raise FileNotFoundError(f"source workbook does not exist: {workbook}")
    expected_source = str(expected_workbook_sha256).strip().lower()
    if len(expected_source) != 64 or any(char not in "0123456789abcdef" for char in expected_source):
        raise ValueError("expected_workbook_sha256 must be 64 lowercase hexadecimal characters")
    actual_source = sha256_file(workbook)
    if actual_source != expected_source:
        raise ValueError(
            f"source workbook SHA-256 mismatch: expected {expected_source}, got {actual_source}"
        )
    try:
        source = pd.read_excel(workbook, engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to read the local workbook") from exc

    result = derive_protocol_metadata(source)
    semantic_payload = semantic_metadata_bytes(result.metadata)
    semantic_digest = _validate_expected(result, semantic_payload, expected_contract)
    csv_payload = canonical_csv_bytes(result.metadata)
    csv_digest = hashlib.sha256(csv_payload).hexdigest()

    cwru_split = _split_manifest(
        result.metadata,
        dataset_id=CWRU_DATASET_ID,
        dataset_name=CWRU_NAME,
        metadata_semantic_sha256=semantic_digest,
    )
    xjtu_split = _split_manifest(
        result.metadata,
        dataset_id=XJTU_DATASET_ID,
        dataset_name=XJTU_NAME,
        metadata_semantic_sha256=semantic_digest,
    )
    cwru_split_payload = _json_bytes(cwru_split)
    xjtu_split_payload = _json_bytes(xjtu_split)

    output_csv = Path(output_csv_path)
    metadata_manifest = Path(metadata_manifest_path)
    cwru_manifest = Path(cwru_split_manifest_path)
    xjtu_manifest = Path(xjtu_split_manifest_path)
    manifest_value = {
        "schema_version": 2,
        "paper_id": "P05",
        "protocol_id": "P05-G040-v3.2",
        "source_workbook": {
            "path": str(workbook.resolve()),
            "sha256": actual_source,
            "size_bytes": int(workbook.stat().st_size),
        },
        "generator": {
            "name": "scripts/build_p05_metadata.py",
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "cwru_split_seed": CWRU_SEED,
        },
        "derived_metadata": {
            "file": output_csv.name,
            "csv_sha256": csv_digest,
            "row_count": int(len(result.metadata)),
            "columns": list(PROTOCOL_COLUMNS),
            "integer_columns": sorted(INTEGER_COLUMNS),
            "semantic_serialization": {
                "top_level": "bare_list_of_row_objects",
                "sort": "stable_mergesort_by_Dataset_id_then_Id",
                "sort_keys": True,
                "separators": [",", ":"],
                "ensure_ascii": True,
                "allow_nan": False,
                "payload_bytes": len(semantic_payload),
                "sha256": semantic_digest,
            },
        },
        "summary": result.summary,
        "split_manifests": {
            "CWRU": {
                "file": cwru_manifest.name,
                "sha256": hashlib.sha256(cwru_split_payload).hexdigest(),
            },
            "XJTU": {
                "file": xjtu_manifest.name,
                "sha256": hashlib.sha256(xjtu_split_payload).hexdigest(),
            },
        },
    }
    metadata_manifest_payload = _json_bytes(manifest_value)

    if sha256_file(workbook) != actual_source:
        raise RuntimeError("source workbook changed while the protocol package was being derived")
    statuses = _write_missing_outputs(
        {
            output_csv: csv_payload,
            metadata_manifest: metadata_manifest_payload,
            cwru_manifest: cwru_split_payload,
            xjtu_manifest: xjtu_split_payload,
        },
        source_workbook=workbook,
    )
    return {
        "workbook_sha256": actual_source,
        "metadata_semantic_sha256": semantic_digest,
        "metadata_csv_sha256": csv_digest,
        "row_count": int(len(result.metadata)),
        "outputs": statuses,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True, help="Read-only source metadata workbook")
    parser.add_argument(
        "--workbook-sha256",
        required=True,
        help="Required expected SHA-256 for the source workbook",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metadata-manifest", required=True)
    parser.add_argument("--cwru-split-manifest", required=True)
    parser.add_argument("--xjtu-split-manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_p05_metadata_package(
        workbook_path=args.workbook,
        expected_workbook_sha256=args.workbook_sha256,
        output_csv_path=args.output_csv,
        metadata_manifest_path=args.metadata_manifest,
        cwru_split_manifest_path=args.cwru_split_manifest,
        xjtu_split_manifest_path=args.xjtu_split_manifest,
        expected_contract=PRODUCTION_CONTRACT,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
