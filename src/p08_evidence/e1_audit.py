"""Independent, fail-closed audit of one materialized P08 E1 run.

The runner is not an authority for this module.  Every L01--L11 decision is
reconstructed from regular files below the run directory.  In particular, a
recorded ``status: pass`` or a bare boolean is never sufficient evidence.

The caller supplies the digest map it intends to retain.  The map must be
non-empty and must agree exactly with the independently reopened
``artifact_manifest.sha256``.  The manifest, in turn, must cover every regular
file other than itself.  This makes an in-memory runner assertion useful only
as a claim to be checked, never as the check itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from hashlib import sha256
import hmac
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Literal

import numpy as np
import yaml

from src.p08_evidence.e1_data import (
    CLASS_IDS,
    EVALUATION_RATES_HZ,
    GENERATOR_VERSION,
    PROTOCOL_ID,
    SOURCE_SHARED_BAND_HZ,
    canonical_json_sha256,
    iter_rate_copies,
    split_underlying_ids,
)
from src.p08_evidence.runtime import (
    ALLOWED_PHYSICAL_GPU_INDICES,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


EXPERIMENT_ID = "P08-E1"
PARTITION_HASH_FIELD = "partition_id_set_sha256"
SEALED_LABEL_COPY_NAME = "sealed_label_table_after_prediction_hashes.json"
# A deterministic engine-owned fixture exercises the audit without pretending
# to be the paper authority. Formal runs bind the path and digest declared in
# their resolved configuration and retain that authority source separately.
PROTOCOL_FIXTURE_BYTES = b"protocol_id: P08-LOSO-v1.1\nfixture_scope: audit_unit_test_only\n"
PROTOCOL_SOURCE_SHA256 = sha256(PROTOCOL_FIXTURE_BYTES).hexdigest()
PROTOCOL_SOURCE_PATH = "protocol/p08_e1_protocol_fixture.yaml"
PROTOCOL_SNAPSHOT_PATH = "protocol_snapshot/p08_e1_protocol.yaml"
AuditRunState = Literal["scored_pending_final_audit", "completed"]
_AUDIT_RUN_STATES = frozenset({"scored_pending_final_audit", "completed"})

# ``leakage_audit.json`` is the output of this module and therefore cannot be
# an input required to compute itself.  A retained final run may include it;
# manifest coverage below will verify it like every other extra artifact.
REQUIRED_EVIDENCE_FILES = frozenset(
    {
        "resolved_config.yaml",
        "command.txt",
        "provenance.json",
        "environment.yml",
        "source_manifest.json",
        PROTOCOL_SNAPSHOT_PATH,
        "fold_manifest.json",
        "partition_disjointness.json",
        "data_manifest_pretest.json",
        "loader_partition_log.json",
        "training_input_schema.json",
        "normalization.json",
        "normalization_recompute.json",
        "source_sampling_rate_table.json",
        "contract_checks.json",
        "epoch_log.jsonl",
        "collation_assertion_log.jsonl",
        "selection_trace.jsonl",
        "selected.ckpt",
        "checkpoint.sha256",
        "target_eval_manifest.json",
        "target_decode_log.json",
        "window_predictions.parquet",
        "record_predictions.parquet",
        "prediction.sha256",
        SEALED_LABEL_COPY_NAME,
        "scored_records.parquet",
        "scorer_join_log.json",
        "metrics.json",
        "run_status.json",
        "stdout.log",
        "stderr.log",
    }
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_SELECTION_KEY_TOKENS = frozenset(
    {"test", "target", "holdout", "heldout"}
)
_FORBIDDEN_UNLABELED_COLUMNS = frozenset(
    {
        "class_id",
        "label",
        "labels",
        "target_label",
        "true_label",
        "ground_truth",
        "underlying_id",
        "source_signal_handle",
        "source_underlying_id",
    }
)
_DISTRIBUTED_MARKERS = ("ddp", "fsdp", "deepspeed")


class AuditEvidenceError(RuntimeError):
    """A materialized artifact does not establish its claimed assertion."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditEvidenceError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditEvidenceError(f"expected JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise AuditEvidenceError(
                f"{path.name}:{line_number} is not a JSON object"
            )
        rows.append(value)
    if not rows:
        raise AuditEvidenceError(f"required JSONL artifact is empty: {path.name}")
    return rows


def _parse_time(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise AuditEvidenceError(f"{field} is missing")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise AuditEvidenceError(f"{field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AuditEvidenceError(f"{field} must include an explicit UTC offset")
    return parsed


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value.lower()) is not None


def _safe_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise AuditEvidenceError(f"unsafe artifact path in manifest: {value!r}")
    if "\n" in value or "\r" in value:
        raise AuditEvidenceError("artifact manifest path contains a newline")
    return relative


def partition_id_set_sha256(pairs: Sequence[tuple[int, int]]) -> str:
    """Hash a partition as sorted ``[class_id, underlying_id]`` identities."""

    normalized = sorted([int(class_id), int(underlying_id)] for class_id, underlying_id in pairs)
    if len({(row[0], row[1]) for row in normalized}) != len(normalized):
        raise ValueError("partition identity list contains duplicates")
    return sha256_bytes(canonical_json_bytes(normalized))


def _signal_handle(class_id: int, underlying_id: int) -> str:
    return canonical_json_sha256(
        {
            "generator_version": GENERATOR_VERSION,
            "class_id": int(class_id),
            "underlying_id": int(underlying_id),
        }
    )


def _verify_manifest_and_claimed_digests(
    run_root: Path, artifact_digests: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    if not isinstance(artifact_digests, Mapping) or not artifact_digests:
        errors.append("artifact_digests must be a non-empty mapping")
        return {}, errors

    manifest_path = run_root / "artifact_manifest.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        errors.append("artifact_manifest.sha256 is missing, non-regular, or symlinked")
        return {}, errors

    entries: dict[str, str] = {}
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return {}, [f"cannot read artifact manifest: {exc}"]
    if not lines:
        errors.append("artifact manifest is empty")
    for line_number, line in enumerate(lines, start=1):
        if "  " not in line:
            errors.append(f"malformed artifact manifest line {line_number}")
            continue
        digest, relative_text = line.split("  ", 1)
        digest = digest.lower()
        try:
            relative = _safe_relative(relative_text)
        except AuditEvidenceError as exc:
            errors.append(str(exc))
            continue
        relative_name = relative.as_posix()
        if not _valid_digest(digest):
            errors.append(f"invalid SHA-256 for {relative_name}")
            continue
        if relative_name == "artifact_manifest.sha256":
            errors.append("artifact manifest must not hash itself")
            continue
        if relative_name in entries:
            errors.append(f"duplicate artifact manifest entry: {relative_name}")
            continue
        candidate = run_root / relative
        if not candidate.is_file() or candidate.is_symlink():
            errors.append(f"manifest artifact missing/non-regular: {relative_name}")
            continue
        try:
            observed = sha256_file(candidate)
        except OSError as exc:
            errors.append(f"cannot hash {relative_name}: {exc}")
            continue
        if observed != digest:
            errors.append(
                f"artifact hash mismatch for {relative_name}: "
                f"expected={digest}, observed={observed}"
            )
        entries[relative_name] = digest

    actual: set[str] = set()
    if run_root.is_dir():
        for candidate in run_root.rglob("*"):
            if candidate.is_symlink():
                errors.append(
                    "symlink is forbidden in evidence run: "
                    + candidate.relative_to(run_root).as_posix()
                )
            elif candidate.is_file() and candidate.name != "artifact_manifest.sha256":
                actual.add(candidate.relative_to(run_root).as_posix())
    if actual != set(entries):
        errors.append(
            "artifact manifest coverage mismatch: "
            f"unhashed={sorted(actual - set(entries))}, "
            f"stale={sorted(set(entries) - actual)}"
        )

    missing_required = sorted(REQUIRED_EVIDENCE_FILES - set(entries))
    if missing_required:
        errors.append(f"required evidence artifacts are missing: {missing_required}")

    claimed = {str(key): str(value).lower() for key, value in artifact_digests.items()}
    if set(claimed) != set(entries):
        errors.append(
            "artifact_digests coverage differs from disk manifest: "
            f"missing={sorted(set(entries)-set(claimed))}, "
            f"stale={sorted(set(claimed)-set(entries))}"
        )
    for relative, digest in entries.items():
        claimed_digest = claimed.get(relative)
        if claimed_digest is None:
            continue
        if not _valid_digest(claimed_digest) or claimed_digest != digest:
            errors.append(f"artifact_digests mismatch for {relative}")

    # Dirty status is itself evidence.  A non-empty status without retained
    # patch content is not reproducible even if every existing file is hashed.
    for status_name, patch_name in (
        ("dirty_status.txt", "dirty.patch"),
        ("paper_dirty_status.txt", "paper_dirty.patch"),
    ):
        status_path = run_root / status_name
        if status_path.is_file() and status_path.read_text(encoding="utf-8").strip():
            if patch_name not in entries:
                errors.append(f"{status_name} is non-empty but {patch_name} is absent")
            elif not (run_root / patch_name).read_bytes():
                errors.append(f"{patch_name} is empty")
    return entries, errors


def _protocol_identity(payload: Mapping[str, Any], *, name: str) -> None:
    if "protocol_id" in payload:
        _require(payload.get("protocol_id") == PROTOCOL_ID, f"{name} protocol mismatch")
    if "experiment_id" in payload:
        _require(
            payload.get("experiment_id") == EXPERIMENT_ID,
            f"{name} experiment mismatch",
        )


def _fold_pairs(
    fold: Mapping[str, Any], split_name: str
) -> set[tuple[int, int]]:
    by_class = fold.get("training_and_validation_underlying_ids_by_class")
    _require(isinstance(by_class, Mapping), "fold manifest lacks source partition IDs")
    result: set[tuple[int, int]] = set()
    for class_id in CLASS_IDS:
        class_entry = by_class.get(str(class_id))
        _require(isinstance(class_entry, Mapping), f"fold manifest lacks class {class_id}")
        values = class_entry.get(split_name)
        _require(isinstance(values, list), f"fold manifest lacks {split_name} IDs")
        _require(
            all(_is_int(value) for value in values),
            f"{split_name} identities must be integers",
        )
        _require(len(values) == len(set(values)), f"duplicate {split_name} identities")
        expected = set(split_underlying_ids(class_id)[split_name])
        observed = set(int(value) for value in values)
        _require(
            observed == expected,
            f"{split_name} identities for class {class_id} differ from frozen split",
        )
        result.update((class_id, value) for value in observed)
    return result


def _expected_test_pairs() -> set[tuple[int, int]]:
    return {
        (class_id, underlying_id)
        for class_id in CLASS_IDS
        for underlying_id in split_underlying_ids(class_id)["test"]
    }


def _sealed_target_mapping(
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    """Reopen the post-prediction sealed-table copy and verify HMAC identities."""

    sealed = _read_json(run_root / SEALED_LABEL_COPY_NAME)
    _protocol_identity(sealed, name=SEALED_LABEL_COPY_NAME)
    _require(sealed.get("status") == "sealed", "sealed label copy status differs")
    _require(
        sealed.get("tokenization") == "HMAC-SHA256",
        "sealed target tokenization differs",
    )
    salt_hex = sealed.get("token_salt_hex")
    _require(
        isinstance(salt_hex, str)
        and len(salt_hex) == 64
        and all(character in "0123456789abcdef" for character in salt_hex),
        "sealed token salt is not a lowercase 256-bit value",
    )
    salt = bytes.fromhex(salt_hex)
    raw_entries = sealed.get("entries")
    _require(
        isinstance(raw_entries, list) and len(raw_entries) == len(_expected_test_pairs()),
        "sealed label copy must contain exactly 204 identities",
    )
    expected_keys = {
        "target_handle",
        "class_id",
        "source_signal_handle",
        "source_underlying_id",
        "opaque_signal_index",
    }
    by_handle: dict[str, dict[str, Any]] = {}
    observed_pairs: set[tuple[int, int]] = set()
    for raw in raw_entries:
        _require(isinstance(raw, Mapping), "sealed label entry is not an object")
        _require(set(raw) == expected_keys, "sealed label entry schema differs")
        class_id = raw.get("class_id")
        underlying_id = raw.get("source_underlying_id")
        opaque_index = raw.get("opaque_signal_index")
        _require(_is_int(class_id) and int(class_id) in CLASS_IDS, "sealed class ID is invalid")
        _require(_is_int(underlying_id), "sealed source underlying ID is invalid")
        _require(_is_int(opaque_index), "sealed opaque signal index is invalid")
        source_handle = str(raw.get("source_signal_handle", ""))
        target_handle = str(raw.get("target_handle", ""))
        expected_source_handle = _signal_handle(int(class_id), int(underlying_id))
        _require(source_handle == expected_source_handle, "sealed source-handle mapping differs")
        expected_target_handle = hmac.new(
            salt, source_handle.encode("utf-8"), sha256
        ).hexdigest()
        _require(target_handle == expected_target_handle, "sealed HMAC target handle differs")
        _require(target_handle not in by_handle, "duplicate sealed target handle")
        by_handle[target_handle] = dict(raw)
        observed_pairs.add((int(class_id), int(underlying_id)))

    expected_pairs = _expected_test_pairs()
    _require(observed_pairs == expected_pairs, "sealed identities differ from frozen test split")
    sorted_handles = sorted(by_handle)
    _require(
        [by_handle[handle]["opaque_signal_index"] for handle in sorted_handles]
        == list(range(len(sorted_handles))),
        "opaque indices are not the rank of lexicographically sorted HMAC handles",
    )
    sorted_mapping = [by_handle[handle] for handle in sorted_handles]
    commitments = {
        "target_handle_set_sha256": sha256_bytes(
            canonical_json_bytes(sorted_handles)
        ),
        "frozen_test_pair_set_sha256": partition_id_set_sha256(
            tuple(expected_pairs)
        ),
        "mapping_commitment_sha256": sha256_bytes(
            canonical_json_bytes(sorted_mapping)
        ),
    }
    _require(sealed.get("entry_count") == len(by_handle), "sealed entry count differs")
    return sealed, by_handle, commitments


def _check_l01(run_root: Path) -> dict[str, Any]:
    fold = _read_json(run_root / "fold_manifest.json")
    partition = _read_json(run_root / "partition_disjointness.json")
    pretest = _read_json(run_root / "data_manifest_pretest.json")
    target = _read_json(run_root / "target_eval_manifest.json")
    decode = _read_json(run_root / "target_decode_log.json")
    _, sealed_by_handle, commitments = _sealed_target_mapping(run_root)
    for name, value in (
        ("fold_manifest", fold),
        ("data_manifest_pretest", pretest),
        ("target_eval_manifest", target),
        ("target_decode_log", decode),
    ):
        _protocol_identity(value, name=name)

    train = _fold_pairs(fold, "train")
    validation = _fold_pairs(fold, "validation")
    test = _expected_test_pairs()
    overlaps = {
        "train_vs_validation": len(train & validation),
        "train_vs_test": len(train & test),
        "validation_vs_test": len(validation & test),
    }
    _require(all(value == 0 for value in overlaps.values()), "partition overlap exists")

    for commitment_name, expected_digest in commitments.items():
        _require(
            target.get(commitment_name) == expected_digest,
            f"target manifest {commitment_name} differs",
        )
        _require(
            decode.get(commitment_name) == expected_digest,
            f"target decode {commitment_name} differs",
        )
    for identity_hash in ("broker_manifest_sha256", "shared_native_payload_sha256"):
        observed = target.get(identity_hash)
        _require(_valid_digest(observed), f"target {identity_hash} is absent")
        _require(
            decode.get(identity_hash) == observed,
            f"target/decode {identity_hash} binding differs",
        )

    entries = target.get("entries")
    _require(isinstance(entries, list) and entries, "target manifest has no entries")
    allowed_target_entry_fields = {
        "signal_handle",
        "opaque_signal_index",
        "original_rate_hz",
        "model_rate_numerator_hz",
        "model_rate_denominator",
        "sample_count",
    }
    rates_by_handle: dict[str, set[int]] = {}
    index_by_handle: dict[str, int] = {}
    for entry in entries:
        _require(isinstance(entry, Mapping), "target entry is not an object")
        _require(
            set(entry) == allowed_target_entry_fields,
            "target manifest entry exposes source/label fields or changed schema",
        )
        handle = str(entry.get("signal_handle", ""))
        opaque_index = entry.get("opaque_signal_index")
        rate = entry.get("original_rate_hz")
        _require(handle in sealed_by_handle, "unknown HMAC target handle")
        _require(_is_int(opaque_index), "target opaque_signal_index is invalid")
        _require(
            int(opaque_index) == sealed_by_handle[handle]["opaque_signal_index"],
            "target handle-to-opaque-index mapping differs",
        )
        prior_index = index_by_handle.setdefault(handle, int(opaque_index))
        _require(prior_index == int(opaque_index), "one handle maps to multiple opaque indices")
        _require(_is_int(rate), "target sampling rate is invalid")
        rates_by_handle.setdefault(handle, set()).add(int(rate))
    _require(
        set(rates_by_handle) == set(sealed_by_handle),
        "target HMAC handle set differs from sealed commitment",
    )
    expected_rates = set(EVALUATION_RATES_HZ)
    _require(
        all(rates == expected_rates for rates in rates_by_handle.values()),
        "one or more target signals lack the exact six-rate grid",
    )
    _require(
        len(entries) == len(sealed_by_handle) * len(EVALUATION_RATES_HZ),
        "target manifest has duplicate or missing rate copies",
    )
    _require(target.get("labels_present") is False, "target manifest exposes labels")
    _require(target.get("source_identity_present") is False, "target manifest exposes source identity")
    _require(decode.get("labels_present") is False, "target decode exposes labels")
    _require(decode.get("source_identity_present") is False, "target decode exposes source identity")

    target_summary = fold.get("target_test")
    _require(isinstance(target_summary, Mapping), "fold target summary is missing")
    _require(
        target_summary.get("frozen_test_pair_set_sha256")
        == commitments["frozen_test_pair_set_sha256"],
        "fold frozen-test partition commitment differs",
    )
    _require(
        target_summary.get("underlying_signal_count") == len(test),
        "sealed target count differs",
    )
    _require(target_summary.get("labels_or_class_counts_visible") is False, "target summary exposes labels")
    _require(
        "opaque_signal_handle_set_sha256" not in target_summary,
        "pre-evaluation fold manifest retains a deterministic target-handle hash",
    )

    expected_hashes = {
        "train": partition_id_set_sha256(tuple(train)),
        "validation": partition_id_set_sha256(tuple(validation)),
        "test": partition_id_set_sha256(tuple(test)),
    }
    recorded_hashes = partition.get(PARTITION_HASH_FIELD)
    if recorded_hashes is None:
        recorded_hashes = partition.get("partition_recording_id_hashes")
    _require(isinstance(recorded_hashes, Mapping), "partition ID hashes are absent")
    _require(dict(recorded_hashes) == expected_hashes, "partition ID hashes differ")
    _require(partition.get("counts") == {key: len(value) for key, value in {
        "train": train, "validation": validation, "test": test
    }.items()}, "partition counts differ")
    _require(partition.get("overlap_counts") == overlaps, "overlap report differs")
    _require(
        partition.get("all_rate_copies_inherit_underlying_split") is True,
        "rate-copy split inheritance is not recorded",
    )

    for split_name in ("train", "validation"):
        split_manifest = pretest.get(split_name)
        _require(isinstance(split_manifest, Mapping), f"pretest {split_name} manifest missing")
        _require(_valid_digest(split_manifest.get("bank_sha256")), f"{split_name} bank hash missing")
    _require(pretest.get("target_state") == "sealed", "pretest target was not sealed")
    return {
        "partition_id_set_sha256": expected_hashes,
        "overlap_counts": overlaps,
        **commitments,
        "target_underlying_signal_count": len(test),
        "target_rate_copy_count": len(entries),
    }


def _check_l02(run_root: Path) -> dict[str, Any]:
    loader = _read_json(run_root / "loader_partition_log.json")
    pretest = _read_json(run_root / "data_manifest_pretest.json")
    visible = loader.get("training_process_visible_splits")
    _require(visible == ["train", "validation"], "training loader visibility differs")
    for field in ("target_dataset_object_count", "target_label_table_count"):
        value = loader.get(field)
        _require(_is_int(value) and value == 0, f"{field} must be observed numeric zero")
    for split_name in ("train", "validation"):
        expected = pretest.get(split_name, {}).get("bank_sha256")
        _require(
            loader.get(f"{split_name}_bank_sha256") == expected and _valid_digest(expected),
            f"loader {split_name} bank hash differs",
        )
        count = loader.get(f"{split_name}_rate_copy_count")
        _require(_is_int(count) and count > 0, f"loader {split_name} count is invalid")
    return {
        "training_process_visible_splits": visible,
        "target_dataset_object_count": 0,
        "target_label_table_count": 0,
    }


def _independent_source_recompute(fold: Mapping[str, Any]) -> dict[str, Any]:
    allowed = _fold_pairs(fold, "train")
    count = 0
    mean = 0.0
    m2 = 0.0
    ordered_digest = sha256()
    bank_digest = sha256()
    bank_identity = {
        "generator_version": GENERATOR_VERSION,
        "split": "train",
        "ordering": ["class_id", "underlying_id", "original_rate_hz"],
    }
    bank_digest.update(canonical_json_bytes(bank_identity))
    rate_counts = {str(rate): 0 for rate in EVALUATION_RATES_HZ}
    rate_copy_count = 0
    bank_sample_count = 0
    for copy in iter_rate_copies(split="train"):
        identity = (int(copy.class_id), int(copy.underlying_id))
        if identity not in allowed:
            continue
        sample_digest = copy.sample_sha256
        ordered_digest.update(
            canonical_json_bytes(
                {
                    "class_id": copy.class_id,
                    "underlying_id": copy.underlying_id,
                    "rate_hz": copy.sample_rate_hz,
                    "sample_sha256": sample_digest,
                }
            )
        )
        bank_digest.update(bytes.fromhex(sample_digest))
        rate_counts[str(copy.sample_rate_hz)] += 1
        rate_copy_count += 1
        bank_sample_count += int(copy.samples.size)
        for raw_value in copy.samples:
            value = float(raw_value)
            count += 1
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
    _require(count >= 2, "independent normalization has fewer than two samples")
    standard_deviation = math.sqrt(m2 / count)
    _require(
        math.isfinite(mean)
        and math.isfinite(standard_deviation)
        and standard_deviation > 0.0,
        "independent normalization is non-finite/degenerate",
    )
    base = {
        "ordered_input_hash": ordered_digest.hexdigest(),
        "sample_count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "algorithm": "deterministic_float64_welford_population_ddof_0",
        "dtype": "float64_fit_float64_apply_then_float32_cast",
        "iteration_order": [
            "class_id_sorted",
            "underlying_id_sorted",
            "exact_sampling_rate_hz_sorted",
            "sample_index_ascending",
        ],
    }
    normalization = {
        **base,
        "canonical_json_sha256": canonical_json_sha256(base),
    }
    return {
        "normalization": normalization,
        "train_bank_sha256": bank_digest.hexdigest(),
        "rate_counts": rate_counts,
        "rate_copy_count": rate_copy_count,
        "sample_count": bank_sample_count,
    }


def _check_l03(run_root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    normalization = _read_json(run_root / "normalization.json")
    recompute = _read_json(run_root / "normalization_recompute.json")
    pretest = _read_json(run_root / "data_manifest_pretest.json")
    expected = source["normalization"]
    _require(normalization == expected, "normalization.json differs from independent Welford recompute")
    _require(recompute.get("original") == expected, "normalization recompute original differs")
    _require(recompute.get("recomputed") == expected, "normalization recompute mapping differs")
    _require(
        recompute.get("regenerated_train_bank_sha256") == source["train_bank_sha256"],
        "normalization recompute train-bank hash differs",
    )
    train_manifest = pretest.get("train")
    _require(isinstance(train_manifest, Mapping), "pretest train manifest is missing")
    _require(
        train_manifest.get("bank_sha256") == source["train_bank_sha256"],
        "pretest train bank differs from independently regenerated bank",
    )
    _require(
        train_manifest.get("rate_copy_count") == source["rate_copy_count"],
        "pretest train rate-copy count differs",
    )
    _require(
        train_manifest.get("sample_count") == source["sample_count"],
        "pretest train sample count differs",
    )
    return {
        "ordered_input_hash": expected["ordered_input_hash"],
        "normalization_canonical_json_sha256": expected["canonical_json_sha256"],
        "regenerated_train_bank_sha256": source["train_bank_sha256"],
    }


def _check_l04(run_root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    table = _read_json(run_root / "source_sampling_rate_table.json")
    resolved = yaml.safe_load((run_root / "resolved_config.yaml").read_text(encoding="utf-8"))
    _require(isinstance(resolved, Mapping), "resolved config is not a mapping")
    counts = source["rate_counts"]
    _require(table.get("scope") == "analytic_train_split_only", "source-rate scope differs")
    _require(table.get("rate_copy_counts_by_hz") == counts, "source-rate counts differ")
    observed_rates = [int(rate) for rate, count in counts.items() if int(count) > 0]
    _require(observed_rates, "no source-training rates were independently observed")
    cutoff = min(rate / 2.0 for rate in observed_rates)
    _require(cutoff == SOURCE_SHARED_BAND_HZ, "independent shared cutoff differs from protocol")
    _require(table.get("stored_shared_cutoff_hz") == cutoff, "stored cutoff differs")
    _require(table.get("recomputed_shared_cutoff_hz") == cutoff, "recorded recompute cutoff differs")
    try:
        config_cutoff = resolved["base_config"]["data"]["generator"]["source_shared_band_hz"]
    except (KeyError, TypeError) as exc:
        raise AuditEvidenceError("resolved config lacks source shared-band cutoff") from exc
    _require(config_cutoff == cutoff, "resolved config cutoff differs")
    return {"source_rate_counts": counts, "independently_recomputed_cutoff_hz": cutoff}


def _key_tokens(key: Any) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(key).lower())
        if token
    }


def _forbidden_keys(value: Any, *, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if _key_tokens(key) & _FORBIDDEN_SELECTION_KEY_TOKENS:
                result.append(child_prefix)
            result.extend(_forbidden_keys(child, prefix=child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_forbidden_keys(child, prefix=f"{prefix}[{index}]"))
    return result


def _checkpoint_digest(run_root: Path) -> str:
    recorded = (run_root / "checkpoint.sha256").read_text(encoding="utf-8").strip().lower()
    _require(_valid_digest(recorded), "checkpoint.sha256 is not a SHA-256")
    observed = sha256_file(run_root / "selected.ckpt")
    _require(recorded == observed, "checkpoint hash differs from selected.ckpt")
    return observed


def _check_l05(run_root: Path) -> dict[str, Any]:
    rows = _read_jsonl(run_root / "selection_trace.jsonl")
    forbidden = _forbidden_keys(rows)
    _require(not forbidden, f"selection trace contains target/test fields: {forbidden}")
    _require(sum(row.get("selected") is True for row in rows) == 1, "selection must retain exactly one candidate")
    for row in rows:
        criterion = str(row.get("selection_criterion", "")).lower()
        _require("validation" in criterion, "selection criterion is not source validation")
    checkpoint_digest = _checkpoint_digest(run_root)
    provenance = _read_json(run_root / "provenance.json")
    target = _read_json(run_root / "target_eval_manifest.json")
    selection_time = max(
        _parse_time(row.get("completed_at_utc"), field="selection completed_at_utc")
        for row in rows
    )
    checkpoint_time = _parse_time(
        provenance.get("checkpoint_written_at_utc"), field="checkpoint_written_at_utc"
    )
    target_time = _parse_time(target.get("written_at_utc"), field="target written_at_utc")
    _require(selection_time < checkpoint_time < target_time, "selection/checkpoint/target recorded order differs")
    selection_mtime = (run_root / "selection_trace.jsonl").stat().st_mtime_ns
    checkpoint_mtime = (run_root / "selected.ckpt").stat().st_mtime_ns
    checkpoint_hash_mtime = (run_root / "checkpoint.sha256").stat().st_mtime_ns
    target_mtime = (run_root / "target_eval_manifest.json").stat().st_mtime_ns
    _require(
        selection_mtime < checkpoint_mtime <= checkpoint_hash_mtime < target_mtime,
        "selection/checkpoint/target filesystem order differs",
    )
    _require(provenance.get("checkpoint_sha256") == checkpoint_digest, "provenance checkpoint hash differs")
    _require(target.get("checkpoint_sha256") == checkpoint_digest, "target manifest checkpoint hash differs")
    _require(
        target.get("unsealed_after") == ["selection_trace_finalized", "checkpoint_sha256_written"],
        "target unseal prerequisites differ",
    )
    return {
        "candidate_count": len(rows),
        "checkpoint_sha256": checkpoint_digest,
        "recorded_order": [selection_time.isoformat(), checkpoint_time.isoformat(), target_time.isoformat()],
        "mtime_ns_order": [selection_mtime, checkpoint_mtime, checkpoint_hash_mtime, target_mtime],
    }


def _contract_event(contract: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    event = contract.get(key)
    if event is None and isinstance(contract.get("events"), list):
        matches = [
            value
            for value in contract["events"]
            if isinstance(value, Mapping) and value.get("check_id") == key
        ]
        _require(len(matches) == 1, f"contract event {key} is missing/duplicated")
        event = matches[0]
    _require(isinstance(event, Mapping), f"contract event {key} must be a structured exception log")
    _require(event.get("rejected") is True, f"contract event {key} did not record rejection")
    _require(event.get("exception_type") == "ValueError", f"contract event {key} exception type differs")
    message = event.get("exception_message")
    _require(isinstance(message, str) and message.strip(), f"contract event {key} has no exception message")
    batch_size = event.get("batch_size")
    _require(_is_int(batch_size) and batch_size > 1, f"contract event {key} lacks a concrete batch size")
    return event


def _check_contract_argument(run_root: Path, key: str, forbidden_argument: str) -> dict[str, Any]:
    contract = _read_json(run_root / "contract_checks.json")
    event = _contract_event(contract, key)
    _require(event.get("forbidden_argument") == forbidden_argument, f"{key} attempted argument differs")
    return {
        "check_id": key,
        "exception_type": event["exception_type"],
        "exception_message_sha256": sha256_bytes(str(event["exception_message"]).encode("utf-8")),
        "forbidden_argument": forbidden_argument,
        "batch_size": event["batch_size"],
    }


def _check_l06(run_root: Path) -> dict[str, Any]:
    return _check_contract_argument(
        run_root, "dataset_id_prompt_rejected", "dataset_ids"
    )


def _check_l07(run_root: Path) -> dict[str, Any]:
    return _check_contract_argument(
        run_root, "system_selected_head_rejected", "system_id"
    )


def _check_l08(run_root: Path) -> dict[str, Any]:
    contract = _read_json(run_root / "contract_checks.json")
    event = _contract_event(contract, "sampling_rate_length_mismatch_rejected")
    _require(
        event.get("forbidden_argument") == "sampling_rate_hz",
        "sampling-rate mismatch event attempted argument differs",
    )
    metadata_count = event.get("metadata_count")
    _require(_is_int(metadata_count) and metadata_count >= 0, "metadata_count is absent")
    _require(metadata_count != event.get("batch_size"), "mismatch exception log does not contain a mismatch")

    rows = _read_jsonl(run_root / "collation_assertion_log.jsonl")
    for index, row in enumerate(rows):
        _require(row.get("batch_original_rate_homogeneous") is True, f"collation row {index} mixed rates")
        for field in (
            "metadata_length_mismatch_count",
            "sampling_rate_scalar_broadcast_count",
        ):
            value = row.get(field)
            _require(_is_int(value) and value == 0, f"collation row {index} {field} is not zero")
        batch_count = row.get("batch_count")
        rate_counts = row.get("rate_batch_counts")
        class_counts = row.get("class_example_counts")
        _require(_is_int(batch_count) and batch_count > 0, f"collation row {index} batch count invalid")
        _require(isinstance(rate_counts, Mapping), f"collation row {index} rate counts absent")
        _require(set(rate_counts) == {str(rate) for rate in EVALUATION_RATES_HZ}, f"collation row {index} rate grid differs")
        _require(all(_is_int(value) and value > 0 for value in rate_counts.values()), f"collation row {index} has empty rate bucket")
        _require(sum(int(value) for value in rate_counts.values()) == batch_count, f"collation row {index} batch totals differ")
        _require(isinstance(class_counts, Mapping) and set(class_counts) == {str(value) for value in CLASS_IDS}, f"collation row {index} class counts differ")
        _require(all(_is_int(value) and value > 0 for value in class_counts.values()), f"collation row {index} lacks a class")
        _require(len(set(int(value) for value in class_counts.values())) == 1, f"collation row {index} is not class balanced")
    return {
        "collation_epoch_rows": len(rows),
        "metadata_length_mismatch_count": 0,
        "sampling_rate_scalar_broadcast_count": 0,
        "exception_type": event["exception_type"],
    }


def _decode_times(decode: Mapping[str, Any]) -> tuple[datetime, datetime]:
    return (
        _parse_time(decode.get("decode_started_at_utc"), field="decode_started_at_utc"),
        _parse_time(decode.get("decode_completed_at_utc"), field="decode_completed_at_utc"),
    )


def _check_l09(run_root: Path) -> dict[str, Any]:
    checkpoint_digest = _checkpoint_digest(run_root)
    provenance = _read_json(run_root / "provenance.json")
    target = _read_json(run_root / "target_eval_manifest.json")
    decode = _read_json(run_root / "target_decode_log.json")
    scorer = _read_json(run_root / "scorer_join_log.json")
    checkpoint_time = _parse_time(provenance.get("checkpoint_written_at_utc"), field="checkpoint_written_at_utc")
    target_time = _parse_time(target.get("written_at_utc"), field="target written_at_utc")
    decode_start, decode_end = _decode_times(decode)
    prediction_time = _parse_time(
        scorer.get("prediction_written_at_utc"), field="prediction_written_at_utc"
    )
    _require(
        checkpoint_time < decode_start <= decode_end < target_time <= prediction_time,
        "checkpoint/target-decode/prediction recorded order differs",
    )
    _require(decode.get("checkpoint_sha256") == checkpoint_digest, "decode checkpoint hash differs")
    _require(
        decode.get("normalization_sha256") == sha256_file(run_root / "normalization.json"),
        "decode normalization hash differs",
    )
    _require(
        target.get("normalization_sha256")
        == sha256_file(run_root / "normalization.json"),
        "target manifest normalization hash differs",
    )
    _require(
        decode.get("target_eval_manifest_sha256") == sha256_file(run_root / "target_eval_manifest.json"),
        "decode target-manifest hash differs",
    )
    _require(decode.get("labels_present") is False, "target decode log exposes labels")
    _require(
        decode.get("source_identity_present") is False,
        "target decode log exposes source identity",
    )
    prediction_digest = sha256_file(run_root / "record_predictions.parquet")
    _require(
        (run_root / "prediction.sha256").read_text(encoding="utf-8").strip()
        == prediction_digest,
        "durable prediction sidecar differs",
    )
    _require(
        (run_root / "checkpoint.sha256").stat().st_mtime_ns
        < (run_root / "target_eval_manifest.json").stat().st_mtime_ns
        < (run_root / "target_decode_log.json").stat().st_mtime_ns
        < (run_root / "record_predictions.parquet").stat().st_mtime_ns
        <= (run_root / "prediction.sha256").stat().st_mtime_ns,
        "checkpoint/target/decode/prediction filesystem order differs",
    )
    return {
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_written_at_utc": checkpoint_time.isoformat(),
        "decode_started_at_utc": decode_start.isoformat(),
        "decode_completed_at_utc": decode_end.isoformat(),
        "prediction_written_at_utc": prediction_time.isoformat(),
    }


def _check_environment_snapshot(
    run_root: Path, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    environment_path = run_root / "environment.yml"
    try:
        document = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AuditEvidenceError("environment.yml cannot be parsed as YAML 1.2/JSON") from exc
    _require(isinstance(document, Mapping), "environment snapshot is not a mapping")
    _require(
        document.get("schema") == "p08.environment-snapshot/v1",
        "environment snapshot schema differs",
    )
    environment = document.get("environment")
    _require(
        isinstance(environment, Mapping)
        and environment.get("name") == "LQ_signal",
        "environment snapshot name differs",
    )
    privacy = document.get("privacy_contract")
    expected_privacy_fields = {
        "absolute_prefix_recorded",
        "channel_or_package_urls_recorded",
        "environment_variables_recorded",
        "host_or_user_identifiers_recorded",
        "timestamps_recorded",
    }
    _require(
        isinstance(privacy, Mapping)
        and set(privacy) == expected_privacy_fields
        and all(value is False for value in privacy.values()),
        "environment privacy contract is incomplete or not all false",
    )
    loaded = document.get("loaded_modules")
    _require(isinstance(loaded, list), "environment loaded_modules is absent")
    expected_modules = ["numpy", "pyarrow", "scipy", "torch"]
    observed_modules: list[str] = []
    for record in loaded:
        _require(isinstance(record, Mapping), "loaded-module record is not an object")
        _require(
            set(record) == {"module", "loaded_path", "sha256"},
            "loaded-module record schema differs",
        )
        module = str(record.get("module", ""))
        loaded_path = str(record.get("loaded_path", ""))
        relative = Path(loaded_path)
        _require(
            loaded_path
            and not relative.is_absolute()
            and ".." not in relative.parts,
            f"loaded module {module!r} path is not prefix-relative",
        )
        _require(_valid_digest(record.get("sha256")), f"loaded module {module!r} hash is invalid")
        observed_modules.append(module)
    _require(observed_modules == expected_modules, "critical loaded-module set/order differs")
    counts = document.get("counts")
    _require(
        isinstance(counts, Mapping)
        and counts.get("loaded_modules") == len(expected_modules),
        "environment loaded-module count differs",
    )
    digest = sha256_file(environment_path)
    _require(
        provenance.get("environment_yml_sha256") == digest,
        "provenance environment_yml_sha256 differs",
    )
    return {
        "schema": document["schema"],
        "environment": "LQ_signal",
        "loaded_modules": observed_modules,
        "environment_yml_sha256": digest,
    }


def _check_protocol_binding(
    run_root: Path,
    provenance: Mapping[str, Any],
    status: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = yaml.safe_load(
        (run_root / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    _require(isinstance(resolved, Mapping), "resolved config is not a mapping")
    try:
        protocol = resolved["base_config"]["protocol"]
    except (KeyError, TypeError) as exc:
        raise AuditEvidenceError("resolved config lacks its protocol binding") from exc
    _require(isinstance(protocol, Mapping), "resolved protocol binding is not a mapping")
    _require(protocol.get("id") == PROTOCOL_ID, "resolved protocol is not P08-LOSO-v1.1")
    protocol_source_path = protocol.get("source_path")
    protocol_source_sha256 = protocol.get("source_sha256")
    _require(
        isinstance(protocol_source_path, str) and bool(protocol_source_path.strip()),
        "resolved protocol source path is absent",
    )
    _require(
        _valid_digest(protocol_source_sha256),
        "resolved protocol source SHA-256 is invalid",
    )

    source_manifest = _read_json(run_root / "source_manifest.json")
    _require(
        set(source_manifest) == {"files", "source_manifest_sha256"},
        "source manifest schema differs",
    )
    files = source_manifest.get("files")
    _require(isinstance(files, list) and files, "source manifest file list is empty")
    self_digest = sha256_bytes(canonical_json_bytes({"files": files}))
    _require(
        source_manifest.get("source_manifest_sha256") == self_digest,
        "source manifest canonical self-hash differs",
    )
    rows: dict[str, Mapping[str, Any]] = {}
    for row in files:
        _require(isinstance(row, Mapping), "source manifest row is not an object")
        _require(set(row) == {"path", "bytes", "sha256"}, "source manifest row schema differs")
        path = str(row.get("path", ""))
        _require(path and path not in rows, "source manifest has an empty/duplicate path")
        _require(_is_int(row.get("bytes")) and int(row["bytes"]) >= 0, "source manifest byte count is invalid")
        _require(_valid_digest(row.get("sha256")), "source manifest row SHA-256 is invalid")
        rows[path] = row
    source_row = rows.get(protocol_source_path)
    _require(source_row is not None, "source manifest omits protocol source")
    _require(
        source_row.get("sha256") == protocol_source_sha256,
        "source manifest protocol hash differs",
    )
    snapshot_path = run_root / PROTOCOL_SNAPSHOT_PATH
    _require(
        sha256_file(snapshot_path) == protocol_source_sha256,
        "retained protocol snapshot hash differs",
    )
    _require(
        source_row.get("bytes") == snapshot_path.stat().st_size,
        "source-manifest/snapshot byte counts differ",
    )
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    _require(
        isinstance(snapshot, Mapping)
        and snapshot.get("protocol_id") == PROTOCOL_ID,
        "retained protocol snapshot is not P08-LOSO-v1.1",
    )
    for name, payload in (("provenance", provenance), ("run_status", status)):
        _require(
            payload.get("protocol_source_sha256") == protocol_source_sha256,
            f"{name} protocol_source_sha256 differs",
        )
    _require(
        provenance.get("source_manifest_sha256") == self_digest,
        "provenance source_manifest_sha256 differs",
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_source_sha256": protocol_source_sha256,
        "source_manifest_sha256": self_digest,
    }


def _check_l10(run_root: Path) -> dict[str, Any]:
    provenance = _read_json(run_root / "provenance.json")
    status = _read_json(run_root / "run_status.json")
    command = (run_root / "command.txt").read_text(encoding="utf-8").strip()
    _require(command.startswith("conda run -n LQ_signal"), "evidence command prefix differs")
    _require(provenance.get("command") == command, "provenance command differs from command.txt")
    _require(provenance.get("conda_environment") == "LQ_signal", "conda environment differs")
    environment_summary = _check_environment_snapshot(run_root, provenance)
    protocol_summary = _check_protocol_binding(run_root, provenance, status)
    observed_indices: dict[str, int] = {}
    for preflight_name in ("gpu_preflight", "evaluation_gpu_preflight"):
        preflight = provenance.get(preflight_name)
        _require(isinstance(preflight, Mapping), f"{preflight_name} record is absent")
        physical = preflight.get("physical_gpu_indices")
        _require(
            isinstance(physical, list)
            and len(physical) == 1
            and _is_int(physical[0]),
            f"{preflight_name} must contain one physical index",
        )
        physical_index = int(physical[0])
        _require(
            physical_index in ALLOWED_PHYSICAL_GPU_INDICES
            and physical_index != 2,
            f"{preflight_name} physical GPU is forbidden/unapproved",
        )
        _require(preflight.get("status") == "pass", f"{preflight_name} status is not pass")
        _require(preflight.get("mode") == "cuda", f"{preflight_name} is not CUDA")
        _require(preflight.get("multi_gpu") is False, f"{preflight_name} reports multi-GPU")
        _require(
            preflight.get("world_size") == 1
            and preflight.get("local_world_size") == 1,
            f"{preflight_name} distributed world size differs",
        )
        strategy = str(preflight.get("trainer_strategy", "")).lower()
        _require(
            not any(marker in strategy for marker in _DISTRIBUTED_MARKERS),
            f"{preflight_name} contains a distributed strategy",
        )
        _require(
            preflight.get("cuda_visible_devices") == str(physical_index),
            f"{preflight_name} CUDA visibility does not identify physical GPU",
        )
        _require(preflight.get("cuda_device_count") == 1, f"{preflight_name} device count is not one")
        _require(
            preflight.get("visible_to_physical_gpu_map") == {"0": physical_index},
            f"{preflight_name} visible-to-physical map differs",
        )
        observed_indices[preflight_name] = physical_index
    return {
        "physical_gpu_indices_by_stage": observed_indices,
        "world_size": 1,
        "local_world_size": 1,
        "multi_gpu": False,
        "environment_snapshot": environment_summary,
        "protocol_binding": protocol_summary,
    }


def _table_dict(path: Path) -> tuple[list[str], dict[str, list[Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - LQ_signal ships pyarrow
        raise AuditEvidenceError("pyarrow is required for evidence audit") from exc
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise AuditEvidenceError(f"cannot reopen parquet artifact {path.name}: {exc}") from exc
    return list(table.schema.names), table.to_pydict()


def _check_unlabeled_columns(columns: Sequence[str], *, name: str) -> None:
    normalized = {str(value).lower() for value in columns}
    forbidden = sorted(normalized & _FORBIDDEN_UNLABELED_COLUMNS)
    _require(not forbidden, f"{name} contains labeled columns: {forbidden}")
    _require("class_id" not in normalized, f"{name} contains class_id")


def _check_l11(
    run_root: Path, *, expected_run_state: AuditRunState
) -> dict[str, Any]:
    schema = _read_json(run_root / "training_input_schema.json")
    loader = _read_json(run_root / "loader_partition_log.json")
    target = _read_json(run_root / "target_eval_manifest.json")
    decode = _read_json(run_root / "target_decode_log.json")
    scorer = _read_json(run_root / "scorer_join_log.json")
    provenance = _read_json(run_root / "provenance.json")
    metrics = _read_json(run_root / "metrics.json")
    status = _read_json(run_root / "run_status.json")
    checkpoint_digest = _checkpoint_digest(run_root)
    _, sealed_by_handle, commitments = _sealed_target_mapping(run_root)

    _require(schema.get("target_object_constructed") is False, "training schema constructed target object")
    _require(
        schema.get("model_input_fields") == ["signal", "sampling_rate_hz"],
        "training model-input schema differs",
    )
    forbidden_fields = set(str(value) for value in schema.get("forbidden_fields", ()))
    _require(
        {"dataset_id", "system_id", "target_signal", "target_label"}.issubset(forbidden_fields),
        "training schema omits a forbidden field",
    )
    _require(loader.get("target_dataset_object_count") == 0, "training loader saw target object")
    _require(loader.get("target_label_table_count") == 0, "training loader saw target labels")
    _require(target.get("labels_present") is False, "target manifest contains labels")
    _require(decode.get("labels_present") is False, "decode stage contains labels")
    _require(target.get("source_identity_present") is False, "target manifest contains source identity")
    _require(decode.get("source_identity_present") is False, "decode stage contains source identity")

    window_columns, window = _table_dict(run_root / "window_predictions.parquet")
    record_columns, predictions = _table_dict(run_root / "record_predictions.parquet")
    scored_columns, scored = _table_dict(run_root / "scored_records.parquet")
    _check_unlabeled_columns(window_columns, name="window_predictions.parquet")
    _check_unlabeled_columns(record_columns, name="record_predictions.parquet")
    _require("class_id" in scored_columns, "scored records lack class_id")
    _require(set(scored_columns) == set(record_columns) | {"class_id"}, "scored table changed non-label columns")
    row_count = len(predictions.get("signal_handle", ()))
    _require(row_count > 0, "record predictions are empty")
    _require(len(scored.get("class_id", ())) == row_count, "scored/prediction row counts differ")
    for column in record_columns:
        _require(predictions[column] == scored[column], f"scorer altered prediction column {column}")
    required_prediction_columns = {
        "signal_handle",
        "opaque_signal_index",
        "original_rate_hz",
        "predicted_class",
        *(f"p_class_{class_id}" for class_id in CLASS_IDS),
    }
    _require(required_prediction_columns.issubset(record_columns), "record prediction schema is incomplete")
    probability = np.asarray(
        [[predictions[f"p_class_{class_id}"][index] for class_id in CLASS_IDS] for index in range(row_count)],
        dtype=np.float64,
    )
    _require(np.isfinite(probability).all(), "prediction probability is non-finite")
    _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6), "probabilities do not sum to one")
    predicted = np.asarray(predictions["predicted_class"], dtype=np.int64)
    _require(np.array_equal(predicted, np.argmax(probability, axis=1)), "predicted_class differs from argmax")

    for index in range(row_count):
        handle = str(predictions["signal_handle"][index])
        opaque_index = predictions["opaque_signal_index"][index]
        _require(handle in sealed_by_handle, "prediction contains an unknown HMAC handle")
        _require(_is_int(opaque_index), "prediction opaque index is invalid")
        _require(
            int(opaque_index) == sealed_by_handle[handle]["opaque_signal_index"],
            "prediction HMAC-handle/opaque-index mapping differs",
        )
        _require(
            int(scored["class_id"][index]) == sealed_by_handle[handle]["class_id"],
            "scored class differs from the post-prediction sealed mapping",
        )

    prediction_keys = [
        (str(predictions["signal_handle"][index]), int(predictions["original_rate_hz"][index]))
        for index in range(row_count)
    ]
    _require(len(prediction_keys) == len(set(prediction_keys)), "record predictions contain duplicate signal/rate keys")
    entries = target.get("entries")
    _require(isinstance(entries, list), "target entries are absent")
    target_keys = [
        (str(entry.get("signal_handle")), int(entry.get("original_rate_hz")))
        for entry in entries
        if isinstance(entry, Mapping) and _is_int(entry.get("original_rate_hz"))
    ]
    _require(set(prediction_keys) == set(target_keys) and len(prediction_keys) == len(target_keys), "target/prediction keys differ")

    prediction_digest = sha256_file(run_root / "record_predictions.parquet")
    scored_digest = sha256_file(run_root / "scored_records.parquet")
    sealed_copy_digest = sha256_file(run_root / SEALED_LABEL_COPY_NAME)
    prediction_sidecar = (
        run_root / "prediction.sha256"
    ).read_text(encoding="utf-8").strip()
    _require(prediction_sidecar == prediction_digest, "prediction sidecar hash differs")
    _require(scorer.get("checkpoint_sha256") == checkpoint_digest, "scorer checkpoint hash differs")
    _require(scorer.get("prediction_sha256_before_label_join") == prediction_digest, "scorer prediction hash differs")
    _require(
        scorer.get("sealed_label_table_after_prediction_hashes_sha256")
        == sealed_copy_digest,
        "scorer sealed-table copy hash differs",
    )
    _require(
        scorer.get("all_four_prediction_hashes_verified_before_label_open") is True,
        "scorer did not record the four-arm prediction-hash gate",
    )
    _require(scorer.get("scored_records_sha256") == scored_digest, "scorer output hash differs")
    _require(metrics.get("prediction_sha256_before_label_join") == prediction_digest, "metrics prediction hash differs")
    _require(metrics.get("scored_records_sha256") == scored_digest, "metrics scored hash differs")
    _require(
        metrics.get("sealed_label_table_sha256") == sealed_copy_digest,
        "metrics sealed-table hash differs",
    )
    _require(
        scorer.get("metrics_sha256") == sha256_file(run_root / "metrics.json"),
        "scorer metrics hash differs",
    )
    _require(provenance.get("checkpoint_sha256") == checkpoint_digest, "provenance checkpoint differs")
    for commitment_name, expected_digest in commitments.items():
        _require(target.get(commitment_name) == expected_digest, f"target {commitment_name} differs")
        _require(decode.get(commitment_name) == expected_digest, f"decode {commitment_name} differs")

    checkpoint_time = _parse_time(provenance.get("checkpoint_written_at_utc"), field="checkpoint_written_at_utc")
    target_time = _parse_time(target.get("written_at_utc"), field="target written_at_utc")
    decode_start, decode_end = _decode_times(decode)
    prediction_time = _parse_time(scorer.get("prediction_written_at_utc"), field="prediction_written_at_utc")
    prediction_gate_time = _parse_time(
        scorer.get("all_prediction_hashes_gate_completed_at_utc"),
        field="all_prediction_hashes_gate_completed_at_utc",
    )
    sealed_open_time = _parse_time(
        scorer.get("sealed_label_table_opened_at_utc"),
        field="sealed_label_table_opened_at_utc",
    )
    sealed_copy_time = _parse_time(
        scorer.get("sealed_label_table_copied_at_utc"),
        field="sealed_label_table_copied_at_utc",
    )
    join_time = _parse_time(scorer.get("scorer_joined_at_utc"), field="scorer_joined_at_utc")
    scorer_end = _parse_time(scorer.get("scorer_completed_at_utc"), field="scorer_completed_at_utc")
    _require(
        checkpoint_time < decode_start <= decode_end < target_time <= prediction_time
        < prediction_gate_time <= sealed_open_time <= sealed_copy_time <= join_time <= scorer_end,
        "checkpoint/decode/prediction/scorer recorded order differs",
    )
    window_mtime = (run_root / "window_predictions.parquet").stat().st_mtime_ns
    prediction_mtime = (run_root / "record_predictions.parquet").stat().st_mtime_ns
    prediction_hash_mtime = (run_root / "prediction.sha256").stat().st_mtime_ns
    sealed_copy_mtime = (run_root / SEALED_LABEL_COPY_NAME).stat().st_mtime_ns
    scored_mtime = (run_root / "scored_records.parquet").stat().st_mtime_ns
    metrics_mtime = (run_root / "metrics.json").stat().st_mtime_ns
    scorer_log_mtime = (run_root / "scorer_join_log.json").stat().st_mtime_ns
    _require(
        window_mtime <= prediction_mtime <= prediction_hash_mtime
        < sealed_copy_mtime < scored_mtime < metrics_mtime < scorer_log_mtime,
        "prediction-hash/sealed-copy/scorer filesystem order differs",
    )

    if expected_run_state == "scored_pending_final_audit":
        _require(
            status.get("status") == "running"
            and status.get("phase") == "scored_pending_final_audit"
            and status.get("mode") == "formal_evidence",
            "run is not in the exact scored_pending_final_audit state",
        )
    else:
        _require(
            status.get("status") == "completed"
            and status.get("mode") == "formal_evidence",
            "run is not completed formal evidence",
        )
    _require(status.get("checkpoint_sha256") == checkpoint_digest, "run-status checkpoint hash differs")
    _require(status.get("metrics_sha256") == sha256_file(run_root / "metrics.json"), "run-status metrics hash differs")
    _require(not (run_root / "stderr.log").read_text(encoding="utf-8"), "stderr.log is non-empty")
    return {
        "unlabeled_record_count": row_count,
        "record_predictions_sha256": prediction_digest,
        "sealed_label_table_after_prediction_hashes_sha256": sealed_copy_digest,
        "scored_records_sha256": scored_digest,
        "class_id_absent_from_unlabeled_parquet": True,
        "audited_run_state": expected_run_state,
        "recorded_order": [
            checkpoint_time.isoformat(),
            decode_start.isoformat(),
            prediction_time.isoformat(),
            prediction_gate_time.isoformat(),
            sealed_open_time.isoformat(),
            sealed_copy_time.isoformat(),
            join_time.isoformat(),
            scorer_end.isoformat(),
        ],
    }


_ITEM_PATHS: dict[str, tuple[str, ...]] = {
    "L01": (
        "fold_manifest.json",
        "partition_disjointness.json",
        "data_manifest_pretest.json",
        "target_eval_manifest.json",
        "target_decode_log.json",
        SEALED_LABEL_COPY_NAME,
    ),
    "L02": ("loader_partition_log.json", "data_manifest_pretest.json"),
    "L03": (
        "fold_manifest.json",
        "normalization.json",
        "normalization_recompute.json",
        "data_manifest_pretest.json",
    ),
    "L04": ("source_sampling_rate_table.json", "resolved_config.yaml"),
    "L05": (
        "selection_trace.jsonl",
        "selected.ckpt",
        "checkpoint.sha256",
        "provenance.json",
        "target_eval_manifest.json",
    ),
    "L06": ("contract_checks.json",),
    "L07": ("contract_checks.json",),
    "L08": ("contract_checks.json", "collation_assertion_log.jsonl"),
    "L09": (
        "checkpoint.sha256",
        "provenance.json",
        "target_eval_manifest.json",
        "target_decode_log.json",
        "record_predictions.parquet",
        "prediction.sha256",
        "scorer_join_log.json",
    ),
    "L10": (
        "command.txt",
        "environment.yml",
        "resolved_config.yaml",
        "source_manifest.json",
        PROTOCOL_SNAPSHOT_PATH,
        "provenance.json",
        "run_status.json",
    ),
    "L11": (
        "training_input_schema.json",
        "loader_partition_log.json",
        "target_eval_manifest.json",
        "target_decode_log.json",
        "window_predictions.parquet",
        "record_predictions.parquet",
        "prediction.sha256",
        SEALED_LABEL_COPY_NAME,
        "scored_records.parquet",
        "scorer_join_log.json",
        "metrics.json",
        "run_status.json",
    ),
}

_ITEM_EXPECTED = {
    "L01": "frozen partition hashes recompute and all partition intersections are empty",
    "L02": "training loaders expose train/validation and observe zero target objects/labels",
    "L03": "source-train-only float64 Welford mapping and bank hash recompute exactly",
    "L04": "source-training rate table independently yields the frozen 6000 Hz cutoff",
    "L05": "selection contains no target/test field and predates checkpoint and target",
    "L06": "dataset identity call is logged as a concrete ValueError rejection",
    "L07": "system-selected head call is logged as a concrete ValueError rejection",
    "L08": "rate-length mismatch is rejected and every collation row records zero broadcast",
    "L09": "target decode and inference start only after checkpoint finalization",
    "L10": "one allowed physical GPU, unit world sizes, and non-distributed strategy",
    "L11": "target remains sealed through selection; decode, prediction hash, then scorer join",
}


def _audit_item(
    item_id: str,
    checker: Callable[[], dict[str, Any]],
    *,
    manifest_entries: Mapping[str, str],
) -> dict[str, Any]:
    paths = _ITEM_PATHS[item_id]
    evidence_hashes = {
        path: manifest_entries[path]
        for path in paths
        if path in manifest_entries
    }
    try:
        _require(len(evidence_hashes) == len(paths), "one or more item evidence hashes are absent")
        observed: Any = checker()
        status = "pass"
    except Exception as exc:  # a malformed artifact must become a recorded FAIL
        status = "fail"
        observed = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    payload = {
        "item_id": item_id,
        "status": status,
        "evidence_paths": list(paths),
        "expected_value": _ITEM_EXPECTED[item_id],
        "observed_value": observed,
        "evidence_file_sha256": evidence_hashes,
    }
    payload["evidence_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def audit_run_artifacts(
    run_root: str | Path,
    *,
    artifact_digests: Mapping[str, str],
    expected_run_state: AuditRunState = "completed",
) -> dict[str, Any]:
    """Reopen one run and independently recompute L01--L11.

    Evidence deficiencies are returned as ``status: fail`` rather than raised,
    making the result safe to serialize as ``leakage_audit.json``.  Programming
    errors inside a checker are also fail-closed and retain their exception
    type in the affected item's observed value.
    """

    root = Path(run_root).resolve()
    if expected_run_state not in _AUDIT_RUN_STATES:
        integrity_errors = [
            "expected_run_state must be 'scored_pending_final_audit' or "
            f"'completed', got {expected_run_state!r}"
        ]
        entries: dict[str, str] = {}
    else:
        entries, integrity_errors = _verify_manifest_and_claimed_digests(
            root, artifact_digests
        )
    if integrity_errors:
        items = []
        for item_id in _ITEM_PATHS:
            payload = {
                "item_id": item_id,
                "status": "fail",
                "evidence_paths": list(_ITEM_PATHS[item_id]),
                "expected_value": _ITEM_EXPECTED[item_id],
                "observed_value": {
                    "error_type": "ArtifactIntegrityError",
                    "errors": integrity_errors,
                },
                "evidence_file_sha256": {
                    path: entries[path]
                    for path in _ITEM_PATHS[item_id]
                    if path in entries
                },
            }
            payload["evidence_sha256"] = sha256_bytes(canonical_json_bytes(payload))
            items.append(payload)
        return {
            "protocol_id": PROTOCOL_ID,
            "experiment_id": EXPERIMENT_ID,
            "status": "fail",
            "audited_run_state": expected_run_state,
            "artifact_integrity": {
                "status": "fail",
                "errors": integrity_errors,
                "verified_entry_count": len(entries),
            },
            "items": items,
        }

    # L03 and L04 share one independently regenerated source mapping.  This is
    # computed by the auditor itself, never accepted from runner memory.
    source_cache: dict[str, Any] = {}

    def source() -> Mapping[str, Any]:
        if not source_cache:
            source_cache.update(
                _independent_source_recompute(_read_json(root / "fold_manifest.json"))
            )
        return source_cache

    checkers: dict[str, Callable[[], dict[str, Any]]] = {
        "L01": lambda: _check_l01(root),
        "L02": lambda: _check_l02(root),
        "L03": lambda: _check_l03(root, source()),
        "L04": lambda: _check_l04(root, source()),
        "L05": lambda: _check_l05(root),
        "L06": lambda: _check_l06(root),
        "L07": lambda: _check_l07(root),
        "L08": lambda: _check_l08(root),
        "L09": lambda: _check_l09(root),
        "L10": lambda: _check_l10(root),
        "L11": lambda: _check_l11(
            root, expected_run_state=expected_run_state
        ),
    }
    items = [
        _audit_item(item_id, checkers[item_id], manifest_entries=entries)
        for item_id in _ITEM_PATHS
    ]
    status = "pass" if all(item["status"] == "pass" for item in items) else "fail"
    return {
        "protocol_id": PROTOCOL_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "audited_run_state": expected_run_state,
        "artifact_integrity": {
            "status": "pass",
            "errors": [],
            "verified_entry_count": len(entries),
        },
        "items": items,
    }


__all__ = [
    "AuditEvidenceError",
    "AuditRunState",
    "PARTITION_HASH_FIELD",
    "REQUIRED_EVIDENCE_FILES",
    "SEALED_LABEL_COPY_NAME",
    "audit_run_artifacts",
    "partition_id_set_sha256",
]
