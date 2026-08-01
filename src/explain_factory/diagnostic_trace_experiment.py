"""Frozen finite-census experiment for the P06 diagnostic trace verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import sys
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import yaml

from .diagnostic_trace_contract import (
    DEFAULT_CONTRACT,
    DiagnosticTrace,
    EdgeWitness,
    LanguagePayload,
    RepresentationPayload,
    SignalPayload,
    Stage,
    StageRecord,
    SymbolicPayload,
    make_edge_witness,
    payload_digest,
    verify_trace,
)


PROTOCOL_ID = "P06-PROTOCOL-v1"
UNIVERSE_ID = "U-P06-v1"
FOREIGN_ROOT = hashlib.sha256(b"P06-FOREIGN-ROOT-v1").hexdigest()
VOCABULARY = ("outer_race_fault", "inner_race_fault", "ball_fault", "normal")
TREATMENT_IDS = (
    "proposed",
    "B1-label-only",
    "B2-schema-only",
    "B3-posthoc-consistency",
    "B4-untyped-provenance",
    "B5-rule-no-provenance",
)
REPLAY_IDS = ("R0", "R1", "R2", "R3", "R4")


@dataclass(frozen=True)
class BaseSpec:
    base_id: str
    raw_values: tuple[float, ...]
    sampling_rate_hz: float
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    symbol: str
    support_feature: str


@dataclass(frozen=True)
class TraceCase:
    case_id: str
    base_id: str
    family: str
    mutation_id: str
    expected_valid: bool
    trace: DiagnosticTrace


BASE_SPECS = (
    BaseSpec(
        "V00",
        (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0),
        12_000.0,
        ("bpfo_energy", "rms"),
        (0.82, 0.31),
        "outer_race_fault",
        "bpfo_energy",
    ),
    BaseSpec(
        "V01",
        (0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5),
        12_000.0,
        ("bpfi_energy", "kurtosis"),
        (0.77, 4.20),
        "inner_race_fault",
        "bpfi_energy",
    ),
    BaseSpec(
        "V02",
        (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
        25_600.0,
        ("bsf_energy", "crest_factor"),
        (0.68, 3.40),
        "ball_fault",
        "bsf_energy",
    ),
    BaseSpec(
        "V03",
        (0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1),
        25_600.0,
        ("rms", "kurtosis"),
        (0.10, 3.00),
        "normal",
        "rms",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_digest(values: Sequence[float]) -> str:
    encoded = struct.pack("<8f", *values)
    return hashlib.sha256(encoded).hexdigest()


def _records(trace: DiagnosticTrace) -> dict[Stage, StageRecord]:
    return {record.stage: record for record in trace.records}


def _bind(records: Iterable[StageRecord]) -> DiagnosticTrace:
    record_tuple = tuple(records)
    by_stage = {record.stage: record for record in record_tuple}
    witnesses = tuple(
        make_edge_witness(
            by_stage[edge.source_stage],
            by_stage[edge.target_stage],
            edge.predicate_id,
        )
        for edge in DEFAULT_CONTRACT.edges
    )
    return DiagnosticTrace(records=record_tuple, witnesses=witnesses)


def build_valid_trace(spec: BaseSpec) -> DiagnosticTrace:
    sample_id = f"p06-v1-{spec.base_id}"
    signal = StageRecord(
        Stage.SIGNAL,
        sample_id,
        SignalPayload(
            content_digest=_raw_digest(spec.raw_values),
            sampling_rate_hz=spec.sampling_rate_hz,
            channel_names=("drive_end",),
            sample_count=8,
        ),
    )
    signal_digest = payload_digest(signal.payload)
    representation = StageRecord(
        Stage.REPRESENTATION,
        sample_id,
        RepresentationPayload(
            source_signal_digest=signal_digest,
            root_signal_digest=signal_digest,
            feature_names=spec.feature_names,
            feature_values=spec.feature_values,
        ),
    )
    symbolic = StageRecord(
        Stage.SYMBOLIC,
        sample_id,
        SymbolicPayload(
            source_representation_digest=payload_digest(representation.payload),
            root_signal_digest=signal_digest,
            symbols=(spec.symbol,),
            support_features=(spec.support_feature,),
        ),
    )
    language = StageRecord(
        Stage.LANGUAGE,
        sample_id,
        LanguagePayload(
            source_symbol_digest=payload_digest(symbolic.payload),
            root_signal_digest=signal_digest,
            text=f"The verified symbol is {spec.symbol}.",
            mentioned_symbols=(spec.symbol,),
        ),
    )
    return _bind((signal, representation, symbolic, language))


def _missing_stage(trace: DiagnosticTrace, stage: Stage) -> DiagnosticTrace:
    return DiagnosticTrace(
        records=tuple(record for record in trace.records if record.stage != stage),
        witnesses=tuple(
            witness
            for witness in trace.witnesses
            if witness.source_stage != stage and witness.target_stage != stage
        ),
    )


def _missing_witness(
    trace: DiagnosticTrace,
    source: Stage,
    target: Stage,
) -> DiagnosticTrace:
    return replace(
        trace,
        witnesses=tuple(
            witness
            for witness in trace.witnesses
            if not (witness.source_stage == source and witness.target_stage == target)
        ),
    )


def _wrong_type(trace: DiagnosticTrace, stage: Stage) -> DiagnosticTrace:
    by_stage = _records(trace)
    next_stage = {
        Stage.SIGNAL: Stage.REPRESENTATION,
        Stage.REPRESENTATION: Stage.SYMBOLIC,
        Stage.SYMBOLIC: Stage.LANGUAGE,
        Stage.LANGUAGE: Stage.SIGNAL,
    }[stage]
    records = tuple(
        replace(record, payload=by_stage[next_stage].payload)
        if record.stage == stage
        else record
        for record in trace.records
    )
    return _bind(records)


def _swapped_symbol(trace: DiagnosticTrace, replacement_symbol: str) -> DiagnosticTrace:
    records = list(trace.records)
    language = _records(trace)[Stage.LANGUAGE]
    if not isinstance(language.payload, LanguagePayload):
        raise TypeError("valid base must contain LanguagePayload")
    replacement = replace(
        language,
        payload=replace(
            language.payload,
            text=f"The verified symbol is {replacement_symbol}.",
            mentioned_symbols=(replacement_symbol,),
        ),
    )
    records[records.index(language)] = replacement
    return _bind(records)


def _stale_sample(trace: DiagnosticTrace, stage: Stage, base_id: str) -> DiagnosticTrace:
    records = tuple(
        replace(record, sample_id=f"stale-{stage.value}-{base_id}")
        if record.stage == stage
        else record
        for record in trace.records
    )
    return replace(trace, records=records)


def _non_compositional(trace: DiagnosticTrace, mask: int) -> DiagnosticTrace:
    by_stage = _records(trace)
    representation = by_stage[Stage.REPRESENTATION]
    symbolic = by_stage[Stage.SYMBOLIC]
    language = by_stage[Stage.LANGUAGE]
    if not isinstance(representation.payload, RepresentationPayload):
        raise TypeError("valid base must contain RepresentationPayload")
    if not isinstance(symbolic.payload, SymbolicPayload):
        raise TypeError("valid base must contain SymbolicPayload")
    if not isinstance(language.payload, LanguagePayload):
        raise TypeError("valid base must contain LanguagePayload")

    representation = replace(
        representation,
        payload=replace(
            representation.payload,
            root_signal_digest=(
                FOREIGN_ROOT if mask & 0b100 else representation.payload.root_signal_digest
            ),
        ),
    )
    symbolic = replace(
        symbolic,
        payload=replace(
            symbolic.payload,
            source_representation_digest=payload_digest(representation.payload),
            root_signal_digest=(
                FOREIGN_ROOT if mask & 0b010 else symbolic.payload.root_signal_digest
            ),
        ),
    )
    language = replace(
        language,
        payload=replace(
            language.payload,
            source_symbol_digest=payload_digest(symbolic.payload),
            root_signal_digest=(
                FOREIGN_ROOT if mask & 0b001 else language.payload.root_signal_digest
            ),
        ),
    )
    return _bind((by_stage[Stage.SIGNAL], representation, symbolic, language))


def generate_universe() -> tuple[TraceCase, ...]:
    cases: list[TraceCase] = []
    edge_labels = (
        (Stage.SIGNAL, Stage.REPRESENTATION, "signal-representation"),
        (Stage.REPRESENTATION, Stage.SYMBOLIC, "representation-symbolic"),
        (Stage.SYMBOLIC, Stage.LANGUAGE, "symbolic-language"),
    )
    for index, spec in enumerate(BASE_SPECS):
        trace = build_valid_trace(spec)
        cases.append(
            TraceCase(
                f"U1-{spec.base_id}-VALID",
                spec.base_id,
                "valid",
                "valid",
                True,
                trace,
            )
        )
        for stage in Stage:
            cases.append(
                TraceCase(
                    f"U1-{spec.base_id}-MISS-STAGE-{stage.value}",
                    spec.base_id,
                    "missing",
                    f"missing-stage-{stage.value}",
                    False,
                    _missing_stage(trace, stage),
                )
            )
        for source, target, label in edge_labels:
            cases.append(
                TraceCase(
                    f"U1-{spec.base_id}-MISS-WITNESS-{label}",
                    spec.base_id,
                    "missing",
                    f"missing-witness-{label}",
                    False,
                    _missing_witness(trace, source, target),
                )
            )
        for stage in Stage:
            cases.append(
                TraceCase(
                    f"U1-{spec.base_id}-TYPE-{stage.value}",
                    spec.base_id,
                    "wrong_type",
                    f"wrong-type-{stage.value}",
                    False,
                    _wrong_type(trace, stage),
                )
            )
        replacement_symbol = BASE_SPECS[(index + 1) % len(BASE_SPECS)].symbol
        cases.append(
            TraceCase(
                f"U1-{spec.base_id}-SWAP-SYMBOL",
                spec.base_id,
                "swapped_symbol",
                "swapped-symbol",
                False,
                _swapped_symbol(trace, replacement_symbol),
            )
        )
        for stage in (Stage.REPRESENTATION, Stage.SYMBOLIC, Stage.LANGUAGE):
            cases.append(
                TraceCase(
                    f"U1-{spec.base_id}-STALE-{stage.value}",
                    spec.base_id,
                    "stale_sample",
                    f"stale-sample-{stage.value}",
                    False,
                    _stale_sample(trace, stage, spec.base_id),
                )
            )
        for mask in range(1, 8):
            cases.append(
                TraceCase(
                    f"U1-{spec.base_id}-ROOT-{mask:03b}",
                    spec.base_id,
                    "non_compositional",
                    f"root-mask-{mask:03b}",
                    False,
                    _non_compositional(trace, mask),
                )
            )

    ordered = tuple(sorted(cases, key=lambda case: case.case_id))
    ids = [case.case_id for case in ordered]
    families = {family: sum(case.family == family for case in ordered) for family in {
        case.family for case in ordered
    }}
    expected_families = {
        "valid": 4,
        "missing": 28,
        "wrong_type": 16,
        "swapped_symbol": 4,
        "stale_sample": 12,
        "non_compositional": 28,
    }
    if len(ordered) != 92 or len(set(ids)) != 92 or families != expected_families:
        raise RuntimeError(
            f"frozen universe mismatch: count={len(ordered)}, unique={len(set(ids))}, families={families}"
        )
    return ordered


def _payload_object(payload: object) -> dict[str, object]:
    if not is_dataclass(payload) or isinstance(payload, type):
        return {"payload_type": type(payload).__name__, "value": repr(payload)}
    return {"payload_type": type(payload).__name__, "value": asdict(payload)}


def _trace_object(trace: DiagnosticTrace) -> dict[str, object]:
    return {
        "records": [
            {
                "stage": record.stage.value,
                "sample_id": record.sample_id,
                "payload": _payload_object(record.payload),
            }
            for record in trace.records
        ],
        "witnesses": [
            {
                "source_stage": witness.source_stage.value,
                "target_stage": witness.target_stage.value,
                "sample_id": witness.sample_id,
                "source_digest": witness.source_digest,
                "target_digest": witness.target_digest,
                "predicate_id": witness.predicate_id,
            }
            for witness in trace.witnesses
        ],
    }


def _case_object(case: TraceCase) -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "universe_id": UNIVERSE_ID,
        "case_id": case.case_id,
        "base_id": case.base_id,
        "family": case.family,
        "mutation_id": case.mutation_id,
        "expected_valid": case.expected_valid,
        "trace": _trace_object(case.trace),
    }


def _jsonl_bytes(items: Iterable[dict[str, object]]) -> bytes:
    lines = [
        json.dumps(item, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for item in items
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def manifest_bytes(cases: Sequence[TraceCase]) -> bytes:
    return _jsonl_bytes(_case_object(case) for case in sorted(cases, key=lambda c: c.case_id))


def order_cases(cases: Sequence[TraceCase], replay_id: str) -> tuple[TraceCase, ...]:
    ordered = tuple(sorted(cases, key=lambda case: case.case_id))
    if replay_id == "R0":
        return ordered
    if replay_id == "R1":
        return tuple(reversed(ordered))
    offsets = {"R2": 1, "R3": 17, "R4": 43}
    if replay_id not in offsets:
        raise ValueError(f"unknown replay_id: {replay_id}")
    offset = offsets[replay_id]
    return ordered[offset:] + ordered[:offset]


def _record_for(trace: DiagnosticTrace, stage: Stage) -> StageRecord | None:
    matching = [record for record in trace.records if record.stage == stage]
    return matching[0] if len(matching) == 1 else None


def _baseline_label_only(trace: DiagnosticTrace) -> tuple[bool, str]:
    language = _record_for(trace, Stage.LANGUAGE)
    symbols = getattr(language.payload, "mentioned_symbols", ()) if language else ()
    accepted = any(isinstance(symbol, str) and symbol in VOCABULARY for symbol in symbols)
    return accepted, "known_language_symbol" if accepted else "no_known_language_symbol"


def _baseline_schema_only(trace: DiagnosticTrace) -> tuple[bool, str]:
    required = {
        Stage.SIGNAL: {"content_digest", "sampling_rate_hz", "channel_names", "sample_count"},
        Stage.REPRESENTATION: {
            "source_signal_digest",
            "root_signal_digest",
            "feature_names",
            "feature_values",
        },
        Stage.SYMBOLIC: {
            "source_representation_digest",
            "root_signal_digest",
            "symbols",
            "support_features",
        },
        Stage.LANGUAGE: {
            "source_symbol_digest",
            "root_signal_digest",
            "text",
            "mentioned_symbols",
        },
    }
    for stage, fields in required.items():
        record = _record_for(trace, stage)
        if record is None or not is_dataclass(record.payload):
            return False, f"missing_or_non_dataclass_{stage.value}"
        if not fields.issubset(asdict(record.payload)):
            return False, f"missing_fields_{stage.value}"
    return True, "required_fields_present"


def _baseline_posthoc(trace: DiagnosticTrace) -> tuple[bool, str]:
    symbolic = _record_for(trace, Stage.SYMBOLIC)
    language = _record_for(trace, Stage.LANGUAGE)
    if symbolic is None or language is None:
        return False, "missing_symbolic_or_language"
    symbols = set(getattr(symbolic.payload, "symbols", ()))
    mentioned = tuple(getattr(language.payload, "mentioned_symbols", ()))
    text = str(getattr(language.payload, "text", ""))
    accepted = bool(mentioned) and set(mentioned).issubset(symbols) and all(
        symbol in text for symbol in mentioned
    )
    return accepted, "posthoc_match" if accepted else "posthoc_mismatch"


def _baseline_untyped_provenance(trace: DiagnosticTrace) -> tuple[bool, str]:
    if any(_record_for(trace, stage) is None for stage in Stage):
        return False, "missing_or_duplicate_stage"
    expected_edges = {
        (edge.source_stage, edge.target_stage) for edge in DEFAULT_CONTRACT.edges
    }
    if len(trace.witnesses) != 3:
        return False, "witness_count"
    by_stage = _records(trace)
    for source, target in expected_edges:
        matching = [
            witness
            for witness in trace.witnesses
            if witness.source_stage == source and witness.target_stage == target
        ]
        if len(matching) != 1:
            return False, "witness_edge"
        witness = matching[0]
        try:
            if witness.source_digest != payload_digest(by_stage[source].payload):
                return False, "source_digest"
            if witness.target_digest != payload_digest(by_stage[target].payload):
                return False, "target_digest"
        except (TypeError, ValueError):
            return False, "unhashable_payload"
    return True, "untyped_edges_match"


def _baseline_rule_no_provenance(trace: DiagnosticTrace) -> tuple[bool, str]:
    representation = _record_for(trace, Stage.REPRESENTATION)
    symbolic = _record_for(trace, Stage.SYMBOLIC)
    language = _record_for(trace, Stage.LANGUAGE)
    if representation is None or symbolic is None or language is None:
        return False, "missing_rule_stage"
    features = set(getattr(representation.payload, "feature_names", ()))
    support = set(getattr(symbolic.payload, "support_features", ()))
    symbols = set(getattr(symbolic.payload, "symbols", ()))
    mentioned = set(getattr(language.payload, "mentioned_symbols", ()))
    accepted = bool(support) and bool(mentioned) and support.issubset(
        features
    ) and mentioned.issubset(symbols)
    return accepted, "local_membership" if accepted else "local_membership_failed"


BASELINES: dict[str, Callable[[DiagnosticTrace], tuple[bool, str]]] = {
    "B1-label-only": _baseline_label_only,
    "B2-schema-only": _baseline_schema_only,
    "B3-posthoc-consistency": _baseline_posthoc,
    "B4-untyped-provenance": _baseline_untyped_provenance,
    "B5-rule-no-provenance": _baseline_rule_no_provenance,
}


def evaluate_case(case: TraceCase) -> tuple[dict[str, object], ...]:
    proposed = verify_trace(case.trace)
    common = {
        "case_id": case.case_id,
        "base_id": case.base_id,
        "family": case.family,
        "mutation_id": case.mutation_id,
        "expected_valid": case.expected_valid,
    }
    rows: list[dict[str, object]] = [
        {
            **common,
            "treatment_id": "proposed",
            "accepted": proposed.accepted,
            "correct": proposed.accepted == case.expected_valid,
            "reason_codes": [failure.code.value for failure in proposed.failures],
            "checked_predicates": list(proposed.checked_predicates),
            "trace_fingerprint": proposed.trace_fingerprint,
        }
    ]
    for baseline_id, baseline in BASELINES.items():
        accepted, reason = baseline(case.trace)
        rows.append(
            {
                **common,
                "treatment_id": baseline_id,
                "accepted": accepted,
                "correct": accepted == case.expected_valid,
                "reason_codes": [reason],
                "checked_predicates": [],
                "trace_fingerprint": proposed.trace_fingerprint,
            }
        )
    return tuple(rows)


def _load_protocol(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol must be a YAML mapping")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol_id mismatch")
    if value.get("protocol_state") != "human_approved":
        raise ValueError("protocol is not human approved")
    if value.get("finite_universe", {}).get("total_case_count") != 92:
        raise ValueError("protocol universe count mismatch")
    if len(value.get("baselines", [])) != 5:
        raise ValueError("protocol baseline count mismatch")
    code_mapping = value.get("code_mapping", {})
    contract_path = Path(__file__).with_name("diagnostic_trace_contract.py")
    if code_mapping.get("implementation_sha256") != _sha256(contract_path):
        raise ValueError("frozen contract implementation hash mismatch")
    if code_mapping.get("experiment_module_sha256") != _sha256(Path(__file__)):
        raise ValueError("frozen experiment runner hash mismatch")
    return value


def run_replay(
    output_dir: Path,
    protocol_path: Path,
    replay_id: str,
) -> dict[str, object]:
    if replay_id not in REPLAY_IDS:
        raise ValueError(f"replay_id must be one of {REPLAY_IDS}")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices not in {"", "-1"}:
        raise RuntimeError(
            "CPU-only evidence runs must set CUDA_VISIBLE_DEVICES to an empty string or -1"
        )
    protocol = _load_protocol(protocol_path)
    cases = generate_universe()
    canonical_manifest = manifest_bytes(cases)
    (output_dir / "replays").mkdir(parents=True, exist_ok=True)
    replay_dir = output_dir / "replays" / replay_id
    replay_dir.mkdir(exist_ok=False)
    manifest_path = replay_dir / "case_manifest.jsonl"
    manifest_path.write_bytes(canonical_manifest)

    started = datetime.now(timezone.utc)
    verdict_rows: list[dict[str, object]] = []
    for case in order_cases(cases, replay_id):
        verdict_rows.extend(evaluate_case(case))
    verdict_path = replay_dir / "verdicts.jsonl"
    verdict_path.write_bytes(_jsonl_bytes(verdict_rows))
    completed = datetime.now(timezone.utc)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "universe_id": UNIVERSE_ID,
        "replay_id": replay_id,
        "case_count": len(cases),
        "treatment_count": len(TREATMENT_IDS),
        "verdict_row_count": len(verdict_rows),
        "case_manifest_sha256": _sha256(manifest_path),
        "verdicts_sha256": _sha256(verdict_path),
        "protocol_sha256": _sha256(protocol_path),
        "contract_sha256": protocol["code_mapping"]["implementation_sha256"],
        "runner_sha256": _sha256(Path(__file__)),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "command_prefix": "conda run -n LQ_signal",
        "cuda_visible_devices": cuda_visible_devices,
        "physical_gpu_indices": [],
        "multi_gpu": False,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
    }
    metadata_path = replay_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize_v1(output_dir: Path, protocol_path: Path) -> dict[str, object]:
    protocol = _load_protocol(protocol_path)
    replay_metadata: dict[str, dict[str, object]] = {}
    replay_rows: dict[str, list[dict[str, object]]] = {}
    for replay_id in REPLAY_IDS:
        replay_dir = output_dir / "replays" / replay_id
        metadata_path = replay_dir / "metadata.json"
        verdict_path = replay_dir / "verdicts.jsonl"
        manifest_path = replay_dir / "case_manifest.jsonl"
        if not metadata_path.is_file() or not verdict_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete replay: {replay_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("conda_environment") != "LQ_signal":
            raise ValueError(f"wrong conda environment for {replay_id}")
        if metadata.get("physical_gpu_indices") != [] or metadata.get("multi_gpu") is not False:
            raise ValueError(f"GPU contract violated for {replay_id}")
        replay_metadata[replay_id] = metadata
        replay_rows[replay_id] = _read_jsonl(verdict_path)

    manifest_hashes = {
        str(metadata["case_manifest_sha256"]) for metadata in replay_metadata.values()
    }
    runner_hashes = {str(metadata["runner_sha256"]) for metadata in replay_metadata.values()}
    contract_hashes = {str(metadata["contract_sha256"]) for metadata in replay_metadata.values()}
    protocol_hashes = {str(metadata["protocol_sha256"]) for metadata in replay_metadata.values()}
    if not all(len(values) == 1 for values in (
        manifest_hashes,
        runner_hashes,
        contract_hashes,
        protocol_hashes,
    )):
        raise ValueError("replay provenance hashes disagree")

    stable_fields = (
        "accepted",
        "reason_codes",
        "checked_predicates",
        "trace_fingerprint",
    )
    indexed: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
    for replay_id, rows in replay_rows.items():
        index = {(str(row["case_id"]), str(row["treatment_id"])): row for row in rows}
        if len(index) != 92 * 6:
            raise ValueError(f"unexpected verdict row count for {replay_id}: {len(index)}")
        indexed[replay_id] = index

    reference = indexed["R0"]
    disagreements: list[dict[str, object]] = []
    for replay_id in REPLAY_IDS[1:]:
        if set(indexed[replay_id]) != set(reference):
            raise ValueError(f"verdict keys differ for {replay_id}")
        for key, reference_row in reference.items():
            observed = indexed[replay_id][key]
            differing = [field for field in stable_fields if observed[field] != reference_row[field]]
            if differing:
                disagreements.append(
                    {
                        "replay_id": replay_id,
                        "case_id": key[0],
                        "treatment_id": key[1],
                        "fields": differing,
                    }
                )

    proposed_rows = [row for row in reference.values() if row["treatment_id"] == "proposed"]
    valid_rows = [row for row in proposed_rows if row["expected_valid"]]
    invalid_rows = [row for row in proposed_rows if not row["expected_valid"]]
    valid_accepted = sum(bool(row["accepted"]) for row in valid_rows)
    invalid_rejected = sum(not bool(row["accepted"]) for row in invalid_rows)
    family_summary: dict[str, dict[str, int | float]] = {}
    for family in ("missing", "wrong_type", "swapped_symbol", "stale_sample", "non_compositional"):
        rows = [row for row in invalid_rows if row["family"] == family]
        detected = sum(not bool(row["accepted"]) for row in rows)
        family_summary[family] = {
            "total": len(rows),
            "detected": detected,
            "completeness": detected / len(rows),
        }

    strict_distinction: dict[str, int] = {}
    for baseline_id in TREATMENT_IDS[1:]:
        count = 0
        for proposed_row in invalid_rows:
            key = (str(proposed_row["case_id"]), baseline_id)
            if not bool(proposed_row["accepted"]) and bool(reference[key]["accepted"]):
                count += 1
        strict_distinction[baseline_id] = count

    thresholds = protocol["decision_thresholds"]["C2"]
    checks = {
        "valid_accepted": valid_accepted == int(thresholds["valid_accepted"]),
        "invalid_rejected": invalid_rejected == int(thresholds["invalid_rejected"]),
        "all_family_completeness": all(
            summary["completeness"] >= float(thresholds["family_detection_completeness_min"])
            for summary in family_summary.values()
        ),
        "replay_agreement": len(disagreements) == int(thresholds["replay_disagreement_count"]),
        "strict_distinction_each_baseline": all(
            count >= int(thresholds["strict_distinction_count_min_each_baseline"])
            for count in strict_distinction.values()
        ),
    }
    outcome = "supported" if all(checks.values()) else "refuted"

    canonical_manifest = output_dir / "case_manifest.jsonl"
    canonical_verdicts = output_dir / "verdicts.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_dir / "replays" / "R0" / "case_manifest.jsonl", canonical_manifest)
    shutil.copyfile(output_dir / "replays" / "R0" / "verdicts.jsonl", canonical_verdicts)

    result: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "universe_id": UNIVERSE_ID,
        "status": "completed",
        "outcome": outcome,
        "claim_scope": ["C2"],
        "case_count": 92,
        "valid_case_count": len(valid_rows),
        "invalid_case_count": len(invalid_rows),
        "valid_accepted": valid_accepted,
        "invalid_rejected": invalid_rejected,
        "false_rejection_count": len(valid_rows) - valid_accepted,
        "invalid_false_acceptance_count": len(invalid_rows) - invalid_rejected,
        "family_summary": family_summary,
        "strict_distinction_count_by_baseline": strict_distinction,
        "replay_count": len(REPLAY_IDS),
        "random_seed_count": 0,
        "replay_disagreement_count": len(disagreements),
        "replay_disagreements": disagreements,
        "threshold_checks": checks,
        "case_manifest_sha256": _sha256(canonical_manifest),
        "verdicts_sha256": _sha256(canonical_verdicts),
        "runner_sha256": next(iter(runner_hashes)),
        "contract_sha256": next(iter(contract_hashes)),
        "protocol_sha256": next(iter(protocol_hashes)),
        "conda_environment": "LQ_signal",
        "command_prefix": "conda run -n LQ_signal",
        "physical_gpu_indices": [],
        "multi_gpu": False,
        "replay_metadata": replay_metadata,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_path = output_dir / "run.log"
    log_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "outcome": outcome,
                "result_sha256": _sha256(result_path),
                "replay_ids": list(REPLAY_IDS),
                "threshold_checks": checks,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="run one clean deterministic replay")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--protocol-path", type=Path, required=True)
    run_parser.add_argument("--replay-id", choices=REPLAY_IDS, required=True)
    summary_parser = subparsers.add_parser("summarize", help="compare all replay artifacts")
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    summary_parser.add_argument("--protocol-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        value = run_replay(args.output_dir.resolve(), args.protocol_path.resolve(), args.replay_id)
    else:
        value = summarize_v1(args.output_dir.resolve(), args.protocol_path.resolve())
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINES",
    "BASE_SPECS",
    "REPLAY_IDS",
    "TREATMENT_IDS",
    "TraceCase",
    "build_valid_trace",
    "evaluate_case",
    "generate_universe",
    "manifest_bytes",
    "order_cases",
    "run_replay",
    "summarize_v1",
]
