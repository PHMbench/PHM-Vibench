"""Frozen ablation, sensitivity, efficiency, and failure analysis for P06."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from .diagnostic_trace_contract import (
    DEFAULT_CONTRACT,
    PREDICATE_REGISTRY,
    DiagnosticTrace,
    EdgeWitness,
    FailureCode,
    LanguagePayload,
    PredicateVerdict,
    RepresentationPayload,
    SignalPayload,
    Stage,
    StageRecord,
    SymbolicPayload,
    VerificationFailure,
    VerificationResult,
    make_edge_witness,
    payload_digest,
    verify_trace,
)
from .diagnostic_trace_experiment import (
    BASE_SPECS,
    PROTOCOL_ID,
    REPLAY_IDS,
    UNIVERSE_ID,
    build_valid_trace,
    generate_universe,
)


FULL_EVIDENCE_ID = "P06-FULL-EVIDENCE-v1"
DIAGNOSTIC_SET_ID = "D-P06-v1"
SENSITIVITY_SET_ID = "S-P06-v1"
EXPECTED_PROTOCOL_SHA256 = "0169081e24bf8218e74d46aae9671028a3a57f4c9f840f83a00eb08ef43d4ce4"
EXPECTED_CONTRACT_SHA256 = "95dad7b36c1e809c0cf850eb662a874bf9fed178863e1cc838094ce8d985868e"
EXPECTED_PRIMARY_RUNNER_SHA256 = "544c02a5b6aeafacb9b91aa50d9b9761129fcadfecb31e79144d2c8fe254c3f6"
ABLATION_IDS = (
    "A0-full",
    "A1-no-content-digest",
    "A2-no-type",
    "A3-no-sample",
    "A4-no-witness-binding",
    "A5-no-signal-representation",
    "A6-no-representation-symbol",
    "A7-no-symbol-language",
    "A8-no-root-composition",
)


@dataclass(frozen=True)
class EvidenceCase:
    case_id: str
    set_id: str
    base_id: str
    family: str
    mutation_id: str
    expected_valid: bool
    trace: DiagnosticTrace


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _cascade(
    trace: DiagnosticTrace,
    *,
    signal_payload: SignalPayload | None = None,
    representation_payload: RepresentationPayload | None = None,
    symbolic_payload: SymbolicPayload | None = None,
    language_payload: LanguagePayload | None = None,
) -> DiagnosticTrace:
    records = _records(trace)
    signal = replace(
        records[Stage.SIGNAL],
        payload=signal_payload or records[Stage.SIGNAL].payload,
    )
    if not isinstance(signal.payload, SignalPayload):
        raise TypeError("cascade requires SignalPayload")
    signal_digest = payload_digest(signal.payload)

    original_representation = records[Stage.REPRESENTATION].payload
    if not isinstance(original_representation, RepresentationPayload):
        raise TypeError("cascade requires RepresentationPayload")
    representation_value = representation_payload or original_representation
    representation_value = replace(
        representation_value,
        source_signal_digest=signal_digest,
        root_signal_digest=signal_digest,
    )
    representation = replace(records[Stage.REPRESENTATION], payload=representation_value)

    original_symbolic = records[Stage.SYMBOLIC].payload
    if not isinstance(original_symbolic, SymbolicPayload):
        raise TypeError("cascade requires SymbolicPayload")
    symbolic_value = symbolic_payload or original_symbolic
    symbolic_value = replace(
        symbolic_value,
        source_representation_digest=payload_digest(representation_value),
        root_signal_digest=signal_digest,
    )
    symbolic = replace(records[Stage.SYMBOLIC], payload=symbolic_value)

    original_language = records[Stage.LANGUAGE].payload
    if not isinstance(original_language, LanguagePayload):
        raise TypeError("cascade requires LanguagePayload")
    language_value = language_payload or original_language
    language_value = replace(
        language_value,
        source_symbol_digest=payload_digest(symbolic_value),
        root_signal_digest=signal_digest,
    )
    language = replace(records[Stage.LANGUAGE], payload=language_value)
    return _bind((signal, representation, symbolic, language))


def _flip_hex(value: str) -> str:
    if not value:
        return "0"
    replacement = "0" if value[0].lower() != "0" else "1"
    return replacement + value[1:]


def generate_diagnostic_cases() -> tuple[EvidenceCase, ...]:
    cases: list[EvidenceCase] = []
    for spec in BASE_SPECS:
        base = build_valid_trace(spec)

        witnesses = list(base.witnesses)
        witnesses[0] = replace(
            witnesses[0], source_digest=_flip_hex(witnesses[0].source_digest)
        )
        cases.append(
            EvidenceCase(
                f"D1-{spec.base_id}-WITNESS-SOURCE",
                DIAGNOSTIC_SET_ID,
                spec.base_id,
                "witness_source_digest",
                "flip-first-witness-source-digest",
                False,
                replace(base, witnesses=tuple(witnesses)),
            )
        )

        witnesses = list(base.witnesses)
        witnesses[-1] = replace(
            witnesses[-1], target_digest=_flip_hex(witnesses[-1].target_digest)
        )
        cases.append(
            EvidenceCase(
                f"D2-{spec.base_id}-WITNESS-TARGET",
                DIAGNOSTIC_SET_ID,
                spec.base_id,
                "witness_target_digest",
                "flip-last-witness-target-digest",
                False,
                replace(base, witnesses=tuple(witnesses)),
            )
        )

        witnesses = list(base.witnesses)
        witnesses[1] = replace(
            witnesses[1], predicate_id="unregistered_predicate_v1"
        )
        cases.append(
            EvidenceCase(
                f"D3-{spec.base_id}-PREDICATE-ID",
                DIAGNOSTIC_SET_ID,
                spec.base_id,
                "witness_predicate",
                "substitute-middle-predicate-id",
                False,
                replace(base, witnesses=tuple(witnesses)),
            )
        )

        signal = _records(base)[Stage.SIGNAL].payload
        if not isinstance(signal, SignalPayload):
            raise TypeError("valid base requires SignalPayload")
        malformed_signal = replace(signal, content_digest="0" * 63)
        cases.append(
            EvidenceCase(
                f"D4-{spec.base_id}-CONTENT-DIGEST",
                DIAGNOSTIC_SET_ID,
                spec.base_id,
                "malformed_content_digest",
                "truncate-signal-content-digest",
                False,
                _cascade(base, signal_payload=malformed_signal),
            )
        )

        records = _records(base)
        representation = records[Stage.REPRESENTATION].payload
        if not isinstance(representation, RepresentationPayload):
            raise TypeError("valid base requires RepresentationPayload")
        invalid_values = (math.inf,) + representation.feature_values[1:]
        invalid_representation = replace(
            records[Stage.REPRESENTATION],
            payload=replace(representation, feature_values=invalid_values),
        )
        cases.append(
            EvidenceCase(
                f"D5-{spec.base_id}-NONFINITE",
                DIAGNOSTIC_SET_ID,
                spec.base_id,
                "nonfinite_feature",
                "replace-first-feature-with-positive-infinity",
                False,
                replace(
                    base,
                    records=tuple(
                        invalid_representation if record.stage == Stage.REPRESENTATION else record
                        for record in base.records
                    ),
                ),
            )
        )

        symbolic = records[Stage.SYMBOLIC].payload
        if not isinstance(symbolic, SymbolicPayload):
            raise TypeError("valid base requires SymbolicPayload")
        unknown_support = replace(symbolic, support_features=("unknown_feature",))
        cases.append(
            EvidenceCase(
                f"D6-{spec.base_id}-UNKNOWN-SUPPORT",
                DIAGNOSTIC_SET_ID,
                spec.base_id,
                "unknown_support_feature",
                "replace-symbolic-support-feature",
                False,
                _cascade(base, symbolic_payload=unknown_support),
            )
        )

    ordered = tuple(sorted(cases, key=lambda case: case.case_id))
    if len(ordered) != 24 or len({case.case_id for case in ordered}) != 24:
        raise RuntimeError("D-P06-v1 must contain 24 unique cases")
    return ordered


def generate_sensitivity_cases() -> tuple[EvidenceCase, ...]:
    cases: list[EvidenceCase] = []
    for spec in BASE_SPECS:
        base = build_valid_trace(spec)
        cases.append(
            EvidenceCase(
                f"S1-{spec.base_id}-REVERSE-ORDER",
                SENSITIVITY_SET_ID,
                spec.base_id,
                "order_only",
                "reverse-records-and-witnesses",
                True,
                replace(
                    base,
                    records=tuple(reversed(base.records)),
                    witnesses=tuple(reversed(base.witnesses)),
                ),
            )
        )
        cases.append(
            EvidenceCase(
                f"S2-{spec.base_id}-ROTATE-ORDER",
                SENSITIVITY_SET_ID,
                spec.base_id,
                "order_only",
                "rotate-records-and-witnesses-one",
                True,
                replace(
                    base,
                    records=base.records[1:] + base.records[:1],
                    witnesses=base.witnesses[1:] + base.witnesses[:1],
                ),
            )
        )

        renamed_id = f"renamed-p06-v1-{spec.base_id}"
        cases.append(
            EvidenceCase(
                f"S3-{spec.base_id}-RENAME-SAMPLE",
                SENSITIVITY_SET_ID,
                spec.base_id,
                "semantic_preserving",
                "consistent-sample-id-rename",
                True,
                DiagnosticTrace(
                    records=tuple(
                        replace(record, sample_id=renamed_id) for record in base.records
                    ),
                    witnesses=tuple(
                        replace(witness, sample_id=renamed_id)
                        for witness in base.witnesses
                    ),
                ),
            )
        )

        records = _records(base)
        representation = records[Stage.REPRESENTATION].payload
        if not isinstance(representation, RepresentationPayload):
            raise TypeError("valid base requires RepresentationPayload")
        permuted = replace(
            representation,
            feature_names=tuple(reversed(representation.feature_names)),
            feature_values=tuple(reversed(representation.feature_values)),
        )
        cases.append(
            EvidenceCase(
                f"S4-{spec.base_id}-PERMUTE-FEATURES",
                SENSITIVITY_SET_ID,
                spec.base_id,
                "semantic_preserving",
                "paired-feature-name-value-permutation",
                True,
                _cascade(base, representation_payload=permuted),
            )
        )

        language = records[Stage.LANGUAGE].payload
        if not isinstance(language, LanguagePayload):
            raise TypeError("valid base requires LanguagePayload")
        wording = replace(
            language,
            text=f"Verified diagnosis: {spec.symbol}.",
        )
        cases.append(
            EvidenceCase(
                f"S5-{spec.base_id}-ALT-WORDING",
                SENSITIVITY_SET_ID,
                spec.base_id,
                "semantic_preserving",
                "alternative-language-wording",
                True,
                _cascade(base, language_payload=wording),
            )
        )

    ordered = tuple(sorted(cases, key=lambda case: case.case_id))
    if len(ordered) != 20 or len({case.case_id for case in ordered}) != 20:
        raise RuntimeError("S-P06-v1 must contain 20 unique cases")
    return ordered


def generate_full_evidence_cases() -> tuple[EvidenceCase, ...]:
    primary = tuple(
        EvidenceCase(
            case.case_id,
            UNIVERSE_ID,
            case.base_id,
            case.family,
            case.mutation_id,
            case.expected_valid,
            case.trace,
        )
        for case in generate_universe()
    )
    cases = tuple(sorted(
        primary + generate_diagnostic_cases() + generate_sensitivity_cases(),
        key=lambda case: case.case_id,
    ))
    if len(cases) != 136 or len({case.case_id for case in cases}) != 136:
        raise RuntimeError("full evidence universe must contain 136 unique cases")
    return cases


def _signal_to_representation_without_content_digest(
    source: StageRecord,
    target: StageRecord,
) -> PredicateVerdict:
    if not isinstance(source.payload, SignalPayload) or not isinstance(
        target.payload, RepresentationPayload
    ):
        return PredicateVerdict(False, "predicate received incompatible payload types")
    signal = source.payload
    representation = target.payload
    if signal.sampling_rate_hz <= 0 or signal.sample_count <= 0:
        return PredicateVerdict(False, "signal dimensions and sampling rate must be positive")
    if not signal.channel_names or len(set(signal.channel_names)) != len(signal.channel_names):
        return PredicateVerdict(False, "signal channel names must be non-empty and unique")
    if representation.source_signal_digest != payload_digest(signal):
        return PredicateVerdict(False, "representation is not bound to the source signal")
    if not representation.feature_names or len(representation.feature_names) != len(
        representation.feature_values
    ):
        return PredicateVerdict(False, "feature names and values must be non-empty and aligned")
    if len(set(representation.feature_names)) != len(representation.feature_names):
        return PredicateVerdict(False, "feature names must be unique")
    if not all(math.isfinite(value) for value in representation.feature_values):
        return PredicateVerdict(False, "feature values must be finite")
    return PredicateVerdict(True, "signal-to-representation predicate satisfied")


def _finish(
    trace: DiagnosticTrace,
    checked_predicates: list[str],
    failures: list[VerificationFailure],
) -> VerificationResult:
    fingerprint = verify_trace(trace).trace_fingerprint
    return VerificationResult(
        accepted=not failures,
        contract_id=DEFAULT_CONTRACT.contract_id,
        trace_fingerprint=fingerprint,
        checked_predicates=tuple(checked_predicates),
        failures=tuple(failures),
    )


def verify_with_ablation(
    trace: DiagnosticTrace,
    ablation_id: str,
) -> VerificationResult:
    if ablation_id not in ABLATION_IDS:
        raise ValueError(f"unknown ablation_id: {ablation_id}")
    if ablation_id == "A0-full":
        return verify_trace(trace)

    failures: list[VerificationFailure] = []
    checked_predicates: list[str] = []
    expected_stages = {spec.stage for spec in DEFAULT_CONTRACT.stages}
    records_by_stage: dict[Stage, list[StageRecord]] = {
        stage: [] for stage in expected_stages
    }
    for record in trace.records:
        if record.stage not in expected_stages:
            failures.append(
                VerificationFailure(
                    FailureCode.UNEXPECTED_STAGE,
                    str(record.stage),
                    "record stage is absent from the contract",
                )
            )
            continue
        records_by_stage[record.stage].append(record)
    for spec in DEFAULT_CONTRACT.stages:
        count = len(records_by_stage[spec.stage])
        if count != 1:
            failures.append(
                VerificationFailure(
                    FailureCode.STAGE_CARDINALITY,
                    spec.stage.value,
                    f"expected exactly one record, observed {count}",
                )
            )
    if any(len(records_by_stage[spec.stage]) != 1 for spec in DEFAULT_CONTRACT.stages):
        return _finish(trace, checked_predicates, failures)

    records = {
        stage: stage_records[0] for stage, stage_records in records_by_stage.items()
    }
    actual_type_ok: dict[Stage, bool] = {}
    for spec in DEFAULT_CONTRACT.stages:
        record = records[spec.stage]
        actual_type_ok[spec.stage] = isinstance(record.payload, spec.payload_type)
        if not actual_type_ok[spec.stage] and ablation_id != "A2-no-type":
            failures.append(
                VerificationFailure(
                    FailureCode.TYPE_MISMATCH,
                    spec.stage.value,
                    f"expected {spec.payload_type.__name__}, observed {type(record.payload).__name__}",
                )
            )

    canonical_sample_id = records[Stage.SIGNAL].sample_id
    if ablation_id != "A3-no-sample":
        if not canonical_sample_id.strip():
            failures.append(
                VerificationFailure(
                    FailureCode.SAMPLE_BINDING,
                    Stage.SIGNAL.value,
                    "sample_id must be non-empty",
                )
            )
        for spec in DEFAULT_CONTRACT.stages[1:]:
            if records[spec.stage].sample_id != canonical_sample_id:
                failures.append(
                    VerificationFailure(
                        FailureCode.SAMPLE_BINDING,
                        spec.stage.value,
                        "stage sample_id differs from the signal sample_id",
                    )
                )

    digests: dict[Stage, str] = {}
    for spec in DEFAULT_CONTRACT.stages:
        try:
            digests[spec.stage] = payload_digest(records[spec.stage].payload)
        except (TypeError, ValueError) as error:
            failures.append(
                VerificationFailure(
                    FailureCode.PAYLOAD_INVALID,
                    spec.stage.value,
                    str(error),
                )
            )

    ignore_witnesses = ablation_id == "A4-no-witness-binding"
    expected_edges = {
        (edge.source_stage, edge.target_stage) for edge in DEFAULT_CONTRACT.edges
    }
    if not ignore_witnesses:
        for witness in sorted(
            (
                witness
                for witness in trace.witnesses
                if (witness.source_stage, witness.target_stage) not in expected_edges
            ),
            key=lambda item: (
                item.source_stage.value,
                item.target_stage.value,
                item.predicate_id,
            ),
        ):
            failures.append(
                VerificationFailure(
                    FailureCode.WITNESS_UNEXPECTED,
                    f"{witness.source_stage.value}->{witness.target_stage.value}",
                    "witness edge is absent from the contract",
                )
            )

    skipped_predicate = {
        "A5-no-signal-representation": "signal_to_representation_v1",
        "A6-no-representation-symbol": "representation_to_symbol_v1",
        "A7-no-symbol-language": "symbol_to_language_v1",
    }.get(ablation_id)

    for edge in DEFAULT_CONTRACT.edges:
        location = f"{edge.source_stage.value}->{edge.target_stage.value}"
        witness_ok = True
        if not ignore_witnesses:
            matching = [
                witness
                for witness in trace.witnesses
                if witness.source_stage == edge.source_stage
                and witness.target_stage == edge.target_stage
            ]
            if len(matching) != 1:
                failures.append(
                    VerificationFailure(
                        FailureCode.WITNESS_CARDINALITY,
                        location,
                        f"expected exactly one witness, observed {len(matching)}",
                    )
                )
                continue
            witness = matching[0]
            if witness.predicate_id != edge.predicate_id:
                witness_ok = False
                failures.append(
                    VerificationFailure(
                        FailureCode.WITNESS_PREDICATE,
                        location,
                        f"expected {edge.predicate_id}, observed {witness.predicate_id}",
                    )
                )
            if ablation_id != "A3-no-sample" and witness.sample_id != canonical_sample_id:
                witness_ok = False
                failures.append(
                    VerificationFailure(
                        FailureCode.WITNESS_SAMPLE,
                        location,
                        "witness sample_id differs from the signal sample_id",
                    )
                )
            if digests.get(edge.source_stage) != witness.source_digest:
                witness_ok = False
                failures.append(
                    VerificationFailure(
                        FailureCode.WITNESS_SOURCE_DIGEST,
                        location,
                        "witness source digest differs from the bound payload",
                    )
                )
            if digests.get(edge.target_stage) != witness.target_digest:
                witness_ok = False
                failures.append(
                    VerificationFailure(
                        FailureCode.WITNESS_TARGET_DIGEST,
                        location,
                        "witness target digest differs from the bound payload",
                    )
                )

        if edge.predicate_id == skipped_predicate:
            continue
        types_allow_execution = (
            actual_type_ok[edge.source_stage] and actual_type_ok[edge.target_stage]
        ) or ablation_id == "A2-no-type"
        if witness_ok and types_allow_execution:
            predicate = (
                _signal_to_representation_without_content_digest
                if ablation_id == "A1-no-content-digest"
                and edge.predicate_id == "signal_to_representation_v1"
                else PREDICATE_REGISTRY[edge.predicate_id]
            )
            try:
                verdict = predicate(records[edge.source_stage], records[edge.target_stage])
            except (TypeError, ValueError) as error:
                failures.append(
                    VerificationFailure(
                        FailureCode.PAYLOAD_INVALID,
                        location,
                        f"predicate input could not be canonicalized: {error}",
                    )
                )
            else:
                checked_predicates.append(edge.predicate_id)
                if not verdict.accepted:
                    failures.append(
                        VerificationFailure(
                            FailureCode.PREDICATE_FAILED,
                            location,
                            verdict.detail,
                        )
                    )

    if (
        ablation_id != "A8-no-root-composition"
        and all(actual_type_ok.values())
        and Stage.SIGNAL in digests
    ):
        expected_root = digests[Stage.SIGNAL]
        root_values = {
            Stage.REPRESENTATION: records[Stage.REPRESENTATION].payload.root_signal_digest,
            Stage.SYMBOLIC: records[Stage.SYMBOLIC].payload.root_signal_digest,
            Stage.LANGUAGE: records[Stage.LANGUAGE].payload.root_signal_digest,
        }
        mismatched = [
            stage.value
            for stage in (Stage.REPRESENTATION, Stage.SYMBOLIC, Stage.LANGUAGE)
            if root_values[stage] != expected_root
        ]
        if mismatched:
            failures.append(
                VerificationFailure(
                    FailureCode.COMPOSITION_ROOT,
                    "signal->language",
                    "root signal digest mismatch at: " + ", ".join(mismatched),
                )
            )
    return _finish(trace, checked_predicates, failures)


def independent_invariant_violations(trace: DiagnosticTrace) -> tuple[str, ...]:
    violations: list[str] = []
    by_stage: dict[Stage, list[StageRecord]] = {stage: [] for stage in Stage}
    for record in trace.records:
        if record.stage in by_stage:
            by_stage[record.stage].append(record)
        else:
            violations.append("unexpected_stage")
    if any(len(by_stage[stage]) != 1 for stage in Stage):
        return tuple(sorted(set(violations + ["stage_cardinality"])))
    records = {stage: by_stage[stage][0] for stage in Stage}
    expected_types = {
        Stage.SIGNAL: SignalPayload,
        Stage.REPRESENTATION: RepresentationPayload,
        Stage.SYMBOLIC: SymbolicPayload,
        Stage.LANGUAGE: LanguagePayload,
    }
    if any(not isinstance(records[stage].payload, expected_types[stage]) for stage in Stage):
        violations.append("type_preservation")
        return tuple(sorted(set(violations)))
    sample_ids = {records[stage].sample_id for stage in Stage}
    if len(sample_ids) != 1 or not next(iter(sample_ids)).strip():
        violations.append("sample_identity")
    try:
        digests = {stage: payload_digest(records[stage].payload) for stage in Stage}
    except (TypeError, ValueError):
        violations.append("payload_canonicalization")
        return tuple(sorted(set(violations)))
    for edge in DEFAULT_CONTRACT.edges:
        matching = [
            witness
            for witness in trace.witnesses
            if witness.source_stage == edge.source_stage
            and witness.target_stage == edge.target_stage
        ]
        if len(matching) != 1:
            violations.append("witness_cardinality")
            continue
        witness = matching[0]
        if (
            witness.predicate_id != edge.predicate_id
            or witness.source_digest != digests[edge.source_stage]
            or witness.target_digest != digests[edge.target_stage]
            or witness.sample_id != records[Stage.SIGNAL].sample_id
        ):
            violations.append("witness_validity")
        verdict = PREDICATE_REGISTRY[edge.predicate_id](
            records[edge.source_stage], records[edge.target_stage]
        )
        if not verdict.accepted:
            violations.append("local_predicate")
    signal_digest = digests[Stage.SIGNAL]
    if any(
        getattr(records[stage].payload, "root_signal_digest") != signal_digest
        for stage in (Stage.REPRESENTATION, Stage.SYMBOLIC, Stage.LANGUAGE)
    ):
        violations.append("root_composition")
    return tuple(sorted(set(violations)))


def _normalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _normalize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        label = "+Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
        return {"nonfinite_float": label}
    return value


def _trace_object(trace: DiagnosticTrace) -> dict[str, object]:
    return {
        "records": [
            {
                "stage": record.stage.value,
                "sample_id": record.sample_id,
                "payload_type": type(record.payload).__name__,
                "payload": _normalize(record.payload),
            }
            for record in trace.records
        ],
        "witnesses": [_normalize(witness) for witness in trace.witnesses],
    }


def _case_object(case: EvidenceCase) -> dict[str, object]:
    return {
        "full_evidence_id": FULL_EVIDENCE_ID,
        "protocol_id": PROTOCOL_ID,
        "case_id": case.case_id,
        "set_id": case.set_id,
        "base_id": case.base_id,
        "family": case.family,
        "mutation_id": case.mutation_id,
        "expected_valid": case.expected_valid,
        "trace": _trace_object(case.trace),
    }


def _jsonl_bytes(items: Iterable[dict[str, object]]) -> bytes:
    lines = [
        json.dumps(
            _normalize(item),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in items
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def manifest_bytes(cases: Sequence[EvidenceCase]) -> bytes:
    return _jsonl_bytes(
        _case_object(case) for case in sorted(cases, key=lambda item: item.case_id)
    )


def order_cases(
    cases: Sequence[EvidenceCase], replay_id: str
) -> tuple[EvidenceCase, ...]:
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


def _load_protocol_lock(path: Path) -> dict[str, object]:
    if _sha256(path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("approved protocol file hash differs from the E2 evidence lock")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol must be a mapping")
    if value.get("protocol_id") != PROTOCOL_ID or value.get("protocol_state") != "human_approved":
        raise ValueError("protocol identity or approval state mismatch")
    mapping = value.get("code_mapping", {})
    contract_path = Path(__file__).with_name("diagnostic_trace_contract.py")
    primary_runner_path = Path(__file__).with_name("diagnostic_trace_experiment.py")
    if _sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("production contract changed after E2 lock")
    if _sha256(primary_runner_path) != EXPECTED_PRIMARY_RUNNER_SHA256:
        raise ValueError("primary E2 runner changed after E2 lock")
    if mapping.get("implementation_sha256") != EXPECTED_CONTRACT_SHA256:
        raise ValueError("protocol contract hash mismatch")
    if mapping.get("experiment_module_sha256") != EXPECTED_PRIMARY_RUNNER_SHA256:
        raise ValueError("protocol primary-runner hash mismatch")
    return value


def run_ablation_replay(
    output_dir: Path,
    protocol_path: Path,
    ablation_id: str,
    replay_id: str,
) -> dict[str, object]:
    if ablation_id not in ABLATION_IDS:
        raise ValueError(f"ablation_id must be one of {ABLATION_IDS}")
    if replay_id not in REPLAY_IDS:
        raise ValueError(f"replay_id must be one of {REPLAY_IDS}")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices not in {"", "-1"}:
        raise RuntimeError(
            "CPU-only evidence runs must set CUDA_VISIBLE_DEVICES to an empty string or -1"
        )
    _load_protocol_lock(protocol_path)
    cases = generate_full_evidence_cases()
    canonical_manifest = manifest_bytes(cases)
    replay_root = output_dir / "replays" / ablation_id
    replay_root.mkdir(parents=True, exist_ok=True)
    replay_dir = replay_root / replay_id
    replay_dir.mkdir(exist_ok=False)
    manifest_path = replay_dir / "case_manifest.jsonl"
    manifest_path.write_bytes(canonical_manifest)

    started = datetime.now(timezone.utc)
    rows: list[dict[str, object]] = []
    for case in order_cases(cases, replay_id):
        started_ns = time.perf_counter_ns()
        verdict = verify_with_ablation(case.trace, ablation_id)
        elapsed_ns = time.perf_counter_ns() - started_ns
        independent = (
            list(independent_invariant_violations(case.trace))
            if ablation_id == "A0-full" and verdict.accepted
            else []
        )
        rows.append(
            {
                "ablation_id": ablation_id,
                "replay_id": replay_id,
                "case_id": case.case_id,
                "set_id": case.set_id,
                "base_id": case.base_id,
                "family": case.family,
                "mutation_id": case.mutation_id,
                "expected_valid": case.expected_valid,
                "accepted": verdict.accepted,
                "correct": verdict.accepted == case.expected_valid,
                "reason_codes": [failure.code.value for failure in verdict.failures],
                "checked_predicates": list(verdict.checked_predicates),
                "trace_fingerprint": verdict.trace_fingerprint,
                "independent_invariant_violations": independent,
                "elapsed_ns": elapsed_ns,
            }
        )
    verdict_path = replay_dir / "verdicts.jsonl"
    verdict_path.write_bytes(_jsonl_bytes(rows))
    completed = datetime.now(timezone.utc)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "full_evidence_id": FULL_EVIDENCE_ID,
        "protocol_id": PROTOCOL_ID,
        "ablation_id": ablation_id,
        "replay_id": replay_id,
        "case_count": len(cases),
        "verdict_row_count": len(rows),
        "case_manifest_sha256": _sha256(manifest_path),
        "verdicts_sha256": _sha256(verdict_path),
        "protocol_sha256": _sha256(protocol_path),
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "primary_runner_sha256": EXPECTED_PRIMARY_RUNNER_SHA256,
        "full_evidence_runner_sha256": _sha256(Path(__file__)),
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
    (replay_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _percentile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarize_full_evidence(
    output_dir: Path,
    protocol_path: Path,
) -> dict[str, object]:
    _load_protocol_lock(protocol_path)
    indexed: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    metadata_values: list[dict[str, object]] = []
    manifest_hashes: set[str] = set()
    runner_hashes: set[str] = set()
    for ablation_id in ABLATION_IDS:
        for replay_id in REPLAY_IDS:
            replay_dir = output_dir / "replays" / ablation_id / replay_id
            metadata_path = replay_dir / "metadata.json"
            verdict_path = replay_dir / "verdicts.jsonl"
            manifest_path = replay_dir / "case_manifest.jsonl"
            if not all(path.is_file() for path in (metadata_path, verdict_path, manifest_path)):
                raise FileNotFoundError(f"incomplete ablation replay: {ablation_id}/{replay_id}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("conda_environment") != "LQ_signal":
                raise ValueError(f"wrong conda environment: {ablation_id}/{replay_id}")
            if metadata.get("physical_gpu_indices") != [] or metadata.get("multi_gpu") is not False:
                raise ValueError(f"GPU contract violated: {ablation_id}/{replay_id}")
            rows = _read_jsonl(verdict_path)
            index = {str(row["case_id"]): row for row in rows}
            if len(index) != 136:
                raise ValueError(f"unexpected row count: {ablation_id}/{replay_id}")
            indexed[(ablation_id, replay_id)] = index
            metadata_values.append(metadata)
            manifest_hashes.add(str(metadata["case_manifest_sha256"]))
            runner_hashes.add(str(metadata["full_evidence_runner_sha256"]))
    if len(manifest_hashes) != 1 or len(runner_hashes) != 1:
        raise ValueError("ablation provenance hashes disagree")

    stable_fields = (
        "accepted",
        "reason_codes",
        "checked_predicates",
        "trace_fingerprint",
        "independent_invariant_violations",
    )
    disagreements: list[dict[str, object]] = []
    for ablation_id in ABLATION_IDS:
        reference = indexed[(ablation_id, "R0")]
        for replay_id in REPLAY_IDS[1:]:
            observed = indexed[(ablation_id, replay_id)]
            if set(observed) != set(reference):
                raise ValueError(f"case keys disagree: {ablation_id}/{replay_id}")
            for case_id, reference_row in reference.items():
                differing = [
                    field
                    for field in stable_fields
                    if observed[case_id][field] != reference_row[field]
                ]
                if differing:
                    disagreements.append(
                        {
                            "ablation_id": ablation_id,
                            "replay_id": replay_id,
                            "case_id": case_id,
                            "fields": differing,
                        }
                    )

    a0 = indexed[("A0-full", "R0")]
    u_rows = [row for row in a0.values() if row["set_id"] == UNIVERSE_ID]
    d_rows = [row for row in a0.values() if row["set_id"] == DIAGNOSTIC_SET_ID]
    s_rows = [row for row in a0.values() if row["set_id"] == SENSITIVITY_SET_ID]
    u_valid = [row for row in u_rows if row["expected_valid"]]
    u_invalid = [row for row in u_rows if not row["expected_valid"]]
    independent_violations = [
        {"case_id": row["case_id"], "violations": row["independent_invariant_violations"]}
        for row in a0.values()
        if row["accepted"] and row["independent_invariant_violations"]
    ]
    failure_profile = Counter(
        code
        for row in u_invalid + d_rows
        for code in row["reason_codes"]
    )

    ablation_summary: dict[str, dict[str, object]] = {}
    invalid_case_ids = {
        case_id for case_id, row in a0.items() if not bool(row["expected_valid"])
    }
    valid_case_ids = {
        case_id for case_id, row in a0.items() if bool(row["expected_valid"])
    }
    for ablation_id in ABLATION_IDS:
        rows = indexed[(ablation_id, "R0")]
        newly_accepted = sorted(
            case_id
            for case_id in invalid_case_ids
            if not bool(a0[case_id]["accepted"]) and bool(rows[case_id]["accepted"])
        )
        newly_rejected = sorted(
            case_id
            for case_id in valid_case_ids
            if bool(a0[case_id]["accepted"]) and not bool(rows[case_id]["accepted"])
        )
        family_counts = Counter(
            str(rows[case_id]["family"]) for case_id in newly_accepted
        )
        ablation_summary[ablation_id] = {
            "newly_accepted_invalid_count": len(newly_accepted),
            "newly_accepted_invalid_case_ids": newly_accepted,
            "newly_accepted_by_family": dict(sorted(family_counts.items())),
            "newly_rejected_valid_count": len(newly_rejected),
            "newly_rejected_valid_case_ids": newly_rejected,
        }

    efficiency: dict[str, dict[str, int]] = {}
    hard_binary_brier: dict[str, float] = {}
    for ablation_id in ABLATION_IDS:
        elapsed = [
            int(row["elapsed_ns"])
            for replay_id in REPLAY_IDS
            for row in indexed[(ablation_id, replay_id)].values()
        ]
        efficiency[ablation_id] = {
            "evaluation_count": len(elapsed),
            "median_elapsed_ns": _percentile(elapsed, 0.50),
            "p95_elapsed_ns": _percentile(elapsed, 0.95),
        }
        reference_rows = indexed[(ablation_id, "R0")].values()
        hard_binary_brier[ablation_id] = sum(
            float(bool(row["accepted"]) != bool(row["expected_valid"]))
            for row in reference_rows
        ) / 136.0

    checks = {
        "u_valid_accepted_4_of_4": sum(bool(row["accepted"]) for row in u_valid) == 4,
        "u_invalid_rejected_88_of_88": sum(not bool(row["accepted"]) for row in u_invalid) == 88,
        "diagnostics_rejected_24_of_24": sum(not bool(row["accepted"]) for row in d_rows) == 24,
        "sensitivity_accepted_20_of_20": sum(bool(row["accepted"]) for row in s_rows) == 20,
        "zero_replay_disagreements": len(disagreements) == 0,
        "zero_independent_invariant_violations": len(independent_violations) == 0,
        "content_digest_attribution": ablation_summary["A1-no-content-digest"]["newly_accepted_invalid_count"] >= 4,
        "sample_binding_attribution": ablation_summary["A3-no-sample"]["newly_accepted_invalid_count"] >= 12,
        "witness_binding_attribution": ablation_summary["A4-no-witness-binding"]["newly_accepted_invalid_count"] >= 24,
        "signal_representation_attribution": ablation_summary["A5-no-signal-representation"]["newly_accepted_invalid_count"] >= 4,
        "representation_symbol_attribution": ablation_summary["A6-no-representation-symbol"]["newly_accepted_invalid_count"] >= 4,
        "symbol_language_attribution": ablation_summary["A7-no-symbol-language"]["newly_accepted_invalid_count"] >= 4,
        "root_composition_attribution": ablation_summary["A8-no-root-composition"]["newly_accepted_invalid_count"] >= 28,
    }

    canonical_manifest = output_dir / "case_manifest.jsonl"
    canonical_verdicts = output_dir / "verdicts.jsonl"
    result_path = output_dir / "result.json"
    log_path = output_dir / "run.log"
    if any(path.exists() for path in (canonical_manifest, canonical_verdicts, result_path, log_path)):
        raise FileExistsError("full-evidence summary artifacts already exist")
    shutil.copyfile(
        output_dir / "replays" / "A0-full" / "R0" / "case_manifest.jsonl",
        canonical_manifest,
    )
    canonical_rows = [
        row
        for ablation_id in ABLATION_IDS
        for row in indexed[(ablation_id, "R0")].values()
    ]
    canonical_verdicts.write_bytes(
        _jsonl_bytes(sorted(canonical_rows, key=lambda row: (str(row["ablation_id"]), str(row["case_id"]))))
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "full_evidence_id": FULL_EVIDENCE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "completed",
        "outcome": "supported" if all(checks.values()) else "refuted",
        "claim_scope": ["C1", "C2"],
        "case_count": 136,
        "ablation_count": len(ABLATION_IDS),
        "replay_count": len(REPLAY_IDS),
        "clean_process_count": len(ABLATION_IDS) * len(REPLAY_IDS),
        "random_seed_count": 0,
        "u_valid_accepted": sum(bool(row["accepted"]) for row in u_valid),
        "u_invalid_rejected": sum(not bool(row["accepted"]) for row in u_invalid),
        "diagnostics_rejected": sum(not bool(row["accepted"]) for row in d_rows),
        "sensitivity_accepted": sum(bool(row["accepted"]) for row in s_rows),
        "independent_invariant_violations": independent_violations,
        "failure_code_profile": dict(sorted(failure_profile.items())),
        "ablation_summary": ablation_summary,
        "replay_disagreement_count": len(disagreements),
        "replay_disagreements": disagreements,
        "efficiency": {
            "status": "descriptive_process_local",
            "clock": "time.perf_counter_ns",
            "warmup_policy": "none_predeclared_small_census",
            "hardware_normalization": "none",
            "superiority_claim_permitted": False,
            "by_ablation": efficiency,
        },
        "calibration": {
            "status": "not_applicable",
            "probabilistic_outputs_available": False,
            "reason": "the verifier emits a deterministic Boolean contract verdict, not a probability",
            "probabilistic_calibration_claim_permitted": False,
            "hard_binary_brier_diagnostic_by_ablation": hard_binary_brier,
        },
        "threshold_checks": checks,
        "case_manifest_sha256": _sha256(canonical_manifest),
        "verdicts_sha256": _sha256(canonical_verdicts),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "primary_runner_sha256": EXPECTED_PRIMARY_RUNNER_SHA256,
        "full_evidence_runner_sha256": next(iter(runner_hashes)),
        "conda_environment": "LQ_signal",
        "command_prefix": "conda run -n LQ_signal",
        "physical_gpu_indices": [],
        "multi_gpu": False,
        "metadata_count": len(metadata_values),
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "outcome": result["outcome"],
                "result_sha256": _sha256(result_path),
                "clean_process_count": result["clean_process_count"],
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
    run_parser = subparsers.add_parser("run", help="run one ablation/replay process")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--protocol-path", type=Path, required=True)
    run_parser.add_argument("--ablation-id", choices=ABLATION_IDS, required=True)
    run_parser.add_argument("--replay-id", choices=REPLAY_IDS, required=True)
    summary_parser = subparsers.add_parser("summarize", help="summarize all 45 processes")
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    summary_parser.add_argument("--protocol-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        value = run_ablation_replay(
            args.output_dir.resolve(),
            args.protocol_path.resolve(),
            args.ablation_id,
            args.replay_id,
        )
    else:
        value = summarize_full_evidence(
            args.output_dir.resolve(), args.protocol_path.resolve()
        )
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABLATION_IDS",
    "DIAGNOSTIC_SET_ID",
    "EvidenceCase",
    "FULL_EVIDENCE_ID",
    "SENSITIVITY_SET_ID",
    "generate_diagnostic_cases",
    "generate_full_evidence_cases",
    "generate_sensitivity_cases",
    "independent_invariant_violations",
    "manifest_bytes",
    "order_cases",
    "run_ablation_replay",
    "summarize_full_evidence",
    "verify_with_ablation",
]
