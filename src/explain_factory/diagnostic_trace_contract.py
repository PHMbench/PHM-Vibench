"""Typed, sample-bound diagnostic trace contracts.

The verifier in this module is deliberately independent of model training and
uses only the Python standard library.  Adapters may construct trace records,
but they may not replace the predicate registry or relax the default contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, TypeAlias


class Stage(str, Enum):
    """The ordered stages in a diagnostic explanation trace."""

    SIGNAL = "signal"
    REPRESENTATION = "representation"
    SYMBOLIC = "symbolic"
    LANGUAGE = "language"


@dataclass(frozen=True)
class SignalPayload:
    content_digest: str
    sampling_rate_hz: float
    channel_names: tuple[str, ...]
    sample_count: int


@dataclass(frozen=True)
class RepresentationPayload:
    source_signal_digest: str
    root_signal_digest: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]


@dataclass(frozen=True)
class SymbolicPayload:
    source_representation_digest: str
    root_signal_digest: str
    symbols: tuple[str, ...]
    support_features: tuple[str, ...]


@dataclass(frozen=True)
class LanguagePayload:
    source_symbol_digest: str
    root_signal_digest: str
    text: str
    mentioned_symbols: tuple[str, ...]


Payload: TypeAlias = (
    SignalPayload | RepresentationPayload | SymbolicPayload | LanguagePayload
)


@dataclass(frozen=True)
class StageRecord:
    stage: Stage
    sample_id: str
    payload: Payload


@dataclass(frozen=True)
class EdgeWitness:
    source_stage: Stage
    target_stage: Stage
    sample_id: str
    source_digest: str
    target_digest: str
    predicate_id: str


@dataclass(frozen=True)
class DiagnosticTrace:
    records: tuple[StageRecord, ...]
    witnesses: tuple[EdgeWitness, ...]


@dataclass(frozen=True)
class StageSpec:
    stage: Stage
    payload_type: type[object]


@dataclass(frozen=True)
class EdgeSpec:
    source_stage: Stage
    target_stage: Stage
    predicate_id: str


@dataclass(frozen=True)
class DiagnosticTraceContract:
    contract_id: str
    stages: tuple[StageSpec, ...]
    edges: tuple[EdgeSpec, ...]
    digest_algorithm: str = "sha256-canonical-json-v1"


class FailureCode(str, Enum):
    STAGE_CARDINALITY = "stage_cardinality"
    UNEXPECTED_STAGE = "unexpected_stage"
    TYPE_MISMATCH = "type_mismatch"
    SAMPLE_BINDING = "sample_binding"
    PAYLOAD_INVALID = "payload_invalid"
    WITNESS_CARDINALITY = "witness_cardinality"
    WITNESS_UNEXPECTED = "witness_unexpected"
    WITNESS_PREDICATE = "witness_predicate"
    WITNESS_SAMPLE = "witness_sample"
    WITNESS_SOURCE_DIGEST = "witness_source_digest"
    WITNESS_TARGET_DIGEST = "witness_target_digest"
    PREDICATE_FAILED = "predicate_failed"
    COMPOSITION_ROOT = "composition_root"


@dataclass(frozen=True)
class VerificationFailure:
    code: FailureCode
    location: str
    detail: str


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    contract_id: str
    trace_fingerprint: str
    checked_predicates: tuple[str, ...]
    failures: tuple[VerificationFailure, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "contract_id": self.contract_id,
            "trace_fingerprint": self.trace_fingerprint,
            "checked_predicates": list(self.checked_predicates),
            "failures": [
                {
                    "code": failure.code.value,
                    "location": failure.location,
                    "detail": failure.detail,
                }
                for failure in self.failures
            ],
        }


@dataclass(frozen=True)
class PredicateVerdict:
    accepted: bool
    detail: str


Predicate: TypeAlias = Callable[[StageRecord, StageRecord], PredicateVerdict]


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def payload_digest(payload: object) -> str:
    """Return a deterministic digest for a frozen dataclass payload."""

    if not is_dataclass(payload) or isinstance(payload, type):
        raise TypeError("payload must be a dataclass instance")
    encoded = json.dumps(
        asdict(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_edge_witness(
    source: StageRecord,
    target: StageRecord,
    predicate_id: str,
) -> EdgeWitness:
    """Bind an adjacent pair to one sample and two immutable payload digests."""

    if source.sample_id != target.sample_id:
        raise ValueError("cannot bind records from different samples")
    return EdgeWitness(
        source_stage=source.stage,
        target_stage=target.stage,
        sample_id=source.sample_id,
        source_digest=payload_digest(source.payload),
        target_digest=payload_digest(target.payload),
        predicate_id=predicate_id,
    )


def _signal_to_representation(
    source: StageRecord,
    target: StageRecord,
) -> PredicateVerdict:
    if not isinstance(source.payload, SignalPayload) or not isinstance(
        target.payload, RepresentationPayload
    ):
        return PredicateVerdict(False, "predicate received incompatible payload types")
    signal = source.payload
    representation = target.payload
    if not _is_sha256_hex(signal.content_digest):
        return PredicateVerdict(False, "signal content digest must be SHA-256 hex")
    if signal.sampling_rate_hz <= 0 or signal.sample_count <= 0:
        return PredicateVerdict(False, "signal dimensions and sampling rate must be positive")
    if not signal.channel_names or len(set(signal.channel_names)) != len(
        signal.channel_names
    ):
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


def _representation_to_symbol(
    source: StageRecord,
    target: StageRecord,
) -> PredicateVerdict:
    if not isinstance(source.payload, RepresentationPayload) or not isinstance(
        target.payload, SymbolicPayload
    ):
        return PredicateVerdict(False, "predicate received incompatible payload types")
    representation = source.payload
    symbolic = target.payload
    if symbolic.source_representation_digest != payload_digest(representation):
        return PredicateVerdict(False, "symbolic payload is not bound to the representation")
    if not symbolic.symbols or len(set(symbolic.symbols)) != len(symbolic.symbols):
        return PredicateVerdict(False, "symbols must be non-empty and unique")
    if not symbolic.support_features:
        return PredicateVerdict(False, "at least one supporting feature is required")
    if not set(symbolic.support_features).issubset(representation.feature_names):
        return PredicateVerdict(False, "symbol support refers to an unknown feature")
    return PredicateVerdict(True, "representation-to-symbol predicate satisfied")


def _symbol_to_language(
    source: StageRecord,
    target: StageRecord,
) -> PredicateVerdict:
    if not isinstance(source.payload, SymbolicPayload) or not isinstance(
        target.payload, LanguagePayload
    ):
        return PredicateVerdict(False, "predicate received incompatible payload types")
    symbolic = source.payload
    language = target.payload
    if language.source_symbol_digest != payload_digest(symbolic):
        return PredicateVerdict(False, "language payload is not bound to the symbolic payload")
    if not language.text.strip():
        return PredicateVerdict(False, "language text must be non-empty")
    if not language.mentioned_symbols or len(set(language.mentioned_symbols)) != len(
        language.mentioned_symbols
    ):
        return PredicateVerdict(False, "mentioned symbols must be non-empty and unique")
    if not set(language.mentioned_symbols).issubset(symbolic.symbols):
        return PredicateVerdict(False, "language mentions a symbol absent from the symbolic stage")
    if not all(symbol in language.text for symbol in language.mentioned_symbols):
        return PredicateVerdict(False, "language text omits a declared mentioned symbol")
    return PredicateVerdict(True, "symbol-to-language predicate satisfied")


PREDICATE_REGISTRY: Mapping[str, Predicate] = MappingProxyType(
    {
        "signal_to_representation_v1": _signal_to_representation,
        "representation_to_symbol_v1": _representation_to_symbol,
        "symbol_to_language_v1": _symbol_to_language,
    }
)


DEFAULT_CONTRACT = DiagnosticTraceContract(
    contract_id="p06-diagnostic-trace-v1",
    stages=(
        StageSpec(Stage.SIGNAL, SignalPayload),
        StageSpec(Stage.REPRESENTATION, RepresentationPayload),
        StageSpec(Stage.SYMBOLIC, SymbolicPayload),
        StageSpec(Stage.LANGUAGE, LanguagePayload),
    ),
    edges=(
        EdgeSpec(
            Stage.SIGNAL,
            Stage.REPRESENTATION,
            "signal_to_representation_v1",
        ),
        EdgeSpec(
            Stage.REPRESENTATION,
            Stage.SYMBOLIC,
            "representation_to_symbol_v1",
        ),
        EdgeSpec(Stage.SYMBOLIC, Stage.LANGUAGE, "symbol_to_language_v1"),
    ),
)


def _trace_fingerprint(trace: DiagnosticTrace) -> str:
    records = sorted(
        (
            record.stage.value,
            record.sample_id,
            payload_digest(record.payload),
        )
        for record in trace.records
    )
    witnesses = sorted(
        (
            witness.source_stage.value,
            witness.target_stage.value,
            witness.sample_id,
            witness.source_digest,
            witness.target_digest,
            witness.predicate_id,
        )
        for witness in trace.witnesses
    )
    encoded = json.dumps(
        {"records": records, "witnesses": witnesses},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _result(
    trace: DiagnosticTrace,
    contract: DiagnosticTraceContract,
    checked_predicates: list[str],
    failures: list[VerificationFailure],
) -> VerificationResult:
    try:
        fingerprint = _trace_fingerprint(trace)
    except (TypeError, ValueError):
        fingerprint = "unavailable"
    return VerificationResult(
        accepted=not failures,
        contract_id=contract.contract_id,
        trace_fingerprint=fingerprint,
        checked_predicates=tuple(checked_predicates),
        failures=tuple(failures),
    )


def verify_trace(
    trace: DiagnosticTrace,
    contract: DiagnosticTraceContract = DEFAULT_CONTRACT,
) -> VerificationResult:
    """Verify a trace deterministically and fail closed on every obligation."""

    failures: list[VerificationFailure] = []
    checked_predicates: list[str] = []
    expected_stages = {spec.stage for spec in contract.stages}
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

    for spec in contract.stages:
        count = len(records_by_stage[spec.stage])
        if count != 1:
            failures.append(
                VerificationFailure(
                    FailureCode.STAGE_CARDINALITY,
                    spec.stage.value,
                    f"expected exactly one record, observed {count}",
                )
            )

    if any(len(records_by_stage[spec.stage]) != 1 for spec in contract.stages):
        return _result(trace, contract, checked_predicates, failures)

    records = {
        stage: stage_records[0] for stage, stage_records in records_by_stage.items()
    }
    type_ok: dict[Stage, bool] = {}
    for spec in contract.stages:
        record = records[spec.stage]
        type_ok[spec.stage] = isinstance(record.payload, spec.payload_type)
        if not type_ok[spec.stage]:
            failures.append(
                VerificationFailure(
                    FailureCode.TYPE_MISMATCH,
                    spec.stage.value,
                    f"expected {spec.payload_type.__name__}, observed {type(record.payload).__name__}",
                )
            )

    signal_record = records[Stage.SIGNAL]
    canonical_sample_id = signal_record.sample_id
    if not canonical_sample_id.strip():
        failures.append(
            VerificationFailure(
                FailureCode.SAMPLE_BINDING,
                Stage.SIGNAL.value,
                "sample_id must be non-empty",
            )
        )
    for spec in contract.stages[1:]:
        if records[spec.stage].sample_id != canonical_sample_id:
            failures.append(
                VerificationFailure(
                    FailureCode.SAMPLE_BINDING,
                    spec.stage.value,
                    "stage sample_id differs from the signal sample_id",
                )
            )

    digests: dict[Stage, str] = {}
    for spec in contract.stages:
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

    expected_edges = {
        (edge.source_stage, edge.target_stage) for edge in contract.edges
    }
    unexpected_witnesses = sorted(
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
    )
    for witness in unexpected_witnesses:
        failures.append(
            VerificationFailure(
                FailureCode.WITNESS_UNEXPECTED,
                f"{witness.source_stage.value}->{witness.target_stage.value}",
                "witness edge is absent from the contract",
            )
        )

    for edge in contract.edges:
        location = f"{edge.source_stage.value}->{edge.target_stage.value}"
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
        witness_ok = True
        if witness.predicate_id != edge.predicate_id:
            witness_ok = False
            failures.append(
                VerificationFailure(
                    FailureCode.WITNESS_PREDICATE,
                    location,
                    f"expected {edge.predicate_id}, observed {witness.predicate_id}",
                )
            )
        if witness.sample_id != canonical_sample_id:
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

        if (
            witness_ok
            and type_ok[edge.source_stage]
            and type_ok[edge.target_stage]
            and edge.predicate_id in PREDICATE_REGISTRY
        ):
            verdict = PREDICATE_REGISTRY[edge.predicate_id](
                records[edge.source_stage],
                records[edge.target_stage],
            )
            checked_predicates.append(edge.predicate_id)
            if not verdict.accepted:
                failures.append(
                    VerificationFailure(
                        FailureCode.PREDICATE_FAILED,
                        location,
                        verdict.detail,
                    )
                )
        elif edge.predicate_id not in PREDICATE_REGISTRY:
            failures.append(
                VerificationFailure(
                    FailureCode.WITNESS_PREDICATE,
                    location,
                    "contract predicate is absent from the immutable registry",
                )
            )

    if all(type_ok.values()) and Stage.SIGNAL in digests:
        expected_root = digests[Stage.SIGNAL]
        root_values = {
            Stage.REPRESENTATION: records[
                Stage.REPRESENTATION
            ].payload.root_signal_digest,
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

    return _result(trace, contract, checked_predicates, failures)


__all__ = [
    "DEFAULT_CONTRACT",
    "DiagnosticTrace",
    "DiagnosticTraceContract",
    "EdgeWitness",
    "FailureCode",
    "LanguagePayload",
    "PREDICATE_REGISTRY",
    "RepresentationPayload",
    "SignalPayload",
    "Stage",
    "StageRecord",
    "SymbolicPayload",
    "VerificationFailure",
    "VerificationResult",
    "make_edge_witness",
    "payload_digest",
    "verify_trace",
]
