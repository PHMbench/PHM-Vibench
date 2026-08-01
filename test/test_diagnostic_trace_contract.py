from __future__ import annotations

import hashlib
from dataclasses import replace

from src.explain_factory.diagnostic_trace_contract import (
    DEFAULT_CONTRACT,
    DiagnosticTrace,
    FailureCode,
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


def _bind(records: tuple[StageRecord, ...]) -> DiagnosticTrace:
    by_stage = {record.stage: record for record in records}
    witnesses = tuple(
        make_edge_witness(
            by_stage[edge.source_stage],
            by_stage[edge.target_stage],
            edge.predicate_id,
        )
        for edge in DEFAULT_CONTRACT.edges
    )
    return DiagnosticTrace(records=records, witnesses=witnesses)


def _valid_trace() -> DiagnosticTrace:
    sample_id = "sample-0001"
    signal = StageRecord(
        stage=Stage.SIGNAL,
        sample_id=sample_id,
        payload=SignalPayload(
            content_digest=hashlib.sha256(b"sample-0001-waveform-v1").hexdigest(),
            sampling_rate_hz=12_000.0,
            channel_names=("drive_end",),
            sample_count=2_048,
        ),
    )
    signal_digest = payload_digest(signal.payload)
    representation = StageRecord(
        stage=Stage.REPRESENTATION,
        sample_id=sample_id,
        payload=RepresentationPayload(
            source_signal_digest=signal_digest,
            root_signal_digest=signal_digest,
            feature_names=("bpfo_energy", "rms"),
            feature_values=(0.82, 0.31),
        ),
    )
    symbolic = StageRecord(
        stage=Stage.SYMBOLIC,
        sample_id=sample_id,
        payload=SymbolicPayload(
            source_representation_digest=payload_digest(representation.payload),
            root_signal_digest=signal_digest,
            symbols=("outer_race_fault",),
            support_features=("bpfo_energy",),
        ),
    )
    language = StageRecord(
        stage=Stage.LANGUAGE,
        sample_id=sample_id,
        payload=LanguagePayload(
            source_symbol_digest=payload_digest(symbolic.payload),
            root_signal_digest=signal_digest,
            text="The verified symbol is outer_race_fault.",
            mentioned_symbols=("outer_race_fault",),
        ),
    )
    return _bind((signal, representation, symbolic, language))


def _codes(trace: DiagnosticTrace) -> set[FailureCode]:
    return {failure.code for failure in verify_trace(trace).failures}


def test_valid_trace_is_accepted_deterministically() -> None:
    trace = _valid_trace()
    first = verify_trace(trace)
    second = verify_trace(trace)

    assert first == second
    assert first.accepted is True
    assert first.failures == ()
    assert first.checked_predicates == tuple(
        edge.predicate_id for edge in DEFAULT_CONTRACT.edges
    )


def test_missing_witness_is_rejected() -> None:
    trace = _valid_trace()
    mutated = replace(trace, witnesses=trace.witnesses[:1] + trace.witnesses[2:])

    assert FailureCode.WITNESS_CARDINALITY in _codes(mutated)


def test_wrong_payload_type_is_rejected() -> None:
    trace = _valid_trace()
    records = list(trace.records)
    records[1] = replace(records[1], payload=trace.records[0].payload)
    mutated = replace(trace, records=tuple(records))

    assert FailureCode.TYPE_MISMATCH in _codes(mutated)


def test_swapped_symbol_is_rejected_by_language_predicate() -> None:
    trace = _valid_trace()
    records = list(trace.records)
    language = records[3]
    assert isinstance(language.payload, LanguagePayload)
    records[3] = replace(
        language,
        payload=replace(
            language.payload,
            text="The verified symbol is inner_race_fault.",
            mentioned_symbols=("inner_race_fault",),
        ),
    )
    mutated = _bind(tuple(records))

    assert FailureCode.PREDICATE_FAILED in _codes(mutated)


def test_stale_sample_is_rejected_by_sample_binding() -> None:
    trace = _valid_trace()
    records = list(trace.records)
    records[3] = replace(records[3], sample_id="sample-stale")
    mutated = replace(trace, records=tuple(records))

    assert FailureCode.SAMPLE_BINDING in _codes(mutated)


def test_non_compositional_trace_fails_after_all_local_predicates_pass() -> None:
    trace = _valid_trace()
    signal, representation, symbolic, language = trace.records
    assert isinstance(representation.payload, RepresentationPayload)
    assert isinstance(symbolic.payload, SymbolicPayload)
    assert isinstance(language.payload, LanguagePayload)

    stale_root = "0" * 64
    representation = replace(
        representation,
        payload=replace(representation.payload, root_signal_digest=stale_root),
    )
    symbolic = replace(
        symbolic,
        payload=replace(
            symbolic.payload,
            source_representation_digest=payload_digest(representation.payload),
            root_signal_digest=stale_root,
        ),
    )
    language = replace(
        language,
        payload=replace(
            language.payload,
            source_symbol_digest=payload_digest(symbolic.payload),
            root_signal_digest=stale_root,
        ),
    )
    mutated = _bind((signal, representation, symbolic, language))
    result = verify_trace(mutated)

    assert result.checked_predicates == tuple(
        edge.predicate_id for edge in DEFAULT_CONTRACT.edges
    )
    assert [failure.code for failure in result.failures] == [
        FailureCode.COMPOSITION_ROOT
    ]


def test_payload_and_trace_fingerprints_are_order_stable() -> None:
    trace = _valid_trace()
    reordered = replace(
        trace,
        records=tuple(reversed(trace.records)),
        witnesses=tuple(reversed(trace.witnesses)),
    )

    assert verify_trace(reordered) == verify_trace(trace)
