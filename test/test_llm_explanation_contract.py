from __future__ import annotations

from dataclasses import dataclass

import pytest

from phmfactory.explanation import (
    EvidenceAtom,
    EvidencePath,
    PHMExplanationState,
    PredictionState,
    build_llm_packet,
    explain_with_llm,
    freeze_mapping,
    parse_llm_explanation,
    state_from_tspn_uxfd_fuzzy_trace,
    state_from_xoan_report,
)


def _minimal_state() -> PHMExplanationState:
    return PHMExplanationState(
        sample_id="case-1",
        task="fault_diagnosis",
        model_family="test",
        trace_kind="test_trace",
        prediction=PredictionState(
            label="outer_race_fault",
            class_index=1,
            confidence=0.8,
            logits=(0.1, 1.2),
        ),
        evidence_atoms=(
            EvidenceAtom(
                id="e:order",
                kind="spectral_order",
                name="BPFO order",
                value=3.58,
                unit="order",
            ),
        ),
        evidence_paths=(
            EvidencePath(
                id="p:outer",
                atom_ids=("e:order",),
                relation="supports outer-race fault claim",
            ),
        ),
        operating_conditions=freeze_mapping(
            {"speed_rpm": 1800}, name="operating_conditions"
        ),
        capabilities=("prediction", "typed_evidence", "structural_path"),
    )


def test_state_rejects_unknown_path_atom() -> None:
    with pytest.raises(ValueError, match="unknown atoms"):
        PHMExplanationState(
            sample_id="case-1",
            task="fault_diagnosis",
            model_family="test",
            trace_kind="test_trace",
            prediction=PredictionState(label="normal"),
            evidence_atoms=(EvidenceAtom(id="e:1", kind="feature", name="one"),),
            evidence_paths=(
                EvidencePath(id="p:1", atom_ids=("e:missing",), relation="uses"),
            ),
        )


def test_xoan_report_adapter_preserves_path_and_uncertainty() -> None:
    report = {
        "relaxed_logits": [[0.2, 1.7, -0.1]],
        "serialized_paths": ["D1(MA3(x))"],
        "logit_relative_rmse": [0.02],
        "predictive_entropy": [0.21],
        "dictionary_insufficiency_score": [0.14],
        "relative_rmse": [0.03],
        "dictionary_manifest": {"operators": ["D1", "MA3"]},
        "score_calibration_state": "uncalibrated",
        "insufficiency_score_id": "p07_dictionary_insufficiency_v2",
    }

    state = state_from_xoan_report(
        report,
        sample_id="xoan-1",
        class_names=("normal", "inner", "outer"),
        operating_conditions={"speed_rpm": 1200},
    )

    assert state.prediction.label == "inner"
    assert state.evidence_paths[0].id == "path:selected"
    assert state.evidence_atoms[0].value == "D1(MA3(x))"
    assert "structural_path" in state.capabilities
    assert "additive_contribution" not in state.capabilities
    assert dict(state.uncertainty.metrics)["predictive_entropy"] == pytest.approx(0.21)


@dataclass
class _FuzzyTrace:
    normalized_rule_firing: list[list[float]]
    rule_contributions: list[list[list[float]]]


@dataclass
class _FuzzyOutput:
    logits: list[list[float]]
    non_fuzzy_logits: list[list[float]]
    fuzzy_scale: float
    fuzzy_trace: _FuzzyTrace


def test_fuzzy_adapter_exports_reconstructable_contributions() -> None:
    output = _FuzzyOutput(
        logits=[[0.55, 1.05]],
        non_fuzzy_logits=[[0.2, 0.4]],
        fuzzy_scale=1.0,
        fuzzy_trace=_FuzzyTrace(
            normalized_rule_firing=[[0.7, 0.3]],
            rule_contributions=[[[0.25, 0.45], [0.10, 0.20]]],
        ),
    )

    state = state_from_tspn_uxfd_fuzzy_trace(
        output,
        sample_id="fuzzy-1",
        class_names=("normal", "fault"),
        rule_names=("high impulsiveness", "order family"),
        rule_relations={0: ("supports", "fault")},
    )

    assert state.prediction.label == "fault"
    assert len(state.evidence_paths) == 2
    assert len(state.contributions) == 6
    assert state.mechanism_relations[0].status == "mechanism-constrained"
    assert dict(state.uncertainty.metrics)[
        "max_logit_reconstruction_residual"
    ] == pytest.approx(0.0)


def test_llm_packet_is_provider_neutral_and_contains_no_hidden_reasoning_request() -> None:
    packet = build_llm_packet(_minimal_state(), audience="maintenance engineer")
    assert packet["state"]["schema_version"] == "phm-eir/v1"
    assert "chain of thought" in packet["constraints"][-1]
    assert "provider" not in packet


def test_parser_rejects_invented_evidence() -> None:
    payload = {
        "summary": "Outer-race fault.",
        "claims": [
            {
                "text": "The result is supported by an invented feature.",
                "evidence_ids": ["e:invented"],
                "path_ids": [],
                "relation_ids": [],
                "uncertainty": "",
            }
        ],
        "limitations": [],
    }
    with pytest.raises(ValueError, match="unknown evidence"):
        parse_llm_explanation(payload, _minimal_state())


def test_explain_with_llm_uses_one_explicit_callback_without_fallback() -> None:
    calls = []

    def generate(packet):
        calls.append(packet)
        return {
            "summary": "The model predicts an outer-race fault.",
            "claims": [
                {
                    "text": "The model-native path includes the BPFO-order evidence.",
                    "evidence_ids": ["e:order"],
                    "path_ids": ["p:outer"],
                    "relation_ids": [],
                    "uncertainty": "The state reports confidence 0.8.",
                }
            ],
            "limitations": [
                "No mechanism-constrained relation was supplied in this state."
            ],
        }

    explanation = explain_with_llm(_minimal_state(), generate)
    assert len(calls) == 1
    assert explanation.claims[0].path_ids == ("p:outer",)
    assert explanation.limitations
