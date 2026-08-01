from __future__ import annotations

import json
import os
from pathlib import Path

from src.explain_factory.diagnostic_trace_contract import verify_trace
from src.explain_factory.diagnostic_trace_full_evidence import (
    ABLATION_IDS,
    REPLAY_IDS,
    generate_diagnostic_cases,
    generate_full_evidence_cases,
    generate_sensitivity_cases,
    independent_invariant_violations,
    manifest_bytes,
    run_ablation_replay,
    summarize_full_evidence,
    verify_with_ablation,
)


P06_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = P06_ROOT / "paper" / "experiments" / "config_bridge.yaml"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_auxiliary_set_counts_and_unique_ids() -> None:
    diagnostics = generate_diagnostic_cases()
    sensitivity = generate_sensitivity_cases()
    full = generate_full_evidence_cases()
    assert len(diagnostics) == 24
    assert len(sensitivity) == 20
    assert len(full) == 136
    assert len({case.case_id for case in full}) == 136


def test_full_verifier_fails_closed_and_accepts_benign_sensitivity() -> None:
    diagnostics = generate_diagnostic_cases()
    sensitivity = generate_sensitivity_cases()
    assert all(not verify_trace(case.trace).accepted for case in diagnostics)
    assert all(verify_trace(case.trace).accepted for case in sensitivity)
    by_prefix = {
        prefix: [case for case in diagnostics if case.case_id.startswith(prefix)]
        for prefix in ("D1-", "D4-", "D5-", "D6-")
    }
    assert all(
        "witness_source_digest"
        in [failure.code.value for failure in verify_trace(case.trace).failures]
        for case in by_prefix["D1-"]
    )
    assert all(
        "predicate_failed"
        in [failure.code.value for failure in verify_trace(case.trace).failures]
        for case in by_prefix["D4-"] + by_prefix["D6-"]
    )
    assert all(
        "payload_invalid"
        in [failure.code.value for failure in verify_trace(case.trace).failures]
        for case in by_prefix["D5-"]
    )


def test_predeclared_single_obligation_ablation_attribution() -> None:
    cases = generate_full_evidence_cases()
    observed = {
        ablation_id: sum(
            not case.expected_valid
            and verify_with_ablation(case.trace, ablation_id).accepted
            for case in cases
        )
        for ablation_id in ABLATION_IDS
    }
    assert observed == {
        "A0-full": 0,
        "A1-no-content-digest": 4,
        "A2-no-type": 0,
        "A3-no-sample": 12,
        "A4-no-witness-binding": 24,
        "A5-no-signal-representation": 4,
        "A6-no-representation-symbol": 4,
        "A7-no-symbol-language": 4,
        "A8-no-root-composition": 28,
    }


def test_manifest_is_strict_json_with_explicit_nonfinite_sentinel() -> None:
    encoded = manifest_bytes(generate_full_evidence_cases()).decode("utf-8")
    rows = [json.loads(line) for line in encoded.splitlines()]
    assert len(rows) == 136
    assert '"nonfinite_float":"+Infinity"' in encoded
    assert ":Infinity" not in encoded


def test_independent_decoder_finds_no_violation_in_accepted_full_traces() -> None:
    for case in generate_full_evidence_cases():
        verdict = verify_trace(case.trace)
        if verdict.accepted:
            assert independent_invariant_violations(case.trace) == (), case.case_id


def test_45_clean_slots_summarize_and_refuse_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "full"
    for ablation_id in ABLATION_IDS:
        for replay_id in REPLAY_IDS:
            metadata = run_ablation_replay(
                output_dir, PROTOCOL_PATH, ablation_id, replay_id
            )
            assert metadata["case_count"] == 136
    try:
        run_ablation_replay(output_dir, PROTOCOL_PATH, "A0-full", "R0")
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing ablation replay must never be overwritten")

    result = summarize_full_evidence(output_dir, PROTOCOL_PATH)
    assert result["status"] == "completed"
    assert result["outcome"] == "supported"
    assert result["clean_process_count"] == 45
    assert result["u_valid_accepted"] == 4
    assert result["u_invalid_rejected"] == 88
    assert result["diagnostics_rejected"] == 24
    assert result["sensitivity_accepted"] == 20
    assert result["replay_disagreement_count"] == 0
    assert result["calibration"]["status"] == "not_applicable"
    assert all(result["threshold_checks"].values())
