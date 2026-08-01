from __future__ import annotations

import json
import os
from pathlib import Path

from src.explain_factory.diagnostic_trace_contract import FailureCode, verify_trace
from src.explain_factory.diagnostic_trace_experiment import (
    BASELINES,
    REPLAY_IDS,
    generate_universe,
    order_cases,
    run_replay,
    summarize_v1,
)


P06_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = P06_ROOT / "paper" / "experiments" / "config_bridge.yaml"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_frozen_universe_counts_and_ids() -> None:
    cases = generate_universe()
    assert len(cases) == 92
    assert len({case.case_id for case in cases}) == 92
    assert sum(case.expected_valid for case in cases) == 4
    assert {
        family: sum(case.family == family for case in cases)
        for family in {
            "valid",
            "missing",
            "wrong_type",
            "swapped_symbol",
            "stale_sample",
            "non_compositional",
        }
    } == {
        "valid": 4,
        "missing": 28,
        "wrong_type": 16,
        "swapped_symbol": 4,
        "stale_sample": 12,
        "non_compositional": 28,
    }


def test_proposed_verifier_matches_all_frozen_labels() -> None:
    for case in generate_universe():
        assert verify_trace(case.trace).accepted is case.expected_valid, case.case_id


def test_non_compositional_cases_pass_local_predicates_then_fail_root() -> None:
    cases = [case for case in generate_universe() if case.family == "non_compositional"]
    assert len(cases) == 28
    for case in cases:
        result = verify_trace(case.trace)
        assert len(result.checked_predicates) == 3
        assert [failure.code for failure in result.failures] == [
            FailureCode.COMPOSITION_ROOT
        ]


def test_each_baseline_has_a_strictly_distinguished_invalid_case() -> None:
    invalid_cases = [case for case in generate_universe() if not case.expected_valid]
    for baseline_id, baseline in BASELINES.items():
        count = sum(
            baseline(case.trace)[0] and not verify_trace(case.trace).accepted
            for case in invalid_cases
        )
        assert count >= 1, baseline_id


def test_replay_orders_are_permutations() -> None:
    cases = generate_universe()
    expected = {case.case_id for case in cases}
    for replay_id in REPLAY_IDS:
        observed = [case.case_id for case in order_cases(cases, replay_id)]
        assert len(observed) == 92
        assert set(observed) == expected


def test_replay_refuses_artifact_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "e2"
    run_replay(output_dir, PROTOCOL_PATH, "R0")
    try:
        run_replay(output_dir, PROTOCOL_PATH, "R0")
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing replay directory must never be overwritten")


def test_five_clean_replay_artifacts_summarize(tmp_path: Path) -> None:
    output_dir = tmp_path / "e2"
    for replay_id in REPLAY_IDS:
        metadata = run_replay(output_dir, PROTOCOL_PATH, replay_id)
        assert metadata["case_count"] == 92
        assert metadata["verdict_row_count"] == 552
    result = summarize_v1(output_dir, PROTOCOL_PATH)

    assert result["status"] == "completed"
    assert result["outcome"] == "supported"
    assert result["valid_accepted"] == 4
    assert result["invalid_rejected"] == 88
    assert result["replay_disagreement_count"] == 0
    assert all(result["threshold_checks"].values())
    assert json.loads((output_dir / "result.json").read_text())["outcome"] == "supported"
