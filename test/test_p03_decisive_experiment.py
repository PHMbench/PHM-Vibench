from __future__ import annotations

import json
from pathlib import Path

from src.utils.claim_evidence_verifier import sha256_json, verify_report
from src.utils.p03_decisive_experiment import (
    CONDITIONS,
    REGIMES,
    build_fixture,
    enforce_verdicts,
    run_experiment,
    score_with_oracle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY_ROOT / "configs" / "experiments" / "p03" / "decisive_mechanism.yaml"


def test_all_regimes_use_valid_graphs_and_shared_candidates() -> None:
    for regime in REGIMES:
        fixture = build_fixture(seed=42, case_index=0, regime=regime)
        candidate = fixture["candidate_report"]
        before = sha256_json(candidate)

        result = verify_report(fixture["graph"], candidate)
        proposed = enforce_verdicts(candidate, result)

        assert sha256_json(candidate) == before
        assert len(candidate["claims"]) == 5
        assert len(proposed["claims"]) + len(proposed["abstentions"]) == 5


def test_independent_oracle_detects_fail_open_negative_control() -> None:
    fixture = build_fixture(seed=123, case_index=1, regime="conflicting")
    candidate = fixture["candidate_report"]
    verification = verify_report(fixture["graph"], candidate)
    proposed = enforce_verdicts(candidate, verification)

    baseline_counts = score_with_oracle(candidate, fixture["oracle"])
    negative_counts = score_with_oracle(candidate, fixture["oracle"])
    proposed_counts = score_with_oracle(proposed, fixture["oracle"])

    assert baseline_counts == negative_counts
    assert baseline_counts["unsupported_emitted"] == 2
    assert baseline_counts["contradicted_emitted"] == 2
    assert proposed_counts["unsupported_emitted"] == 0
    assert proposed_counts["contradicted_emitted"] == 0
    assert proposed_counts["correct_abstentions"] == 2


def test_full_controlled_experiment_is_hashed_and_bounded(tmp_path: Path) -> None:
    output = tmp_path / "decisive.json"

    summary = run_experiment(CONFIG, output)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert summary["status"] == "completed"
    assert summary["mechanism_outcome"] == "supported_on_controlled_fixtures"
    assert summary["paper_claim_outcome"] == "inconclusive"
    assert summary["seeds"] == 5
    assert artifact["design"]["matched_blocks"] == 125
    assert artifact["accepted_for_c1"] is False
    assert artifact["fairness_checks"]["negative_control_byte_identical_to_baseline"] is True
    assert set(artifact["aggregate_metrics"]) == set(CONDITIONS)
    assert output.with_suffix(".json.sha256").is_file()
