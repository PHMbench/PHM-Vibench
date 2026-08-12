from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from src.utils.claim_evidence_verifier import (
    evaluate_rule,
    run_config,
    validate_graph,
    verify_report,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "p03"
    / "verifiable_report_smoke.yaml"
)


def _smoke_inputs() -> tuple[dict, dict]:
    config = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    return copy.deepcopy(config["graph"]), copy.deepcopy(config["report"])


def _verdict(result: dict, claim_id: str) -> dict:
    return next(item for item in result["verdicts"] if item["claim_id"] == claim_id)


def test_smoke_config_is_deterministic_and_fail_closed() -> None:
    graph, report = _smoke_inputs()

    first = verify_report(graph, report)
    second = verify_report(copy.deepcopy(graph), copy.deepcopy(report))

    assert first == second
    assert first["metrics"]["supported_claims"] == 1
    assert first["metrics"]["insufficient_claims"] == 2
    assert _verdict(first, "CLAIM-DIAG")["verdict"] == "supported"
    assert _verdict(first, "CLAIM-CAUSE")["reason"] == "conflict"
    assert _verdict(first, "CLAIM-UNKNOWN")["reason"] == "unknown_claim"


def test_one_sided_active_contradiction_is_reported() -> None:
    graph, report = _smoke_inputs()
    graph["relations"] = [
        relation for relation in graph["relations"] if relation["id"] != "REL-11"
    ]
    claim = next(item for item in report["claims"] if item["claim_id"] == "CLAIM-CAUSE")
    claim["evidence_ids"] = ["EVID-03"]

    result = verify_report(graph, report)

    assert _verdict(result, "CLAIM-CAUSE") == {
        "claim_id": "CLAIM-CAUSE",
        "cited_evidence_ids": ["EVID-03"],
        "active_support_ids": [],
        "active_contradiction_ids": ["EVID-03"],
        "verdict": "contradicted",
        "reason": "active_contradiction",
    }


def test_uncited_contradiction_still_forces_conflict() -> None:
    graph, report = _smoke_inputs()
    claim = next(item for item in report["claims"] if item["claim_id"] == "CLAIM-CAUSE")
    assert claim["evidence_ids"] == ["EVID-02"]

    result = verify_report(graph, report)

    assert _verdict(result, "CLAIM-CAUSE")["reason"] == "conflict"
    assert _verdict(result, "CLAIM-CAUSE")["active_contradiction_ids"] == ["EVID-03"]


def test_missing_provenance_makes_evidence_ineligible() -> None:
    graph, report = _smoke_inputs()
    graph["relations"] = [
        relation for relation in graph["relations"] if relation["id"] != "REL-03"
    ]

    result = verify_report(graph, report)

    diagnosis = _verdict(result, "CLAIM-DIAG")
    assert diagnosis["verdict"] == "insufficient"
    assert diagnosis["reason"] == "invalid_citation"
    assert diagnosis["invalid_citation_ids"] == ["EVID-01"]


def test_assertion_payload_mismatch_fails_closed() -> None:
    graph, report = _smoke_inputs()
    claim = next(item for item in report["claims"] if item["claim_id"] == "CLAIM-DIAG")
    claim["assertion"]["object"] = "outer_race"

    result = verify_report(graph, report)

    assert _verdict(result, "CLAIM-DIAG")["reason"] == "assertion_mismatch"


def test_rule_language_rejects_executable_or_unknown_operators() -> None:
    with pytest.raises(ValueError, match="operator must be one of"):
        evaluate_rule(
            {"finding": {"value": 1}},
            {
                "path": "finding.value",
                "operator": "__import__('os').system",
                "expected": 1,
            },
        )


def test_graph_rejects_duplicate_node_ids() -> None:
    graph, _ = _smoke_inputs()
    graph["nodes"].append(copy.deepcopy(graph["nodes"][0]))

    with pytest.raises(ValueError, match="duplicate node id"):
        validate_graph(graph)


def test_empty_report_is_not_scored_as_perfect() -> None:
    graph, _ = _smoke_inputs()
    report = {"schema_version": 1, "report_id": "empty", "claims": [], "abstentions": []}

    result = verify_report(graph, report)

    assert result["metrics"]["unsupported_claim_rate"] is None
    assert result["metrics"]["selective_risk"] is None
    assert result["metrics"]["coverage"] == 0.0


def test_cli_config_writes_hashed_artifact(tmp_path: Path) -> None:
    output = tmp_path / "verification.json"

    summary = run_config(SMOKE_CONFIG, output)

    assert output.is_file()
    digest_path = output.with_suffix(".json.sha256")
    assert digest_path.is_file()
    expected_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert summary["output_sha256"] == expected_digest
    assert digest_path.read_text(encoding="utf-8").split()[0] == expected_digest
    assert summary["metrics"]["expected_abstentions"] == 2
