"""Deterministic verification for typed diagnostic claim--evidence graphs.

The verifier is deliberately independent of model generation.  It checks a
structured report against a frozen graph and declared allow-listed rules; it
does not infer whether the upstream diagnostic evidence is physically true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


NODE_TYPES = {"source", "test", "result", "evidence", "claim"}
CLAIM_TYPES = {
    "numeric",
    "diagnosis",
    "causal",
    "recommendation",
    "uncertainty",
    "observation",
}
STRUCTURAL_RELATIONS = {
    "derived_from": ("evidence", "result"),
    "produced_by": ("result", "test"),
    "recorded_in": ("result", "source"),
}
SEMANTIC_RELATIONS = {
    "supports": ("evidence", "claim"),
    "contradicts": ("evidence", "claim"),
}
RULE_OPERATORS = {"eq", "neq", "gt", "ge", "lt", "le", "in", "contains", "exists"}
VERDICTS = {"supported", "contradicted", "insufficient"}

_MISSING = object()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible input canonically for deterministic hashing."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _require_sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{location} must be a sequence")
    return value


def _require_identifier(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _lookup(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_rule(rule: Mapping[str, Any], location: str = "rule") -> None:
    path = rule.get("path")
    operator = rule.get("operator")
    if not isinstance(path, str) or not path.strip() or path.startswith(".") or path.endswith("."):
        raise ValueError(f"{location}.path must be a dot-separated payload path")
    if operator not in RULE_OPERATORS:
        raise ValueError(
            f"{location}.operator must be one of {sorted(RULE_OPERATORS)}, got {operator!r}"
        )
    if operator != "exists" and "expected" not in rule:
        raise ValueError(f"{location}.expected is required for operator {operator!r}")
    if operator == "exists" and "expected" in rule and not isinstance(rule["expected"], bool):
        raise ValueError(f"{location}.expected must be boolean for operator 'exists'")
    if operator == "in" and (
        "expected" not in rule
        or not isinstance(rule["expected"], (list, tuple, set))
    ):
        raise ValueError(f"{location}.expected must be a collection for operator 'in'")
    canonical_json_bytes(dict(rule))


def evaluate_rule(payload: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    """Evaluate one allow-listed rule without interpreting executable input."""

    _require_mapping(payload, "payload")
    _require_mapping(rule, "rule")
    _validate_rule(rule)
    actual = _lookup(payload, str(rule["path"]))
    operator = str(rule["operator"])

    if operator == "exists":
        expected = bool(rule.get("expected", True))
        return (actual is not _MISSING) is expected
    if actual is _MISSING:
        return False

    expected = rule["expected"]
    if operator == "eq":
        return actual == expected
    if operator == "neq":
        return actual != expected
    if operator in {"gt", "ge", "lt", "le"}:
        if not _finite_number(actual) or not _finite_number(expected):
            return False
        if operator == "gt":
            return float(actual) > float(expected)
        if operator == "ge":
            return float(actual) >= float(expected)
        if operator == "lt":
            return float(actual) < float(expected)
        return float(actual) <= float(expected)
    if operator == "in":
        return actual in expected
    if operator == "contains":
        if isinstance(actual, Mapping):
            return expected in actual
        if isinstance(actual, (str, list, tuple, set)):
            return expected in actual
        return False
    raise AssertionError(f"unhandled rule operator: {operator}")


def validate_graph(graph: Mapping[str, Any]) -> None:
    """Validate graph schema, endpoint types, and rule safety."""

    graph = _require_mapping(graph, "graph")
    nodes_raw = _require_sequence(graph.get("nodes"), "graph.nodes")
    relations_raw = _require_sequence(graph.get("relations"), "graph.relations")
    if not nodes_raw:
        raise ValueError("graph.nodes must not be empty")

    nodes: dict[str, Mapping[str, Any]] = {}
    for index, raw_node in enumerate(nodes_raw):
        node = _require_mapping(raw_node, f"graph.nodes[{index}]")
        node_id = _require_identifier(node.get("id"), f"graph.nodes[{index}].id")
        if node_id in nodes:
            raise ValueError(f"duplicate node id: {node_id}")
        node_type = node.get("type")
        if node_type not in NODE_TYPES:
            raise ValueError(f"node {node_id} has unsupported type {node_type!r}")
        payload = _require_mapping(node.get("payload", {}), f"node {node_id}.payload")
        canonical_json_bytes(dict(payload))
        if node_type == "claim":
            claim_type = payload.get("claim_type")
            if claim_type not in CLAIM_TYPES:
                raise ValueError(
                    f"claim {node_id} claim_type must be one of {sorted(CLAIM_TYPES)}"
                )
            assertion = _require_mapping(
                payload.get("assertion"), f"claim {node_id}.payload.assertion"
            )
            canonical_json_bytes(dict(assertion))
        nodes[node_id] = node

    relation_ids: set[str] = set()
    for index, raw_relation in enumerate(relations_raw):
        relation = _require_mapping(raw_relation, f"graph.relations[{index}]")
        relation_id = _require_identifier(
            relation.get("id"), f"graph.relations[{index}].id"
        )
        if relation_id in relation_ids:
            raise ValueError(f"duplicate relation id: {relation_id}")
        relation_ids.add(relation_id)
        relation_type = relation.get("type")
        expected_types = STRUCTURAL_RELATIONS.get(str(relation_type))
        if expected_types is None:
            expected_types = SEMANTIC_RELATIONS.get(str(relation_type))
        if expected_types is None:
            raise ValueError(
                f"relation {relation_id} has unsupported type {relation_type!r}"
            )
        source_id = _require_identifier(
            relation.get("from"), f"relation {relation_id}.from"
        )
        target_id = _require_identifier(
            relation.get("to"), f"relation {relation_id}.to"
        )
        if source_id not in nodes or target_id not in nodes:
            raise ValueError(
                f"relation {relation_id} references unknown endpoint "
                f"{source_id!r}->{target_id!r}"
            )
        actual_types = (str(nodes[source_id]["type"]), str(nodes[target_id]["type"]))
        if actual_types != expected_types:
            raise ValueError(
                f"relation {relation_id} requires endpoint types {expected_types}, "
                f"got {actual_types}"
            )
        if relation_type in SEMANTIC_RELATIONS:
            rule = _require_mapping(
                relation.get("rule"), f"relation {relation_id}.rule"
            )
            _validate_rule(rule, f"relation {relation_id}.rule")

    canonical_json_bytes(dict(graph))


def _index_graph(
    graph: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    nodes = {str(node["id"]): node for node in graph["nodes"]}
    relations = [relation for relation in graph["relations"]]
    return nodes, relations


def complete_evidence_ids(graph: Mapping[str, Any]) -> set[str]:
    """Return evidence nodes with a complete result/test/source provenance path."""

    validate_graph(graph)
    nodes, relations = _index_graph(graph)
    evidence_to_results: dict[str, set[str]] = defaultdict(set)
    result_to_tests: dict[str, set[str]] = defaultdict(set)
    result_to_sources: dict[str, set[str]] = defaultdict(set)

    for relation in relations:
        relation_type = relation["type"]
        if relation_type == "derived_from":
            evidence_to_results[str(relation["from"])].add(str(relation["to"]))
        elif relation_type == "produced_by":
            result_to_tests[str(relation["from"])].add(str(relation["to"]))
        elif relation_type == "recorded_in":
            result_to_sources[str(relation["from"])].add(str(relation["to"]))

    complete: set[str] = set()
    for evidence_id, result_ids in evidence_to_results.items():
        if nodes[evidence_id]["type"] != "evidence":
            continue
        if any(result_to_tests[result_id] and result_to_sources[result_id] for result_id in result_ids):
            complete.add(evidence_id)
    return complete


def _active_semantic_evidence(
    graph: Mapping[str, Any],
) -> dict[str, dict[str, set[str]]]:
    nodes, relations = _index_graph(graph)
    complete = complete_evidence_ids(graph)
    active: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"supports": set(), "contradicts": set()}
    )
    for relation in relations:
        relation_type = str(relation["type"])
        if relation_type not in SEMANTIC_RELATIONS:
            continue
        evidence_id = str(relation["from"])
        claim_id = str(relation["to"])
        if evidence_id not in complete:
            continue
        evidence_payload = _require_mapping(
            nodes[evidence_id].get("payload", {}), f"evidence {evidence_id}.payload"
        )
        if evaluate_rule(evidence_payload, relation["rule"]):
            active[claim_id][relation_type].add(evidence_id)
    return active


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _validate_report(report: Mapping[str, Any]) -> None:
    report = _require_mapping(report, "report")
    claims = _require_sequence(report.get("claims", []), "report.claims")
    abstentions = _require_sequence(report.get("abstentions", []), "report.abstentions")
    emitted_ids: set[str] = set()
    for index, raw_claim in enumerate(claims):
        claim = _require_mapping(raw_claim, f"report.claims[{index}]")
        claim_id = _require_identifier(
            claim.get("claim_id"), f"report.claims[{index}].claim_id"
        )
        if claim_id in emitted_ids:
            raise ValueError(f"duplicate emitted claim id: {claim_id}")
        emitted_ids.add(claim_id)
        assertion = _require_mapping(
            claim.get("assertion"), f"report claim {claim_id}.assertion"
        )
        canonical_json_bytes(dict(assertion))
        evidence_ids = _require_sequence(
            claim.get("evidence_ids", []), f"report claim {claim_id}.evidence_ids"
        )
        normalized_evidence_ids = [
            _require_identifier(value, f"report claim {claim_id}.evidence_ids")
            for value in evidence_ids
        ]
        if len(normalized_evidence_ids) != len(set(normalized_evidence_ids)):
            raise ValueError(f"report claim {claim_id} has duplicate evidence ids")

    abstention_ids: set[str] = set()
    for index, raw_abstention in enumerate(abstentions):
        abstention = _require_mapping(raw_abstention, f"report.abstentions[{index}]")
        claim_id = _require_identifier(
            abstention.get("claim_id"), f"report.abstentions[{index}].claim_id"
        )
        if claim_id in abstention_ids:
            raise ValueError(f"duplicate abstention claim id: {claim_id}")
        if claim_id in emitted_ids:
            raise ValueError(f"claim {claim_id} is both emitted and abstained")
        abstention_ids.add(claim_id)
    canonical_json_bytes(dict(report))


def verify_report(
    graph: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return deterministic verdicts and guardrail metrics for one report."""

    validate_graph(graph)
    _validate_report(report)
    nodes, _ = _index_graph(graph)
    complete = complete_evidence_ids(graph)
    active = _active_semantic_evidence(graph)
    claim_nodes = {
        node_id: node for node_id, node in nodes.items() if node["type"] == "claim"
    }
    evidence_nodes = {
        node_id for node_id, node in nodes.items() if node["type"] == "evidence"
    }

    verdicts: list[dict[str, Any]] = []
    known_emitted: set[str] = set()
    for raw_claim in report.get("claims", []):
        claim = _require_mapping(raw_claim, "report claim")
        claim_id = str(claim["claim_id"])
        cited = {str(value) for value in claim.get("evidence_ids", [])}
        base = {
            "claim_id": claim_id,
            "cited_evidence_ids": sorted(cited),
            "active_support_ids": [],
            "active_contradiction_ids": [],
        }
        if claim_id not in claim_nodes:
            verdicts.append({**base, "verdict": "insufficient", "reason": "unknown_claim"})
            continue

        known_emitted.add(claim_id)
        canonical_assertion = _require_mapping(
            claim_nodes[claim_id]["payload"]["assertion"],
            f"claim {claim_id}.assertion",
        )
        if canonical_json_bytes(dict(claim["assertion"])) != canonical_json_bytes(
            dict(canonical_assertion)
        ):
            verdicts.append(
                {**base, "verdict": "insufficient", "reason": "assertion_mismatch"}
            )
            continue

        support = set(active[claim_id]["supports"])
        contradiction = set(active[claim_id]["contradicts"])
        base["active_support_ids"] = sorted(support)
        base["active_contradiction_ids"] = sorted(contradiction)
        invalid_citations = sorted(
            evidence_id
            for evidence_id in cited
            if evidence_id not in evidence_nodes or evidence_id not in complete
        )
        unrelated_citations = sorted(cited - support - contradiction)
        if invalid_citations:
            verdicts.append(
                {
                    **base,
                    "verdict": "insufficient",
                    "reason": "invalid_citation",
                    "invalid_citation_ids": invalid_citations,
                }
            )
        elif unrelated_citations:
            verdicts.append(
                {
                    **base,
                    "verdict": "insufficient",
                    "reason": "unrelated_citation",
                    "unrelated_citation_ids": unrelated_citations,
                }
            )
        elif support and contradiction:
            verdicts.append({**base, "verdict": "insufficient", "reason": "conflict"})
        elif contradiction:
            verdicts.append({**base, "verdict": "contradicted", "reason": "active_contradiction"})
        elif support and cited.intersection(support):
            verdicts.append({**base, "verdict": "supported", "reason": "active_cited_support"})
        elif support:
            verdicts.append({**base, "verdict": "insufficient", "reason": "support_not_cited"})
        else:
            verdicts.append({**base, "verdict": "insufficient", "reason": "no_active_evidence"})

    if any(verdict["verdict"] not in VERDICTS for verdict in verdicts):
        raise AssertionError("internal verifier produced an invalid verdict")

    expected_abstentions = {
        claim_id
        for claim_id in claim_nodes
        if bool(active[claim_id]["supports"]) == bool(active[claim_id]["contradicts"])
    }
    abstention_ids = {
        str(item["claim_id"])
        for item in report.get("abstentions", [])
        if isinstance(item, Mapping)
    }
    correct_abstentions = abstention_ids.intersection(expected_abstentions)
    incorrect_abstentions = abstention_ids - expected_abstentions

    counts = Counter(str(verdict["verdict"]) for verdict in verdicts)
    emitted_count = len(verdicts)
    unsupported_count = emitted_count - counts["supported"]
    metrics = {
        "emitted_claims": emitted_count,
        "canonical_claim_opportunities": len(claim_nodes),
        "supported_claims": counts["supported"],
        "contradicted_claims": counts["contradicted"],
        "insufficient_claims": counts["insufficient"],
        "unsupported_claim_rate": _safe_rate(unsupported_count, emitted_count),
        "contradiction_rate": _safe_rate(counts["contradicted"], emitted_count),
        "evidence_consistency": _safe_rate(counts["supported"], emitted_count),
        "coverage": _safe_rate(len(known_emitted), len(claim_nodes)),
        "selective_risk": _safe_rate(unsupported_count, emitted_count),
        "abstentions": len(abstention_ids),
        "expected_abstentions": len(expected_abstentions),
        "correct_abstentions": len(correct_abstentions),
        "abstention_precision": _safe_rate(len(correct_abstentions), len(abstention_ids)),
        "abstention_recall": _safe_rate(len(correct_abstentions), len(expected_abstentions)),
    }

    policy = {
        "conflict": "insufficient",
        "unknown_claim": "insufficient",
        "assertion_matching": "canonical_exact",
        "contradictions_considered_when_uncited": True,
        "support_requires_citation": True,
    }
    verification_id = sha256_json(
        {"graph": graph, "report": report, "policy": policy}
    )
    return {
        "schema_version": 1,
        "verification_id": verification_id,
        "graph_sha256": sha256_json(graph),
        "report_sha256": sha256_json(report),
        "policy": policy,
        "verdicts": verdicts,
        "abstention_audit": {
            "expected_claim_ids": sorted(expected_abstentions),
            "correct_claim_ids": sorted(correct_abstentions),
            "incorrect_claim_ids": sorted(incorrect_abstentions),
        },
        "metrics": metrics,
        "failures": [
            {
                "claim_id": verdict["claim_id"],
                "verdict": verdict["verdict"],
                "reason": verdict["reason"],
            }
            for verdict in verdicts
            if verdict["verdict"] != "supported"
        ],
    }


def _nested_value(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _assert_expected(result: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for path, expected_value in expected.items():
        actual = _nested_value(result, str(path))
        if actual is _MISSING or actual != expected_value:
            raise ValueError(
                f"smoke expectation failed for {path}: expected {expected_value!r}, "
                f"got {None if actual is _MISSING else actual!r}"
            )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> tuple[str, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    digest = sha256_file(path)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    digest_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest, digest_path


def run_config(config_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config = _require_mapping(config, "config")
    graph = _require_mapping(config.get("graph"), "config.graph")
    report = _require_mapping(config.get("report"), "config.report")

    started = time.perf_counter()
    result = verify_report(graph, report)
    runtime_ms = (time.perf_counter() - started) * 1000.0
    expected = config.get("expected", {})
    if expected:
        _assert_expected(result, _require_mapping(expected, "config.expected"))

    target_value = output_path if output_path is not None else config.get("output_path")
    if not target_value:
        raise ValueError("an output path is required by --output or config.output_path")
    target = Path(str(target_value))
    artifact = {
        "schema_version": 1,
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "runtime_ms": runtime_ms,
        "deterministic_result": result,
    }
    digest, digest_path = _atomic_write_json(target, artifact)
    return {
        "output_path": str(target),
        "output_sha256": digest,
        "output_sha256_path": str(digest_path),
        "verification_id": result["verification_id"],
        "metrics": result["metrics"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a structured diagnostic report against a typed evidence graph."
    )
    parser.add_argument("--config", required=True, help="YAML graph/report configuration")
    parser.add_argument("--output", help="JSON artifact path; overrides config.output_path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_config(args.config, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
