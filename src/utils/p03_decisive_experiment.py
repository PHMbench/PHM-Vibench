"""Run P03's smallest fair controlled-fixture mechanism experiment.

This runner intentionally does not call an LLM.  It sends byte-identical
structured candidate reports to a no-verifier baseline, the fail-closed typed
verifier, and a fail-open negative control.  A manifest-derived oracle, rather
than verifier verdicts, scores all three outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.utils.claim_evidence_verifier import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    verify_report,
)


CLAIM_TYPES = ("numeric", "diagnosis", "causal", "recommendation", "uncertainty")
REGIMES = (
    "complete",
    "missing",
    "noisy",
    "conflicting",
    "fluent_but_unsupported",
)
CONDITIONS = (
    "structured_no_verifier",
    "typed_verifier",
    "fail_open_negative_control",
)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{location} must be a sequence")
    return value


def _affected_slots(case_index: int) -> set[int]:
    return {case_index % len(CLAIM_TYPES), (case_index + 2) % len(CLAIM_TYPES)}


def _claim_value(seed: int, case_index: int, slot_index: int) -> Any:
    if CLAIM_TYPES[slot_index] == "numeric":
        return round(1.0 + (seed % 17) * 0.01 + case_index * 0.1, 6)
    return f"{CLAIM_TYPES[slot_index]}-value-{(seed + case_index + slot_index) % 11}"


def build_fixture(seed: int, case_index: int, regime: str) -> dict[str, Any]:
    """Build one graph, shared candidate report, and independent oracle manifest."""

    if regime not in REGIMES:
        raise ValueError(f"unknown evidence regime: {regime}")
    affected = _affected_slots(case_index)
    nodes: list[dict[str, Any]] = [
        {
            "id": "SRC-OBSERVED",
            "type": "source",
            "payload": {
                "source_kind": "controlled_fixture",
                "seed": seed,
                "case_index": case_index,
            },
        },
        {
            "id": "TEST-FIXTURE",
            "type": "test",
            "payload": {"name": "p03_controlled_fixture", "version": 1},
        },
    ]
    relations: list[dict[str, Any]] = []
    report_claims: list[dict[str, Any]] = []
    oracle_slots: dict[str, dict[str, Any]] = {}
    relation_index = 0

    def add_relation(relation_type: str, source: str, target: str, **extra: Any) -> None:
        nonlocal relation_index
        relation_index += 1
        relation = {
            "id": f"REL-{relation_index:03d}",
            "type": relation_type,
            "from": source,
            "to": target,
        }
        relation.update(extra)
        relations.append(relation)

    for slot_index, claim_type in enumerate(CLAIM_TYPES):
        suffix = f"{slot_index + 1:02d}"
        claim_id = f"CLAIM-{claim_type.upper()}"
        result_id = f"RESULT-{suffix}"
        evidence_id = f"EVID-{suffix}"
        value = _claim_value(seed, case_index, slot_index)
        assertion = {
            "subject": "controlled-bearing-case",
            "predicate": f"{claim_type}_value",
            "value": value,
        }
        evidence_payload: dict[str, Any] = {"finding": {"value": value}}
        if regime == "fluent_but_unsupported" and slot_index in affected:
            evidence_payload["narrative"] = (
                f"The {claim_type} statement sounds technically plausible but "
                "has no complete source-test-result chain."
            )

        nodes.extend(
            [
                {"id": result_id, "type": "result", "payload": {"slot": claim_type}},
                {"id": evidence_id, "type": "evidence", "payload": evidence_payload},
                {
                    "id": claim_id,
                    "type": "claim",
                    "payload": {"claim_type": claim_type, "assertion": assertion},
                },
            ]
        )
        add_relation("derived_from", evidence_id, result_id)
        add_relation("produced_by", result_id, "TEST-FIXTURE")
        if not (
            slot_index in affected
            and regime in {"missing", "fluent_but_unsupported"}
        ):
            add_relation("recorded_in", result_id, "SRC-OBSERVED")
        add_relation(
            "supports",
            evidence_id,
            claim_id,
            rule={"path": "finding.value", "operator": "eq", "expected": value},
        )

        if regime == "conflicting" and slot_index in affected:
            conflict_result_id = f"RESULT-CONFLICT-{suffix}"
            conflict_evidence_id = f"EVID-CONFLICT-{suffix}"
            conflict_value = f"conflict-{value}"
            nodes.extend(
                [
                    {
                        "id": conflict_result_id,
                        "type": "result",
                        "payload": {"slot": claim_type, "conflict": True},
                    },
                    {
                        "id": conflict_evidence_id,
                        "type": "evidence",
                        "payload": {"finding": {"value": conflict_value}},
                    },
                ]
            )
            add_relation("derived_from", conflict_evidence_id, conflict_result_id)
            add_relation("produced_by", conflict_result_id, "TEST-FIXTURE")
            add_relation("recorded_in", conflict_result_id, "SRC-OBSERVED")
            add_relation(
                "contradicts",
                conflict_evidence_id,
                claim_id,
                rule={"path": "finding.value", "operator": "neq", "expected": value},
            )

        report_claims.append(
            {
                "claim_id": claim_id,
                "assertion": assertion,
                "evidence_ids": [evidence_id],
            }
        )
        requires_abstention = slot_index in affected and regime in {
            "missing",
            "conflicting",
            "fluent_but_unsupported",
        }
        oracle_slots[claim_id] = {
            "claim_type": claim_type,
            "requires_abstention": requires_abstention,
            "unsupported_if_emitted": requires_abstention,
            "contradicted_if_emitted": (
                slot_index in affected and regime == "conflicting"
            ),
        }

    if regime == "noisy":
        for distractor_index in range(3):
            suffix = f"NOISE-{distractor_index + 1:02d}"
            result_id = f"RESULT-{suffix}"
            evidence_id = f"EVID-{suffix}"
            nodes.extend(
                [
                    {"id": result_id, "type": "result", "payload": {"noise": True}},
                    {
                        "id": evidence_id,
                        "type": "evidence",
                        "payload": {
                            "finding": {"value": f"irrelevant-{seed}-{case_index}-{distractor_index}"},
                            "narrative": "Fluent but non-target diagnostic context.",
                        },
                    },
                ]
            )
            add_relation("derived_from", evidence_id, result_id)
            add_relation("produced_by", result_id, "TEST-FIXTURE")
            if distractor_index < 2:
                add_relation("recorded_in", result_id, "SRC-OBSERVED")

    graph = {
        "schema_version": 1,
        "graph_id": f"g050-seed-{seed}-case-{case_index}-{regime}",
        "nodes": nodes,
        "relations": relations,
    }
    candidate_report = {
        "schema_version": 1,
        "report_id": f"candidate-seed-{seed}-case-{case_index}-{regime}",
        "claims": report_claims,
        "abstentions": [],
    }
    oracle = {
        "schema_version": 1,
        "case_id": f"seed-{seed}-case-{case_index}",
        "regime": regime,
        "slots": oracle_slots,
    }
    return {"graph": graph, "candidate_report": candidate_report, "oracle": oracle}


def enforce_verdicts(candidate_report: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep supported claims and explicitly abstain from every failed claim."""

    verdict_by_id = {
        str(item["claim_id"]): str(item["verdict"])
        for item in _sequence(result.get("verdicts"), "result.verdicts")
        if isinstance(item, Mapping)
    }
    kept: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    for raw_claim in _sequence(candidate_report.get("claims"), "candidate_report.claims"):
        claim = dict(_mapping(raw_claim, "candidate claim"))
        claim_id = str(claim["claim_id"])
        verdict = verdict_by_id.get(claim_id, "insufficient")
        if verdict == "supported":
            kept.append(claim)
        else:
            abstentions.append(
                {"claim_id": claim_id, "reason": f"verifier_{verdict}"}
            )
    return {
        "schema_version": 1,
        "report_id": f"{candidate_report.get('report_id')}-verified",
        "claims": kept,
        "abstentions": abstentions,
    }


def score_with_oracle(report: Mapping[str, Any], oracle: Mapping[str, Any]) -> dict[str, int]:
    """Score a report without reading any verifier verdict."""

    slots = _mapping(oracle.get("slots"), "oracle.slots")
    emitted = {
        str(item["claim_id"])
        for item in _sequence(report.get("claims", []), "report.claims")
        if isinstance(item, Mapping) and "claim_id" in item
    }
    abstained = {
        str(item["claim_id"])
        for item in _sequence(report.get("abstentions", []), "report.abstentions")
        if isinstance(item, Mapping) and "claim_id" in item
    }
    if emitted & abstained:
        raise ValueError("a claim cannot be both emitted and abstained")
    if (emitted | abstained) - set(slots):
        raise ValueError("report contains a claim absent from the independent oracle")

    counts = {
        "opportunities": len(slots),
        "emitted": 0,
        "supported_emitted": 0,
        "unsupported_emitted": 0,
        "contradicted_emitted": 0,
        "error_emitted": 0,
        "abstentions": 0,
        "required_abstentions": 0,
        "correct_abstentions": 0,
        "schema_failures": 0,
    }
    for claim_id, raw_state in slots.items():
        state = _mapping(raw_state, f"oracle slot {claim_id}")
        requires_abstention = bool(state["requires_abstention"])
        counts["required_abstentions"] += int(requires_abstention)
        if claim_id in emitted:
            counts["emitted"] += 1
            unsupported = bool(state["unsupported_if_emitted"])
            contradicted = bool(state["contradicted_if_emitted"])
            counts["unsupported_emitted"] += int(unsupported)
            counts["contradicted_emitted"] += int(contradicted)
            counts["error_emitted"] += int(unsupported or contradicted)
            counts["supported_emitted"] += int(not unsupported and not contradicted)
        elif claim_id in abstained:
            counts["abstentions"] += 1
            counts["correct_abstentions"] += int(requires_abstention)
        else:
            counts["schema_failures"] += 1
            counts["unsupported_emitted"] += 1
            counts["error_emitted"] += 1
    return counts


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summarize_counts(counts: Mapping[str, int]) -> dict[str, Any]:
    opportunities = int(counts["opportunities"])
    emitted = int(counts["emitted"])
    abstentions = int(counts["abstentions"])
    required = int(counts["required_abstentions"])
    return {
        **{key: int(value) for key, value in counts.items()},
        "unsupported_claim_rate": _rate(int(counts["unsupported_emitted"]), opportunities),
        "contradiction_rate": _rate(int(counts["contradicted_emitted"]), opportunities),
        "coverage": _rate(emitted, opportunities),
        "evidence_consistency": _rate(int(counts["supported_emitted"]), emitted),
        "selective_risk": _rate(int(counts["error_emitted"]), emitted),
        "abstention_precision": _rate(int(counts["correct_abstentions"]), abstentions),
        "abstention_recall": _rate(int(counts["correct_abstentions"]), required),
        "schema_failure_rate": _rate(int(counts["schema_failures"]), opportunities),
    }


def _add_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _git_state(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
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
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def _validate_config(config: Mapping[str, Any]) -> None:
    design = _mapping(config.get("design"), "config.design")
    seeds = [int(value) for value in _sequence(design.get("seeds"), "design.seeds")]
    if len(seeds) < 5 or len(seeds) != len(set(seeds)):
        raise ValueError("design requires at least five unique seeds")
    if tuple(design.get("evidence_regimes", [])) != REGIMES:
        raise ValueError("evidence regimes differ from the frozen G050 contract")
    if tuple(design.get("claim_types", [])) != CLAIM_TYPES:
        raise ValueError("claim types differ from the frozen G050 contract")
    if tuple(design.get("conditions", [])) != CONDITIONS:
        raise ValueError("conditions differ from the frozen G050 contract")
    execution = _mapping(config.get("execution"), "config.execution")
    if execution.get("conda_environment") != "LQ_signal":
        raise ValueError("G050 evidence must use conda environment LQ_signal")
    if execution.get("physical_gpu_indices") != [] or execution.get("multi_gpu") is not False:
        raise ValueError("this controlled G050 experiment must run on CPU with multi_gpu false")
    if not str(execution.get("command", "")).startswith("conda run -n LQ_signal"):
        raise ValueError("recorded command lacks the required conda prefix")


def run_experiment(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = _mapping(yaml.safe_load(config_file.read_text(encoding="utf-8")), "config")
    _validate_config(config)
    design = _mapping(config["design"], "design")
    seeds = [int(value) for value in design["seeds"]]
    cases_per_seed = int(design["base_cases_per_seed"])
    if cases_per_seed != 5:
        raise ValueError("the frozen controlled design requires five base cases per seed")

    count_keys = (
        "opportunities",
        "emitted",
        "supported_emitted",
        "unsupported_emitted",
        "contradicted_emitted",
        "error_emitted",
        "abstentions",
        "required_abstentions",
        "correct_abstentions",
        "schema_failures",
    )
    totals = {condition: {key: 0 for key in count_keys} for condition in CONDITIONS}
    by_seed = {
        seed: {condition: {key: 0 for key in count_keys} for condition in CONDITIONS}
        for seed in seeds
    }
    by_regime = {
        regime: {condition: {key: 0 for key in count_keys} for condition in CONDITIONS}
        for regime in REGIMES
    }
    verification_latencies_ms: list[float] = []
    candidate_hashes: list[str] = []
    negative_control_matches = True
    block_count = 0

    started = time.perf_counter()
    for seed in seeds:
        for case_index in range(cases_per_seed):
            for regime in REGIMES:
                fixture = build_fixture(seed, case_index, regime)
                graph = _mapping(fixture["graph"], "fixture.graph")
                candidate = _mapping(fixture["candidate_report"], "fixture.candidate")
                oracle = _mapping(fixture["oracle"], "fixture.oracle")
                candidate_hash = sha256_json(candidate)
                candidate_hashes.append(candidate_hash)

                verify_started = time.perf_counter()
                verification = verify_report(graph, candidate)
                verification_latencies_ms.append(
                    (time.perf_counter() - verify_started) * 1000.0
                )
                outputs = {
                    "structured_no_verifier": copy.deepcopy(candidate),
                    "typed_verifier": enforce_verdicts(candidate, verification),
                    "fail_open_negative_control": copy.deepcopy(candidate),
                }
                negative_control_matches &= (
                    canonical_json_bytes(outputs["structured_no_verifier"])
                    == canonical_json_bytes(outputs["fail_open_negative_control"])
                )
                for condition, report in outputs.items():
                    counts = score_with_oracle(report, oracle)
                    _add_counts(totals[condition], counts)
                    _add_counts(by_seed[seed][condition], counts)
                    _add_counts(by_regime[regime][condition], counts)
                block_count += 1

    elapsed_seconds = time.perf_counter() - started
    aggregate_metrics = {
        condition: summarize_counts(counts) for condition, counts in totals.items()
    }
    per_seed_metrics = {
        str(seed): {
            condition: summarize_counts(counts)
            for condition, counts in condition_counts.items()
        }
        for seed, condition_counts in by_seed.items()
    }
    per_regime_metrics = {
        regime: {
            condition: summarize_counts(counts)
            for condition, counts in condition_counts.items()
        }
        for regime, condition_counts in by_regime.items()
    }

    expected = _mapping(config.get("expected_contract_signal"), "expected_contract_signal")
    for condition in CONDITIONS:
        expected_metrics = _mapping(expected.get(condition), f"expected {condition}")
        for metric_name, expected_value in expected_metrics.items():
            actual = aggregate_metrics[condition].get(metric_name)
            if actual != expected_value:
                raise ValueError(
                    f"metric contract mismatch for {condition}.{metric_name}: "
                    f"expected {expected_value!r}, got {actual!r}"
                )
    if negative_control_matches is not bool(expected["negative_control_matches_baseline"]):
        raise ValueError("negative-control identity contract failed")

    typed = aggregate_metrics["typed_verifier"]
    baseline = aggregate_metrics["structured_no_verifier"]
    typed_reduces_both = (
        float(typed["unsupported_claim_rate"]) < float(baseline["unsupported_claim_rate"])
        and float(typed["contradiction_rate"]) < float(baseline["contradiction_rate"])
    )
    guardrails = _mapping(_mapping(config["metric"], "metric")["guardrails"], "guardrails")
    coverage_ok = all(
        float(per_regime_metrics[regime]["typed_verifier"]["coverage"])
        >= float(guardrails["minimum_coverage_each_regime"])
        for regime in REGIMES
    )
    abstention_ok = (
        float(typed["abstention_precision"]) >= float(guardrails["minimum_abstention_precision"])
        and float(typed["abstention_recall"]) >= float(guardrails["minimum_abstention_recall"])
    )
    latency = {
        "count": len(verification_latencies_ms),
        "mean": statistics.fmean(verification_latencies_ms),
        "p50": _percentile(verification_latencies_ms, 50),
        "p95": _percentile(verification_latencies_ms, 95),
        "p99": _percentile(verification_latencies_ms, 99),
        "max": max(verification_latencies_ms),
    }
    latency_ok = latency["p95"] <= float(guardrails["maximum_verifier_p95_ms"])
    mechanism_signal = all(
        (typed_reduces_both, negative_control_matches, coverage_ok, abstention_ok, latency_ok)
    )

    repository_root = Path(__file__).resolve().parents[2]
    artifact = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "status": "completed",
        "scientific_scope": config["scientific_scope"],
        "evidence_tier": config["evidence_tier"],
        "accepted_for_c1": False,
        "mechanism_outcome": (
            "supported_on_controlled_fixtures" if mechanism_signal else "refuted"
        ),
        "paper_claim_outcome": "inconclusive" if mechanism_signal else "refuted",
        "boundary": (
            "No LLM renderer, real diagnostic case manifest, learned checker, or "
            "independent human audit is present."
        ),
        "execution": {
            **dict(_mapping(config["execution"], "execution")),
            "elapsed_seconds": elapsed_seconds,
            "repository_state": _git_state(repository_root),
            "config_path": str(config_file),
            "config_sha256": sha256_file(config_file),
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "seeds": seeds,
            "base_cases_per_seed": cases_per_seed,
            "regimes": list(REGIMES),
            "conditions": list(CONDITIONS),
            "matched_blocks": block_count,
            "claim_opportunities_per_condition": totals[CONDITIONS[0]]["opportunities"],
        },
        "fairness_checks": {
            "candidate_hash_count": len(candidate_hashes),
            "candidate_hashes_unique_by_block": len(set(candidate_hashes)) == len(candidate_hashes),
            "same_candidate_object_fed_to_all_conditions": True,
            "negative_control_byte_identical_to_baseline": negative_control_matches,
            "scorer_reads_verifier_verdicts": False,
        },
        "aggregate_metrics": aggregate_metrics,
        "per_seed_metrics": per_seed_metrics,
        "per_regime_metrics": per_regime_metrics,
        "verifier_latency_ms": latency,
        "decision_checks": {
            "typed_reduces_both_primary_metrics": typed_reduces_both,
            "negative_control_matches_baseline": negative_control_matches,
            "coverage_guardrail_passed": coverage_ok,
            "abstention_guardrails_passed": abstention_ok,
            "latency_guardrail_passed": latency_ok,
        },
    }
    output = Path(output_path).resolve()
    digest = _atomic_write_json(output, artifact)
    return {
        "status": artifact["status"],
        "mechanism_outcome": artifact["mechanism_outcome"],
        "paper_claim_outcome": artifact["paper_claim_outcome"],
        "output_path": str(output),
        "output_sha256": digest,
        "matched_blocks": block_count,
        "seeds": len(seeds),
        "primary_metrics": {
            condition: {
                "unsupported_claim_rate": aggregate_metrics[condition]["unsupported_claim_rate"],
                "contradiction_rate": aggregate_metrics[condition]["contradiction_rate"],
                "coverage": aggregate_metrics[condition]["coverage"],
            }
            for condition in CONDITIONS
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_experiment(args.config, args.output)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    log_path = Path(args.log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(rendered, encoding="utf-8")
    log_digest = sha256_file(log_path)
    log_path.with_suffix(log_path.suffix + ".sha256").write_text(
        f"{log_digest}  {log_path.name}\n", encoding="utf-8"
    )
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
