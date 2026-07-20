#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


START_MARKER = "<!-- AUTORESEARCH_SUBMISSION_BINDING:START -->"
END_MARKER = "<!-- AUTORESEARCH_SUBMISSION_BINDING:END -->"

REQUIRED_REVIEW_TICKETS = {
    "moe-minimal-demo-bootstrap",
    "moe-vibench-smoke-bootstrap",
    "moe-runtime-sanity",
    "moe-seed-stability",
    "moe-routing-analysis",
    "moe-stability-strategy",
    "moe-dataset-bridge",
    "moe-expert-ablation",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def latest_file(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no files matched: {pattern}")
    return candidates[-1]


def normalize_dataset_entries(values: List[Any]) -> List[str]:
    normalized: List[str] = []
    for item in values:
        if isinstance(item, dict):
            label = item.get("dataset_name") or item.get("dataset") or item.get("dataset_id")
        else:
            label = item
        if label is None:
            continue
        normalized.append(str(label))
    return normalized


def replace_marked_block(path: Path, lines: Iterable[str]) -> None:
    new_block = "\n".join([START_MARKER, *lines, END_MARKER]) + "\n"
    if path.exists():
        original = path.read_text(encoding="utf-8")
        if START_MARKER in original and END_MARKER in original:
            before, rest = original.split(START_MARKER, 1)
            _, after = rest.split(END_MARKER, 1)
            updated = before.rstrip() + "\n\n" + new_block + after.lstrip("\n")
        else:
            updated = original.rstrip() + "\n\n" + new_block
    else:
        updated = new_block
    path.write_text(updated, encoding="utf-8")


def read_ticket_queue(paper_root: Path) -> List[Dict[str, Any]]:
    queue_path = paper_root / "autoresearch" / "ticket_queue.jsonl"
    rows: List[Dict[str, Any]] = []
    if not queue_path.exists():
        return rows
    for raw_line in queue_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def accepted_tickets(paper_root: Path) -> List[Dict[str, Any]]:
    return [row for row in read_ticket_queue(paper_root) if row.get("status") in {"accepted", "completed"}]


def accepted_ticket_ids(paper_root: Path) -> List[str]:
    return [row["ticket_id"] for row in accepted_tickets(paper_root)]


def accepted_ticket_lookup(paper_root: Path) -> Dict[str, Dict[str, Any]]:
    return {row["ticket_id"]: row for row in accepted_tickets(paper_root)}


def ticket_result_path(paper_root: Path, ticket_id: str, relative_path: str) -> Path:
    ticket = accepted_ticket_lookup(paper_root).get(ticket_id)
    if ticket:
        run_id = ((ticket.get("result_ref") or {}).get("run_id"))
        if run_id:
            candidate = paper_root / "results" / "autoresearch" / run_id / relative_path
            if candidate.exists():
                return candidate
    return latest_file(paper_root / "results" / "autoresearch", f"**/{relative_path}")


def collect_sources(paper_root: Path) -> Dict[str, Path]:
    return {
        "dataset_bridge": ticket_result_path(paper_root, "moe-dataset-bridge", "dataset_bridge/dataset_bridge_summary.json"),
        "stability": ticket_result_path(paper_root, "moe-seed-stability", "seed_stability/stability_summary.json"),
        "routing": ticket_result_path(paper_root, "moe-routing-analysis", "routing_analysis/analysis_summary.json"),
        "stability_strategy": ticket_result_path(paper_root, "moe-stability-strategy", "stability_strategy/stability_strategy_summary.json"),
        "expert_ablation": ticket_result_path(paper_root, "moe-expert-ablation", "expert_ablation/ablation_summary.json"),
    }


def mean_entropy(weights: List[float]) -> float:
    clipped = [max(min(float(value), 1.0), 1e-12) for value in weights]
    return float(-sum(value * math.log(value) for value in clipped))


def build_routing_claims(routing: Dict[str, Any], source: Path) -> Dict[str, Any]:
    route_entropy_mean = routing.get("route_entropy_mean")
    if route_entropy_mean is None:
        samples = routing.get("sample_routing_analysis") or routing.get("path_signatures") or []
        entropies = [float(item["routing_entropy"]) for item in samples if item.get("routing_entropy") is not None]
        if entropies:
            route_entropy_mean = float(sum(entropies) / len(entropies))
    expert_usage_distribution = routing.get("expert_usage_distribution")
    expert_activations = (
        routing.get("expert_activations")
        or (routing.get("explanations") or {}).get("expert_activations")
        or routing.get("expert_statistics")
        or {}
    )
    if expert_usage_distribution is None and expert_activations.get("mean_weights") is not None:
        expert_usage_distribution = [float(value) for value in expert_activations["mean_weights"]]
    if route_entropy_mean is None and expert_usage_distribution:
        route_entropy_mean = mean_entropy(list(expert_usage_distribution))
    path_signature_examples = routing.get("path_signature_examples")
    if path_signature_examples is None:
        sample_explanations = routing.get("sample_explanations") or (routing.get("explanations") or {}).get("sample_explanations") or []
        labels = [
            str(((item.get("routing_decision") or {}).get("selected_expert")))
            for item in sample_explanations
            if ((item.get("routing_decision") or {}).get("selected_expert")) is not None
        ]
        if not labels:
            path_signatures = routing.get("path_signatures") or []
            labels = [f"expert_{int(item['dominant_expert'])}" for item in path_signatures if item.get("dominant_expert") is not None]
        unique_labels: List[str] = []
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)
        path_signature_examples = unique_labels[:3]
    return {
        "source": str(source),
        "route_entropy_mean": route_entropy_mean,
        "expert_usage_distribution": expert_usage_distribution,
        "path_signature_examples": path_signature_examples or [],
    }


def build_review_evidence(output_dir: Path, paper_root: Path) -> Path:
    accepted = accepted_ticket_ids(paper_root)
    accepted_set = set(accepted)
    sources = collect_sources(paper_root)
    dataset_bridge = load_json(sources["dataset_bridge"])
    stability = load_json(sources["stability"])
    routing = load_json(sources["routing"])
    strategy = load_json(sources["stability_strategy"])
    ablation = load_json(sources["expert_ablation"])
    routing_claims = build_routing_claims(routing, sources["routing"])
    dataset_labels = normalize_dataset_entries(
        dataset_bridge.get("successful_datasets", dataset_bridge.get("datasets", []))
    )

    missing_tickets = sorted(REQUIRED_REVIEW_TICKETS - accepted_set)
    open_issues = [f"missing accepted ticket: {ticket_id}" for ticket_id in missing_tickets]
    if dataset_bridge.get("mean_test_acc") is None:
        open_issues.append("dataset bridge mean_test_acc missing")
    if routing_claims.get("route_entropy_mean") is None:
        open_issues.append("routing route_entropy_mean missing")
    if not routing_claims.get("expert_usage_distribution"):
        open_issues.append("routing expert_usage_distribution missing")
    ready = not open_issues

    summary = {
        "ready": ready,
        "accepted_ticket_ids": accepted,
        "expected_claims": ["stability", "routing", "datasets", "expert_ablation", "stability_strategy"],
        "open_issues": open_issues,
        "claims": {
            "stability": {
                "source": str(sources["stability"]),
                "mean_accuracy": stability.get("mean_accuracy"),
                "std_accuracy": stability.get("std_accuracy"),
                "ci95_accuracy": stability.get("ci95_accuracy"),
                "cv_percent": stability.get("cv_percent"),
            },
            "routing": {
                **routing_claims,
            },
            "datasets": {
                "source": str(sources["dataset_bridge"]),
                "datasets": dataset_labels,
                "mean_test_acc": dataset_bridge.get("mean_test_acc"),
            },
            "expert_ablation": {
                "source": str(sources["expert_ablation"]),
                "expert_counts": ablation.get("expert_counts"),
                "successful_expert_counts": ablation.get("successful_expert_counts"),
                "curve_rows": ablation.get("curve_rows", []),
            },
            "stability_strategy": {
                "source": str(sources["stability_strategy"]),
                "strategies": strategy.get("strategies", []),
                "metric": strategy.get("metric"),
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "claim_evidence_map.json"
    dump_json(summary_path, summary)

    lines = [
        "# MoE Claim-Evidence Map",
        "",
        f"- ready: `{ready}`",
        f"- accepted_ticket_ids: `{', '.join(accepted)}`",
        f"- open_issues: `{'; '.join(open_issues) if open_issues else 'none'}`",
        "",
        "## Claims",
        "",
        f"- stability: `{sources['stability']}`",
        f"- routing: `{sources['routing']}`",
        f"- datasets: `{sources['dataset_bridge']}`",
        f"- expert_ablation: `{sources['expert_ablation']}`",
        f"- stability_strategy: `{sources['stability_strategy']}`",
    ]
    (output_dir / "claim_evidence_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def build_manuscript_binding(output_dir: Path, paper_root: Path) -> Path:
    accepted = accepted_ticket_ids(paper_root)
    accepted_set = set(accepted)
    if "moe-review-evidence" not in accepted_set:
        raise RuntimeError("moe-review-evidence must be accepted before manuscript binding")

    sources = collect_sources(paper_root)
    dataset_bridge = load_json(sources["dataset_bridge"])
    stability = load_json(sources["stability"])
    routing = load_json(sources["routing"])
    ablation = load_json(sources["expert_ablation"])
    review_map = load_json(latest_file(paper_root / "results" / "autoresearch", "**/review_evidence/claim_evidence_map.json"))
    routing_claims = build_routing_claims(routing, sources["routing"])
    dataset_labels = normalize_dataset_entries(
        dataset_bridge.get("successful_datasets", dataset_bridge.get("datasets", []))
    )
    manuscript_sync_path: Path | None = None
    accepted_lookup = accepted_ticket_lookup(paper_root)
    if accepted_lookup.get("moe-manuscript-truth-sync"):
        manuscript_sync_path = ticket_result_path(
            paper_root,
            "moe-manuscript-truth-sync",
            "manuscript_truth_sync/manuscript_truth_sync_summary.json",
        )
    manuscript_sync = load_json(manuscript_sync_path) if manuscript_sync_path else None
    open_issues: List[str] = []
    if not review_map.get("ready"):
        open_issues.append("review evidence not ready")
    if manuscript_sync is None:
        open_issues.append("manuscript truth sync missing")
    elif not manuscript_sync.get("all_targets_synced"):
        open_issues.append("manuscript truth sync incomplete")
    if dataset_bridge.get("mean_test_acc") is None:
        open_issues.append("dataset bridge mean_test_acc missing")
    if routing_claims.get("route_entropy_mean") is None:
        open_issues.append("routing route_entropy_mean missing")
    if not routing_claims.get("expert_usage_distribution"):
        open_issues.append("routing expert_usage_distribution missing")

    internal_binding_ready = not open_issues
    parent_gate_blockers = [
        "parent UXFD accepted-run artifact gate not satisfied",
        "parent UXFD 2x4090 GPU queue not accepted",
        "parent UXFD cross-paper submission gate not passed",
    ]
    binding_summary = {
        "bound": True,
        "internal_binding_ready": internal_binding_ready,
        "submission_ready": False,
        "submission_ready_policy": (
            "This script binds internal MoE evidence only. External IEEE "
            "submission readiness is controlled by the parent UXFD gate."
        ),
        "parent_gate_blockers": parent_gate_blockers,
        "accepted_ticket_ids": accepted,
        "datasets": dataset_labels,
        "mean_test_acc": dataset_bridge.get("mean_test_acc"),
        "stability": {
            "mean_accuracy": stability.get("mean_accuracy"),
            "std_accuracy": stability.get("std_accuracy"),
            "ci95_accuracy": stability.get("ci95_accuracy"),
            "cv_percent": stability.get("cv_percent"),
        },
        "routing": routing_claims,
        "ablation_curve": ablation.get("curve_rows", []),
        "tables": ["mean_std_ci_cv_table", "routing_analysis_table", "ablation_matrix"],
        "figures": ["expert_activation_heatmap", "path_signature_visualization", "stability_curve"],
        "review_map": str(latest_file(paper_root / "results" / "autoresearch", "**/review_evidence/claim_evidence_map.json")),
        "manuscript_truth_sync": str(manuscript_sync_path) if manuscript_sync_path else None,
        "open_issues": open_issues,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "manuscript_binding_summary.json"
    dump_json(summary_path, binding_summary)

    manuscript_lines = [
        "## Internal Evidence Binding Snapshot",
        "",
        f"- status: `{'bound' if internal_binding_ready else 'needs_followup'}`",
        "- external_submission_ready: `false`",
        "- external_submission_ready_policy: `controlled by the parent UXFD gate`",
        f"- accepted_ticket_ids: `{', '.join(accepted)}`",
        f"- datasets: `{', '.join(str(item) for item in binding_summary['datasets'])}`",
        f"- dataset_bridge_source: `{sources['dataset_bridge']}`",
        f"- stability_source: `{sources['stability']}`",
        f"- routing_source: `{sources['routing']}`",
        f"- expert_ablation_source: `{sources['expert_ablation']}`",
        f"- mean_test_acc: `{dataset_bridge.get('mean_test_acc')}`",
        f"- mean_accuracy: `{stability.get('mean_accuracy')}`",
        f"- std_accuracy: `{stability.get('std_accuracy')}`",
        f"- ci95_accuracy: `{stability.get('ci95_accuracy')}`",
        f"- cv_percent: `{stability.get('cv_percent')}`",
        f"- route_entropy_mean: `{routing_claims.get('route_entropy_mean')}`",
        f"- expert_usage_distribution: `{routing_claims.get('expert_usage_distribution')}`",
        f"- manuscript_truth_sync: `{binding_summary['manuscript_truth_sync']}`",
        f"- ablation_curve_rows: `{len(ablation.get('curve_rows', []))}`",
        f"- review_map_ready: `{review_map.get('ready')}`",
        f"- internal_blockers: `{'; '.join(open_issues) if open_issues else 'none'}`",
        f"- parent_gate_blockers: `{'; '.join(parent_gate_blockers)}`",
    ]
    blueprint_lines = [
        f"- manuscript_status: `{'bound' if internal_binding_ready else 'needs_followup'}`",
        "- external_submission_ready: `false`",
        "- external_submission_ready_policy: `controlled by the parent UXFD gate`",
        f"- dataset_bridge: `accepted ({sources['dataset_bridge']})`",
        f"- expert_ablation: `accepted ({sources['expert_ablation']})`",
        f"- review_evidence: `accepted ({binding_summary['review_map']})`",
        f"- manuscript_truth_sync: `{binding_summary['manuscript_truth_sync']}`",
        "- manuscript_binding: `accepted`",
        f"- datasets: `{', '.join(str(item) for item in binding_summary['datasets'])}`",
        f"- mean_test_acc: `{dataset_bridge.get('mean_test_acc')}`",
        f"- stability_cv_percent: `{stability.get('cv_percent')}`",
        f"- route_entropy_mean: `{routing_claims.get('route_entropy_mean')}`",
        f"- ablation_curve_rows: `{len(ablation.get('curve_rows', []))}`",
    ]
    replace_marked_block(paper_root / "manuscript" / "AUTORESEARCH_EVIDENCE.md", manuscript_lines)
    replace_marked_block(paper_root / "paper_blueprint.md", blueprint_lines)
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Bind accepted MoE internal evidence into reviewer and manuscript summaries.")
    parser.add_argument("--mode", choices=["review-evidence", "manuscript-binding"], required=True)
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paper_root = Path(args.paper_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.mode == "review-evidence":
        summary_path = build_review_evidence(output_dir, paper_root)
    else:
        summary_path = build_manuscript_binding(output_dir, paper_root)

    print(json.dumps(load_json(summary_path), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
