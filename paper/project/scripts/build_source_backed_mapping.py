#!/usr/bin/env python3
"""Build a source-backed cross-paper mapping report for Paper 06.

The report is intentionally read-only. It checks sibling UXFD submodules for
concrete files and mapping terms, then records paths that support the
neural-symbolic framework mapping. It does not create accepted performance,
GPU, TOP-reproduction, or SOTA evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PAPER_ROOT = Path(__file__).resolve().parents[1]
UXFD_ROOT = PAPER_ROOT.parent
DEFAULT_JSON = PAPER_ROOT / "report" / "source_backed_mapping_report.json"
DEFAULT_MD = PAPER_ROOT / "report" / "source_backed_mapping_report.md"


@dataclass(frozen=True)
class MappingTarget:
    paper_id: str
    mapping_role: str
    evidence_paths: Sequence[str]
    required_terms: Sequence[str]
    layer_terms: Dict[str, Sequence[str]]


TARGETS: Sequence[MappingTarget] = (
    MappingTarget(
        paper_id="1D-2D_fusion_explainable",
        mapping_role="signal-to-representation fusion and alignment evidence",
        evidence_paths=(
            "VIBENCH.md",
            "submission_prep/baseline_ablation_matrix.yaml",
            "configs/vibench/min.yaml",
            "code/models/fusion_aligned.py",
            "code/alignment/physical_alignment.py",
        ),
        required_terms=("fusion", "alignment", "signal_processing_2d"),
        layer_terms={
            "signal_layer": ("1D", "2D", "spectral", "frequency"),
            "neural_layer": ("features_1d", "features_2d", "fusion_layers"),
            "constraint_layer": ("physical", "semantic", "geometric", "alignment_loss"),
            "evidence_layer": ("VIBENCH", "baseline_ablation_matrix", "accepted_evidence_status"),
        },
    ),
    MappingTarget(
        paper_id="MOE_explainable",
        mapping_role="expert-routing and physics-constrained mixture evidence",
        evidence_paths=(
            "VIBENCH.md",
            "submission_prep/baseline_ablation_matrix.yaml",
            "configs/vibench/min.yaml",
            "code/moe_model.py",
            "code/router/statistical_router.py",
        ),
        required_terms=("expert", "router", "load_balance", "sparsity"),
        layer_terms={
            "signal_layer": ("low_pass", "harmonic", "envelope", "frequency"),
            "neural_layer": ("expert_outputs", "routing_weights", "moe"),
            "constraint_layer": ("load_balance", "sparsity", "orthogonal", "diversity"),
            "evidence_layer": ("route", "expert activation", "baseline_ablation_matrix"),
        },
    ),
    MappingTarget(
        paper_id="Paper_fuzzy_XFD",
        mapping_role="fuzzy rule, membership, and decision-path evidence",
        evidence_paths=(
            "VIBENCH.md",
            "submission_prep/baseline_ablation_matrix.yaml",
            "configs/vibench/min.yaml",
            "code/fuzzy_system/rule_base.py",
            "code/fuzzy_system/membership_functions.py",
        ),
        required_terms=("fuzzy", "rule", "membership"),
        layer_terms={
            "signal_layer": ("feature", "fault", "diagnosis"),
            "neural_layer": ("NSN", "TSPN_UXFD", "decision_configs"),
            "constraint_layer": ("membership", "rule", "predicate"),
            "evidence_layer": ("rule-level", "active_rules", "baseline_ablation_matrix"),
        },
    ),
    MappingTarget(
        paper_id="Explainable_FD_Toolkit",
        mapping_role="explanation schema, metric, manifest, and toolkit evidence",
        evidence_paths=(
            "VIBENCH.md",
            "submission_prep/baseline_ablation_matrix.yaml",
            "configs/vibench/min.yaml",
            "scripts/run_toolkit_ablations.py",
            "scripts/validate_schema.py",
        ),
        required_terms=("explain", "schema", "manifest"),
        layer_terms={
            "signal_layer": ("fault", "diagnosis", "dataset"),
            "neural_layer": ("model", "baseline", "NSN"),
            "constraint_layer": ("schema", "metric", "snapshot"),
            "evidence_layer": ("manifest", "artifact", "accepted_evidence"),
        },
    ),
    MappingTarget(
        paper_id="LLM_Explainable_FD_Toolkit",
        mapping_role="LLM evidence-chain and unsupported-claim control evidence",
        evidence_paths=(
            "VIBENCH.md",
            "submission_prep/baseline_ablation_matrix.yaml",
            "configs/vibench/min.yaml",
            "code/llm_explainable_toolkit/core/intermediate_representation.py",
            "experiments/scripts/run_llm_evidence_smoke.py",
        ),
        required_terms=("llm", "evidence", "unsupported"),
        layer_terms={
            "signal_layer": ("diagnosis", "fault", "time"),
            "neural_layer": ("intermediate", "adapter", "model"),
            "constraint_layer": ("unsupported", "hallucination", "checker"),
            "evidence_layer": ("run_meta", "metrics", "evidence"),
        },
    ),
    MappingTarget(
        paper_id="TII_operator_attention",
        mapping_role="operator-attention and signal-operator evidence",
        evidence_paths=(
            "VIBENCH.md",
            "submission_prep/baseline_ablation_matrix.yaml",
            "configs/vibench/min.yaml",
            "code/synthetic_signals/operator_validation.py",
            "code/synthetic_verification.py",
        ),
        required_terms=("operator", "attention", "FFT"),
        layer_terms={
            "signal_layer": ("FFT", "Hilbert", "signal", "operator"),
            "neural_layer": ("attention", "operator_attention", "NSN"),
            "constraint_layer": ("temperature", "identity", "subset"),
            "evidence_layer": ("validation", "baseline_ablation_matrix", "accepted_evidence"),
        },
    ),
)


def _normal_forms(text: str) -> Sequence[str]:
    lower = text.lower()
    return (
        lower,
        lower.replace("_", " "),
        lower.replace("-", " "),
        lower.replace("_", "").replace("-", ""),
    )


def _term_matches(term: str, text: str) -> bool:
    term_forms = _normal_forms(term)
    text_forms = _normal_forms(text)
    return any(term_form in text_form for term_form in term_forms for text_form in text_forms)


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _collect_target(target: MappingTarget) -> Dict[str, object]:
    paper_root = UXFD_ROOT / target.paper_id
    evidence_records = []
    all_text = ""

    for relative_path in target.evidence_paths:
        path = paper_root / relative_path
        text = _read_text(path)
        all_text += "\n" + text
        terms_to_check = set(target.required_terms)
        for terms in target.layer_terms.values():
            terms_to_check.update(terms)
        matched_terms = sorted(term for term in terms_to_check if _term_matches(term, text))
        evidence_records.append(
            {
                "path": relpath(path, PAPER_ROOT),
                "exists": path.exists(),
                "matched_terms": matched_terms,
            }
        )

    required_matches = sorted(
        term for term in target.required_terms if _term_matches(term, all_text)
    )
    layer_support = {
        layer: sorted(term for term in terms if _term_matches(term, all_text))
        for layer, terms in target.layer_terms.items()
    }
    missing_required = sorted(set(target.required_terms) - set(required_matches))
    missing_layers = sorted(layer for layer, terms in layer_support.items() if not terms)
    source_backed = (
        all(record["exists"] for record in evidence_records)
        and not missing_required
        and len(missing_layers) == 0
    )

    return {
        "paper_id": target.paper_id,
        "mapping_role": target.mapping_role,
        "source_backed": source_backed,
        "accepted_evidence": False,
        "required_terms": list(target.required_terms),
        "matched_required_terms": required_matches,
        "missing_required_terms": missing_required,
        "layer_support": layer_support,
        "missing_layers": missing_layers,
        "evidence": evidence_records,
    }


def build_report() -> Dict[str, object]:
    papers = [_collect_target(target) for target in TARGETS]
    source_backed = all(paper["source_backed"] for paper in papers)
    return {
        "report_id": "paper06_source_backed_cross_method_mapping",
        "paper_id": "Neuralsymbolic_theory",
        "source_backed": source_backed,
        "accepted_evidence": False,
        "paper_count": len(papers),
        "papers": papers,
        "limitations": [
            "This report is source-introspection evidence only.",
            "It does not prove model performance, mapping impact, TOP-method reproduction, GPU feasibility, or SOTA.",
            "Accepted train/eval evidence still requires same-protocol logs, metrics, run_meta.yaml, and local GPU metadata.",
        ],
    }


def render_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Paper 06 Source-Backed Cross-Method Mapping",
        "",
        f"- Source-backed: `{str(report['source_backed']).lower()}`",
        "- Accepted evidence: `false`",
        f"- Papers checked: `{report['paper_count']}`",
        "",
        "| Paper | Mapping role | Source-backed | Matched required terms | Evidence paths |",
        "|---|---|---:|---|---|",
    ]
    for paper in report["papers"]:
        evidence_paths = "<br>".join(
            record["path"] for record in paper["evidence"] if record["exists"]
        )
        matched_terms = ", ".join(paper["matched_required_terms"])
        lines.append(
            "| {paper_id} | {role} | `{source}` | {terms} | {paths} |".format(
                paper_id=paper["paper_id"],
                role=paper["mapping_role"],
                source=str(paper["source_backed"]).lower(),
                terms=matched_terms,
                paths=evidence_paths,
            )
        )

    lines.extend(
        [
            "",
            "## Layer Support",
            "",
        ]
    )
    for paper in report["papers"]:
        lines.append(f"### {paper['paper_id']}")
        for layer, terms in paper["layer_support"].items():
            term_text = ", ".join(terms) if terms else "MISSING"
            lines.append(f"- `{layer}`: {term_text}")
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
        ]
    )
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build source-backed Paper06 cross-method mapping evidence."
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        "source_backed={source_backed} accepted_evidence=false papers={paper_count}".format(
            source_backed=str(report["source_backed"]).lower(),
            paper_count=report["paper_count"],
        )
    )
    return 0 if report["source_backed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
