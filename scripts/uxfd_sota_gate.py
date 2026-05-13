from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from scripts.uxfd_gpu_queue import DEFAULT_QUEUE


DEFAULT_SOTA_ROOT = Path("paper/UXFD_paper/results/sota_aggregates")
DEFAULT_ACCEPTED_RUN_ROOT = Path("paper/UXFD_paper/results/accepted_runs")
AGGREGATE_FILENAME = "sota_aggregate.yaml"
ACCEPTED_CLAIM_SCOPES = ("exact_sota", "representative_only", "bounded_non_sota")
ACCEPTED_TOP_SCOPES = ("exact", "representative")
STATISTIC_FIELDS = ("mean", "std", "ci95_low", "ci95_high")
DISALLOWED_REF_MARKERS = ("todo", "template", "smoke", "demo", "dummy", "pending")


@dataclass(frozen=True)
class SotaPaperRecord:
    paper_id: str
    aggregate_path: str
    accepted: bool
    issues: Tuple[str, ...]


@dataclass(frozen=True)
class SotaGateReport:
    ready: bool
    aggregate_root: str
    accepted_run_root: str
    records: Tuple[SotaPaperRecord, ...]
    blockers: Tuple[str, ...]
    expected_papers: int
    accepted_papers: int


def _load_yaml(path: Path) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _coerce_seed_set(value: Any) -> Optional[Tuple[int, ...]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    seeds: List[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        seeds.append(item)
    return tuple(seeds)


def _paper_requirements(queue_path: Path) -> Tuple[Mapping[str, Any], ...]:
    queue = _load_yaml(queue_path)
    top_bindings_by_paper: Dict[str, List[str]] = {}
    for binding in queue.get("top_representative_bindings", ()):
        paper_id = str(binding.get("paper_id", ""))
        binding_id = str(binding.get("binding_id", ""))
        if paper_id and binding_id:
            top_bindings_by_paper.setdefault(paper_id, []).append(binding_id)

    requirements: List[Mapping[str, Any]] = []
    for item in queue.get("paper_queue", ()):
        paper_id = str(item.get("paper_id", ""))
        matrix_path = Path(str(item.get("matrix_path", "")))
        baseline_ids: Tuple[str, ...] = ()
        if matrix_path.exists():
            matrix = _load_yaml(matrix_path)
            baseline_ids = tuple(
                str(entry.get("id", ""))
                for entry in matrix.get("baselines", ())
                if isinstance(entry, Mapping) and entry.get("id")
            )
        requirements.append(
            {
                "paper_id": paper_id,
                "minimum_seeds": int(item.get("minimum_seeds", 3)),
                "baseline_ids": baseline_ids,
                "top_binding_ids": tuple(top_bindings_by_paper.get(paper_id, ())),
            }
        )
    return tuple(requirements)


def _validate_statistics(prefix: str, payload: Mapping[str, Any]) -> Tuple[str, ...]:
    issues: List[str] = []
    statistics = payload.get("statistics", {})
    if not isinstance(statistics, Mapping):
        return (f"{prefix}.statistics must be a mapping",)
    for field in STATISTIC_FIELDS:
        if not _is_number(statistics.get(field)):
            issues.append(f"{prefix}.statistics.{field} must be numeric")
    return tuple(issues)


def _has_effect_or_test(payload: Mapping[str, Any]) -> bool:
    if _is_number(payload.get("effect_size_vs_proposed")):
        return True
    paired_test = payload.get("paired_test", {})
    return (
        isinstance(paired_test, Mapping)
        and bool(str(paired_test.get("name", "")).strip())
        and _is_number(paired_test.get("p_value"))
    )


def _validate_accepted_run_refs(
    prefix: str,
    payload: Mapping[str, Any],
    proposed_seeds: Tuple[int, ...],
    accepted_run_root: Path,
    paper_id: str,
) -> Tuple[str, ...]:
    refs = payload.get("accepted_run_refs")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)) or not refs:
        return (f"{prefix}.accepted_run_refs must be a non-empty list",)

    issues: List[str] = []
    if len(refs) < len(proposed_seeds):
        issues.append(
            f"{prefix}.accepted_run_refs must cover every matched seed"
        )
    expected_entry_id = str(payload.get("entry_id") or payload.get("binding_id") or "")
    referenced_seeds: set[int] = set()
    root = accepted_run_root.resolve()
    for index, ref in enumerate(refs):
        if not isinstance(ref, str) or not ref.strip():
            issues.append(f"{prefix}.accepted_run_refs[{index}] must be a non-empty string")
            continue
        lowered = ref.lower()
        marker = next((item for item in DISALLOWED_REF_MARKERS if item in lowered), "")
        if marker:
            issues.append(f"{prefix}.accepted_run_refs[{index}] must not reference {marker}")
        ref_path = Path(ref)
        if ref_path.is_absolute():
            issues.append(f"{prefix}.accepted_run_refs[{index}] must be relative")
            continue
        candidate = (accepted_run_root / ref_path).resolve()
        if not candidate.is_relative_to(root):
            issues.append(f"{prefix}.accepted_run_refs[{index}] must stay inside accepted_run_root")
            continue
        if not candidate.exists():
            issues.append(f"{prefix}.accepted_run_refs[{index}] does not exist")
            continue
        if candidate.is_dir():
            if not (candidate / "run_meta.yaml").exists():
                issues.append(
                    f"{prefix}.accepted_run_refs[{index}] directory lacks run_meta.yaml"
                )
                continue
            run_meta_path = candidate / "run_meta.yaml"
        elif candidate.name != "run_meta.yaml":
            issues.append(f"{prefix}.accepted_run_refs[{index}] must reference run_meta.yaml")
            continue
        else:
            run_meta_path = candidate

        run_meta = _load_yaml(run_meta_path)
        if str(run_meta.get("paper_id", "")) != paper_id:
            issues.append(f"{prefix}.accepted_run_refs[{index}] paper_id mismatch")
        if (
            expected_entry_id
            and str(run_meta.get("entry_id", "")) != expected_entry_id
        ):
            issues.append(f"{prefix}.accepted_run_refs[{index}] entry_id mismatch")
        seed = run_meta.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            issues.append(f"{prefix}.accepted_run_refs[{index}] seed must be an integer")
        elif seed not in proposed_seeds:
            issues.append(
                f"{prefix}.accepted_run_refs[{index}] seed is not in matched seed set"
            )
        else:
            referenced_seeds.add(seed)
        if str(run_meta.get("evidence_level", "")).strip() != "accepted_same_protocol":
            issues.append(
                f"{prefix}.accepted_run_refs[{index}] evidence_level must be "
                "accepted_same_protocol"
            )
    missing_seeds = sorted(set(proposed_seeds) - referenced_seeds)
    if missing_seeds:
        issues.append(
            f"{prefix}.accepted_run_refs must cover matched run_meta seeds: "
            + ",".join(str(seed) for seed in missing_seeds)
        )
    return tuple(issues)


def _validate_comparison_entry(
    prefix: str,
    payload: Mapping[str, Any],
    proposed_seeds: Tuple[int, ...],
    accepted_run_root: Path,
    paper_id: str,
) -> Tuple[str, ...]:
    issues: List[str] = []
    seed_values = _coerce_seed_set(payload.get("seed_values"))
    if seed_values != proposed_seeds:
        issues.append(f"{prefix}.seed_values must match proposed seed set")
    issues.extend(_validate_statistics(prefix, payload))
    if not _has_effect_or_test(payload):
        issues.append(
            f"{prefix} must include numeric effect_size_vs_proposed or paired_test.p_value"
        )
    issues.extend(
        _validate_accepted_run_refs(
            prefix, payload, proposed_seeds, accepted_run_root, paper_id
        )
    )
    return tuple(issues)


def _validate_aggregate(
    path: Path,
    requirement: Mapping[str, Any],
    accepted_run_root: Path,
) -> SotaPaperRecord:
    paper_id = str(requirement["paper_id"])
    if not path.exists():
        return SotaPaperRecord(
            paper_id=paper_id,
            aggregate_path=str(path),
            accepted=False,
            issues=("missing sota_aggregate.yaml",),
        )

    issues: List[str] = []
    data = _load_yaml(path)
    if data.get("accepted_sota_evidence") is not True:
        issues.append("accepted_sota_evidence must be true")
    if str(data.get("paper_id", "")) != paper_id:
        issues.append("paper_id must match queue paper_id")

    claim_scope = str(data.get("claim_scope", ""))
    if claim_scope not in ACCEPTED_CLAIM_SCOPES:
        issues.append(
            "claim_scope must be one of exact_sota, representative_only, bounded_non_sota"
        )

    proposed = data.get("proposed", {})
    if not isinstance(proposed, Mapping):
        issues.append("proposed must be a mapping")
        proposed = {}
    proposed_seeds = _coerce_seed_set(proposed.get("seed_values"))
    minimum_seeds = int(requirement["minimum_seeds"])
    if proposed_seeds is None:
        issues.append("proposed.seed_values must be an integer list")
        proposed_seeds = ()
    elif len(proposed_seeds) < minimum_seeds:
        issues.append(f"proposed.seed_values must include at least {minimum_seeds} seeds")
    issues.extend(_validate_statistics("proposed", proposed))
    issues.extend(
        _validate_accepted_run_refs(
            "proposed",
            proposed,
            proposed_seeds,
            accepted_run_root,
            paper_id,
        )
    )

    comparators = data.get("comparators", ())
    if not isinstance(comparators, Sequence) or isinstance(comparators, (str, bytes)):
        comparators = ()
        issues.append("comparators must be a list")
    comparator_ids = {
        str(entry.get("entry_id", ""))
        for entry in comparators
        if isinstance(entry, Mapping)
    }
    baseline_ids = set(requirement["baseline_ids"])
    missing_baselines = sorted(baseline_ids - comparator_ids)
    if missing_baselines:
        issues.append("missing baseline comparators: " + ",".join(missing_baselines))
    if len([item for item in comparator_ids if item.startswith("B")]) < 6:
        issues.append("at least six baseline comparators are required")
    for entry in comparators:
        if not isinstance(entry, Mapping):
            issues.append("comparators entries must be mappings")
            continue
        entry_id = str(entry.get("entry_id", ""))
        issues.extend(
            _validate_comparison_entry(
                f"comparators[{entry_id or '?'}]",
                entry,
                proposed_seeds,
                accepted_run_root,
                paper_id,
            )
        )

    top_entries = data.get("top_representatives", ())
    if not isinstance(top_entries, Sequence) or isinstance(top_entries, (str, bytes)):
        top_entries = ()
        issues.append("top_representatives must be a list")
    top_ids = {
        str(entry.get("binding_id", ""))
        for entry in top_entries
        if isinstance(entry, Mapping)
    }
    required_top_ids = set(requirement["top_binding_ids"])
    missing_top_ids = sorted(required_top_ids - top_ids)
    if missing_top_ids:
        issues.append("missing TOP representative bindings: " + ",".join(missing_top_ids))
    for entry in top_entries:
        if not isinstance(entry, Mapping):
            issues.append("top_representatives entries must be mappings")
            continue
        binding_id = str(entry.get("binding_id", ""))
        scope = str(entry.get("scope", ""))
        if scope not in ACCEPTED_TOP_SCOPES:
            issues.append(f"top_representatives[{binding_id or '?'}].scope must be exact or representative")
        if claim_scope == "exact_sota" and scope != "exact":
            issues.append(
                f"top_representatives[{binding_id or '?'}].scope must be exact for exact_sota"
            )
        issues.extend(
            _validate_comparison_entry(
                f"top_representatives[{binding_id or '?'}]",
                entry,
                proposed_seeds,
                accepted_run_root,
                paper_id,
            )
        )

    return SotaPaperRecord(
        paper_id=paper_id,
        aggregate_path=str(path),
        accepted=not issues,
        issues=tuple(issues),
    )


def evaluate_sota_gate(
    aggregate_root: Path = DEFAULT_SOTA_ROOT,
    queue_path: Path = DEFAULT_QUEUE,
    accepted_run_root: Path = DEFAULT_ACCEPTED_RUN_ROOT,
) -> SotaGateReport:
    blockers: List[str] = []
    if not aggregate_root.exists():
        blockers.append(f"sota aggregate root does not exist: {aggregate_root}")

    records: List[SotaPaperRecord] = []
    for requirement in _paper_requirements(queue_path):
        paper_id = str(requirement["paper_id"])
        aggregate_path = aggregate_root / paper_id / AGGREGATE_FILENAME
        record = _validate_aggregate(aggregate_path, requirement, accepted_run_root)
        records.append(record)
        if not record.accepted:
            blockers.append(f"{record.aggregate_path}: {len(record.issues)} issues")

    return SotaGateReport(
        ready=not blockers and bool(records),
        aggregate_root=str(aggregate_root),
        accepted_run_root=str(accepted_run_root),
        records=tuple(records),
        blockers=tuple(blockers),
        expected_papers=len(records),
        accepted_papers=sum(1 for record in records if record.accepted),
    )


def render_markdown(report: SotaGateReport) -> str:
    lines = [
        "# UXFD SOTA Gate",
        "",
        f"- Ready: `{report.ready}`",
        f"- Aggregate root: `{report.aggregate_root}`",
        f"- Accepted run root: `{report.accepted_run_root}`",
        f"- Accepted papers: `{report.accepted_papers}/{report.expected_papers}`",
        f"- Blockers: `{len(report.blockers)}`",
        "",
        "| Paper | Accepted | Issues | Aggregate |",
        "|---|---:|---:|---|",
    ]
    for record in report.records:
        lines.append(
            f"| `{record.paper_id}` | `{record.accepted}` | "
            f"{len(record.issues)} | `{record.aggregate_path}` |"
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.blockers:
        lines.append(f"- {blocker}")
    return "\n".join(lines) + "\n"


def build_payload(report: SotaGateReport) -> Mapping[str, Any]:
    return asdict(report)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate UXFD SOTA aggregate evidence")
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_SOTA_ROOT)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--accepted-run-root", type=Path, default=DEFAULT_ACCEPTED_RUN_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_sota_gate(
        args.aggregate_root,
        queue_path=args.queue,
        accepted_run_root=args.accepted_run_root,
    )
    output = (
        json.dumps(build_payload(report), indent=2) + "\n"
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")

    if report.ready or args.allow_not_ready:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
