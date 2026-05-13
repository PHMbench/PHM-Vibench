from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


DEFAULT_RECENT_WORK_README = Path("paper/UXFD_paper/goal/08_recent_work_citation_readme.md")
DEFAULT_QUEUE = Path("paper/UXFD_paper/goal/09_gpu_execution_queue.yaml")
DEFAULT_MATRIX_ROOT = Path("paper/UXFD_paper")

LOW_TIER_MARKERS = (
    "Scientific Reports",
    "MDPI",
    "IEEE TIM",
    "IEEE Transactions on Instrumentation and Measurement",
    "IEEE Access",
    "Applied Sciences",
    "Electronics",
    "Sensors",
    "Mathematics",
)

REQUIRED_2026_TOP_IDS = (
    "RWTOP2026-TIMESEG",
    "RWTOP2026-TIMESLIVER",
    "RWTOP2026-PGRFNET",
    "RWTOP2026-GTM",
    "RWTOP2026-CSLSTM",
    "RWTOP2026-PROTOTS",
    "RWTOP2026-CALTSFM",
    "RWTOP2026-TSPULSE",
)

REQUIRED_2026_SOURCE_URLS = {
    "RWTOP2026-TIMESEG": "https://openreview.net/forum?id=alt9mSWULk",
    "RWTOP2026-TIMESLIVER": "https://openreview.net/forum?id=MDRp9XhGtS",
    "RWTOP2026-PGRFNET": "https://openreview.net/forum?id=3hS7EtL4bV",
    "RWTOP2026-GTM": "https://openreview.net/forum?id=PWM6FERWz9",
    "RWTOP2026-CSLSTM": "https://openreview.net/forum?id=2VtveTkmzW",
    "RWTOP2026-PROTOTS": "https://openreview.net/forum?id=IbcdVwzLrp",
    "RWTOP2026-CALTSFM": "https://openreview.net/forum?id=nGBN7UjHcy",
    "RWTOP2026-TSPULSE": "https://openreview.net/forum?id=Kw2mvnzCoc",
}

EVIDENCE_READY_STATUSES = frozenset(
    {
        "accepted_gpu_and_artifacts",
        "accepted_exact_artifacts",
        "accepted_representative_artifacts",
    }
)
ALLOWED_ACCEPTED_POOL_YEARS = frozenset({"2024", "2025", "2026"})
ALLOWED_ACCEPTED_POOL_VENUE_TIERS = frozenset({"top-conference", "top-journal"})
ALLOWED_ACCEPTED_POOL_STATUSES = frozenset(
    {
        "exact-runnable",
        "representative-runnable",
        "literature-only",
        "resource-blocked",
        "blocked",
    }
)


@dataclass(frozen=True)
class PaperRecentWorkCoverage:
    paper: str
    top_ids: Tuple[str, ...]
    top_count: int
    has_2026: bool
    runnable_minimum: str
    policy_ready: bool


@dataclass(frozen=True)
class MatrixRecentWorkCoverage:
    paper_id: str
    matrix_path: str
    top_ids: Tuple[str, ...]
    top_count: int
    has_2026: bool
    unknown_ids: Tuple[str, ...]
    missing_exact_status_ids: Tuple[str, ...]
    unscoped_exact_claim_ids: Tuple[str, ...]
    policy_ready: bool


@dataclass(frozen=True)
class TopRepresentativeBinding:
    binding_id: str
    paper_id: str
    external_work_id: str
    status: str
    exact_reproduction_status: str
    local_proxy_matrix_entries: Tuple[str, ...]
    evidence_ready: bool
    representative_only: bool


@dataclass(frozen=True)
class RecentWorkGateReport:
    ready: bool
    policy_ready: bool
    evidence_ready: bool
    source_verification_ready: bool
    accepted_pool_rows: int
    accepted_pool_ids: Tuple[str, ...]
    top_2026_ids: Tuple[str, ...]
    low_tier_violations: Tuple[str, ...]
    per_paper_coverage: Tuple[PaperRecentWorkCoverage, ...]
    matrix_coverage: Tuple[MatrixRecentWorkCoverage, ...]
    bindings: Tuple[TopRepresentativeBinding, ...]
    policy_blockers: Tuple[str, ...]
    evidence_blockers: Tuple[str, ...]
    blockers: Tuple[str, ...]


def _section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing section: {heading}")
    rest = text[start + len(marker) :]
    next_heading = rest.find("\n## ")
    if next_heading >= 0:
        return rest[:next_heading]
    return rest


def _markdown_table_rows(section: str) -> Tuple[Tuple[str, ...], ...]:
    rows: List[Tuple[str, ...]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = tuple(cell.strip() for cell in stripped.strip("|").split("|"))
        if not cells or not cells[0]:
            continue
        first_cell = cells[0].strip()
        if first_cell in {"ID", "Paper"}:
            continue
        if set(first_cell) <= {"-", ":"}:
            continue
        rows.append(cells)
    return tuple(rows)


def _extract_top_ids(text: str) -> Tuple[str, ...]:
    return tuple(re.findall(r"RWTOP20\d{2}-[A-Z0-9]+", text))


def _coded_status(cell: str) -> str:
    match = re.search(r"`([^`]+)`", cell)
    if match:
        return match.group(1).strip()
    return cell.strip()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _accepted_pool(text: str) -> Tuple[Tuple[str, ...], ...]:
    section = _section(text, "Accepted TOP Method Pool")
    return tuple(row for row in _markdown_table_rows(section) if row[0].startswith("RWTOP20"))


def _per_paper_coverage(text: str) -> Tuple[PaperRecentWorkCoverage, ...]:
    section = _section(text, "Per-Paper TOP-Source Minimums")
    coverages: List[PaperRecentWorkCoverage] = []
    for row in _markdown_table_rows(section):
        if len(row) < 3:
            continue
        top_ids = _extract_top_ids(row[1])
        has_2026 = any(top_id.startswith("RWTOP2026-") for top_id in top_ids)
        runnable_minimum = row[2]
        coverages.append(
            PaperRecentWorkCoverage(
                paper=row[0],
                top_ids=top_ids,
                top_count=len(top_ids),
                has_2026=has_2026,
                runnable_minimum=runnable_minimum,
                policy_ready=len(top_ids) >= 3 and has_2026 and bool(runnable_minimum),
            )
        )
    return tuple(coverages)


def _top_bindings(queue_path: Path) -> Tuple[TopRepresentativeBinding, ...]:
    queue = _load_yaml(queue_path)
    bindings: List[TopRepresentativeBinding] = []
    for item in queue.get("top_representative_bindings", []):
        status = str(item.get("status", ""))
        exact_status = str(item.get("exact_reproduction_status", ""))
        representative_only = (
            "not exact" in exact_status.lower()
            or "evaluation protocol only" in exact_status.lower()
            or "representative" in exact_status.lower()
        )
        bindings.append(
            TopRepresentativeBinding(
                binding_id=str(item.get("binding_id", "")),
                paper_id=str(item.get("paper_id", "")),
                external_work_id=str(item.get("external_work_id", "")),
                status=status,
                exact_reproduction_status=exact_status,
                local_proxy_matrix_entries=tuple(
                    str(entry) for entry in item.get("local_proxy_matrix_entries", [])
                ),
                evidence_ready=status in EVIDENCE_READY_STATUSES,
                representative_only=representative_only,
            )
        )
    return tuple(bindings)


def _matrix_recent_work_coverage(
    matrix_root: Path,
    accepted_ids: Sequence[str],
) -> Tuple[MatrixRecentWorkCoverage, ...]:
    accepted_id_set = set(accepted_ids)
    coverages: List[MatrixRecentWorkCoverage] = []
    for matrix_path in sorted(matrix_root.glob("*/submission_prep/baseline_ablation_matrix.yaml")):
        matrix = _load_yaml(matrix_path)
        top_items = tuple(
            item
            for item in matrix.get("top_recent_work", [])
            if isinstance(item, Mapping) and item.get("id")
        )
        top_ids = tuple(str(item.get("id", "")) for item in top_items)
        unknown_ids = tuple(top_id for top_id in top_ids if top_id not in accepted_id_set)
        has_2026 = any(top_id.startswith("RWTOP2026-") for top_id in top_ids)
        missing_exact_status_ids: List[str] = []
        unscoped_exact_claim_ids: List[str] = []
        for item in top_items:
            top_id = str(item.get("id", ""))
            exact_status = str(item.get("exact_reproduction_status", "")).strip()
            exact_status_lower = exact_status.lower()
            if not exact_status:
                missing_exact_status_ids.append(top_id)
                continue
            has_exact_claim = "exact" in exact_status_lower
            accepted_exact = "accepted" in exact_status_lower and "exact" in exact_status_lower
            scoped_as_non_exact = any(
                marker in exact_status_lower
                for marker in (
                    "not exact",
                    "representative",
                    "pending",
                    "resource-blocked",
                    "blocked",
                    "feasibility",
                    "until",
                )
            )
            if has_exact_claim and not accepted_exact and not scoped_as_non_exact:
                unscoped_exact_claim_ids.append(top_id)
        coverages.append(
            MatrixRecentWorkCoverage(
                paper_id=str(matrix.get("paper_id", matrix_path.parent.parent.name)),
                matrix_path=str(matrix_path),
                top_ids=top_ids,
                top_count=len(top_ids),
                has_2026=has_2026,
                unknown_ids=unknown_ids,
                missing_exact_status_ids=tuple(missing_exact_status_ids),
                unscoped_exact_claim_ids=tuple(unscoped_exact_claim_ids),
                policy_ready=(
                    len(top_ids) >= 3
                    and has_2026
                    and not unknown_ids
                    and not missing_exact_status_ids
                    and not unscoped_exact_claim_ids
                ),
            )
        )
    return tuple(coverages)


def evaluate_recent_work_gate(
    recent_work_readme: Path = DEFAULT_RECENT_WORK_README,
    queue_path: Path = DEFAULT_QUEUE,
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
) -> RecentWorkGateReport:
    text = recent_work_readme.read_text(encoding="utf-8")
    accepted_pool_section = _section(text, "Accepted TOP Method Pool")
    try:
        source_verification_section = _section(text, "Live Source Verification")
    except ValueError:
        source_verification_section = ""
    accepted_rows = _accepted_pool(text)
    accepted_ids = tuple(row[0] for row in accepted_rows)
    accepted_status_by_id = {
        row[0]: _coded_status(row[6]) for row in accepted_rows if len(row) > 6
    }
    top_2026_ids = tuple(top_id for top_id in accepted_ids if top_id.startswith("RWTOP2026-"))
    low_tier_violations = tuple(
        marker for marker in LOW_TIER_MARKERS if marker in accepted_pool_section
    )
    per_paper = _per_paper_coverage(text)
    matrix_coverage = _matrix_recent_work_coverage(matrix_root, accepted_ids)
    bindings = _top_bindings(queue_path)

    policy_blockers: List[str] = []
    source_verification_blockers: List[str] = []
    if not source_verification_section:
        source_verification_blockers.append("missing Live Source Verification section")
    else:
        if "2026-05-14" not in source_verification_section:
            source_verification_blockers.append(
                "Live Source Verification section lacks check date 2026-05-14"
            )
        if (
            "does not make any TOP representative `evidence_ready`"
            not in source_verification_section
        ):
            source_verification_blockers.append(
                "Live Source Verification section must state source checks do not make "
                "TOP representatives evidence_ready"
            )
        for top_id, url in REQUIRED_2026_SOURCE_URLS.items():
            if url not in source_verification_section:
                source_verification_blockers.append(
                    f"Live Source Verification missing primary URL for {top_id}: {url}"
                )
    policy_blockers.extend(source_verification_blockers)
    if len(accepted_rows) < 10:
        policy_blockers.append("accepted TOP method pool has fewer than 10 entries")
    for row in accepted_rows:
        if len(row) < 7:
            policy_blockers.append(f"{row[0]}: accepted TOP pool row is incomplete")
            continue
        year = row[1].strip()
        venue_tier = row[2].strip("` ")
        initial_status = _coded_status(row[6])
        if year not in ALLOWED_ACCEPTED_POOL_YEARS:
            policy_blockers.append(
                f"{row[0]}: accepted TOP pool year {year!r} is outside 2024-2026"
            )
        if venue_tier not in ALLOWED_ACCEPTED_POOL_VENUE_TIERS:
            policy_blockers.append(
                f"{row[0]}: accepted TOP pool venue tier {venue_tier!r} is not top-tier"
            )
        if initial_status not in ALLOWED_ACCEPTED_POOL_STATUSES:
            policy_blockers.append(
                f"{row[0]}: unsupported reproduction status {initial_status!r}"
            )
    if low_tier_violations:
        policy_blockers.append(
            "accepted TOP method pool contains rejected low-tier markers: "
            + ", ".join(low_tier_violations)
        )
    if "ICLR 2026 Poster" not in accepted_pool_section:
        policy_blockers.append("accepted TOP method pool lacks ICLR 2026 Poster coverage")
    missing_2026 = tuple(
        top_id for top_id in REQUIRED_2026_TOP_IDS if top_id not in accepted_ids
    )
    if missing_2026:
        policy_blockers.append("missing required 2026 TOP IDs: " + ", ".join(missing_2026))
    if len(per_paper) != 7:
        policy_blockers.append(f"per-paper TOP minimum table has {len(per_paper)} rows, not 7")
    for coverage in per_paper:
        if coverage.top_count < 3:
            policy_blockers.append(f"{coverage.paper}: fewer than three TOP recent methods")
        if not coverage.has_2026:
            policy_blockers.append(f"{coverage.paper}: missing 2026 TOP method")
        if not coverage.runnable_minimum:
            policy_blockers.append(f"{coverage.paper}: missing runnable minimum")
    if len(matrix_coverage) != 7:
        policy_blockers.append(
            f"paper-local baseline/ablation matrices with TOP coverage: {len(matrix_coverage)}, not 7"
        )
    for coverage in matrix_coverage:
        if coverage.top_count < 3:
            policy_blockers.append(
                f"{coverage.paper_id}: matrix has fewer than three TOP recent methods"
            )
        if not coverage.has_2026:
            policy_blockers.append(f"{coverage.paper_id}: matrix missing 2026 TOP method")
        if coverage.unknown_ids:
            policy_blockers.append(
                f"{coverage.paper_id}: matrix IDs absent from accepted TOP pool: "
                + ", ".join(coverage.unknown_ids)
            )
        if coverage.missing_exact_status_ids:
            policy_blockers.append(
                f"{coverage.paper_id}: matrix TOP entries missing "
                "exact_reproduction_status: "
                + ", ".join(coverage.missing_exact_status_ids)
            )
        if coverage.unscoped_exact_claim_ids:
            policy_blockers.append(
                f"{coverage.paper_id}: matrix TOP entries claim exact reproduction "
                "without accepted exact artifacts or representative/resource scope: "
                + ", ".join(coverage.unscoped_exact_claim_ids)
            )

    queue_paper_ids = {
        str(item.get("paper_id", ""))
        for item in _load_yaml(queue_path).get("paper_queue", [])
    }
    binding_paper_ids = {binding.paper_id for binding in bindings}
    if len(bindings) != 7:
        policy_blockers.append(f"TOP representative binding table has {len(bindings)} rows, not 7")
    if queue_paper_ids and binding_paper_ids != queue_paper_ids:
        missing = sorted(queue_paper_ids - binding_paper_ids)
        extra = sorted(binding_paper_ids - queue_paper_ids)
        policy_blockers.append(f"TOP binding paper mismatch; missing={missing}, extra={extra}")
    for binding in bindings:
        if not binding.external_work_id.startswith("RWTOP2026-"):
            policy_blockers.append(
                f"{binding.binding_id}: external work is not a 2026 TOP representative"
            )
        if binding.external_work_id not in accepted_ids:
            policy_blockers.append(
                f"{binding.binding_id}: external work ID is absent from accepted pool"
            )
        else:
            pool_status = accepted_status_by_id.get(binding.external_work_id, "")
            if (
                "representative-runnable" not in pool_status
                and "exact-runnable" not in pool_status
            ):
                policy_blockers.append(
                    f"{binding.binding_id}: external work {binding.external_work_id} "
                    f"is {pool_status!r}, not runnable representative evidence"
                )
        if not binding.local_proxy_matrix_entries:
            policy_blockers.append(f"{binding.binding_id}: no local proxy matrix entries")
        if binding.status not in EVIDENCE_READY_STATUSES and binding.status != "pending_gpu_and_artifacts":
            policy_blockers.append(f"{binding.binding_id}: unsupported binding status {binding.status!r}")
        if binding.status == "pending_gpu_and_artifacts" and not binding.representative_only:
            policy_blockers.append(
                f"{binding.binding_id}: pending binding must not be described as exact evidence"
            )

    evidence_blockers = tuple(
        f"{binding.binding_id}: TOP representative artifacts are still {binding.status}"
        for binding in bindings
        if not binding.evidence_ready
    )
    policy_ready = not policy_blockers
    evidence_ready = policy_ready and not evidence_blockers
    ready = policy_ready and evidence_ready
    blockers = tuple(policy_blockers) + evidence_blockers
    return RecentWorkGateReport(
        ready=ready,
        policy_ready=policy_ready,
        evidence_ready=evidence_ready,
        source_verification_ready=not source_verification_blockers,
        accepted_pool_rows=len(accepted_rows),
        accepted_pool_ids=accepted_ids,
        top_2026_ids=top_2026_ids,
        low_tier_violations=low_tier_violations,
        per_paper_coverage=per_paper,
        matrix_coverage=matrix_coverage,
        bindings=bindings,
        policy_blockers=tuple(policy_blockers),
        evidence_blockers=evidence_blockers,
        blockers=blockers,
    )


def build_payload(report: RecentWorkGateReport) -> Mapping[str, Any]:
    return asdict(report)


def render_markdown(report: RecentWorkGateReport) -> str:
    lines = [
        "# UXFD Recent Work Gate",
        "",
        f"- Ready: `{report.ready}`",
        f"- Policy ready: `{report.policy_ready}`",
        f"- Evidence ready: `{report.evidence_ready}`",
        f"- Source verification ready: `{report.source_verification_ready}`",
        f"- Accepted TOP method rows: `{report.accepted_pool_rows}`",
        f"- 2026 TOP IDs: `{len(report.top_2026_ids)}`",
        f"- Low-tier violations: `{len(report.low_tier_violations)}`",
        f"- Paper-local matrix coverage rows: `{len(report.matrix_coverage)}`",
        f"- TOP representative bindings: `{len(report.bindings)}`",
        "",
        "| Paper | TOP Methods | Has 2026 | Runnable Minimum | Policy Ready |",
        "|---|---:|---:|---|---:|",
    ]
    for coverage in report.per_paper_coverage:
        runnable_minimum = coverage.runnable_minimum.replace("|", "\\|")
        lines.append(
            f"| `{coverage.paper}` | {coverage.top_count} | `{coverage.has_2026}` | "
            f"{runnable_minimum} | `{coverage.policy_ready}` |"
        )
    lines.extend(
        [
            "",
            "## Paper-Local Matrix Coverage",
            "",
            "| Paper ID | TOP Methods | Has 2026 | Unknown IDs | Exact Status Issues | Policy Ready |",
            "|---|---:|---:|---|---|---:|",
        ]
    )
    for coverage in report.matrix_coverage:
        unknown_ids = ", ".join(coverage.unknown_ids) if coverage.unknown_ids else "-"
        exact_issues = []
        if coverage.missing_exact_status_ids:
            exact_issues.append("missing=" + ",".join(coverage.missing_exact_status_ids))
        if coverage.unscoped_exact_claim_ids:
            exact_issues.append("unscoped=" + ",".join(coverage.unscoped_exact_claim_ids))
        exact_issue_text = "; ".join(exact_issues) if exact_issues else "-"
        lines.append(
            f"| `{coverage.paper_id}` | {coverage.top_count} | `{coverage.has_2026}` | "
            f"{unknown_ids} | {exact_issue_text} | `{coverage.policy_ready}` |"
        )
    lines.extend(
        [
            "",
            "## TOP Representative Bindings",
            "",
            "| Binding | Paper | Work | Local Proxy Entries | Exact Reproduction Status | Status | Evidence Ready |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for binding in report.bindings:
        proxy_entries = ", ".join(binding.local_proxy_matrix_entries)
        exact_status = binding.exact_reproduction_status.replace("|", "\\|")
        lines.append(
            f"| `{binding.binding_id}` | `{binding.paper_id}` | `{binding.external_work_id}` | "
            f"`{proxy_entries}` | {exact_status} | `{binding.status}` | "
            f"`{binding.evidence_ready}` |"
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.blockers:
        lines.append(f"- {blocker}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate UXFD TOP recent-work readiness")
    parser.add_argument("--recent-work-readme", type=Path, default=DEFAULT_RECENT_WORK_README)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_recent_work_gate(args.recent_work_readme, args.queue, args.matrix_root)
    if args.format == "json":
        output = json.dumps(build_payload(report), indent=2) + "\n"
    else:
        output = render_markdown(report)

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
