from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.uxfd_submodule_dirty_triage import (
    DO_NOT_AUTO_COMMIT,
    OWNER_REVIEW_ACTION_PACKET,
    OWNER_REVIEW_DECISION_TEMPLATE,
    OWNER_REVIEW_EVIDENCE_INDEX,
    OWNER_REVIEW_RECOMMENDATIONS,
    evaluate_dirty_triage,
)


DEFAULT_DECISION_FILE = Path("paper/UXFD_paper/results/submodule_owner_review_decisions.json")
ALLOWED_DECISIONS = frozenset(
    {
        "commit_after_review",
        "rewrite_then_commit",
        "discard_from_submodule",
    }
)
PENDING_DECISION = "pending_owner_review"
APPROVED_DECISION_STATUS = "owner_review_decisions"
TEMPLATE_STATUS = "template_only_not_owner_approved"
PLACEHOLDER_REVIEWERS = frozenset({"todo", "paper-owner", "owner", "reviewer"})


@dataclass(frozen=True)
class OwnerReviewRecord:
    submodule: str
    path: str
    decision: str
    reviewer: str
    review_date: str
    issues: Tuple[str, ...]


@dataclass(frozen=True)
class OwnerReviewGateReport:
    ready: bool
    decision_file: str
    template_file: str
    source_path: str
    source_is_template: bool
    expected_records: int
    records: Tuple[OwnerReviewRecord, ...]
    pending_records: int
    approved_records: int
    blockers: Tuple[str, ...]


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _owner_review_entries() -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(_owner_review_meta()))


def _owner_review_meta() -> Mapping[Tuple[str, str], Mapping[str, Any]]:
    dirty = evaluate_dirty_triage()
    return {
        (entry.submodule, entry.path): {
            "current_status": entry.status,
            "category": entry.category,
            "risk_markers": tuple(entry.risk_markers),
        }
        for entry in dirty.entries
        if entry.recommended_action == DO_NOT_AUTO_COMMIT
    }


def _string_list(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _expected_supporting_files() -> Mapping[str, str]:
    return {
        "action_packet": str(OWNER_REVIEW_ACTION_PACKET),
        "recommendations": str(OWNER_REVIEW_RECOMMENDATIONS),
        "evidence_index": str(OWNER_REVIEW_EVIDENCE_INDEX),
    }


def _record_issues(
    record: Mapping[str, Any],
    current_meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, ...]:
    issues: List[str] = []
    decision = str(record.get("decision", "")).strip()
    reviewer = str(record.get("reviewer", "")).strip()
    review_date = str(record.get("review_date", "")).strip()
    recommended = set(_string_list(record.get("recommended_decisions")))
    risk_markers = _string_list(record.get("risk_markers"))
    if current_meta is not None:
        current_status = str(record.get("current_status", "")).strip()
        expected_status = str(current_meta.get("current_status", "")).strip()
        if current_status != expected_status:
            issues.append(
                "current_status does not match current dirty triage: "
                f"{current_status or '<missing>'} != {expected_status or '<missing>'}"
            )
        category = str(record.get("category", "")).strip()
        expected_category = str(current_meta.get("category", "")).strip()
        if category != expected_category:
            issues.append(
                "category does not match current dirty triage: "
                f"{category or '<missing>'} != {expected_category or '<missing>'}"
            )
        expected_risk_markers = tuple(
            str(item) for item in current_meta.get("risk_markers", ())
        )
        if risk_markers != expected_risk_markers:
            issues.append("risk_markers do not match current dirty triage")

    if decision == PENDING_DECISION:
        issues.append("decision is still pending_owner_review")
    elif decision not in ALLOWED_DECISIONS:
        issues.append(f"decision is not allowed: {decision or '<missing>'}")
    if recommended and not recommended.issubset(ALLOWED_DECISIONS):
        issues.append("recommended_decisions contains values outside the allowed set")
    if decision in ALLOWED_DECISIONS:
        reviewer_key = reviewer.strip().lower()
        if not reviewer or reviewer_key in PLACEHOLDER_REVIEWERS:
            issues.append("approved decision requires a non-TODO reviewer")
        try:
            parsed_review_date = date.fromisoformat(review_date)
        except ValueError:
            issues.append("approved decision requires an ISO YYYY-MM-DD review_date")
        else:
            if parsed_review_date > date.today():
                issues.append("approved decision review_date cannot be in the future")
        notes = str(record.get("notes", "")).strip()
        if decision == "commit_after_review" and risk_markers and (not notes or notes == "TODO"):
            issues.append("commit_after_review with risk markers requires notes")
    return tuple(issues)


def _records_from_payload(
    payload: Mapping[str, Any],
    current_meta_by_key: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
) -> Tuple[OwnerReviewRecord, ...]:
    raw_records = payload.get("records", ())
    if not isinstance(raw_records, list):
        return ()
    records: List[OwnerReviewRecord] = []
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            continue
        key = (str(raw.get("submodule", "")), str(raw.get("path", "")))
        current_meta = (
            current_meta_by_key.get(key)
            if current_meta_by_key is not None
            else None
        )
        records.append(
            OwnerReviewRecord(
                submodule=str(raw.get("submodule", "")),
                path=str(raw.get("path", "")),
                decision=str(raw.get("decision", "")),
                reviewer=str(raw.get("reviewer", "")),
                review_date=str(raw.get("review_date", "")),
                issues=_record_issues(raw, current_meta=current_meta),
            )
        )
    return tuple(records)


def _duplicate_keys(records: Iterable[OwnerReviewRecord]) -> Tuple[str, ...]:
    seen = set()
    duplicates: List[str] = []
    for record in records:
        key = (record.submodule, record.path)
        if key in seen:
            duplicates.append("|".join(key))
        seen.add(key)
    return tuple(sorted(duplicates))


def evaluate_owner_review_gate(
    decision_file: Path = DEFAULT_DECISION_FILE,
    template_file: Path = OWNER_REVIEW_DECISION_TEMPLATE,
) -> OwnerReviewGateReport:
    blockers: List[str] = []
    source_path = decision_file
    source_is_template = False
    if decision_file.exists():
        source_path = decision_file
    elif template_file.exists():
        source_path = template_file
        source_is_template = True
        blockers.append(f"owner decision file missing: {decision_file}")
    else:
        blockers.append(f"owner decision template missing: {template_file}")
        return OwnerReviewGateReport(
            ready=False,
            decision_file=str(decision_file),
            template_file=str(template_file),
            source_path=str(source_path),
            source_is_template=False,
            expected_records=len(_owner_review_entries()),
            records=(),
            pending_records=0,
            approved_records=0,
            blockers=tuple(blockers),
        )

    try:
        payload = _load_json(source_path)
    except json.JSONDecodeError as exc:
        blockers.append(f"{source_path}: invalid JSON: {exc.msg}")
        payload = {}

    allowed = set(_string_list(payload.get("allowed_decisions")))
    if allowed != ALLOWED_DECISIONS:
        blockers.append("allowed_decisions does not match owner-review policy")
    if payload.get("supporting_files") != _expected_supporting_files():
        blockers.append("supporting_files does not match owner-review support policy")
    payload_status = str(payload.get("status", "")).strip()
    if source_is_template and payload_status != TEMPLATE_STATUS:
        blockers.append("template source must be marked template_only_not_owner_approved")
    if not source_is_template and payload_status != APPROVED_DECISION_STATUS:
        blockers.append("owner decision file must be marked owner_review_decisions")

    expected_meta = _owner_review_meta()
    records = _records_from_payload(payload, current_meta_by_key=expected_meta)
    expected_keys = set(expected_meta)
    record_keys = {(record.submodule, record.path) for record in records}
    missing = sorted(expected_keys - record_keys)
    extra = sorted(record_keys - expected_keys)
    duplicates = _duplicate_keys(records)
    if missing:
        blockers.append(
            "decision records missing current owner-review entries: "
            + ",".join("|".join(key) for key in missing)
        )
    if extra:
        blockers.append(
            "decision records include non-current owner-review entries: "
            + ",".join("|".join(key) for key in extra)
        )
    if duplicates:
        blockers.append("decision records contain duplicates: " + ",".join(duplicates))

    pending_records = sum(record.decision == PENDING_DECISION for record in records)
    approved_records = sum(record.decision in ALLOWED_DECISIONS for record in records)
    record_issue_count = sum(len(record.issues) for record in records)
    if pending_records:
        blockers.append(f"{pending_records} owner-review decisions are still pending")
    if record_issue_count:
        blockers.append(f"{record_issue_count} owner-review record issues remain")
    if source_is_template:
        blockers.append("template file is not owner approval")

    return OwnerReviewGateReport(
        ready=not blockers and len(records) == len(expected_keys),
        decision_file=str(decision_file),
        template_file=str(template_file),
        source_path=str(source_path),
        source_is_template=source_is_template,
        expected_records=len(expected_keys),
        records=records,
        pending_records=pending_records,
        approved_records=approved_records,
        blockers=tuple(blockers),
    )


def build_payload(report: OwnerReviewGateReport) -> Mapping[str, Any]:
    return asdict(report)


def render_markdown(report: OwnerReviewGateReport) -> str:
    lines = [
        "# UXFD Owner Review Gate",
        "",
        "Status: owner-decision validation only. This report is not accepted experiment evidence.",
        "",
        f"- Ready: `{report.ready}`",
        f"- Source path: `{report.source_path}`",
        f"- Source is template: `{report.source_is_template}`",
        f"- Expected records: `{report.expected_records}`",
        f"- Records: `{len(report.records)}`",
        f"- Pending records: `{report.pending_records}`",
        f"- Approved records: `{report.approved_records}`",
        "",
        "## Blockers",
        "",
    ]
    if report.blockers:
        lines.extend(f"- {blocker}" for blocker in report.blockers)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Owner Decision Workflow",
            "",
            "This gate cannot approve the template by itself. Paper owners must:",
            "",
            f"1. Read `{OWNER_REVIEW_ACTION_PACKET}`, `{OWNER_REVIEW_RECOMMENDATIONS}`, and `{OWNER_REVIEW_EVIDENCE_INDEX}`, then inspect each dirty file before changing decisions.",
            f"2. Copy `{report.template_file}` to `{report.decision_file}` only after owner review is ready to record.",
            f"3. Change top-level `status` to `{APPROVED_DECISION_STATUS}`.",
            f"4. Replace every `{PENDING_DECISION}` with one allowed decision: "
            + ", ".join(f"`{decision}`" for decision in sorted(ALLOWED_DECISIONS))
            + ".",
            "5. Use a real reviewer name and ISO `YYYY-MM-DD` review date for every approved decision.",
            "6. Rerun `python -m scripts.uxfd_owner_review_gate`; do not stage, delete, or promote submodule files from the template alone.",
        ]
    )
    lines.extend(
        [
            "",
            "## Records",
            "",
            "| Submodule | Path | Decision | Reviewer | Review date | Issues |",
            "|---|---|---|---|---|---|",
        ]
    )
    for record in report.records:
        issues = ", ".join(record.issues) if record.issues else "-"
        lines.append(
            f"| `{record.submodule}` | `{record.path}` | `{record.decision}` | "
            f"`{record.reviewer}` | `{record.review_date}` | {issues} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate UXFD submodule owner-review decisions")
    parser.add_argument("--decision-file", type=Path, default=DEFAULT_DECISION_FILE)
    parser.add_argument("--template-file", type=Path, default=OWNER_REVIEW_DECISION_TEMPLATE)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_owner_review_gate(
        decision_file=args.decision_file,
        template_file=args.template_file,
    )
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
