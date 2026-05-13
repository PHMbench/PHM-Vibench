from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Iterable, List, Mapping, Optional, Sequence, Tuple


PAPER_SUBMODULES = (
    Path("paper/UXFD_paper/Explainable_FD_Toolkit"),
    Path("paper/UXFD_paper/1D-2D_fusion_explainable"),
    Path("paper/UXFD_paper/LLM_Explainable_FD_Toolkit"),
    Path("paper/UXFD_paper/MOE_explainable"),
    Path("paper/UXFD_paper/Paper_fuzzy_XFD"),
    Path("paper/UXFD_paper/Neuralsymbolic_theory"),
    Path("paper/UXFD_paper/TII_operator_attention"),
)

DO_NOT_AUTO_COMMIT = "do_not_auto_commit_without_owner_review"
PRESERVE_SESSION = "preserve_or_ignore_session_workspace"
PROMOTE_ONLY_THROUGH_GATE = "promote_only_through_accepted_artifact_gate"
TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
BINARY_OR_LARGE_SUFFIXES = {".npy", ".npz", ".pth", ".png", ".pdf"}
OWNER_REVIEW_RECOMMENDATIONS = Path(
    "paper/UXFD_paper/results/submodule_owner_review_recommendations.md"
)
OWNER_REVIEW_ACTION_PACKET = Path(
    "paper/UXFD_paper/results/submodule_owner_review_action_packet.md"
)
OWNER_REVIEW_EVIDENCE_INDEX = Path(
    "paper/UXFD_paper/results/submodule_owner_review_evidence_index.md"
)
OWNER_REVIEW_DECISION_TEMPLATE = Path(
    "paper/UXFD_paper/results/submodule_owner_review_decisions.template.json"
)
OWNER_ALLOWED_DECISIONS = (
    "commit_after_review",
    "rewrite_then_commit",
    "discard_from_submodule",
)


@dataclass(frozen=True)
class DirtyEntry:
    submodule: str
    status: str
    path: str
    category: str
    recommended_action: str
    risk_markers: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DirtySubmoduleSummary:
    submodule: str
    total: int
    modified: int
    untracked: int
    categories: Mapping[str, int]


@dataclass(frozen=True)
class DirtyTriageReport:
    clean: bool
    summaries: Tuple[DirtySubmoduleSummary, ...]
    entries: Tuple[DirtyEntry, ...]


def _classify_path(path: str) -> Tuple[str, str]:
    if path.startswith((".agent/", ".claude/", ".codex/")):
        return "agent_workspace", PRESERVE_SESSION
    if path.startswith("sessions/"):
        return "session_workspace", PRESERVE_SESSION
    if path.endswith("AUTORESEARCH_EVIDENCE.md"):
        return "historical_autoresearch_evidence_draft", DO_NOT_AUTO_COMMIT
    if path == "doc/demo_explanation.txt":
        return "generated_or_result_artifact", PROMOTE_ONLY_THROUGH_GATE
    if path.startswith(("outputs/", "results/", "benchmark_results/", "autoresearch/")):
        return "experiment_output", PROMOTE_ONLY_THROUGH_GATE
    if path.endswith((".log", ".npy", ".npz", ".pth", ".png", ".pdf", ".csv", ".json")):
        return "generated_or_result_artifact", PROMOTE_ONLY_THROUGH_GATE
    if path.startswith("plan/") or path in {
        "EXPERIMENT_DESIGN.md",
        "innovation_contract.md",
        "paper_blueprint.md",
        "program.md",
    }:
        return "planning_or_contract_draft", DO_NOT_AUTO_COMMIT
    if path.startswith(("manuscript/", "paper_draft/")):
        return "manuscript_draft", DO_NOT_AUTO_COMMIT
    if path.startswith(("scripts/", "code/")):
        return "source_or_experiment_script", DO_NOT_AUTO_COMMIT
    if path in {"CORE.md", "README.md"} or path.endswith(".md"):
        return "project_document", DO_NOT_AUTO_COMMIT
    return "unclassified", DO_NOT_AUTO_COMMIT


def _content_risk_markers(submodule: Path, relative_path: str) -> Tuple[str, ...]:
    path = submodule / relative_path
    if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
        return ()

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ("unreadable_text_artifact",)

    markers: List[str] = []
    if "PHM-Vibench copy 2" in text:
        markers.append("stale_exec_root")
    if "--config_dir" in text or "config_dir:" in text:
        markers.append("deprecated_config_dir_dispatch")
    if (
        "Submission-Ready Binding Snapshot" in text
        or re.search(r"status:\s*`?ready`?", text, re.IGNORECASE)
        or "论文就绪" in text
        or "投稿状态" in text
        or "可直接用于论文" in text
    ):
        markers.append("unaccepted_readiness_claim")
    if "accepted: `True`" in text or "accepted: True" in text:
        markers.append("historical_accepted_claim")
    for match in re.finditer(r"CUDA_VISIBLE_DEVICES=([0-9,]+)", text):
        devices = {device.strip() for device in match.group(1).split(",") if device.strip()}
        if devices and not devices.issubset({"0", "1"}):
            markers.append("nonlocal_gpu_binding")
            break
    return tuple(markers)


def _path_risk_markers(status: str, relative_path: str, category: str) -> Tuple[str, ...]:
    markers: List[str] = []
    suffix = Path(relative_path).suffix.lower()

    if status.strip() != "??" and category in {
        "experiment_output",
        "generated_or_result_artifact",
    }:
        markers.append("tracked_generated_artifact_dirty")
    if suffix in BINARY_OR_LARGE_SUFFIXES:
        markers.append("binary_or_large_artifact")
    return tuple(markers)


def _git_status_entries(submodule: Path) -> Tuple[DirtyEntry, ...]:
    result = subprocess.run(
        ["git", "-C", str(submodule), "status", "--porcelain=v1", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    entries: List[DirtyEntry] = []
    for raw in result.stdout.split("\0"):
        if not raw:
            continue
        status = raw[:2]
        path = raw[3:] if len(raw) > 3 else ""
        category, action = _classify_path(path)
        risk_markers = _path_risk_markers(status, path, category) + _content_risk_markers(
            submodule,
            path,
        )
        entries.append(
            DirtyEntry(
                submodule=str(submodule),
                status=status.strip() or status,
                path=path,
                category=category,
                recommended_action=action,
                risk_markers=risk_markers,
            )
        )
    return tuple(entries)


def _summarize_entries(entries: Iterable[DirtyEntry]) -> Tuple[DirtySubmoduleSummary, ...]:
    grouped: DefaultDict[str, List[DirtyEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.submodule].append(entry)

    summaries = []
    for submodule in sorted(grouped):
        records = grouped[submodule]
        categories = Counter(entry.category for entry in records)
        summaries.append(
            DirtySubmoduleSummary(
                submodule=submodule,
                total=len(records),
                modified=sum(1 for entry in records if entry.status != "??"),
                untracked=sum(1 for entry in records if entry.status == "??"),
                categories=dict(sorted(categories.items())),
            )
        )
    return tuple(summaries)


def _action_counts(entries: Iterable[DirtyEntry]) -> Mapping[str, int]:
    return dict(sorted(Counter(entry.recommended_action for entry in entries).items()))


def _risk_marker_counts(entries: Iterable[DirtyEntry]) -> Mapping[str, int]:
    counter: Counter[str] = Counter()
    for entry in entries:
        counter.update(entry.risk_markers)
    return dict(sorted(counter.items()))


def _action_counts_by_submodule(entries: Iterable[DirtyEntry]) -> Mapping[str, Mapping[str, int]]:
    grouped: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        grouped[entry.submodule].update((entry.recommended_action,))
    return {
        submodule: dict(sorted(counter.items()))
        for submodule, counter in sorted(grouped.items())
    }


def _review_command(entry: DirtyEntry) -> str:
    base = f"git -C {shlex.quote(entry.submodule)}"
    path = shlex.quote(entry.path)
    if entry.status == "??":
        return f"{base} status --short -- {path}"
    return f"{base} diff -- {path}"


def _content_review_command(entry: DirtyEntry) -> str:
    if entry.status != "??" or Path(entry.path).suffix.lower() not in TEXT_SUFFIXES:
        return _review_command(entry)
    path = shlex.quote(str(Path(entry.submodule) / entry.path))
    return f"sed -n '1,220p' -- {path}"


def _recommended_owner_decisions(entry: DirtyEntry) -> Tuple[str, ...]:
    if entry.category == "historical_autoresearch_evidence_draft":
        return ("discard_from_submodule", "rewrite_then_commit")
    if entry.category == "planning_or_contract_draft" or entry.risk_markers:
        return ("rewrite_then_commit", "discard_from_submodule")
    return OWNER_ALLOWED_DECISIONS


def _owner_review_note(entry: DirtyEntry) -> str:
    markers = set(entry.risk_markers)
    notes: List[str] = []
    if "nonlocal_gpu_binding" in markers:
        notes.append("Rewrite nonlocal GPU references to local GPU 0,1 policy.")
    if "deprecated_config_dir_dispatch" in markers:
        notes.append(
            "Rewrite deprecated config_dir dispatch to maintained python main.py --config flow."
        )
    if (
        "unaccepted_readiness_claim" in markers
        or "historical_accepted_claim" in markers
    ):
        notes.append(
            "Remove or relabel historical readiness/accepted-evidence wording because "
            "current accepted_runs=0 and submission_ready=false gates still block the paper."
        )
    if "stale_exec_root" in markers:
        notes.append("Remove stale execution-root references before any commit.")
    if notes:
        return " ".join(notes)
    if entry.category == "planning_or_contract_draft":
        return "Useful planning draft only after current-root, parent-gated rewrite."
    return "TODO"


def _owner_decision_template(entries: Iterable[DirtyEntry]) -> Tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "submodule": entry.submodule,
            "path": entry.path,
            "current_status": entry.status,
            "category": entry.category,
            "risk_markers": list(entry.risk_markers),
            "recommended_decisions": list(_recommended_owner_decisions(entry)),
            "decision": "pending_owner_review",
            "reviewer": "TODO",
            "review_date": "TODO",
            "notes": _owner_review_note(entry),
        }
        for entry in entries
        if entry.recommended_action == DO_NOT_AUTO_COMMIT
    )


def _owner_review_packets(entries: Iterable[DirtyEntry]) -> Tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "submodule": entry.submodule,
            "path": entry.path,
            "status": entry.status,
            "category": entry.category,
            "risk_markers": list(entry.risk_markers),
            "review_command": _review_command(entry),
            "content_review_command": _content_review_command(entry),
            "decision_state": "pending_owner_review",
            "allowed_decisions": list(OWNER_ALLOWED_DECISIONS),
            "recommended_decisions": list(_recommended_owner_decisions(entry)),
            "default_next_action": (
                "paper owner must choose an allowed decision before this entry is "
                "staged, rewritten, or cleaned up"
            ),
        }
        for entry in entries
        if entry.recommended_action == DO_NOT_AUTO_COMMIT
    )


def _owner_resolution_gates() -> Tuple[Mapping[str, str], ...]:
    return (
        {
            "decision": "commit_after_review",
            "required_gate": (
                "owner confirms the file is intentional source/docs and removes or "
                "justifies stale exec roots, deprecated config dispatch, nonlocal GPU "
                "bindings, readiness claims, and historical accepted-claim wording"
            ),
        },
        {
            "decision": "rewrite_then_commit",
            "required_gate": (
                "owner rewrites the file, reruns dirty triage, and records why any "
                "remaining risk marker is acceptable before staging"
            ),
        },
        {
            "decision": "discard_from_submodule",
            "required_gate": (
                "owner explicitly marks the untracked draft or generated change as "
                "discardable; do not delete it automatically from this triage"
            ),
        },
    )


def _owner_recommendations_payload() -> Mapping[str, Any]:
    return {
        "path": str(OWNER_REVIEW_RECOMMENDATIONS),
        "exists": OWNER_REVIEW_RECOMMENDATIONS.is_file(),
        "action_packet_path": str(OWNER_REVIEW_ACTION_PACKET),
        "action_packet_exists": OWNER_REVIEW_ACTION_PACKET.is_file(),
        "evidence_index_path": str(OWNER_REVIEW_EVIDENCE_INDEX),
        "evidence_index_exists": OWNER_REVIEW_EVIDENCE_INDEX.is_file(),
        "decision_template_path": str(OWNER_REVIEW_DECISION_TEMPLATE),
        "decision_template_exists": OWNER_REVIEW_DECISION_TEMPLATE.is_file(),
        "status": "decision_support_only",
        "required_use": (
            "paper owners should read the action packet, recommendation note, and "
            "line-level evidence index before "
            "choosing commit_after_review, rewrite_then_commit, or discard_from_submodule"
        ),
    }


def evaluate_dirty_triage(
    submodules: Sequence[Path] = PAPER_SUBMODULES,
) -> DirtyTriageReport:
    entries: List[DirtyEntry] = []
    for submodule in submodules:
        entries.extend(_git_status_entries(submodule))
    entry_tuple = tuple(entries)
    return DirtyTriageReport(
        clean=not entry_tuple,
        summaries=_summarize_entries(entry_tuple),
        entries=entry_tuple,
    )


def build_payload(report: DirtyTriageReport) -> Mapping[str, Any]:
    payload = dict(asdict(report))
    payload["action_counts"] = _action_counts(report.entries)
    payload["risk_marker_counts"] = _risk_marker_counts(report.entries)
    payload["owner_decision_template"] = _owner_decision_template(report.entries)
    payload["owner_review_packets"] = _owner_review_packets(report.entries)
    payload["owner_resolution_gates"] = _owner_resolution_gates()
    payload["owner_review_recommendations"] = _owner_recommendations_payload()
    return json.loads(json.dumps(payload))


def render_markdown(report: DirtyTriageReport) -> str:
    action_counts = _action_counts(report.entries)
    action_counts_by_submodule = _action_counts_by_submodule(report.entries)
    risk_counts = _risk_marker_counts(report.entries)
    auto_commit_safe = sum(
        1
        for entry in report.entries
        if entry.recommended_action
        not in {DO_NOT_AUTO_COMMIT, PRESERVE_SESSION, PROMOTE_ONLY_THROUGH_GATE}
    )
    lines = [
        "# UXFD Submodule Dirty Triage",
        "",
        "Status: blocker triage only. This report is not accepted experiment evidence.",
        "",
        f"- Clean: `{report.clean}`",
        f"- Dirty entries: `{len(report.entries)}`",
        "",
        "## Summary",
        "",
        "| Submodule | Total | Modified | Untracked | Categories |",
        "|---|---:|---:|---:|---|",
    ]
    for summary in report.summaries:
        categories = ", ".join(f"{name}={count}" for name, count in summary.categories.items())
        lines.append(
            f"| `{summary.submodule}` | {summary.total} | {summary.modified} | "
            f"{summary.untracked} | {categories} |"
        )

    action_summary = ", ".join(f"{name}={count}" for name, count in action_counts.items())
    risk_summary = ", ".join(f"{name}={count}" for name, count in risk_counts.items())
    lines.extend(
        [
            "",
            "## Commit-Blocking Verdict",
            "",
            f"- Auto-commit safe entries: `{auto_commit_safe}`",
            f"- Action counts: `{action_summary or '-'}`",
            f"- Risk marker counts: `{risk_summary or '-'}`",
            "- Verdict: do not auto-commit these dirty submodule entries. Commit only owner-reviewed source/docs, and promote result artifacts only through the accepted artifact gate.",
            "",
            "## Triage Rules",
            "",
            f"- `{PRESERVE_SESSION}`: preserve or ignore until the owning paper owner decides.",
            f"- `{PROMOTE_ONLY_THROUGH_GATE}`: do not commit as accepted evidence; promote only through `scripts.uxfd_artifact_gate` after real runs.",
            f"- `{DO_NOT_AUTO_COMMIT}`: inspect with the paper owner before staging.",
            "- Risk markers flag stale paths, deprecated config dispatch, unaccepted readiness claims, historical accepted-claim wording, or GPU bindings outside `0,1`.",
            "",
            "## Owner Review Queue",
            "",
            "Use this queue to resolve the dirty-submodule blocker without promoting generated artifacts as accepted evidence.",
            "",
            "| Submodule | Owner-review entries | Artifact-gate-only entries | Preserve/ignore entries | First non-destructive check |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for summary in report.summaries:
        counts = action_counts_by_submodule.get(summary.submodule, {})
        lines.append(
            f"| `{summary.submodule}` | {counts.get(DO_NOT_AUTO_COMMIT, 0)} | "
            f"{counts.get(PROMOTE_ONLY_THROUGH_GATE, 0)} | "
            f"{counts.get(PRESERVE_SESSION, 0)} | "
            f"`git -C {summary.submodule} status --short` |"
        )

    recommendations = _owner_recommendations_payload()
    lines.extend(
        [
            "",
            "## Owner Review Recommendations",
            "",
            f"- Decision-support report: `{recommendations['path']}`",
            f"- Exists: `{recommendations['exists']}`",
            f"- Owner action packet: `{recommendations['action_packet_path']}`",
            f"- Action packet exists: `{recommendations['action_packet_exists']}`",
            f"- Evidence index: `{recommendations['evidence_index_path']}`",
            f"- Evidence index exists: `{recommendations['evidence_index_exists']}`",
            f"- Machine-readable decision template: `{recommendations['decision_template_path']}`",
            f"- Template exists: `{recommendations['decision_template_exists']}`",
            f"- Status: `{recommendations['status']}`",
            f"- Required use: {recommendations['required_use']}.",
        ]
    )

    owner_entries = [
        entry for entry in report.entries if entry.recommended_action == DO_NOT_AUTO_COMMIT
    ]
    artifact_entries = [
        entry
        for entry in report.entries
        if entry.recommended_action == PROMOTE_ONLY_THROUGH_GATE
    ]
    lines.extend(
        [
            "",
            "## Owner-Review Entry Checklist",
            "",
            "These entries require an explicit paper-owner decision before any staging.",
            "Allowed decisions: `commit_after_review`, `rewrite_then_commit`, or `discard_from_submodule`.",
            "",
            "| Submodule | Status | Category | Risk Markers | Review Command | Path |",
            "|---|---|---|---|---|---|",
        ]
    )
    for entry in owner_entries:
        markers = ", ".join(entry.risk_markers) if entry.risk_markers else "-"
        lines.append(
            f"| `{entry.submodule}` | `{entry.status}` | `{entry.category}` | "
            f"`{markers}` | `{_review_command(entry)}` | `{entry.path}` |"
        )

    lines.extend(
        [
            "",
            "## Owner Decision Template",
            "",
            "Copy these rows into a paper-owner review note before staging any owner-review entry.",
            "The default `pending_owner_review` value is intentionally not commit-safe.",
            "",
            "| Submodule | Path | Current Status | Category | Risk Markers | Recommended Decisions | Decision | Reviewer | Review Date | Notes |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for entry in owner_entries:
        markers = ", ".join(entry.risk_markers) if entry.risk_markers else "-"
        recommended = ", ".join(_recommended_owner_decisions(entry))
        lines.append(
            f"| `{entry.submodule}` | `{entry.path}` | `{entry.status}` | "
            f"`{entry.category}` | `{markers}` | `{recommended}` | "
            f"`pending_owner_review` | `TODO` | `TODO` | {_owner_review_note(entry)} |"
        )

    lines.extend(
        [
            "",
            "## Owner Review Packets",
            "",
            "Each packet is also emitted in `submodule_dirty_triage.json` for automation.",
            "",
            "| Submodule | Path | Decision State | Risk Markers | Status Command | Content Review Command | Default next action |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for packet in _owner_review_packets(report.entries):
        markers = ", ".join(packet["risk_markers"]) if packet["risk_markers"] else "-"
        lines.append(
            f"| `{packet['submodule']}` | `{packet['path']}` | "
            f"`{packet['decision_state']}` | `{markers}` | "
            f"`{packet['review_command']}` | "
            f"`{packet['content_review_command']}` | {packet['default_next_action']} |"
        )

    lines.extend(
        [
            "",
            "## Owner Resolution Gates",
            "",
            "These gates define when an owner-review entry may stop blocking the parent handoff.",
            "",
            "| Decision | Required gate before staging or cleanup |",
            "|---|---|",
        ]
    )
    for gate in _owner_resolution_gates():
        lines.append(
            f"| `{gate['decision']}` | {gate['required_gate']} |"
        )

    lines.extend(
        [
            "",
            "## Artifact-Gate Promotion Checklist",
            "",
            "These entries must not be committed as accepted evidence. Recreate or promote them only through `paper/UXFD_paper/results/accepted_runs` after real Q0-passed runs.",
            "",
            "| Submodule | Status | Category | Risk Markers | Review Command | Path |",
            "|---|---|---|---|---|---|",
        ]
    )
    for entry in artifact_entries:
        markers = ", ".join(entry.risk_markers) if entry.risk_markers else "-"
        lines.append(
            f"| `{entry.submodule}` | `{entry.status}` | `{entry.category}` | "
            f"`{markers}` | `{_review_command(entry)}` | `{entry.path}` |"
        )

    lines.extend(
        [
            "",
            "## Entries",
            "",
            "| Submodule | Status | Category | Action | Risk Markers | Path |",
            "|---|---|---|---|---|---|",
        ]
    )
    for entry in report.entries:
        markers = ", ".join(entry.risk_markers) if entry.risk_markers else "-"
        lines.append(
            f"| `{entry.submodule}` | `{entry.status}` | `{entry.category}` | "
            f"`{entry.recommended_action}` | `{markers}` | `{entry.path}` |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Triage dirty UXFD paper submodules")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = evaluate_dirty_triage()
    if args.format == "json":
        output = json.dumps(build_payload(report), indent=2) + "\n"
    else:
        output = render_markdown(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if report.clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
