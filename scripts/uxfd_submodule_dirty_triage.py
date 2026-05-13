from __future__ import annotations

import argparse
import json
import re
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
    return asdict(report)


def render_markdown(report: DirtyTriageReport) -> str:
    action_counts = _action_counts(report.entries)
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
