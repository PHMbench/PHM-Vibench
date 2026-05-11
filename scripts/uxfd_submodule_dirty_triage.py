from __future__ import annotations

import argparse
import json
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


@dataclass(frozen=True)
class DirtyEntry:
    submodule: str
    status: str
    path: str
    category: str
    recommended_action: str


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
        entries.append(
            DirtyEntry(
                submodule=str(submodule),
                status=status.strip() or status,
                path=path,
                category=category,
                recommended_action=action,
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

    lines.extend(
        [
            "",
            "## Triage Rules",
            "",
            f"- `{PRESERVE_SESSION}`: preserve or ignore until the owning paper owner decides.",
            f"- `{PROMOTE_ONLY_THROUGH_GATE}`: do not commit as accepted evidence; promote only through `scripts.uxfd_artifact_gate` after real runs.",
            f"- `{DO_NOT_AUTO_COMMIT}`: inspect with the paper owner before staging.",
            "",
            "## Entries",
            "",
            "| Submodule | Status | Category | Action | Path |",
            "|---|---|---|---|---|",
        ]
    )
    for entry in report.entries:
        lines.append(
            f"| `{entry.submodule}` | `{entry.status}` | `{entry.category}` | "
            f"`{entry.recommended_action}` | `{entry.path}` |"
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
