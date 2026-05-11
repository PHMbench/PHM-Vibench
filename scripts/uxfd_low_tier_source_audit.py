from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple


DEFAULT_ROOT = Path("paper/UXFD_paper")

DIRECT_MARKERS = (
    "Scientific Reports",
    "MDPI",
    "IEEE TIM",
    "IEEE Transactions on Instrumentation and Measurement",
    "IEEE Access",
)

EXACT_JOURNAL_MARKERS = (
    "Applied Sciences",
    "Electronics",
    "Sensors",
    "Mathematics",
)

TEXT_SUFFIXES = {".bib", ".bbl", ".md", ".tex", ".txt", ".rst"}
EXCLUDED_PARTS = {
    ".agent",
    ".claude",
    ".codex",
    "accepted_run_templates",
    "autoresearch",
    "goal",
    "outputs",
    "results",
    "sessions",
}
BLOCKER_PARTS = {"manuscript", "paper_draft"}
BLOCKER_NAMES = {"ref.bib", "references.bib"}


@dataclass(frozen=True)
class LowTierFinding:
    paper_id: str
    path: str
    line: int
    marker: str
    severity: str
    text: str


@dataclass(frozen=True)
class LowTierSourceAuditReport:
    ready: bool
    root: str
    findings: Tuple[LowTierFinding, ...]
    blockers: Tuple[str, ...]
    blocker_count: int
    triage_count: int


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def _paper_id(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except (IndexError, ValueError):
        return ""


def _severity(root: Path, path: Path) -> str:
    try:
        relative_parts = set(path.relative_to(root).parts)
    except ValueError:
        relative_parts = set(path.parts)
    if path.name in BLOCKER_NAMES or path.suffix == ".bbl":
        return "blocker"
    if relative_parts & BLOCKER_PARTS:
        return "blocker"
    return "triage"


def _exact_journal_marker(line: str) -> str:
    for marker in EXACT_JOURNAL_MARKERS:
        pattern = rf"\bjournal\s*=\s*[{{\"]\s*{re.escape(marker)}\s*[}}\",]"
        if re.search(pattern, line, flags=re.IGNORECASE):
            return marker
    return ""


def _matched_marker(line: str) -> str:
    for marker in DIRECT_MARKERS:
        if re.search(rf"\b{re.escape(marker)}\b", line, flags=re.IGNORECASE):
            return marker
    return _exact_journal_marker(line)


def _iter_text_files(root: Path) -> Tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in TEXT_SUFFIXES
            and not _is_excluded(path.relative_to(root))
        )
    )


def evaluate_low_tier_source_audit(root: Path = DEFAULT_ROOT) -> LowTierSourceAuditReport:
    findings: List[LowTierFinding] = []
    for path in _iter_text_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for index, line in enumerate(lines, start=1):
            marker = _matched_marker(line)
            if not marker:
                continue
            finding = LowTierFinding(
                paper_id=_paper_id(root, path),
                path=str(path),
                line=index,
                marker=marker,
                severity=_severity(root, path),
                text=line.strip()[:220],
            )
            findings.append(finding)

    blockers = tuple(
        f"{finding.path}:{finding.line}: {finding.marker}"
        for finding in findings
        if finding.severity == "blocker"
    )
    triage_count = sum(1 for finding in findings if finding.severity == "triage")
    return LowTierSourceAuditReport(
        ready=not blockers,
        root=str(root),
        findings=tuple(findings),
        blockers=blockers,
        blocker_count=len(blockers),
        triage_count=triage_count,
    )


def build_payload(report: LowTierSourceAuditReport) -> Mapping[str, Any]:
    return asdict(report)


def render_markdown(report: LowTierSourceAuditReport) -> str:
    lines = [
        "# UXFD Low-Tier Source Audit",
        "",
        "Status: source-hygiene triage only. This report is not citation replacement evidence.",
        "",
        f"- Ready: `{report.ready}`",
        f"- Root: `{report.root}`",
        f"- Findings: `{len(report.findings)}`",
        f"- Blockers: `{report.blocker_count}`",
        f"- Triage-only findings: `{report.triage_count}`",
        "",
        "| Severity | Paper | Marker | Location | Text |",
        "|---|---|---|---|---|",
    ]
    for finding in report.findings:
        text = finding.text.replace("|", "\\|")
        location = f"{finding.path}:{finding.line}"
        lines.append(
            f"| `{finding.severity}` | `{finding.paper_id}` | `{finding.marker}` | "
            f"`{location}` | {text} |"
        )
    lines.extend(["", "## Blockers"])
    if report.blockers:
        lines.append("")
    for blocker in report.blockers:
        lines.append(f"- {blocker}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit UXFD low-tier citation/source markers")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_low_tier_source_audit(args.root)
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
