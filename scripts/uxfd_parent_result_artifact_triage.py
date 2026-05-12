from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SCAN_ROOTS = (Path("paper/UXFD_paper/results/figures"),)
DEFAULT_OUTPUT = Path("paper/UXFD_paper/results/parent_result_artifact_triage.md")
PROMOTE_ONLY_THROUGH_GATE = "promote_only_through_accepted_artifact_gate"


@dataclass(frozen=True)
class ParentArtifactEntry:
    path: str
    size_bytes: int
    category: str
    recommended_action: str


@dataclass(frozen=True)
class ParentArtifactTriageReport:
    clean: bool
    scan_roots: Tuple[str, ...]
    entries: Tuple[ParentArtifactEntry, ...]


def evaluate_parent_artifact_triage(
    scan_roots: Sequence[Path] = DEFAULT_SCAN_ROOTS,
) -> ParentArtifactTriageReport:
    entries: List[ParentArtifactEntry] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            entries.append(
                ParentArtifactEntry(
                    path=str(path),
                    size_bytes=path.stat().st_size,
                    category="generated_or_result_artifact",
                    recommended_action=PROMOTE_ONLY_THROUGH_GATE,
                )
            )
    return ParentArtifactTriageReport(
        clean=not entries,
        scan_roots=tuple(str(root) for root in scan_roots),
        entries=tuple(entries),
    )


def build_payload(report: ParentArtifactTriageReport) -> Mapping[str, Any]:
    return asdict(report)


def render_markdown(report: ParentArtifactTriageReport) -> str:
    lines = [
        "# UXFD Parent Result Artifact Triage",
        "",
        "Status: triage only. This report is not accepted experiment evidence.",
        "",
        f"- Clean: `{report.clean}`",
        f"- Scanned roots: `{', '.join(report.scan_roots)}`",
        f"- Result artifacts: `{len(report.entries)}`",
        "",
        "## Policy",
        "",
        "- Parent-level result images are not submission evidence.",
        "- Promote only real run outputs that pass `scripts.uxfd_artifact_gate`.",
        "- Do not stage generated figures from smoke or demo runs as accepted artifacts.",
        "",
        "## Entries",
        "",
        "| Path | Size Bytes | Category | Action |",
        "|---|---:|---|---|",
    ]
    for entry in report.entries:
        lines.append(
            f"| `{entry.path}` | {entry.size_bytes} | `{entry.category}` | "
            f"`{entry.recommended_action}` |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Triage UXFD parent result artifacts")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = evaluate_parent_artifact_triage()
    if args.format == "json":
        output = json.dumps(build_payload(report), indent=2) + "\n"
    else:
        output = render_markdown(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
