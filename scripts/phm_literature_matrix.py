from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional


DEFAULT_INVENTORY = Path("docs/literature/phm_2025_plus.csv")
MIN_YEAR = 2025
SUPPORT_STATUSES = {
    "represented",
    "candidate-baseline",
    "literature-only",
    "dependency-blocked",
    "unsupported",
}
REQUIRED_FIELDS = (
    "id",
    "year",
    "title",
    "authors",
    "venue",
    "url",
    "task_family",
    "method_family",
    "repo_surface",
    "support_status",
)


@dataclass(frozen=True)
class LiteratureEntry:
    id: str
    year: int
    title: str
    authors: str
    venue: str
    url: str
    doi: str
    task_family: str
    method_family: str
    repo_surface: str
    support_status: str
    notes: str


@dataclass(frozen=True)
class InventoryReport:
    total_entries: int
    min_year: int
    max_year: int
    counts_by_task_family: Mapping[str, int]
    counts_by_method_family: Mapping[str, int]
    counts_by_support_status: Mapping[str, int]


class InventoryValidationError(ValueError):
    """Raised when the PHM literature inventory violates its contract."""


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _read_rows(path: Path) -> List[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_inventory(path: Path = DEFAULT_INVENTORY) -> List[LiteratureEntry]:
    rows = _read_rows(path)
    entries: List[LiteratureEntry] = []
    row_errors: list[str] = []

    for idx, row in enumerate(rows, start=2):
        missing = [field for field in REQUIRED_FIELDS if not (row.get(field) or "").strip()]
        if missing:
            row_errors.append(f"row {idx}: missing required fields: {', '.join(missing)}")
            continue

        year_raw = (row.get("year") or "").strip()
        try:
            year = int(year_raw)
        except ValueError:
            row_errors.append(f"row {idx}: year must be an integer, got {year_raw!r}")
            continue

        entries.append(
            LiteratureEntry(
                id=(row.get("id") or "").strip(),
                year=year,
                title=(row.get("title") or "").strip(),
                authors=(row.get("authors") or "").strip(),
                venue=(row.get("venue") or "").strip(),
                url=(row.get("url") or "").strip(),
                doi=(row.get("doi") or "").strip(),
                task_family=(row.get("task_family") or "").strip(),
                method_family=(row.get("method_family") or "").strip(),
                repo_surface=(row.get("repo_surface") or "").strip(),
                support_status=(row.get("support_status") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
        )

    if row_errors:
        raise InventoryValidationError("\n".join(row_errors))
    return entries


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def validate_inventory(entries: List[LiteratureEntry], *, min_count: int = 50) -> InventoryReport:
    issues: list[str] = []

    if len(entries) < min_count:
        issues.append(f"expected at least {min_count} entries, got {len(entries)}")

    old_entries = [entry.id for entry in entries if entry.year < MIN_YEAR]
    if old_entries:
        issues.append(f"entries older than {MIN_YEAR}: {', '.join(old_entries)}")

    invalid_status = [
        f"{entry.id}:{entry.support_status}"
        for entry in entries
        if entry.support_status not in SUPPORT_STATUSES
    ]
    if invalid_status:
        issues.append(f"invalid support_status values: {', '.join(invalid_status)}")

    duplicate_ids = _duplicates(entry.id for entry in entries)
    if duplicate_ids:
        issues.append(f"duplicate ids: {', '.join(duplicate_ids)}")

    duplicate_titles = _duplicates(_normalize_key(entry.title) for entry in entries)
    if duplicate_titles:
        issues.append(f"duplicate titles: {', '.join(duplicate_titles)}")

    duplicate_urls = _duplicates(entry.url for entry in entries)
    if duplicate_urls:
        issues.append(f"duplicate urls: {', '.join(duplicate_urls)}")

    if entries:
        task_counts = Counter(entry.task_family for entry in entries)
        method_counts = Counter(entry.method_family for entry in entries)
        status_counts = Counter(entry.support_status for entry in entries)
        years = [entry.year for entry in entries]
    else:
        task_counts = Counter()
        method_counts = Counter()
        status_counts = Counter()
        years = [0]

    if len(task_counts) < 5:
        issues.append(f"expected at least 5 task families, got {len(task_counts)}")
    if len(method_counts) < 8:
        issues.append(f"expected at least 8 method families, got {len(method_counts)}")

    if issues:
        raise InventoryValidationError("\n".join(issues))

    return InventoryReport(
        total_entries=len(entries),
        min_year=min(years),
        max_year=max(years),
        counts_by_task_family=dict(sorted(task_counts.items())),
        counts_by_method_family=dict(sorted(method_counts.items())),
        counts_by_support_status=dict(sorted(status_counts.items())),
    )


def _render_count_table(title: str, counts: Mapping[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| Key | Count |", "|---|---:|"]
    lines.extend(f"| `{key}` | {count} |" for key, count in counts.items())
    return lines + [""]


def render_markdown(report: InventoryReport) -> str:
    lines = [
        "# PHM 2025+ Literature Matrix",
        "",
        f"- Total entries: `{report.total_entries}`",
        f"- Year range: `{report.min_year}`-`{report.max_year}`",
        "",
    ]
    lines.extend(_render_count_table("Task Families", report.counts_by_task_family))
    lines.extend(_render_count_table("Method Families", report.counts_by_method_family))
    lines.extend(_render_count_table("Support Status", report.counts_by_support_status))
    return "\n".join(lines).rstrip() + "\n"


def build_report(path: Path = DEFAULT_INVENTORY, *, min_count: int = 50) -> InventoryReport:
    entries = load_inventory(path)
    return validate_inventory(entries, min_count=min_count)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize 2025+ PHM literature")
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args(argv)

    try:
        report = build_report(Path(args.inventory), min_count=args.min_count)
    except InventoryValidationError as exc:
        print(f"[FAIL] PHM literature inventory validation failed:\n{exc}")
        return 1

    if args.format == "json":
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
