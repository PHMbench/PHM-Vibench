#!/usr/bin/env python3
"""Validate content-level migration status for PHMFactory paper gitlinks."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
TRACKER = ROOT / "docs/archive/audits/phmfactory-v0.3-paper-submodule-migration-status.yaml"
ALLOWLIST = ROOT / ".github/phmfactory-v0.3-submodules.allowlist.yml"
SHA40 = re.compile(r"[0-9a-f]{40}")
SHA64 = re.compile(r"[0-9a-f]{64}")
REVIEWED_STATUSES = {"target_ci_passed", "target_reviewed", "target_merged"}


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str
    release_only: bool = False


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def collect_findings() -> tuple[Finding, ...]:
    tracker = _load(TRACKER)
    allowlist = _load(ALLOWLIST)
    findings: list[Finding] = []

    papers = tracker.get("papers") or []
    if not isinstance(papers, list) or len(papers) != 8:
        findings.append(Finding("PAPER_COUNT_INVALID", "tracker must contain exactly 8 papers"))
        papers = papers if isinstance(papers, list) else []

    tracked: dict[str, dict[str, Any]] = {}
    for item in papers:
        if not isinstance(item, dict):
            findings.append(Finding("PAPER_ENTRY_INVALID", repr(item)))
            continue
        path = str(item.get("source_path") or "")
        if not path:
            findings.append(Finding("PAPER_SOURCE_PATH_MISSING", repr(item)))
            continue
        if path in tracked:
            findings.append(Finding("PAPER_SOURCE_PATH_DUPLICATE", path))
        tracked[path] = item

    legacy_items = allowlist.get("legacy_entries") or []
    expected = {
        str(item.get("path")): str(item.get("gitlink_commit"))
        for item in legacy_items
        if isinstance(item, dict)
        and item.get("path")
        and item.get("action") == "migrate_then_remove"
    }
    if set(tracked) != set(expected):
        missing = sorted(set(expected) - set(tracked))
        extra = sorted(set(tracked) - set(expected))
        findings.append(
            Finding(
                "PAPER_TRACKER_PATH_DRIFT",
                f"missing={missing}, extra={extra}",
            )
        )

    for path, item in sorted(tracked.items()):
        source_commit = str(item.get("source_commit") or "")
        if source_commit != expected.get(path):
            findings.append(
                Finding(
                    "PAPER_SOURCE_COMMIT_MISMATCH",
                    f"{path}: {source_commit!r} != {expected.get(path)!r}",
                )
            )
        elif not SHA40.fullmatch(source_commit):
            findings.append(Finding("PAPER_SOURCE_COMMIT_INVALID", path))

        coverage = str(item.get("coverage_status") or "")
        review = str(item.get("target_review_status") or "")
        safe = item.get("safe_to_remove") is True

        if coverage == "complete":
            source_count = item.get("source_blob_count")
            snapshot_count = item.get("snapshot_exact_count")
            archive_count = item.get("archive_or_overlay_exact_count")
            uncovered = item.get("uncovered_count")
            if not all(
                isinstance(value, int)
                for value in (source_count, snapshot_count, archive_count, uncovered)
            ):
                findings.append(Finding("PAPER_COVERAGE_COUNTS_INVALID", path))
            elif snapshot_count + archive_count + uncovered != source_count:
                findings.append(Finding("PAPER_COVERAGE_SUM_INVALID", path))
            if uncovered != 0:
                findings.append(Finding("PAPER_COVERAGE_INCOMPLETE", path))
            for key in ("coverage_manifest_sha256", "source_archive_sha256"):
                if not SHA64.fullmatch(str(item.get(key) or "")):
                    findings.append(Finding("PAPER_COVERAGE_HASH_INVALID", f"{path}: {key}"))
            if not item.get("target_repository") or not item.get("target_pr"):
                findings.append(Finding("PAPER_TARGET_EVIDENCE_MISSING", path))
            if not SHA40.fullmatch(str(item.get("target_head") or "")):
                findings.append(Finding("PAPER_TARGET_HEAD_INVALID", path))
        elif coverage not in {"not_started", "in_progress"}:
            findings.append(Finding("PAPER_COVERAGE_STATUS_INVALID", f"{path}: {coverage!r}"))

        removable = coverage == "complete" and review in REVIEWED_STATUSES
        if safe and not removable:
            findings.append(
                Finding("PAPER_PREMATURE_SAFE_TO_REMOVE", f"{path}: review={review!r}")
            )
        if safe and item.get("uncovered_count") != 0:
            findings.append(Finding("PAPER_SAFE_WITH_UNCOVERED_BLOBS", path))
        if not safe:
            findings.append(
                Finding(
                    "PAPER_MIGRATION_PENDING",
                    f"{path}: coverage={coverage}, review={review}",
                    release_only=True,
                )
            )

    return tuple(sorted(findings, key=lambda item: (item.release_only, item.code, item.detail)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("policy", "release"), default="policy")
    args = parser.parse_args()

    try:
        findings = collect_findings()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Paper migration tracker ERROR: {exc}", file=sys.stderr)
        return 1

    for finding in findings:
        kind = "release-blocker" if finding.release_only else "policy-error"
        print(f"- {kind} {finding.code}: {finding.detail}")

    active = findings if args.mode == "release" else tuple(
        finding for finding in findings if not finding.release_only
    )
    if active:
        print(f"Paper migration tracker FAIL: {len(active)} active finding(s)", file=sys.stderr)
        return 1

    pending = sum(finding.release_only for finding in findings)
    print(f"Paper migration tracker PASS: structural contract valid, {pending} pending paper(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
