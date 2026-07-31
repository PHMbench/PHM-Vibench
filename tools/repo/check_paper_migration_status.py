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


def _sha40(value: Any) -> bool:
    return SHA40.fullmatch(str(value or "")) is not None


def _sha64(value: Any) -> bool:
    return SHA64.fullmatch(str(value or "")) is not None


def _validate_foundation(path: str, item: dict[str, Any], findings: list[Finding]) -> bool:
    foundation_start = len(findings)
    if item.get("source_repository") != "liq22/PHM-Vibench-Paper-2025-Metric":
        findings.append(Finding("FOUNDATION_SOURCE_REPOSITORY_INVALID", path))
    if item.get("source_path_count") != 257 or item.get("unique_git_blob_oids") != 243:
        findings.append(Finding("FOUNDATION_SOURCE_COUNTS_INVALID", path))

    partition = item.get("partition") or {}
    if not isinstance(partition, dict):
        findings.append(Finding("FOUNDATION_PARTITION_INVALID", path))
        partition = {}
    if partition.get("program_repository") != "AI4Engineering-L/PHM-Paper-Program-2026":
        findings.append(Finding("FOUNDATION_PROGRAM_REPOSITORY_INVALID", path))
    if partition.get("program_pr") != 3 or not _sha40(partition.get("program_merge_commit")):
        findings.append(Finding("FOUNDATION_PROGRAM_EVIDENCE_INVALID", path))
    for key in ("partition_manifest_sha256", "partition_manifest_gzip_sha256"):
        if not _sha64(partition.get(key)):
            findings.append(Finding("FOUNDATION_PARTITION_HASH_INVALID", f"{path}: {key}"))
    for key in (
        "unassigned_count",
        "cross_authority_path_overlap_count",
        "cross_authority_mutable_oid_overlap_count",
    ):
        if partition.get(key) != 0:
            findings.append(Finding("FOUNDATION_PARTITION_OVERLAP", f"{path}: {key}"))

    dispositions = item.get("disposition_counts") or {}
    expected_dispositions = {
        "p08_import": 34,
        "p09_import": 28,
        "duplicate_alias_to_p08": 8,
        "phmfactory_reference": 2,
        "provenance_quarantine": 182,
        "source_metadata": 3,
    }
    if dispositions != expected_dispositions or sum(dispositions.values()) != 257:
        findings.append(Finding("FOUNDATION_DISPOSITION_COUNTS_INVALID", path))

    targets = item.get("target_imports") or []
    if not isinstance(targets, list) or len(targets) != 2:
        findings.append(Finding("FOUNDATION_TARGET_COUNT_INVALID", path))
        targets = targets if isinstance(targets, list) else []
    by_id = {
        str(target.get("paper_id")): target
        for target in targets
        if isinstance(target, dict) and target.get("paper_id")
    }
    target_contract = {
        "P08": ("AI4Engineering-L/P08-HSE-Prompt-CDDG", 34),
        "P09": ("AI4Engineering-L/P09-HSE-Prompt-GFS", 28),
    }
    for paper_id, (repository, count) in target_contract.items():
        target = by_id.get(paper_id) or {}
        if target.get("target_repository") != repository or target.get("target_pr") != 2:
            findings.append(Finding("FOUNDATION_TARGET_EVIDENCE_INVALID", paper_id))
        if not _sha40(target.get("target_head")) or not _sha40(target.get("target_merge_commit")):
            findings.append(Finding("FOUNDATION_TARGET_SHA_INVALID", paper_id))
        if not isinstance(target.get("target_validation_run"), int):
            findings.append(Finding("FOUNDATION_TARGET_RUN_INVALID", paper_id))
        if target.get("imported_path_count") != count:
            findings.append(Finding("FOUNDATION_TARGET_COUNT_INVALID", paper_id))
        for key in ("selection_manifest_sha256", "source_selection_archive_sha256"):
            if not _sha64(target.get(key)):
                findings.append(Finding("FOUNDATION_TARGET_HASH_INVALID", f"{paper_id}: {key}"))
        if target.get("target_review_status") != "target_merged":
            findings.append(Finding("FOUNDATION_TARGET_NOT_MERGED", paper_id))
        if target.get("active_scientific_evidence") is not False:
            findings.append(Finding("FOUNDATION_CLAIM_BOUNDARY_INVALID", paper_id))

    return (
        item.get("coverage_status") == "complete"
        and item.get("target_review_status") == "targets_merged"
        and item.get("uncovered_count") == 0
        and item.get("safe_to_remove") is True
        and len(findings) == foundation_start
    )


def _validate_single_target(path: str, item: dict[str, Any], findings: list[Finding]) -> bool:
    coverage = str(item.get("coverage_status") or "")
    review = str(item.get("target_review_status") or "")
    safe = item.get("safe_to_remove") is True

    if coverage == "complete":
        values = (
            item.get("source_blob_count"),
            item.get("snapshot_exact_count"),
            item.get("archive_or_overlay_exact_count"),
            item.get("uncovered_count"),
        )
        if not all(isinstance(value, int) for value in values):
            findings.append(Finding("PAPER_COVERAGE_COUNTS_INVALID", path))
        elif values[1] + values[2] + values[3] != values[0]:
            findings.append(Finding("PAPER_COVERAGE_SUM_INVALID", path))
        if item.get("uncovered_count") != 0:
            findings.append(Finding("PAPER_COVERAGE_INCOMPLETE", path))
        for key in ("coverage_manifest_sha256", "source_archive_sha256"):
            if not _sha64(item.get(key)):
                findings.append(Finding("PAPER_COVERAGE_HASH_INVALID", f"{path}: {key}"))
        if not item.get("target_repository") or not item.get("target_pr"):
            findings.append(Finding("PAPER_TARGET_EVIDENCE_MISSING", path))
        if not _sha40(item.get("target_head")) or not _sha40(item.get("target_merge_commit")):
            findings.append(Finding("PAPER_TARGET_SHA_INVALID", path))
    elif coverage not in {"not_started", "in_progress"}:
        findings.append(Finding("PAPER_COVERAGE_STATUS_INVALID", f"{path}: {coverage!r}"))

    removable = coverage == "complete" and review in REVIEWED_STATUSES
    if safe and not removable:
        findings.append(Finding("PAPER_PREMATURE_SAFE_TO_REMOVE", f"{path}: review={review!r}"))
    if safe and item.get("uncovered_count") != 0:
        findings.append(Finding("PAPER_SAFE_WITH_UNCOVERED_BLOBS", path))
    return safe and removable and item.get("uncovered_count") == 0


def collect_findings() -> tuple[Finding, ...]:
    tracker = _load(TRACKER)
    allowlist = _load(ALLOWLIST)
    findings: list[Finding] = []

    if tracker.get("schema_version") != 2:
        findings.append(Finding("TRACKER_SCHEMA_INVALID", "schema_version must be 2"))
    papers = tracker.get("papers") or []
    if not isinstance(papers, list) or len(papers) != 8:
        findings.append(Finding("PAPER_COUNT_INVALID", "tracker must contain exactly 8 papers"))
        papers = papers if isinstance(papers, list) else []

    tracked: dict[str, dict[str, Any]] = {}
    for item in papers:
        if not isinstance(item, dict) or not item.get("source_path"):
            findings.append(Finding("PAPER_ENTRY_INVALID", repr(item)))
            continue
        path = str(item["source_path"])
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
        findings.append(
            Finding(
                "PAPER_TRACKER_PATH_DRIFT",
                f"missing={sorted(set(expected)-set(tracked))}, extra={sorted(set(tracked)-set(expected))}",
            )
        )

    for path, item in sorted(tracked.items()):
        if str(item.get("source_commit") or "") != expected.get(path):
            findings.append(Finding("PAPER_SOURCE_COMMIT_MISMATCH", path))
        elif not _sha40(item.get("source_commit")):
            findings.append(Finding("PAPER_SOURCE_COMMIT_INVALID", path))

        if item.get("paper_id") == "foundation-metric":
            removable = _validate_foundation(path, item, findings)
        else:
            removable = _validate_single_target(path, item, findings)
        if not removable:
            findings.append(
                Finding(
                    "PAPER_MIGRATION_PENDING",
                    f"{path}: coverage={item.get('coverage_status')}, review={item.get('target_review_status')}",
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
