#!/usr/bin/env python3
"""Validate the PHMFactory deny-by-default Git submodule contract."""

from __future__ import annotations

import argparse
import configparser
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = ROOT / ".github/phmfactory-v0.3-submodules.allowlist.yml"
GITMODULES_PATH = ROOT / ".gitmodules"
TARGET_BACKEND_PATH = "packages/phm-data-factory"
TARGET_BACKEND_URL = "https://github.com/PHMbench/phm-data-factory.git"
SHA40 = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str
    release_only: bool = False


def _load_allowlist() -> dict[str, Any]:
    payload = yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("submodule allowlist must contain a YAML mapping")
    return payload


def _configured_submodules() -> dict[str, dict[str, str]]:
    if not GITMODULES_PATH.is_file():
        return {}
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.read(GITMODULES_PATH, encoding="utf-8")
    result: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        if not section.startswith('submodule "'):
            continue
        path = parser.get(section, "path", fallback="").strip()
        url = parser.get(section, "url", fallback="").strip()
        branch = parser.get(section, "branch", fallback="").strip()
        if not path:
            raise ValueError(f"{section} has no path")
        if path in result:
            raise ValueError(f"duplicate submodule path: {path}")
        result[path] = {"url": url, "branch": branch}
    return result


def _gitlink(path: str) -> str:
    result = subprocess.run(
        ["git", "ls-tree", "HEAD", "--", path],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if not result:
        return ""
    fields = result.split(None, 3)
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        return ""
    return fields[2]


def collect_findings() -> tuple[Finding, ...]:
    findings: list[Finding] = []
    allowlist = _load_allowlist()
    if allowlist.get("schema_version") != 2:
        findings.append(Finding("ALLOWLIST_SCHEMA_INVALID", "schema_version must be 2"))
    if allowlist.get("policy") != "deny_by_default":
        findings.append(Finding("ALLOWLIST_POLICY_INVALID", "policy must be deny_by_default"))

    allowed = allowlist.get("allowed_submodules") or []
    if not isinstance(allowed, list) or len(allowed) != 1:
        findings.append(
            Finding("ALLOWED_SUBMODULE_COUNT_INVALID", "exactly one backend candidate is permitted")
        )
        candidate: dict[str, Any] = {}
    else:
        candidate = allowed[0] if isinstance(allowed[0], dict) else {}

    if candidate.get("name") != "phm-data-factory":
        findings.append(Finding("BACKEND_NAME_INVALID", "candidate name must be phm-data-factory"))
    if candidate.get("path") != TARGET_BACKEND_PATH:
        findings.append(
            Finding("BACKEND_PATH_INVALID", f"candidate path must be {TARGET_BACKEND_PATH}")
        )
    if candidate.get("target_url") != TARGET_BACKEND_URL:
        findings.append(
            Finding("BACKEND_URL_INVALID", f"candidate URL must be {TARGET_BACKEND_URL}")
        )
    if candidate.get("expected_owner") != "PHMbench":
        findings.append(Finding("BACKEND_OWNER_INVALID", "expected_owner must be PHMbench"))
    if candidate.get("optional") is not True:
        findings.append(Finding("BACKEND_NOT_OPTIONAL", "backend candidate must remain optional"))
    if candidate.get("license") != "Apache-2.0":
        findings.append(Finding("BACKEND_LICENSE_INVALID", "license must be Apache-2.0"))

    status = str(candidate.get("status") or "")
    if status not in {"blocked_pending_org_transfer", "approved"}:
        findings.append(Finding("BACKEND_STATUS_INVALID", f"unsupported status: {status!r}"))

    reviewed = str(candidate.get("reviewed_source_tree_commit") or "")
    if not SHA40.fullmatch(reviewed):
        findings.append(
            Finding("BACKEND_REVIEWED_COMMIT_INVALID", "reviewed source tree commit is not 40 hex")
        )

    pinned = str(candidate.get("pinned_commit") or "")
    if status == "approved" and not SHA40.fullmatch(pinned):
        findings.append(
            Finding("BACKEND_PIN_MISSING", "approved backend requires a 40-hex pinned_commit")
        )
    if status != "approved" and pinned:
        findings.append(
            Finding("BACKEND_PIN_PREMATURE", "blocked backend must not advertise a final pinned_commit")
        )

    legacy_items = allowlist.get("legacy_entries") or []
    legacy: dict[str, dict[str, Any]] = {}
    if not isinstance(legacy_items, list):
        findings.append(Finding("LEGACY_ALLOWLIST_INVALID", "legacy_entries must be a list"))
    else:
        for item in legacy_items:
            if not isinstance(item, dict) or not item.get("path"):
                findings.append(Finding("LEGACY_ENTRY_INVALID", f"invalid legacy entry: {item!r}"))
                continue
            path = str(item["path"])
            if path in legacy:
                findings.append(Finding("LEGACY_ENTRY_DUPLICATE", path))
            legacy[path] = item

    configured = _configured_submodules()
    for path, entry in sorted(configured.items()):
        gitlink = _gitlink(path)
        if path == TARGET_BACKEND_PATH:
            if status != "approved":
                findings.append(
                    Finding(
                        "BACKEND_PRESENT_BEFORE_APPROVAL",
                        "backend gitlink exists while allowlist status is not approved",
                    )
                )
            if entry["url"] != TARGET_BACKEND_URL:
                findings.append(
                    Finding("BACKEND_GITMODULE_URL_INVALID", f"found {entry['url']!r}")
                )
            if entry["branch"]:
                findings.append(
                    Finding("BACKEND_BRANCH_TRACKING_FORBIDDEN", f"branch={entry['branch']!r}")
                )
            if pinned and gitlink != pinned:
                findings.append(
                    Finding(
                        "BACKEND_GITLINK_MISMATCH",
                        f"gitlink={gitlink!r}, pinned_commit={pinned!r}",
                    )
                )
            continue

        item = legacy.get(path)
        if item is None:
            findings.append(Finding("UNKNOWN_SUBMODULE", path))
            continue
        expected = str(item.get("gitlink_commit") or "")
        if gitlink != expected:
            findings.append(
                Finding(
                    "LEGACY_GITLINK_DRIFT",
                    f"{path}: gitlink={gitlink!r}, expected={expected!r}",
                )
            )
        findings.append(
            Finding("LEGACY_SUBMODULE_REMAINS", path, release_only=True)
        )

    for path, item in sorted(legacy.items()):
        action = str(item.get("action") or "")
        if action == "migrated_and_removed" and path in configured:
            findings.append(
                Finding("REMOVED_LEGACY_SUBMODULE_RETURNED", path)
            )

    if TARGET_BACKEND_PATH not in configured:
        findings.append(
            Finding(
                "PHM_DATA_FACTORY_BACKEND_PENDING",
                f"{status}: organization-owned backend gitlink is not integrated",
                release_only=True,
            )
        )
    elif status != "approved":
        findings.append(
            Finding(
                "PHM_DATA_FACTORY_BACKEND_PENDING",
                f"backend status is {status!r}",
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
    except (OSError, ValueError, configparser.Error, yaml.YAMLError) as exc:
        print(f"Submodule policy ERROR: {exc}", file=sys.stderr)
        return 1

    active = findings if args.mode == "release" else tuple(
        finding for finding in findings if not finding.release_only
    )
    for finding in findings:
        label = "release-blocker" if finding.release_only else "policy-error"
        print(f"- {label} {finding.code}: {finding.detail}")

    if active:
        print(f"Submodule policy FAIL: {len(active)} active finding(s)", file=sys.stderr)
        return 1

    pending = sum(finding.release_only for finding in findings)
    print(f"Submodule policy PASS: structural contract valid, {pending} release blocker(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
