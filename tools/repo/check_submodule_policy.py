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
DEFERRAL_PATH = ROOT / "docs/releases/v0.3.0-backend-deferral.yaml"
GITMODULES_PATH = ROOT / ".gitmodules"
TARGET_BACKEND_PATH = "packages/phm-data-factory"
TARGET_BACKEND_URL = "https://github.com/PHMbench/phm-data-factory.git"
SHA40 = re.compile(r"[0-9a-f]{40}")
DEFERRED_STATUS = "deferred_to_v0.3.1"
APPROVED_STATUS = "approved"


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str
    release_only: bool = False


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return payload


def _load_allowlist() -> dict[str, Any]:
    return _load_yaml(ALLOWLIST_PATH, "submodule allowlist")


def _load_deferral() -> dict[str, Any]:
    return _load_yaml(DEFERRAL_PATH, "backend deferral contract")


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


def _gitlinks() -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    gitlinks: dict[str, str] = {}
    for line in result.splitlines():
        if "\t" not in line:
            continue
        metadata, path = line.split("\t", 1)
        fields = metadata.split()
        if len(fields) == 3 and fields[0] == "160000" and fields[1] == "commit":
            gitlinks[path] = fields[2]
    return gitlinks


def _deferral_errors(payload: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    decision = payload.get("decision") or {}
    ownership = payload.get("ownership") or {}
    state = payload.get("v0.3.0_repository_state") or {}
    behavior = payload.get("behavior_without_backend") or {}
    claims = payload.get("claim_boundary") or {}

    exact = {
        "schema_version": (payload.get("schema_version"), 1),
        "release": (str(payload.get("release") or ""), "0.3.0"),
        "component": (payload.get("component"), "phm-data-factory"),
        "decision.status": (decision.get("status"), DEFERRED_STATUS),
        "decision.included_in_v0.3.0": (decision.get("included_in_v0.3.0"), False),
        "decision.required_for_core": (decision.get("required_for_core"), False),
        "decision.release_blocking_for_v0.3.0": (
            decision.get("release_blocking_for_v0.3.0"),
            False,
        ),
        "decision.target_release": (str(decision.get("target_release") or ""), "0.3.1"),
        "ownership.target_repository": (
            ownership.get("target_repository"),
            "PHMbench/phm-data-factory",
        ),
        "ownership.organization_owned_required": (
            ownership.get("organization_owned_required"),
            True,
        ),
        "ownership.immutable_pin_required": (
            ownership.get("immutable_pin_required"),
            True,
        ),
        "ownership.personal_repository_url_forbidden": (
            ownership.get("personal_repository_url_forbidden"),
            True,
        ),
        "v0.3.0_repository_state.gitlink_required": (state.get("gitlink_required"), False),
        "v0.3.0_repository_state.gitlink_present_allowed": (
            state.get("gitlink_present_allowed"),
            False,
        ),
        "v0.3.0_repository_state.placeholder_gitlink_allowed": (
            state.get("placeholder_gitlink_allowed"),
            False,
        ),
        "v0.3.0_repository_state.branch_tracking_allowed": (
            state.get("branch_tracking_allowed"),
            False,
        ),
        "v0.3.0_repository_state.runtime_import_allowed": (
            state.get("runtime_import_allowed"),
            False,
        ),
        "behavior_without_backend.public_cli_works": (behavior.get("public_cli_works"), True),
        "behavior_without_backend.dummy_smoke_works": (behavior.get("dummy_smoke_works"), True),
        "behavior_without_backend.silent_fallback_allowed": (
            behavior.get("silent_fallback_allowed"),
            False,
        ),
        "behavior_without_backend.explicit_selection_error_required": (
            behavior.get("explicit_selection_error_required"),
            True,
        ),
        "claim_boundary.backend_integrated": (claims.get("backend_integrated"), False),
        "claim_boundary.backend_supported": (claims.get("backend_supported"), False),
        "claim_boundary.live_iotdb_supported": (claims.get("live_iotdb_supported"), False),
        "claim_boundary.performance_claim_authorized": (
            claims.get("performance_claim_authorized"),
            False,
        ),
    }
    for field, (actual, expected) in exact.items():
        if actual != expected:
            errors.append(f"{field}={actual!r}, expected {expected!r}")

    gates = payload.get("v0.3.1_entry_gate") or []
    required_gates = {
        "organization_owned_public_https_repository",
        "compatible_explicit_license",
        "immutable_reviewed_commit",
        "bounded_adapter_pr",
        "no_protected_runtime_rewrite",
        "explicit_missing_backend_error",
        "core_paths_pass_without_backend",
    }
    if not isinstance(gates, list) or set(gates) != required_gates:
        errors.append("v0.3.1_entry_gate must contain the exact approved gate set")
    return tuple(errors)


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
    if status not in {DEFERRED_STATUS, APPROVED_STATUS}:
        findings.append(Finding("BACKEND_STATUS_INVALID", f"unsupported status: {status!r}"))

    reviewed = str(candidate.get("reviewed_source_tree_commit") or "")
    if not SHA40.fullmatch(reviewed):
        findings.append(
            Finding("BACKEND_REVIEWED_COMMIT_INVALID", "reviewed source tree commit is not 40 hex")
        )

    pinned = str(candidate.get("pinned_commit") or "")
    if status == APPROVED_STATUS and not SHA40.fullmatch(pinned):
        findings.append(
            Finding("BACKEND_PIN_MISSING", "approved backend requires a 40-hex pinned_commit")
        )
    if status != APPROVED_STATUS and pinned:
        findings.append(
            Finding("BACKEND_PIN_PREMATURE", "deferred backend must not advertise a final pin")
        )

    if status == DEFERRED_STATUS:
        if not DEFERRAL_PATH.is_file():
            findings.append(Finding("BACKEND_DEFERRAL_MISSING", str(DEFERRAL_PATH.relative_to(ROOT))))
        else:
            for error in _deferral_errors(_load_deferral()):
                findings.append(Finding("BACKEND_DEFERRAL_INVALID", error))
    elif status == APPROVED_STATUS and DEFERRAL_PATH.is_file():
        findings.append(
            Finding(
                "BACKEND_DEFERRAL_CONFLICT",
                "approved backend cannot retain the v0.3.0 exclusion contract",
            )
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
    gitlinks = _gitlinks()
    for path in sorted(set(gitlinks) - set(configured)):
        findings.append(Finding("UNKNOWN_SUBMODULE", path))

    for path, entry in sorted(configured.items()):
        gitlink = gitlinks.get(path, "")
        if not gitlink:
            findings.append(Finding("CONFIGURED_SUBMODULE_GITLINK_MISSING", path))
            continue
        if path == TARGET_BACKEND_PATH:
            if status != APPROVED_STATUS:
                findings.append(
                    Finding(
                        "BACKEND_PRESENT_BEFORE_APPROVAL",
                        "backend gitlink exists while v0.3.0 defers integration",
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

        if path in legacy:
            findings.append(Finding("REMOVED_LEGACY_SUBMODULE_RETURNED", path))
        else:
            findings.append(Finding("UNKNOWN_SUBMODULE", path))

    if status == APPROVED_STATUS and TARGET_BACKEND_PATH not in configured:
        findings.append(
            Finding(
                "PHM_DATA_FACTORY_BACKEND_PENDING",
                "approved backend is missing its exact configured gitlink",
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
    except (
        OSError,
        ValueError,
        configparser.Error,
        yaml.YAMLError,
        subprocess.CalledProcessError,
    ) as exc:
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
