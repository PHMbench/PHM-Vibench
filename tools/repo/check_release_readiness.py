#!/usr/bin/env python3
"""Audit PHMFactory v0.3 release readiness without mutating repository state."""

from __future__ import annotations

import argparse
import configparser
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
TARGET_VERSION = "0.3.0"
TARGET_REPOSITORY = "PHMbench/phmfactory"
TARGET_BACKEND_PATH = "packages/phm-data-factory"
TARGET_BACKEND_URL = "https://github.com/PHMbench/phm-data-factory.git"
REQUIRED_BUNDLE_HASHES = ("metadata", "signals")
FLOATING_REVISIONS = {"main", "master", "latest", "develop", "development", ""}
V020_PROVENANCE_PATH = "docs/releases/v0.2.0-rc-provenance.yaml"
SUBMODULE_ALLOWLIST_PATH = ".github/phmfactory-v0.3-submodules.allowlist.yml"
V020_BASELINE_COMMIT = "a331769d4005018bc833534ecf4efeb5e8a5a78d"
V020_EXPECTED_PROVENANCE: dict[str, Any] = {
    "project_name": "PHM-Vibench",
    "version_label": "v0.2.0",
    "status": "release_candidate",
    "formal_release": False,
    "baseline_commit": V020_BASELINE_COMMIT,
    "tag_present": False,
    "superseded_by": "v0.3.0",
}


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _toml_version(text: str) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _python_version(text: str) -> str:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _git_tags() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "tag", "--list"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


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


def _configured_submodules() -> dict[str, dict[str, str]]:
    path = ROOT / ".gitmodules"
    if not path.is_file():
        return {}
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.read(path, encoding="utf-8")
    configured: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        if not section.startswith('submodule "'):
            continue
        submodule_path = parser.get(section, "path", fallback="").strip()
        if not submodule_path:
            continue
        configured[submodule_path] = {
            "url": parser.get(section, "url", fallback="").strip(),
            "branch": parser.get(section, "branch", fallback="").strip(),
        }
    return configured


def _v020_provenance_error() -> str:
    path = ROOT / V020_PROVENANCE_PATH
    if not path.is_file():
        return f"no visible v0.2* Git tag and {V020_PROVENANCE_PATH} is absent"

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return f"could not parse {V020_PROVENANCE_PATH}: {exc}"

    if not isinstance(payload, dict):
        return f"{V020_PROVENANCE_PATH} must contain a YAML mapping"

    mismatches = []
    for key, expected in V020_EXPECTED_PROVENANCE.items():
        actual = payload.get(key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r}, expected {expected!r}")
    if mismatches:
        return "; ".join(mismatches)
    return ""


def _submodule_findings() -> tuple[Finding, ...]:
    findings: list[Finding] = []
    try:
        allowlist = yaml.safe_load(_read(SUBMODULE_ALLOWLIST_PATH)) or {}
    except (OSError, yaml.YAMLError) as exc:
        return (Finding("SUBMODULE_POLICY_INVALID", str(exc)),)
    if not isinstance(allowlist, dict):
        return (Finding("SUBMODULE_POLICY_INVALID", "allowlist is not a mapping"),)

    allowed = allowlist.get("allowed_submodules") or []
    candidate = allowed[0] if isinstance(allowed, list) and len(allowed) == 1 else {}
    if not isinstance(candidate, dict):
        candidate = {}
    status = str(candidate.get("status") or "")
    target_url = str(candidate.get("target_url") or "")
    pinned_commit = str(candidate.get("pinned_commit") or "")

    configured = _configured_submodules()
    backend = configured.get(TARGET_BACKEND_PATH)
    backend_ready = (
        status == "approved"
        and target_url == TARGET_BACKEND_URL
        and re.fullmatch(r"[0-9a-f]{40}", pinned_commit) is not None
        and backend is not None
        and backend.get("url") == TARGET_BACKEND_URL
        and not backend.get("branch")
        and _gitlink(TARGET_BACKEND_PATH) == pinned_commit
    )
    if not backend_ready:
        findings.append(
            Finding(
                "PHM_DATA_FACTORY_BACKEND_PENDING",
                f"status={status!r}, configured={backend is not None}, target_url={target_url!r}",
            )
        )

    legacy_items = allowlist.get("legacy_entries") or []
    legacy_paths = {
        str(item.get("path"))
        for item in legacy_items
        if isinstance(item, dict)
        and item.get("path")
        and item.get("action") == "migrate_then_remove"
    }
    remaining = sorted(path for path in configured if path in legacy_paths)
    if remaining:
        findings.append(
            Finding(
                "LEGACY_SUBMODULES_REMAIN",
                f"{len(remaining)} legacy paper gitlink(s): {', '.join(remaining)}",
            )
        )

    unknown = sorted(
        path for path in configured if path != TARGET_BACKEND_PATH and path not in legacy_paths
    )
    if unknown:
        findings.append(
            Finding("UNKNOWN_SUBMODULES_PRESENT", ", ".join(unknown))
        )

    return tuple(findings)


def collect_findings() -> tuple[Finding, ...]:
    findings: list[Finding] = []

    pyproject = _read("pyproject.toml")
    package_init = _read("phmfactory/__init__.py")
    readme = _read("README.md")
    citation = yaml.safe_load(_read("CITATION.cff")) or {}
    changelog = _read("CHANGELOG.md")
    manifest = yaml.safe_load(
        _read("phmfactory/data_sources/manifests/cwru-demo-v1.yaml")
    ) or {}

    project_version = _toml_version(pyproject)
    package_version = _python_version(package_init)
    if project_version != package_version:
        findings.append(
            Finding(
                "VERSION_MISMATCH",
                f"pyproject={project_version!r}, package={package_version!r}",
            )
        )
    if project_version != TARGET_VERSION:
        findings.append(
            Finding(
                "VERSION_NOT_FINAL",
                f"expected {TARGET_VERSION!r}, found {project_version!r}",
            )
        )

    if not readme.startswith("# PHMFactory\n"):
        findings.append(Finding("README_BRAND_PENDING", "README heading is not PHMFactory"))

    if citation.get("title") != "PHMFactory":
        findings.append(
            Finding("CITATION_BRAND_PENDING", f"title={citation.get('title')!r}")
        )
    expected_url = f"https://github.com/{TARGET_REPOSITORY}"
    for field in ("repository-code", "url"):
        if citation.get(field) != expected_url:
            findings.append(
                Finding(
                    "CITATION_REPOSITORY_PENDING",
                    f"{field}={citation.get(field)!r}, expected {expected_url!r}",
                )
            )

    if not re.search(r"^##\s+v?0\.3\.0\b", changelog, flags=re.MULTILINE):
        findings.append(Finding("CHANGELOG_V030_MISSING", "no v0.3.0 section"))
    if not (ROOT / "RELEASE_NOTES_v0.3.0.md").is_file():
        findings.append(
            Finding("RELEASE_NOTES_V030_MISSING", "RELEASE_NOTES_v0.3.0.md absent")
        )

    providers = manifest.get("providers") or {}
    for provider_name, provider in sorted(providers.items()):
        revision = str((provider or {}).get("revision") or "")
        if revision.casefold() in FLOATING_REVISIONS:
            findings.append(
                Finding(
                    "CWRU_REVISION_FLOATING",
                    f"{provider_name} revision={revision!r}",
                )
            )

    expected_hashes = manifest.get("expected_sha256") or {}
    filename_hash_keys = {
        str((manifest.get("files") or {}).get(key, {}).get("filename") or "")
        for key in REQUIRED_BUNDLE_HASHES
    }
    conflicting_filename_keys = sorted(
        key for key in filename_hash_keys if key and expected_hashes.get(key)
    )
    if conflicting_filename_keys:
        findings.append(
            Finding(
                "CWRU_HASH_KEY_CONFLICT",
                f"filename hash keys are forbidden; use logical keys: {conflicting_filename_keys}",
            )
        )
    for key in REQUIRED_BUNDLE_HASHES:
        value = str(expected_hashes.get(key) or "")
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            findings.append(
                Finding("CWRU_HASH_MISSING", f"expected_sha256.{key} is not pinned")
            )

    findings.extend(_submodule_findings())

    tags = _git_tags()
    if not any(tag.startswith("v0.2") for tag in tags):
        provenance_error = _v020_provenance_error()
        if provenance_error:
            findings.append(Finding("V020_PROVENANCE_UNRESOLVED", provenance_error))
    if any(tag in {"v0.3.0", "0.3.0"} for tag in tags):
        findings.append(
            Finding("V030_TAG_ALREADY_EXISTS", "release tag exists before readiness pass")
        )

    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if repository and repository != TARGET_REPOSITORY:
        findings.append(
            Finding(
                "REPOSITORY_RENAME_PENDING",
                f"current={repository!r}, expected={TARGET_REPOSITORY!r}",
            )
        )

    return tuple(sorted(findings, key=lambda item: (item.code, item.detail)))


def _print_findings(findings: Iterable[Finding]) -> None:
    findings = tuple(findings)
    if not findings:
        print("PHMFactory v0.3 release readiness PASS: 0 blockers")
        return
    print(f"PHMFactory v0.3 release readiness BLOCKED: {len(findings)} blocker(s)")
    for finding in findings:
        print(f"- {finding.code}: {finding.detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "release"), default="audit")
    args = parser.parse_args()

    try:
        findings = collect_findings()
    except (OSError, configparser.Error, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(f"Release readiness ERROR: {exc}", file=sys.stderr)
        return 1

    _print_findings(findings)
    if args.mode == "release" and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
