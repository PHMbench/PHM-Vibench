#!/usr/bin/env python3
"""Audit PHMFactory v0.3 release readiness without mutating repository state."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
TARGET_VERSION = "0.3.0"
TARGET_REPOSITORY = "PHMbench/phmfactory"
REQUIRED_BUNDLE_HASHES = ("metadata", "signals")
FLOATING_REVISIONS = {"main", "master", "latest", "develop", "development", ""}
V020_PROVENANCE_PATH = Path("docs/archive/audits/phmfactory-v0.2-provenance.md")
V020_BASELINE_SHA = "a331769d4005018bc833534ecf4efeb5e8a5a78d"
V020_PROVENANCE_MARKER = "provenance_status: resolved_without_final_tag"


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str


def _read(path: str | Path) -> str:
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


def _v020_provenance_resolved(tags: tuple[str, ...]) -> tuple[bool, str]:
    tagged = tuple(tag for tag in tags if tag.startswith("v0.2"))
    if tagged:
        return True, f"visible tags={tagged!r}"

    path = ROOT / V020_PROVENANCE_PATH
    if not path.is_file():
        return False, f"no v0.2* tag and {V020_PROVENANCE_PATH} is absent"

    text = path.read_text(encoding="utf-8")
    required = (
        V020_PROVENANCE_MARKER,
        f"baseline_sha: {V020_BASELINE_SHA}",
        "v0.2 status: release candidate, not a tagged final release",
    )
    missing = tuple(marker for marker in required if marker not in text)
    if missing:
        return False, f"{V020_PROVENANCE_PATH} is missing markers: {missing!r}"
    return True, f"resolved by {V020_PROVENANCE_PATH}"


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
    for key in REQUIRED_BUNDLE_HASHES:
        value = str(expected_hashes.get(key) or "")
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            findings.append(
                Finding("CWRU_HASH_MISSING", f"expected_sha256.{key} is not pinned")
            )

    tags = _git_tags()
    provenance_ok, provenance_detail = _v020_provenance_resolved(tags)
    if not provenance_ok:
        findings.append(Finding("V020_PROVENANCE_UNRESOLVED", provenance_detail))
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

    findings = collect_findings()
    _print_findings(findings)
    if args.mode == "release" and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
