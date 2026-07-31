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
SHA40 = re.compile(r"[0-9a-f]{40}")
V020_PROVENANCE_PATH = "docs/releases/v0.2.0-rc-provenance.yaml"
SUBMODULE_ALLOWLIST_PATH = ".github/phmfactory-v0.3-submodules.allowlist.yml"
BACKEND_DEFERRAL_PATH = "docs/releases/v0.3.0-backend-deferral.yaml"
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
DEFERRED_STATUS = "deferred_to_v0.3.1"


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


def _gitlink(path: str) -> str:
    return _gitlinks().get(path, "")


def _is_immutable_revision(value: str) -> bool:
    return SHA40.fullmatch(value.casefold()) is not None


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


def _yaml_mapping(path: str) -> dict[str, Any]:
    payload = yaml.safe_load(_read(path)) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _v020_provenance_error() -> str:
    path = ROOT / V020_PROVENANCE_PATH
    if not path.is_file():
        return f"no visible v0.2* Git tag and {V020_PROVENANCE_PATH} is absent"
    payload = _yaml_mapping(V020_PROVENANCE_PATH)
    mismatches = []
    for key, expected in V020_EXPECTED_PROVENANCE.items():
        actual = payload.get(key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r}, expected {expected!r}")
    return "; ".join(mismatches)


def _backend_deferral_error(payload: dict[str, Any]) -> str:
    decision = payload.get("decision") or {}
    ownership = payload.get("ownership") or {}
    state = payload.get("v0.3.0_repository_state") or {}
    behavior = payload.get("behavior_without_backend") or {}
    claims = payload.get("claim_boundary") or {}
    checks = {
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
        "v0.3.0_repository_state.gitlink_present_allowed": (
            state.get("gitlink_present_allowed"),
            False,
        ),
        "v0.3.0_repository_state.runtime_import_allowed": (
            state.get("runtime_import_allowed"),
            False,
        ),
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
    mismatches = [
        f"{field}={actual!r}, expected {expected!r}"
        for field, (actual, expected) in checks.items()
        if actual != expected
    ]
    return "; ".join(mismatches)


def _submodule_findings() -> tuple[Finding, ...]:
    findings: list[Finding] = []
    allowlist = _yaml_mapping(SUBMODULE_ALLOWLIST_PATH)
    allowed = allowlist.get("allowed_submodules") or []
    candidate = allowed[0] if isinstance(allowed, list) and len(allowed) == 1 else {}
    if not isinstance(candidate, dict):
        candidate = {}

    status = str(candidate.get("status") or "")
    target_url = str(candidate.get("target_url") or "")
    pinned_commit = str(candidate.get("pinned_commit") or "")
    configured = _configured_submodules()
    gitlinks = _gitlinks()

    raw_unknown = sorted(set(gitlinks) - set(configured))
    if raw_unknown:
        findings.append(Finding("UNKNOWN_SUBMODULES_PRESENT", ", ".join(raw_unknown)))

    legacy_items = allowlist.get("legacy_entries") or []
    legacy_paths = {
        str(item.get("path"))
        for item in legacy_items
        if isinstance(item, dict) and item.get("path")
    }
    remaining = sorted(path for path in configured if path in legacy_paths)
    if remaining:
        findings.append(
            Finding(
                "LEGACY_SUBMODULES_REMAIN",
                f"{len(remaining)} legacy gitlink(s): {', '.join(remaining)}",
            )
        )

    unknown = sorted(
        path for path in configured if path != TARGET_BACKEND_PATH and path not in legacy_paths
    )
    if unknown:
        findings.append(Finding("UNKNOWN_SUBMODULES_PRESENT", ", ".join(unknown)))

    backend = configured.get(TARGET_BACKEND_PATH)
    if status == DEFERRED_STATUS:
        if not (ROOT / BACKEND_DEFERRAL_PATH).is_file():
            findings.append(
                Finding("BACKEND_DEFERRAL_INVALID", f"{BACKEND_DEFERRAL_PATH} is absent")
            )
        else:
            error = _backend_deferral_error(_yaml_mapping(BACKEND_DEFERRAL_PATH))
            if error:
                findings.append(Finding("BACKEND_DEFERRAL_INVALID", error))
        if backend is not None or _gitlink(TARGET_BACKEND_PATH):
            findings.append(
                Finding(
                    "BACKEND_DEFERRAL_INVALID",
                    "v0.3.0 deferral requires no configured backend or gitlink",
                )
            )
        if pinned_commit:
            findings.append(
                Finding("BACKEND_DEFERRAL_INVALID", "deferred backend must not set pinned_commit")
            )
        if target_url != TARGET_BACKEND_URL:
            findings.append(
                Finding("BACKEND_DEFERRAL_INVALID", f"target_url={target_url!r}")
            )
    else:
        backend_ready = (
            status == "approved"
            and target_url == TARGET_BACKEND_URL
            and SHA40.fullmatch(pinned_commit) is not None
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

    return tuple(findings)


def collect_findings() -> tuple[Finding, ...]:
    findings: list[Finding] = []

    pyproject = _read("pyproject.toml")
    package_init = _read("phmfactory/__init__.py")
    readme = _read("README.md")
    citation = _yaml_mapping("CITATION.cff")
    changelog = _read("CHANGELOG.md")
    manifest = _yaml_mapping("phmfactory/data_sources/manifests/cwru-demo-v1.yaml")

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
        findings.append(Finding("CITATION_BRAND_PENDING", f"title={citation.get('title')!r}"))
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
    release_pin_required = manifest.get("release_pin_required") is True
    for provider_name, provider in sorted(providers.items()):
        revision = str((provider or {}).get("revision") or "")
        if (
            release_pin_required and not _is_immutable_revision(revision)
        ) or revision.casefold() in FLOATING_REVISIONS:
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
        findings.append(Finding("V030_TAG_ALREADY_EXISTS", "release tag exists before readiness pass"))

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
    except (
        OSError,
        ValueError,
        configparser.Error,
        yaml.YAMLError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Release readiness ERROR: {exc}", file=sys.stderr)
        return 1

    _print_findings(findings)
    if args.mode == "release" and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
