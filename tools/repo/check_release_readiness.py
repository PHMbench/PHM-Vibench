#!/usr/bin/env python3
"""Audit PHMFactory v0.3.0-rc1 readiness without mutating repository state."""

from __future__ import annotations

import argparse
import configparser
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
TARGET_VERSION = "0.3.0rc1"
TARGET_RELEASE_LABEL = "v0.3.0-rc1"
TARGET_REPOSITORY = "PHMbench/PHM-Vibench"
TARGET_BACKEND_PATH = "packages/phm-data-factory"
TARGET_BACKEND_URL = "https://github.com/PHMbench/phm-data-factory.git"
V020_PROVENANCE_PATH = "docs/releases/v0.2.0-rc-provenance.yaml"
SUBMODULE_ALLOWLIST_PATH = ".github/phmfactory-v0.3-submodules.allowlist.yml"
BACKEND_DEFERRAL_PATH = "docs/releases/v0.3.0-backend-deferral.yaml"
BASELINE_REGISTRY_PATH = "configs/config_registry.csv"
BASELINE_REGISTRY_ID = "baseline_01_mfpt_global_average_linear"
BASELINE_CONFIG_PATH = "configs/baselines/01_mfpt/mfpt_global_average_linear.yaml"
BASELINE_REQUIRED_PATHS = (
    BASELINE_CONFIG_PATH,
    "scripts/prepare_mfpt_baseline.py",
    "src/data_factory/reader/RM_007_MFPT.py",
    "test/test_mfpt_baseline.py",
    ".github/workflows/mfpt-baseline.yml",
)
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
SHA40 = re.compile(r"[0-9a-f]{40}")


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _cwru_contract_errors(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return scientific CWRU manifest errors without treating file hashes as semantics."""

    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("bundle_id") != "cwru-demo-v1":
        errors.append("bundle_id must be 'cwru-demo-v1'")
    if payload.get("dataset_name") != "CWRU":
        errors.append("dataset_name must be 'CWRU'")

    files = _mapping(payload.get("files"))
    declared_files: dict[str, str] = {}
    for logical_name in ("metadata", "signals"):
        entry = _mapping(files.get(logical_name))
        filename = entry.get("filename")
        if not _nonempty(filename):
            errors.append(f"files.{logical_name}.filename must be non-empty")
        else:
            declared_files[logical_name] = str(filename).strip()
        if entry.get("required") is not True:
            errors.append(f"files.{logical_name}.required must be true")

    metadata = _mapping(payload.get("metadata"))
    if not _nonempty(metadata.get("id_column")):
        errors.append("metadata.id_column must be non-empty")
    required_columns = {
        str(value).strip()
        for value in metadata.get("required_columns") or ()
        if _nonempty(value)
    }
    missing_columns = sorted({"Dataset_id", "Label", "Domain_id"} - required_columns)
    if missing_columns:
        errors.append(f"metadata.required_columns missing {missing_columns}")

    selector = _mapping(metadata.get("selector"))
    if not _nonempty(selector.get("column")):
        errors.append("metadata.selector.column must be non-empty")
    selector_values = [
        str(value).strip()
        for value in selector.get("values") or ()
        if _nonempty(value)
    ]
    if not selector_values:
        errors.append("metadata.selector.values must be non-empty")

    aliases = _mapping(metadata.get("column_aliases"))
    for logical_name in ("sample_length", "channel_count"):
        values = [
            str(value).strip()
            for value in aliases.get(logical_name) or ()
            if _nonempty(value)
        ]
        if not values:
            errors.append(f"metadata.column_aliases.{logical_name} must be non-empty")

    providers = _mapping(payload.get("providers"))
    if not providers:
        errors.append("providers must contain at least one provider")
    for provider_name, raw_provider in sorted(providers.items()):
        provider = _mapping(raw_provider)
        if not _nonempty(provider.get("repo_id")):
            errors.append(f"providers.{provider_name}.repo_id must be non-empty")
        if not _nonempty(provider.get("revision")):
            errors.append(f"providers.{provider_name}.revision must be explicit")
        provider_files = _mapping(provider.get("files"))
        for logical_name in ("metadata", "signals"):
            expected = declared_files.get(logical_name)
            actual = provider_files.get(logical_name)
            if expected and actual != expected:
                errors.append(
                    f"providers.{provider_name}.files.{logical_name}={actual!r}, "
                    f"expected {expected!r}"
                )

    return tuple(errors)


def _read_registry_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "category", "path", "pipeline", "status", "protocol_status"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        return [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _baseline_valid_error(rows: Iterable[Mapping[str, str]]) -> str:
    matches = [row for row in rows if row.get("id") == BASELINE_REGISTRY_ID]
    if len(matches) != 1:
        return f"expected exactly one {BASELINE_REGISTRY_ID!r} row, found {len(matches)}"
    row = matches[0]
    expected = {
        "category": "baseline",
        "path": BASELINE_CONFIG_PATH,
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "status": "sanity_ok",
        "protocol_status": "baseline_valid",
    }
    mismatches = [
        f"{field}={row.get(field)!r}, expected {value!r}"
        for field, value in expected.items()
        if row.get(field) != value
    ]
    return "; ".join(mismatches)


def collect_findings() -> tuple[Finding, ...]:
    findings: list[Finding] = []

    pyproject = _read("pyproject.toml")
    package_init = _read("phmfactory/__init__.py")
    readme = _read("README.md")
    citation = _yaml_mapping("CITATION.cff")
    changelog = _read("CHANGELOG.md")
    cwru_manifest = _yaml_mapping("phmfactory/data_sources/manifests/cwru-demo-v1.yaml")

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
                "VERSION_NOT_RC1",
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

    cwru_errors = _cwru_contract_errors(cwru_manifest)
    if cwru_errors:
        findings.append(Finding("CWRU_CONTRACT_INVALID", "; ".join(cwru_errors)))

    registry_path = ROOT / BASELINE_REGISTRY_PATH
    if not registry_path.is_file():
        findings.append(Finding("BASELINE_VALID_REFERENCE_INVALID", f"{registry_path} absent"))
    else:
        baseline_error = _baseline_valid_error(_read_registry_rows(registry_path))
        if baseline_error:
            findings.append(Finding("BASELINE_VALID_REFERENCE_INVALID", baseline_error))
    missing_baseline_paths = [
        path for path in BASELINE_REQUIRED_PATHS if not (ROOT / path).is_file()
    ]
    if missing_baseline_paths:
        findings.append(
            Finding(
                "BASELINE_VALID_REFERENCE_INVALID",
                "missing reviewed baseline path(s): " + ", ".join(missing_baseline_paths),
            )
        )

    findings.extend(_submodule_findings())

    tags = _git_tags()
    if not any(tag.startswith("v0.2") for tag in tags):
        provenance_error = _v020_provenance_error()
        if provenance_error:
            findings.append(Finding("V020_PROVENANCE_UNRESOLVED", provenance_error))
    reserved_tags = {"v0.3.0rc1", "0.3.0rc1", "v0.3.0-rc1", "0.3.0-rc1"}
    if any(tag in reserved_tags for tag in tags):
        findings.append(
            Finding("V030RC1_TAG_ALREADY_EXISTS", "release-candidate tag exists before readiness pass")
        )
    if any(tag in {"v0.3.0", "0.3.0"} for tag in tags):
        findings.append(Finding("V030_TAG_ALREADY_EXISTS", "final release tag already exists"))

    return tuple(sorted(findings, key=lambda item: (item.code, item.detail)))


def _print_findings(findings: Iterable[Finding]) -> None:
    findings = tuple(findings)
    if not findings:
        print(f"PHMFactory {TARGET_RELEASE_LABEL} readiness PASS: 0 blockers")
        return
    print(
        f"PHMFactory {TARGET_RELEASE_LABEL} readiness BLOCKED: "
        f"{len(findings)} blocker(s)"
    )
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
        csv.Error,
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
