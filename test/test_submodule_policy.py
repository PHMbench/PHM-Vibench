from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/repo/check_submodule_policy.py"
SPEC = importlib.util.spec_from_file_location("check_submodule_policy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)


def test_repository_policy_has_no_structural_errors() -> None:
    findings = policy.collect_findings()
    structural = [finding for finding in findings if not finding.release_only]
    assert structural == []
    codes = {finding.code for finding in findings}
    assert "PHM_DATA_FACTORY_BACKEND_PENDING" in codes
    assert "LEGACY_SUBMODULE_REMAINS" not in codes


def test_unknown_submodule_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        policy,
        "_configured_submodules",
        lambda: {"packages/unreviewed": {"url": "https://example.invalid/x.git", "branch": ""}},
    )
    monkeypatch.setattr(policy, "_gitlinks", lambda: {"packages/unreviewed": "a" * 40})
    findings = policy.collect_findings()
    assert any(
        finding.code == "UNKNOWN_SUBMODULE" and finding.detail == "packages/unreviewed"
        for finding in findings
    )


def test_unconfigured_raw_gitlink_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(policy, "_configured_submodules", lambda: {})
    monkeypatch.setattr(
        policy,
        "_gitlinks",
        lambda: {"arbitrary/raw-gitlink": "d" * 40},
    )
    findings = policy.collect_findings()
    assert any(
        finding.code == "UNKNOWN_SUBMODULE"
        and finding.detail == "arbitrary/raw-gitlink"
        for finding in findings
    )


def test_personal_backend_url_is_not_allowlisted(monkeypatch) -> None:
    allowlist = policy._load_allowlist()
    candidate = dict(allowlist["allowed_submodules"][0])
    candidate["target_url"] = "https://github.com/example-user/phm-data-factory.git"
    changed = dict(allowlist)
    changed["allowed_submodules"] = [candidate]
    monkeypatch.setattr(policy, "_load_allowlist", lambda: changed)
    monkeypatch.setattr(policy, "_configured_submodules", lambda: {})
    monkeypatch.setattr(policy, "_gitlinks", lambda: {})

    findings = policy.collect_findings()
    assert any(finding.code == "BACKEND_URL_INVALID" for finding in findings)


def test_approved_backend_requires_exact_gitlink(monkeypatch) -> None:
    allowlist = policy._load_allowlist()
    candidate = dict(allowlist["allowed_submodules"][0])
    candidate.update(status="approved", pinned_commit="b" * 40)
    changed = dict(allowlist)
    changed["allowed_submodules"] = [candidate]
    monkeypatch.setattr(policy, "_load_allowlist", lambda: changed)
    monkeypatch.setattr(
        policy,
        "_configured_submodules",
        lambda: {
            policy.TARGET_BACKEND_PATH: {
                "url": policy.TARGET_BACKEND_URL,
                "branch": "",
            }
        },
    )
    monkeypatch.setattr(
        policy,
        "_gitlinks",
        lambda: {policy.TARGET_BACKEND_PATH: "c" * 40},
    )

    findings = policy.collect_findings()
    assert any(finding.code == "BACKEND_GITLINK_MISMATCH" for finding in findings)
