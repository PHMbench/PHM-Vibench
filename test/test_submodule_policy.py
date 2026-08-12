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


def test_repository_policy_has_no_findings() -> None:
    assert policy.collect_findings() == ()


def test_invalid_deferral_is_rejected() -> None:
    errors = policy._deferral_errors({})
    assert errors
    assert any("decision.status" in error for error in errors)


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


def test_gitlinks_reads_the_staged_index(monkeypatch) -> None:
    output = (
        "100644 " + "1" * 40 + " 0\tREADME.md\n"
        "160000 " + "2" * 40 + " 0\tpackages/provider\n"
    )
    monkeypatch.setattr(
        policy.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": output})(),
    )
    assert policy._gitlinks() == {"packages/provider": "2" * 40}


def test_personal_backend_url_is_not_allowlisted(monkeypatch) -> None:
    allowlist = policy._load_allowlist()
    candidate = dict(allowlist["allowed_submodules"][0])
    candidate["target_url"] = "https://github.com/example-user/phm-data-factory.git"
    changed = dict(allowlist)
    changed["allowed_submodules"] = [candidate]
    monkeypatch.setattr(policy, "_load_allowlist", lambda: changed)

    findings = policy.collect_findings()
    assert any(finding.code == "BACKEND_URL_INVALID" for finding in findings)


def test_deferred_backend_must_be_absent(monkeypatch) -> None:
    allowlist = policy._load_allowlist()
    candidate = dict(allowlist["allowed_submodules"][0])
    candidate.update(status="deferred_to_v0.3.1", pinned_commit=None)
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
        lambda: {policy.TARGET_BACKEND_PATH: "a" * 40},
    )

    findings = policy.collect_findings()
    assert any(finding.code == "BACKEND_PRESENT_BEFORE_APPROVAL" for finding in findings)


def test_approved_backend_requires_exact_gitlink(monkeypatch) -> None:
    allowlist = policy._load_allowlist()
    candidate = dict(allowlist["allowed_submodules"][0])
    candidate.update(
        status="approved",
        reviewed_source_tree_commit=policy.TARGET_BACKEND_COMMIT,
        pinned_commit=policy.TARGET_BACKEND_COMMIT,
    )
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


def test_approved_backend_requires_released_commit(monkeypatch) -> None:
    allowlist = policy._load_allowlist()
    candidate = dict(allowlist["allowed_submodules"][0])
    candidate.update(
        status="approved",
        reviewed_source_tree_commit="b" * 40,
        pinned_commit="b" * 40,
    )
    changed = dict(allowlist)
    changed["allowed_submodules"] = [candidate]
    monkeypatch.setattr(policy, "_load_allowlist", lambda: changed)
    monkeypatch.setattr(policy, "_configured_submodules", lambda: {})
    monkeypatch.setattr(policy, "_gitlinks", lambda: {})

    findings = policy.collect_findings()
    codes = {finding.code for finding in findings}
    assert "BACKEND_REVIEWED_COMMIT_INVALID" in codes
    assert "BACKEND_PIN_INVALID" in codes
