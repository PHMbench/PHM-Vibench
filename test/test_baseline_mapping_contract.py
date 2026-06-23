from __future__ import annotations

from scripts.baseline_mapping import BASELINE_ROLES, derive_baselines, validate_baselines
from scripts.model_support_matrix import derive_model_support
from scripts.task_experiment_matrix import derive_matrix


def test_baseline_model_refs_and_roles_are_registered() -> None:
    baselines = derive_baselines()
    model_keys = set(derive_model_support().model_statuses)

    assert baselines
    for baseline in baselines:
        assert baseline.baseline_role in BASELINE_ROLES
        assert baseline.model_ref in model_keys


def test_baseline_task_families_link_to_slice2_matrix() -> None:
    task_keys = set(derive_matrix().family_statuses)

    for baseline in derive_baselines():
        assert baseline.task_family in task_keys


def test_blocked_or_unverified_baselines_have_reasons() -> None:
    for baseline in derive_baselines():
        if baseline.baseline_role in {"blocked", "unverified"}:
            assert baseline.blocker_reason
        if baseline.evidence_status in {"dependency-blocked", "failed", "skipped"}:
            assert baseline.blocker_reason


def test_baseline_mapping_validates_without_open_contract_issues() -> None:
    assert validate_baselines(derive_baselines()) == ()
