from __future__ import annotations

import pytest

from tools.repo.check_pr_base_policy import (
    HOTFIX_LABEL,
    MAIN_SYNC_LABEL,
    RELEASE_LABEL,
    evaluate_pr_base,
)


@pytest.mark.parametrize(
    "head",
    (
        "feat/new-capability",
        "fix/one-defect",
        "docs/one-authority",
        "test/one-contract",
        "ci/one-gate",
        "cleanup/one-ledger",
        "migration/one-source-sha",
        "research/one-protocol",
    ),
)
def test_routine_topics_target_dev(head: str) -> None:
    decision = evaluate_pr_base(base="dev", head=head, number=200)
    assert decision.allowed is True
    assert decision.code == "routine-dev-pr"


def test_routine_topic_cannot_target_main() -> None:
    decision = evaluate_pr_base(base="main", head="feat/new-capability", number=200)
    assert decision.allowed is False
    assert decision.code == "routine-pr-targets-main"


def test_release_promotion_requires_maintainer_label() -> None:
    denied = evaluate_pr_base(base="main", head="dev", number=201)
    assert denied.allowed is False

    allowed = evaluate_pr_base(
        base="main",
        head="dev",
        number=201,
        labels=[RELEASE_LABEL],
    )
    assert allowed.allowed is True
    assert allowed.code == "release-promotion"


def test_hotfix_requires_authorization_label() -> None:
    denied = evaluate_pr_base(base="main", head="hotfix/cve", number=202)
    assert denied.allowed is False

    allowed = evaluate_pr_base(
        base="main",
        head="hotfix/cve",
        number=202,
        labels=[HOTFIX_LABEL],
    )
    assert allowed.allowed is True
    assert allowed.code == "emergency-hotfix"


def test_transition_pr_127_is_narrowly_scoped() -> None:
    allowed = evaluate_pr_base(
        base="main",
        head="agent/v030-canonical-integration-r2",
        number=127,
    )
    assert allowed.allowed is True
    assert allowed.code == "transition-pr-127"

    wrong_number = evaluate_pr_base(
        base="main",
        head="agent/v030-canonical-integration-r2",
        number=128,
    )
    assert wrong_number.allowed is False

    wrong_head = evaluate_pr_base(base="main", head="other", number=127)
    assert wrong_head.allowed is False


def test_main_to_dev_sync_requires_label() -> None:
    denied = evaluate_pr_base(base="dev", head="main", number=203)
    assert denied.allowed is False

    allowed = evaluate_pr_base(
        base="dev",
        head="main",
        number=203,
        labels=[MAIN_SYNC_LABEL],
    )
    assert allowed.allowed is True
    assert allowed.code == "main-to-dev-sync"
