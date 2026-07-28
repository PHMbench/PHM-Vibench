from __future__ import annotations

import pytest

from tools.repo.check_pr_base_policy import (
    CANONICAL_REPOSITORY,
    HOTFIX_LABEL,
    MAIN_SYNC_LABEL,
    RELEASE_LABEL,
    evaluate_pr_base,
)

FORK_REPOSITORY = "attacker/PHM-Vibench"


def decide(
    *,
    base: str,
    head: str,
    number: int,
    labels=(),
    base_repo: str = CANONICAL_REPOSITORY,
    head_repo: str = CANONICAL_REPOSITORY,
):
    return evaluate_pr_base(
        base=base,
        head=head,
        base_repo=base_repo,
        head_repo=head_repo,
        number=number,
        labels=labels,
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
    decision = decide(base="dev", head=head, number=200)
    assert decision.allowed is True
    assert decision.code == "routine-dev-pr"


def test_ordinary_fork_topic_can_target_dev() -> None:
    decision = decide(
        base="dev",
        head="fix/external-contribution",
        head_repo=FORK_REPOSITORY,
        number=200,
    )
    assert decision.allowed is True
    assert decision.code == "routine-dev-pr"


def test_managed_base_must_belong_to_canonical_repository() -> None:
    decision = decide(
        base="dev",
        head="fix/external-contribution",
        base_repo=FORK_REPOSITORY,
        head_repo=FORK_REPOSITORY,
        number=200,
    )
    assert decision.allowed is False
    assert decision.code == "unexpected-base-repository"


def test_routine_topic_cannot_target_main() -> None:
    decision = decide(base="main", head="feat/new-capability", number=200)
    assert decision.allowed is False
    assert decision.code == "routine-pr-targets-main"


def test_release_promotion_requires_maintainer_label_and_same_repo() -> None:
    denied = decide(base="main", head="dev", number=201)
    assert denied.allowed is False

    allowed = decide(
        base="main",
        head="dev",
        number=201,
        labels=[RELEASE_LABEL],
    )
    assert allowed.allowed is True
    assert allowed.code == "release-promotion"

    fork = decide(
        base="main",
        head="dev",
        head_repo=FORK_REPOSITORY,
        number=201,
        labels=[RELEASE_LABEL],
    )
    assert fork.allowed is False


@pytest.mark.parametrize("head", ("dev", "release/v0.3.0"))
def test_fork_cannot_impersonate_release_source(head: str) -> None:
    decision = decide(
        base="main",
        head=head,
        head_repo=FORK_REPOSITORY,
        number=204,
        labels=[RELEASE_LABEL],
    )
    assert decision.allowed is False
    assert decision.code == "routine-pr-targets-main"


def test_hotfix_requires_authorization_label_and_same_repo() -> None:
    denied = decide(base="main", head="hotfix/cve", number=202)
    assert denied.allowed is False

    allowed = decide(
        base="main",
        head="hotfix/cve",
        number=202,
        labels=[HOTFIX_LABEL],
    )
    assert allowed.allowed is True
    assert allowed.code == "emergency-hotfix"

    fork = decide(
        base="main",
        head="hotfix/cve",
        head_repo=FORK_REPOSITORY,
        number=202,
        labels=[HOTFIX_LABEL],
    )
    assert fork.allowed is False


def test_transition_pr_127_is_narrowly_scoped_to_same_repo() -> None:
    allowed = decide(
        base="main",
        head="agent/v030-canonical-integration-r2",
        number=127,
    )
    assert allowed.allowed is True
    assert allowed.code == "transition-pr-127"

    wrong_number = decide(
        base="main",
        head="agent/v030-canonical-integration-r2",
        number=128,
    )
    assert wrong_number.allowed is False

    wrong_head = decide(base="main", head="other", number=127)
    assert wrong_head.allowed is False

    fork = decide(
        base="main",
        head="agent/v030-canonical-integration-r2",
        head_repo=FORK_REPOSITORY,
        number=127,
    )
    assert fork.allowed is False


def test_main_to_dev_sync_requires_label_and_same_repo() -> None:
    denied = decide(base="dev", head="main", number=203)
    assert denied.allowed is False

    allowed = decide(
        base="dev",
        head="main",
        number=203,
        labels=[MAIN_SYNC_LABEL],
    )
    assert allowed.allowed is True
    assert allowed.code == "main-to-dev-sync"

    fork = decide(
        base="dev",
        head="main",
        head_repo=FORK_REPOSITORY,
        number=203,
        labels=[MAIN_SYNC_LABEL],
    )
    assert fork.allowed is False
    assert fork.code == "unsupported-dev-source"
