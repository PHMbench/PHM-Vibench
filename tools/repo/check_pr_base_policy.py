#!/usr/bin/env python3
"""Enforce the documented main/dev pull-request topology.

Privileged operations require both an auditable maintainer-controlled label and
canonical repository identity. A fork may contribute an ordinary topic branch to
``dev``, but it cannot impersonate this repository's long-lived branches merely by
using the same branch name.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

CANONICAL_REPOSITORY = "PHMbench/PHM-Vibench"
RELEASE_LABEL = "release-promotion-approved"
HOTFIX_LABEL = "emergency-hotfix-approved"
MAIN_SYNC_LABEL = "main-sync-approved"
TRANSITION_PR = 127
DEV_PREFIXES = (
    "feat/",
    "fix/",
    "docs/",
    "test/",
    "ci/",
    "cleanup/",
    "migration/",
    "research/",
    "release/",
)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    message: str


def _normalise_labels(labels: Iterable[str]) -> frozenset[str]:
    return frozenset(str(label).strip() for label in labels if str(label).strip())


def evaluate_pr_base(
    *,
    base: str,
    head: str,
    base_repo: str,
    head_repo: str,
    number: int,
    labels: Iterable[str] = (),
) -> PolicyDecision:
    """Return one deterministic decision for the supplied PR topology."""

    base = str(base).strip()
    head = str(head).strip()
    base_repo = str(base_repo).strip()
    head_repo = str(head_repo).strip()
    label_set = _normalise_labels(labels)
    base_is_canonical = base_repo == CANONICAL_REPOSITORY
    same_canonical_repository = base_is_canonical and head_repo == CANONICAL_REPOSITORY

    if base in {"main", "dev"} and not base_is_canonical:
        return PolicyDecision(
            False,
            "unexpected-base-repository",
            f"Managed base {base!r} must belong to {CANONICAL_REPOSITORY}.",
        )

    if base == "main":
        if (
            number == TRANSITION_PR
            and head == "agent/v030-canonical-integration-r2"
            and same_canonical_repository
        ):
            return PolicyDecision(
                True,
                "transition-pr-127",
                "Canonical v0.3 PR #127 is the sole documented transition exception.",
            )
        if (
            (head == "dev" or head.startswith("release/"))
            and RELEASE_LABEL in label_set
            and same_canonical_repository
        ):
            return PolicyDecision(
                True,
                "release-promotion",
                "Authorized same-repository release promotion into main.",
            )
        if (
            head.startswith("hotfix/")
            and HOTFIX_LABEL in label_set
            and same_canonical_repository
        ):
            return PolicyDecision(
                True,
                "emergency-hotfix",
                "Authorized same-repository emergency hotfix; a back-sync PR to dev is required.",
            )
        return PolicyDecision(
            False,
            "routine-pr-targets-main",
            "Routine work must target dev. Main accepts only labelled same-repository "
            "release promotions, labelled same-repository emergency hotfixes, and the "
            "exact transition PR #127.",
        )

    if base == "dev":
        if (
            head == "main"
            and MAIN_SYNC_LABEL in label_set
            and same_canonical_repository
        ):
            return PolicyDecision(
                True,
                "main-to-dev-sync",
                "Authorized same-repository synchronization of main ancestry into dev.",
            )
        if any(head.startswith(prefix) for prefix in DEV_PREFIXES):
            return PolicyDecision(
                True,
                "routine-dev-pr",
                "Routine topic PR correctly targets dev; fork topic branches are permitted.",
            )
        return PolicyDecision(
            False,
            "unsupported-dev-source",
            "Dev accepts focused topic branches or a labelled same-repository main-to-dev sync.",
        )

    return PolicyDecision(
        True,
        "unmanaged-base",
        f"Base branch {base!r} is outside the main/dev governance workflow.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--base-repo", required=True)
    parser.add_argument("--head-repo", required=True)
    parser.add_argument("--number", required=True, type=int)
    parser.add_argument(
        "--labels-json",
        default="[]",
        help="JSON array containing pull-request label names.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        labels = json.loads(args.labels_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"labels-json must be valid JSON: {exc}") from exc
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise SystemExit("labels-json must be a JSON array of strings")

    decision = evaluate_pr_base(
        base=args.base,
        head=args.head,
        base_repo=args.base_repo,
        head_repo=args.head_repo,
        number=args.number,
        labels=labels,
    )
    status = "ALLOW" if decision.allowed else "DENY"
    print(f"{status} [{decision.code}] {decision.message}")
    return 0 if decision.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
