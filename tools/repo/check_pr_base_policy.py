#!/usr/bin/env python3
"""Enforce the documented main/dev pull-request topology.

The policy is deliberately independent of GitHub branch-name inference for
privileged main-branch operations: release promotions and emergency hotfixes
also require maintainer-controlled labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

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
    number: int,
    labels: Iterable[str] = (),
) -> PolicyDecision:
    """Return one deterministic decision for the supplied PR topology."""

    base = str(base).strip()
    head = str(head).strip()
    label_set = _normalise_labels(labels)

    if base == "main":
        if number == TRANSITION_PR and head == "agent/v030-canonical-integration-r2":
            return PolicyDecision(
                True,
                "transition-pr-127",
                "Canonical v0.3 PR #127 is the sole documented transition exception.",
            )
        if (head == "dev" or head.startswith("release/")) and RELEASE_LABEL in label_set:
            return PolicyDecision(
                True,
                "release-promotion",
                "Authorized release promotion into main.",
            )
        if head.startswith("hotfix/") and HOTFIX_LABEL in label_set:
            return PolicyDecision(
                True,
                "emergency-hotfix",
                "Authorized emergency hotfix; a back-sync PR to dev is required.",
            )
        return PolicyDecision(
            False,
            "routine-pr-targets-main",
            "Routine work must target dev. Main accepts only labelled release promotions, "
            "labelled emergency hotfixes, and transition PR #127.",
        )

    if base == "dev":
        if head == "main" and MAIN_SYNC_LABEL in label_set:
            return PolicyDecision(
                True,
                "main-to-dev-sync",
                "Authorized synchronization of the post-release main ancestry into dev.",
            )
        if any(head.startswith(prefix) for prefix in DEV_PREFIXES):
            return PolicyDecision(True, "routine-dev-pr", "Routine topic PR correctly targets dev.")
        return PolicyDecision(
            False,
            "unsupported-dev-source",
            "Dev accepts focused topic branches or a labelled main-to-dev synchronization PR.",
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
        number=args.number,
        labels=labels,
    )
    status = "ALLOW" if decision.allowed else "DENY"
    print(f"{status} [{decision.code}] {decision.message}")
    return 0 if decision.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
