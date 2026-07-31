from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.repo import check_release_readiness as readiness


@pytest.mark.parametrize(
    "revision",
    ("main", "master", "release/v0.3", "refs/heads/main", "v0.3.0", ""),
)
def test_mutable_revision_forms_are_not_immutable(revision: str) -> None:
    assert not readiness._is_immutable_revision(revision)


def test_full_commit_revision_is_immutable() -> None:
    assert readiness._is_immutable_revision("a" * 40)


def test_gitlinks_enumerates_raw_unconfigured_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (
        "100644 blob " + "1" * 40 + "\tREADME.md\n"
        "160000 commit " + "2" * 40 + "\tunconfigured/raw\n"
    )
    monkeypatch.setattr(
        readiness.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    assert readiness._gitlinks() == {"unconfigured/raw": "2" * 40}
