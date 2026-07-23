from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "repo" / "check_case_collisions.py"
SPEC = importlib.util.spec_from_file_location("check_case_collisions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _groups(paths: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {values for _, values in MODULE._collisions(paths)}


def test_detects_case_colliding_files() -> None:
    groups = _groups(("README.md", "readme.md"))
    assert ("README.md", "readme.md") in groups


def test_detects_case_colliding_directory_prefixes() -> None:
    groups = _groups(("Docs/guide.md", "docs/reference.md"))
    assert ("Docs", "docs") in groups


def test_detects_unicode_normalization_collisions() -> None:
    composed = "caf\u00e9/README.md"
    decomposed = "cafe\u0301/README.md"
    groups = _groups((composed, decomposed))
    assert tuple(sorted((composed, decomposed))) in groups


def test_accepts_portable_distinct_paths() -> None:
    assert MODULE._collisions(("docs/guide.md", "src/readme.py")) == ()
