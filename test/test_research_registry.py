from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from scripts.gen_research_atlas import render_atlas
from scripts.research_registry import REGISTRY_COLUMNS, read_registry, validate_rows


REGISTRY_PATH = Path("research/2025_2026/method_registry.csv")


def test_repository_research_registry_is_valid():
    rows = read_registry(REGISTRY_PATH)

    assert len(rows) >= 25
    assert validate_rows(rows) == []


def test_registry_rejects_preprint_runtime_promotion():
    row = read_registry(REGISTRY_PATH)[0]
    invalid = replace(
        row,
        publication_status="preprint",
        implementation_maturity="exploratory_runtime",
    )

    errors = validate_rows([invalid])

    assert any("preprint/submission cannot exceed research_only" in error for error in errors)


def test_registry_rejects_duplicate_ids():
    row = read_registry(REGISTRY_PATH)[0]

    errors = validate_rows([row, row])

    assert any("duplicate method_id" in error for error in errors)


def test_registry_rejects_code_without_license_status():
    row = next(row for row in read_registry(REGISTRY_PATH) if row.code_url)
    invalid = replace(row, code_license_status="not_applicable")

    errors = validate_rows([invalid])

    assert any("code URL cannot use not_applicable" in error for error in errors)


def test_registry_schema_is_exact(tmp_path):
    path = tmp_path / "registry.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS[:-1])
        writer.writeheader()

    try:
        read_registry(path)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("schema mismatch should fail")


def test_generated_atlas_is_deterministic():
    rows = read_registry(REGISTRY_PATH)

    first = render_atlas(rows, REGISTRY_PATH)
    second = render_atlas(list(reversed(rows)), REGISTRY_PATH)

    assert first == second
    assert "Registry presence is research evidence" in first
