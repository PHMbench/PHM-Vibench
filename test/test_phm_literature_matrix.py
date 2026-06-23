from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.phm_literature_matrix import (
    InventoryValidationError,
    SUPPORT_STATUSES,
    LiteratureEntry,
    load_inventory,
    validate_inventory,
)


def test_default_inventory_contains_at_least_50_post_2025_entries() -> None:
    entries = load_inventory()
    report = validate_inventory(entries, min_count=50)

    assert report.total_entries >= 50
    assert report.min_year >= 2025
    assert "rul" in report.counts_by_task_family
    assert "fault_diagnosis" in report.counts_by_task_family
    assert "domain_generalization" in report.counts_by_task_family
    assert "few_shot" in report.counts_by_task_family
    assert len(report.counts_by_method_family) >= 8
    assert set(report.counts_by_support_status).issubset(SUPPORT_STATUSES)


def test_inventory_rows_have_required_mapping_fields() -> None:
    for entry in load_inventory():
        assert entry.id.startswith("PHM2025-")
        assert entry.title
        assert entry.venue
        assert entry.url.startswith("https://")
        assert entry.task_family
        assert entry.method_family
        assert entry.repo_surface
        assert entry.support_status in SUPPORT_STATUSES


def test_validator_rejects_old_entries() -> None:
    entry = LiteratureEntry(
        id="PHM2024-001",
        year=2024,
        title="Old PHM paper",
        authors="Example",
        venue="Example",
        url="https://example.com/old",
        doi="",
        task_family="fault_diagnosis",
        method_family="cnn",
        repo_surface="model_factory.CNN",
        support_status="represented",
        notes="",
    )

    with pytest.raises(InventoryValidationError, match="older than 2025"):
        validate_inventory([entry], min_count=1)


def test_validator_rejects_duplicate_titles_and_urls() -> None:
    entries = [
        LiteratureEntry(
            id="PHM2025-A",
            year=2025,
            title="Duplicate Title",
            authors="A",
            venue="V",
            url="https://example.com/a",
            doi="",
            task_family="fault_diagnosis",
            method_family="cnn",
            repo_surface="model_factory.CNN",
            support_status="represented",
            notes="",
        ),
        LiteratureEntry(
            id="PHM2025-B",
            year=2025,
            title=" duplicate   title ",
            authors="B",
            venue="V",
            url="https://example.com/a",
            doi="",
            task_family="rul",
            method_family="rnn",
            repo_surface="model_factory.RNN",
            support_status="candidate-baseline",
            notes="",
        ),
    ]

    with pytest.raises(InventoryValidationError, match="duplicate titles"):
        validate_inventory(entries, min_count=1)


def test_loader_rejects_missing_required_fields(tmp_path: Path) -> None:
    inventory = tmp_path / "bad.csv"
    with inventory.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "year",
                "title",
                "authors",
                "venue",
                "url",
                "doi",
                "task_family",
                "method_family",
                "repo_surface",
                "support_status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "PHM2025-BAD",
                "year": "2025",
                "title": "",
                "authors": "Example",
                "venue": "Example",
                "url": "https://example.com/bad",
                "doi": "",
                "task_family": "fault_diagnosis",
                "method_family": "cnn",
                "repo_surface": "model_factory.CNN",
                "support_status": "represented",
                "notes": "",
            }
        )

    with pytest.raises(InventoryValidationError, match="missing required fields"):
        load_inventory(inventory)


def test_validator_rejects_invalid_support_status() -> None:
    entry = LiteratureEntry(
        id="PHM2025-STATUS",
        year=2025,
        title="Status test",
        authors="Example",
        venue="Example",
        url="https://example.com/status",
        doi="",
        task_family="fault_diagnosis",
        method_family="cnn",
        repo_surface="model_factory.CNN",
        support_status="smoke-tested",
        notes="",
    )

    with pytest.raises(InventoryValidationError, match="invalid support_status"):
        validate_inventory([entry], min_count=1)
