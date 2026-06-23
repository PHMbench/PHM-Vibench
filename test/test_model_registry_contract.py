from __future__ import annotations

import importlib.util
from pathlib import Path

from scripts.model_support_matrix import (
    SUPPORT_STATUSES,
    derive_model_support,
    load_component_entries,
    load_model_entries,
    maintained_isfm_component_keys,
)


def test_model_registry_rows_have_unique_statuses_and_resolvable_paths() -> None:
    report = derive_model_support()

    assert report.duplicate_model_keys == ()
    assert set(report.model_statuses) == {entry.key for entry in load_model_entries()}
    assert all(item.status in SUPPORT_STATUSES for item in report.model_statuses.values())
    assert all(item.reason for item in report.model_statuses.values())

    for entry in load_model_entries():
        if report.model_statuses[entry.key].status != "failed":
            assert Path(entry.module_path).exists(), entry.key


def test_isfm_component_rows_have_unique_statuses_and_resolvable_paths() -> None:
    report = derive_model_support()

    assert report.duplicate_component_keys == ()
    assert set(report.component_statuses) == {entry.key for entry in load_component_entries()}
    assert all(item.status in SUPPORT_STATUSES for item in report.component_statuses.values())
    assert all(item.reason for item in report.component_statuses.values())

    for entry in load_component_entries():
        if report.component_statuses[entry.key].status != "failed":
            assert Path(entry.module_path).exists(), entry.key


def test_optional_dependency_gap_is_dependency_blocked_not_passing() -> None:
    report = derive_model_support()
    status = report.model_statuses[("X_model", "CI_GNN")]

    if importlib.util.find_spec("torch_geometric") is None:
        assert status.status == "dependency-blocked"
        assert "torch_geometric" in status.reason
    else:
        assert status.status == "smoke-tested"


def test_maintained_isfm_demo_components_are_registered_and_smoke_tested() -> None:
    report = derive_model_support()
    keys = maintained_isfm_component_keys("configs/hydra/experiments/00_smoke/dummy_dg.yaml")

    for key in keys:
        assert key in report.component_statuses
        assert report.component_statuses[key].status == "smoke-tested"


def test_maintained_isfm_demo_model_is_smoke_tested() -> None:
    report = derive_model_support()

    status = report.model_statuses[("ISFM", "M_01_ISFM")]

    assert status.status == "smoke-tested"
    assert "dummy_dg.yaml" in status.reason
