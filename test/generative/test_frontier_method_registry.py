from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.task_factory.Components.generative.registry import (
    export_registry_snapshot,
    get_frontier_method,
    load_frontier_method_registry,
)


def test_frontier_registry_loads_all_methods() -> None:
    registry = load_frontier_method_registry()

    assert len(registry) == 10
    assert registry["ot_nfm"].claim_status == "exploratory"
    assert registry["physical_field_one_step_fm"].blocks_benchmark_valid is True


def test_frontier_registry_rejects_invalid_status(tmp_path: Path) -> None:
    data = yaml.safe_load(Path("configs/registry/generative_frontier_methods.yaml").read_text())
    data["methods"]["ot_nfm"]["claim_status"] = "paper-ready"
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid claim_status"):
        load_frontier_method_registry(path)


def test_frontier_registry_unknown_method_fails_fast() -> None:
    with pytest.raises(KeyError, match="unknown frontier method"):
        get_frontier_method("not_a_method")


def test_frontier_registry_snapshot_is_exportable() -> None:
    snapshot = export_registry_snapshot()
    encoded = json.dumps(snapshot)

    assert "ot_nfm" in encoded
    assert all("blocks_benchmark_valid" in row for row in snapshot)


def test_paperpack_writes_frontier_registry_snapshot(tmp_path: Path) -> None:
    from scripts.paperpack_generative import build_paperpack

    paperpack = build_paperpack(tmp_path)
    snapshot_path = paperpack / "appendix" / "frontier_method_registry.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert snapshot_path.is_file()
    assert len(snapshot) == 10
    assert {row["method_id"] for row in snapshot} >= {"ot_nfm", "euler_mean_flow"}
