from __future__ import annotations

import sys
import types
from pathlib import Path

from scripts import config_inspect, validate_configs


def _write_minimal_config(tmp_path: Path, pipeline: str) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.csv").write_text("sample_id,label\n0,0\n", encoding="utf-8")

    config = tmp_path / "inspect.yaml"
    config.write_text(
        f"""
pipeline: "{pipeline}"
environment:
  output_dir: "{(tmp_path / "out").as_posix()}"
  iterations: 1
  seed: 1
data:
  data_dir: "{data_dir.as_posix()}"
  metadata_file: "metadata.csv"
model:
  type: "Dummy"
  name: "Dummy"
task:
  type: "Classification"
  name: "Dummy"
trainer:
  name: "Default_trainer"
  num_epochs: 2
""",
        encoding="utf-8",
    )
    return config


def test_config_inspect_reports_resolved_sources_targets_and_sanity(tmp_path, monkeypatch) -> None:
    module_name = "src.Pipeline_ConfigInspectTest"
    fake_module = types.ModuleType(module_name)
    fake_module.pipeline = lambda args: {"ok": True}
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    monkeypatch.setattr(config_inspect, "_maybe_import", lambda module: (True, ""))

    config = _write_minimal_config(tmp_path, "Pipeline_ConfigInspectTest")

    result = config_inspect.inspect_config(
        str(config),
        overrides=["trainer.num_epochs=3"],
        local_config=str(tmp_path / "missing_local.yaml"),
    )

    assert result.resolved["pipeline"] == "Pipeline_ConfigInspectTest"
    assert result.resolved["trainer"]["num_epochs"] == 3
    assert result.sources["pipeline"].startswith(f"config:{config.as_posix()}")
    assert result.sources["trainer.num_epochs"] == "cli:--override"
    assert result.targets["pipeline"]["module"] == "src.Pipeline_ConfigInspectTest"
    assert result.targets["factories"]["data_factory"].endswith("build_data")
    assert any(item["check"] == "preflight.pipeline_import" and item["ok"] for item in result.sanity)


def test_validate_configs_iterates_active_registry_rows(tmp_path) -> None:
    registry = tmp_path / "config_registry.csv"
    active = tmp_path / "active.yaml"
    skipped = tmp_path / "skipped.yaml"
    missing = tmp_path / "missing.yaml"
    active.write_text("pipeline: Pipeline_01_default\n", encoding="utf-8")
    skipped.write_text("pipeline: Pipeline_01_default\n", encoding="utf-8")
    registry.write_text(
        "path,status\n"
        f"{active.as_posix()},active\n"
        f"{skipped.as_posix()},/\n"
        f"{missing.as_posix()},active\n",
        encoding="utf-8",
    )

    paths = list(validate_configs.iter_registry_active_configs(registry))

    assert paths == [active, missing]
