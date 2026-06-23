from __future__ import annotations

import sys
import types

import pytest

import main as vibench_main


def _minimal_config_text(tmp_path, pipeline: str) -> str:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.csv").write_text("sample_id,label\n0,0\n", encoding="utf-8")
    output_dir = tmp_path / "results"
    return f"""
pipeline: "{pipeline}"
environment:
  output_dir: "{output_dir.as_posix()}"
  iterations: 1
  seed: 42
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
  num_epochs: 1
"""


def test_main_requires_config() -> None:
    with pytest.raises(SystemExit) as excinfo:
        vibench_main.main([])

    assert excinfo.value.code != 0


def test_main_rejects_missing_config_path(tmp_path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit) as excinfo:
        vibench_main.main(["--config", str(missing)])

    assert excinfo.value.code != 0


def test_main_rejects_config_without_pipeline(tmp_path) -> None:
    config = tmp_path / "no_pipeline.yaml"
    config.write_text("environment: {}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        vibench_main.main(["--config", str(config)])

    assert excinfo.value.code != 0


def test_main_rejects_unknown_pipeline(tmp_path) -> None:
    config = tmp_path / "bad_pipeline.yaml"
    config.write_text('pipeline: "Pipeline_DoesNotExist"\n', encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        vibench_main.main(["--config", str(config)])

    assert excinfo.value.code != 0


def test_main_rejects_invalid_override_syntax(tmp_path) -> None:
    config = tmp_path / "invalid_override.yaml"
    config.write_text(_minimal_config_text(tmp_path, "Pipeline_TestStrict"), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        vibench_main.main(["--config", str(config), "--override", "trainer.num_epochs"])

    assert excinfo.value.code != 0


def test_config_path_is_deprecated_but_compatible(tmp_path, monkeypatch) -> None:
    module_name = "src.Pipeline_TestStrict"
    fake_module = types.ModuleType(module_name)
    called = {}

    def pipeline(args):
        called["config_path"] = args.config_path
        return {"ok": True}

    fake_module.pipeline = pipeline
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    config = tmp_path / "ok.yaml"
    config.write_text(_minimal_config_text(tmp_path, "Pipeline_TestStrict"), encoding="utf-8")

    with pytest.warns(DeprecationWarning):
        result = vibench_main.main(["--config_path", str(config)])

    assert result == {"ok": True}
    assert called["config_path"] == str(config)
