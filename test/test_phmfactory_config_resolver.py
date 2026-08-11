from __future__ import annotations

from pathlib import Path

import pytest
from phmfactory.config import (
    MAINTAINED_PRESETS,
    load_config_dict,
    parse_overrides,
    resolve_config,
    resolve_config_path,
)
from phmfactory.pipelines import PipelineNameDeprecationWarning


def test_parse_overrides_expands_dotted_keys_and_yaml_types() -> None:
    assert parse_overrides(
        ["trainer.max_epochs=2", "task.enabled=true", "labels=[1, 2]"]
    ) == {
        "trainer": {"max_epochs": 2},
        "task": {"enabled": True},
        "labels": [1, 2],
    }


def test_parse_overrides_rejects_malformed_yaml_value() -> None:
    with pytest.raises(ValueError, match="Invalid YAML value"):
        parse_overrides(["labels=[1, 2"])


def test_parse_overrides_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        parse_overrides(["trainer.device="])


def test_recursive_base_config_merge_is_ordered(tmp_path: Path) -> None:
    base_a = tmp_path / "base_a.yaml"
    base_b = tmp_path / "base_b.yaml"
    child = tmp_path / "child.yaml"
    base_a.write_text("model:\n  width: 32\ntrainer:\n  epochs: 1\n", encoding="utf-8")
    base_b.write_text("model:\n  depth: 4\ntrainer:\n  epochs: 2\n", encoding="utf-8")
    child.write_text(
        "base_configs:\n"
        "  a: base_a.yaml\n"
        "  b: base_b.yaml\n"
        "pipeline: Pipeline_01_default\n"
        "model:\n  width: 64\n",
        encoding="utf-8",
    )

    resolved = load_config_dict(child)
    assert resolved["model"] == {"width": 64, "depth": 4}
    assert resolved["trainer"] == {"epochs": 2}


def test_resolve_config_canonicalizes_pipeline_and_applies_override(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "pipeline: Pipeline_01_default\ntrainer:\n  max_epochs: 5\n",
        encoding="utf-8",
    )

    with pytest.warns(PipelineNameDeprecationWarning):
        resolved = resolve_config(
            config,
            override_values=[
                "pipeline=Pipeline_04_unified_metric",
                "trainer.max_epochs=1",
            ],
        )

    assert resolved.pipeline == "Pipeline_04_Unified_Evaluation"
    assert resolved.data["pipeline"] == "Pipeline_04_Unified_Evaluation"
    assert resolved.data["trainer"]["max_epochs"] == 1
    assert resolved.path == config.resolve()


def test_explicit_config_requires_pipeline(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must declare `pipeline`"):
        resolve_config(config)


def test_pipeline_may_be_supplied_by_explicit_override(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")

    resolved = resolve_config(
        config,
        override_values=["pipeline=Pipeline_01_Fault_Diagnosis"],
    )

    assert resolved.pipeline == "Pipeline_01_Fault_Diagnosis"
    assert resolved.data["pipeline"] == "Pipeline_01_Fault_Diagnosis"


def test_non_utf8_config_fails_without_encoding_fallback(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_bytes(
        b"pipeline: Pipeline_01_Fault_Diagnosis\nnotes: \xff\n"
    )

    with pytest.raises(UnicodeError, match="must be valid UTF-8"):
        resolve_config(config)


def test_resolve_config_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("preset, relative_path", sorted(MAINTAINED_PRESETS.items()))
def test_maintained_presets_point_to_tracked_configs(
    preset: str,
    relative_path: str,
) -> None:
    expected = Path(relative_path).resolve()
    assert expected.is_file()
    assert resolve_config_path(preset) == expected


def test_installed_style_resolution_works_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_config("smoke")
    assert resolved.path.is_file()
    assert resolved.path.as_posix().endswith("configs/demo/00_smoke/dummy_dg.yaml")
    assert resolved.data["data"]["metadata_file"] == "metadata_dummy.csv"
    assert resolved.data["trainer"]["device"] == "cpu"


def test_cycle_detection(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base_configs:\n  second: second.yaml\n", encoding="utf-8")
    second.write_text("base_configs:\n  first: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Cyclic base_configs"):
        load_config_dict(first)
