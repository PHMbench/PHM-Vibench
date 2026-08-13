from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.utils.hse.prompt_validator import (
    HSEPromptConfigValidator,
    HSPPromptValidator,
)


def test_compatibility_alias_points_to_read_only_validator() -> None:
    assert HSEPromptConfigValidator is HSPPromptValidator


def test_validator_has_no_configuration_rewrite_surface() -> None:
    validator = HSPPromptValidator()

    assert not hasattr(validator, "fix_config")
    assert not hasattr(validator, "fix_yaml_file")
    assert not any(name.startswith("_fix_") for name in dir(validator))


def test_validation_does_not_modify_input_configuration() -> None:
    config = {
        "environment": {},
        "data": {"batch_size": -1},
        "model": {"fusion_strategy": "invalid"},
        "task": {"contrast_weight": 2.0},
        "trainer": {"max_epochs": 0},
    }
    original = deepcopy(config)

    valid, errors, _warnings = HSPPromptValidator().validate_config(config)

    assert not valid
    assert errors
    assert config == original


def test_yaml_validation_is_read_only(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("model:\n  fusion_strategy: invalid\n", encoding="utf-8")
    original = config_path.read_bytes()

    valid, errors, _warnings = HSPPromptValidator().validate_yaml_file(config_path)

    assert not valid
    assert errors
    assert config_path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [config_path]
