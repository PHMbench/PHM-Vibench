from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

from pydantic import ValidationError
import pytest
import yaml

from phmfactory import cli as public_cli
from phmfactory.commands import preflight
from phmfactory.config import analyze_config, parse_overrides, validate_complete_experiment
from scripts.config_inspect import inspect_config
from scripts.validate_configs import validate_one


def _write_variant(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict], None],
) -> tuple[Path, Path]:
    config = analyze_config("smoke").runtime_config()
    output_dir = tmp_path / "must-not-exist"
    config["environment"]["output_dir"] = str(output_dir)
    mutate(config)
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path, output_dir


@pytest.mark.parametrize("field", ("num_epochs", "test_after_fit"))
def test_classification_lifecycle_fields_are_required_before_pipeline_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    config_path, output_dir = _write_variant(
        tmp_path,
        f"missing-{field}",
        lambda config: config["trainer"].pop(field),
    )

    def fail_pipeline_import(*args, **kwargs):
        del args, kwargs
        pytest.fail("invalid lifecycle config must fail before Pipeline import")

    monkeypatch.setattr(public_cli.importlib, "import_module", fail_pipeline_import)

    validation_errors = validate_one(config_path)
    assert validation_errors
    assert field in "\n".join(validation_errors)

    calls = [
        lambda: analyze_config(config_path),
        lambda: inspect_config(config_path),
        lambda: preflight.run(["--config", str(config_path)]),
        lambda: public_cli.run(
            public_cli.build_parser().parse_args(
                ["--config", str(config_path)]
            )
        ),
    ]
    for call in calls:
        with pytest.raises(ValidationError):
            call()

    assert not output_dir.exists()


def test_legacy_max_epochs_cannot_coexist_as_a_second_epoch_authority(
    tmp_path: Path,
) -> None:
    config_path, output_dir = _write_variant(
        tmp_path,
        "legacy-max-epochs",
        lambda config: config["trainer"].__setitem__("max_epochs", 2),
    )

    with pytest.raises(ValidationError, match="trainer.max_epochs is unsupported"):
        analyze_config(config_path)

    assert not output_dir.exists()


def test_explicit_false_is_a_valid_visible_classification_policy(
    tmp_path: Path,
) -> None:
    config_path, _ = _write_variant(
        tmp_path,
        "training-only",
        lambda config: config["trainer"].__setitem__("test_after_fit", False),
    )

    analysis = analyze_config(config_path)

    assert analysis.effective_config["trainer"]["test_after_fit"] is False


def test_pipeline06_does_not_receive_unused_classification_policy() -> None:
    analysis = analyze_config("configs/demo/10_generative/dummy_generative_cfm.yaml")

    assert analysis.pipeline == "Pipeline_06_Generative_Modeling"
    assert "test_after_fit" not in analysis.effective_config["trainer"]


@pytest.mark.parametrize("token", [
    "trainer..num_epochs=2",
    ".trainer.num_epochs=2",
    "trainer.num_epochs.=2",
    "trainer. .num_epochs=2",
    "trainer. num_epochs=2",
])
def test_empty_or_padded_override_segments_are_rejected(token):
    with pytest.raises(ValueError, match="Invalid override path"):
        parse_overrides([token])


@pytest.mark.parametrize("token,exception,detail", [
    ("trainer.num_epoch=2", ValidationError, "num_epoch"),
    ("trainer..num_epochs=2", ValueError, "Invalid override path"),
])
def test_invalid_overrides_fail_at_each_public_entry_before_execution(
    tmp_path, monkeypatch, token, exception, detail,
):
    config_path, output_dir = _write_variant(tmp_path, "valid-input", lambda config: None)

    def fail_pipeline_import(*args, **kwargs):
        pytest.fail("invalid override must fail before Pipeline import")

    monkeypatch.setattr(public_cli.importlib, "import_module", fail_pipeline_import)
    argv = ["--config", str(config_path), "--override", token]
    calls = [
        lambda: analyze_config(config_path, override_values=[token]),
        lambda: inspect_config(str(config_path), overrides=[token]),
        lambda: preflight.run(argv),
        lambda: public_cli.run(public_cli.build_parser().parse_args(argv)),
    ]
    for call in calls:
        with pytest.raises(exception, match=detail):
            call()
    assert not output_dir.exists()


def test_unknown_default_trainer_yaml_field_is_not_an_ignored_setting(tmp_path):
    config_path, output_dir = _write_variant(
        tmp_path, "misspelled-epoch",
        lambda config: config["trainer"].__setitem__("num_epoch", 2),
    )
    errors = validate_one(config_path)
    assert errors and "num_epoch" in "\n".join(errors)
    with pytest.raises(ValidationError, match="Supported fields:.*num_epochs"):
        analyze_config(config_path)
    assert not output_dir.exists()


def test_valid_optional_trainer_fields_need_not_exist_in_base_yaml(tmp_path):
    config_path, _ = _write_variant(
        tmp_path, "without-optional-selection",
        lambda config: [config["trainer"].pop(key, None) for key in ("min_delta", "save_top_k")],
    )
    config = analyze_config(config_path).runtime_config()
    before = deepcopy(config)
    validate_complete_experiment(config)
    assert config == before
    assert "min_delta" not in config["trainer"]
    assert "save_top_k" not in config["trainer"]

    changed = analyze_config(config_path, override_values=[
        "trainer.min_delta=0.0", "trainer.save_top_k=2",
        "trainer.early_stopping=false", "trainer.deterministic=warn",
    ]).runtime_config()
    assert changed["trainer"]["min_delta"] == 0.0
    assert changed["trainer"]["save_top_k"] == 2
    assert changed["trainer"]["early_stopping"] is False
    assert changed["trainer"]["deterministic"] == "warn"
    assert changed["trainer"]["num_epochs"] == before["trainer"]["num_epochs"]


def test_custom_trainer_and_research_component_extras_remain_explicit():
    config = analyze_config("smoke", override_values=[
        "trainer.name=ResearchTrainer",
        "trainer.research_control.alpha=0.5",
        "model.research_parameter=7",
        "task.research_parameter=false",
    ]).runtime_config()
    assert config["trainer"]["research_control"] == {"alpha": 0.5}
    assert config["model"]["research_parameter"] == 7
    assert config["task"]["research_parameter"] is False
    # Configuration acceptance is not a claim that this example Trainer is installed.


def test_supported_repeated_override_keeps_last_value_and_strict_types():
    config = analyze_config("smoke", override_values=[
        "trainer.num_epochs=1", "trainer.num_epochs=2",
        "trainer.test_after_fit=false",
    ]).runtime_config()
    assert config["trainer"]["num_epochs"] == 2
    assert "num_epoch" not in config["trainer"]
    assert config["trainer"]["test_after_fit"] is False
    for token in ("trainer.num_epochs='2'", "trainer.early_stopping='false'"):
        with pytest.raises(ValidationError):
            analyze_config("smoke", override_values=[token])
