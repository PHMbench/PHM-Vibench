from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import ValidationError
import pytest
import yaml

from phmfactory import cli as public_cli
from phmfactory.commands import preflight
from phmfactory.config import analyze_config
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
