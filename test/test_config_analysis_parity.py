from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from phmfactory import cli as public_cli
from phmfactory.commands import preflight
from phmfactory.config import (
    analyze_config,
    load_config_dict,
    resolve_config,
    semantic_config_sha256,
    validate_complete_experiment,
)
from scripts.config_inspect import inspect_config
from scripts.validate_configs import validate_one
from src.runtime import classification


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPOSITORY_ROOT / "configs" / "demo" / "00_smoke" / "dummy_dg.yaml"


def _minimal_config(path: Path, *, epochs: int) -> None:
    path.write_text(
        "pipeline: Pipeline_01_Fault_Diagnosis\n"
        "environment:\n"
        "  project: config-parity-test\n"
        "  seed: 0\n"
        "  iterations: 1\n"
        "  output_dir: results/test\n"
        "data:\n"
        "  data_dir: data\n"
        "  metadata_file: metadata_dummy.csv\n"
        "model:\n"
        "  type: Baseline\n"
        "  name: GlobalAverageLinear\n"
        "task:\n"
        "  type: DG\n"
        "  name: classification\n"
        "  target_system_id: [0]\n"
        "  loss: CE\n"
        "trainer:\n"
        "  name: Default_trainer\n"
        f"  num_epochs: {epochs}\n"
        "  device: cpu\n"
        "  gpus: 1\n"
        "  monitor: val_loss\n"
        "  monitor_mode: min\n"
        "  test_after_fit: true\n",
        encoding="utf-8",
    )


def _remove_yaml_field(path: Path, field: str) -> None:
    prefix = f"  {field}:"
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not line.startswith(prefix)]
    assert len(kept) == len(lines) - 1, field
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def test_preset_and_explicit_path_have_same_effective_identity() -> None:
    preset = analyze_config("smoke")
    explicit = analyze_config(SMOKE_CONFIG)

    assert preset.effective_config == explicit.effective_config
    assert preset.effective_config_sha256 == explicit.effective_config_sha256


def test_equivalent_yaml_and_cli_override_have_same_effective_identity(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct.yaml"
    overridden = tmp_path / "overridden.yaml"
    _minimal_config(direct, epochs=2)
    _minimal_config(overridden, epochs=1)

    direct_analysis = analyze_config(direct)
    override_analysis = analyze_config(
        overridden,
        override_values=["trainer.num_epochs=2"],
    )

    assert direct_analysis.effective_config == override_analysis.effective_config
    assert (
        direct_analysis.effective_config_sha256
        == override_analysis.effective_config_sha256
    )


def test_precedence_is_base_then_config_then_explicit_local_then_cli(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    local = tmp_path / "machine.yaml"
    base.write_text(
        "environment:\n"
        "  project: config-precedence-test\n"
        "  seed: 0\n"
        "  iterations: 1\n"
        "  output_dir: results/test\n"
        "data:\n"
        "  data_dir: data\n"
        "  metadata_file: metadata_dummy.csv\n"
        "model:\n"
        "  type: Baseline\n"
        "  name: GlobalAverageLinear\n"
        "task:\n"
        "  type: DG\n"
        "  name: classification\n"
        "  target_system_id: [0]\n"
        "  loss: CE\n"
        "trainer:\n"
        "  name: Default_trainer\n"
        "  num_epochs: 1\n"
        "  device: cpu\n"
        "  gpus: 1\n"
        "  monitor: val_loss\n"
        "  monitor_mode: min\n",
        encoding="utf-8",
    )
    config.write_text(
        "base_configs:\n  common: base.yaml\n"
        "pipeline: Pipeline_01_Fault_Diagnosis\n"
        "trainer: {num_epochs: 2}\n",
        encoding="utf-8",
    )
    local.write_text("trainer: {num_epochs: 3, device: cuda}\n", encoding="utf-8")

    local_only = analyze_config(config, local_config=local)
    final = analyze_config(
        config,
        local_config=local,
        override_values=["trainer.num_epochs=4"],
    )

    assert local_only.effective_config["trainer"]["num_epochs"] == 3
    assert local_only.effective_config["trainer"]["device"] == "cuda"
    assert final.effective_config["trainer"]["num_epochs"] == 4
    assert final.effective_config["trainer"]["device"] == "cuda"
    assert final.sources["trainer.num_epochs"] == "cli:--override"
    assert final.sources["trainer.device"] == f"local:{local.resolve()}"
    assert final.local_config_path == local.resolve()


def test_unmentioned_local_yaml_is_not_an_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "experiment.yaml"
    _minimal_config(config, epochs=2)
    hidden = tmp_path / "configs" / "local" / "local.yaml"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("trainer: {num_epochs: 999}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    analysis = analyze_config(config)

    assert analysis.local_config_path is None
    assert analysis.effective_config["trainer"]["num_epochs"] == 2
    assert hidden not in analysis.source_files


def test_base_fragment_loader_remains_separate_from_complete_experiment_validation(
    tmp_path: Path,
) -> None:
    fragment = tmp_path / "trainer.yaml"
    fragment.write_text(
        "trainer:\n  name: Default_trainer\n  num_epochs: 1\n",
        encoding="utf-8",
    )

    assert load_config_dict(fragment) == {
        "trainer": {"name": "Default_trainer", "num_epochs": 1}
    }
    with pytest.raises(ValueError, match="must declare `pipeline`"):
        analyze_config(fragment)


def test_resolve_config_is_a_compatibility_view_of_analysis() -> None:
    analysis = analyze_config("smoke", override_values=["trainer.num_epochs=2"])
    resolved = resolve_config("smoke", override_values=["trainer.num_epochs=2"])

    assert resolved.data == analysis.effective_config
    assert resolved.pipeline == analysis.pipeline
    assert resolved.overrides == analysis.overrides
    assert resolved.path == analysis.path


def test_inspector_and_public_analysis_return_same_config_and_hash(
    tmp_path: Path,
) -> None:
    override = f"environment.output_dir={tmp_path / 'output'}"
    analysis = analyze_config("smoke", override_values=[override])
    inspected = inspect_config("smoke", overrides=[override])

    assert inspected.resolved == analysis.effective_config
    assert inspected.effective_config_sha256 == analysis.effective_config_sha256


def test_validator_accepts_the_same_maintained_smoke_config() -> None:
    assert validate_one(SMOKE_CONFIG) == []


def test_preflight_reports_the_same_effective_hash(
    tmp_path: Path,
) -> None:
    override = f"environment.output_dir={tmp_path / 'preflight'}"
    expected = analyze_config("smoke", override_values=[override])

    report = preflight.run(["--config", "smoke", "--override", override])

    assert report["effective_config_sha256"] == expected.effective_config_sha256
    assert report["pipeline"] == expected.pipeline
    assert not (tmp_path / "preflight").exists()


def test_strict_validation_does_not_rewrite_the_visible_mapping() -> None:
    config = analyze_config("smoke").runtime_config()
    before = deepcopy(config)

    validate_complete_experiment(config)

    assert config == before


@pytest.mark.parametrize(
    "invalid_override",
    [
        'environment.iterations="1"',
        'trainer.test_after_fit="false"',
        'data.num_workers="0"',
    ],
)
def test_public_entrypoints_share_strict_type_rejection_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_override: str,
) -> None:
    output_dir = tmp_path / "must-not-exist"
    output_override = f"environment.output_dir={output_dir}"
    overrides = [output_override, invalid_override]

    def fail_pipeline_import(*args, **kwargs):
        del args, kwargs
        pytest.fail("invalid config must fail before Pipeline import")

    monkeypatch.setattr(public_cli.importlib, "import_module", fail_pipeline_import)

    calls = [
        lambda: analyze_config("smoke", override_values=overrides),
        lambda: inspect_config("smoke", overrides=overrides),
        lambda: preflight.run(
            [
                "--config",
                "smoke",
                "--override",
                output_override,
                "--override",
                invalid_override,
            ]
        ),
        lambda: public_cli.run(
            public_cli.build_parser().parse_args(
                [
                    "--config",
                    "smoke",
                    "--override",
                    output_override,
                    "--override",
                    invalid_override,
                ]
            )
        ),
    ]

    for call in calls:
        with pytest.raises(ValidationError):
            call()

    assert not output_dir.exists()


@pytest.mark.parametrize("field", ("seed", "iterations"))
def test_environment_execution_fields_are_required_at_the_shared_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    config = tmp_path / f"missing-{field}.yaml"
    _minimal_config(config, epochs=1)
    output_dir = tmp_path / "must-not-exist"
    text = config.read_text(encoding="utf-8").replace(
        "  output_dir: results/test",
        f"  output_dir: {output_dir}",
    )
    config.write_text(text, encoding="utf-8")
    _remove_yaml_field(config, field)

    def fail_pipeline_import(*args, **kwargs):
        del args, kwargs
        pytest.fail("missing environment field must fail before Pipeline import")

    monkeypatch.setattr(public_cli.importlib, "import_module", fail_pipeline_import)

    validation_errors = validate_one(config)
    assert validation_errors
    assert f"environment.{field}" in "\n".join(validation_errors)

    calls = [
        lambda: analyze_config(config),
        lambda: inspect_config(config),
        lambda: preflight.run(["--config", str(config)]),
        lambda: public_cli.run(
            public_cli.build_parser().parse_args(["--config", str(config)])
        ),
    ]
    for call in calls:
        with pytest.raises(ValidationError):
            call()

    assert not output_dir.exists()


@pytest.mark.parametrize("field", ("seed", "iterations"))
def test_classification_runtime_has_no_missing_environment_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    output_dir = tmp_path / "must-not-exist"
    environment = SimpleNamespace(
        project="runtime-environment-contract",
        seed=7,
        iterations=1,
        output_dir=str(output_dir),
    )
    delattr(environment, field)
    configs = SimpleNamespace(
        environment=environment,
        data=SimpleNamespace(),
        model=SimpleNamespace(),
        task=SimpleNamespace(name="classification"),
        trainer=SimpleNamespace(test_after_fit=True),
    )
    monkeypatch.setattr(classification, "load_runtime_config", lambda args: configs)
    monkeypatch.setattr(
        classification,
        "build_data",
        lambda *args, **kwargs: pytest.fail(
            "missing environment field must fail before Data Factory construction"
        ),
    )

    with pytest.raises(ValueError, match=rf"environment\.{field} is required"):
        classification.run_classification_pipeline(object())

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "field,value",
    (("seed", "7"), ("iterations", "1")),
)
def test_classification_runtime_does_not_coerce_environment_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    output_dir = tmp_path / "must-not-exist"
    environment = SimpleNamespace(
        project="runtime-environment-contract",
        seed=7,
        iterations=1,
        output_dir=str(output_dir),
    )
    setattr(environment, field, value)
    configs = SimpleNamespace(
        environment=environment,
        data=SimpleNamespace(),
        model=SimpleNamespace(),
        task=SimpleNamespace(name="classification"),
        trainer=SimpleNamespace(test_after_fit=True),
    )
    monkeypatch.setattr(classification, "load_runtime_config", lambda args: configs)
    monkeypatch.setattr(
        classification,
        "build_data",
        lambda *args, **kwargs: pytest.fail(
            "invalid environment type must fail before Data Factory construction"
        ),
    )

    with pytest.raises(TypeError, match=rf"environment\.{field} must be an integer"):
        classification.run_classification_pipeline(object())

    assert not output_dir.exists()


def test_grouped_split_coupling_fails_at_the_shared_public_boundary(
    tmp_path: Path,
) -> None:
    config = tmp_path / "invalid-grouped.yaml"
    _minimal_config(config, epochs=1)
    with config.open("a", encoding="utf-8") as handle:
        handle.write(
            "data:\n"
            "  data_dir: data\n"
            "  metadata_file: metadata_dummy.csv\n"
            "  split:\n"
            "    strategy: grouped_metadata\n"
            "    group_key: File\n"
            "    manifest_path: results/split.json\n"
            "    test_policy: partition\n"
            "    fractions: {train: 0.8, val: 0.1, test: 0.1}\n"
        )

    with pytest.raises(ValidationError, match="requires test_policy=task_defined"):
        analyze_config(config)


def test_semantic_hash_is_stable_for_mapping_order() -> None:
    left = {"pipeline": "P", "trainer": {"device": "cpu", "epochs": 1}}
    right = {"trainer": {"epochs": 1, "device": "cpu"}, "pipeline": "P"}
    assert semantic_config_sha256(left) == semantic_config_sha256(right)
