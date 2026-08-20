from __future__ import annotations

from pathlib import Path

from phmfactory.commands import preflight
from phmfactory.config import analyze_config, resolve_config
from scripts.config_inspect import inspect_config
from scripts.validate_configs import validate_one


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPOSITORY_ROOT / "configs" / "demo" / "00_smoke" / "dummy_dg.yaml"


def _minimal_config(path: Path, *, epochs: int) -> None:
    path.write_text(
        "pipeline: Pipeline_01_Fault_Diagnosis\n"
        "environment:\n"
        "  project: parity-test\n"
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
        "trainer:\n"
        "  name: Default_trainer\n"
        f"  num_epochs: {epochs}\n"
        "  device: cpu\n",
        encoding="utf-8",
    )


def test_preset_and_explicit_path_have_same_effective_config() -> None:
    preset = analyze_config("smoke")
    explicit = analyze_config(SMOKE_CONFIG)

    assert preset.effective_config == explicit.effective_config
    assert preset.pipeline == explicit.pipeline


def test_equivalent_yaml_and_cli_override_have_same_effective_config(
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


def test_precedence_is_base_then_config_then_explicit_local_then_cli(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    local = tmp_path / "machine.yaml"
    base.write_text(
        "environment: {project: parity-test, seed: 0, iterations: 1, output_dir: results/test}\n"
        "data: {data_dir: data, metadata_file: metadata_dummy.csv}\n"
        "model: {type: Baseline, name: GlobalAverageLinear}\n"
        "task: {type: DG, name: classification, target_system_id: [0]}\n"
        "trainer: {name: Default_trainer, num_epochs: 1, device: cpu}\n",
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


def test_resolve_config_is_the_same_analysis_contract() -> None:
    analysis = analyze_config("smoke", override_values=["trainer.num_epochs=2"])
    resolved = resolve_config("smoke", override_values=["trainer.num_epochs=2"])

    assert resolved == analysis
    assert resolved.effective_config == analysis.effective_config


def test_inspector_and_public_analysis_return_same_config(
    tmp_path: Path,
) -> None:
    override = f"environment.output_dir={tmp_path / 'output'}"
    analysis = analyze_config("smoke", override_values=[override])
    inspected = inspect_config("smoke", overrides=[override])

    assert inspected.resolved == analysis.effective_config


def test_validator_accepts_the_same_maintained_smoke_config() -> None:
    assert validate_one(SMOKE_CONFIG) == []


def test_preflight_reports_the_same_pipeline_and_path(
    tmp_path: Path,
) -> None:
    override = f"environment.output_dir={tmp_path / 'preflight'}"
    expected = analyze_config("smoke", override_values=[override])

    report = preflight.run(["--config", "smoke", "--override", override])

    assert report["pipeline"] == expected.pipeline
    assert report["resolved_config_path"] == str(expected.path)
    assert report["output_dir"] == str((tmp_path / "preflight").resolve())
    assert "effective_config_sha256" not in report
    assert "run_spec_sha256" not in report
    assert not (tmp_path / "preflight").exists()
