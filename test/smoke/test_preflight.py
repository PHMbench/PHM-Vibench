from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import main


REPO = Path(__file__).resolve().parents[2]


def _run_main(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )


def test_preflight_accepts_dummy_default_config() -> None:
    result = _run_main("--config", "configs/demo/00_smoke/dummy_dg.yaml", "--preflight-only")

    assert result.returncode == 0, result.stderr
    assert "[OK] preflight passed" in result.stdout
    assert "Pipeline_01_default" in result.stdout


def test_preflight_accepts_dummy_generative_config() -> None:
    result = _run_main(
        "--config",
        "configs/demo/10_generative/dummy_generative_cfm.yaml",
        "--preflight-only",
    )

    assert result.returncode == 0, result.stderr
    assert "[OK] preflight passed" in result.stdout
    assert "Pipeline_06_generative" in result.stdout


def test_preflight_fails_on_malformed_yaml(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("pipeline: [\n", encoding="utf-8")

    result = _run_main("--config", str(config), "--preflight-only")

    assert result.returncode != 0
    assert "Malformed YAML config" in result.stderr


def test_preflight_fails_on_invalid_pipeline(tmp_path: Path) -> None:
    config = tmp_path / "bad_pipeline.yaml"
    config.write_text(
        "\n".join(
            [
                "pipeline: os",
                "environment: {project: bad, output_dir: results, seed: 0, iterations: 1}",
                "data: {data_dir: data, metadata_file: metadata_dummy.csv}",
                "model: {type: generative_model, name: phm_cfm_mlp1d}",
                "task: {type: generative, name: conditional_flow_matching, generative: {mode: train}}",
                "trainer: {name: Default_trainer, device: cpu, gpus: 0, num_epochs: 1}",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_main("--config", str(config), "--preflight-only")

    assert result.returncode != 0
    assert "Unsupported pipeline 'os'" in result.stderr


def test_preflight_fails_on_missing_required_section(tmp_path: Path) -> None:
    config = tmp_path / "missing_section.yaml"
    config.write_text(
        "\n".join(
            [
                "pipeline: Pipeline_01_default",
                "data: {data_dir: data, metadata_file: metadata_dummy.csv}",
                "model: {type: generative_model, name: phm_cfm_mlp1d}",
                "task: {type: generative, name: conditional_flow_matching, generative: {mode: train}}",
                "trainer: {name: Default_trainer, device: cpu, gpus: 0, num_epochs: 1}",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_main("--config", str(config), "--preflight-only")

    assert result.returncode != 0
    assert "missing required section" in result.stderr
    assert "environment" in result.stderr


def test_preflight_fails_on_generative_sample_without_checkpoint() -> None:
    result = _run_main(
        "--config",
        "configs/demo/10_generative/dummy_generative_cfm.yaml",
        "--preflight-only",
        "--override",
        "task.generative.mode=sample",
    )

    assert result.returncode != 0
    assert "sample mode requires checkpoint_path" in result.stderr


def test_preflight_accepts_explicit_untrained_sample_smoke() -> None:
    result = _run_main(
        "--config",
        "configs/demo/10_generative/dummy_generative_cfm.yaml",
        "--preflight-only",
        "--override",
        "task.generative.mode=sample",
        "--override",
        "task.generative.allow_untrained_smoke=true",
    )

    assert result.returncode == 0, result.stderr
    assert "[OK] preflight passed" in result.stdout


def test_preflight_accepts_generative_stage_ledger_path(tmp_path: Path) -> None:
    result = _run_main(
        "--config",
        "configs/demo/10_generative/dummy_generative_cfm.yaml",
        "--preflight-only",
        "--override",
        f"task.generative.stage_ledger_path={tmp_path / 'stage_ledger.json'}",
    )

    assert result.returncode == 0, result.stderr
    assert "[OK] preflight passed" in result.stdout


def test_preflight_does_not_import_pipeline_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class Args:
        config_path = "configs/demo/10_generative/dummy_generative_cfm.yaml"
        local_config = None
        override = None

    def fail_import_module(name: str):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(main.importlib, "import_module", fail_import_module)

    resolved = main.preflight(Args())

    assert resolved["pipeline"] == "Pipeline_06_generative"
