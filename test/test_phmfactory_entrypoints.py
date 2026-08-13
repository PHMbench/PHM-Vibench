from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples import cwru_quickstart
from phmfactory import __version__, cli
from phmfactory.config import (
    MAINTAINED_PRESETS,
    ConfigAnalysis,
    resolve_config_path,
    semantic_config_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _analysis(
    tmp_path: Path,
    *,
    requested: str,
    pipeline: str,
    overrides: dict | None = None,
) -> ConfigAnalysis:
    config = {
        "pipeline": pipeline,
        "environment": {"output_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "config.yaml"
    return ConfigAnalysis(
        requested=requested,
        path=path,
        effective_config=config,
        pipeline=pipeline,
        overrides=overrides or {},
        local_config_path=None,
        source_files=(path,),
        sources={},
        diagnostics=(),
        effective_config_sha256=semantic_config_sha256(config),
    )


def test_public_version_is_v030_development_release() -> None:
    assert __version__ == "0.3.0.dev0"


def test_parser_preserves_legacy_config_path_alias() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--config_path", "legacy.yaml", "--notes", "compat"])
    assert args.config is None
    assert args.config_path == "legacy.yaml"
    assert args.local_config is None
    assert args.notes == "compat"
    assert args.allow_experimental is False


def test_parser_accepts_explicit_local_config() -> None:
    args = cli.build_parser().parse_args(
        ["--config", "experiment.yaml", "--local-config", "machine.yaml"]
    )
    assert args.local_config == "machine.yaml"


def test_config_takes_precedence_over_legacy_alias() -> None:
    args = argparse.Namespace(
        config="public.yaml",
        config_path="legacy.yaml",
        notes="",
        override=None,
    )
    assert cli._resolve_config_path(args) == "public.yaml"


def test_experiment_entrypoint_requires_explicit_config_choice() -> None:
    args = cli.build_parser().parse_args([])

    with pytest.raises(ValueError, match="experiment configuration is required"):
        cli._resolve_config_path(args)


def test_process_entrypoint_discards_structured_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "main", lambda argv=None: {"status": "passed"})
    assert cli.entrypoint(["preflight"]) == 0


def test_process_entrypoint_preserves_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(argv=None):
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(cli, "main", fail)
    with pytest.raises(RuntimeError, match="runtime failed"):
        cli.entrypoint([])


def test_run_dispatches_analyzed_canonical_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    analysis = _analysis(
        tmp_path,
        requested="missing.yaml",
        pipeline="Pipeline_04_Unified_Evaluation",
        overrides={"pipeline": "Pipeline_04_unified_metric"},
    )

    def pipeline(args: argparse.Namespace) -> str:
        observed["requested_config"] = args.requested_config
        observed["config_path"] = args.config_path
        observed["resolved_config_path"] = args.resolved_config_path
        observed["resolved_pipeline"] = args.resolved_pipeline
        observed["config_analysis"] = args.config_analysis
        observed["compiled_run_spec"] = args.compiled_run_spec
        observed["resolved_config_data"] = args.resolved_config_data
        observed["effective_config_sha256"] = args.effective_config_sha256
        observed["run_spec_sha256"] = args.run_spec_sha256
        observed["notes"] = args.notes
        return "sentinel"

    def fake_import(name: str) -> SimpleNamespace:
        observed["module"] = name
        return SimpleNamespace(pipeline=pipeline)

    def fake_analyze(source: str, **kwargs) -> ConfigAnalysis:
        assert source == "missing.yaml"
        assert kwargs["override_values"] == ["pipeline=Pipeline_04_unified_metric"]
        assert kwargs["local_config"] is None
        return analysis

    monkeypatch.setattr(cli, "analyze_config", fake_analyze)
    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    args = argparse.Namespace(
        config="missing.yaml",
        config_path=None,
        local_config=None,
        notes="entrypoint-parity",
        override=["pipeline=Pipeline_04_unified_metric"],
        allow_experimental=True,
    )

    assert cli.run(args) == "sentinel"
    compiled = observed.pop("compiled_run_spec")
    resolved_data = observed.pop("resolved_config_data")
    run_spec_sha256 = observed.pop("run_spec_sha256")
    config_analysis = observed.pop("config_analysis")
    assert compiled.pipeline == "Pipeline_04_Unified_Evaluation"
    assert resolved_data == analysis.effective_config
    assert run_spec_sha256 == compiled.sha256
    assert config_analysis is analysis
    assert observed == {
        "module": "src.Pipeline_04_Unified_Evaluation",
        "requested_config": "missing.yaml",
        "config_path": str(analysis.path),
        "resolved_config_path": str(analysis.path),
        "resolved_pipeline": "Pipeline_04_Unified_Evaluation",
        "effective_config_sha256": analysis.effective_config_sha256,
        "notes": "entrypoint-parity",
    }
    assert Path(args.run_manifest_path).is_file()


@pytest.mark.parametrize("preset", tuple(sorted(MAINTAINED_PRESETS)))
def test_run_passes_maintained_preset_path_to_runtime(
    preset: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def pipeline(args: argparse.Namespace) -> bool:
        observed["requested_config"] = args.requested_config
        observed["config_path"] = args.config_path
        observed["effective_config_sha256"] = args.effective_config_sha256
        return True

    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=pipeline),
    )
    args = argparse.Namespace(
        config=preset,
        config_path=None,
        local_config=None,
        notes="",
        override=[f"environment.output_dir={tmp_path / 'runs'}"],
    )

    assert cli.run(args) is True
    assert observed["requested_config"] == preset
    assert Path(str(observed["config_path"])) == resolve_config_path(preset)
    assert len(str(observed["effective_config_sha256"])) == 64
    assert Path(args.run_manifest_path).is_file()


def test_cwru_quickstart_uses_one_lightning_device_for_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    monkeypatch.setattr(
        cwru_quickstart,
        "download_bundle",
        lambda *args, **kwargs: SimpleNamespace(directory=tmp_path),
    )
    monkeypatch.setattr(
        cwru_quickstart.cli,
        "main",
        lambda argv: observed.extend(argv),
    )
    monkeypatch.setattr(sys, "argv", ["cwru_quickstart.py"])

    cwru_quickstart.main()

    assert "trainer.device=cpu" in observed
    assert "trainer.gpus=1" in observed
    assert "trainer.gpus=0" not in observed


@pytest.mark.parametrize(
    "command",
    (
        [sys.executable, "main.py", "--help"],
        [sys.executable, "-m", "phmfactory", "--help"],
    ),
)
def test_python_entrypoints_share_help_surface(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--config_path" in completed.stdout
    assert "--local-config" in completed.stdout
    assert "--override" in completed.stdout
    assert "--allow-experimental" in completed.stdout


@pytest.mark.parametrize(
    "prefix",
    (
        [sys.executable, "main.py"],
        [sys.executable, "-m", "phmfactory"],
    ),
)
def test_python_process_entrypoints_return_zero_for_preflight(
    prefix: list[str],
    tmp_path: Path,
) -> None:
    target = tmp_path / "preflight-output"
    completed = subprocess.run(
        [
            *prefix,
            "preflight",
            "--config",
            "smoke",
            "--override",
            f"environment.output_dir={target}",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "status=passed" in completed.stdout
    assert "effective_config_sha256=" in completed.stdout
    assert not target.exists()


@pytest.mark.parametrize(
    "prefix",
    (
        [sys.executable, "main.py"],
        [sys.executable, "-m", "phmfactory"],
    ),
)
def test_python_process_entrypoints_return_nonzero_for_invalid_config(
    prefix: list[str],
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.yaml"
    completed = subprocess.run(
        [*prefix, "preflight", "--config", str(missing)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "was not found" in completed.stderr


def test_root_main_is_only_a_process_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(cli, "entrypoint", lambda argv=None: calls.append(argv) or 0)

    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(REPOSITORY_ROOT / "main.py"), run_name="__main__")

    assert error.value.code == 0
    assert calls == [None]


def test_run_dispatches_packaged_base_configs_outside_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def pipeline(args: argparse.Namespace) -> str:
        config = args.compiled_run_spec.runtime_config()
        observed["data"] = config["data"]["metadata_file"]
        observed["model"] = config["model"]["name"]
        observed["task"] = config["task"]["name"]
        observed["trainer"] = config["trainer"]["device"]
        return "dispatched"

    real_import_module = cli.importlib.import_module

    def fake_import(name: str):
        if name == "src.Pipeline_01_Fault_Diagnosis":
            return SimpleNamespace(pipeline=pipeline)
        return real_import_module(name)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    args = argparse.Namespace(
        config="smoke",
        config_path=None,
        local_config=None,
        notes="",
        override=None,
    )

    assert cli.run(args) == "dispatched"
    assert observed == {
        "data": "metadata_dummy.csv",
        "model": "M_01_ISFM",
        "task": "classification",
        "trainer": "cpu",
    }
    assert Path(args.run_manifest_path).is_file()
