from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory import __version__
from phmfactory import cli
from phmfactory.config import MAINTAINED_PRESETS, ResolvedConfig, resolve_config_path
from phmfactory.pipelines import PipelineNameDeprecationWarning


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_public_version_is_v030_development_release() -> None:
    assert __version__ == "0.3.0.dev0"


def test_parser_preserves_legacy_config_path_alias() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--config_path", "legacy.yaml", "--notes", "compat"])
    assert args.config is None
    assert args.config_path == "legacy.yaml"
    assert args.notes == "compat"


def test_config_takes_precedence_over_legacy_alias() -> None:
    args = argparse.Namespace(
        config="public.yaml",
        config_path="legacy.yaml",
        notes="",
        override=None,
    )
    assert cli._resolve_config_path(args) == "public.yaml"


def test_run_dispatches_resolved_canonical_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def pipeline(args: argparse.Namespace) -> str:
        observed["requested_config"] = args.requested_config
        observed["config_path"] = args.config_path
        observed["resolved_config_path"] = args.resolved_config_path
        observed["resolved_pipeline"] = args.resolved_pipeline
        observed["notes"] = args.notes
        return "sentinel"

    def fake_import(name: str) -> SimpleNamespace:
        observed["module"] = name
        return SimpleNamespace(pipeline=pipeline)

    def fake_resolve(source: str, *, override_values: list[str] | None = None) -> ResolvedConfig:
        assert source == "missing.yaml"
        assert override_values == ["pipeline=Pipeline_04_unified_metric"]
        return ResolvedConfig(
            requested=source,
            path=Path("/tmp/missing.yaml"),
            data={"pipeline": "Pipeline_04_Unified_Evaluation"},
            pipeline="Pipeline_04_Unified_Evaluation",
            overrides={"pipeline": "Pipeline_04_unified_metric"},
        )

    monkeypatch.setattr(cli, "resolve_config", fake_resolve)
    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    args = argparse.Namespace(
        config="missing.yaml",
        config_path=None,
        notes="entrypoint-parity",
        override=["pipeline=Pipeline_04_unified_metric"],
    )

    assert cli.run(args) == "sentinel"
    assert observed == {
        "module": "src.Pipeline_04_Unified_Evaluation",
        "requested_config": "missing.yaml",
        "config_path": "/tmp/missing.yaml",
        "resolved_config_path": "/tmp/missing.yaml",
        "resolved_pipeline": "Pipeline_04_Unified_Evaluation",
        "notes": "entrypoint-parity",
    }


@pytest.mark.parametrize("preset", tuple(sorted(MAINTAINED_PRESETS)))
def test_run_passes_maintained_preset_path_to_runtime(
    preset: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def pipeline(args: argparse.Namespace) -> None:
        observed["requested_config"] = args.requested_config
        observed["config_path"] = args.config_path

    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=pipeline),
    )
    args = argparse.Namespace(
        config=preset,
        config_path=None,
        notes="",
        override=None,
    )

    cli.run(args)

    assert observed["requested_config"] == preset
    assert Path(str(observed["config_path"])) == resolve_config_path(preset)


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
    assert "--override" in completed.stdout


def test_root_main_is_only_a_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(cli, "main", lambda argv=None: calls.append(argv))
    runpy.run_path(str(REPOSITORY_ROOT / "main.py"), run_name="__main__")
    assert calls == [None]
