from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory.config import parse_overrides
from scripts import phm


def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs/demo/00_smoke").mkdir(parents=True)
    (root / "configs/local").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "results").mkdir()
    (root / "main.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    (root / "data/metadata_dummy.csv").write_text("Id,File_name\n", encoding="utf-8")
    (root / phm.SMOKE_CONFIG).write_text(
        "pipeline: Pipeline_01_Fault_Diagnosis\n"
        "environment:\n"
        "  output_dir: wrong-output\n"
        "data:\n"
        "  data_dir: wrong-data\n"
        "  metadata_file: wrong.csv\n"
        "  num_workers: 9\n"
        "model:\n"
        "  name: Dummy\n"
        "task:\n"
        "  name: dummy\n"
        "trainer:\n"
        "  num_epochs: 9\n"
        "  device: gpu\n"
        "  gpus: 8\n",
        encoding="utf-8",
    )
    (root / "configs/local/local.yaml").write_text(
        "data:\n"
        "  data_dir: /external/machine/data\n"
        "  metadata_file: external.xlsx\n"
        "  num_workers: 32\n"
        "trainer:\n"
        "  num_epochs: 99\n"
        "  device: gpu\n"
        "environment:\n"
        "  output_dir: /external/results\n",
        encoding="utf-8",
    )
    return root


def fake_importer(name: str) -> object:
    assert name in phm.REQUIRED_MODULES
    return SimpleNamespace(__version__="test")


def command_overrides(command: list[str]) -> dict:
    values = [command[index + 1] for index, item in enumerate(command[:-1]) if item == "--override"]
    return parse_overrides(values)


def test_doctor_passes_without_starting_training(tmp_path: Path, capsys) -> None:
    root = make_repo(tmp_path)
    assert phm.run_doctor(root, importer=fake_importer) == 0
    output = capsys.readouterr().out
    assert "doctor: PASS" in output
    assert "entrypoint:main.py" in output
    assert "config:offline-smoke" in output
    assert "data:dummy-metadata" in output
    assert "dependency:pytorch_lightning" in output


def test_doctor_reports_missing_smoke_config(tmp_path: Path, capsys) -> None:
    root = make_repo(tmp_path)
    (root / phm.SMOKE_CONFIG).unlink()
    assert phm.run_doctor(root, importer=fake_importer) == 1
    output = capsys.readouterr().out
    assert "[FAIL] config:offline-smoke" in output
    assert "remediation:" in output


def test_doctor_reports_broken_pipeline_dependency(tmp_path: Path, capsys) -> None:
    root = make_repo(tmp_path)

    def importer(name: str) -> object:
        if name == "pytorch_lightning":
            raise ImportError("broken lightning")
        return SimpleNamespace(__version__="test")

    assert phm.run_doctor(root, importer=importer) == 1
    output = capsys.readouterr().out
    assert "[FAIL] dependency:pytorch_lightning" in output


def test_demo_restores_dummy_contract_at_cli_precedence(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    observed = {}

    def runner(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    assert phm.run_demo(root, runner=runner) == 0
    command = observed["command"]
    assert command[:4] == [
        phm.sys.executable,
        str(root / "main.py"),
        "--config",
        str((root / phm.SMOKE_CONFIG).resolve()),
    ]
    overrides = command_overrides(command)
    assert overrides["data"]["data_dir"] == str((root / "data").resolve())
    assert overrides["data"]["metadata_file"] == "metadata_dummy.csv"
    assert overrides["data"]["num_workers"] == 0
    assert overrides["trainer"]["num_epochs"] == 1
    assert overrides["trainer"]["device"] == "cpu"
    assert overrides["trainer"]["gpus"] == 1
    assert overrides["environment"]["output_dir"] == str((root / phm.SMOKE_OUTPUT).resolve())
    assert observed["cwd"] == root.resolve()
    assert observed["shell"] is False


def test_demo_dry_run_does_not_spawn(tmp_path: Path) -> None:
    root = make_repo(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not spawn a process")

    assert phm.run_demo(root, dry_run=True, runner=forbidden) == 0


@pytest.mark.parametrize(
    ("epochs", "workers"),
    ((0, 0), (1, -1)),
)
def test_demo_rejects_invalid_bounds(tmp_path: Path, epochs: int, workers: int) -> None:
    root = make_repo(tmp_path)
    with pytest.raises(ValueError):
        phm.demo_command(root, epochs=epochs, num_workers=workers)
