from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import phm


def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs/demo/00_smoke").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "main.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    (root / phm.SMOKE_CONFIG).write_text("pipeline: Pipeline_01_Fault_Diagnosis\n", encoding="utf-8")
    return root


def fake_importer(name: str) -> object:
    assert name in phm.REQUIRED_MODULES
    return SimpleNamespace(__version__="test")


def test_doctor_passes_without_starting_training(tmp_path: Path, capsys) -> None:
    root = make_repo(tmp_path)
    assert phm.run_doctor(root, importer=fake_importer) == 0
    output = capsys.readouterr().out
    assert "doctor: PASS" in output
    assert "entrypoint:main.py" in output
    assert "config:offline-smoke" in output


def test_doctor_reports_missing_smoke_config(tmp_path: Path, capsys) -> None:
    root = make_repo(tmp_path)
    (root / phm.SMOKE_CONFIG).unlink()
    assert phm.run_doctor(root, importer=fake_importer) == 1
    output = capsys.readouterr().out
    assert "[FAIL] config:offline-smoke" in output
    assert "remediation:" in output


def test_demo_uses_argument_list_and_existing_main_path(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    observed = {}

    def runner(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    assert phm.run_demo(root, runner=runner) == 0
    assert observed["command"] == [
        phm.sys.executable,
        str(root / "main.py"),
        "--config",
        str(phm.SMOKE_CONFIG),
        "--override",
        "trainer.num_epochs=1",
        "--override",
        "data.num_workers=0",
    ]
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
