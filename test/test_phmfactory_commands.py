from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory import cli
from phmfactory.commands import demo, doctor, preflight
from phmfactory.commands.common import check_writable_directory
from phmfactory.config import ConfigAnalysis, ResolvedConfig, semantic_config_sha256
import phmfactory.device as device_contract


def _config(
    tmp_path: Path,
    *,
    output: bool = True,
    device: str = "cpu",
) -> dict:
    environment = {
        "seed": 0,
        "iterations": 1,
        **({"output_dir": str(tmp_path / "new" / "outputs")} if output else {}),
    }
    return {
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "environment": environment,
        "data": {},
        "model": {},
        "task": {},
        "trainer": {"device": device, "devices": 1},
    }


def _analysis(
    tmp_path: Path,
    *,
    output: bool = True,
    device: str = "cpu",
) -> ConfigAnalysis:
    config = _config(tmp_path, output=output, device=device)
    path = tmp_path / "smoke.yaml"
    return ConfigAnalysis(
        requested="smoke",
        path=path,
        effective_config=config,
        pipeline="Pipeline_01_Fault_Diagnosis",
        overrides={},
        local_config_path=None,
        source_files=(path,),
        sources={},
        diagnostics=(),
        effective_config_sha256=semantic_config_sha256(config),
    )


def _resolved(tmp_path: Path) -> ResolvedConfig:
    return _analysis(tmp_path).to_resolved_config()


def test_command_router_preserves_legacy_experiment_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed: list[object] = []
    monkeypatch.setattr(
        cli,
        "_run_command",
        lambda name, argv: routed.append((name, list(argv))) or "command",
    )
    monkeypatch.setattr(
        cli,
        "run",
        lambda args: routed.append(("experiment", args.config)) or "experiment",
    )

    assert cli.main(["doctor"]) == "command"
    assert cli.main(["--config", "smoke"]) == "experiment"
    assert routed == [("doctor", []), ("experiment", "smoke")]


def test_no_arguments_only_prints_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "run",
        lambda args: pytest.fail("no-argument invocation must not start a run"),
    )

    assert cli.main([]) == {"status": "help"}

    output = capsys.readouterr().out
    assert "usage: phmfactory" in output
    assert "Run an experiment explicitly" in output


@pytest.mark.parametrize("command", ("doctor", "demo", "preflight", "data"))
def test_named_command_help_is_standard_argparse(command: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--help"])
    assert error.value.code == 0


def test_demo_uses_offline_defaults_and_user_override_wins() -> None:
    observed: list[argparse.Namespace] = []
    result = demo.run(
        ["--override", "trainer.num_epochs=2", "--notes", "demo-test"],
        experiment_runner=lambda args: observed.append(args) or "ok",
    )

    assert result == "ok"
    args = observed[0]
    assert args.config == "smoke"
    assert args.notes == "demo-test"
    assert args.allow_experimental is False
    assert args.override[:3] == list(demo.DEFAULT_OVERRIDES)
    assert args.override[-1] == "trainer.num_epochs=2"
    assert "trainer.gpus=1" not in args.override


def test_preflight_uses_single_analysis_without_importing_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis(tmp_path)
    monkeypatch.setattr(preflight, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(name=name),
    )

    result = preflight.run(["--config", "smoke"])

    assert result["status"] == "passed"
    assert result["pipeline"] == "Pipeline_01_Fault_Diagnosis"
    assert result["effective_config_sha256"] == analysis.effective_config_sha256
    assert len(result["run_spec_sha256"]) == 64
    assert result["requested_device"] == "cpu"
    assert result["resolved_accelerator"] == "cpu"
    assert result["resolved_devices"] == 1
    assert not (tmp_path / "new").exists()


def test_preflight_rejects_unavailable_cuda_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis(tmp_path, device="cuda")
    monkeypatch.setattr(preflight, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(name=name),
    )
    monkeypatch.setattr(
        device_contract,
        "_load_torch",
        lambda: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        ),
    )

    with pytest.raises(RuntimeError, match="CUDA is unavailable.*no CPU fallback"):
        preflight.run(["--config", "smoke"])

    assert not (tmp_path / "new").exists()


def test_preflight_requires_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis(tmp_path, output=False)
    monkeypatch.setattr(preflight, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(name=name),
    )

    with pytest.raises(ValueError, match="environment.output_dir"):
        preflight.run(["--config", "smoke"])


def test_direct_outputs_are_machine_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_direct_outputs(
        {
            "result_dir": "/tmp/result",
            "best_checkpoint": "/tmp/result/best.ckpt",
            "test_metrics": "/tmp/result/all_results.csv",
            "run_summary": "/tmp/result/run_summary.json",
            "primary_metrics": {
                "test_acc": {"count": 1, "mean": 0.75, "sample_std": None}
            },
        }
    )

    output = capsys.readouterr().out
    assert "result_dir=/tmp/result" in output
    assert "best_checkpoint=/tmp/result/best.ckpt" in output
    assert "test_metrics=/tmp/result/all_results.csv" in output
    assert "run_summary=/tmp/result/run_summary.json" in output
    assert '"test_acc"' in output


def test_doctor_exercises_real_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        if name == "torch":
            raise OSError("binary ABI mismatch")
        return SimpleNamespace(__version__="test")

    monkeypatch.setattr(doctor.importlib, "import_module", fake_import)
    monkeypatch.setattr(doctor, "resolve_config", lambda source: _resolved(tmp_path))
    monkeypatch.setattr(
        doctor.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(name=name),
    )
    monkeypatch.setattr(doctor, "check_writable_directory", lambda path: Path(path))

    checks = doctor.collect_checks()
    torch_check = next(check for check in checks if check.name == "import:torch")

    assert imported == list(doctor.CORE_MODULES)
    assert torch_check.passed is False
    assert "OSError: binary ABI mismatch" in torch_check.detail


def test_doctor_failure_has_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "collect_checks",
        lambda: [doctor.DoctorCheck("import:torch", False, "missing")],
    )
    with pytest.raises(SystemExit) as error:
        doctor.run([])
    assert error.value.code == 1


def test_doctor_success_returns_check_records(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = [doctor.DoctorCheck("python", True, "3.10")]
    monkeypatch.setattr(doctor, "collect_checks", lambda: expected)
    assert doctor.run([]) == expected


def test_writable_probe_leaves_missing_target_absent(tmp_path: Path) -> None:
    target = tmp_path / "new" / "nested" / "output"
    assert check_writable_directory(target) == target.resolve()
    assert not target.exists()
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".phmfactory-dir-probe-*"))


def test_writable_probe_preserves_existing_directory_content(tmp_path: Path) -> None:
    target = tmp_path / "output"
    target.mkdir()
    marker = target / "owned-by-user.txt"
    marker.write_text("preserve\n", encoding="utf-8")

    assert check_writable_directory(target) == target.resolve()
    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert not list(target.glob(".phmfactory-write-probe-*"))


def test_writable_probe_never_recursively_deletes_concurrent_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "new" / "output"
    original_write_text = Path.write_text

    def write_text(path: Path, *args, **kwargs):
        result = original_write_text(path, *args, **kwargs)
        if path.name.startswith(".phmfactory-write-probe-"):
            original_write_text(
                path.parent / "concurrent.txt",
                "created by another owner\n",
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(Path, "write_text", write_text)

    assert check_writable_directory(target) == target.resolve()
    assert not target.exists()
    probe_directories = list(tmp_path.glob(".phmfactory-dir-probe-*"))
    assert len(probe_directories) == 1
    assert (probe_directories[0] / "concurrent.txt").read_text(encoding="utf-8") == (
        "created by another owner\n"
    )
