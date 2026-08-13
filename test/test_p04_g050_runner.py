from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.p04 import run_g050_decisive as runner
from src.model_factory.MoE.M_04_RoleConstrainedMoE import Model


def _decisive_args(arm: str = "P0") -> SimpleNamespace:
    config = runner._load_config(runner.G050_CONFIG_PATH)
    return runner._model_args(config, arm)


def test_runner_accepts_only_the_canonical_config_and_requires_mode(
    tmp_path: Path,
) -> None:
    parsed = runner._parse_args(
        ["--output-root", str(tmp_path / "out"), "--mode", "smoke"]
    )
    assert parsed.mode == "smoke"
    assert not hasattr(parsed, "config")

    with pytest.raises(SystemExit):
        runner._parse_args(["--output-root", str(tmp_path / "missing-mode")])
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--config",
                str(tmp_path / "other.yaml"),
                "--output-root",
                str(tmp_path / "out"),
                "--mode",
                "smoke",
            ]
        )


def test_json_artifacts_reject_nested_nonfinite_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        runner._write_json(
            tmp_path / "invalid.json",
            {"tensor": torch.tensor([1.0, float("nan")])},
        )
    with pytest.raises(ValueError, match="NaN or infinity"):
        runner._write_json(
            tmp_path / "invalid-array.json",
            {"array": np.asarray([1.0, float("inf")])},
        )


def test_decisive_model_requires_explicit_parameters() -> None:
    values = vars(_decisive_args()).copy()
    values.pop("low_order_cutoff")
    with pytest.raises(ValueError, match="low_order_cutoff"):
        Model(SimpleNamespace(**values))

    values = vars(_decisive_args()).copy()
    values["scientific_arm"] = ""
    with pytest.raises(ValueError, match="scientific_arm"):
        Model(SimpleNamespace(**values))


def test_empty_router_override_is_rejected_instead_of_inheriting() -> None:
    model = Model(_decisive_args()).eval()
    model.set_compatibility_statistics(torch.zeros(4), torch.ones(4))
    physical = {
        "sample_rate_hz": [12_000.0],
        "rotation_speed_rpm": [1_797.0],
        "load_hp": [0.0],
    }
    with pytest.raises(ValueError, match="unsupported router mode"):
        model(
            torch.randn(1, 256, 1),
            physical_metadata=physical,
            router_mode="",
        )


def test_main_records_data_failure_as_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "failed-run"
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: SimpleNamespace(
            output_root=output_root,
            raw_root=None,
            mode="smoke",
        ),
    )
    monkeypatch.setattr(runner, "_require_gpu5", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        runner,
        "_runtime_git_provenance",
        lambda: {
            "root": str(runner.REPO_ROOT),
            "branch": "test",
            "commit": "0" * 40,
            "dirty": False,
            "changed_paths": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "_run_semantic_gate_tests",
        lambda: {"status": "passed", "observed_pass_count": 1},
    )

    def fail_data(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("sentinel data failure")

    monkeypatch.setattr(runner, "_load_admitted_data", fail_data)
    with pytest.raises(ValueError, match="sentinel data failure"):
        runner.main()

    index = json.loads((output_root / "run_index.json").read_text(encoding="utf-8"))
    assert index["execution_status"] == "failed"
    assert index["execution"]["execution_stage"] == "load_data"
    assert index["failure"] == {
        "message": "sentinel data failure",
        "stage": "load_data",
        "type": "ValueError",
    }
    assert index["runs"] == []
