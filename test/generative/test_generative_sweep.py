from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from scripts import generative_sweep


class _Result:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.stdout = "ok"
        self.stderr = ""


def test_generative_sweep_writes_multi_config_seed_step_rows(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, text, capture_output):  # noqa: ANN001
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out_csv = tmp_path / "sweep.csv"

    code = generative_sweep.run_sweep(
        configs=[
            "configs/demo/10_generative/dummy_generative_cfm.yaml",
            "configs/demo/10_generative/dummy_generative_ddpm.yaml",
        ],
        seeds=[0, 1],
        steps=[4, 8],
        out_csv=out_csv,
    )

    assert code == 0
    assert len(calls) == 8
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(rows) == 8
    assert {row["method"] for row in rows} == {"cfm", "ddpm"}
    assert {row["seed"] for row in rows} == {"0", "1"}
    assert {row["num_steps"] for row in rows} == {"4", "8"}
    assert all(row["returncode"] == "0" for row in rows)


def test_generative_sweep_reports_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, text, capture_output):  # noqa: ANN001
        return _Result(returncode=2)

    monkeypatch.setattr(subprocess, "run", fake_run)
    out_csv = tmp_path / "sweep.csv"

    code = generative_sweep.run_sweep(
        configs="configs/demo/10_generative/dummy_generative_cfm.yaml",
        seeds=[0],
        steps=[4, 8],
        out_csv=out_csv,
    )

    assert code == 1
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["returncode"] == "2"
