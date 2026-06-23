from __future__ import annotations

import argparse

import pytest

from src.configs.preflight import PreflightError, build_preflight_report, run_preflight


def _base_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.csv").write_text("sample_id,label\n0,0\n", encoding="utf-8")
    return {
        "pipeline": "Pipeline_01_default",
        "environment": {
            "output_dir": str(tmp_path / "results"),
            "iterations": 1,
            "seed": 1,
        },
        "data": {
            "data_dir": str(data_dir),
            "metadata_file": "metadata.csv",
        },
        "model": {"type": "Dummy", "name": "Dummy"},
        "task": {"type": "Classification", "name": "Dummy"},
        "trainer": {"num_epochs": 1},
    }


def test_preflight_accepts_minimal_valid_config(tmp_path) -> None:
    report = run_preflight(
        _base_config(tmp_path),
        strict=True,
        require_data=True,
        create_output_dir=True,
    )

    assert all(item["ok"] for item in report)
    assert (tmp_path / "results").is_dir()


def test_preflight_rejects_missing_data_dir(tmp_path) -> None:
    cfg = _base_config(tmp_path)
    cfg["data"]["data_dir"] = str(tmp_path / "missing")

    with pytest.raises(PreflightError) as excinfo:
        run_preflight(cfg, strict=True, require_data=True)

    assert "preflight.data_dir_exists" in str(excinfo.value)


def test_preflight_reports_missing_metadata(tmp_path) -> None:
    cfg = _base_config(tmp_path)
    cfg["data"]["metadata_file"] = "missing.csv"

    report = build_preflight_report(cfg, require_data=True)

    failed = {item["check"] for item in report if not item["ok"]}
    assert "preflight.metadata_file_exists" in failed


def test_preflight_rejects_p02_missing_mode(tmp_path) -> None:
    cfg = _base_config(tmp_path)
    cfg["pipeline"] = "Pipeline_02_pretrain_fewshot"

    with pytest.raises(PreflightError) as excinfo:
        run_preflight(cfg, strict=True)

    assert "preflight.p02_pipeline_mode" in str(excinfo.value)


def test_preflight_rejects_p02_legacy_without_fs_config(tmp_path) -> None:
    cfg = _base_config(tmp_path)
    cfg["pipeline"] = "Pipeline_02_pretrain_fewshot"
    cfg["pipeline_mode"] = "legacy"

    with pytest.raises(PreflightError) as excinfo:
        run_preflight(cfg, args=argparse.Namespace(fs_config_path=None), strict=True)

    assert "preflight.p02_legacy_fs_config" in str(excinfo.value)


def test_preflight_accepts_p02_legacy_with_fs_config(tmp_path) -> None:
    cfg = _base_config(tmp_path)
    cfg["pipeline"] = "Pipeline_02_pretrain_fewshot"
    cfg["pipeline_mode"] = "legacy"

    report = run_preflight(
        cfg,
        args=argparse.Namespace(fs_config_path="configs/demo/00_smoke/dummy_dg.yaml"),
        strict=True,
        create_output_dir=False,
    )

    assert all(item["ok"] for item in report)
