from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from scripts.config_inspect import InspectResult

from frontend.console.adapters.configuration import (
    build_launch_request,
    build_preflight_result,
    parse_override_text,
)
from frontend.console.adapters.runs import compare_protocols, discover_run_records, load_metrics


def test_parse_override_text_ignores_blank_and_comment_lines() -> None:
    text = """
    # smoke defaults
    trainer.num_epochs=1

    data.num_workers=0
    """
    assert parse_override_text(text) == ["trainer.num_epochs=1", "data.num_workers=0"]


def test_build_launch_request_keeps_cli_contract() -> None:
    request = build_launch_request(
        "configs/demo/00_smoke/dummy_dg.yaml",
        ["trainer.num_epochs=1", "data.num_workers=0"],
        notes="smoke",
    )
    assert request.argv[1:4] == ["main.py", "--config", "configs/demo/00_smoke/dummy_dg.yaml"]
    assert request.argv.count("--override") == 2
    assert "--notes" in request.argv
    assert "main.py --config configs/demo/00_smoke/dummy_dg.yaml" in request.shell_command


def test_build_preflight_result_uses_inspect_config(monkeypatch) -> None:
    stub = InspectResult(
        resolved={
            "pipeline": "Pipeline_01_default",
            "environment": {"output_dir": "results/demo", "seed": 0},
            "data": {"metadata_file": "metadata_dummy.csv", "factory_name": "default"},
            "model": {"name": "M_01_ISFM"},
            "task": {"type": "DG", "name": "classification"},
            "trainer": {"num_epochs": 1},
        },
        sources={"task.type": "config:dummy", "trainer.num_epochs": "cli:--override"},
        targets={"pipeline": {"module": "src.Pipeline_01_default"}},
        sanity=[{"check": "pipeline_import", "ok": True, "message": "ok", "fix": ""}],
    )

    monkeypatch.setattr("frontend.console.adapters.configuration.inspect_config", lambda *args, **kwargs: stub)
    result = build_preflight_result(
        "configs/demo/00_smoke/dummy_dg.yaml",
        ["trainer.num_epochs=1"],
        notes="smoke",
    )

    assert result.pipeline_name == "Pipeline_01_default"
    assert result.output_preview.endswith("iter_0")
    assert "environment:" in result.resolved_yaml
    assert result.sources[0]["field"] == "task.type"


def test_discover_run_records_and_protocol_guard_rails(tmp_path: Path) -> None:
    def write_run(run_name: str, target_domain_id: int) -> Path:
        run_dir = tmp_path / "results" / run_name / "iter_0"
        figures_dir = run_dir / "figures"
        artifacts_dir = run_dir / "artifacts"
        figures_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config_snapshot = run_dir / "config_snapshot.yaml"
        config_snapshot.write_text(
            "\n".join(
                [
                    "pipeline: Pipeline_01_default",
                    "environment:",
                    "  output_dir: results/demo/test",
                    "data:",
                    "  factory_name: default",
                    "  metadata_file: metadata_dummy.csv",
                    "task:",
                    "  type: DG",
                    "  name: classification",
                    "  target_system_id: [0]",
                    f"  target_domain_id: [{target_domain_id}]",
                    "model:",
                    "  name: M_01_ISFM",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        metrics_path = run_dir / "test_result_0.csv"
        metrics_path.write_text("test_loss,test_acc\n0.1,1.0\n", encoding="utf-8")
        (figures_dir / "learning_curve.png").write_bytes(b"png")

        manifest = {
            "run_id": run_name,
            "timestamp": f"2026-04-16T00:00:0{target_domain_id}Z",
            "config_snapshot": str(config_snapshot),
            "metrics_path": str(metrics_path),
            "figures_dir": str(figures_dir),
            "metrics_inline": {"test_loss": 0.1 * target_domain_id},
        }
        (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run_dir

    write_run("compatible_a", 1)
    write_run("compatible_b", 2)

    records = discover_run_records(root_dirs=[tmp_path / "results"], limit=10)
    assert len(records) == 2
    assert all(record.evidence_state == "complete" for record in records)

    metrics = load_metrics(records[0])
    assert "test_loss" in metrics

    report = compare_protocols(records)
    assert report.compatible is False
    assert any(row["field"] == "task.target_domain_id" for row in report.hard_mismatches)


def test_registry_module_is_lazy_on_import() -> None:
    sys.modules.pop("frontend.console.adapters.registry", None)
    sys.modules.pop("src.data_factory", None)
    sys.modules.pop("src.trainer_factory", None)

    importlib.import_module("frontend.console.adapters.registry")

    assert "src.data_factory" not in sys.modules
    assert "src.trainer_factory" not in sys.modules
