from __future__ import annotations

from pathlib import Path

import pytest

from frontend.console.adapters.configuration import PreflightResult
from frontend.console.adapters.runs import ProtocolSignature, RunRecord, RunSummary


def _app_test_class():
    return pytest.importorskip("streamlit.testing.v1").AppTest


def _summary(
    run_id: str,
    evidence_state: str = "complete",
) -> RunSummary:
    base = Path("/tmp") / run_id
    return RunSummary(
        run_dir=base,
        manifest_path=base / "artifacts" / "manifest.json",
        manifest={"run_id": run_id, "metrics_inline": {"test_acc": 0.9}},
        config_snapshot=base / "config_snapshot.yaml",
        metrics_path=base / "test_result_0.csv",
        metrics_csv_logger=base / "logs" / "metrics.csv",
        figures_dir=base / "figures",
        predictions_path=base / "artifacts" / "predictions.npz",
        artifacts_dir=base / "artifacts",
        checkpoint_paths=(),
        timestamp="2026-04-17T00:00:00Z",
        run_id=run_id,
        evidence_state=evidence_state,
    )


def _record(
    run_id: str,
    evidence_state: str = "complete",
    summary: str = "Pipeline_01_default / DG / classification",
    target_domain: str = "[0]",
) -> RunRecord:
    base = _summary(run_id, evidence_state=evidence_state)
    return RunRecord(
        **base.__dict__,
        protocol_signature=ProtocolSignature(
            summary=summary,
            hard_fields={
                "pipeline": "Pipeline_01_default",
                "task.type": "DG",
                "task.name": "classification",
                "task.target_domain_id": target_domain,
            },
            soft_fields={"environment.output_dir": "results/demo"},
        ),
    )


def test_compose_requires_fresh_preflight(monkeypatch) -> None:
    AppTest = _app_test_class()

    stub = PreflightResult(
        config_path="configs/demo/00_smoke/dummy_dg.yaml",
        overrides=["trainer.num_epochs=1"],
        shell_command="python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1",
        resolved={"pipeline": "Pipeline_01_default"},
        resolved_yaml="pipeline: Pipeline_01_default\n",
        sources=[{"field": "pipeline", "source": "config"}],
        targets={"pipeline": {"module": "src.Pipeline_01_default"}},
        sanity=[{"check": "pipeline_import", "ok": True, "message": "ok", "fix": ""}],
        output_preview="results/demo/iter_0",
        pipeline_name="Pipeline_01_default",
    )

    monkeypatch.setattr("frontend.console.pages.compose.build_preflight_result", lambda *args, **kwargs: stub)

    app = AppTest.from_file("frontend/pages/02_Compose.py")
    app.run()
    assert app.button[1].label == "Launch via CLI"
    assert app.button[1].disabled is True

    app.button[0].click()
    app.run()
    assert app.button[1].disabled is False
    assert any("pipeline: Pipeline_01_default" in block.value for block in app.code)

    app.text_area[0].input("trainer.num_epochs=2")
    app.run()
    assert app.button[1].disabled is True
    assert any("Inputs changed" in warning.value for warning in app.warning)


def test_runs_page_hydrates_only_selected_run(monkeypatch) -> None:
    AppTest = _app_test_class()

    summaries = [_summary("run_a"), _summary("run_b")]
    hydrate_calls: list[str] = []

    monkeypatch.setattr("frontend.console.pages.runs.discover_recent_runs", lambda limit=100: summaries)

    def hydrate(summary: RunSummary) -> RunRecord:
        hydrate_calls.append(summary.run_id)
        return _record(summary.run_id)

    monkeypatch.setattr("frontend.console.pages.runs.hydrate_run_record", hydrate)
    monkeypatch.setattr("frontend.console.pages.runs.load_config_snapshot", lambda record: {"pipeline": "Pipeline_01_default"})
    monkeypatch.setattr("frontend.console.pages.runs.load_metrics", lambda record: {"test_acc": 0.95})
    monkeypatch.setattr("frontend.console.pages.runs.load_metrics_history", lambda record: [])
    monkeypatch.setattr("frontend.console.pages.runs.figure_files", lambda record: [])

    app = AppTest.from_file("frontend/pages/03_Runs.py")
    app.run()

    assert hydrate_calls == ["run_a"]
    assert any("Selected Run" == subheader.value for subheader in app.subheader)


def test_compare_page_warns_on_incomplete_evidence(monkeypatch) -> None:
    AppTest = _app_test_class()

    summaries = [_summary("baseline"), _summary("candidate", evidence_state="partial")]
    records = {
        "baseline": _record("baseline", evidence_state="complete", target_domain="[0]"),
        "candidate": _record("candidate", evidence_state="partial", target_domain="[1]"),
    }

    monkeypatch.setattr("frontend.console.pages.compare.discover_recent_runs", lambda limit=100: summaries)
    monkeypatch.setattr("frontend.console.pages.compare.hydrate_run_record", lambda summary: records[summary.run_id])
    monkeypatch.setattr(
        "frontend.console.pages.compare.load_metrics",
        lambda record: {"test_acc": 0.9 if record.run_id == "baseline" else 0.8},
    )

    app = AppTest.from_file("frontend/pages/04_Compare.py")
    app.run()

    assert app.selectbox[0].label == "Baseline run"
    assert any("incomplete evidence" in warning.value for warning in app.warning)
    assert any("Protocol signature mismatch" in error.value for error in app.error)


def test_artifacts_page_surfaces_missing_inventory(monkeypatch) -> None:
    AppTest = _app_test_class()

    summary = _summary("run_artifact")
    record = _record("run_artifact")
    base = summary.run_dir

    monkeypatch.setattr("frontend.console.pages.artifacts.discover_recent_runs", lambda limit=100: [summary])
    monkeypatch.setattr("frontend.console.pages.artifacts.hydrate_run_record", lambda selected: record)
    monkeypatch.setattr("frontend.console.pages.artifacts.preview_text", lambda path: "pipeline: Pipeline_01_default\n")
    monkeypatch.setattr("frontend.console.pages.artifacts.load_metrics_history", lambda record: [])
    monkeypatch.setattr("frontend.console.pages.artifacts.figure_files", lambda record: [])
    monkeypatch.setattr("frontend.console.pages.artifacts.preview_predictions", lambda path: {})
    monkeypatch.setattr(
        "frontend.console.pages.artifacts.list_artifacts",
        lambda selected: [
            type("Artifact", (), {"label": "config_snapshot.yaml", "kind": "yaml", "exists": True, "path": base / "config_snapshot.yaml"})(),
            type("Artifact", (), {"label": "figures/", "kind": "directory", "exists": False, "path": base / "figures"})(),
        ],
    )

    app = AppTest.from_file("frontend/pages/06_Artifacts.py")
    app.run()

    assert any("Missing artifacts: 1" in caption.value for caption in app.caption)
    app.selectbox[1].select("figures/")
    app.run()
    assert any("Artifact not present." in info.value for info in app.info)
