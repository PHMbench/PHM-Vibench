import json
from types import SimpleNamespace

from scripts import uxfd_experiment_launch_gate as gate


def test_experiment_launch_gate_blocks_current_non_ready_state_without_live_preflight() -> None:
    report = gate.evaluate_experiment_launch_gate(require_live_preflight=False)

    assert report.ready is False
    assert report.live_preflight_required is False
    assert report.live_preflight_accepted is None
    assert any("owner-review gate not ready" in item for item in report.blockers)
    assert any("gpu queue static gate not executable" in item for item in report.blockers)
    assert not any("submission gate not ready" in item for item in report.blockers)


def test_experiment_launch_gate_requires_live_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        gate,
        "evaluate_owner_review_gate",
        lambda: SimpleNamespace(
            ready=True,
            source_path="owner.json",
            pending_records=0,
            blockers=(),
        ),
    )
    monkeypatch.setattr(
        gate,
        "expand_queue",
        lambda queue_path: (
            SimpleNamespace(
                phase="proposed",
                paper_id="P1",
                command="python main.py --config x.yaml",
                status="accepted_same_protocol",
            ),
        ),
    )
    monkeypatch.setattr(
        gate,
        "validate_queue",
        lambda queue_path: SimpleNamespace(
            can_execute=True,
            resource_reason="static ready",
            structural_issues=(),
        ),
    )
    monkeypatch.setattr(
        gate,
        "run_live_preflight",
        lambda: SimpleNamespace(
            accepted=False,
            reason="blocked: no visible RTX 4090 devices",
        ),
    )

    report = gate.evaluate_experiment_launch_gate()

    assert report.ready is False
    assert report.live_preflight_required is True
    assert report.live_preflight_accepted is False
    assert report.blockers == (
        "live GPU preflight not accepted: blocked: no visible RTX 4090 devices",
    )


def test_experiment_launch_gate_ready_when_launch_inputs_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        gate,
        "evaluate_owner_review_gate",
        lambda: SimpleNamespace(
            ready=True,
            source_path="owner.json",
            pending_records=0,
            blockers=(),
        ),
    )
    monkeypatch.setattr(
        gate,
        "expand_queue",
        lambda queue_path: (
            SimpleNamespace(
                phase="baselines",
                paper_id="P1",
                command="python main.py --config x.yaml",
                status="accepted_same_protocol",
            ),
        ),
    )
    monkeypatch.setattr(
        gate,
        "validate_queue",
        lambda queue_path: SimpleNamespace(
            can_execute=True,
            resource_reason="static ready",
            structural_issues=(),
        ),
    )
    monkeypatch.setattr(
        gate,
        "run_live_preflight",
        lambda: SimpleNamespace(
            accepted=True,
            reason="accepted: local RTX 4090 devices visible",
        ),
    )

    report = gate.evaluate_experiment_launch_gate()

    assert report.ready is True
    assert report.blockers == ()
    assert "Ready: `True`" in gate.render_markdown(report)
    assert "does not require accepted run artifacts" in gate.render_markdown(report)


def test_experiment_launch_gate_cli_writes_json(tmp_path) -> None:
    json_path = tmp_path / "experiment_launch.json"

    assert gate.main(
        [
            "--format",
            "json",
            "--skip-live-preflight",
            "--allow-not-ready",
            "--output",
            str(json_path),
        ]
    ) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["ready"] is False
    assert payload["live_preflight_required"] is False
