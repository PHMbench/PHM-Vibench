import json
from pathlib import Path
from types import SimpleNamespace

from scripts import uxfd_prelaunch_gate as gate


PERSISTED_PRELAUNCH_GATE_JSON = Path(
    "paper/UXFD_paper/results/prelaunch_gate_current.json"
)
PERSISTED_PRELAUNCH_GATE_MD = Path(
    "paper/UXFD_paper/results/prelaunch_gate_current.md"
)


def test_prelaunch_gate_blocks_current_non_ready_state_without_live_preflight() -> None:
    report = gate.evaluate_prelaunch_gate(require_live_preflight=False)

    assert report.ready is False
    assert report.live_preflight_required is False
    assert report.live_preflight_accepted is None
    assert any("objective audit not achieved" in item for item in report.blockers)
    assert any("owner-review gate not ready" in item for item in report.blockers)
    assert any("gpu queue static gate not executable" in item for item in report.blockers)
    assert any("submission gate not ready" in item for item in report.blockers)


def test_prelaunch_gate_requires_live_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        gate,
        "evaluate_objective_audit",
        lambda queue_path, artifact_root: SimpleNamespace(
            achieved=True,
            met=95,
            not_met=0,
            blocked=0,
            unverified=0,
        ),
    )
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
    monkeypatch.setattr(
        gate,
        "evaluate_submission_gate",
        lambda queue_path, artifact_root, sota_root: SimpleNamespace(
            ready=True,
            blockers=(),
            artifact_gate_accepted=True,
            sota_gate_ready=True,
            recent_work_evidence_ready=True,
        ),
    )

    report = gate.evaluate_prelaunch_gate()

    assert report.ready is False
    assert report.live_preflight_required is True
    assert report.live_preflight_accepted is False
    assert report.blockers == (
        "live GPU preflight not accepted: blocked: no visible RTX 4090 devices",
    )


def test_prelaunch_gate_ready_when_all_inputs_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        gate,
        "evaluate_objective_audit",
        lambda queue_path, artifact_root: SimpleNamespace(
            achieved=True,
            met=95,
            not_met=0,
            blocked=0,
            unverified=0,
        ),
    )
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
    monkeypatch.setattr(
        gate,
        "evaluate_submission_gate",
        lambda queue_path, artifact_root, sota_root: SimpleNamespace(
            ready=True,
            blockers=(),
            artifact_gate_accepted=True,
            sota_gate_ready=True,
            recent_work_evidence_ready=True,
        ),
    )

    report = gate.evaluate_prelaunch_gate()

    assert report.ready is True
    assert report.blockers == ()
    assert "Ready: `True`" in gate.render_markdown(report)


def test_prelaunch_gate_cli_writes_json(tmp_path: Path) -> None:
    json_path = tmp_path / "prelaunch.json"

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
    assert '"ready": false' in json_path.read_text(encoding="utf-8")


def test_persisted_prelaunch_gate_reports_match_current_gate() -> None:
    report = gate.evaluate_prelaunch_gate()

    expected_json = json.dumps(gate.build_payload(report), indent=2) + "\n"
    assert PERSISTED_PRELAUNCH_GATE_JSON.read_text(encoding="utf-8") == expected_json
    assert PERSISTED_PRELAUNCH_GATE_MD.read_text(encoding="utf-8") == gate.render_markdown(
        report
    )
