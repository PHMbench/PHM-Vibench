from pathlib import Path

from scripts.uxfd_goal_status import DEFAULT_STATUS_DIR, generate_status_reports


def test_goal_status_generator_writes_current_non_evidence_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "status"

    written = generate_status_reports(output_dir, generated_on="2026-05-12")

    assert len(written) == 10
    overall = (output_dir / "status_00_overall.md").read_text(encoding="utf-8")
    gpu = (output_dir / "status_09_gpu_execution.md").read_text(encoding="utf-8")
    citation = (output_dir / "status_08_citation_readiness.md").read_text(
        encoding="utf-8"
    )

    assert "not accepted experiment evidence" in overall
    assert "Artifact coverage: `0/104`" in overall
    assert "Queue dry-run entries: `104`" in gpu
    assert "Evidence ready: `False`" in citation


def test_persisted_goal_status_reports_match_generator(tmp_path: Path) -> None:
    output_dir = tmp_path / "status"

    generated = generate_status_reports(output_dir, generated_on="2026-05-12")

    for generated_path in generated:
        persisted_path = DEFAULT_STATUS_DIR / generated_path.name
        assert persisted_path.exists()
        assert persisted_path.read_text(encoding="utf-8") == generated_path.read_text(
            encoding="utf-8"
        )
