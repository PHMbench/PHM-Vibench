import json
from pathlib import Path

import yaml

from scripts.uxfd_artifact_gate import evaluate_artifact_gate
from scripts.uxfd_artifact_scaffold import create_scaffold, main
from scripts.uxfd_gpu_queue import DEFAULT_QUEUE, build_launch_plan, expand_queue


PERSISTED_TEMPLATE_ROOT = Path("paper/UXFD_paper/results/accepted_run_templates")


def test_artifact_scaffold_creates_one_template_per_launch_row(tmp_path: Path) -> None:
    report = create_scaffold(output_root=tmp_path / "templates", queue_path=DEFAULT_QUEUE)
    launch_rows = build_launch_plan(expand_queue(DEFAULT_QUEUE))

    assert len(report.records) == len(launch_rows)
    assert report.note == "templates only; not accepted evidence"
    assert report.validation_can_execute is False

    first = Path(report.records[0].template_path)
    assert first.name == "run_meta.template.yaml"
    assert first.exists()
    assert not (first.parent / "run_meta.yaml").exists()

    data = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert data["accepted_evidence"] is False
    assert data["cuda_visible_devices"] in {"0", "1"}
    assert data["command"].startswith("CUDA_VISIBLE_DEVICES=")


def test_artifact_scaffold_cli_writes_manifest_and_keeps_gate_blocked(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "templates"
    report_path = tmp_path / "scaffold.json"

    assert (
        main(
            [
                "--output-root",
                str(output_root),
                "--format",
                "json",
                "--output",
                str(report_path),
            ]
        )
        == 0
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    gate = evaluate_artifact_gate(output_root)

    assert payload["template_root"] == str(output_root)
    assert len(manifest) == len(payload["records"])
    assert (output_root / "README.md").exists()
    assert gate.accepted is False
    assert any("no run_meta.yaml" in blocker for blocker in gate.blockers)


def test_persisted_artifact_templates_match_current_launch_plan() -> None:
    launch_rows = build_launch_plan(expand_queue(DEFAULT_QUEUE))
    manifest_path = PERSISTED_TEMPLATE_ROOT / "manifest.json"

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) == len(launch_rows) == 97

    manifest_keys = {
        (
            record["queue_id"],
            record["paper_id"],
            record["phase"],
            record["entry_id"],
            record["device"],
        )
        for record in manifest
    }
    launch_keys = {
        (
            row.queue_id,
            row.paper_id,
            row.phase,
            row.entry_id,
            row.device,
        )
        for row in launch_rows
    }
    assert manifest_keys == launch_keys

    for record in manifest:
        template_path = Path(record["template_path"])
        assert template_path.exists(), template_path
        assert template_path.name == "run_meta.template.yaml"
        assert not (template_path.parent / "run_meta.yaml").exists()
        data = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        assert data["accepted_evidence"] is False
        assert data["source_queue_id"] == record["queue_id"]
        assert data["paper_id"] == record["paper_id"]
        assert data["phase"] == record["phase"]
        assert data["entry_id"] == record["entry_id"]
        assert data["cuda_visible_devices"] == record["device"]
