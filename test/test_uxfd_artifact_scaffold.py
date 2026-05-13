import json
from pathlib import Path

import yaml

from scripts.uxfd_artifact_gate import evaluate_artifact_gate
from scripts.uxfd_artifact_scaffold import create_scaffold, main
from scripts.uxfd_gpu_queue import (
    DEFAULT_QUEUE,
    DISALLOWED_LAUNCH_COMMAND_MARKERS,
    build_launch_plan,
    expand_queue,
)


PERSISTED_TEMPLATE_ROOT = Path("paper/UXFD_paper/results/accepted_run_templates")


def _assert_no_disallowed_command_markers(data: dict, template_path: Path) -> None:
    for field in ("command", "original_command", "queue_config_path"):
        value = str(data.get(field, ""))
        lowered = value.lower()
        marker = next(
            (item for item in DISALLOWED_LAUNCH_COMMAND_MARKERS if item in lowered),
            "",
        )
        assert marker == "", f"{template_path}:{field} contains {marker}: {value}"


def test_artifact_scaffold_creates_one_template_per_launch_row(tmp_path: Path) -> None:
    report = create_scaffold(
        output_root=tmp_path / "templates", queue_path=DEFAULT_QUEUE
    )
    queue_rows = expand_queue(DEFAULT_QUEUE)

    assert len(build_launch_plan(queue_rows)) == 97
    assert len(report.records) == len(queue_rows) == 104
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
    assert data["config_path"] == "config.yaml"
    assert data["queue_config_path"].startswith("paper/UXFD_paper/")
    assert str(data["source_tree_status"]).startswith("TODO")
    _assert_no_disallowed_command_markers(data, first)

    top_records = [
        record for record in report.records if record.phase == "top_representatives"
    ]
    assert len(top_records) == 7
    top_template = Path(top_records[0].template_path)
    top_data = yaml.safe_load(top_template.read_text(encoding="utf-8"))
    assert top_data["cuda_visible_devices"] == "0,1"
    assert top_data["gpu_count"] == 2
    _assert_no_disallowed_command_markers(top_data, top_template)


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
    readme = (output_root / "README.md").read_text(encoding="utf-8")
    assert "at least one numeric metric" in readme
    assert "source_tree_status: clean" in readme
    assert "`batch_size` must be a positive integer" in readme
    assert "dirty, modified, unknown, or uncommitted" in readme
    assert gate.accepted is False
    assert any("no run_meta.yaml" in blocker for blocker in gate.blockers)


def test_persisted_artifact_templates_match_current_launch_plan() -> None:
    queue_rows = expand_queue(DEFAULT_QUEUE)
    manifest_path = PERSISTED_TEMPLATE_ROOT / "manifest.json"
    readme = (PERSISTED_TEMPLATE_ROOT / "README.md").read_text(encoding="utf-8")

    assert manifest_path.exists()
    assert "at least one numeric metric" in readme
    assert "source_tree_status: clean" in readme
    assert "`batch_size` must be a positive integer" in readme
    assert "dirty, modified, unknown, or uncommitted" in readme
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(build_launch_plan(queue_rows)) == 97
    assert len(manifest) == len(queue_rows) == 104

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
    queue_keys = set()
    for row in build_launch_plan(queue_rows):
        queue_keys.add(
            (row.queue_id, row.paper_id, row.phase, row.entry_id, row.device)
        )
    for row in queue_rows:
        if row.phase == "top_representatives":
            queue_keys.add((row.queue_id, row.paper_id, row.phase, row.entry_id, "0,1"))
    assert manifest_keys == queue_keys

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
        assert data["config_path"] == "config.yaml"
        assert "source_tree_status" in data
        assert "queue_config_path" in data
        _assert_no_disallowed_command_markers(data, template_path)
        if "python main.py --config" in record["command"]:
            assert data["queue_config_path"].startswith("paper/UXFD_paper/")
