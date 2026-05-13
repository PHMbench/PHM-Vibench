import json
from pathlib import Path

import yaml

from scripts.uxfd_sota_gate import evaluate_sota_gate
from scripts.uxfd_sota_scaffold import DEFAULT_TEMPLATE_ROOT, create_scaffold, main


def test_sota_scaffold_creates_one_template_per_paper(tmp_path: Path) -> None:
    report = create_scaffold(output_root=tmp_path / "templates")

    assert report.note == "templates only; not accepted SOTA evidence"
    assert len(report.records) == 7
    assert {record.minimum_seeds for record in report.records} == {3}
    assert all(record.baselines >= 6 for record in report.records)
    assert all(record.top_representatives >= 1 for record in report.records)

    first = Path(report.records[0].template_path)
    assert first.name == "sota_aggregate.template.yaml"
    assert first.exists()
    assert not (first.parent / "sota_aggregate.yaml").exists()

    data = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert data["template_only"] is True
    assert data["accepted_sota_evidence"] is False
    assert str(data["claim_scope"]).startswith("TODO")
    assert len(data["comparators"]) >= 6
    assert data["top_representatives"]
    assert "mean" in data["proposed"]["statistics"]
    assert "ci95_high" in data["proposed"]["statistics"]
    assert isinstance(data["proposed"]["accepted_run_refs"], list)
    assert len(data["proposed"]["accepted_run_refs"]) == 3
    assert "effect_size_vs_proposed" in data["comparators"][0]
    assert "paired_test" in data["comparators"][0]
    assert isinstance(data["comparators"][0]["accepted_run_refs"], list)
    assert isinstance(data["top_representatives"][0]["accepted_run_refs"], list)


def test_sota_scaffold_cli_writes_manifest_and_keeps_gate_blocked(tmp_path: Path) -> None:
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
    manifest = yaml.safe_load((output_root / "manifest.yaml").read_text(encoding="utf-8"))
    readme = (output_root / "README.md").read_text(encoding="utf-8")
    gate = evaluate_sota_gate(output_root)

    assert payload["template_root"] == str(output_root)
    assert len(payload["records"]) == 7
    assert len(manifest) == 7
    assert "not accepted SOTA evidence" in readme
    assert "mean, std, 95% CI" in readme
    assert "existing relative `run_meta.yaml` paths" in readme
    assert gate.ready is False
    assert all("missing sota_aggregate.yaml" in record.issues for record in gate.records)


def test_persisted_sota_templates_match_current_queue(tmp_path: Path) -> None:
    expected = create_scaffold(output_root=tmp_path / "expected")
    manifest = yaml.safe_load((DEFAULT_TEMPLATE_ROOT / "manifest.yaml").read_text(encoding="utf-8"))
    readme = (DEFAULT_TEMPLATE_ROOT / "README.md").read_text(encoding="utf-8")

    assert len(expected.records) == 7
    assert len(manifest) == 7
    assert "not accepted SOTA evidence" in readme
    assert "effect size or paired test" in readme
    assert {
        (
            record.queue_id,
            record.paper_id,
            record.minimum_seeds,
            record.baselines,
            record.top_representatives,
        )
        for record in expected.records
    } == {
        (
            item["queue_id"],
            item["paper_id"],
            item["minimum_seeds"],
            item["baselines"],
            item["top_representatives"],
        )
        for item in manifest
    }
    for item in manifest:
        template_path = Path(item["template_path"])
        assert template_path.exists()
        data = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        assert data["paper_id"] == item["paper_id"]
        assert data["source_queue_id"] == item["queue_id"]
        assert data["accepted_sota_evidence"] is False
        assert len(data["comparators"]) == item["baselines"]
        assert len(data["top_representatives"]) == item["top_representatives"]
        assert isinstance(data["proposed"]["accepted_run_refs"], list)
        assert all(
            isinstance(entry["accepted_run_refs"], list)
            for entry in data["comparators"]
        )
