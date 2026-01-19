import json
from pathlib import Path

from scripts.collect_uxfd_runs import collect_manifests, write_csv


def test_collect_uxfd_runs_writes_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "run_0"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "paper_id": "Paper_fuzzy_XFD",
        "preset_version": "vibench-min-v1",
        "run_id": "run_0",
        "stage": "test",
        "timestamp": "2026-01-15T00:00:00Z",
        "config_snapshot": str(run_dir / "config_snapshot.yaml"),
        "metrics_path": str(run_dir / "test_result_0.csv"),
        "predictions_path": str(artifacts_dir / "predictions.npz"),
        "metrics_inline": {"test_loss": 0.1, "test_acc": 1.0},
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = collect_manifests(tmp_path)
    assert len(result.rows) == 1
    assert result.rows[0]["paper_id"] == "Paper_fuzzy_XFD"
    assert result.rows[0]["predictions_path"].endswith("predictions.npz")

    out_csv = tmp_path / "reports" / "uxfd_runs.csv"
    write_csv(
        out_csv,
        result.rows,
        preferred_cols=["paper_id", "preset_version", "run_id", "predictions_path"],
    )
    assert out_csv.exists()
    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "paper_id" in header
    assert "predictions_path" in header

