import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER05_ROOT = REPO_ROOT / "paper/UXFD_paper/Paper_fuzzy_XFD"
SCRIPT = PAPER05_ROOT / "scripts/run_reviewer_ablation_smoke.py"
CONDITIONS = ("hard_threshold", "no_safety_fallback", "no_rule_output")


def test_fuzzy_reviewer_ablation_smoke_runner_emits_nonaccepted_artifacts(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--condition",
            "all",
            "--output",
            str(tmp_path),
            "--seed",
            "0",
        ],
        cwd=PAPER05_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "accepted_evidence=False" in completed.stdout
    for condition in CONDITIONS:
        run_dir = tmp_path / condition / "seed_0"
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        run_meta = json.loads((run_dir / "run_meta.yaml").read_text(encoding="utf-8"))

        assert metrics["paper_id"] == "Paper_fuzzy_XFD"
        assert metrics["protocol_id"] == "fuzzy_reviewer_ablation_smoke"
        assert metrics["condition_id"] == condition
        assert metrics["accepted_evidence"] is False
        assert "reviewer_readiness_proxy" in metrics
        assert run_meta["accepted_evidence"] is False
        assert run_meta["metrics_path"] == str(run_dir / "metrics.json")

    hard_threshold = json.loads(
        (tmp_path / "hard_threshold" / "seed_0" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    no_safety = json.loads(
        (tmp_path / "no_safety_fallback" / "seed_0" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    no_rule = json.loads(
        (tmp_path / "no_rule_output" / "seed_0" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert hard_threshold["hard_threshold_rate_proxy"] == 1.0
    assert no_safety["safety_coverage_proxy"] == 0.0
    assert no_rule["rule_trace_coverage_proxy"] == 0.0
