"""Real one-epoch CPU fits through the unmodified public PHMFactory CLI."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model,metrics", [
    ("itransformer", ("acc", "f1")),
    ("timesnet", ("acc", "f1")),
    ("tslanet", ("acc", "f1")),
    ("nlinear", ("mse", "mae")),
    ("sparsetsf", ("mse", "mae")),
    ("fits", ("mse", "mae")),
    ("segrnn", ("mse", "mae")),
    ("softs", ("mse", "mae")),
    ("frets", ("mse", "mae")),
    ("tsmixer", ("mse", "mae")),
])
def test_native_model_public_dummy(model, metrics, tmp_path, record_property):
    config = ROOT / "configs" / "experiments" / "model_integration" / f"{model}_dummy.yaml"
    command = [
        sys.executable, "-m", "phmfactory", "--config", str(config),
        "--override", f"environment.output_dir={tmp_path / model}",
        "--override", f"data.cache_dir={tmp_path / 'cache'}",
    ]
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=240)
    (tmp_path / f"{model}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    values = {}
    fields = ("result_dir", "best_checkpoint", "test_metrics", "run_summary", "primary_metrics")
    for line in result.stdout.splitlines():
        for field in fields:
            prefix = field + "="
            if line.startswith(prefix):
                values[field] = line[len(prefix):]
    assert set(values) == set(fields), result.stdout
    root = Path(values["result_dir"]).resolve()
    assert root.is_dir()
    for field in ("best_checkpoint", "test_metrics", "run_summary"):
        path = Path(values[field]).resolve()
        assert path.is_file() and root in path.parents
    summary = json.loads(Path(values["run_summary"]).read_text(encoding="utf-8"))
    assert {f"test_{metric}_Dummy_Data" for metric in metrics} <= summary["metrics"].keys()
    for metric in summary["metrics"].values():
        assert metric["count"] == 1 and math.isfinite(metric["mean"])
    record_property("model", model)
    record_property("command", " ".join(command))
    record_property("primary_metrics", values["primary_metrics"])
