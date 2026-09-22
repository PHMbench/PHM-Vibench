"""Linux installed-wheel TimesNet run; never substitute an editable/stub model.

Set PHMFACTORY_TEST_PYTHON to the normal-wheel venv interpreter. GNU time records
wall time and peak process RSS for the actual CLI fit/checkpoint/test lifecycle.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def test_installed_timesnet_preflight_fit_checkpoint_metrics(tmp_path, record_property):
    executable = os.environ.get("PHMFACTORY_TEST_PYTHON", sys.executable)
    artifacts = Path(os.environ.get("TIMESNET_ARTIFACT_DIR", str(tmp_path / "artifacts"))).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled", MPLBACKEND="Agg")
    probe = subprocess.run([executable, "-c", '''
import importlib, importlib.resources as r, json, sys, time
start = time.perf_counter()
m = importlib.import_module("src.model_factory.CNN.TimesNet")
print(json.dumps(dict(model_file=m.__file__, python=sys.version,
    import_seconds=time.perf_counter()-start,
    data=str(r.files("data")),
    config=str(r.files("configs")/"experiments/model_integration/timesnet_dummy.yaml"))))
'''], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    installed = json.loads(probe.stdout.strip())
    assert "site-packages" in Path(installed["model_file"]).parts, installed
    assert Path(installed["config"]).is_file()
    assert Path(installed["data"]).is_dir()
    (artifacts / "installed_environment.json").write_text(json.dumps(installed, indent=2), encoding="utf-8")
    output = tmp_path / "run"
    options = ["--config", installed["config"],
               "--override", f"data.data_dir={installed['data']}",
               "--override", f"data.cache_dir={tmp_path / 'cache'}",
               "--override", f"environment.output_dir={output}"]
    preflight = subprocess.run([executable, "-m", "phmfactory", "preflight", *options],
                               cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    (artifacts / "preflight.log").write_text(preflight.stdout + preflight.stderr, encoding="utf-8")
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    assert not output.exists()
    assert not (tmp_path / "cache").exists()
    time_file = artifacts / "cli_resources.json"
    command = [executable, "-m", "phmfactory", *options]
    (artifacts / "command.txt").write_text("\n".join(command) + "\n", encoding="utf-8")
    result = subprocess.run(["/usr/bin/time", "-f", '{"wall_seconds":%e,"max_rss_kib":%M}',
                             "-o", str(time_file), *command],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180)
    log = result.stdout + result.stderr
    (artifacts / "cli.log").write_text(log, encoding="utf-8")
    assert result.returncode == 0, log
    values = {}
    fields = {"result_dir", "best_checkpoint", "test_metrics", "run_summary", "primary_metrics"}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in fields:
            values[key] = value.strip()
    assert fields == values.keys(), log
    root = Path(values["result_dir"]).resolve()
    assert root.is_dir() and output.resolve() in root.parents
    for key in ("best_checkpoint", "test_metrics", "run_summary"):
        path = Path(values[key]).resolve()
        assert path.is_file() and root in path.parents, (key, path)
    assert "Restoring states from the checkpoint path" in log, log
    assert values["best_checkpoint"] in log
    summary = json.loads(Path(values["run_summary"]).read_text(encoding="utf-8"))
    assert summary["iterations"] == 1
    assert {"test_acc_Dummy_Data", "test_f1_Dummy_Data"} <= summary["metrics"].keys()
    for metric in summary["metrics"].values():
        assert metric["count"] == 1 and math.isfinite(metric["mean"])
    (artifacts / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (artifacts / "test_metrics.csv").write_text(Path(values["test_metrics"]).read_text(encoding="utf-8"), encoding="utf-8")
    resources = json.loads(time_file.read_text(encoding="utf-8"))
    assert resources["wall_seconds"] > 0 and resources["max_rss_kib"] > 0
    record_property("import_seconds", installed["import_seconds"])
    record_property("cli_wall_seconds", resources["wall_seconds"])
    record_property("cli_peak_process_mib", resources["max_rss_kib"] / 1024)
