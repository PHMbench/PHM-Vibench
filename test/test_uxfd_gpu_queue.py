import json
import subprocess
from pathlib import Path

from scripts.uxfd_gpu_queue import (
    DEFAULT_QUEUE,
    build_launch_plan,
    build_payload,
    expand_queue,
    main,
    render_shell_plan,
    run_live_preflight,
    summarize_rows,
    validate_queue,
)


PERSISTED_LAUNCH_PLAN = Path("paper/UXFD_paper/results/queue_launch_plan.sh")
PERSISTED_SHARD_DIR = Path("paper/UXFD_paper/results/queue_launch_shards")
PERSISTED_LIVE_PREFLIGHT = Path("paper/UXFD_paper/results/gpu_queue_live_preflight.json")


def test_gpu_queue_expands_all_paper_commands_and_top_bindings() -> None:
    rows = expand_queue(DEFAULT_QUEUE)

    paper_ids = {row.paper_id for row in rows if row.phase != "top_representatives"}
    assert paper_ids == {
        "1D-2D_fusion_explainable",
        "Explainable_FD_Toolkit",
        "LLM_Explainable_FD_Toolkit",
        "MOE_explainable",
        "Neuralsymbolic_theory",
        "Paper_fuzzy_XFD",
        "TII_operator_attention",
    }
    assert sum(row.phase == "proposed" for row in rows) == 7
    assert sum(row.phase == "baselines" for row in rows) >= 42
    assert sum(row.phase == "ablations" for row in rows) >= 42
    assert sum(row.phase == "top_representatives" for row in rows) == 7

    matrix_rows = [row for row in rows if row.phase != "top_representatives"]
    main_py_rows = [row for row in matrix_rows if "python main.py --config" in row.command]
    blocked_rows = [row for row in matrix_rows if row.command.startswith("blocked:")]

    assert main_py_rows
    assert blocked_rows == []
    assert all("CUDA_VISIBLE_DEVICES=0" in row.command for row in main_py_rows)
    assert all(Path(row.matrix_path).exists() for row in matrix_rows)

    summary = summarize_rows(rows)
    assert summary["total"] == len(rows)
    assert summary["top_representatives"] == 7
    assert summary["main_py_commands"] >= 40
    assert summary["blocked"] == 0
    assert summary["per_phase"]["proposed"] == 7
    assert summary["per_phase"]["top_representatives"] == 7


def test_gpu_queue_validation_blocks_execution_until_preflight_passes() -> None:
    validation = validate_queue(DEFAULT_QUEUE)

    assert validation.structural_issues == ()
    assert validation.can_execute is False
    assert "blocked" in validation.resource_reason


def test_gpu_queue_cli_writes_json_manifest_without_preflight_execution(tmp_path: Path) -> None:
    output = tmp_path / "queue" / "dry_run.json"

    assert main(["--format", "json", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["validation"]["can_execute"] is False
    assert payload["validation"]["structural_issues"] == []
    assert len(payload["commands"]) == len(expand_queue(DEFAULT_QUEUE))
    assert payload["summary"]["total"] == len(payload["commands"])
    assert payload["summary"]["top_representatives"] == 7

    blocked_output = tmp_path / "queue" / "blocked.md"
    assert (
        main(["--format", "markdown", "--output", str(blocked_output), "--require-preflight"])
        == 2
    )
    text = blocked_output.read_text(encoding="utf-8")
    assert "Can execute now: `False`" in text
    assert "Total dry-run entries" in text


def test_gpu_queue_payload_summary_matches_expanded_rows() -> None:
    rows = expand_queue(DEFAULT_QUEUE)
    validation = validate_queue(DEFAULT_QUEUE)
    payload = build_payload(rows, validation)

    assert payload["summary"] == summarize_rows(rows)
    assert payload["summary"]["per_paper"]["TII_operator_attention"]["baselines"] >= 6
    assert payload["summary"]["per_paper"]["LLM_Explainable_FD_Toolkit"].get("blocked", 0) == 0


def test_gpu_queue_builds_two_device_launch_plan_without_top_bindings() -> None:
    rows = expand_queue(DEFAULT_QUEUE)
    launch_rows = build_launch_plan(rows)

    assert launch_rows
    assert {row.device for row in launch_rows} == {"0", "1"}
    assert all(row.phase != "top_representatives" for row in launch_rows)
    assert all(row.command.startswith("CUDA_VISIBLE_DEVICES=") for row in launch_rows)
    assert all("paper-local baseline_ablation_matrix.yaml" not in row.command for row in launch_rows)
    assert any(row.command.startswith("CUDA_VISIBLE_DEVICES=1 ") for row in launch_rows)
    assert any(
        row.workdir == "paper/UXFD_paper/Explainable_FD_Toolkit"
        and row.command.startswith("CUDA_VISIBLE_DEVICES=")
        and "python scripts/run_toolkit_ablations.py" in row.command
        for row in launch_rows
    )


def test_gpu_queue_cli_writes_shell_launch_plan_without_running_it(tmp_path: Path) -> None:
    output = tmp_path / "queue" / "launch_plan.sh"

    assert main(["--format", "shell", "--output", str(output)]) == 0

    text = output.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "nvidia-smi -L" in text
    assert "torch.cuda.device_count() >= 2" in text
    assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "CUDA_VISIBLE_DEVICES=1" in text
    assert "(cd paper/UXFD_paper/Explainable_FD_Toolkit && CUDA_VISIBLE_DEVICES=" in text
    assert "paper-local baseline_ablation_matrix.yaml" not in text


def test_gpu_queue_cli_writes_per_gpu_shell_shards(tmp_path: Path) -> None:
    output = tmp_path / "queue" / "launch_plan.sh"
    shard_dir = tmp_path / "queue" / "shards"

    assert (
        main(["--format", "shell", "--output", str(output), "--shard-dir", str(shard_dir)])
        == 0
    )

    gpu0 = (shard_dir / "gpu0.sh").read_text(encoding="utf-8")
    gpu1 = (shard_dir / "gpu1.sh").read_text(encoding="utf-8")
    readme = (shard_dir / "README.md").read_text(encoding="utf-8")

    assert "Launch shard for CUDA device 0" in gpu0
    assert "Launch shard for CUDA device 1" in gpu1
    assert "CUDA_VISIBLE_DEVICES=0" in gpu0
    assert "CUDA_VISIBLE_DEVICES=1" not in gpu0
    assert "CUDA_VISIBLE_DEVICES=1" in gpu1
    assert "CUDA_VISIBLE_DEVICES=0" not in gpu1
    assert "| `0` | `gpu0.sh` |" in readme
    assert "| `1` | `gpu1.sh` |" in readme


def test_persisted_launch_plan_and_shards_match_current_queue() -> None:
    rows = expand_queue(DEFAULT_QUEUE)
    validation = validate_queue(DEFAULT_QUEUE)
    launch_rows = build_launch_plan(rows)

    assert len(launch_rows) == 97
    assert PERSISTED_LAUNCH_PLAN.exists()
    assert PERSISTED_LAUNCH_PLAN.read_text(encoding="utf-8") == render_shell_plan(
        rows,
        validation,
    )

    expected_counts = {"0": 49, "1": 48}
    for device, expected_count in expected_counts.items():
        shard = PERSISTED_SHARD_DIR / f"gpu{device}.sh"
        assert shard.exists()
        text = shard.read_text(encoding="utf-8")
        assert text == render_shell_plan(rows, validation, device_filter=device)
        assert text.count(f"CUDA_VISIBLE_DEVICES={device}") == expected_count
        other_device = "1" if device == "0" else "0"
        assert f"CUDA_VISIBLE_DEVICES={other_device}" not in text

    readme = (PERSISTED_SHARD_DIR / "README.md").read_text(encoding="utf-8")
    assert "These scripts are launch plans, not accepted evidence." in readme
    assert "| `0` | `gpu0.sh` |" in readme
    assert "| `1` | `gpu1.sh` |" in readme


def test_persisted_launch_plan_and_shards_are_shell_syntax_valid() -> None:
    scripts = (
        PERSISTED_LAUNCH_PLAN,
        PERSISTED_SHARD_DIR / "gpu0.sh",
        PERSISTED_SHARD_DIR / "gpu1.sh",
    )

    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_persisted_live_preflight_snapshot_matches_current_queue_shape() -> None:
    rows = expand_queue(DEFAULT_QUEUE)
    validation = validate_queue(DEFAULT_QUEUE)
    payload = json.loads(PERSISTED_LIVE_PREFLIGHT.read_text(encoding="utf-8"))
    expected_payload = json.loads(json.dumps(build_payload(rows, validation)))

    assert payload["validation"] == expected_payload["validation"]
    assert payload["summary"] == summarize_rows(rows)
    assert len(payload["commands"]) == len(rows) == 104

    live = payload["live_preflight"]
    assert live["accepted"] is False
    assert live["nvidia_smi_ok"] is False
    assert live["torch_cuda_available"] is False
    assert live["torch_cuda_device_count"] == 0
    assert live["gpu_names"] == []
    assert "blocked:" in live["reason"]


def test_gpu_queue_live_preflight_is_reported_without_launching_experiments(
    tmp_path: Path,
) -> None:
    output = tmp_path / "queue" / "live_preflight.json"

    assert main(["--format", "json", "--live-preflight", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    live = payload["live_preflight"]
    assert isinstance(live["accepted"], bool)
    assert isinstance(live["nvidia_smi_ok"], bool)
    assert isinstance(live["torch_cuda_available"], bool)
    assert isinstance(live["torch_cuda_device_count"], int)
    assert isinstance(live["gpu_names"], list)
    assert live["reason"]

    direct = run_live_preflight()
    assert isinstance(direct.accepted, bool)
