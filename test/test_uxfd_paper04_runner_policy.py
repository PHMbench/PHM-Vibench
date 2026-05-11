from pathlib import Path


PAPER04_ROOT = Path("paper/UXFD_paper/MOE_explainable")
RUNNER_FILES = (
    PAPER04_ROOT / "scripts/run_real_dataset_probe.py",
    PAPER04_ROOT / "scripts/run_expert_ablation_probe.py",
    PAPER04_ROOT / "scripts/run_dataset_bridge_minimal.py",
)


def test_paper04_probe_runners_do_not_encode_stale_root_or_gpu_policy() -> None:
    forbidden = (
        "PHM-Vibench copy 2",
        "GPU_ID = 2",
        "CUDA_VISIBLE_DEVICES=2",
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "main_com.py",
        "--config_dir",
    )

    for runner in RUNNER_FILES:
        text = runner.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, (runner, marker)


def test_paper04_real_dataset_probe_resolves_current_repo_root() -> None:
    text = (PAPER04_ROOT / "scripts/run_real_dataset_probe.py").read_text(encoding="utf-8")

    assert "PAPER_ROOT = Path(__file__).resolve().parent.parent" in text
    assert "REPO_ROOT = PAPER_ROOT.parents[2]" in text
