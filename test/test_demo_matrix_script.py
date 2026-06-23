from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_demo_matrix_script_defines_smoke_and_full_gates() -> None:
    script = Path("scripts/run_demo_matrix.sh").read_text(encoding="utf-8")

    assert "configs/hydra/experiments/00_smoke/dummy_dg.yaml" in script
    assert "configs/hydra/experiments/01_cross_domain/cwru_dg.yaml" in script
    assert "configs/hydra/experiments/02_cross_system/multi_system_cddg.yaml" in script
    assert "configs/hydra/experiments/03_fewshot/cwru_protonet.yaml" in script
    assert "configs/hydra/experiments/04_cross_system_fewshot/cross_system_tspn.yaml" in script
    assert "configs/hydra/experiments/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml" in script
    assert "configs/hydra/experiments/06_pretrain_cddg/pretrain_hse_cddg.yaml" in script
    assert "PHM_VIBENCH_DATA" in script
    assert "artifacts/manifest.json" in script
    assert "test_result_*.csv" in script
    assert "assert_no_silent_fallback" in script
    assert "test_hse_contrastive_flow_has_nonzero_signal" in script


def test_demo_matrix_full_mode_requires_data_root_before_running() -> None:
    env = os.environ.copy()
    env.pop("PHM_VIBENCH_DATA", None)

    proc = subprocess.run(
        ["bash", "scripts/run_demo_matrix.sh", "--mode", "full"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert proc.returncode == 2
    assert "full matrix requires PHM_VIBENCH_DATA" in proc.stderr
    assert "[RUN]" not in proc.stdout
    assert "[RUN]" not in proc.stderr
