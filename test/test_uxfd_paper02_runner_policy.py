from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import importlib.util

import h5py
import numpy as np
import pandas as pd
import yaml


PAPER02_ROOT = Path("paper/UXFD_paper/1D-2D_fusion_explainable")
RUNNER_DIR = PAPER02_ROOT / "scripts"
RUNNER_FILES = (
    RUNNER_DIR / "run_ablation_study.py",
    RUNNER_DIR / "run_3seed_stability_test.py",
    RUNNER_DIR / "run_multi_dataset_validation.py",
    RUNNER_DIR / "run_ablation_studies.sh",
)


def test_paper02_legacy_runners_do_not_encode_stale_gpu_or_root_policy() -> None:
    forbidden = (
        "GPU_ID = 2",
        "CUDA_VISIBLE_DEVICES=2",
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "configs/unified_baseline",
        "PHM-Vibench copy 2",
        "save/task_THU_018_basic",
        "main_com.py",
    )

    for runner in RUNNER_FILES:
        text = runner.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, (runner, marker)


def test_paper02_ablation_runner_dry_run_uses_current_root_and_gpu_01(tmp_path: Path) -> None:
    runner = RUNNER_DIR / "run_ablation_study.py"
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--output_dir",
            str(tmp_path),
            "--gpu_id",
            "1",
            "--configs",
            "1D_only",
            "No_Statistical",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload_start = result.stdout.find("{")
    assert payload_start >= 0, result.stdout
    payload = json.loads(result.stdout[payload_start:])

    assert payload["dry_run"] is True
    assert payload["allowed_gpu_ids"] == [0, 1]
    assert len(payload["configs"]) == 2

    for item in payload["configs"]:
        config_path = Path(item["config_path"])
        assert config_path.exists()
        assert item["gpu_id"] == 1
        assert item["repo_root"].endswith("PHM-Vibench_fix")

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["pipeline"] == "Pipeline_01_default"
        assert config["trainer"]["device"] == "cuda"
        assert config["model"]["device"] == "cuda"
        assert config["model"]["ablation_flags"]
        assert "configs/base/data/base_cross_domain.yaml" in config["base_configs"]["data"]


def test_paper02_local_hdf5_dataset_loader_uses_current_phm_vibench_layout(
    tmp_path: Path,
) -> None:
    module_path = PAPER02_ROOT / "code/utils/datasets.py"
    spec = importlib.util.spec_from_file_location("paper02_datasets", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metadata = pd.DataFrame(
        {
            "Id": [47044, 47045, 47046, 47047],
            "Dataset_id": [14, 14, 14, 14],
            "Name": ["RM_018_THU24"] * 4,
            "Label": [0, 1, 0, 1],
        }
    )
    metadata.to_excel(tmp_path / "metadata.xlsx", index=False)

    with h5py.File(tmp_path / "RM_018_THU24.h5", "w") as h5_file:
        for row_id in metadata["Id"]:
            h5_file.create_dataset(
                str(row_id),
                data=np.ones((64, 2, 1), dtype=np.float32) * float(row_id),
            )

    dataset = module.PHMVibenchWindowDataset(
        data_dir=tmp_path,
        dataset_task="THU_018_basic",
        seq_len=32,
        num_classes=2,
        max_records=4,
        windows_per_record=2,
    )

    signal, label = dataset[0]
    assert len(dataset) == 8
    assert tuple(signal.shape) == (32,)
    assert label.item() in {0, 1}


def test_paper02_dummy_dataset_supports_short_smoke_windows() -> None:
    module_path = PAPER02_ROOT / "code/utils/datasets.py"
    spec = importlib.util.spec_from_file_location("paper02_datasets_short", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset = module.DummyDataset(num_samples=3, seq_len=16, num_classes=10)
    signal, label = dataset[2]
    assert tuple(signal.shape) == (16,)
    assert 0 <= label.item() < 10
