from __future__ import annotations

from pathlib import Path

from scripts.task_experiment_matrix import SUPPORT_STATUSES, derive_matrix, load_task_families
from src.configs.config_utils import load_config


def _task_dict(config_path: str) -> dict:
    return load_config(config_path).task.__dict__


def test_registry_task_keys_are_unique_and_config_refs_exist() -> None:
    report = derive_matrix()

    assert report.duplicate_task_keys == ()
    assert report.missing_config_task_refs == ()


def test_every_registry_task_family_has_exactly_one_support_status() -> None:
    report = derive_matrix()
    families = load_task_families()

    assert set(report.family_statuses) == {family.key for family in families}
    assert all(item.status in SUPPORT_STATUSES for item in report.family_statuses.values())
    assert all(item.reason for item in report.family_statuses.values())


def test_matrix_coverage_includes_core_phm_families_or_reasons() -> None:
    report = derive_matrix()

    for task_type in ["DG", "CDDG", "FS", "GFS", "pretrain"]:
        matches = [
            item for item in report.family_statuses.values() if item.family.task_type == task_type
        ]
        assert matches, task_type
        assert all(item.status != "unsupported" for item in matches)

    assert "regression" in report.absent_capabilities
    assert "multi-task" in report.absent_capabilities


def test_task_registry_paths_dataset_paths_and_batch_formats_exist() -> None:
    for family in load_task_families():
        assert family.batch_format, family.key
        assert (Path("src/task_factory") / family.path).exists(), family.key
        assert (Path("src/data_factory") / family.dataset_path).exists(), family.key


def test_fewshot_and_gfs_configs_expose_feasibility_fields() -> None:
    fs_task = _task_dict("configs/hydra/experiments/03_fewshot/cwru_protonet.yaml")
    gfs_task = _task_dict("configs/hydra/experiments/04_cross_system_fewshot/cross_system_tspn.yaml")

    assert fs_task["type"] == "FS"
    assert fs_task["name"] == "classification"
    assert fs_task["n_way"] > 0
    assert fs_task["k_shot"] > 0
    assert fs_task["q_query"] > 0

    assert gfs_task["type"] == "GFS"
    assert gfs_task["name"] == "classification"
    assert gfs_task["num_labels"] > 0
    assert gfs_task["num_support"] > 0
    assert gfs_task["num_query"] > 0


def test_dg_cddg_and_pretrain_configs_expose_compatibility_fields() -> None:
    dg_task = _task_dict("configs/hydra/experiments/01_cross_domain/cwru_dg.yaml")
    cddg_task = _task_dict("configs/hydra/experiments/02_cross_system/multi_system_cddg.yaml")
    pretrain_task = _task_dict(
        "configs/hydra/experiments/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml"
    )

    assert dg_task["type"] == "DG"
    assert dg_task["source_domain_id"]
    assert dg_task["target_domain_id"]
    assert dg_task["target_system_id"]

    assert cddg_task["type"] == "CDDG"
    assert cddg_task["target_system_id"]
    assert cddg_task["target_domain_num"] > 0

    assert pretrain_task["type"] == "pretrain"
    assert pretrain_task["name"] == "hse_contrastive"
    assert pretrain_task["contrast_loss"] == "INFONCE"
    assert pretrain_task["contrastive_pairing"] == "simclr_2view"
