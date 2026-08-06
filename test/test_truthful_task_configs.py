from phmfactory.config import resolve_config


def test_fs_compatibility_demo_is_non_episodic_supervised_classification():
    resolved = resolve_config("configs/demo/03_fewshot/cwru_protonet.yaml")
    task = resolved.data["task"]

    assert task["type"] == "FS"
    assert task["name"] == "classification"
    assert task["loss"] == "CE"
    for unused_field in ("n_way", "k_shot", "q_query", "episodes_per_epoch"):
        assert unused_field not in task


def test_gfs_compatibility_demo_is_hierarchically_sampled_ce():
    resolved = resolve_config(
        "configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml"
    )
    task = resolved.data["task"]

    assert task["type"] == "GFS"
    assert task["name"] == "classification"
    assert task["loss"] == "CE"
    assert task["num_systems"] == 1
    assert task["num_support"] == 5
    assert task["num_query"] == 15


def test_cross_system_compatibility_demo_is_currently_single_system():
    resolved = resolve_config(
        "configs/demo/02_cross_system/multi_system_cddg.yaml"
    )
    assert resolved.data["task"]["target_system_id"] == [1]


def test_pipeline02_demo_is_single_stage_hse_pretraining():
    resolved = resolve_config(
        "configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml"
    )

    assert "stages" not in resolved.data
    assert resolved.data["task"]["type"] == "pretrain"
    assert resolved.data["task"]["name"] == "hse_contrastive"
