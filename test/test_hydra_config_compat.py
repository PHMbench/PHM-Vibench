import pytest

from main import resolve_pipeline_name, validate_pipeline_name
from src.configs.config_utils import load_config, is_hydra_available, merge_with_local_override
from src.utils.config_utils import parse_overrides


def test_load_config_preserves_base_config_composition():
    cfg = load_config("configs/demo/00_smoke/dummy_dg.yaml")

    assert cfg.pipeline == "Pipeline_01_default"
    assert cfg.data.metadata_file == "metadata_dummy.csv"
    assert cfg.data.batch_size == 4
    assert cfg.model.name == "M_01_ISFM"
    assert cfg.trainer.num_epochs == 1


def test_dot_key_overrides_still_merge_into_nested_config():
    cfg = load_config(
        "configs/demo/00_smoke/dummy_dg.yaml",
        {"trainer.num_epochs": 2, "data.batch_size": 8},
    )

    assert cfg.trainer.num_epochs == 2
    assert cfg.data.batch_size == 8


def test_hydra_style_plus_overrides_parse_as_existing_keys():
    overrides = parse_overrides(["+trainer.num_epochs=3", "task.target_system_id=[1, 2]"])

    assert overrides["trainer"]["num_epochs"] == 3
    assert overrides["task"]["target_system_id"] == [1, 2]


def test_hydra_backend_availability_flag_is_boolean():
    assert isinstance(is_hydra_available(), bool)


def test_local_override_is_explicit_opt_in(tmp_path):
    local_cfg = tmp_path / "local.yaml"
    local_cfg.write_text("data:\n  data_dir: /tmp/not_demo\n", encoding="utf-8")

    cfg = merge_with_local_override("configs/demo/10_generative/dummy_generative_cfm.yaml")
    assert cfg.data.data_dir == "data"
    assert cfg.data.metadata_file == "metadata_dummy.csv"

    overridden = merge_with_local_override(
        "configs/demo/10_generative/dummy_generative_cfm.yaml",
        local_cfg,
    )
    assert str(overridden.data.data_dir) == "/tmp/not_demo"


def test_pipeline_name_must_be_whitelisted():
    assert validate_pipeline_name("Pipeline_06_generative") == "Pipeline_06_generative"
    with pytest.raises(ValueError):
        validate_pipeline_name("os")


def test_resolve_pipeline_name_reads_demo_yaml():
    assert (
        resolve_pipeline_name("configs/demo/10_generative/dummy_generative_cfm.yaml")
        == "Pipeline_06_generative"
    )
