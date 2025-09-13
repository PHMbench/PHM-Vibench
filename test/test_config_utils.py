from types import SimpleNamespace
import os
import yaml

from src.utils import load_config, transfer_namespace, path_name


def test_load_config(tmp_path):
    cfg = {
        "data": {"metadata_file": "ds"},
        "model": {"name": "m"},
        "task": {"type": "T", "name": "N"},
        "trainer": {"name": "tr"},
    }
    cfg_file = tmp_path / "cfg.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    loaded = load_config(str(cfg_file))
    assert loaded == cfg


def test_transfer_namespace():
    ns = transfer_namespace({"a": 1, "b": "c"})
    assert isinstance(ns, SimpleNamespace)
    assert ns.a == 1
    assert ns.b == "c"


def test_path_name(tmp_path):
    configs = {
        "data": {"metadata_file": "ds"},
        "model": {"name": "m"},
        "task": {"type": "T", "name": "N"},
        "trainer": {"name": "tr"},
    }
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result_dir, exp_name = path_name(configs, 1)
    finally:
        os.chdir(cwd)
    assert os.path.isdir(result_dir)
    assert exp_name.startswith("ds/M_m/T_TN_")

