try:
    from streamlit_app import (
        list_config_files,
        list_pipeline_modules,
        cast_value,
        update_config,
        config_to_yaml,
    )
except Exception:  # pragma: no cover - optional dependencies
    import pytest
    pytest.skip("streamlit or pandas not installed", allow_module_level=True)


def test_list_config_files():
    files = list_config_files()
    assert any(f.endswith('CWRU.yaml') for f in files)


def test_list_pipeline_modules():
    modules = list_pipeline_modules()
    assert 'Pipeline_01_default' in modules


def test_cast_value():
    assert cast_value(1, '2') == 2
    assert cast_value(True, False) is False
    assert cast_value([1, 2], '[3, 4]') == [3, 4]


def test_update_config():
    base = {"data": {"a": 1}}
    sections = {"data": {"a": 2}}
    cfg = update_config(base, sections, {"foo": "bar"})
    assert cfg["data"]["a"] == 2
    assert cfg["environment"]["foo"] == "bar"


def test_config_to_yaml():
    cfg = {"a": 1}
    yaml_str = config_to_yaml(cfg)
    assert "a: 1" in yaml_str

