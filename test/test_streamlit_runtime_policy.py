from __future__ import annotations

from pathlib import Path

import yaml

from apps.streamlit import runtime_policy as policy
from apps.streamlit.config_service import CONFIG_BLOCKS, ValidationReport


def test_template_resolution_passes_no_hidden_local_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("environment: {}\n", encoding="utf-8")
    seen = {}

    def fake_inspect(repo_root, path, overrides=(), timeout=90.0, local_config_path=None):
        seen["local"] = local_config_path
        assert path == config_path
        assert local_config_path is None
        return ValidationReport(True, ("python",))

    monkeypatch.setattr(policy, "inspect_config", fake_inspect)
    report = policy.inspect_portable_config(tmp_path, config_path)

    assert report.ok
    assert seen["local"] is None


def test_execution_yaml_passes_no_hidden_local_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seen = {}

    def fake_inspect(repo_root, path, overrides=(), timeout=90.0, local_config_path=None):
        seen["path"] = path
        seen["local"] = local_config_path
        assert path.is_file()
        assert local_config_path is None
        return ValidationReport(True, ("python",))

    monkeypatch.setattr(policy, "inspect_config", fake_inspect)
    yaml_text = yaml.safe_dump({block: {} for block in CONFIG_BLOCKS})
    report = policy.inspect_execution_yaml(tmp_path, yaml_text)

    assert report.ok
    assert seen["local"] is None
    assert not seen["path"].exists()
