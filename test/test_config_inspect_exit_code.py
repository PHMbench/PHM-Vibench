from __future__ import annotations

from pathlib import Path

import pytest

from scripts import config_inspect


def test_local_config_is_never_discovered_implicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_local = tmp_path / "configs" / "local" / "local.yaml"
    default_local.parent.mkdir(parents=True)
    default_local.write_text("environment:\n  seed: 99\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert config_inspect._find_local_override_path(None) is None


def test_explicit_missing_local_config_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="--local-config not found"):
        config_inspect._find_local_override_path(str(missing))


def test_explicit_local_config_must_be_a_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--local-config not found"):
        config_inspect._find_local_override_path(str(tmp_path))


def test_config_inspect_returns_nonzero_when_sanity_fails(monkeypatch, capsys) -> None:
    result = config_inspect.InspectResult(
        resolved={},
        sources={},
        targets={},
        sanity=[
            {
                "check": "pipeline_import",
                "ok": False,
                "message": "pipeline import failed",
                "fix": "install the missing dependency",
            }
        ],
    )
    monkeypatch.setattr(config_inspect, "inspect_config", lambda *args, **kwargs: result)

    exit_code = config_inspect.main(["--config", "dummy.yaml"])

    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().out


def test_config_inspect_returns_zero_when_sanity_passes(monkeypatch) -> None:
    result = config_inspect.InspectResult(
        resolved={},
        sources={},
        targets={},
        sanity=[
            {
                "check": "pipeline_import",
                "ok": True,
                "message": "pipeline import passed",
                "fix": "",
            }
        ],
    )
    monkeypatch.setattr(config_inspect, "inspect_config", lambda *args, **kwargs: result)

    assert config_inspect.main(["--config", "dummy.yaml"]) == 0
