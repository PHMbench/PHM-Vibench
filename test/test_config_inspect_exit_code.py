from __future__ import annotations

from scripts import config_inspect


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
