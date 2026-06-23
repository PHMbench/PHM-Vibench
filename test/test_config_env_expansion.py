from __future__ import annotations

import pytest

from src.configs.config_utils import load_config


def _write_minimal_config(path, data_dir: str) -> None:
    path.write_text(
        "\n".join(
            [
                "pipeline: Pipeline_01_default",
                "data:",
                f"  data_dir: {data_dir!r}",
                "  metadata_file: metadata.csv",
                "model:",
                "  name: M_Test",
                "  type: Test",
                "task:",
                "  name: classification",
                "  type: DG",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_env_placeholder_uses_default_when_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PHM_VIBENCH_DATA", raising=False)
    config = tmp_path / "config.yaml"
    _write_minimal_config(config, "${PHM_VIBENCH_DATA:-data}")

    cfg = load_config(config)

    assert cfg.data.data_dir == "data"


def test_env_placeholder_uses_environment_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PHM_VIBENCH_DATA", "/tmp/phm-data")
    config = tmp_path / "config.yaml"
    _write_minimal_config(config, "${PHM_VIBENCH_DATA:-data}")

    cfg = load_config(config)

    assert cfg.data.data_dir == "/tmp/phm-data"


def test_contrastive_pairing_rejects_unknown_value(tmp_path) -> None:
    config = tmp_path / "bad_pairing.yaml"
    config.write_text(
        "\n".join(
            [
                "pipeline: Pipeline_01_default",
                "data:",
                "  data_dir: data",
                "  metadata_file: metadata.csv",
                "model:",
                "  name: M_Test",
                "  type: Test",
                "task:",
                "  name: hse_contrastive",
                "  type: pretrain",
                "  contrast_loss: INFONCE",
                "  temperature: 0.07",
                "  contrastive_pairing: guessed_pairs",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="对比配对方式"):
        load_config(config)
