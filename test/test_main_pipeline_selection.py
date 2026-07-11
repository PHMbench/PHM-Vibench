from types import SimpleNamespace

import main as main_module


def test_main_pipeline_override_selects_top_level_pipeline(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: Pipeline_01_default\n", encoding="utf-8")

    imported: list[str] = []

    def fake_import_module(name: str):
        imported.append(name)
        return SimpleNamespace(pipeline=lambda args: args.config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--config",
            str(config),
            "--override",
            "pipeline=Pipeline_02_pretrain_fewshot",
        ],
    )
    monkeypatch.setattr(main_module.importlib, "import_module", fake_import_module)

    assert main_module.main() == str(config)
    assert imported == ["src.Pipeline_02_pretrain_fewshot"]
