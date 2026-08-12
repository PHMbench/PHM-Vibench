from phmfactory import cli


def test_cli_pipeline_override_selects_canonical_pipeline(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: Pipeline_01_Fault_Diagnosis\n", encoding="utf-8")

    observed = {}

    def fake_run(args):
        observed["pipeline"] = cli._resolve_pipeline(args, args.config)
        return args.config

    monkeypatch.setattr(cli, "run", fake_run)

    assert cli.main(
        [
            "--config",
            str(config),
            "--override",
            "pipeline=Pipeline_02_Pretraining_Few_Shot",
        ]
    ) == str(config)
    assert observed["pipeline"] == "Pipeline_02_Pretraining_Few_Shot"
