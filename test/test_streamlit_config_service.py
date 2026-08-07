from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from apps.streamlit import config_service as cs


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text("# test\n", encoding="utf-8")
    (tmp_path / "configs" / "demo").mkdir(parents=True)
    return tmp_path


def _write_registry(root: Path, body: str) -> None:
    (root / "configs" / "config_registry.csv").write_text(body, encoding="utf-8")


def _inspector_payload() -> dict:
    return {
        "local_config_path": None,
        "resolved": {block: {} for block in cs.CONFIG_BLOCKS},
        "sources": {},
        "targets": {},
        "sanity": [{"check": "ok", "ok": True, "message": "pass"}],
    }


def test_load_registry_preserves_unknown_columns(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write_registry(
        root,
        "id,category,path,description,status,new_column\n"
        "demo,demo,configs/demo/demo.yaml,Demo,sanity_ok,future-value\n",
    )
    (root / "configs" / "demo" / "demo.yaml").write_text(
        "environment: {}\n", encoding="utf-8"
    )

    entry = cs.load_registry(root)[0]

    assert entry.id == "demo"
    assert entry.metadata["new_column"] == "future-value"


def test_missing_registry_is_actionable(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(cs.ConfigPathError, match="does not exist"):
        cs.load_registry(root)


def test_registry_requires_columns(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write_registry(root, "id,path\ndemo,configs/demo/demo.yaml\n")
    with pytest.raises(cs.RegistryError, match="missing required columns"):
        cs.load_registry(root)


def test_resolve_repo_path_rejects_traversal(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    outside = tmp_path.parent / "outside.yaml"
    outside.write_text("x: 1\n", encoding="utf-8")

    with pytest.raises(cs.ConfigPathError, match="escapes"):
        cs.resolve_repo_path(root, "../outside.yaml", yaml_only=True)


def test_resolve_repo_path_rejects_non_config_prefix(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "other").mkdir()
    (root / "other" / "demo.yaml").write_text("x: 1\n", encoding="utf-8")

    with pytest.raises(cs.ConfigPathError, match="must stay under"):
        cs.resolve_repo_path(root, "other/demo.yaml", yaml_only=True)


def test_invalid_yaml_reports_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("environment: [\n", encoding="utf-8")

    with pytest.raises(cs.ConfigFormatError, match="line"):
        cs.load_yaml_mapping(path)


def test_yaml_root_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(cs.ConfigFormatError, match="root must be a mapping"):
        cs.load_yaml_mapping(path)


def test_parse_advanced_yaml_requires_five_blocks() -> None:
    with pytest.raises(cs.ConfigFormatError, match="missing resolved mapping blocks"):
        cs.parse_yaml_text("environment: {}\n")


def test_field_alias_selects_existing_legacy_path(tmp_path: Path) -> None:
    catalog_path = tmp_path / "field_catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "fields": {
                    "learning_rate": {
                        "label": "LR",
                        "widget": "number",
                        "paths": ["task.lr", "trainer.learning_rate"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    catalog = cs.load_catalog(catalog_path)
    resolved = {"trainer": {"learning_rate": 0.01}}

    assert cs.select_field_path(resolved, catalog.fields[0]) == "trainer.learning_rate"


def test_field_alias_uses_canonical_path_when_missing(tmp_path: Path) -> None:
    catalog_path = tmp_path / "field_catalog.yaml"
    catalog_path.write_text(
        "fields:\n  seed:\n    paths: [environment.seed]\n",
        encoding="utf-8",
    )
    catalog = cs.load_catalog(catalog_path)

    assert cs.select_field_path({}, catalog.fields[0]) == "environment.seed"


def test_override_types_round_trip_through_yaml() -> None:
    values = [1, 0.001, True, False, None, "cpu", "a value", [1, 2], {"a": 1}]
    for value in values:
        token = cs.serialize_override_value(value)
        assert yaml.safe_load(token) == value


def test_override_rejects_unsupported_type() -> None:
    with pytest.raises(cs.OverrideError, match="Unsupported override value type"):
        cs.serialize_override_value(object())


def test_override_rejects_invalid_key() -> None:
    with pytest.raises(cs.OverrideError, match="Invalid override key"):
        cs.validate_override_key("trainer.num_epochs;rm -rf")


def test_override_rejects_base_config_mutation() -> None:
    with pytest.raises(cs.OverrideError, match="base_configs"):
        cs.validate_override_key("base_configs.model")


def test_parse_override_lines_preserves_types() -> None:
    parsed = cs.parse_override_lines(
        "# comment\ntrainer.num_epochs=2\ntrainer.device=cpu\ntask.ids=[1, 2]\n"
    )
    assert parsed == (
        ("trainer.num_epochs", 2),
        ("trainer.device", "cpu"),
        ("task.ids", [1, 2]),
    )


def test_parse_override_lines_rejects_duplicate() -> None:
    with pytest.raises(cs.OverrideError, match="Duplicate"):
        cs.parse_override_lines("trainer.num_epochs=1\ntrainer.num_epochs=2\n")


def test_build_main_command_repeats_override_flags(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    config = root / "configs" / "demo" / "demo.yaml"
    config.write_text("environment: {}\n", encoding="utf-8")

    command = cs.build_main_command(
        root,
        config,
        [("trainer.num_epochs", 1), ("trainer.device", "cpu")],
        python_executable="python",
    )

    assert command == (
        "python",
        "main.py",
        "--config",
        "configs/demo/demo.yaml",
        "--override",
        "trainer.num_epochs=1",
        "--override",
        'trainer.device="cpu"',
    )


def test_apply_overrides_does_not_mutate_input() -> None:
    original = {"trainer": {"num_epochs": 10}, "task": {"lr": 0.001}}
    updated = cs.apply_overrides(original, [("trainer.num_epochs", 1)])

    assert updated["trainer"]["num_epochs"] == 1
    assert original["trainer"]["num_epochs"] == 10


def test_inspect_config_parses_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    config = root / "configs" / "demo" / "demo.yaml"
    config.write_text("environment: {}\n", encoding="utf-8")
    payload = _inspector_payload()

    def fake_run(command, **kwargs):
        assert kwargs["cwd"] == str(root)
        assert "--override" in command
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    report = cs.inspect_config(root, config, [("trainer.num_epochs", 1)])

    assert report.ok is True
    assert report.resolved == payload["resolved"]
    assert report.local_config_path is None


def test_inspect_config_reports_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    config = root / "configs" / "demo" / "demo.yaml"
    config.write_text("environment: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        cs.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: torch",
        ),
    )
    report = cs.inspect_config(root, config)

    assert report.ok is False
    assert "core dependencies" in report.error
    assert "torch" in report.stderr


def test_group_entries_is_catalog_driven(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "fields:\n  device:\n    paths: [trainer.device]\n"
        "template_groups:\n"
        "  quick:\n"
        "    include_ids: [smoke]\n"
        "    include_statuses: [sanity_ok]\n",
        encoding="utf-8",
    )
    catalog = cs.load_catalog(catalog_path)
    entries = (
        cs.RegistryEntry("smoke", "demo", "a", "", status="sanity_ok"),
        cs.RegistryEntry("other", "demo", "b", "", status="sanity_ok"),
    )

    assert [item.id for item in cs.group_entries(entries, catalog, "quick")] == [
        "smoke"
    ]


def test_normalize_overrides_last_source_wins() -> None:
    combined = cs.normalize_overrides(
        (
            ("trainer.num_epochs", 2),
            ("trainer.device", "cpu"),
            ("trainer.num_epochs", 3),
        )
    )

    assert combined == (
        ("trainer.num_epochs", 3),
        ("trainer.device", "cpu"),
    )


def test_catalog_aliases_match_maintained_smoke_keyspace() -> None:
    catalog = cs.load_catalog(
        Path(__file__).parents[1] / "apps" / "streamlit" / "field_catalog.yaml"
    )
    smoke = {
        "environment": {"seed": 0, "output_dir": "results/demo"},
        "data": {"batch_size": 4, "num_workers": 0},
        "model": {},
        "task": {"lr": 0.001, "epochs": 10},
        "trainer": {"num_epochs": 1, "device": "cpu"},
    }
    paths = {spec.key: cs.select_field_path(smoke, spec) for spec in catalog.fields}

    assert paths["learning_rate"] == "task.lr"
    assert paths["batch_size"] == "data.batch_size"
    assert paths["num_workers"] == "data.num_workers"
    assert paths["epochs"] == "trainer.num_epochs"


def test_inspect_yaml_text_uses_external_temp_without_local_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    seen = {}

    def fake_inspect(
        repo_root,
        config_path,
        overrides=(),
        timeout=90.0,
        local_config_path=None,
    ):
        seen["path"] = config_path
        seen["local"] = local_config_path
        assert repo_root == root
        assert config_path.is_file()
        assert local_config_path is None
        with pytest.raises(ValueError):
            config_path.resolve().relative_to(root.resolve())
        return cs.ValidationReport(ok=True, command=("python",))

    monkeypatch.setattr(cs, "inspect_config", fake_inspect)
    yaml_text = yaml.safe_dump({block: {} for block in cs.CONFIG_BLOCKS})

    report = cs.inspect_yaml_text(root, yaml_text)

    assert report.ok is True
    assert seen["local"] is None
    assert not seen["path"].exists()
    assert not (root / ".streamlit").exists()


def test_inspect_config_passes_only_an_explicit_local_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    config = root / "configs" / "demo" / "demo.yaml"
    config.write_text("environment: {}\n", encoding="utf-8")
    local = tmp_path / "machine.yaml"
    local.write_text("trainer: {device: cpu}\n", encoding="utf-8")
    payload = _inspector_payload()
    payload["local_config_path"] = str(local.resolve())

    def fake_run(command, **kwargs):
        index = command.index("--local-config")
        assert command[index + 1] == str(local.resolve())
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    report = cs.inspect_config(root, config, local_config_path=local)

    assert report.ok is True
    assert report.local_config_path == str(local.resolve())
