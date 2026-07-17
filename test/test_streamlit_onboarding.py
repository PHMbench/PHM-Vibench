from __future__ import annotations

from pathlib import Path

import pytest

from apps.streamlit import config_service as cs
from apps.streamlit import onboarding as ob


def _profile(**changes):
    values = {
        "template_id": "smoke",
        "title": "Smoke",
        "summary": "Offline smoke",
        "difficulty": "Beginner",
        "data_label": "Bundled",
        "device_label": "CPU",
        "estimated_time": "Short",
        "requires_external_data": False,
        "required_paths": ("configs/demo/smoke.yaml", "data/meta.csv"),
    }
    values.update(changes)
    return ob.TemplateProfile(**values)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text("# test\n", encoding="utf-8")
    (tmp_path / "configs" / "demo").mkdir(parents=True)
    (tmp_path / "configs" / "demo" / "smoke.yaml").write_text(
        "x: 1\n", encoding="utf-8"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "meta.csv").write_text("id\n1\n", encoding="utf-8")
    return tmp_path


def test_load_template_profiles_preserves_user_guidance(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text(
        "version: 1\n"
        "profiles:\n"
        "  smoke:\n"
        "    title: Offline starter\n"
        "    summary: First run\n"
        "    difficulty: Beginner\n"
        "    data_label: Bundled\n"
        "    device_label: CPU\n"
        "    estimated_time: Short\n"
        "    requires_external_data: false\n"
        "    badges: [Offline, CPU]\n"
        "    required_paths: [configs/demo/smoke.yaml]\n",
        encoding="utf-8",
    )

    profile = ob.load_template_profiles(path)["smoke"]

    assert profile.badges == ("Offline", "CPU")
    assert profile.required_paths == ("configs/demo/smoke.yaml",)
    assert profile.requires_external_data is False


def test_profile_paths_reject_repository_escape(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text(
        "profiles:\n"
        "  smoke:\n"
        "    requires_external_data: false\n"
        "    required_paths: [../secret]\n",
        encoding="utf-8",
    )

    with pytest.raises(ob.OnboardingError, match="repository-relative"):
        ob.load_template_profiles(path)


def test_apply_safe_defaults_keeps_run_history() -> None:
    state = {
        "ui_mode": "Advanced",
        "selected_template_id": "other",
        "validation_report": object(),
        "field::other::device": "cuda",
        "active_run_id": "run-1",
        "selected_run_id": "run-1",
    }

    ob.apply_safe_defaults(state, "smoke")

    assert state["ui_mode"] == "Quick Start"
    assert state["selected_template_id"] == "smoke"
    assert state["validation_report"] is None
    assert "field::other::device" not in state
    assert state["active_run_id"] == "run-1"
    assert state["selected_run_id"] == "run-1"


def test_readiness_is_ready_for_complete_offline_environment(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    modules = {"streamlit", "yaml", "torch", "pytorch_lightning"}

    report = ob.collect_environment_readiness(
        root,
        _profile(),
        module_finder=lambda name: object() if name in modules else None,
        version_reader=lambda name: "1.0",
        access_checker=lambda path, mode: True,
    )

    assert report.can_execute is True
    assert report.blocked == ()


def test_readiness_blocks_missing_training_dependency(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    modules = {"streamlit", "yaml", "torch"}

    report = ob.collect_environment_readiness(
        root,
        _profile(),
        module_finder=lambda name: object() if name in modules else None,
        version_reader=lambda name: "1.0",
        access_checker=lambda path, mode: True,
    )

    assert report.can_execute is False
    assert [item.key for item in report.blocked] == ["dependency:pytorch_lightning"]


def test_local_config_is_visible_warning_not_blocker(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    local = root / "configs" / "local"
    local.mkdir()
    (local / "local.yaml").write_text("trainer: {}\n", encoding="utf-8")

    report = ob.collect_environment_readiness(
        root,
        _profile(),
        module_finder=lambda name: object(),
        version_reader=lambda name: "1.0",
        access_checker=lambda path, mode: True,
    )

    assert report.can_execute is True
    assert [item.key for item in report.warnings] == ["local-config"]


def test_bundled_template_requires_declared_assets(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "data" / "meta.csv").unlink()

    status = ob.assess_template_data(root, {"data": {}}, _profile())

    assert status.ready is False
    assert "data/meta.csv" in status.detail


def test_bundled_template_validates_final_overridden_data_path(tmp_path: Path) -> None:
    root = _repo(tmp_path)

    status = ob.assess_template_data(
        root,
        {"data": {"data_dir": "missing", "metadata_file": "meta.csv"}},
        _profile(),
    )

    assert status.ready is False
    assert "Configured smoke data" in status.detail


def test_external_template_resolves_data_root_and_metadata(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (external / "metadata.xlsx").write_bytes(b"test")
    profile = _profile(
        requires_external_data=True,
        required_paths=(),
        data_label="External",
    )

    status = ob.assess_template_data(
        root,
        {"data": {"data_dir": str(external), "metadata_file": "metadata.xlsx"}},
        profile,
    )

    assert status.ready is True
    assert status.data_root == str(external.resolve())
    assert status.metadata_path == str((external / "metadata.xlsx").resolve())


def test_external_template_returns_actionable_missing_path(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    profile = _profile(requires_external_data=True, required_paths=())

    status = ob.assess_template_data(
        root,
        {"data": {"data_dir": "missing", "metadata_file": "meta.xlsx"}},
        profile,
    )

    assert status.ready is False
    assert "configs/local/local.yaml" in status.action


def test_every_maintained_demo_has_explicit_user_profile() -> None:
    root = Path(__file__).parents[1]
    profiles = ob.load_template_profiles(
        root / "apps" / "streamlit" / "template_profiles.yaml"
    )
    maintained = {
        entry.id
        for entry in cs.load_registry(root)
        if entry.category == "demo" and entry.status == "sanity_ok"
    }

    assert maintained
    assert maintained.issubset(profiles.keys())


def test_default_smoke_profile_assets_exist() -> None:
    root = Path(__file__).parents[1]
    profiles = ob.load_template_profiles(
        root / "apps" / "streamlit" / "template_profiles.yaml"
    )
    smoke = profiles["demo_00_smoke_dummy_dg"]

    assert smoke.requires_external_data is False
    assert smoke.required_paths
    assert all((root / path).exists() for path in smoke.required_paths)
