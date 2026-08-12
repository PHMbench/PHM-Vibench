from __future__ import annotations

import json
import subprocess

import pytest

from src.utils.p05_code_snapshot import export_p05_code_snapshot


def _source_tree(tmp_path):
    root = tmp_path / "source"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "main.py").write_text("from src.pkg import value\n", encoding="utf-8")
    (root / "src" / "pkg" / "__init__.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "main.py", "src/pkg/__init__.py"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=P05 Test",
            "-c",
            "user.email=p05@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return root


def test_code_snapshot_covers_commit_and_all_runtime_python_sources(tmp_path) -> None:
    root = _source_tree(tmp_path)
    package = tmp_path / "snapshot"

    created = export_p05_code_snapshot(package, source_root=root)
    reused = export_p05_code_snapshot(package, source_root=root)

    assert created.status == "created"
    assert reused.status == "reused"
    manifest = json.loads(created.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["git_commit"]) == 40
    assert [item["path"] for item in manifest["files"]] == [
        "main.py",
        "src/pkg/__init__.py",
    ]
    assert manifest["file_count"] == 2
    assert reused.semantic_sha256 == created.semantic_sha256


def test_code_snapshot_detects_dirty_and_untracked_runtime_source(tmp_path) -> None:
    root = _source_tree(tmp_path)
    first = export_p05_code_snapshot(tmp_path / "first", source_root=root)
    (root / "src" / "pkg" / "__init__.py").write_text("value = 2\n", encoding="utf-8")
    (root / "src" / "pkg" / "new.py").write_text("new = True\n", encoding="utf-8")
    second = export_p05_code_snapshot(tmp_path / "second", source_root=root)

    assert first.semantic_sha256 != second.semantic_sha256
    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert [item["path"] for item in manifest["files"]] == [
        "main.py",
        "src/pkg/__init__.py",
        "src/pkg/new.py",
    ]


def test_code_snapshot_refuses_conflict_and_source_symlink(tmp_path) -> None:
    root = _source_tree(tmp_path)
    package = tmp_path / "snapshot"
    export_p05_code_snapshot(package, source_root=root)
    (root / "main.py").write_text("changed = True\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="conflicts"):
        export_p05_code_snapshot(package, source_root=root)

    linked = root / "src" / "pkg" / "linked.py"
    linked.symlink_to(root / "main.py")
    with pytest.raises(ValueError, match="source symlink"):
        export_p05_code_snapshot(tmp_path / "linked", source_root=root)
