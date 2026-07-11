from __future__ import annotations

from scripts.validate_docs import iter_doc_files


def test_validate_docs_skips_local_agent_and_obsidian_dirs(tmp_path) -> None:
    (tmp_path / "README.md").write_text("# Root\n", encoding="utf-8")
    for dirname in [".agents", ".claude", ".codex", ".tmp", "obsidian"]:
        path = tmp_path / dirname / "README.md"
        path.parent.mkdir(parents=True)
        path.write_text("[broken](missing.md)\n", encoding="utf-8")

    scanned = {path.relative_to(tmp_path).as_posix() for path in iter_doc_files(tmp_path)}

    assert scanned == {"README.md"}
