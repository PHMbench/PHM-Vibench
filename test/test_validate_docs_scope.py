from __future__ import annotations

from scripts.validate_docs import iter_doc_files


def test_validate_docs_skips_local_agent_and_obsidian_dirs(tmp_path) -> None:
    (tmp_path / "README.md").write_text("# Root\n", encoding="utf-8")
    for dirname in [".agents", ".tmp", "obsidian"]:
        path = tmp_path / dirname / "README.md"
        path.parent.mkdir(parents=True)
        path.write_text("[broken](missing.md)\n", encoding="utf-8")
    for dirname in [
        ".claude/handoffs",
        ".claude/skills/speckit-plan",
        ".codex/claude-team-runs",
    ]:
        path = tmp_path / dirname / "README.md"
        path.parent.mkdir(parents=True)
        path.write_text("[broken](missing.md)\n", encoding="utf-8")
    tracked_agent_doc = tmp_path / ".claude" / "README.md"
    tracked_agent_doc.parent.mkdir(exist_ok=True)
    tracked_agent_doc.write_text("# Tracked agent docs\n", encoding="utf-8")

    scanned = {path.relative_to(tmp_path).as_posix() for path in iter_doc_files(tmp_path)}

    assert scanned == {"README.md", ".claude/README.md"}


def test_validate_docs_skips_initialized_git_submodules(tmp_path) -> None:
    (tmp_path / "README.md").write_text("# Root\n", encoding="utf-8")
    (tmp_path / ".gitmodules").write_text(
        '[submodule "provider"]\n'
        "\tpath = packages/provider\n"
        "\turl = https://example.invalid/provider.git\n",
        encoding="utf-8",
    )
    provider = tmp_path / "packages" / "provider"
    provider.mkdir(parents=True)
    (provider / "README.md").write_text("[broken](missing.md)\n", encoding="utf-8")
    (provider / "AGENTS.md").write_text("# External instructions\n", encoding="utf-8")

    scanned = {path.relative_to(tmp_path).as_posix() for path in iter_doc_files(tmp_path)}

    assert scanned == {"README.md"}
