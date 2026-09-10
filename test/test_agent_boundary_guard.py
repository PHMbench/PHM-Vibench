from __future__ import annotations

import importlib.util
from pathlib import Path

from scripts.validate_docs import check_ai_docs_point_to_readme


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "repo" / "check_agent_boundaries.py"
SPEC = importlib.util.spec_from_file_location("check_agent_boundaries", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _violations(*paths: str) -> set[tuple[str, str]]:
    return set(MODULE._violations(tuple(paths)))


def test_allows_only_exact_shared_root_names() -> None:
    assert not _violations("AGENTS.md", "CLAUDE.md")
    paths = ("agent.md", "agents.md", "claude.MD", "agents_cn.MD", "Codex_agent.md")
    assert _violations(*paths) == {
        ("duplicate or private Agent document", path) for path in paths
    }


def test_rejects_top_level_agent_workspaces() -> None:
    observed = _violations(
        ".claude/commands/run.md",
        ".codex/skills/example/SKILL.md",
        ".agents/config.yaml",
        ".gemini/prompt.md",
    )
    assert len(observed) == 4
    assert all(category == "top-level Agent workspace" for category, _ in observed)


def test_allows_neutral_public_documentation() -> None:
    assert not _violations(
        "docs/developer_guide.md",
        "docs/archive/audits/agent-migration.md",
        "tools/repo/check_agent_boundaries.py",
    )


def test_rejects_nested_instructions_and_local_overrides() -> None:
    paths = (
        "src/data_factory/CLAUDE.md",
        "src/model_factory/Transformer/AGENTS.md",
        "configs/demo/CLAUDE.md",
        "AGENTS.override.md",
        "CLAUDE.local.md",
    )
    assert _violations(*paths) == {
        ("duplicate or private Agent document", path) for path in paths
    }


def test_shared_root_import_passes_documentation_check(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Read README.md.\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
    assert not check_ai_docs_point_to_readme(tmp_path)


def test_shared_root_import_requires_its_target(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
    issues = check_ai_docs_point_to_readme(tmp_path)
    assert [issue.kind for issue in issues] == ["missing_shared_agent_document"]


def test_root_claude_cannot_add_a_second_instruction_body(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Read README.md.\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\nUse a different policy.\n", encoding="utf-8")
    issues = check_ai_docs_point_to_readme(tmp_path)
    assert [issue.kind for issue in issues] == ["invalid_shared_agent_import"]
