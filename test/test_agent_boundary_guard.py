from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "repo" / "check_agent_boundaries.py"
SPEC = importlib.util.spec_from_file_location("check_agent_boundaries", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _violations(*paths: str) -> set[tuple[str, str]]:
    return set(MODULE._violations(tuple(paths)))


def test_rejects_root_agent_documents_case_insensitively() -> None:
    assert ("root Agent document", "CLAUDE.md") in _violations("CLAUDE.md")
    assert ("root Agent document", "agents_cn.MD") in _violations("agents_cn.MD")
    assert ("root Agent document", "Codex_agent.md") in _violations("Codex_agent.md")


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


def test_temporarily_allows_module_level_claude_documents() -> None:
    """Module knowledge is migrated through implementation-aware follow-up PRs."""

    assert not _violations(
        "src/data_factory/CLAUDE.md",
        "src/model_factory/Transformer/CLAUDE.md",
        "configs/demo/CLAUDE.md",
    )
