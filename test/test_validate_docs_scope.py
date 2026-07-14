from __future__ import annotations

from pathlib import Path

from scripts.validate_docs import (
    check_case_collisions,
    check_citation_metadata,
    check_issue_templates,
    check_local_links,
    iter_doc_files,
)


def write(path: Path, content: str = "# Doc\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_docs_scans_maintained_markdown_and_skips_archives(tmp_path) -> None:
    write(tmp_path / "README.md")
    write(tmp_path / "docs" / "guide.md")
    write(tmp_path / ".github" / "ISSUE_TEMPLATE" / "bug.md")
    write(tmp_path / "src" / "package" / "README.md")

    for path in [
        tmp_path / ".agents" / "README.md",
        tmp_path / ".tmp" / "README.md",
        tmp_path / ".claude" / "README.md",
        tmp_path / ".codex" / "README.md",
        tmp_path / "obsidian" / "README.md",
        tmp_path / "paper" / "README.md",
        tmp_path / "dev" / "README.md",
        tmp_path / "docs" / "v0.1.0" / "README.md",
        tmp_path / "docs" / "past" / "README.md",
        tmp_path / "configs" / "v0.0.9" / "README.md",
        tmp_path / "src" / "configs" / "plan" / "README.md",
        tmp_path / "src" / "configs" / "deprecated" / "README.md",
        tmp_path / "test" / "todo_test" / "README.md",
    ]:
        write(path, "[broken](missing.md)\n")

    scanned = {path.relative_to(tmp_path).as_posix() for path in iter_doc_files(tmp_path)}

    assert scanned == {
        ".github/ISSUE_TEMPLATE/bug.md",
        "README.md",
        "docs/guide.md",
        "src/package/README.md",
    }


def test_validate_docs_catches_broken_link_in_ordinary_doc(tmp_path) -> None:
    guide = tmp_path / "docs" / "guide.md"
    write(guide, "[missing](not-there.md)\n")

    issues = check_local_links(tmp_path, [guide])

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("missing_link_target", "docs/guide.md")
    ]


def test_validate_docs_accepts_root_relative_and_encoded_local_links(tmp_path) -> None:
    guide = tmp_path / "docs" / "guide.md"
    write(tmp_path / "README.md")
    write(tmp_path / "docs" / "a file.md")
    write(guide, "[root](/README.md) [space](a%20file.md#section)\n")

    assert check_local_links(tmp_path, [guide]) == []


def test_validate_docs_detects_case_colliding_documents(tmp_path) -> None:
    upper = tmp_path / "docs" / "README.md"
    lower = tmp_path / "docs" / "readme.md"
    write(upper)
    write(lower)

    issues = check_case_collisions(tmp_path, [upper, lower])

    assert len(issues) == 1
    assert issues[0].kind == "case_colliding_documents"


def test_issue_template_requires_valid_front_matter(tmp_path) -> None:
    valid = tmp_path / ".github" / "ISSUE_TEMPLATE" / "valid.md"
    invalid = tmp_path / ".github" / "ISSUE_TEMPLATE" / "invalid.md"
    write(
        valid,
        "---\nname: Bug\nabout: Report a bug\ntitle: '[BUG] '\n---\n\nBody\n",
    )
    write(invalid, "# No front matter\n")

    issues = check_issue_templates(tmp_path)

    assert len(issues) == 1
    assert issues[0].path.endswith("invalid.md")


def test_citation_metadata_requires_canonical_nonempty_fields(tmp_path) -> None:
    write(
        tmp_path / "CITATION.cff",
        """cff-version: 1.2.0
message: Cite the exact commit.
title: PHM-Vibench
type: software
authors:
  - name: PHMbench contributors
repository-code: https://github.com/PHMbench/PHM-Vibench
license: Apache-2.0
""",
    )

    assert check_citation_metadata(tmp_path) == []

    write(tmp_path / "citation.cff", "")
    issues = check_citation_metadata(tmp_path)
    assert any(issue.kind == "noncanonical_citation_filename" for issue in issues)
