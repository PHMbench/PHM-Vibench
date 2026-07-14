"""Documentation consistency checks for maintained repository documentation.

The validator is intentionally offline. It checks current user, contributor,
configuration, release, policy, template, and component documentation while
excluding explicitly historical, research, paper, and agent-workflow trees.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

import yaml


SKIP_TOP_DIRS = {
    ".git",
    ".archive",
    ".pytest_cache",
    ".agents",
    ".tmp",
    ".claude",
    ".codex",
    "__pycache__",
    "obsidian",
    "paper",
    "dev",
}

SKIP_PATH_PREFIXES = {
    ("docs", "v0.1.0"),
    ("docs", "past"),
    ("configs", "v0.0.9"),
    ("src", "configs", "plan"),
    ("src", "configs", "deprecated"),
    ("test", "todo_test"),
}

SKIP_DIR_NAMES = {"__pycache__"}

PLACEHOLDER_PATTERNS = {
    "<YOUR_REPO_URL>": "unresolved repository URL placeholder",
    "<YOUR_repo_URL>": "unresolved repository URL placeholder",
    "[INSERT SECURITY CONTACT EMAIL ADDRESS HERE]": "unresolved security contact placeholder",
    "[INSERT CONTACT METHOD]": "unresolved conduct contact placeholder",
    "python src/main.py --config-name": "obsolete runtime command",
}


@dataclass(frozen=True)
class Issue:
    kind: str
    path: str
    detail: str


def has_prefix(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(parts) >= len(prefix) and parts[: len(prefix)] == prefix


def is_skipped_path(rel: Path) -> bool:
    parts = rel.parts
    if parts and parts[0] in SKIP_TOP_DIRS:
        return True
    if any(has_prefix(parts, prefix) for prefix in SKIP_PATH_PREFIXES):
        return True
    return any(part in SKIP_DIR_NAMES for part in parts)


def iter_doc_files(repo_root: Path) -> Iterable[Path]:
    """Yield all maintained Markdown files in a deterministic order."""

    for path in sorted(repo_root.rglob("*.md")):
        rel = path.relative_to(repo_root)
        if not is_skipped_path(rel):
            yield path


def strip_fenced_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def normalize_link_destination(dest: str) -> str:
    dest = dest.strip()
    if dest.startswith("<") and dest.endswith(">"):
        dest = dest[1:-1].strip()
    dest = dest.split("#", 1)[0]
    dest = dest.split("?", 1)[0]
    return unquote(dest.strip())


def iter_link_destinations(text: str) -> Iterable[str]:
    # Inline links and images: [label](target) / ![alt](target)
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
        yield match.group(1)

    # Reference definitions: [name]: target
    for match in re.finditer(r"(?m)^\s*\[[^\]]+\]:\s*(\S+)", text):
        yield match.group(1)


def check_local_links(repo_root: Path, doc_files: Iterable[Path]) -> list[Issue]:
    issues: list[Issue] = []
    for path in doc_files:
        text = strip_fenced_code_blocks(
            path.read_text(encoding="utf-8", errors="ignore")
        )
        for raw_dest in iter_link_destinations(text):
            dest = normalize_link_destination(raw_dest)
            if not dest or dest.startswith("#"):
                continue
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", dest):
                continue
            if dest.startswith("mailto:") or dest.startswith("@"):
                continue

            if dest.startswith("/"):
                target = (repo_root / dest.lstrip("/")).resolve()
            else:
                target = (path.parent / dest).resolve()

            try:
                target.relative_to(repo_root.resolve())
            except ValueError:
                issues.append(
                    Issue(
                        kind="link_escapes_repository",
                        path=str(path.relative_to(repo_root)),
                        detail=f"{raw_dest} (resolved to {target})",
                    )
                )
                continue

            if not target.exists():
                issues.append(
                    Issue(
                        kind="missing_link_target",
                        path=str(path.relative_to(repo_root)),
                        detail=f"{raw_dest} (resolved to {target})",
                    )
                )
    return issues


def first_n_lines(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[:n])


def check_ai_docs_point_to_readme(repo_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    for doc_name in ["CLAUDE.md", "AGENTS.md", "GEMINI.md"]:
        for path in repo_root.rglob(doc_name):
            rel = path.relative_to(repo_root)
            if is_skipped_path(rel):
                continue
            readme = path.parent / "README.md"
            if not readme.exists():
                issues.append(
                    Issue(
                        kind="missing_readme_for_ai_doc",
                        path=str(rel),
                        detail="Expected sibling README.md",
                    )
                )
                continue
            head = first_n_lines(path, 40)
            if "@README" not in head and "README.md" not in head:
                issues.append(
                    Issue(
                        kind="ai_doc_missing_readme_pointer",
                        path=str(rel),
                        detail="Expected @README or README.md reference near the top",
                    )
                )
    return issues


def check_placeholders(repo_root: Path, doc_files: Iterable[Path]) -> list[Issue]:
    issues: list[Issue] = []
    for path in doc_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, detail in PLACEHOLDER_PATTERNS.items():
            if pattern in text:
                issues.append(
                    Issue(
                        kind="unresolved_placeholder",
                        path=str(path.relative_to(repo_root)),
                        detail=f"{detail}: {pattern}",
                    )
                )
    return issues


def check_nonempty_docs(repo_root: Path, doc_files: Iterable[Path]) -> list[Issue]:
    issues: list[Issue] = []
    for path in doc_files:
        if not path.read_text(encoding="utf-8", errors="ignore").strip():
            issues.append(
                Issue(
                    kind="empty_document",
                    path=str(path.relative_to(repo_root)),
                    detail="Maintained Markdown file is empty",
                )
            )
    return issues


def check_case_collisions(repo_root: Path, doc_files: Iterable[Path]) -> list[Issue]:
    by_casefold: dict[str, list[str]] = {}
    for path in doc_files:
        rel = path.relative_to(repo_root).as_posix()
        by_casefold.setdefault(rel.casefold(), []).append(rel)

    issues: list[Issue] = []
    for paths in sorted(by_casefold.values()):
        if len(paths) > 1:
            issues.append(
                Issue(
                    kind="case_colliding_documents",
                    path=paths[0],
                    detail=", ".join(paths),
                )
            )
    return issues


def parse_front_matter(path: Path) -> dict[str, object] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return None
    data = yaml.safe_load("\n".join(lines[1:end]))
    return data if isinstance(data, dict) else None


def check_issue_templates(repo_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    template_dir = repo_root / ".github" / "ISSUE_TEMPLATE"
    if not template_dir.exists():
        return issues

    for path in sorted(template_dir.glob("*.md")):
        front_matter = parse_front_matter(path)
        rel = str(path.relative_to(repo_root))
        if front_matter is None:
            issues.append(
                Issue(
                    kind="invalid_issue_template_front_matter",
                    path=rel,
                    detail="Expected YAML front matter delimited by ---",
                )
            )
            continue
        for key in ["name", "about", "title"]:
            value = front_matter.get(key)
            if not isinstance(value, str) or not value.strip():
                issues.append(
                    Issue(
                        kind="invalid_issue_template_front_matter",
                        path=rel,
                        detail=f"Missing non-empty string field: {key}",
                    )
                )
    return issues


def check_citation_metadata(repo_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    canonical = repo_root / "CITATION.cff"
    legacy = repo_root / "citation.cff"

    if legacy.exists():
        issues.append(
            Issue(
                kind="noncanonical_citation_filename",
                path="citation.cff",
                detail="Use uppercase CITATION.cff only",
            )
        )

    if not canonical.exists():
        issues.append(
            Issue(
                kind="missing_citation_metadata",
                path="CITATION.cff",
                detail="Expected canonical citation metadata",
            )
        )
        return issues

    try:
        data = yaml.safe_load(canonical.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        issues.append(
            Issue(
                kind="invalid_citation_metadata",
                path="CITATION.cff",
                detail=str(exc),
            )
        )
        return issues

    if not isinstance(data, dict):
        issues.append(
            Issue(
                kind="invalid_citation_metadata",
                path="CITATION.cff",
                detail="Expected a YAML mapping",
            )
        )
        return issues

    for key in ["cff-version", "message", "title", "type", "authors", "repository-code", "license"]:
        if key not in data or data[key] in (None, "", []):
            issues.append(
                Issue(
                    kind="invalid_citation_metadata",
                    path="CITATION.cff",
                    detail=f"Missing required field: {key}",
                )
            )
    return issues


def check_canonical_policy_paths(repo_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    required = [
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "CITATION.cff",
        "LICENSE",
    ]
    for rel in required:
        if not (repo_root / rel).exists():
            issues.append(
                Issue(
                    kind="missing_canonical_policy_file",
                    path=rel,
                    detail="Required repository entry is missing",
                )
            )

    for legacy in ["contributing.md", "citation.cff", "Codex_agent.md"]:
        if (repo_root / legacy).exists():
            issues.append(
                Issue(
                    kind="legacy_or_empty_root_document",
                    path=legacy,
                    detail="Remove or replace with the canonical repository entry",
                )
            )
    return issues


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    doc_files = list(iter_doc_files(repo_root))

    issues: list[Issue] = []
    issues.extend(check_ai_docs_point_to_readme(repo_root))
    issues.extend(check_local_links(repo_root, doc_files))
    issues.extend(check_placeholders(repo_root, doc_files))
    issues.extend(check_nonempty_docs(repo_root, doc_files))
    issues.extend(check_case_collisions(repo_root, doc_files))
    issues.extend(check_issue_templates(repo_root))
    issues.extend(check_citation_metadata(repo_root))
    issues.extend(check_canonical_policy_paths(repo_root))

    if issues:
        print("[FAIL] Documentation checks failed:")
        for issue in issues:
            print(f"- {issue.kind}: {issue.path}: {issue.detail}")
        return 1

    print(f"[OK] Documentation checks passed ({len(doc_files)} maintained Markdown files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
