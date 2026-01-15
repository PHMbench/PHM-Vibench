"""Documentation consistency checks (local, no network).

This module is intentionally lightweight and conservative: it validates that documentation
links resolve and that per-directory AI docs defer shared content to README.md.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SKIP_TOP_DIRS = {
    ".git",
    ".archive",
    ".pytest_cache",
    "__pycache__",
    "paper",  # paper workflows are not part of the core validation gate
}

SKIP_DIR_NAMES = {"__pycache__"}


@dataclass(frozen=True)
class Issue:
    kind: str
    path: str
    detail: str


def iter_doc_files(repo_root: Path) -> Iterable[Path]:
    patterns = [
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        "GEMINI.md",
        "API_REFERENCE.md",
    ]
    for pattern in patterns:
        for path in repo_root.rglob(pattern):
            rel = path.relative_to(repo_root)
            if rel.parts and rel.parts[0] in SKIP_TOP_DIRS:
                continue
            if any(part in SKIP_DIR_NAMES for part in rel.parts):
                continue
            yield path


def strip_fenced_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def check_local_links(repo_root: Path, doc_files: Iterable[Path]) -> list[Issue]:
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    issues: list[Issue] = []
    for path in doc_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = strip_fenced_code_blocks(text)
        for match in link_re.finditer(text):
            dest = match.group(1).strip()
            if not dest or dest.startswith("#"):
                continue
            if re.match(r"^[a-zA-Z]+://", dest) or dest.startswith("mailto:"):
                continue
            dest = dest.split("#", 1)[0]
            if dest.startswith("@"):
                continue
            target = (path.parent / dest).resolve()
            if not target.exists():
                issues.append(
                    Issue(
                        kind="missing_link_target",
                        path=str(path.relative_to(repo_root)),
                        detail=f"{dest} (resolved to {target})",
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
            if rel.parts and rel.parts[0] in SKIP_TOP_DIRS:
                continue
            if any(part in SKIP_DIR_NAMES for part in rel.parts):
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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    doc_files = list(iter_doc_files(repo_root))

    issues: list[Issue] = []
    issues.extend(check_ai_docs_point_to_readme(repo_root))
    issues.extend(check_local_links(repo_root, doc_files))

    if issues:
        print("[FAIL] Documentation checks failed:")
        for issue in issues:
            print(f"- {issue.kind}: {issue.path}: {issue.detail}")
        return 1

    print(f"[OK] Documentation checks passed ({len(doc_files)} files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

