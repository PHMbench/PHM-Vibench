#!/usr/bin/env python3
"""Generate the immutable PHMFactory v0.3 baseline inventories.

This tool is intentionally read-only with respect to runtime code. It reads an
exported snapshot of the frozen main commit and writes audit artifacts under
``docs/archive/audits`` plus the deny-by-default submodule allowlist.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile
from typing import Iterable, Sequence

BASELINE_COMMIT = "a331769d4005018bc833534ecf4efeb5e8a5a78d"
CONTRACT_COMMIT = "d044d2031165cd4186d1da462fb154f101d6d493"
REPOSITORY = "PHMbench/PHM-Vibench"

PROTECTED_PREFIXES = (
    "src/data_factory/reader/",
    "src/data_factory/dataset_task/",
    "src/data_factory/samplers/",
    "src/model_factory/",
    "src/task_factory/",
    "src/trainer_factory/",
)
PROTECTED_EXACT = {
    "src/data_factory/H5DataDict.py",
    "src/data_factory/data_factory.py",
}

BOUNDARY_PREFIXES = (
    ("agent", ".claude/"),
    ("agent", ".codex/"),
    ("development", "dev/"),
    ("historical", ".archive/"),
    ("paper", "paper/"),
    ("result", "results/"),
    ("result", "metrics_reports/"),
    ("report", "reports/"),
    ("plot", "plot/"),
    ("reader_output", "src/data_factory/reader/output/"),
    ("historical", "docs/past/"),
    ("historical", "docs/v0.1.0/"),
    ("historical_config", "configs/v0.0.9/"),
)

ROOT_AGENT_FILES = {
    "AGENTS.md",
    "AGENTS_CN.md",
    "CLAUDE.md",
    "CLAUDE_CN.md",
    "GEMINI.md",
    "Codex_agent.md",
}

PERSONAL_PATTERNS = {
    "linux_home": re.compile(r"/home/[A-Za-z0-9._-]+"),
    "macos_home": re.compile(r"/Users/[A-Za-z0-9._-]+"),
    "windows_home": re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+"),
    "personal_github_ssh": re.compile(r"git@github\.com:"),
    "personal_account": re.compile(r"(?<![A-Za-z0-9_])liq22(?![A-Za-z0-9_])", re.I),
    "personal_environment": re.compile(r"(?<![A-Za-z0-9_])LQ_signal(?![A-Za-z0-9_])"),
    "personal_prefix": re.compile(r"(?<![A-Za-z0-9_])LQ[_-][A-Za-z0-9_-]+"),
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".ps1",
    ".bat",
    ".cff",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def baseline_paths() -> list[str]:
    output = git("ls-tree", "-r", "--name-only", BASELINE_COMMIT)
    return [line for line in output.splitlines() if line]


def is_pipeline(path: str) -> bool:
    name = PurePosixPath(path).name
    return path.startswith("src/") and name.startswith("Pipeline_") and name.endswith(".py")


def is_protected_python(path: str) -> bool:
    if not path.endswith(".py"):
        return False
    return (
        path in PROTECTED_EXACT
        or any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)
        or is_pipeline(path)
    )


def safe_read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > 5_000_000:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def xlsx_search_text(path: Path) -> str:
    try:
        if not zipfile.is_zipfile(path):
            return ""
        chunks: list[str] = []
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.startswith("xl/") and name.endswith(".xml"):
                    chunks.append(archive.read(name).decode("utf-8", errors="replace"))
        return "\n".join(chunks)
    except (OSError, zipfile.BadZipFile, KeyError):
        return ""


def callable_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts: list[str] = []
    posonly_count = len(args.posonlyargs)
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        item = argument.arg
        if argument.annotation is not None:
            item += f": {ast.unparse(argument.annotation)}"
        if default is not None:
            item += f"={ast.unparse(default)}"
        parts.append(item)
        if posonly_count and index + 1 == posonly_count:
            parts.append("/")
    if args.vararg is not None:
        item = f"*{args.vararg.arg}"
        if args.vararg.annotation is not None:
            item += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(item)
    elif args.kwonlyargs:
        parts.append("*")
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        item = argument.arg
        if argument.annotation is not None:
            item += f": {ast.unparse(argument.annotation)}"
        if default is not None:
            item += f"={ast.unparse(default)}"
        parts.append(item)
    if args.kwarg is not None:
        item = f"**{args.kwarg.arg}"
        if args.kwarg.annotation is not None:
            item += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(item)
    return_annotation = ""
    if node.returns is not None:
        return_annotation = f" -> {ast.unparse(node.returns)}"
    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{async_prefix}{node.name}({', '.join(parts)}){return_annotation}"


def callable_records(source: str) -> tuple[list[dict[str, object]], str | None]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], f"{exc.msg} at line {exc.lineno}"
    records: list[dict[str, object]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            dumped = ast.dump(node, annotate_fields=True, include_attributes=False)
            record: dict[str, object] = {
                "name": node.name,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "line": node.lineno,
                "ast_sha256": sha256_bytes(dumped.encode("utf-8")),
            }
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                record["signature"] = callable_signature(node)
            records.append(record)
    return records, None


def exact_word_occurs(text: str, token: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text) is not None


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_submodules(snapshot: Path) -> list[dict[str, str]]:
    modules_path = snapshot / ".gitmodules"
    if not modules_path.exists():
        return []
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(modules_path, encoding="utf-8")
    rows: list[dict[str, str]] = []
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        path = parser.get(section, "path")
        tree_line = git("ls-tree", BASELINE_COMMIT, "--", path)
        fields = tree_line.split()
        gitlink = fields[2] if len(fields) >= 3 else ""
        if path in {"data/Rotor_simulation", "paper/LQ_vibench_fix"}:
            classification = "personal"
        elif path.startswith("paper/"):
            classification = "paper"
        else:
            classification = "unclassified"
        rows.append(
            {
                "name": section.removeprefix('submodule "').removesuffix('"'),
                "path": path,
                "url": parser.get(section, "url", fallback=""),
                "branch": parser.get(section, "branch", fallback=""),
                "gitlink_sha": gitlink,
                "classification": classification,
                "allowlisted": "false",
                "v0_3_action": "migrate_then_remove",
            }
        )
    return sorted(rows, key=lambda item: item["path"])


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()

    snapshot = args.snapshot_root.resolve()
    output_root = args.output_root.resolve()
    if not snapshot.is_dir():
        raise SystemExit(f"Snapshot root does not exist: {snapshot}")
    if git("rev-parse", "HEAD") == BASELINE_COMMIT:
        contract_head = BASELINE_COMMIT
    else:
        contract_head = git("rev-parse", "HEAD")

    paths = baseline_paths()
    protected_paths = sorted(path for path in paths if is_protected_python(path))
    current_protected = sorted(
        path for path in git("ls-files").splitlines() if is_protected_python(path)
    )
    if protected_paths != current_protected:
        missing = sorted(set(protected_paths) - set(current_protected))
        added = sorted(set(current_protected) - set(protected_paths))
        raise SystemExit(f"Protected path-set drift. missing={missing}, added={added}")

    fingerprints: list[dict[str, object]] = []
    parse_errors: list[tuple[str, str]] = []
    for relative in protected_paths:
        baseline_file = snapshot / relative
        current_file = output_root / relative
        baseline_data = baseline_file.read_bytes()
        current_data = current_file.read_bytes()
        if baseline_data != current_data:
            raise SystemExit(f"Protected runtime changed before PR-02 baseline: {relative}")
        source = baseline_data.decode("utf-8", errors="replace")
        callables, parse_error = callable_records(source)
        if parse_error:
            parse_errors.append((relative, parse_error))
        fingerprints.append(
            {
                "path": relative,
                "file_sha256": sha256_bytes(baseline_data),
                "bytes": len(baseline_data),
                "callables": callables,
                "parse_error": parse_error,
            }
        )

    searchable_text: dict[str, str] = {}
    metadata_text: dict[str, str] = {}
    python_text: dict[str, str] = {}
    for relative in paths:
        candidate = snapshot / relative
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        text: str | None = None
        if suffix in TEXT_SUFFIXES:
            text = safe_read_text(candidate)
        elif suffix == ".xlsx":
            text = xlsx_search_text(candidate)
        if text is None:
            continue
        searchable_text[relative] = text
        if suffix == ".py":
            python_text[relative] = text
        if suffix in {".csv", ".tsv", ".yaml", ".yml", ".xlsx", ".xls"} or relative.startswith("data/"):
            metadata_text[relative] = text

    reader_rows: list[dict[str, object]] = []
    reader_paths = sorted(
        path
        for path in paths
        if path.startswith("src/data_factory/reader/")
        and path.count("/") == 3
        and path.endswith(".py")
        and PurePosixPath(path).name not in {"__init__.py", "utils.py"}
    )
    for relative in reader_paths:
        module = PurePosixPath(relative).stem
        data = (snapshot / relative).read_bytes()
        source = data.decode("utf-8", errors="replace")
        callables, parse_error = callable_records(source)
        read_record = next(
            (item for item in callables if item["name"] == "read" and item["kind"] == "function"),
            None,
        )
        metadata_refs = sorted(
            path for path, text in metadata_text.items() if exact_word_occurs(text, module)
        )
        direct_python_consumers = sorted(
            path
            for path, text in python_text.items()
            if path != relative and exact_word_occurs(text, module)
        )
        executable_nodes = callables
        if not source.strip() or (not executable_nodes and not source.replace("#", "").strip()):
            status = "placeholder"
        elif read_record is not None and not module.startswith(("RM_", "Dummy_")):
            status = "unverified"
        elif read_record is not None and (metadata_refs or module == "Dummy_Data"):
            status = "maintained"
        elif read_record is not None:
            status = "unverified"
        else:
            status = "experimental"
        notes: list[str] = []
        if parse_error:
            notes.append(f"parse_error={parse_error}")
        if read_record is None:
            notes.append("no top-level read callable")
        if read_record is not None and not module.startswith(("RM_", "Dummy_")):
            notes.append("legacy non-RM module name and interface require implementation-aware review")
        if status in {"unverified", "placeholder", "experimental"}:
            notes.append("presence is not a maintained-support claim")
        reader_rows.append(
            {
                "module": module,
                "path": relative,
                "status": status,
                "file_sha256": sha256_bytes(data),
                "read_signature": "" if read_record is None else read_record.get("signature", ""),
                "read_ast_sha256": "" if read_record is None else read_record["ast_sha256"],
                "metadata_reference_count": len(metadata_refs),
                "metadata_reference_paths": ";".join(metadata_refs),
                "direct_python_consumer_count": len(direct_python_consumers),
                "direct_python_consumer_paths": ";".join(direct_python_consumers),
                "generic_runtime_consumer": "src/data_factory/data_factory.py",
                "notes": "; ".join(notes),
            }
        )

    personal_rows: list[dict[str, object]] = []
    personal_counts = {name: 0 for name in PERSONAL_PATTERNS}
    for relative, text in searchable_text.items():
        for line_number, line in enumerate(text.splitlines(), start=1):
            for category, pattern in PERSONAL_PATTERNS.items():
                if pattern.search(line):
                    personal_counts[category] += 1
                    personal_rows.append(
                        {
                            "category": category,
                            "path": relative,
                            "line": line_number,
                            "action": "review_then_remove_or_neutralize",
                        }
                    )

    boundary_rows: list[dict[str, object]] = []
    for category, prefix in BOUNDARY_PREFIXES:
        matching = [path for path in paths if path.startswith(prefix)]
        boundary_rows.append(
            {
                "category": category,
                "path": prefix.rstrip("/"),
                "tracked_file_count": len(matching),
                "v0_3_action": "inventory_then_migrate_or_remove",
                "notes": "do not delete before destination/provenance verification",
            }
        )
    for root_file in sorted(ROOT_AGENT_FILES):
        boundary_rows.append(
            {
                "category": "agent",
                "path": root_file,
                "tracked_file_count": int(root_file in paths),
                "v0_3_action": "archive_then_remove",
                "notes": "vendor/personal workflow document",
            }
        )
    pycache_paths = [path for path in paths if "__pycache__/" in path or path.endswith((".pyc", ".pyo"))]
    boundary_rows.append(
        {
            "category": "generated",
            "path": "**/__pycache__/** and *.py[co]",
            "tracked_file_count": len(pycache_paths),
            "v0_3_action": "remove_and_ignore",
            "notes": "generated bytecode",
        }
    )

    submodule_rows = parse_submodules(snapshot)

    audit_dir = output_root / "docs/archive/audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        audit_dir / "phmfactory-v0.3-reader-inventory.csv",
        (
            "module",
            "path",
            "status",
            "file_sha256",
            "read_signature",
            "read_ast_sha256",
            "metadata_reference_count",
            "metadata_reference_paths",
            "direct_python_consumer_count",
            "direct_python_consumer_paths",
            "generic_runtime_consumer",
            "notes",
        ),
        reader_rows,
    )
    write_csv(
        audit_dir / "phmfactory-v0.3-personal-path-inventory.csv",
        ("category", "path", "line", "action"),
        personal_rows,
    )
    write_csv(
        audit_dir / "phmfactory-v0.3-boundary-inventory.csv",
        ("category", "path", "tracked_file_count", "v0_3_action", "notes"),
        boundary_rows,
    )
    write_csv(
        audit_dir / "phmfactory-v0.3-submodule-baseline.csv",
        (
            "name",
            "path",
            "url",
            "branch",
            "gitlink_sha",
            "classification",
            "allowlisted",
            "v0_3_action",
        ),
        submodule_rows,
    )

    fingerprint_rows: list[dict[str, object]] = []
    for item in fingerprints:
        callable_payload = json.dumps(
            item["callables"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        fingerprint_rows.append(
            {
                "path": item["path"],
                "file_sha256": item["file_sha256"],
                "bytes": item["bytes"],
                "callable_count": len(item["callables"]),
                "callable_ast_sha256": sha256_bytes(callable_payload),
                "parse_error": item["parse_error"] or "",
            }
        )
    write_csv(
        audit_dir / "phmfactory-v0.3-protected-runtime-fingerprints.csv",
        (
            "path",
            "file_sha256",
            "bytes",
            "callable_count",
            "callable_ast_sha256",
            "parse_error",
        ),
        fingerprint_rows,
    )

    status_counts: dict[str, int] = {}
    for row in reader_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    tags = git("tag", "--list", "v0.2*").splitlines()
    tags_text = ", ".join(tags) if tags else "none found"
    summary_lines = [
        "# PHMFactory v0.3 Runtime, Reader, and Repository Baseline",
        "",
        "## Immutable basis",
        "",
        f"- Repository: `{REPOSITORY}`",
        f"- Frozen main/runtime commit: `{BASELINE_COMMIT}`",
        f"- Repository-contract commit: `{CONTRACT_COMMIT}`",
        f"- v0.2.x tags visible to the generator: `{tags_text}`",
        "",
        "This PR records evidence only. It does not modify a reader, runtime callable,",
        "submodule, paper workspace, result, package name, Pipeline, config, or test.",
        "",
        "## Protected runtime",
        "",
        f"- Protected Python files fingerprinted: **{len(fingerprints)}**",
        f"- Python parse errors recorded: **{len(parse_errors)}**",
        "- Every protected file was byte-compared with the frozen runtime commit before",
        "  the inventories were emitted.",
        "- Callable fingerprints use `ast.dump(..., include_attributes=False)` and SHA-256.",
        "",
        "Artifacts:",
        "",
        "- `phmfactory-v0.3-protected-runtime-fingerprints.csv`",
        "- `phmfactory-v0.3-reader-inventory.csv`",
        "",
        "## Reader classification",
        "",
        "| Status | Count | Meaning |",
        "| --- | ---: | --- |",
    ]
    meanings = {
        "maintained": "Top-level `read` callable and an active metadata/config reference, or the offline Dummy reader.",
        "compatibility": "Callable exists under a legacy/non-RM module name; retained without a new support claim.",
        "experimental": "Non-empty reader-area module without the standard top-level `read` callable.",
        "unverified": "Top-level `read` callable exists, but no active metadata/config reference was found.",
        "placeholder": "Empty or effectively non-executable placeholder.",
    }
    for status in ("maintained", "compatibility", "experimental", "unverified", "placeholder"):
        summary_lines.append(f"| `{status}` | {status_counts.get(status, 0)} | {meanings[status]} |")
    summary_lines.extend(
        [
            "",
            "Classification is an audit result, not a promise to delete non-maintained",
            "files. `THU.py`, `THU24.py`, and similar compatibility/placeholder files remain",
            "protected until a separate implementation-aware decision.",
            "",
            "## Submodule baseline",
            "",
            f"- Configured submodules: **{len(submodule_rows)}**",
            f"- Allowlisted baseline entries: **{sum(row['allowlisted'] == 'true' for row in submodule_rows)}**",
            "- The frozen baseline does not contain the proposed `phm-data-factory` backend.",
            "- The deny-by-default allowlist records that backend as the sole candidate and",
            "  records every existing baseline entry as legacy/non-allowlisted.",
            "",
            "Artifacts:",
            "",
            "- `phmfactory-v0.3-submodule-baseline.csv`",
            "- `.github/phmfactory-v0.3-submodules.allowlist.yml`",
            "",
            "## Personal and ownership-boundary inventory",
            "",
            "The scanner records path, line, and category without copying line contents into",
            "the public report. This avoids turning an inventory into a second source of",
            "personal configuration values.",
            "",
            "| Category | Matches |",
            "| --- | ---: |",
        ]
    )
    for category in sorted(personal_counts):
        summary_lines.append(f"| `{category}` | {personal_counts[category]} |")
    summary_lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `phmfactory-v0.3-personal-path-inventory.csv`",
            "- `phmfactory-v0.3-boundary-inventory.csv`",
            "",
            "## Interpretation and next actions",
            "",
            "1. PR-03 may remove generated and Agent/personal-only paths only after private-fork",
            "   preservation and reference checks.",
            "2. Reader cleanup PRs must compare against the protected callable fingerprints.",
            "3. Paper/result/submodule deletion requires destination, immutable source SHA,",
            "   content/hash verification, and reviewer confirmation.",
            "4. The proposed backend remains optional and must not make the core wheel, CLI,",
            "   Dummy smoke, or CWRU quickstart depend on an initialized submodule.",
            "5. This baseline does not authorize algorithm changes or broad formatting.",
            "",
            "## Regeneration",
            "",
            "The generated artifacts are deterministic for the frozen snapshot:",
            "",
            "```bash",
            f"mkdir -p /tmp/phmfactory-v030-baseline",
            f"git archive {BASELINE_COMMIT} | tar -x -C /tmp/phmfactory-v030-baseline",
            "python tools/repo/v030_generate_baseline.py \\",
            "  --snapshot-root /tmp/phmfactory-v030-baseline \\",
            "  --output-root .",
            "```",
        ]
    )
    (audit_dir / "phmfactory-v0.3-runtime-reader-baseline.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "reader_count": len(reader_rows),
                "reader_statuses": status_counts,
                "protected_python_files": len(fingerprints),
                "protected_parse_errors": len(parse_errors),
                "submodules": len(submodule_rows),
                "personal_matches": sum(personal_counts.values()),
                "boundary_rows": len(boundary_rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
