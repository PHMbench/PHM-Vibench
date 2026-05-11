from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


UXFD_ROOT = Path("paper/UXFD_paper")
ROOT_COMMAND_RE = re.compile(r"python\s+main\.py\s+--config\s+([^\s\\]+)")


@dataclass(frozen=True)
class UXFDContract:
    submodule_path: Path
    vibench_path: Path
    min_config_path: Path
    maintained_command: str
    expected_artifacts: Tuple[str, ...]
    status: str
    reason: str


@dataclass(frozen=True)
class LatexEntrypoint:
    submodule_path: Path
    tex_path: Path
    status: str
    reason: str = ""


@dataclass(frozen=True)
class ClaimEvidence:
    claim_id: str
    submodule_path: Path
    tex_path: Path
    claim_type: str
    status: str
    reason: str
    artifact_path: str = ""


@dataclass(frozen=True)
class CompileGate:
    tex_path: Path
    command: str
    result: str
    pdf_path: str
    log_path: str
    first_error: str


@dataclass(frozen=True)
class SubmoduleState:
    submodule_path: Path
    submodule_status: str
    parent_gitlink_status: str
    reason: str
    commit_sha: str = ""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def indexed_uxfd_submodules(readme_path: Path = UXFD_ROOT / "README.md") -> Tuple[Path, ...]:
    text = _read_text(readme_path)
    gitmodule_paths = set(gitmodule_uxfd_submodules())
    paths = []
    for match in re.finditer(r"`(paper/UXFD_paper/[^`]+)`", text):
        path = Path(match.group(1))
        if path in gitmodule_paths and path not in paths:
            paths.append(path)
    return tuple(paths)


def gitmodule_uxfd_submodules(gitmodules_path: Path = Path(".gitmodules")) -> Tuple[Path, ...]:
    paths = []
    for line in _read_text(gitmodules_path).splitlines():
        stripped = line.strip()
        if not stripped.startswith("path = "):
            continue
        path = Path(stripped.removeprefix("path = ").strip())
        if path.parent == UXFD_ROOT:
            paths.append(path)
    return tuple(paths)


def _extract_root_command(vibench_path: Path) -> str:
    if not vibench_path.exists():
        return ""
    text = _read_text(vibench_path)
    match = ROOT_COMMAND_RE.search(text)
    if not match:
        return ""
    config_path = match.group(1)
    return f"python main.py --config {config_path} --override trainer.num_epochs=1"


def _extract_expected_artifacts(vibench_path: Path) -> Tuple[str, ...]:
    if not vibench_path.exists():
        return ()
    text = _read_text(vibench_path)
    artifacts = []
    for item in [
        "config_snapshot.yaml",
        "test_result_*.csv",
        "test_result.csv",
        "artifacts/manifest.json",
        "artifacts/data_metadata_snapshot.json",
        "artifacts/predictions.npz",
        "artifacts/distilled/summary.json",
    ]:
        if item in text:
            artifacts.append(item)
    return tuple(artifacts)


def audit_contracts(paths: Optional[Iterable[Path]] = None) -> Tuple[UXFDContract, ...]:
    contracts: List[UXFDContract] = []
    for submodule in paths or indexed_uxfd_submodules():
        vibench = submodule / "VIBENCH.md"
        min_config = submodule / "configs" / "vibench" / "min.yaml"
        command = _extract_root_command(vibench)
        artifacts = _extract_expected_artifacts(vibench)
        reason = ""
        status = "unverified"
        if not vibench.exists():
            status = "blocked"
            reason = "missing VIBENCH.md"
        elif not min_config.exists():
            status = "blocked"
            reason = "missing configs/vibench/min.yaml"
        elif not command:
            status = "paper-local-only"
            reason = "no maintained root CLI command found"
        elif "artifacts/manifest.json" not in artifacts:
            status = "unverified"
            reason = "VIBENCH.md lacks full Slice 1 artifact expectations"
        contracts.append(
            UXFDContract(
                submodule_path=submodule,
                vibench_path=vibench,
                min_config_path=min_config,
                maintained_command=command,
                expected_artifacts=artifacts,
                status=status,
                reason=reason,
            )
        )
    return tuple(contracts)


def discover_latex_entrypoints(paths: Optional[Iterable[Path]] = None) -> Tuple[LatexEntrypoint, ...]:
    records: List[LatexEntrypoint] = []
    for submodule in paths or indexed_uxfd_submodules():
        final_main = submodule / "manuscript" / "final_tex" / "main.tex"
        if final_main.exists():
            records.append(LatexEntrypoint(submodule, final_main, "selected"))
            continue
        candidates = sorted((submodule / "manuscript").rglob("*.tex")) if (submodule / "manuscript").exists() else []
        candidates.extend(sorted((submodule / "paper_draft").rglob("*.tex")) if (submodule / "paper_draft").exists() else [])
        if candidates:
            records.append(LatexEntrypoint(submodule, candidates[0], "non-final", "no manuscript/final_tex/main.tex"))
        else:
            records.append(LatexEntrypoint(submodule, final_main, "missing", "no TeX entrypoint discovered"))
    return tuple(records)


def map_claim_evidence(entrypoints: Optional[Iterable[LatexEntrypoint]] = None) -> Tuple[ClaimEvidence, ...]:
    records = []
    for entrypoint in entrypoints or discover_latex_entrypoints():
        if entrypoint.status != "selected":
            records.append(
                ClaimEvidence(
                    claim_id=f"{entrypoint.submodule_path.name}:entrypoint",
                    submodule_path=entrypoint.submodule_path,
                    tex_path=entrypoint.tex_path,
                    claim_type="text",
                    status="blocked",
                    reason=entrypoint.reason or "no selected final entrypoint",
                )
            )
            continue
        text = _read_text(entrypoint.tex_path)
        has_claim_surface = any(token in text for token in ["\\includegraphics", "\\begin{table}", "\\input{"])
        records.append(
            ClaimEvidence(
                claim_id=f"{entrypoint.submodule_path.name}:claims",
                submodule_path=entrypoint.submodule_path,
                tex_path=entrypoint.tex_path,
                claim_type="text",
                status="blocked",
                reason=(
                    "claim surface requires artifact-level audit"
                    if has_claim_surface
                    else "no figure/table claim surface discovered in selected entrypoint"
                ),
            )
        )
    return tuple(records)


def compile_gates(entrypoints: Optional[Iterable[LatexEntrypoint]] = None) -> Tuple[CompileGate, ...]:
    latexmk = shutil.which("latexmk")
    xelatex = shutil.which("xelatex")
    pdflatex = shutil.which("pdflatex")
    records = []
    for entrypoint in entrypoints or discover_latex_entrypoints():
        pdf_path = str(entrypoint.tex_path.with_suffix(".pdf"))
        log_path = str(entrypoint.tex_path.with_suffix(".log"))
        if entrypoint.status != "selected":
            records.append(CompileGate(entrypoint.tex_path, "", "blocked", "", log_path, entrypoint.reason))
        elif latexmk:
            records.append(CompileGate(entrypoint.tex_path, f"latexmk -pdf {entrypoint.tex_path}", "pending", pdf_path, log_path, ""))
        elif xelatex:
            records.append(CompileGate(entrypoint.tex_path, f"xelatex {entrypoint.tex_path.name}", "pending", pdf_path, log_path, ""))
        elif pdflatex:
            records.append(CompileGate(entrypoint.tex_path, f"pdflatex {entrypoint.tex_path.name}", "pending", pdf_path, log_path, ""))
        else:
            records.append(CompileGate(entrypoint.tex_path, "", "skipped", "", log_path, "missing latexmk/xelatex/pdflatex"))
    return tuple(records)


def submodule_states(paths: Optional[Iterable[Path]] = None) -> Tuple[SubmoduleState, ...]:
    proc = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        check=True,
        capture_output=True,
        text=True,
    )
    status_by_path = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            status_by_path[Path(parts[1])] = (line[0], parts[0].lstrip("+-U"))

    records = []
    for submodule in paths or indexed_uxfd_submodules():
        marker, sha = status_by_path.get(submodule, ("?", ""))
        sub_status = "clean" if marker == " " else "dirty-or-pointer-changed"
        reason = "" if sub_status == "clean" else f"git submodule status marker={marker!r}"
        records.append(
            SubmoduleState(
                submodule_path=submodule,
                submodule_status=sub_status,
                parent_gitlink_status="unknown",
                reason=reason,
                commit_sha=sha,
            )
        )
    return tuple(records)


def render_markdown() -> str:
    lines = ["# UXFD Paper Alignment Audit", "", "## Contracts", ""]
    for contract in audit_contracts():
        reason = f" ({contract.reason})" if contract.reason else ""
        lines.append(f"- `{contract.submodule_path}`: `{contract.status}`{reason}")
    lines.extend(["", "## LaTeX Entrypoints", ""])
    for entrypoint in discover_latex_entrypoints():
        reason = f" ({entrypoint.reason})" if entrypoint.reason else ""
        lines.append(f"- `{entrypoint.submodule_path}`: `{entrypoint.status}` -> `{entrypoint.tex_path}`{reason}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit UXFD paper contracts and alignment gates")
    parser.parse_args(argv)
    print(render_markdown())
    contracts = audit_contracts()
    if any(item.status == "blocked" for item in contracts):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
