#!/usr/bin/env python3
"""Enforce PHMFactory v0.3 dependency ownership boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
NAME_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)")

GOVERNED_FILES = {
    "core": Path("requirements.txt"),
    "streamlit": Path("apps/streamlit/requirements.txt"),
    "modelscope": Path("phmfactory/data_sources/modelscope/requirements.txt"),
    "plotting": Path("plot/requirements.txt"),
    "tests": Path("test/requirements.txt"),
}

EXPECTED_OWNERS = {
    "streamlit": {"streamlit"},
    "modelscope": {"modelscope"},
    "plotting": {"scienceplots", "umap-learn"},
    "tests": {"pytest"},
}

FORBIDDEN_IN_CORE = set().union(*EXPECTED_OWNERS.values()) | {
    "plotly",       # legacy app owns it through app/requirements_gui.txt
    "torchaudio",   # no maintained runtime import in the frozen audit
    "torchvision",  # no maintained runtime import in the frozen audit
    "urllib3",      # transitive library; no direct maintained import
}

REQUIRED_IN_CORE = {
    "h5py",
    "huggingface-hub",
    "openpyxl",
    "pydantic",
    "pyyaml",
    "torch",
}

PROHIBITED_REFERENCE_MARKERS = (
    "git+ssh://",
    "git@github.com:",
    "file://",
    "/home/",
    "/users/",
    "c:\\users\\",
)


@dataclass(frozen=True)
class RequirementFile:
    owner: str
    path: Path
    packages: frozenset[str]


def canonical_name(name: str) -> str:
    """Return a PEP 503-style comparison key."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement_file(owner: str, relative_path: Path) -> RequirementFile:
    path = ROOT / relative_path
    if not path.is_file():
        raise ValueError(f"Missing governed requirements file: {relative_path}")

    packages: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lowered = stripped.lower()
        if any(marker in lowered for marker in PROHIBITED_REFERENCE_MARKERS):
            raise ValueError(
                f"{relative_path}:{line_number}: prohibited local/private reference"
            )
        if stripped.startswith(("-r", "--requirement", "-e", "--editable")):
            raise ValueError(
                f"{relative_path}:{line_number}: requirement indirection is not allowed; "
                "subsystem files must contain incremental packages only"
            )
        match = NAME_PATTERN.match(stripped)
        if match is None:
            raise ValueError(
                f"{relative_path}:{line_number}: cannot parse requirement {stripped!r}"
            )
        package = canonical_name(match.group(1))
        if package in packages:
            raise ValueError(
                f"{relative_path}:{line_number}: duplicate package {package!r}"
            )
        packages.add(package)
    return RequirementFile(owner, relative_path, frozenset(packages))


def validate(files: Iterable[RequirementFile]) -> None:
    by_owner = {item.owner: item for item in files}
    core = by_owner["core"].packages

    missing_core = sorted(REQUIRED_IN_CORE - core)
    if missing_core:
        raise ValueError(f"Core requirements are missing: {missing_core}")

    leaked = sorted(core & FORBIDDEN_IN_CORE)
    if leaked:
        raise ValueError(f"Optional or unused packages leaked into core: {leaked}")

    claimed_by: dict[str, str] = {}
    for owner, expected in EXPECTED_OWNERS.items():
        packages = by_owner[owner].packages
        missing = sorted(expected - packages)
        unexpected = sorted(packages - expected)
        if missing or unexpected:
            raise ValueError(
                f"{by_owner[owner].path}: expected {sorted(expected)}, "
                f"found {sorted(packages)}; missing={missing}, unexpected={unexpected}"
            )
        duplicates = sorted(packages & core)
        if duplicates:
            raise ValueError(
                f"{by_owner[owner].path}: duplicates core packages: {duplicates}"
            )
        for package in packages:
            previous = claimed_by.setdefault(package, owner)
            if previous != owner:
                raise ValueError(
                    f"Optional package {package!r} is claimed by both "
                    f"{previous!r} and {owner!r}"
                )


def main() -> int:
    try:
        files = [
            parse_requirement_file(owner, path)
            for owner, path in GOVERNED_FILES.items()
        ]
        validate(files)
    except ValueError as exc:
        print(f"requirements boundary check failed: {exc}", file=sys.stderr)
        return 1

    for item in files:
        packages = ", ".join(sorted(item.packages)) or "(none)"
        print(f"{item.owner:11} {item.path}: {packages}")
    print("PHMFactory v0.3 dependency ownership: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
