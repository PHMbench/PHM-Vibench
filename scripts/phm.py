"""Small user-facing helpers around the maintained PHMFactory entrypoint."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

MIN_PYTHON = (3, 10)
REQUIRED_MODULES = ("yaml", "torch", "pytorch_lightning", "numpy", "pandas")
SMOKE_CONFIG = Path("configs/demo/00_smoke/dummy_dg.yaml")
SMOKE_OUTPUT = Path("results/demo/dummy_dg_smoke")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    remediation: str = ""


def repository_root() -> Path:
    """Return the repository root containing this module's ``scripts`` package."""
    return Path(__file__).resolve().parents[1]


def _python_check() -> CheckResult:
    version = sys.version_info[:3]
    passed = version >= MIN_PYTHON
    detail = ".".join(str(value) for value in version)
    remediation = "Install Python 3.10 or newer." if not passed else ""
    return CheckResult("python", passed, detail, remediation)


def _file_check(root: Path, relative: Path, name: str) -> CheckResult:
    path = root / relative
    passed = path.is_file()
    return CheckResult(
        name,
        passed,
        str(path),
        f"Run from a complete PHMFactory checkout containing {relative}." if not passed else "",
    )


def _dependency_checks(
    importer: Callable[[str], object] = importlib.import_module,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importer(module_name)
        except Exception as exc:  # import failures may wrap native-library errors
            results.append(
                CheckResult(
                    f"dependency:{module_name}",
                    False,
                    f"{type(exc).__name__}: {exc}",
                    "Install the repository core requirements in the active environment.",
                )
            )
            continue
        version = str(getattr(module, "__version__", "available"))
        results.append(CheckResult(f"dependency:{module_name}", True, version))
    return results


def _writable_output_check(root: Path) -> CheckResult:
    output_root = root / "results"
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".phmfactory-doctor-", dir=output_root):
            pass
    except OSError as exc:
        return CheckResult(
            "output:writable",
            False,
            f"{output_root}: {exc}",
            "Grant write permission or use a writable checkout/output location.",
        )
    return CheckResult("output:writable", True, str(output_root))


def collect_doctor_checks(
    root: Path | None = None,
    *,
    importer: Callable[[str], object] = importlib.import_module,
) -> list[CheckResult]:
    """Collect deterministic environment checks without starting training."""
    resolved_root = (root or repository_root()).resolve()
    checks = [
        _python_check(),
        CheckResult(
            "repository:root",
            (resolved_root / ".git").exists() or (resolved_root / "pyproject.toml").is_file(),
            str(resolved_root),
            "Run the command from a PHMFactory source checkout.",
        ),
        _file_check(resolved_root, Path("main.py"), "entrypoint:main.py"),
        _file_check(resolved_root, SMOKE_CONFIG, "config:offline-smoke"),
        _file_check(resolved_root, Path("data/metadata_dummy.csv"), "data:dummy-metadata"),
        _writable_output_check(resolved_root),
    ]
    checks.extend(_dependency_checks(importer))
    return checks


def run_doctor(
    root: Path | None = None,
    *,
    importer: Callable[[str], object] = importlib.import_module,
) -> int:
    checks = collect_doctor_checks(root, importer=importer)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
        if check.remediation and not check.passed:
            print(f"       remediation: {check.remediation}")
    failed = [check for check in checks if not check.passed]
    print(f"doctor: {'PASS' if not failed else 'FAIL'} ({len(checks) - len(failed)}/{len(checks)})")
    return 0 if not failed else 1


def _flatten_leaves(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, Mapping):
            leaves.extend(_flatten_leaves(item, path))
        else:
            leaves.append((path, item))
    return leaves


def _smoke_override_values(
    root: Path,
    *,
    epochs: int,
    num_workers: int,
) -> list[str]:
    """Return CLI-precedence values that restore the maintained Dummy contract."""
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    from phmfactory.config import resolve_config

    previous = Path.cwd()
    try:
        os.chdir(root)
        resolved = resolve_config(SMOKE_CONFIG)
    finally:
        os.chdir(previous)

    payload = deepcopy(resolved.data)
    payload.pop("pipeline", None)
    data = payload.setdefault("data", {})
    data["data_dir"] = str((root / "data").resolve())
    data["metadata_file"] = "metadata_dummy.csv"
    data["num_workers"] = num_workers
    environment = payload.setdefault("environment", {})
    environment["output_dir"] = str((root / SMOKE_OUTPUT).resolve())
    trainer = payload.setdefault("trainer", {})
    trainer["num_epochs"] = epochs
    trainer["device"] = "cpu"
    trainer["gpus"] = 1

    return [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in _flatten_leaves(payload)]


def demo_command(
    root: Path,
    *,
    epochs: int = 1,
    num_workers: int = 0,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "main.py"),
        "--config",
        str((root / SMOKE_CONFIG).resolve()),
    ]
    for override in _smoke_override_values(
        root,
        epochs=epochs,
        num_workers=num_workers,
    ):
        command.extend(("--override", override))
    return command


def run_demo(
    root: Path | None = None,
    *,
    epochs: int = 1,
    num_workers: int = 0,
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    resolved_root = (root or repository_root()).resolve()
    command = demo_command(resolved_root, epochs=epochs, num_workers=num_workers)
    print("command:", " ".join(command))
    print("local overrides: restored to the maintained Dummy contract at CLI precedence")
    if dry_run:
        print("demo: DRY RUN")
        return 0
    completed = runner(
        command,
        cwd=resolved_root,
        check=False,
        text=True,
        shell=False,
    )
    if completed.returncode == 0:
        print(f"demo: PASS; inspect {resolved_root / SMOKE_OUTPUT}")
    else:
        print(f"demo: FAIL (exit={completed.returncode})", file=sys.stderr)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.phm",
        description="PHMFactory environment and offline-demo helpers.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the local source environment without training.")
    demo = commands.add_parser("demo", help="Run the repository-shipped offline smoke demo.")
    demo.add_argument("--epochs", type=int, default=1)
    demo.add_argument("--num-workers", type=int, default=0)
    demo.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "doctor":
        return run_doctor()
    return run_demo(
        epochs=args.epochs,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
