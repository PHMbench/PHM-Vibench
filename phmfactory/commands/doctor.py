"""Environment diagnostics that never start training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys

from phmfactory.commands.common import check_writable_directory
from phmfactory.config import analyze_config
from phmfactory.pipelines import pipeline_module_name


CORE_MODULES = ("yaml", "torch", "pandas", "pytorch_lightning")


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    passed: bool
    detail: str


def build_parser() -> argparse.ArgumentParser:
    """Build the zero-option doctor parser so standard ``--help`` works."""

    return argparse.ArgumentParser(
        prog="phmfactory doctor",
        description="Validate the installed PHMFactory runtime without training.",
    )


def _module_detail(module: object) -> str:
    version = getattr(module, "__version__", None)
    return f"imported version={version}" if version else "imported"


def collect_checks() -> list[DoctorCheck]:
    """Collect bounded diagnostics without constructing a Pipeline or Trainer."""

    checks: list[DoctorCheck] = []
    python_ok = sys.version_info >= (3, 10)
    checks.append(
        DoctorCheck(
            "python",
            python_ok,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    for module_name in CORE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            checks.append(
                DoctorCheck(
                    f"import:{module_name}",
                    False,
                    f"{type(error).__name__}: {error}",
                )
            )
        else:
            checks.append(
                DoctorCheck(f"import:{module_name}", True, _module_detail(module))
            )

    try:
        analysis = analyze_config("smoke")
    except Exception as error:
        checks.append(
            DoctorCheck("config:smoke", False, f"{type(error).__name__}: {error}")
        )
        return checks

    checks.append(DoctorCheck("config:smoke", True, str(analysis.path)))
    module_name = pipeline_module_name(analysis.pipeline, warn=False)
    try:
        module_found = importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError) as error:
        checks.append(
            DoctorCheck(
                "pipeline:smoke",
                False,
                f"{type(error).__name__}: {error}",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "pipeline:smoke",
                module_found,
                module_name if module_found else f"missing {module_name}",
            )
        )

    environment = analysis.effective_config.get("environment") or {}
    output_dir = environment.get("output_dir")
    try:
        writable = check_writable_directory(str(output_dir))
    except Exception as error:
        checks.append(
            DoctorCheck(
                "output:writable",
                False,
                f"{type(error).__name__}: {error}",
            )
        )
    else:
        checks.append(DoctorCheck("output:writable", True, str(writable)))
    return checks


def run(argv: Sequence[str]) -> list[DoctorCheck]:
    """Print diagnostic records and exit non-zero when any required check fails."""

    build_parser().parse_args(list(argv))
    checks = collect_checks()
    for check in checks:
        label = "PASS" if check.passed else "FAIL"
        print(f"{label} {check.name}: {check.detail}")
    failed = [check for check in checks if not check.passed]
    if failed:
        raise SystemExit(1)
    print(f"doctor=passed checks={len(checks)}")
    return checks
