"""Environment diagnostics that never start training."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from collections.abc import Sequence

from phmfactory.commands.common import check_writable_directory
from phmfactory.config import resolve_config
from phmfactory.pipelines import pipeline_module_name


CORE_MODULES = ("yaml", "torch", "pandas", "pytorch_lightning")


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    passed: bool
    detail: str


def collect_checks() -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    python_ok = sys.version_info >= (3, 10)
    checks.append(
        DoctorCheck(
            "python",
            python_ok,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    for module in CORE_MODULES:
        found = importlib.util.find_spec(module) is not None
        checks.append(DoctorCheck(f"import:{module}", found, "available" if found else "missing"))

    try:
        resolved = resolve_config("smoke")
    except Exception as error:
        checks.append(DoctorCheck("config:smoke", False, str(error)))
        return checks

    checks.append(DoctorCheck("config:smoke", True, str(resolved.path)))
    module_name = pipeline_module_name(resolved.pipeline, warn=False)
    module_found = importlib.util.find_spec(module_name) is not None
    checks.append(
        DoctorCheck(
            "pipeline:smoke",
            module_found,
            module_name if module_found else f"missing {module_name}",
        )
    )

    environment = resolved.data.get("environment") or {}
    output_dir = environment.get("output_dir")
    try:
        writable = check_writable_directory(str(output_dir))
    except Exception as error:
        checks.append(DoctorCheck("output:writable", False, str(error)))
    else:
        checks.append(DoctorCheck("output:writable", True, str(writable)))
    return checks


def run(argv: Sequence[str]) -> list[DoctorCheck]:
    if argv:
        raise SystemExit("phmfactory doctor takes no arguments")
    checks = collect_checks()
    for check in checks:
        label = "PASS" if check.passed else "FAIL"
        print(f"{label} {check.name}: {check.detail}")
    failed = [check for check in checks if not check.passed]
    if failed:
        raise SystemExit(1)
    print(f"doctor=passed checks={len(checks)}")
    return checks
