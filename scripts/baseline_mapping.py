from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

from scripts.model_support_matrix import derive_model_support
from scripts.task_experiment_matrix import derive_matrix


BASELINE_ROLES = {"mandatory", "optional", "blocked", "unverified"}


@dataclass(frozen=True)
class BaselineMapping:
    task_family: Tuple[str, str]
    baseline_role: str
    model_ref: Tuple[str, str]
    config_path: str
    command: str
    evidence_status: str
    blocker_reason: str = ""


def derive_baselines() -> Tuple[BaselineMapping, ...]:
    return (
        BaselineMapping(
            task_family=("DG", "classification"),
            baseline_role="mandatory",
            model_ref=("ISFM", "M_01_ISFM"),
            config_path="configs/hydra/experiments/00_smoke/dummy_dg.yaml",
            command="bash scripts/run_demo_matrix.sh --mode smoke",
            evidence_status="smoke-tested",
        ),
        BaselineMapping(
            task_family=("CDDG", "classification"),
            baseline_role="optional",
            model_ref=("ISFM", "M_01_ISFM"),
            config_path="configs/hydra/experiments/02_cross_system/multi_system_cddg.yaml",
            command="PHM_VIBENCH_DATA=<data-root> bash scripts/run_demo_matrix.sh --mode full",
            evidence_status="real-data-ready",
        ),
        BaselineMapping(
            task_family=("FS", "classification"),
            baseline_role="optional",
            model_ref=("ISFM", "M_01_ISFM"),
            config_path="configs/hydra/experiments/03_fewshot/cwru_protonet.yaml",
            command="PHM_VIBENCH_DATA=<data-root> bash scripts/run_demo_matrix.sh --mode full",
            evidence_status="real-data-ready",
        ),
        BaselineMapping(
            task_family=("GFS", "classification"),
            baseline_role="optional",
            model_ref=("ISFM", "M_01_ISFM"),
            config_path="configs/hydra/experiments/04_cross_system_fewshot/cross_system_tspn.yaml",
            command="PHM_VIBENCH_DATA=<data-root> bash scripts/run_demo_matrix.sh --mode full",
            evidence_status="real-data-ready",
        ),
        BaselineMapping(
            task_family=("pretrain", "hse_contrastive"),
            baseline_role="mandatory",
            model_ref=("ISFM", "M_01_ISFM"),
            config_path="configs/hydra/experiments/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml",
            command=(
                "python -m pytest -q "
                "test/test_hse_contrastive_failfast.py::test_hse_contrastive_flow_has_nonzero_signal"
            ),
            evidence_status="smoke-tested",
        ),
        BaselineMapping(
            task_family=("DG", "classification"),
            baseline_role="blocked",
            model_ref=("X_model", "CI_GNN"),
            config_path="",
            command="python -m pytest -q test/test_x_model_smoke.py",
            evidence_status="dependency-blocked",
            blocker_reason="requires torch_geometric in the current environment",
        ),
    )


def validate_baselines(baselines: Tuple[BaselineMapping, ...]) -> Tuple[str, ...]:
    model_report = derive_model_support()
    task_report = derive_matrix()
    model_keys = set(model_report.model_statuses)
    task_keys = set(task_report.family_statuses)

    issues: List[str] = []
    for baseline in baselines:
        if baseline.baseline_role not in BASELINE_ROLES:
            issues.append(f"{baseline.task_family}: invalid role {baseline.baseline_role}")
        if baseline.model_ref not in model_keys:
            issues.append(f"{baseline.task_family}: unknown model {baseline.model_ref}")
        if baseline.task_family not in task_keys:
            issues.append(f"{baseline.task_family}: unknown task family")
        if baseline.baseline_role in {"blocked", "unverified"} and not baseline.blocker_reason:
            issues.append(f"{baseline.task_family}: {baseline.baseline_role} baseline lacks blocker reason")
        if baseline.evidence_status in {"dependency-blocked", "failed", "skipped"} and not baseline.blocker_reason:
            issues.append(f"{baseline.task_family}: blocked evidence lacks reason")
    return tuple(issues)


def render_markdown(baselines: Tuple[BaselineMapping, ...]) -> str:
    lines = [
        "# Baseline Mapping",
        "",
        "| Task family | Role | Model | Evidence | Blocker / note |",
        "|---|---|---|---|---|",
    ]
    for item in baselines:
        lines.append(
            "| "
            f"`{item.task_family[0]}.{item.task_family[1]}` | `{item.baseline_role}` | "
            f"`{item.model_ref[0]}.{item.model_ref[1]}` | `{item.evidence_status}` | "
            f"{item.blocker_reason or item.command} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive selected PHM baseline mapping")
    parser.parse_args(argv)

    baselines = derive_baselines()
    print(render_markdown(baselines))
    issues = validate_baselines(baselines)
    if issues:
        for issue in issues:
            print(f"[FAIL] {issue}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
