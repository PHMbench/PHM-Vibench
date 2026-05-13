from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import yaml

from scripts.uxfd_owner_review_gate import DEFAULT_DECISION_FILE, evaluate_owner_review_gate
from scripts.uxfd_recent_work_gate import evaluate_recent_work_gate
from scripts.uxfd_submission_gate import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_QUEUE,
    GOAL_DIR,
    PAPER07_GOAL,
    PAPER07_REJECTION_CONTRACT,
    PAPER07_REJECTION_NEEDLES,
    PAPER07_REVIEWER_TRACE,
    PAPER07_REVIEWER_TRACE_NEEDLES,
    REQUIRED_GOAL_FILES,
    evaluate_submission_gate,
)


SPEC_DIR = Path("specs/006-uxfd-ieee-trans-submission-readiness")
CLAUDE_TEAM_DIR = Path(".codex/claude-team-runs/20260511-uxfd-ieee-trans-review")
HANDOFF_PATH = Path(
    ".claude/handoffs/2026-05-11-uxfd-ieee-trans-submission-readiness.md"
)
CONTINUATION_HANDOFF_PATH = Path(
    ".claude/handoffs/2026-05-12-uxfd-goal-continuation.md"
)
EXECUTION_GATE_HANDOFF_PATH = Path(
    ".claude/handoffs/2026-05-13-uxfd-execution-gate-check.md"
)
LATEST_CONTINUATION_HANDOFF_PATH = Path(
    ".claude/handoffs/2026-05-14-uxfd-owner-gpu-blocked-continuation.md"
)

REQUIRED_SPEC_FILES = (
    "spec.md",
    "plan.md",
    "tasks.md",
    "research.md",
    "data-model.md",
    "quickstart.md",
    "contracts/uxfd-ieee-trans-submission-readiness-contract.md",
    "checklists/requirements.md",
    "checklists/submission-readiness.md",
)

CLAUDE_TEAM_OUTPUTS = (
    "report.md",
    "risks.md",
    "test-log.md",
)

CODEX_SUBAGENT_LAUNCH = "CODEX_SUBAGENT_LAUNCH.md"
SUBAGENT_ID_PATTERN = re.compile(
    r"`([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})`"
)

EXECUTION_ARTIFACTS = (
    ("GPU execution runbook", Path("paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md")),
    ("live GPU preflight snapshot", Path("paper/UXFD_paper/results/gpu_queue_live_preflight.json")),
    ("combined GPU launch plan", Path("paper/UXFD_paper/results/queue_launch_plan.sh")),
    ("GPU0 launch shard", Path("paper/UXFD_paper/results/queue_launch_shards/gpu0.sh")),
    ("GPU1 launch shard", Path("paper/UXFD_paper/results/queue_launch_shards/gpu1.sh")),
    (
        "accepted-run template manifest",
        Path("paper/UXFD_paper/results/accepted_run_templates/manifest.json"),
    ),
    (
        "SOTA aggregate template manifest",
        Path("paper/UXFD_paper/results/sota_aggregate_templates/manifest.yaml"),
    ),
    (
        "SOTA aggregate scaffold report",
        Path("paper/UXFD_paper/results/sota_aggregate_templates/scaffold_report.md"),
    ),
    (
        "artifact queue coverage report",
        Path("paper/UXFD_paper/results/artifact_gate_queue_coverage.md"),
    ),
    (
        "SOTA aggregate gate JSON report",
        Path("paper/UXFD_paper/results/sota_gate_current.json"),
    ),
    (
        "SOTA aggregate gate markdown report",
        Path("paper/UXFD_paper/results/sota_gate_current.md"),
    ),
    (
        "submodule dirty triage report",
        Path("paper/UXFD_paper/results/submodule_dirty_triage.md"),
    ),
    (
        "submodule dirty triage JSON report",
        Path("paper/UXFD_paper/results/submodule_dirty_triage.json"),
    ),
    (
        "submodule owner-review recommendations",
        Path("paper/UXFD_paper/results/submodule_owner_review_recommendations.md"),
    ),
    (
        "submodule owner-review action packet",
        Path("paper/UXFD_paper/results/submodule_owner_review_action_packet.md"),
    ),
    (
        "submodule owner-review decision template",
        Path("paper/UXFD_paper/results/submodule_owner_review_decisions.template.json"),
    ),
    (
        "submodule owner-review decision file",
        DEFAULT_DECISION_FILE,
    ),
    (
        "submodule owner-review gate JSON report",
        Path("paper/UXFD_paper/results/submodule_owner_review_gate_current.json"),
    ),
    (
        "submodule owner-review gate markdown report",
        Path("paper/UXFD_paper/results/submodule_owner_review_gate_current.md"),
    ),
    (
        "parent result artifact triage report",
        Path("paper/UXFD_paper/results/parent_result_artifact_triage.md"),
    ),
    (
        "readiness execution backlog",
        Path("paper/UXFD_paper/results/readiness_backlog.md"),
    ),
    (
        "goal clarity audit report",
        Path("paper/UXFD_paper/results/goal_clarity_audit_current.md"),
    ),
    (
        "commit recovery plan",
        Path("paper/UXFD_paper/results/commit_recovery_plan.md"),
    ),
    (
        "low-tier source audit report",
        Path("paper/UXFD_paper/results/low_tier_source_audit.md"),
    ),
)

LAUNCH_SCRIPT_STATIC_GATE_PATHS = (
    Path("paper/UXFD_paper/results/queue_launch_plan.sh"),
    Path("paper/UXFD_paper/results/queue_launch_shards/gpu0.sh"),
    Path("paper/UXFD_paper/results/queue_launch_shards/gpu1.sh"),
)
LAUNCH_SCRIPT_STATIC_GATE_NEEDLES = (
    "Blocked: static queue validation can_execute=False",
    "exit 2",
)
ARTIFACT_GATE_FINITE_METRIC_NEEDLES = (
    "metrics_path JSON must contain at least one numeric metric",
    "metrics_path CSV must contain at least one numeric metric",
    "metrics_path must not contain TODO placeholders",
    "metrics_path JSON numeric metrics must be finite",
    "metrics_path CSV numeric metrics must be finite",
)
SOURCE_TREE_STATUS_NEEDLES = (
    "source_tree_status must be clean",
    "source_tree_status",
)
RUN_CONTROL_NEEDLES = (
    "seed must be a non-negative integer",
    "batch_size must be a positive integer",
)
RUNTIME_NEEDLES = (
    "runtime must be positive HH:MM:SS",
    "RUNTIME_PATTERN",
)
PRECISION_NEEDLES = (
    "precision must be one of fp32, tf32, fp16, bf16, amp",
    "ACCEPTED_PRECISION_VALUES",
)
EVIDENCE_LEVEL_NEEDLES = (
    "evidence_level must be accepted_same_protocol",
    "ACCEPTED_EVIDENCE_LEVEL_VALUES",
)
PREPROCESSING_SIGNATURE_NEEDLES = (
    "preprocessing_signature must match sha256:<64 lowercase hex>",
    "PREPROCESSING_SIGNATURE_PATTERN",
)
SHA_PROVENANCE_NEEDLES = (
    "DISALLOWED_SHA_PROVENANCE_MARKERS",
    "git_sha_or_submodule_sha must not contain",
)
ACCEPTED_RUN_ROOT_README = Path("paper/UXFD_paper/results/accepted_runs/README.md")
ACCEPTED_RUN_ROOT_GATE_NEEDLES = (
    "uxfd_gpu_queue --live-preflight --require-preflight",
    "Blocked: static queue validation can_execute=False",
    "uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage",
)
SOTA_AGGREGATE_TEMPLATE_README = Path(
    "paper/UXFD_paper/results/sota_aggregate_templates/README.md"
)
SOTA_AGGREGATE_ACTIVATION_NEEDLES = (
    "Activation preflight",
    "uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage",
    "must pass before creating `paper/UXFD_paper/results/sota_aggregates`",
    "Do not commit template-derived `sota_aggregate.yaml`",
)
SOTA_COMPARISON_CONTRACT_FIELDS = (
    "single_run_rule",
    "same_protocol_population",
    "seed_protocol",
    "aggregate_statistics",
    "accepted_run_ref_binding",
    "ablation_dependency",
    "top_scope",
    "claim_output",
)
SOTA_COMPARISON_CONTRACT_NEEDLES = (
    "single run",
    "matched seed",
    "minimum_seeds",
    "95% confidence interval",
    "effect size",
    "accepted_run_refs",
    "run_meta.yaml",
    "failure_record",
    "representative top proxy",
    "exact external",
)

PAPER_SUBMODULES = (
    Path("paper/UXFD_paper/Explainable_FD_Toolkit"),
    Path("paper/UXFD_paper/1D-2D_fusion_explainable"),
    Path("paper/UXFD_paper/LLM_Explainable_FD_Toolkit"),
    Path("paper/UXFD_paper/MOE_explainable"),
    Path("paper/UXFD_paper/Paper_fuzzy_XFD"),
    Path("paper/UXFD_paper/Neuralsymbolic_theory"),
    Path("paper/UXFD_paper/TII_operator_attention"),
)

PARENT_GOAL_CHECKPOINT_PATHS = (
    Path(".claude/handoffs/2026-05-12-uxfd-goal-continuation.md"),
    EXECUTION_GATE_HANDOFF_PATH,
    LATEST_CONTINUATION_HANDOFF_PATH,
    Path("paper/UXFD_paper/goal/README.md"),
    Path("paper/UXFD_paper/goal/09_gpu_execution_queue.yaml"),
    Path("paper/UXFD_paper/goal/99_submission_readiness_matrix.md"),
    Path("paper/UXFD_paper/goal/status"),
    Path("paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md"),
    Path("paper/UXFD_paper/results/gpu_queue_live_preflight.json"),
    Path("paper/UXFD_paper/results/.gitignore"),
    Path("paper/UXFD_paper/results/queue_launch_plan.sh"),
    Path("paper/UXFD_paper/results/queue_launch_shards/gpu0.sh"),
    Path("paper/UXFD_paper/results/queue_launch_shards/gpu1.sh"),
    Path("paper/UXFD_paper/results/accepted_runs"),
    Path("paper/UXFD_paper/results/accepted_run_templates"),
    Path("paper/UXFD_paper/results/sota_aggregate_templates"),
    Path("paper/UXFD_paper/results/submission_gate_current.json"),
    Path("paper/UXFD_paper/results/submission_gate_current.md"),
    Path("paper/UXFD_paper/results/recent_work_gate_current.json"),
    Path("paper/UXFD_paper/results/recent_work_gate_current.md"),
    Path("paper/UXFD_paper/results/sota_gate_current.json"),
    Path("paper/UXFD_paper/results/sota_gate_current.md"),
    Path("paper/UXFD_paper/results/submodule_dirty_triage.md"),
    Path("paper/UXFD_paper/results/submodule_dirty_triage.json"),
    Path("paper/UXFD_paper/results/submodule_owner_review_recommendations.md"),
    Path("paper/UXFD_paper/results/submodule_owner_review_action_packet.md"),
    Path("paper/UXFD_paper/results/submodule_owner_review_decisions.template.json"),
    Path("paper/UXFD_paper/results/submodule_owner_review_gate_current.json"),
    Path("paper/UXFD_paper/results/submodule_owner_review_gate_current.md"),
    Path("paper/UXFD_paper/results/parent_result_artifact_triage.md"),
    Path("paper/UXFD_paper/results/goal_clarity_audit_current.md"),
    Path("paper/UXFD_paper/results/commit_recovery_plan.md"),
    Path("paper/UXFD_paper/results/low_tier_source_audit.md"),
    Path("paper/UXFD_paper/results/low_tier_source_audit.json"),
    Path("scripts/uxfd_low_tier_source_audit.py"),
    Path("scripts/uxfd_artifact_gate.py"),
    Path("scripts/uxfd_goal_status.py"),
    Path("scripts/uxfd_objective_audit.py"),
    Path("scripts/uxfd_artifact_scaffold.py"),
    Path("scripts/uxfd_sota_scaffold.py"),
    Path("scripts/uxfd_gpu_queue.py"),
    Path("scripts/uxfd_parent_result_artifact_triage.py"),
    Path("scripts/uxfd_readiness_backlog.py"),
    Path("scripts/uxfd_recent_work_gate.py"),
    Path("scripts/uxfd_sota_gate.py"),
    Path("scripts/uxfd_submission_gate.py"),
    Path("scripts/uxfd_submodule_dirty_triage.py"),
    Path("scripts/uxfd_owner_review_gate.py"),
    Path("test/test_uxfd_low_tier_source_audit.py"),
    Path("test/test_uxfd_goal_status.py"),
    Path("test/test_uxfd_parent_result_artifact_triage.py"),
    Path("test/test_uxfd_artifact_gate.py"),
    Path("test/test_uxfd_artifact_scaffold.py"),
    Path("test/test_uxfd_sota_scaffold.py"),
    Path("test/test_uxfd_gpu_queue.py"),
    Path("test/test_uxfd_paper01_control_docs.py"),
    Path("test/test_uxfd_paper02_control_docs.py"),
    Path("test/test_uxfd_paper02_runner_policy.py"),
    Path("test/test_uxfd_paper04_control_docs.py"),
    Path("test/test_uxfd_paper04_runner_policy.py"),
    Path("test/test_uxfd_paper04_truth_manuscript.py"),
    Path("test/test_uxfd_objective_audit.py"),
    Path("test/test_uxfd_readiness_backlog.py"),
    Path("test/test_uxfd_recent_work_gate.py"),
    Path("test/test_uxfd_sota_gate.py"),
    Path("test/test_uxfd_submission_gate.py"),
    Path("test/test_uxfd_submodule_dirty_triage.py"),
    Path("test/test_uxfd_owner_review_gate.py"),
    Path("test/test_uxfd_goal_clarity.py"),
)


@dataclass(frozen=True)
class ObjectiveAuditItem:
    requirement: str
    evidence: str
    status: str
    details: str


@dataclass(frozen=True)
class ObjectiveAuditReport:
    achieved: bool
    objective: str
    items: Tuple[ObjectiveAuditItem, ...]
    blockers: Tuple[str, ...]
    met: int
    not_met: int
    blocked: int
    unverified: int


def _item(requirement: str, evidence: Path | str, status: str, details: str) -> ObjectiveAuditItem:
    return ObjectiveAuditItem(
        requirement=requirement,
        evidence=str(evidence),
        status=status,
        details=details,
    )


def _exists_item(requirement: str, path: Path) -> ObjectiveAuditItem:
    return _item(
        requirement=requirement,
        evidence=path,
        status="met" if path.exists() else "not_met",
        details="exists" if path.exists() else "missing",
    )


def _text_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8")


def _git_status_lines(path: Path) -> Tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line.strip())


def _git_status_lines_for_paths(paths: Sequence[Path]) -> Tuple[str, ...]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *(str(path) for path in paths)],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line.strip())


def _paper_submodule_cleanliness_item(
    submodule_paths: Sequence[Path] = PAPER_SUBMODULES,
) -> ObjectiveAuditItem:
    dirty: List[str] = []
    unreadable: List[str] = []
    for path in submodule_paths:
        try:
            status_lines = _git_status_lines(path)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            unreadable.append(f"{path} ({exc.__class__.__name__})")
            continue
        if status_lines:
            dirty.append(f"{path.name}:{len(status_lines)}")

    if unreadable:
        return _item(
            requirement="paper submodule working trees clean before parent handoff",
            evidence="git -C <paper_submodule> status --porcelain",
            status="unverified",
            details="unreadable_submodules=" + ", ".join(unreadable),
        )
    if dirty:
        return _item(
            requirement="paper submodule working trees clean before parent handoff",
            evidence="git -C <paper_submodule> status --porcelain",
            status="not_met",
            details="dirty_submodules=" + ", ".join(dirty),
        )
    return _item(
        requirement="paper submodule working trees clean before parent handoff",
        evidence="git -C <paper_submodule> status --porcelain",
        status="met",
        details=f"{len(submodule_paths)} paper submodules clean",
    )


def _parent_goal_checkpoint_item(
    paths: Sequence[Path] = PARENT_GOAL_CHECKPOINT_PATHS,
) -> ObjectiveAuditItem:
    try:
        status_lines = _git_status_lines_for_paths(paths)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return _item(
            requirement="parent UXFD goal-control checkpoint committed",
            evidence="git status --porcelain -- <UXFD goal-control paths>",
            status="unverified",
            details=f"git status failed: {exc.__class__.__name__}",
        )

    if status_lines:
        return _item(
            requirement="parent UXFD goal-control checkpoint committed",
            evidence="git status --porcelain -- <UXFD goal-control paths>",
            status="not_met",
            details=f"dirty_parent_goal_control_paths={len(status_lines)}",
        )

    return _item(
        requirement="parent UXFD goal-control checkpoint committed",
        evidence="git status --porcelain -- <UXFD goal-control paths>",
        status="met",
        details=f"{len(paths)} parent goal-control paths clean",
    )


def _launch_scripts_static_gate_item(
    paths: Sequence[Path] = LAUNCH_SCRIPT_STATIC_GATE_PATHS,
) -> ObjectiveAuditItem:
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        return _item(
            requirement="GPU launch scripts enforce static queue gate",
            evidence=",".join(str(path) for path in paths),
            status="not_met",
            details="missing_paths=" + ",".join(missing_paths),
        )

    missing_needles: List[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for needle in LAUNCH_SCRIPT_STATIC_GATE_NEEDLES:
            if needle not in text:
                missing_needles.append(f"{path}:{needle}")

    if missing_needles:
        return _item(
            requirement="GPU launch scripts enforce static queue gate",
            evidence=",".join(str(path) for path in paths),
            status="not_met",
            details="missing_guard_markers=" + ",".join(missing_needles),
        )

    return _item(
        requirement="GPU launch scripts enforce static queue gate",
        evidence=",".join(str(path) for path in paths),
        status="met",
        details="queue_launch_plan.sh,gpu0.sh,gpu1.sh print blocked reason and exit 2",
    )


def _finite_metrics_contract_item(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    if not queue_path.exists():
        missing.append(str(queue_path))
        queue_contract: Mapping[str, Any] = {}
    else:
        queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
        queue_contract = queue.get("accepted_artifact_contract", {})

    metrics_text = str(queue_contract.get("metrics", "")).lower()
    if queue_contract.get("numeric_metrics_required") is not True:
        missing.append("accepted_artifact_contract.numeric_metrics_required")
    if "numeric metric" not in metrics_text:
        missing.append("accepted_artifact_contract.metrics numeric metric wording")
    if "finite" not in metrics_text:
        missing.append("accepted_artifact_contract.metrics finite metric wording")
    for disallowed_marker in ("todo", "nan", "infinite"):
        if disallowed_marker not in metrics_text:
            missing.append(
                f"accepted_artifact_contract.metrics missing {disallowed_marker} rejection"
            )

    if not artifact_gate_path.exists():
        missing.append(str(artifact_gate_path))
    else:
        gate_text = artifact_gate_path.read_text(encoding="utf-8")
        for needle in ARTIFACT_GATE_FINITE_METRIC_NEEDLES:
            if needle not in gate_text:
                missing.append(f"{artifact_gate_path}:{needle}")

    if missing:
        return _item(
            requirement="accepted metrics contain finite values",
            evidence=f"{queue_path},{artifact_gate_path}",
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="accepted metrics contain finite values",
        evidence=f"{queue_path},{artifact_gate_path}",
        status="met",
        details=(
            "queue contract and artifact gate require at least one finite numeric "
            "metric and reject TODO, NaN, and infinite metric payloads"
        ),
    )


def _source_tree_status_contract_item(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
    artifact_scaffold_path: Path = Path("scripts/uxfd_artifact_scaffold.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    if not queue_path.exists():
        missing.append(str(queue_path))
        queue: Mapping[str, Any] = {}
    else:
        queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))

    if "source tree status" not in queue.get("accepted_run_metadata_required", []):
        missing.append("accepted_run_metadata_required.source tree status")

    for path, needles in (
        (artifact_gate_path, SOURCE_TREE_STATUS_NEEDLES),
        (artifact_scaffold_path, ("source_tree_status",)),
    ):
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path}:{needle}")

    if missing:
        return _item(
            requirement="accepted artifacts require clean source trees",
            evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="accepted artifacts require clean source trees",
        evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
        status="met",
        details="queue contract, artifact gate, and templates require source_tree_status clean",
    )


def _run_control_contract_item(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
    artifact_scaffold_path: Path = Path("scripts/uxfd_artifact_scaffold.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    if not queue_path.exists():
        missing.append(str(queue_path))
        queue_contract: Mapping[str, Any] = {}
    else:
        queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
        queue_contract = queue.get("accepted_artifact_contract", {})

    run_controls = str(queue_contract.get("run_controls", "")).lower()
    for needle in ("seed", "non-negative integer", "batch size", "positive integer"):
        if needle not in run_controls:
            missing.append(f"accepted_artifact_contract.run_controls.{needle}")
    seed_contracts = {
        "seed_uniqueness": ("source_queue_id", "entry_id", "seed", "duplicated"),
        "minimum_seed_coverage": ("minimum_seeds", "distinct", "accepted seed"),
    }
    for field, needles in seed_contracts.items():
        text = str(queue_contract.get(field, "")).lower()
        for needle in needles:
            if needle not in text:
                missing.append(f"accepted_artifact_contract.{field}.{needle}")

    for path, needles in (
        (
            artifact_gate_path,
            RUN_CONTROL_NEEDLES
            + (
                "queue_seed_key",
                "duplicate accepted run_meta.yaml queue+seed keys",
                "queue seed coverage incomplete",
                "minimum_seeds",
            ),
        ),
        (
            artifact_scaffold_path,
            (
                "Run-control rule",
                "batch_size",
                "Seed-uniqueness rule",
                "Minimum-seed rule",
            ),
        ),
    ):
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path}:{needle}")

    if missing:
        return _item(
            requirement="accepted artifacts require numeric run controls",
            evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="accepted artifacts require numeric run controls",
        evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
        status="met",
        details=(
            "queue contract, artifact gate, and templates require integer seed "
            "and batch_size, unique queue+seed keys, and minimum_seeds coverage"
        ),
    )


def _runtime_contract_item(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
    artifact_scaffold_path: Path = Path("scripts/uxfd_artifact_scaffold.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    if not queue_path.exists():
        missing.append(str(queue_path))
        queue_contract: Mapping[str, Any] = {}
    else:
        queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
        queue_contract = queue.get("accepted_artifact_contract", {})

    runtime_text = str(queue_contract.get("runtime", "")).lower()
    for needle in ("positive", "hh:mm:ss"):
        if needle not in runtime_text:
            missing.append(f"accepted_artifact_contract.runtime.{needle}")

    for path, needles in (
        (artifact_gate_path, RUNTIME_NEEDLES),
        (artifact_scaffold_path, ("Runtime rule", "HH:MM:SS")),
    ):
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path}:{needle}")

    if missing:
        return _item(
            requirement="accepted artifacts require positive runtime metadata",
            evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="accepted artifacts require positive runtime metadata",
        evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
        status="met",
        details="queue contract, artifact gate, and templates require positive HH:MM:SS runtime",
    )


def _precision_contract_item(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
    artifact_scaffold_path: Path = Path("scripts/uxfd_artifact_scaffold.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    if not queue_path.exists():
        missing.append(str(queue_path))
        queue_contract: Mapping[str, Any] = {}
    else:
        queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
        queue_contract = queue.get("accepted_artifact_contract", {})

    precision_text = str(queue_contract.get("precision", "")).lower()
    for needle in ("fp32", "tf32", "fp16", "bf16", "amp"):
        if needle not in precision_text:
            missing.append(f"accepted_artifact_contract.precision.{needle}")

    for path, needles in (
        (artifact_gate_path, PRECISION_NEEDLES),
        (artifact_scaffold_path, ("Precision rule", "precision")),
    ):
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path}:{needle}")

    if missing:
        return _item(
            requirement="accepted artifacts require enumerated precision metadata",
            evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="accepted artifacts require enumerated precision metadata",
        evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
        status="met",
        details="queue contract, artifact gate, and templates require precision enum",
    )


def _evidence_level_contract_item(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
    artifact_scaffold_path: Path = Path("scripts/uxfd_artifact_scaffold.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    if not queue_path.exists():
        missing.append(str(queue_path))
        queue_contract: Mapping[str, Any] = {}
    else:
        queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
        queue_contract = queue.get("accepted_artifact_contract", {})

    evidence_level_text = str(queue_contract.get("evidence_level", "")).lower()
    for needle in ("accepted_same_protocol", "smoke", "demo", "dummy", "template", "pending"):
        if needle not in evidence_level_text:
            missing.append(f"accepted_artifact_contract.evidence_level.{needle}")

    for path, needles in (
        (artifact_gate_path, EVIDENCE_LEVEL_NEEDLES),
        (artifact_scaffold_path, ("Evidence-level rule", "evidence_level")),
    ):
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path}:{needle}")

    if missing:
        return _item(
            requirement="accepted artifacts require accepted_same_protocol evidence level",
            evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="accepted artifacts require accepted_same_protocol evidence level",
        evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
        status="met",
        details=(
            "queue contract, artifact gate, and templates reject non-accepted "
            "smoke/demo/dummy/template/pending evidence levels"
        ),
    )


def _preprocessing_signature_contract_item(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
    artifact_scaffold_path: Path = Path("scripts/uxfd_artifact_scaffold.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    if not queue_path.exists():
        missing.append(str(queue_path))
        queue_contract: Mapping[str, Any] = {}
    else:
        queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
        queue_contract = queue.get("accepted_artifact_contract", {})

    signature_text = str(queue_contract.get("preprocessing_signature", "")).lower()
    for needle in ("sha256", "64 lowercase hex"):
        if needle not in signature_text:
            missing.append(f"accepted_artifact_contract.preprocessing_signature.{needle}")

    for path, needles in (
        (artifact_gate_path, PREPROCESSING_SIGNATURE_NEEDLES),
        (artifact_scaffold_path, ("Protocol-signature rule", "preprocessing_signature")),
    ):
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path}:{needle}")

    if missing:
        return _item(
            requirement="accepted artifacts require hashed preprocessing signatures",
            evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="accepted artifacts require hashed preprocessing signatures",
        evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
        status="met",
        details="queue contract, artifact gate, and templates require sha256 preprocessing_signature",
    )


def _sha_provenance_contract_item(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
    artifact_scaffold_path: Path = Path("scripts/uxfd_artifact_scaffold.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    if not queue_path.exists():
        missing.append(str(queue_path))
        queue_contract: Mapping[str, Any] = {}
    else:
        queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
        queue_contract = queue.get("accepted_artifact_contract", {})

    sha_text = str(queue_contract.get("sha_provenance", "")).lower()
    for marker in ("dirty", "modified", "unknown", "uncommitted"):
        if marker not in sha_text:
            missing.append(f"accepted_artifact_contract.sha_provenance.{marker}")

    for path, needles in (
        (artifact_gate_path, SHA_PROVENANCE_NEEDLES),
        (artifact_scaffold_path, ("Provenance rule", "git_sha_or_submodule_sha")),
    ):
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path}:{needle}")

    if missing:
        return _item(
            requirement="accepted artifacts require clean SHA provenance",
            evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="accepted artifacts require clean SHA provenance",
        evidence=f"{queue_path},{artifact_gate_path},{artifact_scaffold_path}",
        status="met",
        details="queue contract, artifact gate, and templates reject dirty SHA provenance markers",
    )


def _accepted_run_root_activation_gate_item(
    accepted_run_root_readme: Path = ACCEPTED_RUN_ROOT_README,
    gpu_queue_path: Path = Path("scripts/uxfd_gpu_queue.py"),
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
    artifact_scaffold_path: Path = Path("scripts/uxfd_artifact_scaffold.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    for path, needles in (
        (accepted_run_root_readme, ACCEPTED_RUN_ROOT_GATE_NEEDLES),
        (artifact_scaffold_path, ACCEPTED_RUN_ROOT_GATE_NEEDLES),
        (
            gpu_queue_path,
            (
                "--require-preflight",
                "Blocked: static queue validation can_execute=False",
            ),
        ),
        (artifact_gate_path, ("require_queue_coverage", "queue coverage incomplete")),
    ):
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path}:{needle}")

    if missing:
        return _item(
            requirement="accepted-run evidence root requires GPU and queue preflight",
            evidence=(
                f"{accepted_run_root_readme},{gpu_queue_path},"
                f"{artifact_gate_path},{artifact_scaffold_path}"
            ),
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="accepted-run evidence root requires GPU and queue preflight",
        evidence=(
            f"{accepted_run_root_readme},{gpu_queue_path},"
            f"{artifact_gate_path},{artifact_scaffold_path}"
        ),
        status="met",
        details=(
            "accepted_runs root and templates require live GPU preflight, static "
            "queue gate clearance, and artifact gate queue coverage before promotion"
        ),
    )


def _sota_aggregate_activation_gate_item(
    template_readme: Path = SOTA_AGGREGATE_TEMPLATE_README,
    sota_scaffold_path: Path = Path("scripts/uxfd_sota_scaffold.py"),
    sota_gate_path: Path = Path("scripts/uxfd_sota_gate.py"),
    artifact_gate_path: Path = Path("scripts/uxfd_artifact_gate.py"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    for path, needles in (
        (template_readme, SOTA_AGGREGATE_ACTIVATION_NEEDLES),
        (sota_scaffold_path, SOTA_AGGREGATE_ACTIVATION_NEEDLES),
        (
            sota_gate_path,
            (
                "accepted_run_refs",
                "does not exist",
                "run_meta.yaml",
                "accepted_run_root",
            ),
        ),
        (artifact_gate_path, ("require_queue_coverage", "queue coverage incomplete")),
    ):
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path}:{needle}")

    if missing:
        return _item(
            requirement="SOTA aggregate activation requires accepted run coverage",
            evidence=(
                f"{template_readme},{sota_scaffold_path},"
                f"{sota_gate_path},{artifact_gate_path}"
            ),
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="SOTA aggregate activation requires accepted run coverage",
        evidence=(
            f"{template_readme},{sota_scaffold_path},"
            f"{sota_gate_path},{artifact_gate_path}"
        ),
        status="met",
        details=(
            "SOTA templates require artifact gate queue coverage before aggregate "
            "creation, and SOTA gate requires existing accepted run_meta refs"
        ),
    )


def _sota_comparison_contract_item(
    queue_path: Path = DEFAULT_QUEUE,
    runbook_path: Path = Path("paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md"),
) -> ObjectiveAuditItem:
    missing: List[str] = []
    if not queue_path.exists():
        missing.append(str(queue_path))
        queue: Mapping[str, Any] = {}
    else:
        queue = yaml.safe_load(queue_path.read_text(encoding="utf-8")) or {}

    contract = queue.get("sota_comparison_contract", {})
    if not isinstance(contract, Mapping):
        missing.append("sota_comparison_contract")
        contract = {}

    for field in SOTA_COMPARISON_CONTRACT_FIELDS:
        if not str(contract.get(field, "")).strip():
            missing.append(f"sota_comparison_contract.{field}")

    contract_text = " ".join(str(value) for value in contract.values()).lower()
    cross_gate = queue.get("cross_paper_gate", {})
    cross_gate_text = str(cross_gate.get("sota_rule", "")).lower()
    combined_text = f"{contract_text} {cross_gate_text}"
    for needle in SOTA_COMPARISON_CONTRACT_NEEDLES:
        if needle.lower() not in combined_text:
            missing.append(f"sota_comparison_contract.{needle}")
    if "multi-seed" not in cross_gate_text:
        missing.append("cross_paper_gate.sota_rule.multi-seed")

    if not runbook_path.exists():
        missing.append(str(runbook_path))
    else:
        runbook_text = runbook_path.read_text(encoding="utf-8").lower()
        for needle in (
            "single accepted run is only a run artifact",
            "accepted_run_refs",
            "run_meta.yaml",
            "matched seed set",
            "95% confidence interval",
            "effect size",
            "cannot be silently removed",
        ):
            if needle not in runbook_text:
                missing.append(f"{runbook_path}:{needle}")

    if missing:
        return _item(
            requirement="SOTA comparison requires multi-seed same-protocol aggregate evidence",
            evidence=f"{queue_path},{runbook_path}",
            status="not_met",
            details="missing=" + ",".join(missing),
        )

    return _item(
        requirement="SOTA comparison requires multi-seed same-protocol aggregate evidence",
        evidence=f"{queue_path},{runbook_path}",
        status="met",
        details=(
            "queue/runbook block single-run SOTA and require matched seeds, "
            "accepted run refs, aggregate statistics, failure records, and "
            "exact-vs-representative TOP scope"
        ),
    )


def _owner_review_decision_gate_item() -> ObjectiveAuditItem:
    report = evaluate_owner_review_gate()
    return _item(
        requirement="submodule owner-review decision gate",
        evidence=report.source_path,
        status="met" if report.ready else "not_met",
        details=(
            f"ready={report.ready}, pending_records={report.pending_records}, "
            f"blockers={len(report.blockers)}"
        ),
    )


def _subagent_execution_item(
    team_dir: Path = CLAUDE_TEAM_DIR,
    launch_blocked: bool = False,
) -> ObjectiveAuditItem:
    launch_path = team_dir / CODEX_SUBAGENT_LAUNCH
    output_paths = tuple(team_dir / filename for filename in CLAUDE_TEAM_OUTPUTS)
    outputs_ready = all(path.exists() for path in output_paths)
    if not launch_path.exists():
        return _item(
            requirement="six xhigh/subagent or Claude Team execution evidence",
            evidence=team_dir,
            status="blocked" if launch_blocked else "unverified",
            details=(
                "local subagent launch log missing; launch log records policy block"
                if launch_blocked
                else "local subagent launch log missing"
            ),
        )

    text = launch_path.read_text(encoding="utf-8")
    subagent_ids = set(SUBAGENT_ID_PATTERN.findall(text))
    has_xhigh_marker = "reasoning_effort=xhigh" in text
    evidence_ready = len(subagent_ids) == 6 and has_xhigh_marker and outputs_ready
    if evidence_ready:
        return _item(
            requirement="six xhigh/subagent or Claude Team execution evidence",
            evidence=team_dir,
            status="met",
            details="subagents=6, xhigh=True, deliverables=3",
        )

    missing_outputs = [path.name for path in output_paths if not path.exists()]
    return _item(
        requirement="six xhigh/subagent or Claude Team execution evidence",
        evidence=team_dir,
        status="blocked" if launch_blocked else "unverified",
        details=(
            f"subagents={len(subagent_ids)}, xhigh={has_xhigh_marker}, "
            f"missing_deliverables={','.join(missing_outputs) or 'none'}"
        ),
    )


def evaluate_objective_audit(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> ObjectiveAuditReport:
    objective = (
        "Execute the UXFD seven-paper goal package, use Spec Kit/Claude Team/"
        "handoff workflow, maintain TOP recent-work and 2x4090 constraints, and "
        "drive all seven papers toward IEEE Transactions submission readiness."
    )
    items: List[ObjectiveAuditItem] = []

    for filename in REQUIRED_GOAL_FILES:
        items.append(_exists_item(f"named goal file {filename}", GOAL_DIR / filename))

    for filename in REQUIRED_SPEC_FILES:
        items.append(_exists_item(f"Spec Kit artifact {filename}", SPEC_DIR / filename))

    items.append(_exists_item("handoff document", HANDOFF_PATH))
    items.append(_exists_item("continuation handoff document", CONTINUATION_HANDOFF_PATH))
    items.append(_exists_item("execution gate handoff document", EXECUTION_GATE_HANDOFF_PATH))
    items.append(
        _exists_item("latest continuation handoff document", LATEST_CONTINUATION_HANDOFF_PATH)
    )
    items.append(_exists_item("Claude Team task spec", CLAUDE_TEAM_DIR / "TASK_SPEC.md"))
    items.append(_exists_item("Claude Team launch log", CLAUDE_TEAM_DIR / "LAUNCH_LOG.md"))
    items.append(
        _exists_item(
            "Codex xhigh subagent launch log",
            CLAUDE_TEAM_DIR / CODEX_SUBAGENT_LAUNCH,
        )
    )

    launch_log = CLAUDE_TEAM_DIR / "LAUNCH_LOG.md"
    launch_blocked = _text_contains(launch_log, "Prepared but not launched") or _text_contains(
        launch_log, "rejected by policy"
    )
    items.append(_subagent_execution_item(launch_blocked=launch_blocked))
    for filename in CLAUDE_TEAM_OUTPUTS:
        path = CLAUDE_TEAM_DIR / filename
        items.append(
            _item(
                requirement=f"Claude Team deliverable {filename}",
                evidence=path,
                status="not_met" if not path.exists() else "met",
                details=(
                    "missing because team launch is blocked and local subagent synthesis is absent"
                    if not path.exists()
                    else "exists"
                ),
            )
        )

    for requirement, path in EXECUTION_ARTIFACTS:
        items.append(_exists_item(requirement, path))
    items.append(_launch_scripts_static_gate_item())
    items.append(_finite_metrics_contract_item(queue_path=queue_path))
    items.append(_source_tree_status_contract_item(queue_path=queue_path))
    items.append(_run_control_contract_item(queue_path=queue_path))
    items.append(_runtime_contract_item(queue_path=queue_path))
    items.append(_precision_contract_item(queue_path=queue_path))
    items.append(_evidence_level_contract_item(queue_path=queue_path))
    items.append(_preprocessing_signature_contract_item(queue_path=queue_path))
    items.append(_sha_provenance_contract_item(queue_path=queue_path))
    items.append(_accepted_run_root_activation_gate_item())
    items.append(_sota_aggregate_activation_gate_item())
    items.append(_sota_comparison_contract_item(queue_path=queue_path))
    items.append(_owner_review_decision_gate_item())

    paper07_rejection_ready = _text_contains(
        PAPER07_GOAL,
        PAPER07_REJECTION_NEEDLES[0],
    ) and all(
        _text_contains(PAPER07_GOAL, needle)
        for needle in PAPER07_REJECTION_NEEDLES[1:3]
    ) and all(
        _text_contains(PAPER07_REJECTION_CONTRACT, needle)
        for needle in PAPER07_REJECTION_NEEDLES[3:]
    ) and all(
        _text_contains(PAPER07_REVIEWER_TRACE, needle)
        for needle in PAPER07_REVIEWER_TRACE_NEEDLES
    )
    items.append(
        _item(
            requirement="Paper07 rejection-recovery innovation contract",
            evidence=f"{PAPER07_GOAL},{PAPER07_REJECTION_CONTRACT},{PAPER07_REVIEWER_TRACE}",
            status="met" if paper07_rejection_ready else "not_met",
            details=(
                "goal and submodule contract encode rejection recovery, DSOA v2, "
                "reviewer traceability, Q0 preflight, and non-SOTA/non-ready stop rules"
                if paper07_rejection_ready
                else "missing required Paper07 rejection-recovery goal or contract phrases"
            ),
        )
    )

    items.append(_paper_submodule_cleanliness_item())
    items.append(_parent_goal_checkpoint_item())

    submission = evaluate_submission_gate(queue_path=queue_path, artifact_root=artifact_root)
    recent = evaluate_recent_work_gate(queue_path=queue_path)

    items.append(
        _item(
            requirement="seven paper-local baseline/ablation matrices",
            evidence="submission_prep/baseline_ablation_matrix.yaml",
            status="met" if len(submission.papers) == 7 else "not_met",
            details=f"{len(submission.papers)} matrices discovered by submission gate",
        )
    )
    for paper in submission.papers:
        matrix_ready = paper.baselines >= 6 and paper.ablations >= 6
        items.append(
            _item(
                requirement=f"{paper.paper_id}: 6+ baselines and 6+ ablations",
                evidence=paper.matrix_path,
                status="met" if matrix_ready else "not_met",
                details=(
                    f"baselines={paper.baselines}, ablations={paper.ablations}, "
                    f"submission_ready={paper.submission_ready}"
                ),
            )
        )
        items.append(
            _item(
                requirement=f"{paper.paper_id}: IEEE Transactions submission-ready",
                evidence=paper.matrix_path,
                status="not_met" if not paper.submission_ready else "met",
                details=f"strict blockers remaining={len(paper.strict_blockers)}",
            )
        )

    items.append(
        _item(
            requirement="TOP recent-work policy",
            evidence=GOAL_DIR / "08_recent_work_citation_readme.md",
            status="met" if recent.policy_ready else "not_met",
            details=(
                f"accepted_pool_rows={recent.accepted_pool_rows}, "
                f"2026_ids={len(recent.top_2026_ids)}, "
                f"low_tier_violations={len(recent.low_tier_violations)}, "
                f"source_verification_ready={recent.source_verification_ready}"
            ),
        )
    )
    items.append(
        _item(
            requirement="low-tier source hygiene",
            evidence=Path("paper/UXFD_paper/results/low_tier_source_audit.md"),
            status="met" if submission.low_tier_source_ready else "not_met",
            details=(
                f"findings={submission.low_tier_source_findings}, "
                f"blockers={submission.low_tier_source_blocker_count}, "
                f"triage={submission.low_tier_source_triage_count}"
            ),
        )
    )
    items.append(
        _item(
            requirement="TOP representative accepted artifacts",
            evidence=GOAL_DIR / "09_gpu_execution_queue.yaml",
            status="met" if recent.evidence_ready else "not_met",
            details=f"pending_or_blocked_bindings={len(recent.evidence_blockers)}",
        )
    )
    queue_resource_reason = submission.queue_resource_reason.lower()
    queue_status = (
        "met"
        if submission.queue_can_execute
        else "blocked"
        if "blocked" in queue_resource_reason
        else "not_met"
    )
    items.append(
        _item(
            requirement="2x4090 GPU queue executable",
            evidence=queue_path,
            status=queue_status,
            details=submission.queue_resource_reason,
        )
    )
    items.append(
        _item(
            requirement="accepted run artifact metadata",
            evidence=artifact_root,
            status="met" if submission.artifact_gate_accepted else "not_met",
            details=(
                f"records={submission.artifact_gate_records}, "
                f"blockers={len(submission.artifact_gate_blockers)}"
            ),
        )
    )
    items.append(
        _item(
            requirement="cross-paper submission gate",
            evidence="scripts.uxfd_submission_gate",
            status="met" if submission.ready else "not_met",
            details=f"ready={submission.ready}, blockers={len(submission.blockers)}",
        )
    )

    blockers = tuple(
        f"{item.requirement}: {item.details}"
        for item in items
        if item.status != "met"
    )
    status_counts: Mapping[str, int] = {
        status: sum(1 for item in items if item.status == status)
        for status in ("met", "not_met", "blocked", "unverified")
    }
    achieved = not blockers
    return ObjectiveAuditReport(
        achieved=achieved,
        objective=objective,
        items=tuple(items),
        blockers=blockers,
        met=status_counts["met"],
        not_met=status_counts["not_met"],
        blocked=status_counts["blocked"],
        unverified=status_counts["unverified"],
    )


def build_payload(report: ObjectiveAuditReport) -> Mapping[str, Any]:
    return asdict(report)


def render_markdown(report: ObjectiveAuditReport) -> str:
    lines = [
        "# UXFD Objective Audit",
        "",
        f"- Achieved: `{report.achieved}`",
        f"- Met: `{report.met}`",
        f"- Not met: `{report.not_met}`",
        f"- Blocked: `{report.blocked}`",
        f"- Unverified: `{report.unverified}`",
        "",
        "## Objective",
        "",
        report.objective,
        "",
        "## Prompt-to-Artifact Checklist",
        "",
        "| Status | Requirement | Evidence | Details |",
        "|---|---|---|---|",
    ]
    for item in report.items:
        details = item.details.replace("|", "\\|")
        lines.append(
            f"| `{item.status}` | {item.requirement} | `{item.evidence}` | {details} |"
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.blockers:
        lines.append(f"- {blocker}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the UXFD active-thread objective")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-not-achieved", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_objective_audit(queue_path=args.queue, artifact_root=args.artifact_root)
    if args.format == "json":
        output = json.dumps(build_payload(report), indent=2) + "\n"
    else:
        output = render_markdown(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")

    if report.achieved or args.allow_not_achieved:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
