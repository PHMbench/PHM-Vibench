"""Documentation consistency checks (local, no network).

This module is intentionally lightweight and conservative: it validates that documentation
links resolve and that per-directory AI docs defer shared content to README.md.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml


SKIP_TOP_DIRS = {
    ".git",
    "__pycache__",
    "05_04",
    "data",
    "paper",  # paper workflows are not part of the core validation gate
    "results",
    "obsidian",  # local knowledge base / archive vault, not validated
}

SKIP_DIR_NAMES = {"__pycache__"}

FORBIDDEN_PHM_GENERATIVE_PATHS = {
    "docs/phm_generative": (
        "PHM generative module docs belong in owning module READMEs; "
        "process artifacts belong under the active specs/<feature>/ directory."
    ),
    "docs/generative": (
        "PHM generative module docs belong in owning module READMEs; "
        "process artifacts belong under the active specs/<feature>/ directory."
    ),
    "src/phm_factory": "Generative runtime must use existing PHM-Vibench factories.",
    "projects": "Do not create a parallel project tree for PHM generative work.",
    "projects/phm_generative": (
        "Process artifacts belong under specs/<active-feature>/, not projects/."
    ),
    "packs": "Do not create top-level packs; use module READMEs and specs/.",
    "templates": "Do not create top-level templates for PHM generative work.",
    "schemas": "Do not create top-level schemas for PHM generative work.",
}
PHM_GENERATIVE_LEGACY_DOC_REFERENCE_RE = re.compile(
    r"docs/(?:phm_generative|generative)(?:/|$|[^A-Za-z0-9_-])"
)
PHM_GENERATIVE_REFERENCE_INDEXES = (
    "configs/config_registry.csv",
    "docs/CONFIG_ATLAS.md",
    "docs/README.md",
)

REQUIRED_PHM_GENERATIVE_READMES = (
    "src/task_factory/task/generative/README.md",
    "src/model_factory/generative_model/README.md",
    "src/task_factory/Components/generative/README.md",
    "src/task_factory/Components/generative/losses/README.md",
    "src/task_factory/Components/generative/metrics/README.md",
    "src/task_factory/Components/generative/manifests/README.md",
    "src/task_factory/Components/generative/samplers/README.md",
    "src/data_factory/ID/README.md",
    "configs/paper/phm_generative/README.md",
    "scripts/README.md",
)

PHM_GENERATIVE_README_REQUIRED_SNIPPETS = {
    "src/task_factory/task/generative/README.md": (
        "python main.py --config <yaml>",
        "Future pipeline:",
        "src/Pipeline_06_generative.py",
        "src/model_factory/generative_model/",
        "src/task_factory/Components/generative/losses/",
        "fault_label",
        "domain_id",
        "domain_id -> load/rpm/system_id/sampling_rate",
        "domain_id,load,rpm,system_id,sampling_rate,description,dataset_name,notes",
        "domain_map_path",
        "domain_map_hash",
        "`load`, `rpm`, `system_id`, and `sampling_rate` are not direct V0 model condition keys",
        "python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only",
        "conda activate LQ_signal",
        "PHM\\text{-}GenBench",
        "small\\ verified\\ goal",
        "docs/materials goals",
        "demo-only goals",
        "runtime goals",
        "paperpack goals",
        "research-only goals",
        "GOAL-GEN-020",
        "Do not implement `GOAL-GEN-005+`",
        "environment / data / model / task / trainer",
    ),
    "src/model_factory/generative_model/README.md": (
        "Rectified Flow",
        "DDPM",
        "Score SDE",
        "Mamba/SSM",
        "MeanFlow",
        "Drifting Models",
        "reference_only: true",
        "copy_code_allowed: false",
        "paper_reference",
        "code_reference",
        "implementation_language",
        "license_status",
        "code_uncertain",
        "verify license before copying",
        "Mamba/SSM is a backbone, not a generative loss",
        "Do not carry hidden cache",
        "probability-flow time",
        "stateless",
    ),
    "src/task_factory/Components/generative/README.md": (
        "src/task_factory/Components/generative/losses/",
        "src/task_factory/Components/generative/samplers/",
        "src/task_factory/Components/generative/metrics/",
        "src/task_factory/Components/generative/manifests/",
        "eval-only",
        "benchmark-valid",
    ),
    "src/task_factory/Components/generative/losses/README.md": (
        "Conditional Flow Matching",
        "src/task_factory/Components/generative/losses/flow_matching.py",
        "src/task_factory/Components/generative/losses/rectified_flow.py",
        "src/task_factory/Components/generative/losses/ddpm.py",
        "src/task_factory/Components/generative/losses/score_sde.py",
        "[N, C, L]",
        "x_t = (1-t)z + tx_1",
        "u_t = x_1 - z",
        "fault_label: [N]",
        "domain_id: [N]",
        "eval-only",
    ),
    "src/task_factory/Components/generative/metrics/README.md": (
        "evaluation-only",
        "TSTR/TRTS",
        "missing status and reason",
        "tables/table_quality_mean_std.csv",
        "tables/table_utility_mean_std.csv",
        "tables/table_efficiency_mean_std.csv",
        "tables/table_leakage.csv",
        "tables/table_ablation.csv",
        "figure_sources/spectra_overlay.csv",
        "figure_sources/temporal_overlay.csv",
        "figure_sources/metric_barplot.csv",
        "figure_sources/dataset_method_heatmap.csv",
        "figure_sources/missing_metric_audit.csv",
        "figure_sources/manifest_index.json",
        "synthetic manifest paths",
        "metric source paths",
        "appendix/run_index.csv",
        "appendix/manifest_completeness.csv",
        "appendix/missing_metrics.csv",
        "appendix/missing_metrics.md",
    ),
    "src/task_factory/Components/generative/manifests/README.md": (
        "benchmark-valid",
        "source split is `train`",
        "condition counts for `fault_label` and `domain_id`",
        "domain map path and hash",
        "fault_label",
        "domain_id",
        "standardization",
        "robust_scaler",
        "MinMaxScaler is not allowed",
        "params artifact",
        "params hash",
        "metric status and missing reasons",
        "target_test",
    ),
    "src/task_factory/Components/generative/samplers/README.md": (
        "[N, C, L]",
        "fault_label",
        "domain_id",
        "finite values, shape, dtype, and device",
        "no hidden CPU fallback",
    ),
    "src/data_factory/ID/README.md": (
        "fault_label",
        "domain_id",
        "load",
        "rpm",
        "domain map",
    ),
    "configs/paper/phm_generative/README.md": (
        "python main.py --config <yaml>",
        "six_dataset_benchmark_matrix.yaml",
        "GPU 6",
        "GPU 7",
        "conda activate LQ_signal",
        "CUDA_VISIBLE_DEVICES=6 python -c",
        "CUDA_VISIBLE_DEVICES=7 python -c",
        "blocked_run_status_ledger.csv",
        "no CPU fallback",
    ),
    "scripts/README.md": (
        "scripts.validate_docs",
        "docs/phm_generative/",
        "docs/generative/",
        "tables/table_quality_mean_std.csv",
        "tables/table_utility_mean_std.csv",
        "tables/table_efficiency_mean_std.csv",
        "tables/table_leakage.csv",
        "tables/table_ablation.csv",
        "figure_sources/spectra_overlay.csv",
        "figure_sources/temporal_overlay.csv",
        "figure_sources/metric_barplot.csv",
        "figure_sources/dataset_method_heatmap.csv",
        "figure_sources/missing_metric_audit.csv",
        "six_dataset_benchmark_matrix.yaml",
        "gpu_preflight_report.json",
        "blocked_run_status_ledger.csv",
        "conda activate LQ_signal",
        "CUDA_VISIBLE_DEVICES=6",
        "CUDA_VISIBLE_DEVICES=7",
        "appendix/run_index.csv",
        "appendix/manifest_completeness.csv",
        "appendix/missing_metrics.csv",
        "appendix/missing_metrics.md",
        "SUBMISSION_READY",
        "NOT_SUBMISSION_READY",
    ),
}

V2_GOAL_REQUIRED_SECTIONS = (
    "## Goal ID",
    "## Objective",
    "## Scope",
    "## Required Behavior",
    "## Acceptance Criteria",
    "## Validation Commands",
)
V2_GOAL_LEGACY_DOC_TARGETS = (
    "docs/phm_generative/",
    "docs/generative/",
)
V2_GOAL_LEGACY_DOC_PROHIBITION_HINTS = (
    "do not",
    "must not",
    "forbid",
    "forbidden",
    "avoid",
    "without",
    "not create",
    "not used",
    "test ! -e",
    "structure_violation",
)
CORE_EXPECTED_GOAL_FILES = (
    "GOAL-GEN-000-repo-native-doc-pack.md",
    "GOAL-GEN-001-domain-id-contract.md",
    "GOAL-GEN-002-task-components-loss-spec.md",
    "GOAL-GEN-003-codex-claude-handoff.md",
    "GOAL-GEN-004-frontier-reference-map.md",
    "GOAL-GEN-M1-REPO-NATIVE.md",
)
CORE_GOAL_REQUIRED_SNIPPETS = {
    "GOAL-GEN-000-repo-native-doc-pack.md": ("README", "src/task_factory"),
    "GOAL-GEN-001-domain-id-contract.md": ("README", "domain_id"),
    "GOAL-GEN-002-task-components-loss-spec.md": ("README", "losses"),
    "GOAL-GEN-003-codex-claude-handoff.md": (
        "specs/<active-feature>",
        "README",
        "Subagent/teammate acceleration",
    ),
    "GOAL-GEN-004-frontier-reference-map.md": ("README", "reference"),
    "GOAL-GEN-M1-REPO-NATIVE.md": (
        "README",
        "specs/<active-feature>",
        "Subagent/teammate acceleration",
    ),
}

CLAUDE_TEAM_REQUIRED_FILES = (
    "TASK_SPEC.md",
    "report.md",
    "risks.md",
    "test-log.md",
)
CLAUDE_TASK_SPEC_REQUIRED_SNIPPETS = (
    "Read-only `review` mode",
    "Edits are not allowed",
    "Do not push",
    "delete",
    "read secrets",
    "report.md",
    "risks.md",
    "test-log.md",
)
CLAUDE_REVIEW_ALLOWED_DECISIONS = {"APPROVE", "REQUEST_CHANGES", "BLOCKING"}
CLAUDE_REVIEW_REQUIRED_TAGS = (
    "BLOCKING_ISSUES",
    "NON_BLOCKING_ISSUES",
    "FIX_INSTRUCTION",
)
GOAL_GEN_003_GOAL = ".specify/goals/v2/GOAL-GEN-003-codex-claude-handoff.md"
GOAL_GEN_003_REVIEW_README = (
    "specs/002-phm-genbench-frontier/reviews/README.md"
)
GOAL_GEN_003_REVIEW_TEMPLATE = (
    "specs/002-phm-genbench-frontier/reviews/claude-team/"
    "phm-gen-general-review-template/TASK_SPEC.md"
)
GOAL_GEN_003_HANDOFF_README = (
    "specs/002-phm-genbench-frontier/handoffs/README.md"
)
GOAL_GEN_003_REVIEW_REQUIRED_SNIPPETS = (
    "phm-gen-architect",
    "phm-gen-loss-reviewer",
    "phm-gen-leakage-reviewer",
    "src/task_factory/task/generative/README.md",
    "src/task_factory/Components/generative/README.md",
    "src/task_factory/Components/generative/losses/README.md",
    "src/task_factory/Components/generative/manifests/README.md",
    "src/model_factory/generative_model/README.md",
    "REVIEW_DECISION",
    "BLOCKING_ISSUES",
    "NON_BLOCKING_ISSUES",
    "FIX_INSTRUCTION",
)
GOAL_GEN_003_TEMPLATE_REQUIRED_SNIPPETS = (
    "Read-only `review` mode",
    "Edits are not allowed",
    "phm-gen-architect",
    "phm-gen-loss-reviewer",
    "phm-gen-leakage-reviewer",
    "CFM target remains `x1 - z`",
    "`fault_label` and `domain_id` are the only default model condition keys",
    "`load` and `rpm` remain domain-map metadata",
    "eval-only in V0",
    "MeanFlow and Drifting remain research-only",
    "No output is trusted without Codex verification",
    "report.md",
    "risks.md",
    "test-log.md",
    "BLOCKED_NOT_RUN",
)
GOAL_GEN_003_HANDOFF_README_REQUIRED_SNIPPETS = (
    "Goal ID",
    "Objective",
    "Files changed",
    "Runtime behavior changed: yes/no",
    "Contracts touched",
    "Validation commands run",
    "Validation results",
    "Known risks",
    "Required reviewers",
    "Required context files",
    "Review output format",
    "verified results from blocked work",
    "REVIEW_DECISION",
    "FIX_INSTRUCTION",
)

M2_REVIEW_HANDOFF_GOAL = ".specify/goals/v2/GOAL-GEN-M2-006-review-handoff.md"
M2_SPECKIT_FREEZE_GOAL = ".specify/goals/v2/GOAL-GEN-M2-000-speckit-freeze.md"
M2_SIX_DATASET_MATRIX_GOAL = (
    ".specify/goals/v2/GOAL-GEN-M2-001-six-dataset-matrix-gpu.md"
)
M2_PAPER_DRAFT_GOAL = ".specify/goals/v2/GOAL-GEN-M2-005-markdown-paper-draft.md"
M2_REAL_RUNS_GOAL = ".specify/goals/v2/GOAL-GEN-M2-003-real-runs-evidence.md"
M2_FEATURE_DIR = "specs/002-phm-genbench-frontier"
M2_SIX_DATASET_MATRIX = "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml"
M2_DRY_RUN_PLAN = (
    "results/paper/phm_generative/six_dataset_submission_v1/"
    "dry_run_current_audit/run_plan.csv"
)
M2_EXPECTED_GOAL_FILES = (
    "GOAL-GEN-M2-000-speckit-freeze.md",
    "GOAL-GEN-M2-001-six-dataset-matrix-gpu.md",
    "GOAL-GEN-M2-002-multidataset-aggregation.md",
    "GOAL-GEN-M2-003-real-runs-evidence.md",
    "GOAL-GEN-M2-004-figures-tables.md",
    "GOAL-GEN-M2-005-markdown-paper-draft.md",
    "GOAL-GEN-M2-006-review-handoff.md",
)
M2_REAL_RUNS_GOAL_REQUIRED_SNIPPETS = (
    "conda activate LQ_signal",
    "CUDA_VISIBLE_DEVICES=6 python -c",
    "CUDA_VISIBLE_DEVICES=7 python -c",
    "--preflight-gpu",
    "--dry-run",
    "--output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight",
    "--execute",
    "--stages train",
    "--stages sample",
    "--stages eval",
    "--stages paperpack",
    "--from-runs results/paper/phm_generative/six_dataset_submission_v1/runs",
    "torch.cuda.is_available() == True",
    "one visible device",
    "BLOCKED_GPU_PREFLIGHT",
    "Do not mark this goal complete",
    "do not reroute to CPU",
)
M2_REVIEW_HANDOFF_GOAL_REQUIRED_SNIPPETS = (
    "Claude Code Teams",
    "read-only review mode",
    "unapproved external service",
    "blocked review",
    "Codex verification",
    "Codex must inspect reports",
    "Claude teammates are advisory reviewers",
    "Each teammate receives a bounded scope",
    "Subagent/teammate acceleration",
    "specs/002-phm-genbench-frontier/",
)
M2_DOWNSTREAM_GOAL_REQUIRED_SNIPPETS = {
    "GOAL-GEN-M2-002-multidataset-aggregation.md": (
        "This goal's real-evidence completion is task `T048`",
        "depends on `T047`",
        "fixture aggregation is not a substitute",
    ),
    "GOAL-GEN-M2-004-figures-tables.md": (
        "This goal's final-evidence completion is task `T049`",
        "depends on `T048`",
        "must not be used as final paper table or figure evidence",
    ),
    "GOAL-GEN-M2-005-markdown-paper-draft.md": (
        "This goal's submission-ready completion is task `T050`",
        "depends on `T049`",
        "not completion of the submission-ready objective",
    ),
    "GOAL-GEN-M2-006-review-handoff.md": (
        "This goal's final review completion is task `T051`",
        "depends on `T050`",
        "not final approval",
    ),
}
M2_SPECKIT_REQUIRED_FILES = (
    "spec.md",
    "plan.md",
    "tasks.md",
    "research.md",
    "data-model.md",
    "quickstart.md",
    "contracts/generative-benchmark-contract.md",
    "checklists/requirements.md",
    "checklists/benchmark-readiness.md",
    "analysis/m2-cross-artifact-analysis.md",
    "m2/README.md",
    "m2/goals.md",
)
ACTIVE_FEATURE_FORBIDDEN_SNIPPETS = (
    "GOAL-FFU",
)
ACTIVE_FEATURE_QUICKSTART_REQUIRED_SNIPPETS = (
    "LQ_signal",
    "torchmetrics",
    "Feature_factory-update",
    "branch-name caveat",
    "not as M2 evidence completion",
)
M2_ANALYSIS_REQUIRED_SNIPPETS = (
    "M2-002 Aggregation Contract Impact",
    "M2-004 figures/tables",
    "M2-005 paper draft",
    "benchmark_effect_summary.csv",
    "benchmark_effect_report.md",
    "benchmark_effect_manifest.json",
    "missing_metrics.md",
    "dataset / method / metric",
    "manifest_paths",
    "metric_source_paths",
    "baseline_method",
    "configured_dataset_count",
    "observed_datasets",
    "observed_configured_datasets",
    "observed_configured_dataset_count",
    "missing_datasets",
    "unexpected_datasets",
    "min_datasets",
    "min_datasets_met",
    "input_gaps",
    "SUBMISSION_READY",
    "Matrix-external datasets cannot satisfy the six-dataset paper claim",
)
M2_GPU_PREFLIGHT_REPORT = (
    "specs/002-phm-genbench-frontier/reviews/codex/"
    "2026-05-12-gpu-preflight-report.json"
)
M2_GPU_PREFLIGHT_EXPECTED_SOURCE_REPORT = (
    "results/paper/phm_generative/six_dataset_submission_v1/"
    "gpu_preflight/gpu_preflight_report.json"
)
GOAL_GEN_COMPLETION_AUDIT = (
    ".specify/goals/v2/staus/COMPLETION-AUDIT-2026-05-16-GOAL-GEN.md"
)
GOAL_GEN_COMPLETION_AUDIT_REQUIRED_SNIPPETS = (
    "**Audit decision**: NOT COMPLETE",
    ".specify/goals/v2/GOAL-GEN-000-repo-native-doc-pack.md",
    ".specify/goals/v2/GOAL-GEN-001-domain-id-contract.md",
    ".specify/goals/v2/GOAL-GEN-002-task-components-loss-spec.md",
    ".specify/goals/v2/GOAL-GEN-003-codex-claude-handoff.md",
    ".specify/goals/v2/GOAL-GEN-004-frontier-reference-map.md",
    ".specify/goals/v2/GOAL-GEN-M1-REPO-NATIVE.md",
    ".specify/goals/v2/GOAL-GEN-M2-000-speckit-freeze.md",
    ".specify/goals/v2/GOAL-GEN-M2-001-six-dataset-matrix-gpu.md",
    ".specify/goals/v2/GOAL-GEN-M2-002-multidataset-aggregation.md",
    ".specify/goals/v2/GOAL-GEN-M2-003-real-runs-evidence.md",
    ".specify/goals/v2/GOAL-GEN-M2-004-figures-tables.md",
    ".specify/goals/v2/GOAL-GEN-M2-005-markdown-paper-draft.md",
    ".specify/goals/v2/GOAL-GEN-M2-006-review-handoff.md",
    "specs/002-phm-genbench-frontier/spec.md",
    "GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE",
    "results/paper/phm_generative/six_dataset_submission_v1/runs",
    "`T047`",
    "`T048`",
    "`T049`",
    "`T050`",
    "`T051`",
    "`NOT_SUBMISSION_READY`",
    'update_goal(status="complete")',
    "must not be called",
)
GOAL_GEN_STATUS_REPORT = ".specify/goals/v2/staus/STATUS-2026-05-16.md"
GOAL_GEN_STATUS_REPORT_REQUIRED_SNIPPETS = (
    "Subagent Acceleration Status",
    "SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-01-goal-status-consistency.md",
    "SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-02-gpu-run-evidence.md",
    "SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-03-paper-readiness.md",
    "SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-04-validation-guardrails.md",
    "SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-05-speckit-workflow.md",
    "SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-06-handoff-team-review.md",
    "SUBAGENT-SUMMARY-2026-05-16-GOAL-GEN-M2-STATUS.md",
    "COMPLETION-AUDIT-2026-05-16-GOAL-GEN.md",
    "GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE is blocked",
    "T047-T051",
    "NOT_SUBMISSION_READY",
)
M2_GPU_RUNBOOK = (
    "specs/002-phm-genbench-frontier/reviews/codex/"
    "2026-05-11-m2-gpu-runbook.md"
)
M2_RUN_STATUS_LEDGER = (
    "specs/002-phm-genbench-frontier/reviews/codex/"
    "2026-05-11-m2-run-status-ledger.csv"
)
M2_RUN_STATUS_LEDGER_MARKDOWN = (
    "specs/002-phm-genbench-frontier/reviews/codex/"
    "2026-05-11-m2-run-status-ledger.md"
)

M2_RUN_STATUS_REQUIRED_FIELDS = (
    "dataset",
    "dataset_name",
    "method",
    "method_label",
    "seed",
    "planned_stages",
    "status",
    "reason",
)
M2_RUN_STATUS_EXPECTED_DATASETS = frozenset(
    {
        "RM_001_CWRU",
        "RM_002_XJTU",
        "RM_003_FEMTO",
        "RM_008_UNSW",
        "RM_024_JUST",
        "RM_027_PU",
    }
)
M2_RUN_STATUS_EXPECTED_DATASET_NAMES = {
    "RM_001_CWRU": "CWRU",
    "RM_002_XJTU": "XJTU",
    "RM_003_FEMTO": "FEMTO",
    "RM_008_UNSW": "UNSW",
    "RM_024_JUST": "JUST",
    "RM_027_PU": "PU",
}
M2_RUN_STATUS_EXPECTED_METHODS = frozenset(
    {
        "cfm_grid",
        "ddpm_train_distribution",
        "rectified_flow_grid",
    }
)
M2_RUN_STATUS_EXPECTED_METHOD_LABELS = {
    "cfm_grid": "Conditional Flow Matching",
    "ddpm_train_distribution": "DDPM Epsilon",
    "rectified_flow_grid": "Rectified Flow",
}
M2_SIX_DATASET_EXPECTED_OUTPUT_DIR = (
    "results/paper/phm_generative/six_dataset_submission_v1"
)
M2_SIX_DATASET_EXPECTED_BASELINE = "cfm_grid"
M2_SIX_DATASET_EXPECTED_DATA_DIR = "/home/user/data/PHMbenchdata/PHM-Vibench"
M2_SIX_DATASET_EXPECTED_METADATA = "metadata.xlsx"
M2_SIX_DATASET_ALLOWED_CONDITION_POLICIES = frozenset({"grid", "train_distribution"})
M2_RUN_STATUS_EXPECTED_SEEDS = frozenset({"0", "1"})
M2_RUN_STATUS_EXPECTED_STAGES = "train;sample;eval;paperpack"
M2_RUN_STATUS_ALLOWED_TERMINAL_STATUSES = frozenset({"complete", "failed"})
M2_RUN_STATUS_LEDGER_MARKDOWN_REQUIRED_SNIPPETS = (
    "# M2 Run Status Ledger",
    "Machine-readable copy:",
    "2026-05-11-m2-run-status-ledger.csv",
    "BLOCKED_GPU_PREFLIGHT",
    "GPU 6",
    "GPU 7",
    "nvidia-smi",
    "train/sample/eval/paperpack",
    "results/paper/phm_generative/six_dataset_submission_v1/runs",
    "## Downstream Readiness",
    "Ready for M2-004 figures/tables: no.",
    "Ready for M2-005 paper draft: no.",
    "## Resume Rule",
)
M2_GPU_RUNBOOK_REQUIRED_SECTIONS = (
    "## Current Blocker",
    "## Resume Gates",
    "## Execution Sequence",
    "## Evidence Aggregation",
    "## Completion Rule",
)
M2_GPU_RUNBOOK_REQUIRED_SNIPPETS = (
    "GPU 6",
    "GPU 7",
    "conda activate LQ_signal",
    "CUDA_VISIBLE_DEVICES=6 python -c",
    "CUDA_VISIBLE_DEVICES=7 python -c",
    "CUDA_VISIBLE_DEVICES=6,7 python -c",
    "torch.cuda.is_available()",
    "`torch.cuda.device_count()` is exactly `1`",
    "env CUDA_VISIBLE_DEVICES=6",
    "env CUDA_VISIBLE_DEVICES=7",
    "trainer.device=cuda",
    "trainer.gpus=1",
    "144 commands",
    "--execute",
    "--preflight-gpu",
    "--stages train",
    "sample",
    "eval",
    "paperpack",
    "--from-runs",
    "Do not route the paper benchmark to CPU",
)

M2_PAPER_REQUIRED_FILES = (
    "PAPER_DRAFT.md",
    "evidence_gaps.md",
    "submission_readiness.md",
)
M2_PAPER_READY_EVIDENCE_FILES = (
    "results/paper/phm_generative/six_dataset_submission_v1/effect/"
    "benchmark_effect_summary.csv",
    "results/paper/phm_generative/six_dataset_submission_v1/effect/"
    "benchmark_effect_manifest.json",
)
M2_PAPER_READY_MIN_DATASETS = 6
M2_PAPER_DRAFT_REQUIRED_SECTIONS = (
    "## Abstract",
    "## Experimental Setting",
    "## Metrics",
    "## Results",
    "## Evidence And Reproducibility",
    "## Limitations",
)
M2_PAPER_DRAFT_REQUIRED_SNIPPETS = (
    "phm_genbench_six_dataset_submission_v1",
    "fault_label",
    "domain_id",
    "FFT and spectral calculations are evaluation-only evidence",
    "benchmark-valid quality and utility evidence",
    "evidence gaps",
)
M2_PAPER_NOT_READY_DRAFT_REQUIRED_SNIPPETS = (
    "not submission-ready",
    "No numerical claim",
    "No computable benchmark rows",
)
M2_PAPER_GAPS_REQUIRED_SNIPPETS = (
    "Summary:",
    "Manifest:",
    "Evidence gaps:",
)
M2_PAPER_READINESS_REQUIRED_SNIPPETS = (
    "Status:",
    "Promotion rule:",
    "SUBMISSION_READY",
    "source paths are traceable",
)
M2_PAPER_FORBIDDEN_PLACEHOLDER_RE = re.compile(
    r"\bTODO\b|\bTBD\b|<[^>\n]*placeholder[^>\n]*>|\{\{|\}\}",
    flags=re.IGNORECASE,
)

HANDOFF_REQUIRED_SECTIONS = (
    "## Current State",
    "## Goal ID",
    "## Objective",
    "## Files Changed",
    "## Runtime Behavior Changed",
    "## Contracts Touched",
    "## Validation Commands Run",
    "## Validation Results",
    "## Known Risks",
    "## Required Reviewers",
    "## Required Context Files",
    "## Review Output Format",
    "## Next Steps",
)

FEATURE_SPEC_REQUIRED_FR_IDS = tuple(f"FR-{index:03d}" for index in range(1, 16))
FEATURE_SPEC_REQUIRED_SC_IDS = tuple(f"SC-{index:03d}" for index in range(1, 9))
FEATURE_SPEC_REQUIRED_SNIPPETS = (
    "PHM-GenBench Frontier",
    "configuration-first execution",
    "5-block configs",
    "factory-first extension",
    "evidence-gated validity",
    "strict preflight validation",
    "condition sampling",
    "normalization parameter artifacts",
    "config/protocol hashes",
    "missing value status and reason",
    "table CSVs",
    "figure-source CSVs",
    "existing factories",
    "exploratory validity",
    "Claude Code Teams",
    "read-only plan/review mode",
    "subagent/teammate acceleration scopes",
    "handoff documents",
    "active Speckit feature directory",
    "README of the owning module",
    "MUST NOT accumulate a separate PHM generative docs tree",
    "Benchmark-effect aggregation",
    "configured, observed, missing, and unexpected dataset coverage",
    "At least five generative families",
    "Six-dataset readiness checks",
)

CONSTITUTION_REQUIRED_SNIPPETS = (
    "python main.py --config <yaml>",
    "5-block structure",
    "environment",
    "data",
    "model",
    "task",
    "trainer",
    "src/Pipeline_06_generative.py",
    "src/model_factory/generative_model/",
    "src/task_factory/task/generative/",
    "src/task_factory/Components/generative/",
    "src/phm_factory/",
    "benchmark-valid",
    "exploratory",
    "docs-only",
    "config hash",
    "protocol hash",
    "normalization artifact and hash",
    "condition counts",
    "source split",
    "leakage checks",
    "metric status/reason reporting",
    "FFT",
    "eval-only",
    "MUST NOT be silently introduced as training losses",
    "MeanFlow",
    "Drifting",
    "validity_status: exploratory",
    "Mamba and SSM modules are backbones, not losses",
    "stateless",
    "README next to the owning module or config directory",
    "active Speckit feature directory",
    "MUST NOT accumulate a separate PHM generative docs tree",
    "Constitution",
    "Specification",
    "Implementation plan",
    "Requirements-quality checklist",
    "Tasks",
    "Cross-artifact analysis",
    "Implementation",
    "python -m scripts.validate_docs",
    "python -m scripts.validate_configs",
    "python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only",
    (
        "python main.py --config configs/demo/10_generative/"
        "dummy_generative_cfm.yaml --preflight-only"
    ),
)

ROOT_PHM_GENBENCH_GUIDANCE_REQUIRED_SNIPPETS = {
    "AGENTS.md": (
        ".specify/memory/constitution.md",
        ".specify/goals/",
        "src/task_factory/task/generative/",
        "src/model_factory/generative_model/",
        "src/task_factory/Components/generative/",
    ),
    "CLAUDE.md": (
        ".specify/memory/constitution.md",
        "specs/002-phm-genbench-frontier/reviews/README.md",
        "specs/002-phm-genbench-frontier/handoffs/README.md",
        "src/Pipeline_06_generative.py",
        "python main.py --config <yaml>",
        "environment",
        "data",
        "model",
        "task",
        "trainer",
    ),
    "docs/README.md": (
        "canonical",
        "module-specific usage belongs in the module",
        "../src/task_factory/task/generative/README.md",
        "../src/model_factory/generative_model/README.md",
        "../src/task_factory/Components/generative/README.md",
    ),
}


@dataclass(frozen=True)
class Issue:
    kind: str
    path: str
    detail: str


def iter_doc_files(repo_root: Path) -> Iterable[Path]:
    names = {
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        "GEMINI.md",
        "API_REFERENCE.md",
    }
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        rel = root_path.relative_to(repo_root)
        if rel.parts and rel.parts[0] in SKIP_TOP_DIRS:
            dirs[:] = []
            continue
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIR_NAMES and (not rel.parts and d not in SKIP_TOP_DIRS or rel.parts)
        ]
        for name in files:
            if name in names:
                yield root_path / name


def strip_fenced_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def _contains_normalized(text: str, snippet: str) -> bool:
    haystack = re.sub(r"\s+", " ", text).casefold()
    needle = re.sub(r"\s+", " ", snippet).casefold()
    return needle in haystack


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
    for path in iter_doc_files(repo_root):
        if path.name not in {"CLAUDE.md", "AGENTS.md", "GEMINI.md"}:
            continue
        rel = path.relative_to(repo_root)
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


def check_phm_generative_docs_placement(repo_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    for rel, detail in FORBIDDEN_PHM_GENERATIVE_PATHS.items():
        if (repo_root / rel).exists():
            issues.append(
                Issue(
                    kind="forbidden_phm_generative_path",
                    path=rel,
                    detail=detail,
                )
            )
    for rel in PHM_GENERATIVE_REFERENCE_INDEXES:
        path = repo_root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = PHM_GENERATIVE_LEGACY_DOC_REFERENCE_RE.search(text)
        if match is not None:
            issues.append(
                Issue(
                    kind="legacy_phm_generative_doc_reference",
                    path=rel,
                    detail=(
                        f"{match.group(0)} must not be referenced from maintained "
                        "indexes; use module READMEs or the active feature spec."
                    ),
                )
            )
    return issues


def check_required_phm_generative_readmes(repo_root: Path) -> list[Issue]:
    if not (repo_root / ".specify" / "goals" / "v2").exists():
        return []

    issues: list[Issue] = []
    for rel in REQUIRED_PHM_GENERATIVE_READMES:
        if not (repo_root / rel).is_file():
            issues.append(
                Issue(
                    kind="missing_phm_generative_module_readme",
                    path=rel,
                    detail=(
                        "PHM generative docs must live in owning module/config "
                        "README files, not in a central docs tree."
                    ),
                )
            )
    return issues


def check_required_phm_generative_readme_content(repo_root: Path) -> list[Issue]:
    if not (repo_root / ".specify" / "goals" / "v2").exists():
        return []

    issues: list[Issue] = []
    for rel, snippets in PHM_GENERATIVE_README_REQUIRED_SNIPPETS.items():
        path = repo_root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in snippets:
            if not _contains_normalized(text, snippet):
                issues.append(
                    Issue(
                        kind="missing_phm_generative_readme_contract_text",
                        path=rel,
                        detail=snippet,
                    )
                )
    return issues


def _section_value_after(lines: list[str], section: str) -> Optional[str]:
    for i, line in enumerate(lines):
        if line.strip() != section:
            continue
        for value in lines[i + 1 :]:
            value = value.strip()
            if value:
                return value
        return None
    return None


def _markdown_section_lines(lines: list[str], section: str) -> list[str]:
    for i, line in enumerate(lines):
        if line.strip() != section:
            continue
        section_lines: list[str] = []
        for value in lines[i + 1 :]:
            if value.startswith("## "):
                break
            section_lines.append(value)
        return section_lines
    return []


def _is_legacy_doc_prohibition(lines: list[str], index: int) -> bool:
    context = "\n".join(lines[max(0, index - 1) : index + 1]).lower()
    return any(hint in context for hint in V2_GOAL_LEGACY_DOC_PROHIBITION_HINTS)


def check_v2_goal_contracts(repo_root: Path) -> list[Issue]:
    goal_dir = repo_root / ".specify" / "goals" / "v2"
    if not goal_dir.exists():
        return []

    issues: list[Issue] = []
    for path in sorted(goal_dir.glob("GOAL-GEN*.md")):
        rel = str(path.relative_to(repo_root))
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for section in V2_GOAL_REQUIRED_SECTIONS:
            if section not in text:
                issues.append(
                    Issue(
                        kind="v2_goal_missing_section",
                        path=rel,
                        detail=section,
                    )
                )
        goal_id = _section_value_after(lines, "## Goal ID")
        if not goal_id or not goal_id.startswith("GOAL-GEN"):
            issues.append(
                Issue(
                    kind="v2_goal_invalid_goal_id",
                    path=rel,
                    detail="Expected non-empty GOAL-GEN* value after ## Goal ID",
                )
            )
        elif not path.stem.upper().startswith(goal_id.upper()):
            issues.append(
                Issue(
                    kind="v2_goal_id_filename_mismatch",
                    path=rel,
                    detail=f"Goal ID {goal_id} does not match filename {path.name}",
                )
            )
        scope_lines = _markdown_section_lines(lines, "## Scope")
        for index, line in enumerate(scope_lines):
            for target in V2_GOAL_LEGACY_DOC_TARGETS:
                if target not in line or _is_legacy_doc_prohibition(scope_lines, index):
                    continue
                issues.append(
                    Issue(
                        kind="v2_goal_legacy_docs_allowed_target",
                        path=rel,
                        detail=(
                            f"{target} must not be listed as an allowed goal "
                            "target; use owning module READMEs or the active "
                            "spec feature directory."
                        ),
                    )
                )
    return issues


def check_v2_core_goal_queue(repo_root: Path) -> list[Issue]:
    goal_dir = repo_root / ".specify" / "goals" / "v2"
    if not goal_dir.exists():
        return []

    has_core_goal = any((goal_dir / filename).exists() for filename in CORE_EXPECTED_GOAL_FILES)
    if not has_core_goal:
        return []

    issues: list[Issue] = []
    for filename in CORE_EXPECTED_GOAL_FILES:
        path = goal_dir / filename
        rel = str(path.relative_to(repo_root))
        if not path.is_file():
            issues.append(
                Issue(
                    kind="missing_core_goal_file",
                    path=rel,
                    detail="Core GOAL-GEN queue requires 000 through 004 and M1.",
                )
            )
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in CORE_GOAL_REQUIRED_SNIPPETS[filename]:
            if snippet not in text:
                issues.append(
                    Issue(
                        kind="core_goal_missing_required_scope_text",
                        path=rel,
                        detail=snippet,
                    )
                )
    return issues


def check_goal_gen_003_review_templates(repo_root: Path) -> list[Issue]:
    if not (repo_root / GOAL_GEN_003_GOAL).exists():
        return []

    checks = {
        GOAL_GEN_003_REVIEW_README: GOAL_GEN_003_REVIEW_REQUIRED_SNIPPETS,
        GOAL_GEN_003_REVIEW_TEMPLATE: GOAL_GEN_003_TEMPLATE_REQUIRED_SNIPPETS,
        GOAL_GEN_003_HANDOFF_README: GOAL_GEN_003_HANDOFF_README_REQUIRED_SNIPPETS,
    }
    issues: list[Issue] = []
    for rel, snippets in checks.items():
        path = repo_root / rel
        if not path.is_file():
            issues.append(
                Issue(
                    kind="missing_goal_gen_003_review_template_artifact",
                    path=rel,
                    detail="GOAL-GEN-003 requires this feature-scoped artifact.",
                )
            )
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in snippets:
            if not _contains_normalized(text, snippet):
                issues.append(
                    Issue(
                        kind="goal_gen_003_review_template_missing_text",
                        path=rel,
                        detail=snippet,
                    )
                )
    return issues


def check_v2_m2_goal_queue(repo_root: Path) -> list[Issue]:
    goal_dir = repo_root / ".specify" / "goals" / "v2"
    if not goal_dir.exists():
        return []

    existing_m2_goals = sorted(goal_dir.glob("GOAL-GEN-M2-*.md"))
    if not existing_m2_goals:
        return []

    issues: list[Issue] = []
    for filename in M2_EXPECTED_GOAL_FILES:
        path = goal_dir / filename
        if not path.is_file():
            issues.append(
                Issue(
                    kind="missing_m2_goal_file",
                    path=str(path.relative_to(repo_root)),
                    detail="M2 queue requires GOAL-GEN-M2-000 through M2-006.",
                )
            )

    for path in existing_m2_goals:
        rel = str(path.relative_to(repo_root))
        text = path.read_text(encoding="utf-8", errors="ignore")
        if M2_FEATURE_DIR not in text:
            issues.append(
                Issue(
                    kind="m2_goal_missing_active_feature_reference",
                    path=rel,
                    detail=f"Expected explicit {M2_FEATURE_DIR} reference.",
                )
            )

    review_goal = goal_dir / "GOAL-GEN-M2-006-review-handoff.md"
    if review_goal.is_file():
        review_text = review_goal.read_text(encoding="utf-8", errors="ignore")
        for snippet in M2_REVIEW_HANDOFF_GOAL_REQUIRED_SNIPPETS:
            if not _contains_normalized(review_text, snippet):
                issues.append(
                    Issue(
                        kind="m2_review_goal_missing_claude_team_contract_text",
                        path=str(review_goal.relative_to(repo_root)),
                        detail=snippet,
                    )
                )
    for filename, snippets in M2_DOWNSTREAM_GOAL_REQUIRED_SNIPPETS.items():
        path = goal_dir / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in snippets:
            if not _contains_normalized(text, snippet):
                issues.append(
                    Issue(
                        kind="m2_downstream_goal_missing_dependency_text",
                        path=str(path.relative_to(repo_root)),
                        detail=snippet,
                    )
                )
    return issues


def check_feature_m2_real_runs_goal_contract(repo_root: Path) -> list[Issue]:
    path = repo_root / M2_REAL_RUNS_GOAL
    if not path.is_file():
        return []

    rel = str(path.relative_to(repo_root))
    text = path.read_text(encoding="utf-8", errors="ignore")
    issues: list[Issue] = []
    for snippet in M2_REAL_RUNS_GOAL_REQUIRED_SNIPPETS:
        if not _contains_normalized(text, snippet):
            issues.append(
                Issue(
                    kind="m2_real_runs_goal_missing_required_text",
                    path=rel,
                    detail=snippet,
                )
            )
    return issues


def check_feature_speckit_artifacts(repo_root: Path) -> list[Issue]:
    if not (repo_root / M2_SPECKIT_FREEZE_GOAL).exists():
        return []

    issues: list[Issue] = []
    feature_dir = repo_root / M2_FEATURE_DIR
    if not feature_dir.exists():
        return [
            Issue(
                kind="missing_m2_feature_dir",
                path=M2_FEATURE_DIR,
                detail="M2 Speckit freeze requires the active feature directory.",
            )
        ]

    for rel in M2_SPECKIT_REQUIRED_FILES:
        path = feature_dir / rel
        if not path.is_file():
            issues.append(
                Issue(
                    kind="missing_m2_speckit_artifact",
                    path=str(path.relative_to(repo_root)),
                    detail="M2 Speckit freeze requires this feature artifact.",
                )
            )
            continue
        if rel.startswith(("reviews/", "handoffs/")):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in ACTIVE_FEATURE_FORBIDDEN_SNIPPETS:
            if snippet in text:
                issues.append(
                    Issue(
                        kind="active_feature_artifact_forbidden_legacy_goal_text",
                        path=str(path.relative_to(repo_root)),
                        detail=snippet,
                    )
                )
        if rel == "quickstart.md":
            for snippet in ACTIVE_FEATURE_QUICKSTART_REQUIRED_SNIPPETS:
                if not _contains_normalized(text, snippet):
                    issues.append(
                        Issue(
                            kind="m2_quickstart_missing_execution_caveat",
                            path=str(path.relative_to(repo_root)),
                            detail=snippet,
                        )
                    )

    tasks_path = feature_dir / "tasks.md"
    preflight_report = repo_root / M2_GPU_PREFLIGHT_REPORT
    if tasks_path.is_file() and preflight_report.is_file():
        try:
            preflight_payload = json.loads(preflight_report.read_text(encoding="utf-8"))
            preflight_failed = preflight_payload.get("passed") is False
        except json.JSONDecodeError:
            preflight_failed = False
        if preflight_failed:
            tasks_text = tasks_path.read_text(encoding="utf-8", errors="ignore")
            required_open_tasks = {
                "T047/M2-003 real GPU execution": r"^- \[ \].*T047.*M2-003.*real.*GPU",
                "T048/M2-002 real aggregation": r"^- \[ \].*T048.*M2-002.*Aggregate.*real",
                "T049/M2-004 final figures/tables": r"^- \[ \].*T049.*M2-004.*final.*tables.*figure",
                "T050/M2-005 submission draft": r"^- \[ \].*T050.*M2-005.*submission.*draft",
                "T051/M2-006 final review": r"^- \[ \].*T051.*M2-006.*final.*review",
            }
            missing_open_tasks = [
                name
                for name, pattern in required_open_tasks.items()
                if not re.search(pattern, tasks_text, flags=re.IGNORECASE | re.MULTILINE)
            ]
            if missing_open_tasks:
                issues.append(
                    Issue(
                        kind="m2_tasks_missing_open_real_gpu_run_task",
                        path=str(tasks_path.relative_to(repo_root)),
                        detail=(
                            "Failed GPU preflight requires open T047-T051 "
                            "evidence-chain tasks; do not present real GPU "
                            "execution, aggregation, figures, draft, or review "
                            "work as complete. Missing: "
                            + ", ".join(missing_open_tasks)
                        ),
                    )
                )

    checklist_dir = feature_dir / "checklists"
    for path in sorted(checklist_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        incomplete = re.findall(r"^- \[ \]", text, flags=re.M)
        if incomplete:
            issues.append(
                Issue(
                    kind="incomplete_m2_speckit_checklist",
                    path=str(path.relative_to(repo_root)),
                    detail=f"{len(incomplete)} unchecked item(s).",
                )
            )
    return issues


def check_feature_m2_analysis_contract(repo_root: Path) -> list[Issue]:
    if not (repo_root / ".specify" / "goals" / "v2" / "GOAL-GEN-M2-002-multidataset-aggregation.md").exists():
        return []

    path = repo_root / M2_FEATURE_DIR / "analysis" / "m2-cross-artifact-analysis.md"
    if not path.is_file():
        return [
            Issue(
                kind="missing_m2_cross_artifact_analysis",
                path=str(path.relative_to(repo_root)),
                detail="M2 aggregation requires feature-scoped cross-artifact analysis.",
            )
        ]

    rel = str(path.relative_to(repo_root))
    text = path.read_text(encoding="utf-8", errors="ignore")
    issues: list[Issue] = []
    for snippet in M2_ANALYSIS_REQUIRED_SNIPPETS:
        if not _contains_normalized(text, snippet):
            issues.append(
                Issue(
                    kind="m2_cross_artifact_analysis_missing_contract_text",
                    path=rel,
                    detail=snippet,
                )
            )
    return issues


def check_feature_six_dataset_matrix(repo_root: Path) -> list[Issue]:
    if not (repo_root / M2_SIX_DATASET_MATRIX_GOAL).exists():
        return []

    path = repo_root / M2_SIX_DATASET_MATRIX
    if not path.is_file():
        return [
            Issue(
                kind="missing_m2_six_dataset_matrix",
                path=M2_SIX_DATASET_MATRIX,
                detail="M2 six-dataset GPU goal requires the paper matrix.",
            )
        ]

    rel = str(path.relative_to(repo_root))
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return [
            Issue(
                kind="invalid_m2_six_dataset_matrix_yaml",
                path=rel,
                detail=str(exc),
            )
        ]
    if not isinstance(payload, dict):
        return [
            Issue(
                kind="invalid_m2_six_dataset_matrix_shape",
                path=rel,
                detail="Expected YAML mapping.",
            )
        ]

    issues: list[Issue] = []
    benchmark = payload.get("benchmark") or {}
    resource = benchmark.get("resource") or {}
    data_check = benchmark.get("data_check") or {}
    datasets = payload.get("datasets") or []
    methods = payload.get("methods") or []
    overrides = benchmark.get("overrides") or {}
    if not isinstance(benchmark, dict):
        benchmark = {}
    if not isinstance(resource, dict):
        resource = {}
    if not isinstance(data_check, dict):
        data_check = {}
    if not isinstance(overrides, dict):
        overrides = {}
    if not isinstance(datasets, list):
        datasets = []
    if not isinstance(methods, list):
        methods = []

    if benchmark.get("id") != "phm_genbench_six_dataset_submission_v1":
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_benchmark_id",
                path=rel,
                detail="Expected phm_genbench_six_dataset_submission_v1.",
            )
        )
    if benchmark.get("min_datasets") != 6:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_min_datasets",
                path=rel,
                detail="Expected min_datasets=6.",
            )
        )
    if benchmark.get("output_dir") != M2_SIX_DATASET_EXPECTED_OUTPUT_DIR:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_output_dir",
                path=rel,
                detail=f"Expected output_dir {M2_SIX_DATASET_EXPECTED_OUTPUT_DIR}.",
            )
        )
    if benchmark.get("baseline_method") != M2_SIX_DATASET_EXPECTED_BASELINE:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_baseline_method",
                path=rel,
                detail=f"Expected baseline_method {M2_SIX_DATASET_EXPECTED_BASELINE}.",
            )
        )
    if data_check.get("data_dir") != M2_SIX_DATASET_EXPECTED_DATA_DIR:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_data_dir",
                path=rel,
                detail=f"Expected data_check.data_dir {M2_SIX_DATASET_EXPECTED_DATA_DIR}.",
            )
        )
    if data_check.get("metadata_file") != M2_SIX_DATASET_EXPECTED_METADATA:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_metadata_file",
                path=rel,
                detail=f"Expected data_check.metadata_file {M2_SIX_DATASET_EXPECTED_METADATA}.",
            )
        )
    if [int(seed) for seed in benchmark.get("seeds", [])] != [0, 1]:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_seeds",
                path=rel,
                detail="Expected seeds [0, 1].",
            )
        )
    if [str(gpu_id) for gpu_id in resource.get("gpu_ids", [])] != ["6", "7"]:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_gpu_ids",
                path=rel,
                detail="Expected resource.gpu_ids [6, 7].",
            )
        )
    if resource.get("max_parallel_runs") != 2:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_parallelism",
                path=rel,
                detail="Expected resource.max_parallel_runs=2.",
            )
        )
    if resource.get("require_cuda") is not True:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_require_cuda_not_true",
                path=rel,
                detail="Expected resource.require_cuda=true.",
            )
        )
    if overrides.get("trainer.device") != "cuda":
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_trainer_device_not_cuda",
                path=rel,
                detail="Expected trainer.device=cuda.",
            )
        )
    if overrides.get("trainer.gpus") != 1:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_trainer_gpus_not_one",
                path=rel,
                detail="Expected trainer.gpus=1.",
            )
        )
    if overrides.get("data.normalization") != "standardization":
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_normalization",
                path=rel,
                detail="Expected data.normalization=standardization.",
            )
        )

    dataset_ids = {
        str(item.get("dataset"))
        for item in datasets
        if isinstance(item, dict) and item.get("dataset")
    }
    if dataset_ids != M2_RUN_STATUS_EXPECTED_DATASETS:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_datasets",
                path=rel,
                detail=(
                    f"Expected datasets {sorted(M2_RUN_STATUS_EXPECTED_DATASETS)}, "
                    f"found {sorted(dataset_ids)}."
                ),
            )
        )
    for item in datasets:
        if not isinstance(item, dict):
            issues.append(
                Issue(
                    kind="m2_six_dataset_matrix_invalid_dataset_entry",
                    path=rel,
                    detail="Each dataset entry must be a mapping.",
                )
            )
            continue
        dataset = str(item.get("dataset") or "<missing>")
        item_overrides = item.get("overrides") or {}
        protocol = item.get("protocol") or {}
        if not item.get("name") or not isinstance(item.get("dataset_id"), int):
            issues.append(
                Issue(
                    kind="m2_six_dataset_matrix_dataset_missing_identity",
                    path=rel,
                    detail=f"{dataset} requires name and integer dataset_id.",
                )
            )
        if not isinstance(item_overrides, dict):
            item_overrides = {}
        for key in (
            "task.target_system_id",
            "task.source_domain_id",
            "task.target_domain_id",
        ):
            value = item_overrides.get(key)
            if not isinstance(value, list) or not value:
                issues.append(
                    Issue(
                        kind="m2_six_dataset_matrix_dataset_missing_override",
                        path=rel,
                        detail=f"{dataset} requires non-empty {key}.",
                    )
                )
        if not isinstance(protocol, dict):
            protocol = {}
        if not protocol.get("utility") or not protocol.get("notes"):
            issues.append(
                Issue(
                    kind="m2_six_dataset_matrix_dataset_missing_protocol",
                    path=rel,
                    detail=f"{dataset} requires protocol.utility and protocol.notes.",
                )
            )
    method_ids = {
        str(item.get("method"))
        for item in methods
        if isinstance(item, dict) and item.get("method")
    }
    if method_ids != M2_RUN_STATUS_EXPECTED_METHODS:
        issues.append(
            Issue(
                kind="m2_six_dataset_matrix_invalid_methods",
                path=rel,
                detail=(
                    f"Expected methods {sorted(M2_RUN_STATUS_EXPECTED_METHODS)}, "
                    f"found {sorted(method_ids)}."
                ),
            )
        )
    for item in methods:
        if not isinstance(item, dict):
            issues.append(
                Issue(
                    kind="m2_six_dataset_matrix_invalid_method_entry",
                    path=rel,
                    detail="Each method entry must be a mapping.",
                )
            )
            continue
        method = str(item.get("method") or "<missing>")
        train_config = str(item.get("train_config") or "")
        if not item.get("label"):
            issues.append(
                Issue(
                    kind="m2_six_dataset_matrix_method_missing_label",
                    path=rel,
                    detail=f"{method} requires label.",
                )
            )
        if item.get("condition_sampling_policy") not in (
            M2_SIX_DATASET_ALLOWED_CONDITION_POLICIES
        ):
            issues.append(
                Issue(
                    kind="m2_six_dataset_matrix_invalid_condition_policy",
                    path=rel,
                    detail=(
                        f"{method} requires one of "
                        f"{sorted(M2_SIX_DATASET_ALLOWED_CONDITION_POLICIES)}."
                    ),
                )
            )
        if not train_config or not (repo_root / train_config).is_file():
            issues.append(
                Issue(
                    kind="m2_six_dataset_matrix_missing_method_config",
                    path=rel,
                    detail=f"{method} train_config not found: {train_config or '<missing>'}.",
                )
            )
    return issues


def check_feature_m2_dry_run_plan(repo_root: Path) -> list[Issue]:
    if not (repo_root / M2_SIX_DATASET_MATRIX_GOAL).exists():
        return []

    path = repo_root / M2_DRY_RUN_PLAN
    if not path.is_file():
        return [
            Issue(
                kind="missing_m2_dry_run_plan",
                path=M2_DRY_RUN_PLAN,
                detail="M2 six-dataset matrix goal requires a dry-run run_plan.csv.",
            )
        ]

    rel = str(path.relative_to(repo_root))
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except csv.Error as exc:
        return [
            Issue(
                kind="invalid_m2_dry_run_plan_csv",
                path=rel,
                detail=str(exc),
            )
        ]

    issues: list[Issue] = []
    expected_stages = frozenset({"train", "sample", "eval", "paperpack"})
    expected_keys = {
        (dataset, method, seed, stage)
        for dataset in M2_RUN_STATUS_EXPECTED_DATASETS
        for method in M2_RUN_STATUS_EXPECTED_METHODS
        for seed in M2_RUN_STATUS_EXPECTED_SEEDS
        for stage in expected_stages
    }
    if len(rows) != len(expected_keys):
        issues.append(
            Issue(
                kind="invalid_m2_dry_run_plan_row_count",
                path=rel,
                detail=f"Expected {len(expected_keys)} rows, found {len(rows)}.",
            )
        )

    seen_keys: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(rows, start=2):
        dataset = (row.get("dataset") or "").strip()
        method = (row.get("method") or "").strip()
        seed = (row.get("seed") or "").strip()
        stage = (row.get("stage") or "").strip()
        gpu_id = (row.get("gpu_id") or "").strip()
        command = (row.get("command") or "").strip()
        key = (dataset, method, seed, stage)
        if all(key):
            if key in seen_keys:
                issues.append(
                    Issue(
                        kind="duplicate_m2_dry_run_plan_row",
                        path=rel,
                        detail=f"Duplicate dataset/method/seed/stage row: {key}.",
                    )
                )
            seen_keys.add(key)
        if gpu_id not in {"6", "7"}:
            issues.append(
                Issue(
                    kind="invalid_m2_dry_run_plan_gpu",
                    path=rel,
                    detail=f"Row {index} expected GPU 6 or 7, found {gpu_id}.",
                )
            )
        if "CUDA_VISIBLE_DEVICES=" not in command:
            issues.append(
                Issue(
                    kind="m2_dry_run_plan_command_missing_cuda_visible_devices",
                    path=rel,
                    detail=f"Row {index} command must pin CUDA_VISIBLE_DEVICES.",
                )
            )
        if stage != "paperpack":
            if "trainer.device=cuda" not in command or "trainer.gpus=1" not in command:
                issues.append(
                    Issue(
                        kind="m2_dry_run_plan_command_missing_cuda_trainer_override",
                        path=rel,
                        detail=(
                            f"Row {index} main.py command must keep "
                            "trainer.device=cuda and trainer.gpus=1."
                        ),
                    )
                )
        elif "scripts.paperpack_generative" not in command:
            issues.append(
                Issue(
                    kind="m2_dry_run_plan_paperpack_command_invalid",
                    path=rel,
                    detail=f"Row {index} paperpack command must call paperpack.",
                )
            )

    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)
        unexpected = sorted(seen_keys - expected_keys)
        issues.append(
            Issue(
                kind="invalid_m2_dry_run_plan_matrix",
                path=rel,
                detail=(
                    f"Missing rows: {missing[:3]}"
                    f"{'...' if len(missing) > 3 else ''}; "
                    f"unexpected rows: {unexpected[:3]}"
                    f"{'...' if len(unexpected) > 3 else ''}."
                ),
            )
        )

    return issues


def check_feature_review_handoff_artifacts(repo_root: Path) -> list[Issue]:
    specs_dir = repo_root / "specs"
    issues: list[Issue] = []
    if (repo_root / M2_REVIEW_HANDOFF_GOAL).exists():
        feature_dir = repo_root / M2_FEATURE_DIR
        if not feature_dir.exists():
            issues.append(
                Issue(
                    kind="missing_m2_feature_dir",
                    path=M2_FEATURE_DIR,
                    detail=(
                        "M2 review/handoff goal requires the active feature "
                        "directory."
                    ),
                )
            )
        else:
            concrete_runs = [
                path
                for path in (feature_dir / "reviews" / "claude-team").glob("*")
                if path.is_dir() and "template" not in path.name
            ]
            concrete_handoffs = [
                path
                for path in (feature_dir / "handoffs").glob("*.md")
                if path.name != "README.md"
            ]
            if not concrete_runs:
                issues.append(
                    Issue(
                        kind="missing_m2_claude_team_run",
                        path=str(
                            (feature_dir / "reviews" / "claude-team").relative_to(
                                repo_root
                            )
                        ),
                        detail=(
                            "M2 review/handoff goal requires a concrete "
                            "Claude team run artifact."
                        ),
                    )
                )
            if not concrete_handoffs:
                issues.append(
                    Issue(
                        kind="missing_m2_handoff",
                        path=str((feature_dir / "handoffs").relative_to(repo_root)),
                        detail=(
                            "M2 review/handoff goal requires a concrete "
                            "handoff artifact."
                        ),
                    )
                )
    if not specs_dir.exists():
        return issues
    for run_dir in sorted(specs_dir.glob("*/reviews/claude-team/*")):
        if not run_dir.is_dir() or "template" in run_dir.name:
            continue
        rel_run = run_dir.relative_to(repo_root)
        for filename in CLAUDE_TEAM_REQUIRED_FILES:
            path = run_dir / filename
            if not path.is_file():
                issues.append(
                    Issue(
                        kind="missing_claude_team_artifact",
                        path=str(rel_run / filename),
                        detail=(
                            "Claude team runs require TASK_SPEC, report, "
                            "risks, and test-log."
                        ),
                )
            )
        task_spec = run_dir / "TASK_SPEC.md"
        if task_spec.is_file():
            task_text = task_spec.read_text(encoding="utf-8", errors="ignore")
            for snippet in CLAUDE_TASK_SPEC_REQUIRED_SNIPPETS:
                if snippet not in task_text:
                    issues.append(
                        Issue(
                            kind="claude_team_task_spec_missing_required_text",
                            path=str(task_spec.relative_to(repo_root)),
                            detail=snippet,
                        )
                    )
            teammates_match = re.search(
                r"^## Teammates\s*(.*?)(?:^## |\Z)",
                task_text,
                flags=re.S | re.M,
            )
            teammate_lines = (
                [
                    line
                    for line in teammates_match.group(1).splitlines()
                    if line.strip().startswith("- ")
                ]
                if teammates_match
                else []
            )
            if len(teammate_lines) < 3:
                issues.append(
                    Issue(
                        kind="claude_team_task_spec_insufficient_teammates",
                        path=str(task_spec.relative_to(repo_root)),
                        detail="Expected at least three scoped reviewer teammates.",
                    )
                )
        report = run_dir / "report.md"
        if report.is_file():
            report_text = report.read_text(encoding="utf-8", errors="ignore")
            decision_match = re.search(
                r"<REVIEW_DECISION>\s*([^<]+?)\s*</REVIEW_DECISION>",
                report_text,
            )
            if decision_match is None:
                issues.append(
                    Issue(
                        kind="claude_team_report_missing_review_decision",
                        path=str(report.relative_to(repo_root)),
                        detail="Expected machine-readable <REVIEW_DECISION> tag.",
                    )
                )
            elif decision_match.group(1).strip() not in CLAUDE_REVIEW_ALLOWED_DECISIONS:
                issues.append(
                    Issue(
                        kind="claude_team_report_invalid_review_decision",
                        path=str(report.relative_to(repo_root)),
                        detail=(
                            "Expected APPROVE, REQUEST_CHANGES, or BLOCKING; "
                            f"found {decision_match.group(1).strip()}."
                        ),
                    )
                )
            for tag in CLAUDE_REVIEW_REQUIRED_TAGS:
                if f"<{tag}>" not in report_text or f"</{tag}>" not in report_text:
                    issues.append(
                        Issue(
                            kind="claude_team_report_missing_required_tag",
                            path=str(report.relative_to(repo_root)),
                            detail=tag,
                        )
                    )
            fix_match = re.search(
                r"<FIX_INSTRUCTION>\s*(.*?)\s*</FIX_INSTRUCTION>",
                report_text,
                flags=re.S,
            )
            if fix_match is None:
                issues.append(
                    Issue(
                        kind="claude_team_report_missing_fix_instruction",
                        path=str(report.relative_to(repo_root)),
                        detail="Expected closing </FIX_INSTRUCTION> tag.",
                    )
                )
            elif not fix_match.group(1).strip():
                issues.append(
                    Issue(
                        kind="claude_team_report_empty_fix_instruction",
                        path=str(report.relative_to(repo_root)),
                        detail="Expected non-empty Codex-ready fix instruction.",
                    )
                )
            elif not report_text.rstrip().endswith("</FIX_INSTRUCTION>"):
                issues.append(
                    Issue(
                        kind="claude_team_report_trailing_text_after_fix_instruction",
                        path=str(report.relative_to(repo_root)),
                        detail="Claude review report must end with </FIX_INSTRUCTION>.",
                    )
                )
            if "BLOCKED_NOT_RUN" in report_text:
                if (
                    decision_match is not None
                    and decision_match.group(1).strip() != "BLOCKING"
                ):
                    issues.append(
                        Issue(
                            kind="blocked_claude_review_non_blocking_decision",
                            path=str(report.relative_to(repo_root)),
                            detail=(
                                "Blocked Claude reviews must use "
                                "<REVIEW_DECISION>BLOCKING</REVIEW_DECISION>."
                            ),
                        )
                    )
                for filename in ("risks.md", "test-log.md"):
                    path = run_dir / filename
                    text = (
                        path.read_text(encoding="utf-8", errors="ignore")
                        if path.is_file()
                        else ""
                    )
                    if "BLOCKED_NOT_RUN" not in text:
                        issues.append(
                            Issue(
                                kind="blocked_claude_artifact_missing_status",
                                path=str(path.relative_to(repo_root)),
                                detail=(
                                    "Blocked Claude reviews must mark each "
                                    "output BLOCKED_NOT_RUN."
                                ),
                            )
                        )

    for path in sorted(specs_dir.glob("*/handoffs/*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(repo_root))
        for section in HANDOFF_REQUIRED_SECTIONS:
            if section not in text:
                issues.append(
                    Issue(
                        kind="handoff_missing_section",
                        path=rel,
                        detail=section,
                    )
                )
    return issues


def _draft_status(text: str) -> Optional[str]:
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("**Draft status:**") or line.startswith("Status:")):
            continue
        for status in ("SUBMISSION_READY", "NOT_SUBMISSION_READY"):
            if f"`{status}`" in line:
                return status
    return None


def check_feature_paper_artifacts(repo_root: Path) -> list[Issue]:
    if not (repo_root / M2_PAPER_DRAFT_GOAL).exists():
        return []

    issues: list[Issue] = []
    paper_dir = repo_root / M2_FEATURE_DIR / "paper"
    if not paper_dir.exists():
        return [
            Issue(
                kind="missing_m2_paper_dir",
                path=str(paper_dir.relative_to(repo_root)),
                detail=(
                    "M2 paper draft goal requires feature-scoped paper "
                    "artifacts."
                ),
            )
        ]

    for filename in M2_PAPER_REQUIRED_FILES:
        path = paper_dir / filename
        if not path.is_file():
            issues.append(
                Issue(
                    kind="missing_m2_paper_artifact",
                    path=str(path.relative_to(repo_root)),
                    detail=(
                        "M2 paper draft goal requires draft, gaps, and "
                        "readiness files."
                    ),
                )
            )

    draft = paper_dir / "PAPER_DRAFT.md"
    readiness = paper_dir / "submission_readiness.md"
    gaps = paper_dir / "evidence_gaps.md"
    if not draft.is_file() or not readiness.is_file() or not gaps.is_file():
        return issues

    draft_text = draft.read_text(encoding="utf-8", errors="ignore")
    readiness_text = readiness.read_text(encoding="utf-8", errors="ignore")
    gaps_text = gaps.read_text(encoding="utf-8", errors="ignore")
    draft_status = _draft_status(draft_text)
    readiness_status = _draft_status(readiness_text)
    if draft_status is None:
        issues.append(
            Issue(
                kind="m2_paper_draft_missing_status",
                path=str(draft.relative_to(repo_root)),
                detail="Expected SUBMISSION_READY or NOT_SUBMISSION_READY status.",
            )
        )
    if readiness_status is None:
        issues.append(
            Issue(
                kind="m2_paper_readiness_missing_status",
                path=str(readiness.relative_to(repo_root)),
                detail="Expected SUBMISSION_READY or NOT_SUBMISSION_READY status.",
            )
        )
    if draft_status and readiness_status and draft_status != readiness_status:
        issues.append(
            Issue(
                kind="m2_paper_status_mismatch",
                path=str(readiness.relative_to(repo_root)),
                detail=f"Draft status {draft_status} != readiness {readiness_status}.",
            )
        )
    preflight_report = repo_root / M2_GPU_PREFLIGHT_REPORT
    paper_ready = draft_status == "SUBMISSION_READY" and readiness_status == "SUBMISSION_READY"
    if (
        (draft_status == "SUBMISSION_READY" or readiness_status == "SUBMISSION_READY")
        and preflight_report.is_file()
    ):
        try:
            preflight_payload = json.loads(preflight_report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            preflight_payload = {}
        if preflight_payload.get("passed") is False:
            issues.append(
                Issue(
                    kind="m2_paper_ready_while_gpu_preflight_failed",
                    path=str(draft.relative_to(repo_root)),
                    detail="SUBMISSION_READY paper artifacts require passing GPU preflight.",
                )
            )
    if paper_ready:
        ledger_path = repo_root / M2_RUN_STATUS_LEDGER
        if not ledger_path.is_file():
            issues.append(
                Issue(
                    kind="m2_paper_ready_missing_run_status_ledger",
                    path=str(draft.relative_to(repo_root)),
                    detail="SUBMISSION_READY paper artifacts require run status ledger.",
                )
            )
        else:
            try:
                with ledger_path.open("r", encoding="utf-8", newline="") as handle:
                    for index, row in enumerate(csv.DictReader(handle), start=2):
                        status = (row.get("status") or "").strip()
                        if status.upper().startswith("BLOCKED"):
                            issues.append(
                                Issue(
                                    kind="m2_paper_ready_with_blocked_run_status",
                                    path=str(draft.relative_to(repo_root)),
                                    detail=(
                                        f"Row {index} in run status ledger is "
                                        f"still {status}."
                                    ),
                                )
                            )
                            break
                        if status.lower() != "complete":
                            issues.append(
                                Issue(
                                    kind="m2_paper_ready_with_incomplete_run_status",
                                    path=str(draft.relative_to(repo_root)),
                                    detail=(
                                        f"Row {index} in run status ledger is "
                                        f"{status or '<empty>'}, not complete."
                                    ),
                                )
                            )
                            break
            except csv.Error as exc:
                issues.append(
                    Issue(
                        kind="m2_paper_ready_invalid_run_status_ledger",
                        path=str(ledger_path.relative_to(repo_root)),
                        detail=str(exc),
                    )
                )
    if draft_status == "NOT_SUBMISSION_READY" and "- " not in gaps_text:
        issues.append(
            Issue(
                kind="m2_paper_missing_evidence_gaps",
                path=str(gaps.relative_to(repo_root)),
                detail="NOT_SUBMISSION_READY drafts must list evidence gaps.",
            )
        )
    if readiness_status == "NOT_SUBMISSION_READY" and "- " not in readiness_text:
        issues.append(
            Issue(
                kind="m2_paper_readiness_missing_reason",
                path=str(readiness.relative_to(repo_root)),
                detail="NOT_SUBMISSION_READY readiness must list evidence gaps.",
            )
        )
    if draft_status == "SUBMISSION_READY" and readiness_status == "SUBMISSION_READY":
        for evidence_file in M2_PAPER_READY_EVIDENCE_FILES:
            evidence_path = repo_root / evidence_file
            if not evidence_path.is_file():
                issues.append(
                    Issue(
                        kind="m2_paper_ready_missing_evidence_file",
                        path=str(draft.relative_to(repo_root)),
                        detail=f"Missing ready evidence file: {evidence_file}",
                    )
                )
        summary_path = repo_root / M2_PAPER_READY_EVIDENCE_FILES[0]
        manifest_path = repo_root / M2_PAPER_READY_EVIDENCE_FILES[1]
        if manifest_path.is_file():
            try:
                ready_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                issues.append(
                    Issue(
                        kind="m2_paper_ready_invalid_manifest_json",
                        path=str(manifest_path.relative_to(repo_root)),
                        detail=str(exc),
                    )
                )
                ready_manifest = {}
            if ready_manifest:
                observed_count = ready_manifest.get("observed_configured_dataset_count")
                try:
                    observed_count_int = int(observed_count)
                except (TypeError, ValueError):
                    observed_count_int = -1
                has_manifest_gap = any(
                    ready_manifest.get(field)
                    for field in ("input_gaps", "missing_datasets", "unexpected_datasets")
                )
                if (
                    ready_manifest.get("min_datasets_met") is not True
                    or observed_count_int < M2_PAPER_READY_MIN_DATASETS
                    or has_manifest_gap
                ):
                    issues.append(
                        Issue(
                            kind="m2_paper_ready_manifest_not_ready",
                            path=str(manifest_path.relative_to(repo_root)),
                            detail=(
                                "Ready manifest must have min_datasets_met=true, "
                                "observed_configured_dataset_count>=6, and no "
                                "input/missing/unexpected dataset gaps."
                            ),
                        )
                    )
                observed_datasets = ready_manifest.get("observed_configured_datasets")
                if not isinstance(observed_datasets, list) or {
                    str(dataset) for dataset in observed_datasets
                } != set(M2_RUN_STATUS_EXPECTED_DATASETS):
                    issues.append(
                        Issue(
                            kind="m2_paper_ready_manifest_dataset_mismatch",
                            path=str(manifest_path.relative_to(repo_root)),
                            detail=(
                                "Ready manifest observed_configured_datasets "
                                "must match the six configured paper datasets."
                            ),
                        )
                    )
            else:
                issues.append(
                    Issue(
                        kind="m2_paper_ready_manifest_not_ready",
                        path=str(manifest_path.relative_to(repo_root)),
                        detail="Ready manifest must not be empty.",
                    )
                )
        if summary_path.is_file():
            try:
                with summary_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except csv.Error as exc:
                issues.append(
                    Issue(
                        kind="m2_paper_ready_invalid_summary_csv",
                        path=str(summary_path.relative_to(repo_root)),
                        detail=str(exc),
                    )
                )
                rows = []
            evidence_by_dataset: dict[str, set[str]] = {}
            missing_sources = False
            missing_source_files = False
            non_valid_status = False
            invalid_n = False
            for row in rows:
                dataset = (row.get("dataset") or "").strip()
                category = (row.get("category") or "").strip()
                status = (row.get("benchmark_status") or "").strip()
                if status and status != "benchmark-valid":
                    non_valid_status = True
                if not dataset or category not in {"quality", "utility"}:
                    continue
                if status != "benchmark-valid":
                    continue
                try:
                    n_value = int((row.get("n") or "").strip())
                except ValueError:
                    n_value = 0
                if n_value <= 0:
                    invalid_n = True
                    continue
                if not (row.get("metric_source_paths") or "").strip() or not (
                    row.get("manifest_paths") or ""
                ).strip():
                    missing_sources = True
                else:
                    for source_field in ("metric_source_paths", "manifest_paths"):
                        for source in str(row.get(source_field) or "").split(";"):
                            source = source.strip()
                            if not source:
                                continue
                            source_rel = Path(source)
                            if (
                                source_rel.is_absolute()
                                or ".." in source_rel.parts
                                or not (repo_root / source_rel).is_file()
                            ):
                                missing_source_files = True
                evidence_by_dataset.setdefault(dataset, set()).add(category)
            ready_datasets = {
                dataset
                for dataset, categories in evidence_by_dataset.items()
                if {"quality", "utility"}.issubset(categories)
            }
            if len(ready_datasets) < M2_PAPER_READY_MIN_DATASETS:
                issues.append(
                    Issue(
                        kind="m2_paper_ready_summary_insufficient_datasets",
                        path=str(summary_path.relative_to(repo_root)),
                        detail=(
                            "Ready summary must include benchmark-valid quality "
                            "and utility rows for at least six datasets."
                        ),
                    )
                )
            if ready_datasets != set(M2_RUN_STATUS_EXPECTED_DATASETS):
                issues.append(
                    Issue(
                        kind="m2_paper_ready_summary_dataset_mismatch",
                        path=str(summary_path.relative_to(repo_root)),
                        detail=(
                            "Ready summary datasets must match the six "
                            "configured paper datasets."
                        ),
                    )
                )
            if missing_sources:
                issues.append(
                    Issue(
                        kind="m2_paper_ready_summary_missing_source_paths",
                        path=str(summary_path.relative_to(repo_root)),
                        detail=(
                            "Ready summary quality/utility rows must include "
                            "metric_source_paths and manifest_paths."
                        ),
                    )
                )
            if missing_source_files:
                issues.append(
                    Issue(
                        kind="m2_paper_ready_summary_missing_source_files",
                        path=str(summary_path.relative_to(repo_root)),
                        detail=(
                            "Ready summary metric_source_paths and "
                            "manifest_paths must point to existing repository "
                            "files."
                        ),
                    )
                )
            if invalid_n:
                issues.append(
                    Issue(
                        kind="m2_paper_ready_summary_invalid_n",
                        path=str(summary_path.relative_to(repo_root)),
                        detail=(
                            "Ready summary benchmark-valid quality/utility rows "
                            "must have n > 0."
                        ),
                    )
                )
            if non_valid_status:
                issues.append(
                    Issue(
                        kind="m2_paper_ready_summary_non_valid_status",
                        path=str(summary_path.relative_to(repo_root)),
                        detail="Ready summary rows must be benchmark-valid.",
                    )
                )
    placeholder = M2_PAPER_FORBIDDEN_PLACEHOLDER_RE.search(draft_text)
    if placeholder:
        issues.append(
            Issue(
                kind="m2_paper_draft_contains_placeholder",
                path=str(draft.relative_to(repo_root)),
                detail=f"Forbidden placeholder token: {placeholder.group(0)}",
            )
        )
    for section in M2_PAPER_DRAFT_REQUIRED_SECTIONS:
        if section not in draft_text:
            issues.append(
                Issue(
                    kind="m2_paper_draft_missing_section",
                    path=str(draft.relative_to(repo_root)),
                    detail=section,
                )
            )
    for snippet in M2_PAPER_DRAFT_REQUIRED_SNIPPETS:
        if not _contains_normalized(draft_text, snippet):
            issues.append(
                Issue(
                    kind="m2_paper_draft_missing_required_text",
                    path=str(draft.relative_to(repo_root)),
                    detail=snippet,
                )
            )
    if draft_status == "NOT_SUBMISSION_READY":
        for snippet in M2_PAPER_NOT_READY_DRAFT_REQUIRED_SNIPPETS:
            if not _contains_normalized(draft_text, snippet):
                issues.append(
                    Issue(
                        kind="m2_paper_not_ready_draft_missing_blocked_claim_text",
                        path=str(draft.relative_to(repo_root)),
                        detail=snippet,
                    )
                )
    for snippet in M2_PAPER_GAPS_REQUIRED_SNIPPETS:
        if not _contains_normalized(gaps_text, snippet):
            issues.append(
                Issue(
                    kind="m2_paper_gaps_missing_required_text",
                    path=str(gaps.relative_to(repo_root)),
                    detail=snippet,
                )
            )
    for snippet in M2_PAPER_READINESS_REQUIRED_SNIPPETS:
        if not _contains_normalized(readiness_text, snippet):
            issues.append(
                Issue(
                    kind="m2_paper_readiness_missing_required_text",
                    path=str(readiness.relative_to(repo_root)),
                    detail=snippet,
                )
            )
    return issues


def check_feature_gpu_preflight_artifact(repo_root: Path) -> list[Issue]:
    if not (repo_root / M2_REAL_RUNS_GOAL).exists():
        return []

    path = repo_root / M2_GPU_PREFLIGHT_REPORT
    if not path.is_file():
        return [
            Issue(
                kind="missing_m2_gpu_preflight_report",
                path=M2_GPU_PREFLIGHT_REPORT,
                detail="M2 real-runs goal requires a reviewable GPU preflight report.",
            )
        ]

    rel = str(path.relative_to(repo_root))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            Issue(
                kind="invalid_m2_gpu_preflight_report_json",
                path=rel,
                detail=str(exc),
            )
        ]

    issues: list[Issue] = []
    source_report = payload.get("source_report")
    if source_report is None:
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_missing_source_report_field",
                path=rel,
                detail=(
                    "Reviewable GPU preflight report must point to the "
                    "canonical generated source_report."
                ),
            )
        )
    elif not isinstance(source_report, str) or not source_report.strip():
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_invalid_source_report",
                path=rel,
                detail="source_report must be a non-empty relative path.",
            )
        )
    elif source_report != M2_GPU_PREFLIGHT_EXPECTED_SOURCE_REPORT:
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_noncanonical_source_report",
                path=rel,
                detail=(
                    "Expected source_report "
                    f"{M2_GPU_PREFLIGHT_EXPECTED_SOURCE_REPORT}."
                ),
            )
        )
    else:
        source_rel = Path(source_report)
        if source_rel.is_absolute() or ".." in source_rel.parts:
            issues.append(
                Issue(
                    kind="m2_gpu_preflight_report_invalid_source_report",
                    path=rel,
                    detail="source_report must stay inside the repository.",
                )
            )
        else:
            source_path = repo_root / source_rel
            if not source_path.is_file():
                issues.append(
                    Issue(
                        kind="m2_gpu_preflight_report_missing_source_report",
                        path=rel,
                        detail=f"source_report does not exist: {source_report}",
                    )
                )
            else:
                try:
                    source_payload = json.loads(
                        source_path.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError as exc:
                    issues.append(
                        Issue(
                            kind="m2_gpu_preflight_report_invalid_source_json",
                            path=str(source_rel),
                            detail=str(exc),
                        )
                    )
                else:
                    for key in (
                        "benchmark_id",
                        "matrix_path",
                        "require_cuda",
                        "gpu_ids",
                        "max_parallel_runs",
                        "passed",
                        "results",
                    ):
                        if source_payload.get(key) != payload.get(key):
                            issues.append(
                                Issue(
                                    kind=(
                                        "m2_gpu_preflight_report_source_mismatch"
                                    ),
                                    path=rel,
                                    detail=f"source_report mismatch for {key}.",
                                )
                            )
                            break
    if payload.get("benchmark_id") != "phm_genbench_six_dataset_submission_v1":
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_invalid_benchmark_id",
                path=rel,
                detail="Expected phm_genbench_six_dataset_submission_v1.",
            )
        )
    if payload.get("matrix_path") != (
        "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml"
    ):
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_invalid_matrix_path",
                path=rel,
                detail="Expected six_dataset_benchmark_matrix.yaml.",
            )
        )
    if payload.get("require_cuda") is not True:
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_require_cuda_not_true",
                path=rel,
                detail="M2 GPU preflight evidence must require CUDA.",
            )
        )
    if payload.get("gpu_ids") != ["6", "7"]:
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_invalid_gpu_ids",
                path=rel,
                detail="Expected GPU ids ['6', '7'].",
            )
        )
    if payload.get("max_parallel_runs") != 2:
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_invalid_parallelism",
                path=rel,
                detail="Expected max_parallel_runs=2.",
            )
        )
    if not isinstance(payload.get("passed"), bool):
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_missing_passed_flag",
                path=rel,
                detail="Expected boolean passed field.",
            )
        )
    results = payload.get("results")
    if not isinstance(results, list):
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_missing_results",
                path=rel,
                detail="Expected results list.",
            )
        )
        return issues
    if {str(item.get("gpu_id")) for item in results if isinstance(item, dict)} != {
        "6",
        "7",
    }:
        issues.append(
            Issue(
                kind="m2_gpu_preflight_report_missing_gpu_results",
                path=rel,
                detail="Expected one result for GPU 6 and one for GPU 7.",
            )
        )
    for item in results:
        if not isinstance(item, dict) or item.get("status") not in {
            "passed",
            "failed",
        }:
            issues.append(
                Issue(
                    kind="m2_gpu_preflight_report_invalid_result_status",
                    path=rel,
                    detail="Each GPU result must be passed or failed.",
                )
            )
            break
    result_statuses = [
        item.get("status") for item in results if isinstance(item, dict)
    ]
    if isinstance(payload.get("passed"), bool) and result_statuses:
        all_passed = all(status == "passed" for status in result_statuses)
        if payload["passed"] != all_passed:
            issues.append(
                Issue(
                    kind="m2_gpu_preflight_report_passed_status_mismatch",
                    path=rel,
                    detail="Report passed flag must match per-GPU statuses.",
                )
            )
    for item in results:
        if not isinstance(item, dict) or item.get("status") != "failed":
            continue
        gpu_id = str(item.get("gpu_id") or "")
        error = str(item.get("error") or "")
        if not error:
            issues.append(
                Issue(
                    kind="m2_gpu_preflight_report_failed_result_missing_error",
                    path=rel,
                    detail=f"GPU {gpu_id or '<missing>'} failed without error.",
                )
            )
        elif gpu_id and gpu_id not in error:
            issues.append(
                Issue(
                    kind="m2_gpu_preflight_report_failed_result_error_missing_gpu",
                    path=rel,
                    detail=f"GPU {gpu_id} error must name the failed GPU.",
                )
            )
    return issues


def check_goal_gen_completion_audit(repo_root: Path) -> list[Issue]:
    preflight_path = repo_root / M2_GPU_PREFLIGHT_REPORT
    if not preflight_path.is_file():
        return []

    try:
        preflight_failed = (
            json.loads(preflight_path.read_text(encoding="utf-8")).get("passed")
            is False
        )
    except json.JSONDecodeError:
        return []
    if not preflight_failed:
        return []

    audit_path = repo_root / GOAL_GEN_COMPLETION_AUDIT
    if not audit_path.is_file():
        return [
            Issue(
                kind="missing_goal_gen_completion_audit",
                path=str(audit_path.relative_to(repo_root)),
                detail="Failed GPU preflight requires a NOT COMPLETE completion audit.",
            )
        ]

    rel = str(audit_path.relative_to(repo_root))
    text = audit_path.read_text(encoding="utf-8", errors="ignore")
    issues: list[Issue] = []
    for snippet in GOAL_GEN_COMPLETION_AUDIT_REQUIRED_SNIPPETS:
        if not _contains_normalized(text, snippet):
            issues.append(
                Issue(
                    kind="goal_gen_completion_audit_missing_text",
                    path=rel,
                    detail=snippet,
                )
            )
    return issues


def check_goal_gen_status_package(repo_root: Path) -> list[Issue]:
    preflight_path = repo_root / M2_GPU_PREFLIGHT_REPORT
    if not preflight_path.is_file():
        return []

    try:
        preflight_failed = (
            json.loads(preflight_path.read_text(encoding="utf-8")).get("passed")
            is False
        )
    except json.JSONDecodeError:
        return []
    if not preflight_failed:
        return []

    status_path = repo_root / GOAL_GEN_STATUS_REPORT
    if not status_path.is_file():
        return [
            Issue(
                kind="missing_goal_gen_status_report",
                path=str(status_path.relative_to(repo_root)),
                detail="Failed GPU preflight requires a current v2 status report.",
            )
        ]

    rel = str(status_path.relative_to(repo_root))
    text = status_path.read_text(encoding="utf-8", errors="ignore")
    issues: list[Issue] = []
    for snippet in GOAL_GEN_STATUS_REPORT_REQUIRED_SNIPPETS:
        if not _contains_normalized(text, snippet):
            issues.append(
                Issue(
                    kind="goal_gen_status_report_missing_text",
                    path=rel,
                    detail=snippet,
                )
            )
    return issues


def check_feature_m2_gpu_runbook(repo_root: Path) -> list[Issue]:
    if not (repo_root / M2_REAL_RUNS_GOAL).exists():
        return []

    path = repo_root / M2_GPU_RUNBOOK
    if not path.is_file():
        return [
            Issue(
                kind="missing_m2_gpu_runbook",
                path=M2_GPU_RUNBOOK,
                detail="M2 real-runs goal requires a feature-scoped GPU runbook.",
            )
        ]

    rel = str(path.relative_to(repo_root))
    text = path.read_text(encoding="utf-8", errors="ignore")
    issues: list[Issue] = []
    for section in M2_GPU_RUNBOOK_REQUIRED_SECTIONS:
        if section not in text:
            issues.append(
                Issue(
                    kind="m2_gpu_runbook_missing_section",
                    path=rel,
                    detail=section,
                )
            )
    for snippet in M2_GPU_RUNBOOK_REQUIRED_SNIPPETS:
        if snippet not in text:
            issues.append(
                Issue(
                    kind="m2_gpu_runbook_missing_required_text",
                    path=rel,
                    detail=snippet,
                )
            )
    return issues


def check_feature_m2_run_status_ledger(repo_root: Path) -> list[Issue]:
    if not (repo_root / M2_REAL_RUNS_GOAL).exists():
        return []

    path = repo_root / M2_RUN_STATUS_LEDGER
    if not path.is_file():
        return [
            Issue(
                kind="missing_m2_run_status_ledger",
                path=M2_RUN_STATUS_LEDGER,
                detail="M2 real-runs goal requires a reviewable run status ledger.",
            )
        ]

    md_path = repo_root / M2_RUN_STATUS_LEDGER_MARKDOWN
    issues: list[Issue] = []
    if not md_path.is_file():
        issues.append(
            Issue(
                kind="missing_m2_run_status_ledger_markdown",
                path=M2_RUN_STATUS_LEDGER_MARKDOWN,
                detail=(
                    "M2 real-runs goal requires a human-readable run status "
                    "ledger under the active feature spec."
                ),
            )
        )
    else:
        md_rel = str(md_path.relative_to(repo_root))
        md_text = md_path.read_text(encoding="utf-8")
        for snippet in M2_RUN_STATUS_LEDGER_MARKDOWN_REQUIRED_SNIPPETS:
            if snippet not in md_text:
                issues.append(
                    Issue(
                        kind="m2_run_status_ledger_markdown_missing_required_text",
                        path=md_rel,
                        detail=snippet,
                    )
                )
        preflight_report_path = repo_root / M2_GPU_PREFLIGHT_REPORT
        if preflight_report_path.is_file():
            try:
                preflight_payload = json.loads(
                    preflight_report_path.read_text(encoding="utf-8")
                )
                source_report = preflight_payload.get("source_report")
            except json.JSONDecodeError:
                source_report = None
            if isinstance(source_report, str):
                source_ledger_rel = (
                    Path(source_report).parent / "blocked_run_status_ledger.csv"
                ).as_posix()
                if source_ledger_rel not in md_text:
                    issues.append(
                        Issue(
                            kind=(
                                "m2_run_status_ledger_markdown_missing_source_ledger"
                            ),
                            path=md_rel,
                            detail=source_ledger_rel,
                        )
                    )
        expected_dataset_mentions = (
            len(M2_RUN_STATUS_EXPECTED_METHODS) * len(M2_RUN_STATUS_EXPECTED_SEEDS)
        )
        expected_method_mentions = (
            len(M2_RUN_STATUS_EXPECTED_DATASETS) * len(M2_RUN_STATUS_EXPECTED_SEEDS)
        )
        expected_blocked_statuses = (
            len(M2_RUN_STATUS_EXPECTED_DATASETS)
            * len(M2_RUN_STATUS_EXPECTED_METHODS)
            * len(M2_RUN_STATUS_EXPECTED_SEEDS)
        )
        incomplete_parts: list[str] = []
        for dataset in sorted(M2_RUN_STATUS_EXPECTED_DATASETS):
            count = md_text.count(dataset)
            if count < expected_dataset_mentions:
                incomplete_parts.append(
                    f"{dataset} appears {count}/{expected_dataset_mentions}"
                )
        for method in sorted(M2_RUN_STATUS_EXPECTED_METHODS):
            count = md_text.count(method)
            if count < expected_method_mentions:
                incomplete_parts.append(
                    f"{method} appears {count}/{expected_method_mentions}"
                )
        blocked_count = md_text.count("BLOCKED_GPU_PREFLIGHT")
        if blocked_count < expected_blocked_statuses:
            incomplete_parts.append(
                "BLOCKED_GPU_PREFLIGHT appears "
                f"{blocked_count}/{expected_blocked_statuses}"
            )
        if incomplete_parts:
            issues.append(
                Issue(
                    kind="m2_run_status_ledger_markdown_incomplete_matrix",
                    path=md_rel,
                    detail="; ".join(incomplete_parts[:6]),
                )
            )

    rel = str(path.relative_to(repo_root))
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing_fields = [
                field
                for field in M2_RUN_STATUS_REQUIRED_FIELDS
                if field not in fieldnames
            ]
            if missing_fields:
                return [
                    Issue(
                        kind="invalid_m2_run_status_ledger_header",
                        path=rel,
                        detail=f"Missing columns: {', '.join(missing_fields)}",
                    )
                ]
            rows = list(reader)
    except csv.Error as exc:
        return [
            Issue(
                kind="invalid_m2_run_status_ledger_csv",
                path=rel,
                detail=str(exc),
            )
        ]

    preflight_failed = False
    source_report: Optional[str] = None
    preflight_report = repo_root / M2_GPU_PREFLIGHT_REPORT
    if preflight_report.is_file():
        try:
            preflight_payload = json.loads(preflight_report.read_text(encoding="utf-8"))
            preflight_failed = preflight_payload.get("passed") is False
            if isinstance(preflight_payload.get("source_report"), str):
                source_report = preflight_payload["source_report"]
        except json.JSONDecodeError:
            preflight_failed = False
    if source_report:
        source_report_path = Path(source_report)
        if source_report_path.is_absolute() or ".." in source_report_path.parts:
            issues.append(
                Issue(
                    kind="invalid_m2_run_status_source_report",
                    path=rel,
                    detail="source_report must be a relative in-repository path.",
                )
            )
            source_ledger_path = None
        else:
            source_ledger_path = (
                repo_root
                / source_report_path.parent
                / "blocked_run_status_ledger.csv"
            )
        if source_ledger_path is not None and not source_ledger_path.is_file():
            issues.append(
                Issue(
                    kind="missing_m2_run_status_source_ledger",
                    path=str(source_ledger_path.relative_to(repo_root)),
                    detail=(
                        "GPU preflight source_report requires sibling "
                        "blocked_run_status_ledger.csv."
                    ),
                )
            )
        elif source_ledger_path is not None:
            try:
                with source_ledger_path.open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    source_reader = csv.DictReader(handle)
                    source_fieldnames = source_reader.fieldnames or []
                    source_rows = list(source_reader)
            except csv.Error as exc:
                issues.append(
                    Issue(
                        kind="invalid_m2_run_status_source_ledger_csv",
                        path=str(source_ledger_path.relative_to(repo_root)),
                        detail=str(exc),
                    )
                )
            else:
                if source_fieldnames != fieldnames or source_rows != rows:
                    issues.append(
                        Issue(
                            kind="m2_run_status_ledger_source_mismatch",
                            path=rel,
                            detail=(
                                "Reviewable run-status ledger must mirror the "
                                "blocked_run_status_ledger.csv next to the "
                                "reviewable GPU preflight source_report."
                            ),
                        )
                    )

    expected_keys = {
        (dataset, method, seed)
        for dataset in M2_RUN_STATUS_EXPECTED_DATASETS
        for method in M2_RUN_STATUS_EXPECTED_METHODS
        for seed in M2_RUN_STATUS_EXPECTED_SEEDS
    }
    seen_keys: set[tuple[str, str, str]] = set()
    expected_row_count = len(expected_keys)

    if len(rows) != expected_row_count:
        issues.append(
            Issue(
                kind="invalid_m2_run_status_ledger_row_count",
                path=rel,
                detail=f"Expected {expected_row_count} rows, found {len(rows)}.",
            )
        )

    for index, row in enumerate(rows, start=2):
        for field in M2_RUN_STATUS_REQUIRED_FIELDS:
            if not (row.get(field) or "").strip():
                issues.append(
                    Issue(
                        kind="invalid_m2_run_status_ledger_empty_field",
                        path=rel,
                        detail=f"Row {index} has empty {field}.",
                    )
                )
                break

        dataset = (row.get("dataset") or "").strip()
        dataset_name = (row.get("dataset_name") or "").strip()
        method = (row.get("method") or "").strip()
        method_label = (row.get("method_label") or "").strip()
        seed = (row.get("seed") or "").strip()
        planned_stages = (row.get("planned_stages") or "").strip()
        status = (row.get("status") or "").strip()
        reason = (row.get("reason") or "").strip()
        status_lower = status.lower()

        expected_dataset_name = M2_RUN_STATUS_EXPECTED_DATASET_NAMES.get(dataset)
        if expected_dataset_name and dataset_name != expected_dataset_name:
            issues.append(
                Issue(
                    kind="invalid_m2_run_status_ledger_dataset_name",
                    path=rel,
                    detail=(
                        f"Row {index} dataset {dataset} expected name "
                        f"{expected_dataset_name}, found {dataset_name or '<empty>'}."
                    ),
                )
            )
        expected_method_label = M2_RUN_STATUS_EXPECTED_METHOD_LABELS.get(method)
        if expected_method_label and method_label != expected_method_label:
            issues.append(
                Issue(
                    kind="invalid_m2_run_status_ledger_method_label",
                    path=rel,
                    detail=(
                        f"Row {index} method {method} expected label "
                        f"{expected_method_label}, found {method_label or '<empty>'}."
                    ),
                )
            )

        if (
            status_lower not in M2_RUN_STATUS_ALLOWED_TERMINAL_STATUSES
            and not status.upper().startswith("BLOCKED")
        ):
            issues.append(
                Issue(
                    kind="invalid_m2_run_status_ledger_status",
                    path=rel,
                    detail=(
                        f"Row {index} has invalid status {status or '<empty>'}; "
                        "expected complete, failed, or BLOCKED_*."
                    ),
                )
            )
        if preflight_failed and status != "BLOCKED_GPU_PREFLIGHT":
            issues.append(
                Issue(
                    kind="invalid_m2_run_status_ledger_unblocked_failed_preflight",
                    path=rel,
                    detail=(
                        f"Row {index} must be BLOCKED_GPU_PREFLIGHT while "
                        "the reviewable GPU preflight report is failed."
                    ),
                )
            )
        if preflight_failed and not re.search(
            r"GPU|CUDA|nvidia-smi|NVIDIA", reason, flags=re.IGNORECASE
        ):
            issues.append(
                Issue(
                    kind="invalid_m2_run_status_ledger_blocked_reason_not_gpu",
                    path=rel,
                    detail=(
                        f"Row {index} blocked reason must identify the GPU/CUDA "
                        "resource blocker."
                    ),
                )
            )
        if planned_stages != M2_RUN_STATUS_EXPECTED_STAGES:
            issues.append(
                Issue(
                    kind="invalid_m2_run_status_ledger_stages",
                    path=rel,
                    detail=(
                        f"Row {index} expected {M2_RUN_STATUS_EXPECTED_STAGES}, "
                        f"found {planned_stages or '<empty>'}."
                    ),
                )
            )
        if status.upper().startswith("BLOCKED") or status.lower() == "failed":
            if not reason:
                issues.append(
                    Issue(
                        kind="invalid_m2_run_status_ledger_missing_reason",
                        path=rel,
                        detail=f"Row {index} has status {status} without reason.",
                    )
                )

        if dataset and method and seed:
            key = (dataset, method, seed)
            if key in seen_keys:
                issues.append(
                    Issue(
                        kind="duplicate_m2_run_status_ledger_row",
                        path=rel,
                        detail=f"Duplicate dataset/method/seed row: {key}.",
                    )
                )
            seen_keys.add(key)

    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)
        unexpected = sorted(seen_keys - expected_keys)
        issues.append(
            Issue(
                kind="invalid_m2_run_status_ledger_matrix",
                path=rel,
                detail=(
                    f"Missing matrix rows: {missing[:3]}"
                    f"{'...' if len(missing) > 3 else ''}; "
                    f"unexpected rows: {unexpected[:3]}"
                    f"{'...' if len(unexpected) > 3 else ''}."
                ),
            )
        )

    return issues


def check_feature_spec_contract(repo_root: Path) -> list[Issue]:
    spec_path = repo_root / M2_FEATURE_DIR / "spec.md"
    if not spec_path.exists() and not (repo_root / M2_SPECKIT_FREEZE_GOAL).exists():
        return []

    if not spec_path.is_file():
        return [
            Issue(
                kind="missing_active_feature_spec",
                path=str(spec_path.relative_to(repo_root)),
                detail="Active PHM-GenBench feature requires spec.md.",
            )
        ]

    rel = str(spec_path.relative_to(repo_root))
    text = spec_path.read_text(encoding="utf-8", errors="ignore")
    issues: list[Issue] = []
    for requirement_id in FEATURE_SPEC_REQUIRED_FR_IDS:
        if requirement_id not in text:
            issues.append(
                Issue(
                    kind="feature_spec_missing_functional_requirement",
                    path=rel,
                    detail=requirement_id,
                )
            )
    for success_id in FEATURE_SPEC_REQUIRED_SC_IDS:
        if success_id not in text:
            issues.append(
                Issue(
                    kind="feature_spec_missing_success_criterion",
                    path=rel,
                    detail=success_id,
                )
            )
    for snippet in FEATURE_SPEC_REQUIRED_SNIPPETS:
        if not _contains_normalized(text, snippet):
            issues.append(
                Issue(
                    kind="feature_spec_missing_contract_text",
                    path=rel,
                    detail=snippet,
                )
            )
    return issues


def check_constitution_contract(repo_root: Path) -> list[Issue]:
    constitution_path = repo_root / ".specify" / "memory" / "constitution.md"
    if not constitution_path.exists() and not (repo_root / ".specify" / "goals" / "v2").exists():
        return []

    if not constitution_path.is_file():
        return [
            Issue(
                kind="missing_phm_genbench_constitution",
                path=".specify/memory/constitution.md",
                detail="PHM-GenBench goals require the Speckit constitution.",
            )
        ]

    rel = str(constitution_path.relative_to(repo_root))
    text = constitution_path.read_text(encoding="utf-8", errors="ignore")
    issues: list[Issue] = []
    for snippet in CONSTITUTION_REQUIRED_SNIPPETS:
        if not _contains_normalized(text, snippet):
            issues.append(
                Issue(
                    kind="constitution_missing_contract_text",
                    path=rel,
                    detail=snippet,
                )
            )
    return issues


def check_root_phm_genbench_guidance(repo_root: Path) -> list[Issue]:
    if not (repo_root / ".specify" / "goals" / "v2").exists():
        return []

    issues: list[Issue] = []
    for rel, snippets in ROOT_PHM_GENBENCH_GUIDANCE_REQUIRED_SNIPPETS.items():
        path = repo_root / rel
        if not path.is_file():
            issues.append(
                Issue(
                    kind="missing_root_phm_genbench_guidance",
                    path=rel,
                    detail="Expected root guidance document.",
                )
            )
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in snippets:
            if not _contains_normalized(text, snippet):
                issues.append(
                    Issue(
                        kind="root_phm_genbench_guidance_missing_text",
                        path=rel,
                        detail=snippet,
                    )
                )
    return issues


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    doc_files = list(iter_doc_files(repo_root))

    issues: list[Issue] = []
    issues.extend(check_constitution_contract(repo_root))
    issues.extend(check_root_phm_genbench_guidance(repo_root))
    issues.extend(check_ai_docs_point_to_readme(repo_root))
    issues.extend(check_phm_generative_docs_placement(repo_root))
    issues.extend(check_required_phm_generative_readmes(repo_root))
    issues.extend(check_required_phm_generative_readme_content(repo_root))
    issues.extend(check_v2_goal_contracts(repo_root))
    issues.extend(check_v2_core_goal_queue(repo_root))
    issues.extend(check_goal_gen_003_review_templates(repo_root))
    issues.extend(check_v2_m2_goal_queue(repo_root))
    issues.extend(check_feature_m2_real_runs_goal_contract(repo_root))
    issues.extend(check_feature_speckit_artifacts(repo_root))
    issues.extend(check_feature_m2_analysis_contract(repo_root))
    issues.extend(check_feature_six_dataset_matrix(repo_root))
    issues.extend(check_feature_m2_dry_run_plan(repo_root))
    issues.extend(check_feature_review_handoff_artifacts(repo_root))
    issues.extend(check_feature_paper_artifacts(repo_root))
    issues.extend(check_feature_gpu_preflight_artifact(repo_root))
    issues.extend(check_goal_gen_completion_audit(repo_root))
    issues.extend(check_goal_gen_status_package(repo_root))
    issues.extend(check_feature_m2_gpu_runbook(repo_root))
    issues.extend(check_feature_m2_run_status_ledger(repo_root))
    issues.extend(check_feature_spec_contract(repo_root))
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
