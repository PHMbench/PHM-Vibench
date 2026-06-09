from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.validate_docs import (
    CONSTITUTION_REQUIRED_SNIPPETS,
    FEATURE_SPEC_REQUIRED_FR_IDS,
    FEATURE_SPEC_REQUIRED_SC_IDS,
    FEATURE_SPEC_REQUIRED_SNIPPETS,
    GOAL_GEN_003_HANDOFF_README_REQUIRED_SNIPPETS,
    GOAL_GEN_003_REVIEW_REQUIRED_SNIPPETS,
    GOAL_GEN_003_TEMPLATE_REQUIRED_SNIPPETS,
    GOAL_GEN_COMPLETION_AUDIT_REQUIRED_SNIPPETS,
    GOAL_GEN_STATUS_REPORT_REQUIRED_SNIPPETS,
    M2_ANALYSIS_REQUIRED_SNIPPETS,
    M2_DOWNSTREAM_GOAL_REQUIRED_SNIPPETS,
    M2_REVIEW_HANDOFF_GOAL_REQUIRED_SNIPPETS,
    M2_PAPER_DRAFT_REQUIRED_SECTIONS,
    M2_PAPER_DRAFT_REQUIRED_SNIPPETS,
    M2_REAL_RUNS_GOAL_REQUIRED_SNIPPETS,
    PHM_GENERATIVE_README_REQUIRED_SNIPPETS,
    REQUIRED_PHM_GENERATIVE_READMES,
    ROOT_PHM_GENBENCH_GUIDANCE_REQUIRED_SNIPPETS,
    check_constitution_contract,
    check_feature_gpu_preflight_artifact,
    check_feature_m2_gpu_runbook,
    check_feature_m2_real_runs_goal_contract,
    check_feature_m2_run_status_ledger,
    check_goal_gen_completion_audit,
    check_goal_gen_status_package,
    check_feature_m2_analysis_contract,
    check_feature_m2_dry_run_plan,
    check_feature_review_handoff_artifacts,
    check_feature_paper_artifacts,
    check_feature_six_dataset_matrix,
    check_feature_spec_contract,
    check_feature_speckit_artifacts,
    check_goal_gen_003_review_templates,
    check_phm_generative_docs_placement,
    check_required_phm_generative_readme_content,
    check_required_phm_generative_readmes,
    check_root_phm_genbench_guidance,
    check_v2_core_goal_queue,
    check_v2_m2_goal_queue,
    check_v2_goal_contracts,
)


M2_LEDGER_DATASETS = (
    ("RM_001_CWRU", "CWRU"),
    ("RM_002_XJTU", "XJTU"),
    ("RM_003_FEMTO", "FEMTO"),
    ("RM_008_UNSW", "UNSW"),
    ("RM_024_JUST", "JUST"),
    ("RM_027_PU", "PU"),
)
M2_LEDGER_METHODS = (
    ("cfm_grid", "Conditional Flow Matching"),
    ("ddpm_train_distribution", "DDPM Epsilon"),
    ("rectified_flow_grid", "Rectified Flow"),
)

M2_SPECKIT_FILES = (
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

M2_GOAL_FILES = (
    "GOAL-GEN-M2-000-speckit-freeze.md",
    "GOAL-GEN-M2-001-six-dataset-matrix-gpu.md",
    "GOAL-GEN-M2-002-multidataset-aggregation.md",
    "GOAL-GEN-M2-003-real-runs-evidence.md",
    "GOAL-GEN-M2-004-figures-tables.md",
    "GOAL-GEN-M2-005-markdown-paper-draft.md",
    "GOAL-GEN-M2-006-review-handoff.md",
)

CORE_GOAL_FILES = (
    "GOAL-GEN-000-repo-native-doc-pack.md",
    "GOAL-GEN-001-domain-id-contract.md",
    "GOAL-GEN-002-task-components-loss-spec.md",
    "GOAL-GEN-003-codex-claude-handoff.md",
    "GOAL-GEN-004-frontier-reference-map.md",
    "GOAL-GEN-M1-REPO-NATIVE.md",
)


def _m2_speckit_artifact_text(rel: str) -> str:
    if "checklists/" in rel:
        return "- [x] done\n"
    if rel == "quickstart.md":
        return (
            "Use the project `LQ_signal` environment because base Python may "
            "lack `torchmetrics`. The `Feature_factory-update` branch-name "
            "caveat is not as M2 evidence completion.\n"
        )
    return "# Artifact\n"


def _write_valid_claude_task_spec(run_dir: Path) -> None:
    (run_dir / "TASK_SPEC.md").write_text(
        "\n".join(
            [
                "# Task",
                "## Mode",
                "Read-only `review` mode first. Edits are not allowed.",
                "## Out Of Scope",
                "Do not push, publish, deploy, delete, or read secrets.",
                "## Teammates",
                "- Dataset protocol auditor.",
                "- Metrics and figures auditor.",
                "- Governance and leakage reviewer.",
                "## Required Outputs",
                "- `report.md`",
                "- `risks.md`",
                "- `test-log.md`",
            ]
        ),
        encoding="utf-8",
    )


def _write_valid_six_dataset_matrix(path: Path) -> None:
    datasets_yaml = "\n".join(
        [
            f'''  - dataset: "{dataset}"
    dataset_id: 1
    name: "{name}"
    overrides:
      task.target_system_id: [1]
      task.source_domain_id: [0, 1]
      task.target_domain_id: [2]
    protocol:
      utility: "domain_shift"
      notes: "{name} protocol."'''
            for dataset, name in M2_LEDGER_DATASETS
        ]
    )
    methods_yaml = "\n".join(
        [
            f'''  - method: "{method}"
    label: "{label}"
    train_config: "configs/paper/phm_generative/{method}.yaml"
    condition_sampling_policy: "grid"'''
            for method, label in M2_LEDGER_METHODS
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
benchmark:
  id: "phm_genbench_six_dataset_submission_v1"
  output_dir: "results/paper/phm_generative/six_dataset_submission_v1"
  baseline_method: "cfm_grid"
  min_datasets: 6
  data_check:
    data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
    metadata_file: "metadata.xlsx"
  seeds: [0, 1]
  resource:
    gpu_ids: [6, 7]
    max_parallel_runs: 2
    require_cuda: true
  overrides:
    data.normalization: "standardization"
    trainer.device: "cuda"
    trainer.gpus: 1
datasets:
{datasets_yaml}
methods:
{methods_yaml}
""".strip(),
        encoding="utf-8",
    )
    for method, _label in M2_LEDGER_METHODS:
        config_path = path.parents[3] / "configs" / "paper" / "phm_generative" / f"{method}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# config\n", encoding="utf-8")


def _write_valid_m2_dry_run_plan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stages = ("train", "sample", "eval", "paperpack")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "benchmark_id",
                "dataset",
                "dataset_id",
                "dataset_name",
                "method",
                "method_label",
                "seed",
                "stage",
                "gpu_id",
                "config",
                "command",
            ]
        )
        for dataset_index, (dataset, dataset_name) in enumerate(M2_LEDGER_DATASETS, start=1):
            for method, method_label in M2_LEDGER_METHODS:
                for seed in ("0", "1"):
                    for stage_index, stage in enumerate(stages):
                        gpu_id = str(6 + stage_index % 2)
                        config = f"configs/paper/phm_generative/{method}_seed{seed}.yaml"
                        if stage == "paperpack":
                            command = (
                                f"env CUDA_VISIBLE_DEVICES={gpu_id} "
                                "python -m scripts.paperpack_generative --run_dir runs"
                            )
                        else:
                            command = (
                                f"env CUDA_VISIBLE_DEVICES={gpu_id} "
                                f"python main.py --config {config} "
                                "--override trainer.device=cuda "
                                "--override trainer.gpus=1"
                            )
                        writer.writerow(
                            [
                                "phm_genbench_six_dataset_submission_v1",
                                dataset,
                                dataset_index,
                                dataset_name,
                                method,
                                method_label,
                                seed,
                                stage,
                                gpu_id,
                                config if stage != "paperpack" else "",
                                command,
                            ]
                        )


def _write_valid_paper_artifacts(
    paper_dir: Path,
    *,
    draft_status: str = "NOT_SUBMISSION_READY",
    readiness_status: str = "NOT_SUBMISSION_READY",
    extra_draft_text: str = "",
) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    section_text = "\n\n".join(
        f"{section}\n{section} content." for section in M2_PAPER_DRAFT_REQUIRED_SECTIONS
    )
    draft_text = "\n".join(
        [
            "# PHM-GenBench",
            f"**Draft status:** `{draft_status}`",
            "**Benchmark ID:** `phm_genbench_six_dataset_submission_v1`",
            section_text,
            "\n".join(M2_PAPER_DRAFT_REQUIRED_SNIPPETS),
            (
                "This draft is not submission-ready. No numerical claim is "
                "made. No computable benchmark rows are available."
                if draft_status == "NOT_SUBMISSION_READY"
                else ""
            ),
            extra_draft_text,
        ]
    )
    (paper_dir / "PAPER_DRAFT.md").write_text(draft_text, encoding="utf-8")
    (paper_dir / "submission_readiness.md").write_text(
        "\n".join(
            [
                "# M2 Submission Readiness",
                f"Status: `{readiness_status}`",
                "Reason:",
                "- missing evidence",
                "Promotion rule:",
                "The draft can be marked `SUBMISSION_READY` only when",
                "source paths are traceable.",
            ]
        ),
        encoding="utf-8",
    )
    (paper_dir / "evidence_gaps.md").write_text(
        "\n".join(
            [
                "# M2 Paper Evidence Gaps",
                "Summary: `summary.csv`",
                "Manifest: `manifest.json`",
                "Evidence gaps:",
                "- missing evidence",
            ]
        ),
        encoding="utf-8",
    )


def _write_ready_effect_evidence(repo_root: Path) -> None:
    effect_dir = (
        repo_root
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "effect"
    )
    effect_dir.mkdir(parents=True)
    rows: list[str] = [
        (
            "dataset,method,metric,category,mean,n,missing_count,rank,"
            "delta_vs_baseline,benchmark_status,metric_source_paths,manifest_paths"
        )
    ]
    dataset_ids = [dataset for dataset, _name in M2_LEDGER_DATASETS]
    for dataset in dataset_ids:
        metric_path = repo_root / "runs" / dataset / "metrics.csv"
        manifest_path = repo_root / "runs" / dataset / "manifest.json"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text("metric,value\nplaceholder,1\n", encoding="utf-8")
        manifest_path.write_text("{}\n", encoding="utf-8")
        rows.append(
            (
                f"{dataset},cfm_grid,temporal_l1,quality,1.0,2,0,1,0.0,"
                f"benchmark-valid,runs/{dataset}/metrics.csv,"
                f"runs/{dataset}/manifest.json"
            )
        )
        rows.append(
            (
                f"{dataset},cfm_grid,tstr_accuracy,utility,0.8,2,0,1,0.0,"
                f"benchmark-valid,runs/{dataset}/metrics.csv,"
                f"runs/{dataset}/manifest.json"
            )
        )
    (effect_dir / "benchmark_effect_summary.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (effect_dir / "benchmark_effect_manifest.json").write_text(
        """
{
  "min_datasets_met": true,
  "observed_configured_datasets": [
    "RM_001_CWRU",
    "RM_002_XJTU",
    "RM_003_FEMTO",
    "RM_008_UNSW",
    "RM_024_JUST",
    "RM_027_PU"
  ],
  "observed_configured_dataset_count": 6,
  "input_gaps": [],
  "missing_datasets": [],
  "unexpected_datasets": []
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    ledger_path = (
        repo_root
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_lines = [
        "dataset,dataset_name,method,method_label,seed,planned_stages,status,reason"
    ]
    for dataset, dataset_name in M2_LEDGER_DATASETS:
        for method, method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                ledger_lines.append(
                    ",".join(
                        [
                            dataset,
                            dataset_name,
                            method,
                            method_label,
                            str(seed),
                            "train;sample;eval;paperpack",
                            "complete",
                            "complete",
                        ]
                    )
                )
    ledger_path.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")


def test_validate_docs_allows_repo_without_deprecated_generative_docs(tmp_path: Path) -> None:
    issues = check_phm_generative_docs_placement(tmp_path)

    assert issues == []


def test_validate_docs_blocks_deprecated_generative_docs_dirs(tmp_path: Path) -> None:
    (tmp_path / "docs" / "phm_generative").mkdir(parents=True)
    (tmp_path / "docs" / "generative").mkdir(parents=True)

    issues = check_phm_generative_docs_placement(tmp_path)

    assert {issue.kind for issue in issues} == {"forbidden_phm_generative_path"}
    assert {issue.path for issue in issues} == {"docs/phm_generative", "docs/generative"}


def test_validate_docs_blocks_forbidden_generative_runtime_and_project_dirs(
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "phm_factory").mkdir(parents=True)
    (tmp_path / "projects" / "phm_generative").mkdir(parents=True)
    (tmp_path / "packs").mkdir()
    (tmp_path / "templates").mkdir()
    (tmp_path / "schemas").mkdir()

    issues = check_phm_generative_docs_placement(tmp_path)

    assert {issue.kind for issue in issues} == {"forbidden_phm_generative_path"}
    assert {
        "src/phm_factory",
        "projects",
        "projects/phm_generative",
        "packs",
        "templates",
        "schemas",
    }.issubset({issue.path for issue in issues})


def test_validate_docs_blocks_legacy_generative_docs_index_references(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "configs" / "config_registry.csv"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "id,docs\nx,docs/phm_generative\n",
        encoding="utf-8",
    )
    atlas = tmp_path / "docs" / "CONFIG_ATLAS.md"
    atlas.parent.mkdir(parents=True)
    atlas.write_text("See docs/generative\n", encoding="utf-8")

    issues = check_phm_generative_docs_placement(tmp_path)

    assert {issue.kind for issue in issues} == {
        "legacy_phm_generative_doc_reference"
    }
    assert {issue.path for issue in issues} == {
        "configs/config_registry.csv",
        "docs/CONFIG_ATLAS.md",
    }


def test_validate_docs_reports_one_legacy_docs_reference_per_index(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "configs" / "config_registry.csv"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "id,docs\nx,docs/phm_generative/PAPER_TABLES_AND_FIGURES.md\n",
        encoding="utf-8",
    )

    issues = check_phm_generative_docs_placement(tmp_path)

    assert [issue.kind for issue in issues] == [
        "legacy_phm_generative_doc_reference"
    ]
    assert issues[0].path == "configs/config_registry.csv"


def test_validate_docs_allows_similar_non_legacy_docs_reference(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "configs" / "config_registry.csv"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "id,docs\nx,docs/generative_model_notes.md\n",
        encoding="utf-8",
    )

    issues = check_phm_generative_docs_placement(tmp_path)

    assert issues == []


def test_validate_docs_accepts_required_phm_generative_readmes(tmp_path: Path) -> None:
    (tmp_path / ".specify" / "goals" / "v2").mkdir(parents=True)
    for rel in REQUIRED_PHM_GENERATIVE_READMES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# README\n", encoding="utf-8")

    issues = check_required_phm_generative_readmes(tmp_path)

    assert issues == []


def test_validate_docs_rejects_missing_required_phm_generative_readmes(tmp_path: Path) -> None:
    (tmp_path / ".specify" / "goals" / "v2").mkdir(parents=True)

    issues = check_required_phm_generative_readmes(tmp_path)

    assert {issue.kind for issue in issues} == {"missing_phm_generative_module_readme"}
    assert {issue.path for issue in issues} == set(REQUIRED_PHM_GENERATIVE_READMES)


def test_validate_docs_accepts_required_phm_generative_readme_content(
    tmp_path: Path,
) -> None:
    (tmp_path / ".specify" / "goals" / "v2").mkdir(parents=True)
    for rel, snippets in PHM_GENERATIVE_README_REQUIRED_SNIPPETS.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(snippets), encoding="utf-8")

    issues = check_required_phm_generative_readme_content(tmp_path)

    assert issues == []


def test_validate_docs_rejects_missing_required_phm_generative_readme_content(
    tmp_path: Path,
) -> None:
    (tmp_path / ".specify" / "goals" / "v2").mkdir(parents=True)
    rel = "src/task_factory/task/generative/README.md"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text("fault_label\n", encoding="utf-8")

    issues = check_required_phm_generative_readme_content(tmp_path)

    assert {issue.kind for issue in issues} == {
        "missing_phm_generative_readme_contract_text"
    }
    assert {issue.path for issue in issues} == {rel}


def test_validate_docs_accepts_phm_genbench_constitution_contract(
    tmp_path: Path,
) -> None:
    (tmp_path / ".specify" / "goals" / "v2").mkdir(parents=True)
    constitution = tmp_path / ".specify" / "memory" / "constitution.md"
    constitution.parent.mkdir(parents=True)
    constitution.write_text("\n".join(CONSTITUTION_REQUIRED_SNIPPETS), encoding="utf-8")

    issues = check_constitution_contract(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_phm_genbench_constitution_contract(
    tmp_path: Path,
) -> None:
    (tmp_path / ".specify" / "goals" / "v2").mkdir(parents=True)
    constitution = tmp_path / ".specify" / "memory" / "constitution.md"
    constitution.parent.mkdir(parents=True)
    constitution.write_text("python main.py --config <yaml>\n", encoding="utf-8")

    issues = check_constitution_contract(tmp_path)

    assert {issue.kind for issue in issues} == {"constitution_missing_contract_text"}
    assert all(issue.path == ".specify/memory/constitution.md" for issue in issues)


def test_validate_docs_accepts_active_feature_spec_contract(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-000-speckit-freeze.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    spec_path = tmp_path / "specs" / "002-phm-genbench-frontier" / "spec.md"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        "\n".join(
            (
                "Feature Specification: PHM-GenBench Frontier",
                *FEATURE_SPEC_REQUIRED_FR_IDS,
                *FEATURE_SPEC_REQUIRED_SC_IDS,
                *FEATURE_SPEC_REQUIRED_SNIPPETS,
            )
        ),
        encoding="utf-8",
    )

    issues = check_feature_spec_contract(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_active_feature_spec_contract(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-000-speckit-freeze.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    spec_path = tmp_path / "specs" / "002-phm-genbench-frontier" / "spec.md"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("FR-001\nSC-001\n", encoding="utf-8")

    issues = check_feature_spec_contract(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "feature_spec_missing_functional_requirement" in kinds
    assert "feature_spec_missing_success_criterion" in kinds
    assert "feature_spec_missing_contract_text" in kinds


def test_validate_docs_accepts_root_phm_genbench_guidance(tmp_path: Path) -> None:
    (tmp_path / ".specify" / "goals" / "v2").mkdir(parents=True)
    for rel, snippets in ROOT_PHM_GENBENCH_GUIDANCE_REQUIRED_SNIPPETS.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(snippets), encoding="utf-8")

    issues = check_root_phm_genbench_guidance(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_root_phm_genbench_guidance(
    tmp_path: Path,
) -> None:
    (tmp_path / ".specify" / "goals" / "v2").mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text(".specify/memory/constitution.md\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("# Claude\n", encoding="utf-8")

    issues = check_root_phm_genbench_guidance(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "missing_root_phm_genbench_guidance" in kinds
    assert "root_phm_genbench_guidance_missing_text" in kinds


def test_validate_docs_accepts_v2_goal_contract_shape(tmp_path: Path) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    (goal_dir / "GOAL-GEN-999-example.md").write_text(
        "\n".join(
            [
                "# GOAL-GEN-999: Example",
                "",
                "## Goal ID",
                "",
                "GOAL-GEN-999",
                "",
                "## Objective",
                "",
                "Example.",
                "",
                "## Scope",
                "",
                "Example.",
                "",
                "## Required Behavior",
                "",
                "Example.",
                "",
                "## Acceptance Criteria",
                "",
                "Example.",
                "",
                "## Validation Commands",
                "",
                "```bash",
                "python -m scripts.validate_docs",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    issues = check_v2_goal_contracts(tmp_path)

    assert issues == []


def test_validate_docs_rejects_v2_goal_without_parseable_goal_id(tmp_path: Path) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    (goal_dir / "GOAL-GEN-999-example.md").write_text(
        "\n".join(
            [
                "# GOAL-GEN-999: Example",
                "",
                "## Objective",
                "",
                "Example.",
                "",
                "## Scope",
                "",
                "Example.",
                "",
                "## Required Behavior",
                "",
                "Example.",
                "",
                "## Acceptance Criteria",
                "",
                "Example.",
                "",
                "## Validation Commands",
                "",
                "```bash",
                "python -m scripts.validate_docs",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    issues = check_v2_goal_contracts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "v2_goal_missing_section",
        "v2_goal_invalid_goal_id",
    }


def test_validate_docs_rejects_v2_goal_id_filename_mismatch(tmp_path: Path) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    (goal_dir / "GOAL-GEN-999-example.md").write_text(
        "\n".join(
            [
                "# GOAL-GEN-999: Example",
                "",
                "## Goal ID",
                "",
                "GOAL-GEN-998",
                "",
                "## Objective",
                "",
                "Example.",
                "",
                "## Scope",
                "",
                "Example.",
                "",
                "## Required Behavior",
                "",
                "Example.",
                "",
                "## Acceptance Criteria",
                "",
                "Example.",
                "",
                "## Validation Commands",
                "",
                "```bash",
                "python -m scripts.validate_docs",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    issues = check_v2_goal_contracts(tmp_path)

    assert [issue.kind for issue in issues] == ["v2_goal_id_filename_mismatch"]


def test_validate_docs_rejects_v2_goal_legacy_docs_allowed_target(
    tmp_path: Path,
) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    (goal_dir / "GOAL-GEN-999-example.md").write_text(
        "\n".join(
            [
                "# GOAL-GEN-999: Example",
                "",
                "## Goal ID",
                "",
                "GOAL-GEN-999",
                "",
                "## Objective",
                "",
                "Example.",
                "",
                "## Scope",
                "",
                "Allowed to add:",
                "",
                "- `docs/phm_generative/README.md`",
                "",
                "## Required Behavior",
                "",
                "Example.",
                "",
                "## Acceptance Criteria",
                "",
                "Example.",
                "",
                "## Validation Commands",
                "",
                "```bash",
                "python -m scripts.validate_docs",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    issues = check_v2_goal_contracts(tmp_path)

    assert [issue.kind for issue in issues] == [
        "v2_goal_legacy_docs_allowed_target"
    ]


def test_validate_docs_allows_v2_goal_legacy_docs_prohibition(
    tmp_path: Path,
) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    (goal_dir / "GOAL-GEN-999-example.md").write_text(
        "\n".join(
            [
                "# GOAL-GEN-999: Example",
                "",
                "## Goal ID",
                "",
                "GOAL-GEN-999",
                "",
                "## Objective",
                "",
                "Example.",
                "",
                "## Scope",
                "",
                "Use module READMEs. Do not create `docs/phm_generative/` or",
                "`docs/generative/`.",
                "",
                "## Required Behavior",
                "",
                "Example.",
                "",
                "## Acceptance Criteria",
                "",
                "Example.",
                "",
                "## Validation Commands",
                "",
                "```bash",
                "python -m scripts.validate_docs",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    issues = check_v2_goal_contracts(tmp_path)

    assert issues == []


def test_validate_docs_accepts_complete_core_goal_queue(tmp_path: Path) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    for filename in CORE_GOAL_FILES:
        (goal_dir / filename).write_text(
            "README\nsrc/task_factory\ndomain_id\nlosses\n"
            "specs/<active-feature>\nreference\n"
            "Subagent/teammate acceleration\n",
            encoding="utf-8",
        )

    issues = check_v2_core_goal_queue(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_core_goal_queue(tmp_path: Path) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    (goal_dir / "GOAL-GEN-000-repo-native-doc-pack.md").write_text(
        "README\n",
        encoding="utf-8",
    )

    issues = check_v2_core_goal_queue(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "missing_core_goal_file" in kinds
    assert "core_goal_missing_required_scope_text" in kinds


def test_validate_docs_accepts_goal_gen_003_review_templates(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-003-codex-claude-handoff.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    artifacts = {
        "specs/002-phm-genbench-frontier/reviews/README.md": (
            GOAL_GEN_003_REVIEW_REQUIRED_SNIPPETS
        ),
        (
            "specs/002-phm-genbench-frontier/reviews/claude-team/"
            "phm-gen-general-review-template/TASK_SPEC.md"
        ): GOAL_GEN_003_TEMPLATE_REQUIRED_SNIPPETS,
        "specs/002-phm-genbench-frontier/handoffs/README.md": (
            GOAL_GEN_003_HANDOFF_README_REQUIRED_SNIPPETS
        ),
    }
    for rel, snippets in artifacts.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(snippets), encoding="utf-8")

    issues = check_goal_gen_003_review_templates(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_goal_gen_003_review_templates(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-003-codex-claude-handoff.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    readme = tmp_path / "specs" / "002-phm-genbench-frontier" / "reviews" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("phm-gen-architect\n", encoding="utf-8")

    issues = check_goal_gen_003_review_templates(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "missing_goal_gen_003_review_template_artifact" in kinds
    assert "goal_gen_003_review_template_missing_text" in kinds


def test_validate_docs_accepts_complete_m2_goal_queue(tmp_path: Path) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    for filename in M2_GOAL_FILES:
        text = "specs/002-phm-genbench-frontier/\n"
        if filename == "GOAL-GEN-M2-006-review-handoff.md":
            text += "\n".join(M2_REVIEW_HANDOFF_GOAL_REQUIRED_SNIPPETS)
        if filename in M2_DOWNSTREAM_GOAL_REQUIRED_SNIPPETS:
            text += "\n" + "\n".join(M2_DOWNSTREAM_GOAL_REQUIRED_SNIPPETS[filename])
        (goal_dir / filename).write_text(text, encoding="utf-8")

    issues = check_v2_m2_goal_queue(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_m2_goal_queue(tmp_path: Path) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    (goal_dir / "GOAL-GEN-M2-000-speckit-freeze.md").write_text(
        "No active feature reference.\n",
        encoding="utf-8",
    )

    issues = check_v2_m2_goal_queue(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "missing_m2_goal_file" in kinds
    assert "m2_goal_missing_active_feature_reference" in kinds


def test_validate_docs_rejects_m2_review_goal_without_claude_team_contract(
    tmp_path: Path,
) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    for filename in M2_GOAL_FILES:
        (goal_dir / filename).write_text(
            "specs/002-phm-genbench-frontier/\n",
            encoding="utf-8",
        )

    issues = check_v2_m2_goal_queue(tmp_path)

    assert "m2_review_goal_missing_claude_team_contract_text" in {
        issue.kind for issue in issues
    }


def test_validate_docs_rejects_m2_downstream_goal_without_dependency_text(
    tmp_path: Path,
) -> None:
    goal_dir = tmp_path / ".specify" / "goals" / "v2"
    goal_dir.mkdir(parents=True)
    for filename in M2_GOAL_FILES:
        text = "specs/002-phm-genbench-frontier/\n"
        if filename == "GOAL-GEN-M2-006-review-handoff.md":
            text += "\n".join(M2_REVIEW_HANDOFF_GOAL_REQUIRED_SNIPPETS)
        (goal_dir / filename).write_text(text, encoding="utf-8")

    issues = check_v2_m2_goal_queue(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "m2_downstream_goal_missing_dependency_text" in kinds
    details = {issue.detail for issue in issues}
    assert "This goal's real-evidence completion is task `T048`" in details
    assert "This goal's final-evidence completion is task `T049`" in details
    assert "This goal's submission-ready completion is task `T050`" in details
    assert "This goal's final review completion is task `T051`" in details


def test_validate_docs_accepts_m2_real_runs_goal_contract(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("\n".join(M2_REAL_RUNS_GOAL_REQUIRED_SNIPPETS), encoding="utf-8")

    issues = check_feature_m2_real_runs_goal_contract(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_m2_real_runs_goal_contract(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("CUDA_VISIBLE_DEVICES=6 python -c\n", encoding="utf-8")

    issues = check_feature_m2_real_runs_goal_contract(tmp_path)

    assert {
        "m2_real_runs_goal_missing_required_text",
    } == {issue.kind for issue in issues}
    assert any(issue.detail == "--stages train" for issue in issues)


def test_validate_docs_accepts_complete_m2_speckit_artifacts(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-000-speckit-freeze.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    feature_dir = tmp_path / "specs" / "002-phm-genbench-frontier"
    for rel in M2_SPECKIT_FILES:
        path = feature_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_m2_speckit_artifact_text(rel), encoding="utf-8")

    issues = check_feature_speckit_artifacts(tmp_path)

    assert issues == []


def test_validate_docs_rejects_m2_quickstart_without_execution_caveat(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-000-speckit-freeze.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    feature_dir = tmp_path / "specs" / "002-phm-genbench-frontier"
    for rel in M2_SPECKIT_FILES:
        path = feature_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_m2_speckit_artifact_text(rel), encoding="utf-8")
    (feature_dir / "quickstart.md").write_text("# Quickstart\n", encoding="utf-8")

    issues = check_feature_speckit_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_quickstart_missing_execution_caveat"
    }


def test_validate_docs_rejects_incomplete_m2_speckit_artifacts(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-000-speckit-freeze.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    feature_dir = tmp_path / "specs" / "002-phm-genbench-frontier"
    checklist = feature_dir / "checklists" / "requirements.md"
    checklist.parent.mkdir(parents=True)
    checklist.write_text("- [ ] missing\n", encoding="utf-8")

    issues = check_feature_speckit_artifacts(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "missing_m2_speckit_artifact" in kinds
    assert "incomplete_m2_speckit_checklist" in kinds


def test_validate_docs_rejects_legacy_goal_text_in_active_speckit_artifacts(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-000-speckit-freeze.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    feature_dir = tmp_path / "specs" / "002-phm-genbench-frontier"
    for rel in M2_SPECKIT_FILES:
        path = feature_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_m2_speckit_artifact_text(rel), encoding="utf-8")
    (feature_dir / "quickstart.md").write_text("GOAL-FFU-P0-002\n", encoding="utf-8")

    issues = check_feature_speckit_artifacts(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "active_feature_artifact_forbidden_legacy_goal_text" in kinds
    assert "m2_quickstart_missing_execution_caveat" in kinds


def test_validate_docs_requires_open_m2_real_gpu_task_when_preflight_failed(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-000-speckit-freeze.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    feature_dir = tmp_path / "specs" / "002-phm-genbench-frontier"
    for rel in M2_SPECKIT_FILES:
        path = feature_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_m2_speckit_artifact_text(rel), encoding="utf-8")
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")

    issues = check_feature_speckit_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_tasks_missing_open_real_gpu_run_task"
    }


def test_validate_docs_accepts_open_m2_real_gpu_task_when_preflight_failed(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-000-speckit-freeze.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    feature_dir = tmp_path / "specs" / "002-phm-genbench-frontier"
    for rel in M2_SPECKIT_FILES:
        path = feature_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_m2_speckit_artifact_text(rel), encoding="utf-8")
    (feature_dir / "tasks.md").write_text(
        "\n".join(
            [
                "- [ ] T047 [M2-003] Execute real six-dataset GPU runs",
                "- [ ] T048 [M2-002] Aggregate real six-dataset run directories",
                "- [ ] T049 [M2-004] Generate final paper tables and figure sources",
                "- [ ] T050 [M2-005] Regenerate the submission draft",
                "- [ ] T051 [M2-006] Run final Codex verification and advisory review",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")

    issues = check_feature_speckit_artifacts(tmp_path)

    assert issues == []


def test_validate_docs_requires_downstream_m2_evidence_tasks_when_preflight_failed(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-000-speckit-freeze.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    feature_dir = tmp_path / "specs" / "002-phm-genbench-frontier"
    for rel in M2_SPECKIT_FILES:
        path = feature_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_m2_speckit_artifact_text(rel), encoding="utf-8")
    (feature_dir / "tasks.md").write_text(
        "- [ ] T047 [M2-003] Execute real six-dataset GPU runs\n",
        encoding="utf-8",
    )
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")

    issues = check_feature_speckit_artifacts(tmp_path)

    assert [issue.kind for issue in issues] == ["m2_tasks_missing_open_real_gpu_run_task"]
    assert "T048/M2-002 real aggregation" in issues[0].detail
    assert "T049/M2-004 final figures/tables" in issues[0].detail
    assert "T050/M2-005 submission draft" in issues[0].detail
    assert "T051/M2-006 final review" in issues[0].detail
    assert "T047/M2-003 real GPU execution" not in issues[0].detail


def test_validate_docs_accepts_m2_analysis_contract(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-002-multidataset-aggregation.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    analysis = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "analysis"
        / "m2-cross-artifact-analysis.md"
    )
    analysis.parent.mkdir(parents=True)
    analysis.write_text("\n".join(M2_ANALYSIS_REQUIRED_SNIPPETS), encoding="utf-8")

    issues = check_feature_m2_analysis_contract(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_m2_analysis_contract(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-002-multidataset-aggregation.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    analysis = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "analysis"
        / "m2-cross-artifact-analysis.md"
    )
    analysis.parent.mkdir(parents=True)
    analysis.write_text("M2 analysis\n", encoding="utf-8")

    issues = check_feature_m2_analysis_contract(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_cross_artifact_analysis_missing_contract_text"
    }


def test_validate_docs_accepts_m2_six_dataset_matrix(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-001-six-dataset-matrix-gpu.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    _write_valid_six_dataset_matrix(
        tmp_path / "configs" / "paper" / "phm_generative" / "six_dataset_benchmark_matrix.yaml"
    )

    issues = check_feature_six_dataset_matrix(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_m2_six_dataset_matrix(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-001-six-dataset-matrix-gpu.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    matrix_path = (
        tmp_path
        / "configs"
        / "paper"
        / "phm_generative"
        / "six_dataset_benchmark_matrix.yaml"
    )
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        """
benchmark:
  id: "wrong"
  min_datasets: 1
  seeds: [0]
  resource:
    gpu_ids: [0]
    max_parallel_runs: 1
    require_cuda: false
  overrides:
    trainer.device: "cpu"
    trainer.gpus: 0
datasets:
  - dataset: "RM_001_CWRU"
methods:
  - method: "cfm_grid"
""".strip(),
        encoding="utf-8",
    )

    issues = check_feature_six_dataset_matrix(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "m2_six_dataset_matrix_invalid_benchmark_id" in kinds
    assert "m2_six_dataset_matrix_invalid_gpu_ids" in kinds
    assert "m2_six_dataset_matrix_invalid_datasets" in kinds
    assert "m2_six_dataset_matrix_invalid_methods" in kinds


def test_validate_docs_accepts_m2_dry_run_plan(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-001-six-dataset-matrix-gpu.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    _write_valid_m2_dry_run_plan(
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "dry_run_current_audit"
        / "run_plan.csv"
    )

    issues = check_feature_m2_dry_run_plan(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_m2_dry_run_plan(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-001-six-dataset-matrix-gpu.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    plan_path = (
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "dry_run_current_audit"
        / "run_plan.csv"
    )
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "benchmark_id",
                "dataset",
                "dataset_id",
                "dataset_name",
                "method",
                "method_label",
                "seed",
                "stage",
                "gpu_id",
                "config",
                "command",
            ]
        )
        writer.writerow(
            [
                "phm_genbench_six_dataset_submission_v1",
                "RM_001_CWRU",
                "1",
                "CWRU",
                "cfm_grid",
                "Conditional Flow Matching",
                "0",
                "train",
                "0",
                "configs/paper/phm_generative/cfm_grid_seed0.yaml",
                "python main.py --config configs/paper/phm_generative/cfm_grid_seed0.yaml",
            ]
        )

    issues = check_feature_m2_dry_run_plan(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "invalid_m2_dry_run_plan_row_count" in kinds
    assert "invalid_m2_dry_run_plan_gpu" in kinds
    assert "m2_dry_run_plan_command_missing_cuda_visible_devices" in kinds
    assert "m2_dry_run_plan_command_missing_cuda_trainer_override" in kinds
    assert "invalid_m2_dry_run_plan_matrix" in kinds


def test_validate_docs_accepts_blocked_claude_review_and_handoff(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "specs"
        / "002-feature"
        / "reviews"
        / "claude-team"
        / "2026-05-11-run"
    )
    run_dir.mkdir(parents=True)
    _write_valid_claude_task_spec(run_dir)
    (run_dir / "report.md").write_text(
        "\n".join(
            [
                "Status: `BLOCKED_NOT_RUN`",
                "<REVIEW_DECISION>BLOCKING</REVIEW_DECISION>",
                "<BLOCKING_ISSUES>",
                "Endpoint approval missing.",
                "</BLOCKING_ISSUES>",
                "<NON_BLOCKING_ISSUES>",
                "None.",
                "</NON_BLOCKING_ISSUES>",
                "<FIX_INSTRUCTION>",
                "Get endpoint approval.",
                "</FIX_INSTRUCTION>",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "risks.md").write_text("Status: `BLOCKED_NOT_RUN`\n", encoding="utf-8")
    (run_dir / "test-log.md").write_text(
        "Status: `BLOCKED_NOT_RUN`\n", encoding="utf-8"
    )
    handoff_dir = tmp_path / "specs" / "002-feature" / "handoffs"
    handoff_dir.mkdir(parents=True)
    (handoff_dir / "2026-05-11.md").write_text(
        "\n".join(
            [
                "## Current State",
                "Blocked.",
                "## Goal ID",
                "GOAL-GEN-M2-006",
                "## Objective",
                "Review.",
                "## Files Changed",
                "Review files.",
                "## Runtime Behavior Changed",
                "No.",
                "## Contracts Touched",
                "Review/handoff.",
                "## Validation Commands Run",
                "python -m scripts.validate_docs",
                "## Validation Results",
                "Passed.",
                "## Known Risks",
                "Review blocked.",
                "## Required Reviewers",
                "Dataset protocol auditor.",
                "## Required Context Files",
                "spec.md.",
                "## Review Output Format",
                "<REVIEW_DECISION>BLOCKING</REVIEW_DECISION>",
                "## Next Steps",
                "Fix GPU.",
            ]
        ),
        encoding="utf-8",
    )

    issues = check_feature_review_handoff_artifacts(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_blocked_claude_review(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "specs"
        / "002-feature"
        / "reviews"
        / "claude-team"
        / "2026-05-11-run"
    )
    run_dir.mkdir(parents=True)
    _write_valid_claude_task_spec(run_dir)
    (run_dir / "report.md").write_text(
        "Status: `BLOCKED_NOT_RUN`\n<REVIEW_DECISION>BLOCKING</REVIEW_DECISION>\n",
        encoding="utf-8",
    )
    (run_dir / "risks.md").write_text("# Risks\n", encoding="utf-8")
    handoff_dir = tmp_path / "specs" / "002-feature" / "handoffs"
    handoff_dir.mkdir(parents=True)
    (handoff_dir / "2026-05-11.md").write_text("## Goal ID\n", encoding="utf-8")

    issues = check_feature_review_handoff_artifacts(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "missing_claude_team_artifact" in kinds
    assert "claude_team_report_missing_fix_instruction" in kinds
    assert "claude_team_report_missing_required_tag" in kinds
    assert "blocked_claude_artifact_missing_status" in kinds
    assert "handoff_missing_section" in kinds


def test_validate_docs_rejects_unsafe_claude_task_spec(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "specs"
        / "002-feature"
        / "reviews"
        / "claude-team"
        / "2026-05-11-run"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "TASK_SPEC.md").write_text(
        "# Task\n## Mode\nImplementation mode.\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        "\n".join(
            [
                "<REVIEW_DECISION>REQUEST_CHANGES</REVIEW_DECISION>",
                "<BLOCKING_ISSUES>",
                "Task spec is unsafe.",
                "</BLOCKING_ISSUES>",
                "<NON_BLOCKING_ISSUES>",
                "None.",
                "</NON_BLOCKING_ISSUES>",
                "<FIX_INSTRUCTION>",
                "Restore read-only review-mode task spec.",
                "</FIX_INSTRUCTION>",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "risks.md").write_text("# Risks\n", encoding="utf-8")
    (run_dir / "test-log.md").write_text("# Tests\n", encoding="utf-8")

    issues = check_feature_review_handoff_artifacts(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "claude_team_task_spec_missing_required_text" in kinds
    assert "claude_team_task_spec_insufficient_teammates" in kinds


def test_validate_docs_rejects_invalid_claude_review_decision(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "specs"
        / "002-feature"
        / "reviews"
        / "claude-team"
        / "2026-05-11-run"
    )
    run_dir.mkdir(parents=True)
    _write_valid_claude_task_spec(run_dir)
    (run_dir / "report.md").write_text(
        "\n".join(
            [
                "<REVIEW_DECISION>OK</REVIEW_DECISION>",
                "<BLOCKING_ISSUES>",
                "None.",
                "</BLOCKING_ISSUES>",
                "<NON_BLOCKING_ISSUES>",
                "None.",
                "</NON_BLOCKING_ISSUES>",
                "<FIX_INSTRUCTION>",
                "Use one of the allowed decisions.",
                "</FIX_INSTRUCTION>",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "risks.md").write_text("# Risks\n", encoding="utf-8")
    (run_dir / "test-log.md").write_text("# Tests\n", encoding="utf-8")

    issues = check_feature_review_handoff_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "claude_team_report_invalid_review_decision"
    }


def test_validate_docs_rejects_blocked_claude_review_with_approve_decision(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "specs"
        / "002-feature"
        / "reviews"
        / "claude-team"
        / "2026-05-11-run"
    )
    run_dir.mkdir(parents=True)
    _write_valid_claude_task_spec(run_dir)
    (run_dir / "report.md").write_text(
        "\n".join(
            [
                "Status: `BLOCKED_NOT_RUN`",
                "<REVIEW_DECISION>APPROVE</REVIEW_DECISION>",
                "<BLOCKING_ISSUES>",
                "Endpoint approval missing.",
                "</BLOCKING_ISSUES>",
                "<NON_BLOCKING_ISSUES>",
                "None.",
                "</NON_BLOCKING_ISSUES>",
                "<FIX_INSTRUCTION>",
                "Get endpoint approval.",
                "</FIX_INSTRUCTION>",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "risks.md").write_text("Status: `BLOCKED_NOT_RUN`\n", encoding="utf-8")
    (run_dir / "test-log.md").write_text(
        "Status: `BLOCKED_NOT_RUN`\n", encoding="utf-8"
    )

    issues = check_feature_review_handoff_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "blocked_claude_review_non_blocking_decision"
    }


def test_validate_docs_rejects_trailing_claude_review_text(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "specs"
        / "002-feature"
        / "reviews"
        / "claude-team"
        / "2026-05-11-run"
    )
    run_dir.mkdir(parents=True)
    _write_valid_claude_task_spec(run_dir)
    (run_dir / "report.md").write_text(
        "\n".join(
            [
                "<REVIEW_DECISION>APPROVE</REVIEW_DECISION>",
                "<BLOCKING_ISSUES>",
                "None.",
                "</BLOCKING_ISSUES>",
                "<NON_BLOCKING_ISSUES>",
                "None.",
                "</NON_BLOCKING_ISSUES>",
                "<FIX_INSTRUCTION>",
                "No patch required.",
                "</FIX_INSTRUCTION>",
                "extra trailing text",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "risks.md").write_text("# Risks\n", encoding="utf-8")
    (run_dir / "test-log.md").write_text("# Tests\n", encoding="utf-8")

    issues = check_feature_review_handoff_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "claude_team_report_trailing_text_after_fix_instruction"
    }


def test_validate_docs_requires_m2_review_handoff_artifacts_when_goal_exists(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-006-review-handoff.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    (tmp_path / "specs" / "002-phm-genbench-frontier").mkdir(parents=True)

    issues = check_feature_review_handoff_artifacts(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "missing_m2_claude_team_run" in kinds
    assert "missing_m2_handoff" in kinds


def test_validate_docs_accepts_m2_paper_artifacts(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(paper_dir)

    issues = check_feature_paper_artifacts(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_m2_paper_artifacts(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="NOT_SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {"m2_paper_status_mismatch"}


def test_validate_docs_rejects_submission_ready_paper_without_effect_evidence(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert "m2_paper_ready_missing_evidence_file" in {issue.kind for issue in issues}


def test_validate_docs_rejects_submission_ready_paper_when_gpu_preflight_failed(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    report_goal = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    report_goal.write_text("# Goal\n", encoding="utf-8")
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        """
{
  "benchmark_id": "phm_genbench_six_dataset_submission_v1",
  "matrix_path": "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml",
  "require_cuda": true,
  "gpu_ids": ["6", "7"],
  "max_parallel_runs": 2,
  "passed": false,
  "results": [
    {"gpu_id": "6", "status": "failed", "error": "GPU 6 failed CUDA preflight"},
    {"gpu_id": "7", "status": "failed", "error": "GPU 7 failed CUDA preflight"}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    _write_ready_effect_evidence(tmp_path)
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_paper_ready_while_gpu_preflight_failed"
    }


def test_validate_docs_rejects_submission_ready_paper_with_blocked_run_ledger(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    _write_ready_effect_evidence(tmp_path)
    ledger_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        "\n".join(
            [
                "dataset,dataset_name,method,method_label,seed,planned_stages,status,reason",
                (
                    "RM_001_CWRU,CWRU,cfm_grid,Conditional Flow Matching,0,"
                    "train;sample;eval;paperpack,BLOCKED_GPU_PREFLIGHT,"
                    "GPU 6/7 torch CUDA preflight failed"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_paper_ready_with_blocked_run_status"
    }


def test_validate_docs_rejects_submission_ready_paper_with_failed_run_ledger(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    _write_ready_effect_evidence(tmp_path)
    ledger_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_text = ledger_path.read_text(encoding="utf-8")
    ledger_path.write_text(
        ledger_text.replace(
            "RM_027_PU,PU,rectified_flow_grid,Rectified Flow,1,"
            "train;sample;eval;paperpack,complete,complete",
            "RM_027_PU,PU,rectified_flow_grid,Rectified Flow,1,"
            "train;sample;eval;paperpack,failed,training failed",
        ),
        encoding="utf-8",
    )
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_paper_ready_with_incomplete_run_status"
    }


def test_validate_docs_rejects_submission_ready_paper_without_run_ledger(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    _write_ready_effect_evidence(tmp_path)
    ledger_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_path.unlink()
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_paper_ready_missing_run_status_ledger"
    }


def test_validate_docs_rejects_submission_ready_paper_with_incomplete_effect_evidence(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    effect_dir = (
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "effect"
    )
    effect_dir.mkdir(parents=True)
    (effect_dir / "benchmark_effect_summary.csv").write_text(
        "dataset,method,metric,category,benchmark_status,metric_source_paths,manifest_paths\n",
        encoding="utf-8",
    )
    (effect_dir / "benchmark_effect_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "m2_paper_ready_manifest_not_ready" in kinds
    assert "m2_paper_ready_summary_insufficient_datasets" in kinds


def test_validate_docs_rejects_submission_ready_paper_with_wrong_manifest_datasets(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    _write_ready_effect_evidence(tmp_path)
    manifest_path = (
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "effect"
        / "benchmark_effect_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["observed_configured_datasets"] = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
    ]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert "m2_paper_ready_manifest_dataset_mismatch" in {
        issue.kind for issue in issues
    }


def test_validate_docs_rejects_submission_ready_paper_with_wrong_summary_datasets(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    _write_ready_effect_evidence(tmp_path)
    summary_path = (
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "effect"
        / "benchmark_effect_summary.csv"
    )
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(
        summary_text.replace("RM_027_PU", "D6"), encoding="utf-8"
    )
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert "m2_paper_ready_summary_dataset_mismatch" in {
        issue.kind for issue in issues
    }


def test_validate_docs_rejects_submission_ready_paper_with_zero_summary_n(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    _write_ready_effect_evidence(tmp_path)
    summary_path = (
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "effect"
        / "benchmark_effect_summary.csv"
    )
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(
        summary_text.replace(
            "RM_027_PU,cfm_grid,temporal_l1,quality,1.0,2,0,1,0.0,",
            "RM_027_PU,cfm_grid,temporal_l1,quality,1.0,0,0,1,0.0,",
        ),
        encoding="utf-8",
    )
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert "m2_paper_ready_summary_invalid_n" in {issue.kind for issue in issues}


def test_validate_docs_rejects_submission_ready_paper_with_missing_source_files(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    _write_ready_effect_evidence(tmp_path)
    missing_metric = tmp_path / "runs" / "RM_027_PU" / "metrics.csv"
    missing_metric.unlink()
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(
        paper_dir,
        draft_status="SUBMISSION_READY",
        readiness_status="SUBMISSION_READY",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert "m2_paper_ready_summary_missing_source_files" in {
        issue.kind for issue in issues
    }


def test_validate_docs_rejects_not_ready_paper_without_readiness_reason(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(paper_dir)
    (paper_dir / "submission_readiness.md").write_text(
        "\n".join(
            [
                "# M2 Submission Readiness",
                "Status: `NOT_SUBMISSION_READY`",
                "Promotion rule:",
                "The draft can be marked `SUBMISSION_READY` only when",
                "source paths are traceable.",
            ]
        ),
        encoding="utf-8",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_paper_readiness_missing_reason"
    }


def test_validate_docs_rejects_not_ready_paper_without_blocked_claim_text(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(paper_dir)
    draft = paper_dir / "PAPER_DRAFT.md"
    draft.write_text(
        draft.read_text(encoding="utf-8").replace(
            "No numerical claim is made. No computable benchmark rows are available.",
            "",
        ),
        encoding="utf-8",
    )

    issues = check_feature_paper_artifacts(tmp_path)

    assert "m2_paper_not_ready_draft_missing_blocked_claim_text" in {
        issue.kind for issue in issues
    }


def test_validate_docs_rejects_placeholder_m2_paper_draft(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    _write_valid_paper_artifacts(paper_dir, extra_draft_text="TODO: add claim.")

    issues = check_feature_paper_artifacts(tmp_path)

    assert "m2_paper_draft_contains_placeholder" in {issue.kind for issue in issues}


def test_validate_docs_rejects_structurally_incomplete_m2_paper_draft(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-005-markdown-paper-draft.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    paper_dir = tmp_path / "specs" / "002-phm-genbench-frontier" / "paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "PAPER_DRAFT.md").write_text(
        "**Draft status:** `NOT_SUBMISSION_READY`\n", encoding="utf-8"
    )
    (paper_dir / "submission_readiness.md").write_text(
        "Status: `NOT_SUBMISSION_READY`\n", encoding="utf-8"
    )
    (paper_dir / "evidence_gaps.md").write_text(
        "Evidence gaps:\n- missing evidence\n", encoding="utf-8"
    )

    issues = check_feature_paper_artifacts(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "m2_paper_draft_missing_section" in kinds
    assert "m2_paper_draft_missing_required_text" in kinds
    assert "m2_paper_gaps_missing_required_text" in kinds
    assert "m2_paper_readiness_missing_required_text" in kinds


def test_validate_docs_rejects_m2_gpu_preflight_report_without_source_report(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        """
{
  "benchmark_id": "phm_genbench_six_dataset_submission_v1",
  "matrix_path": "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml",
  "require_cuda": true,
  "gpu_ids": ["6", "7"],
  "max_parallel_runs": 2,
  "passed": false,
  "results": [
    {"gpu_id": "6", "status": "failed", "error": "GPU 6 failed CUDA preflight"},
    {"gpu_id": "7", "status": "failed", "error": "GPU 7 failed CUDA preflight"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    issues = check_feature_gpu_preflight_artifact(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_gpu_preflight_report_missing_source_report_field"
    }


def test_validate_docs_requires_completion_audit_when_gpu_preflight_failed(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")

    issues = check_goal_gen_completion_audit(tmp_path)

    assert {issue.kind for issue in issues} == {"missing_goal_gen_completion_audit"}


def test_validate_docs_accepts_completion_audit_when_gpu_preflight_failed(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")
    audit_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "staus"
        / "COMPLETION-AUDIT-2026-05-16-GOAL-GEN.md"
    )
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        "\n".join(GOAL_GEN_COMPLETION_AUDIT_REQUIRED_SNIPPETS),
        encoding="utf-8",
    )

    issues = check_goal_gen_completion_audit(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_completion_audit_when_gpu_preflight_failed(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")
    audit_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "staus"
        / "COMPLETION-AUDIT-2026-05-16-GOAL-GEN.md"
    )
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text("GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE\n", encoding="utf-8")

    issues = check_goal_gen_completion_audit(tmp_path)

    assert {issue.kind for issue in issues} == {
        "goal_gen_completion_audit_missing_text"
    }
    assert any(issue.detail == "**Audit decision**: NOT COMPLETE" for issue in issues)


def test_validate_docs_rejects_completion_audit_missing_objective_artifact(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")
    audit_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "staus"
        / "COMPLETION-AUDIT-2026-05-16-GOAL-GEN.md"
    )
    audit_path.parent.mkdir(parents=True)
    missing_artifact = ".specify/goals/v2/GOAL-GEN-M2-006-review-handoff.md"
    audit_path.write_text(
        "\n".join(
            snippet
            for snippet in GOAL_GEN_COMPLETION_AUDIT_REQUIRED_SNIPPETS
            if snippet != missing_artifact
        ),
        encoding="utf-8",
    )

    issues = check_goal_gen_completion_audit(tmp_path)

    assert {issue.kind for issue in issues} == {
        "goal_gen_completion_audit_missing_text"
    }
    assert [issue.detail for issue in issues] == [missing_artifact]


def test_validate_docs_requires_status_report_when_gpu_preflight_failed(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")

    issues = check_goal_gen_status_package(tmp_path)

    assert {issue.kind for issue in issues} == {"missing_goal_gen_status_report"}


def test_validate_docs_accepts_status_report_when_gpu_preflight_failed(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")
    status_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "staus"
        / "STATUS-2026-05-16.md"
    )
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        "\n".join(GOAL_GEN_STATUS_REPORT_REQUIRED_SNIPPETS),
        encoding="utf-8",
    )

    issues = check_goal_gen_status_package(tmp_path)

    assert issues == []


def test_validate_docs_rejects_incomplete_status_report_when_gpu_preflight_failed(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"passed": false}\n', encoding="utf-8")
    status_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "staus"
        / "STATUS-2026-05-16.md"
    )
    status_path.parent.mkdir(parents=True)
    status_path.write_text("Subagent Acceleration Status\n", encoding="utf-8")

    issues = check_goal_gen_status_package(tmp_path)

    assert {issue.kind for issue in issues} == {
        "goal_gen_status_report_missing_text"
    }
    assert any(
        issue.detail == "COMPLETION-AUDIT-2026-05-16-GOAL-GEN.md"
        for issue in issues
    )


def test_validate_docs_accepts_m2_gpu_preflight_report_source_report(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    source_path = (
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "gpu_preflight"
        / "gpu_preflight_report.json"
    )
    source_path.parent.mkdir(parents=True)
    report_json = """
{
  "benchmark_id": "phm_genbench_six_dataset_submission_v1",
  "matrix_path": "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml",
  "require_cuda": true,
  "gpu_ids": ["6", "7"],
  "max_parallel_runs": 2,
  "passed": false,
  "results": [
    {"gpu_id": "6", "status": "failed", "error": "GPU 6 failed CUDA preflight"},
    {"gpu_id": "7", "status": "failed", "error": "GPU 7 failed CUDA preflight"}
  ]
}
""".strip()
    source_path.write_text(report_json, encoding="utf-8")
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        report_json.replace(
            '"results": [',
            (
                '"source_report": '
                '"results/paper/phm_generative/six_dataset_submission_v1/'
                'gpu_preflight/gpu_preflight_report.json",\n'
                '  "results": ['
            ),
        ),
        encoding="utf-8",
    )

    issues = check_feature_gpu_preflight_artifact(tmp_path)

    assert issues == []


def test_validate_docs_rejects_noncanonical_m2_gpu_preflight_source_report(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    source_path = (
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "gpu_preflight_current_resume"
        / "gpu_preflight_report.json"
    )
    source_path.parent.mkdir(parents=True)
    report_json = """
{
  "benchmark_id": "phm_genbench_six_dataset_submission_v1",
  "matrix_path": "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml",
  "require_cuda": true,
  "gpu_ids": ["6", "7"],
  "max_parallel_runs": 2,
  "passed": false,
  "results": [
    {"gpu_id": "6", "status": "failed", "error": "GPU 6 failed CUDA preflight"},
    {"gpu_id": "7", "status": "failed", "error": "GPU 7 failed CUDA preflight"}
  ]
}
""".strip()
    source_path.write_text(report_json, encoding="utf-8")
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        report_json.replace(
            '"results": [',
            (
                '"source_report": '
                '"results/paper/phm_generative/six_dataset_submission_v1/'
                'gpu_preflight_current_resume/gpu_preflight_report.json",\n'
                '  "results": ['
            ),
        ),
        encoding="utf-8",
    )

    issues = check_feature_gpu_preflight_artifact(tmp_path)

    assert "m2_gpu_preflight_report_noncanonical_source_report" in {
        issue.kind for issue in issues
    }


def test_validate_docs_rejects_missing_m2_gpu_preflight_source_report(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        """
{
  "benchmark_id": "phm_genbench_six_dataset_submission_v1",
  "matrix_path": "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml",
  "require_cuda": true,
  "gpu_ids": ["6", "7"],
  "max_parallel_runs": 2,
  "passed": false,
  "source_report": "results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight/gpu_preflight_report.json",
  "results": [
    {"gpu_id": "6", "status": "failed", "error": "GPU 6 failed CUDA preflight"},
    {"gpu_id": "7", "status": "failed", "error": "GPU 7 failed CUDA preflight"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    issues = check_feature_gpu_preflight_artifact(tmp_path)

    assert {issue.kind for issue in issues} == {
        "m2_gpu_preflight_report_missing_source_report"
    }


def test_validate_docs_rejects_missing_m2_gpu_preflight_report(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")

    issues = check_feature_gpu_preflight_artifact(tmp_path)

    assert {issue.kind for issue in issues} == {"missing_m2_gpu_preflight_report"}


def test_validate_docs_rejects_inconsistent_m2_gpu_preflight_report(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        """
{
  "benchmark_id": "phm_genbench_six_dataset_submission_v1",
  "matrix_path": "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml",
  "require_cuda": true,
  "gpu_ids": ["6", "7"],
  "max_parallel_runs": 2,
  "passed": true,
  "results": [
    {"gpu_id": "6", "status": "failed"},
    {"gpu_id": "7", "status": "passed"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    issues = check_feature_gpu_preflight_artifact(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "m2_gpu_preflight_report_passed_status_mismatch" in kinds
    assert "m2_gpu_preflight_report_failed_result_missing_error" in kinds


def test_validate_docs_accepts_m2_gpu_runbook(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    runbook_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-gpu-runbook.md"
    )
    runbook_path.parent.mkdir(parents=True)
    runbook_path.write_text(
        "\n".join(
            [
                "# M2 GPU Runbook",
                "## Current Blocker",
                "GPU 6 and GPU 7 are blocked.",
                "## Resume Gates",
                "eval \"$(conda shell.bash hook)\" && conda activate LQ_signal &&",
                "CUDA_VISIBLE_DEVICES=6 python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())\"",
                "CUDA_VISIBLE_DEVICES=7 python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())\"",
                "CUDA_VISIBLE_DEVICES=6,7 python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())\"",
                "torch.cuda.is_available() is True.",
                "`torch.cuda.device_count()` is exactly `1` for single-GPU probes.",
                "--preflight-gpu",
                "## Execution Sequence",
                "env CUDA_VISIBLE_DEVICES=6 python main.py",
                "env CUDA_VISIBLE_DEVICES=7 python main.py",
                "trainer.device=cuda",
                "trainer.gpus=1",
                "144 commands",
                "--execute --preflight-gpu --stages train",
                "Repeat for sample, eval, and paperpack.",
                "Do not route the paper benchmark to CPU.",
                "## Evidence Aggregation",
                "--from-runs results/paper/phm_generative/six_dataset_submission_v1/runs",
                "## Completion Rule",
                "Complete only after real evidence exists.",
            ]
        ),
        encoding="utf-8",
    )

    issues = check_feature_m2_gpu_runbook(tmp_path)

    assert issues == []


def test_validate_docs_rejects_missing_m2_gpu_runbook(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")

    issues = check_feature_m2_gpu_runbook(tmp_path)

    assert {issue.kind for issue in issues} == {"missing_m2_gpu_runbook"}


def test_validate_docs_accepts_m2_run_status_ledger(tmp_path: Path) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    ledger_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "dataset,dataset_name,method,method_label,seed,planned_stages,status,reason"
    ]
    for dataset, dataset_name in M2_LEDGER_DATASETS:
        for method, method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                lines.append(
                    ",".join(
                        [
                            dataset,
                            dataset_name,
                            method,
                            method_label,
                            str(seed),
                            "train;sample;eval;paperpack",
                            "BLOCKED_GPU_PREFLIGHT",
                            "GPU 6/7 torch CUDA preflight failed",
                        ]
                    )
                )
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    markdown_lines = [
        "# M2 Run Status Ledger",
        "Machine-readable copy:",
        "2026-05-11-m2-run-status-ledger.csv",
        "Current status: `BLOCKED_GPU_PREFLIGHT`.",
        "GPU 6 failed CUDA preflight.",
        "GPU 7 failed CUDA preflight.",
        "nvidia-smi cannot communicate with the NVIDIA driver.",
        "Each run covers train/sample/eval/paperpack.",
        "results/paper/phm_generative/six_dataset_submission_v1/runs",
        "## Downstream Readiness",
        "Ready for M2-004 figures/tables: no.",
        "Ready for M2-005 paper draft: no.",
        "| Dataset | Method | Seed | Status |",
        "| --- | --- | --- | --- |",
    ]
    for dataset, _dataset_name in M2_LEDGER_DATASETS:
        for method, _method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                markdown_lines.append(
                    f"| {dataset} | {method} | {seed} | BLOCKED_GPU_PREFLIGHT |"
                )
    markdown_lines.append("## Resume Rule")
    (ledger_path.with_suffix(".md")).write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    issues = check_feature_m2_run_status_ledger(tmp_path)

    assert issues == []


def test_validate_docs_rejects_m2_run_status_markdown_missing_source_ledger(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    review_dir = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
    )
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "2026-05-12-gpu-preflight-report.json").write_text(
        (
            '{"passed": false, "source_report": '
            '"results/paper/phm_generative/six_dataset_submission_v1/'
            'gpu_preflight_fixture/gpu_preflight_report.json"}\n'
        ),
        encoding="utf-8",
    )
    source_ledger = (
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "gpu_preflight_fixture"
        / "blocked_run_status_ledger.csv"
    )
    source_ledger.parent.mkdir(parents=True)
    header = (
        "benchmark_id,dataset,dataset_name,method,method_label,seed,"
        "planned_stages,status,reason"
    )
    lines = [header]
    review_lines = [header]
    markdown_lines = [
        "# M2 Run Status Ledger",
        "Machine-readable copy:",
        "2026-05-11-m2-run-status-ledger.csv",
        "Current status: `BLOCKED_GPU_PREFLIGHT`.",
        "GPU 6 failed CUDA preflight.",
        "GPU 7 failed CUDA preflight.",
        "nvidia-smi cannot communicate with the NVIDIA driver.",
        "Each run covers train/sample/eval/paperpack.",
        "results/paper/phm_generative/six_dataset_submission_v1/runs",
        "## Downstream Readiness",
        "Ready for M2-004 figures/tables: no.",
        "Ready for M2-005 paper draft: no.",
    ]
    for dataset, dataset_name in M2_LEDGER_DATASETS:
        for method, method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                row = ",".join(
                    [
                        "phm_genbench_six_dataset_submission_v1",
                        dataset,
                        dataset_name,
                        method,
                        method_label,
                        str(seed),
                        "train;sample;eval;paperpack",
                        "BLOCKED_GPU_PREFLIGHT",
                        "GPU 6/7 torch CUDA preflight failed",
                    ]
                )
                lines.append(row)
                review_lines.append(row)
                markdown_lines.append(
                    f"| {dataset} | {method} | {seed} | BLOCKED_GPU_PREFLIGHT |"
                )
    markdown_lines.append("## Resume Rule")
    source_ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger_path = review_dir / "2026-05-11-m2-run-status-ledger.csv"
    ledger_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    ledger_path.with_suffix(".md").write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )

    issues = check_feature_m2_run_status_ledger(tmp_path)

    assert "m2_run_status_ledger_markdown_missing_source_ledger" in {
        issue.kind for issue in issues
    }


def test_validate_docs_rejects_missing_m2_run_status_ledger(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")

    issues = check_feature_m2_run_status_ledger(tmp_path)

    assert {issue.kind for issue in issues} == {"missing_m2_run_status_ledger"}


def test_validate_docs_rejects_missing_m2_run_status_ledger_markdown(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    ledger_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "dataset,dataset_name,method,method_label,seed,planned_stages,status,reason"
    ]
    for dataset, dataset_name in M2_LEDGER_DATASETS:
        for method, method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                lines.append(
                    ",".join(
                        [
                            dataset,
                            dataset_name,
                            method,
                            method_label,
                            str(seed),
                            "train;sample;eval;paperpack",
                            "BLOCKED_GPU_PREFLIGHT",
                            "GPU 6/7 torch CUDA preflight failed",
                        ]
                    )
                )
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    issues = check_feature_m2_run_status_ledger(tmp_path)

    assert {issue.kind for issue in issues} == {
        "missing_m2_run_status_ledger_markdown"
    }


def test_validate_docs_rejects_incomplete_m2_run_status_ledger_markdown(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    ledger_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "dataset,dataset_name,method,method_label,seed,planned_stages,status,reason"
    ]
    for dataset, dataset_name in M2_LEDGER_DATASETS:
        for method, method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                lines.append(
                    ",".join(
                        [
                            dataset,
                            dataset_name,
                            method,
                            method_label,
                            str(seed),
                            "train;sample;eval;paperpack",
                            "BLOCKED_GPU_PREFLIGHT",
                            "GPU 6/7 torch CUDA preflight failed",
                        ]
                    )
                )
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ledger_path.with_suffix(".md")).write_text(
        "\n".join(
            [
                "# M2 Run Status Ledger",
                "Machine-readable copy:",
                "2026-05-11-m2-run-status-ledger.csv",
                "Current status: `BLOCKED_GPU_PREFLIGHT`.",
                "GPU 6 failed CUDA preflight.",
                "GPU 7 failed CUDA preflight.",
                "nvidia-smi cannot communicate with the NVIDIA driver.",
                "Each run covers train/sample/eval/paperpack.",
                "results/paper/phm_generative/six_dataset_submission_v1/runs",
                "## Resume Rule",
            ]
        ),
        encoding="utf-8",
    )

    issues = check_feature_m2_run_status_ledger(tmp_path)

    assert "m2_run_status_ledger_markdown_incomplete_matrix" in {
        issue.kind for issue in issues
    }


def test_validate_docs_rejects_m2_run_status_ledger_source_mismatch(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    review_dir = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
    )
    review_dir.mkdir(parents=True, exist_ok=True)
    report_path = review_dir / "2026-05-12-gpu-preflight-report.json"
    report_path.write_text(
        (
            '{"passed": false, "source_report": '
            '"results/paper/phm_generative/six_dataset_submission_v1/'
            'gpu_preflight_fixture/gpu_preflight_report.json"}\n'
        ),
        encoding="utf-8",
    )
    source_ledger = (
        tmp_path
        / "results"
        / "paper"
        / "phm_generative"
        / "six_dataset_submission_v1"
        / "gpu_preflight_fixture"
        / "blocked_run_status_ledger.csv"
    )
    source_ledger.parent.mkdir(parents=True)
    source_ledger.write_text(
        (
            "benchmark_id,dataset,dataset_name,method,method_label,seed,"
            "planned_stages,status,reason\n"
            "phm_genbench_six_dataset_submission_v1,RM_001_CWRU,CWRU,"
            "cfm_grid,Conditional Flow Matching,0,train;sample;eval;paperpack,"
            "BLOCKED_GPU_PREFLIGHT,different source ledger\n"
        ),
        encoding="utf-8",
    )
    ledger_path = review_dir / "2026-05-11-m2-run-status-ledger.csv"
    lines = [
        "dataset,dataset_name,method,method_label,seed,planned_stages,status,reason"
    ]
    markdown_lines = [
        "# M2 Run Status Ledger",
        "Machine-readable copy:",
        "2026-05-11-m2-run-status-ledger.csv",
        "Current status: `BLOCKED_GPU_PREFLIGHT`.",
        "GPU 6 failed CUDA preflight.",
        "GPU 7 failed CUDA preflight.",
        "nvidia-smi cannot communicate with the NVIDIA driver.",
        "Each run covers train/sample/eval/paperpack.",
        "results/paper/phm_generative/six_dataset_submission_v1/runs",
        "## Downstream Readiness",
        "Ready for M2-004 figures/tables: no.",
        "Ready for M2-005 paper draft: no.",
    ]
    for dataset, dataset_name in M2_LEDGER_DATASETS:
        for method, method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                lines.append(
                    ",".join(
                        [
                            dataset,
                            dataset_name,
                            method,
                            method_label,
                            str(seed),
                            "train;sample;eval;paperpack",
                            "BLOCKED_GPU_PREFLIGHT",
                            "GPU 6/7 torch CUDA preflight failed",
                        ]
                    )
                )
                markdown_lines.append(
                    f"| {dataset} | {method} | {seed} | BLOCKED_GPU_PREFLIGHT |"
                )
    markdown_lines.append("## Resume Rule")
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger_path.with_suffix(".md").write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )

    issues = check_feature_m2_run_status_ledger(tmp_path)

    assert "m2_run_status_ledger_source_mismatch" in {
        issue.kind for issue in issues
    }


def test_validate_docs_rejects_unblocked_ledger_when_gpu_preflight_failed(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    report_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-12-gpu-preflight-report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        """
{
  "benchmark_id": "phm_genbench_six_dataset_submission_v1",
  "matrix_path": "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml",
  "require_cuda": true,
  "gpu_ids": ["6", "7"],
  "max_parallel_runs": 2,
  "passed": false,
  "results": [
    {"gpu_id": "6", "status": "failed", "error": "GPU 6 failed CUDA preflight"},
    {"gpu_id": "7", "status": "failed", "error": "GPU 7 failed CUDA preflight"}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    ledger_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "dataset,dataset_name,method,method_label,seed,planned_stages,status,reason"
    ]
    for dataset, dataset_name in M2_LEDGER_DATASETS:
        for method, method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                lines.append(
                    ",".join(
                        [
                            dataset,
                            dataset_name,
                            method,
                            method_label,
                            str(seed),
                            "train;sample;eval;paperpack",
                            "COMPLETE",
                            "not a resource blocker",
                        ]
                    )
                )
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    issues = check_feature_m2_run_status_ledger(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "invalid_m2_run_status_ledger_unblocked_failed_preflight" in kinds
    assert "invalid_m2_run_status_ledger_blocked_reason_not_gpu" in kinds


def test_validate_docs_rejects_m2_run_status_ledger_label_mismatch(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    ledger_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "dataset,dataset_name,method,method_label,seed,planned_stages,status,reason"
    ]
    for dataset, dataset_name in M2_LEDGER_DATASETS:
        for method, method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                if dataset == "RM_001_CWRU" and method == "cfm_grid" and seed == 0:
                    dataset_name = "wrong_dataset_name"
                    method_label = "wrong_method_label"
                lines.append(
                    ",".join(
                        [
                            dataset,
                            dataset_name,
                            method,
                            method_label,
                            str(seed),
                            "train;sample;eval;paperpack",
                            "BLOCKED_GPU_PREFLIGHT",
                            "GPU 6/7 torch CUDA preflight failed",
                        ]
                    )
                )
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    issues = check_feature_m2_run_status_ledger(tmp_path)

    kinds = {issue.kind for issue in issues}
    assert "invalid_m2_run_status_ledger_dataset_name" in kinds
    assert "invalid_m2_run_status_ledger_method_label" in kinds


def test_validate_docs_rejects_m2_run_status_ledger_invalid_status(
    tmp_path: Path,
) -> None:
    goal_path = (
        tmp_path
        / ".specify"
        / "goals"
        / "v2"
        / "GOAL-GEN-M2-003-real-runs-evidence.md"
    )
    goal_path.parent.mkdir(parents=True)
    goal_path.write_text("# Goal\n", encoding="utf-8")
    ledger_path = (
        tmp_path
        / "specs"
        / "002-phm-genbench-frontier"
        / "reviews"
        / "codex"
        / "2026-05-11-m2-run-status-ledger.csv"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "dataset,dataset_name,method,method_label,seed,planned_stages,status,reason"
    ]
    for dataset, dataset_name in M2_LEDGER_DATASETS:
        for method, method_label in M2_LEDGER_METHODS:
            for seed in (0, 1):
                status = "UNKNOWN" if (
                    dataset == "RM_001_CWRU" and method == "cfm_grid" and seed == 0
                ) else "BLOCKED_GPU_PREFLIGHT"
                lines.append(
                    ",".join(
                        [
                            dataset,
                            dataset_name,
                            method,
                            method_label,
                            str(seed),
                            "train;sample;eval;paperpack",
                            status,
                            "GPU 6/7 torch CUDA preflight failed",
                        ]
                    )
                )
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    issues = check_feature_m2_run_status_ledger(tmp_path)

    assert "invalid_m2_run_status_ledger_status" in {issue.kind for issue in issues}
