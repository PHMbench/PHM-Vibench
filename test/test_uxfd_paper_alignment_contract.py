from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from scripts.uxfd_paper_alignment import (
    audit_contracts,
    compile_gates,
    discover_latex_entrypoints,
    gitmodule_uxfd_submodules,
    indexed_uxfd_submodules,
    map_claim_evidence,
    submodule_states,
)


GOAL_DIR = Path("paper/UXFD_paper/goal")
PAPER07_MATRIX = Path(
    "paper/UXFD_paper/TII_operator_attention/submission_prep/"
    "baseline_ablation_matrix.yaml"
)
PAPER05_MATRIX = Path(
    "paper/UXFD_paper/Paper_fuzzy_XFD/submission_prep/"
    "baseline_ablation_matrix.yaml"
)
PAPER04_MATRIX = Path(
    "paper/UXFD_paper/MOE_explainable/submission_prep/"
    "baseline_ablation_matrix.yaml"
)
PAPER01_MATRIX = Path(
    "paper/UXFD_paper/Explainable_FD_Toolkit/submission_prep/"
    "baseline_ablation_matrix.yaml"
)

LOW_TIER_MARKERS = (
    "Scientific Reports",
    "MDPI",
    "IEEE TIM",
    "IEEE Transactions on Instrumentation and Measurement",
    "IEEE Access",
    "Applied Sciences",
    "Electronics",
    "Sensors",
    "Mathematics",
)


def _section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    assert start >= 0, f"missing section {heading!r}"
    rest = text[start + len(marker) :]
    next_heading = rest.find("\n## ")
    if next_heading >= 0:
        return rest[:next_heading]
    return rest


def test_seven_indexed_uxfd_submodules_match_gitmodules() -> None:
    indexed = set(indexed_uxfd_submodules())
    gitmodules = set(gitmodule_uxfd_submodules())

    assert len(indexed) == 7
    assert indexed == gitmodules


def test_each_indexed_submodule_has_vibench_and_min_config() -> None:
    for contract in audit_contracts():
        assert contract.vibench_path.exists(), contract.submodule_path
        assert contract.min_config_path.exists(), contract.submodule_path


def test_each_vibench_and_min_config_are_tracked_by_submodule_git() -> None:
    for contract in audit_contracts():
        result = subprocess.run(
            [
                "git",
                "-C",
                str(contract.submodule_path),
                "ls-files",
                "--error-unmatch",
                "VIBENCH.md",
                "configs/vibench/min.yaml",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            contract.submodule_path,
            result.stdout,
            result.stderr,
        )


def test_each_vibench_declares_root_cli_or_paper_local_status() -> None:
    for contract in audit_contracts():
        assert contract.maintained_command or contract.status == "paper-local-only"
        if contract.maintained_command:
            assert contract.maintained_command.startswith("python main.py --config ")
            assert str(contract.min_config_path) in contract.maintained_command


def test_each_vibench_records_local_gpu_binding_policy() -> None:
    for contract in audit_contracts():
        text = contract.vibench_path.read_text(encoding="utf-8")

        assert "CUDA_VISIBLE_DEVICES=0" in text or "CUDA_VISIBLE_DEVICES=1" in text, (
            contract.submodule_path,
            "missing local GPU binding in VIBENCH.md",
        )


def test_artifact_expectations_are_recorded_or_blocked() -> None:
    for contract in audit_contracts():
        if "artifacts/manifest.json" in contract.expected_artifacts:
            assert "artifacts/data_metadata_snapshot.json" in contract.expected_artifacts
        else:
            assert contract.status in {"unverified", "blocked", "paper-local-only"}
            assert contract.reason


def test_latex_entrypoint_discovery_records_selected_or_blocker_status() -> None:
    entrypoints = discover_latex_entrypoints()

    assert len(entrypoints) == 7
    for entrypoint in entrypoints:
        assert entrypoint.status in {"selected", "non-final", "missing"}
        if entrypoint.status == "selected":
            assert entrypoint.tex_path.name == "main.tex"
            assert entrypoint.tex_path.exists()
        else:
            assert entrypoint.reason


def test_claim_evidence_records_artifact_source_or_blocker_fields() -> None:
    for claim in map_claim_evidence():
        assert claim.claim_id
        assert claim.claim_type
        assert claim.status in {"verified", "blocked", "unresolved", "external-source"}
        if claim.status != "verified":
            assert claim.reason
        else:
            assert claim.artifact_path and Path(claim.artifact_path).exists()


def test_compile_gate_records_have_required_fields() -> None:
    for gate in compile_gates():
        assert gate.tex_path
        assert gate.result in {"pending", "pass", "fail", "skipped", "blocked"}
        if gate.result == "pending":
            assert gate.command
            assert gate.pdf_path
            assert gate.log_path
        else:
            assert gate.first_error


def test_submodule_states_include_all_uxfd_submodules() -> None:
    states = submodule_states()

    assert {state.submodule_path for state in states} == set(indexed_uxfd_submodules())
    for state in states:
        assert state.commit_sha
        assert state.submodule_status in {"clean", "dirty-or-pointer-changed"}


def test_goal_files_require_six_baselines_ablations_and_sota_gate() -> None:
    paper_goal_files = sorted(GOAL_DIR.glob("0[1-7]_*.md"))

    assert len(paper_goal_files) == 7
    for goal_file in paper_goal_files:
        text = goal_file.read_text(encoding="utf-8")
        baseline_section = _section(text, "Baseline Suite")

        baselines = [
            line
            for line in baseline_section.splitlines()
            if line.strip().startswith("- ")
        ]
        assert len(baselines) >= 6, goal_file
        assert "## Ablation Suite" in text, goal_file
        assert "## SOTA Optimization Gate" in text, goal_file


def test_recent_work_readme_defines_reproduction_status_and_commands() -> None:
    readme = GOAL_DIR / "08_recent_work_citation_readme.md"
    text = readme.read_text(encoding="utf-8")

    assert "exact-runnable" in text
    assert "representative-runnable" in text
    assert "literature-only" in text
    assert "resource-blocked" in text
    assert "blocked" in text
    assert "CUDA_VISIBLE_DEVICES=0,1" in text
    assert "RTX 4090" in text
    assert text.count("| RWTOP20") >= 10
    assert "top-conference" in text
    assert "top-journal" in text
    assert "python -m scripts.baseline_mapping" in text


def test_recent_work_accepted_pool_excludes_low_tier_sources() -> None:
    readme = GOAL_DIR / "08_recent_work_citation_readme.md"
    text = readme.read_text(encoding="utf-8")
    accepted_pool = _section(text, "Accepted TOP Method Pool")

    assert "ICML" in accepted_pool
    assert "ICLR" in accepted_pool
    assert "NeurIPS" in accepted_pool
    assert "CVPR" in accepted_pool
    assert "Information Fusion" in accepted_pool
    for marker in LOW_TIER_MARKERS:
        assert marker not in accepted_pool


def test_paper_goals_require_top_recent_work_quota() -> None:
    paper_goal_files = sorted(GOAL_DIR.glob("0[1-7]_*.md"))

    assert len(paper_goal_files) == 7
    for goal_file in paper_goal_files:
        text = goal_file.read_text(encoding="utf-8")
        quota_section = _section(text, "TOP Recent-Work Quota")
        top_method_lines = [
            line
            for line in quota_section.splitlines()
            if line.strip().startswith("- RWTOP")
        ]

        assert len(top_method_lines) >= 3, goal_file
        assert (
            "representative-runnable" in quota_section
            or "exact-runnable" in quota_section
        ), goal_file


def test_overall_goal_defines_two_4090_compute_gate() -> None:
    text = (GOAL_DIR / "00_overall_goal.md").read_text(encoding="utf-8")
    compute_section = _section(text, "Compute Resource Gate")

    assert "CUDA_VISIBLE_DEVICES=0,1" in compute_section
    assert "RTX" in compute_section
    assert "4090" in compute_section
    assert "0" in compute_section
    assert "1" in compute_section
    assert "resource-blocked" in compute_section
    assert "cloud" in compute_section


def test_paper_goals_require_compute_budget_sections() -> None:
    paper_goal_files = sorted(GOAL_DIR.glob("0[1-7]_*.md"))

    assert len(paper_goal_files) == 7
    for goal_file in paper_goal_files:
        text = goal_file.read_text(encoding="utf-8")
        compute_section = _section(text, "Compute Budget")

        assert "CUDA_VISIBLE_DEVICES" in compute_section, goal_file
        assert "RTX 4090" in compute_section, goal_file
        assert "0,1" in compute_section, goal_file
        assert "resource-blocked" in compute_section, goal_file
        assert "OOM" in compute_section, goal_file


def test_operator_attention_goal_has_rejection_recovery_requirements() -> None:
    text = (GOAL_DIR / "07_tii_operator_attention.md").read_text(encoding="utf-8")

    assert "## Rejection-Recovery Focus" in text
    assert "Dynamic Sparse Operator Attention v2" in text
    assert "OAS" in text
    assert "OSS" in text
    assert "OCS" in text


def test_operator_attention_baseline_ablation_matrix_is_command_bound_not_ready() -> None:
    assert PAPER07_MATRIX.exists()
    matrix = yaml.safe_load(PAPER07_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert matrix["evidence_level"] == "config-target validated only"
    assert len(matrix["baselines"]) >= 6
    assert len(matrix["ablations"]) >= 6
    assert all(
        "pass in LQ_signal" in entry.get("dummy_smoke_status", "")
        for entry in matrix["baselines"]
    )
    assert (
        sum(
            "pass in LQ_signal" in entry.get("dummy_smoke_status", "")
            or "same run as B01" in entry.get("dummy_smoke_status", "")
            for entry in matrix["ablations"]
        )
        >= 6
    )

    for entry in matrix["baselines"] + matrix["ablations"]:
        assert entry["config_target_validated"] is True
        assert "CUDA_VISIBLE_DEVICES=0" in entry["command"]
        assert "python main.py --config" in entry["command"]
        assert (
            "pending" in entry["accepted_evidence_status"]
            or "blocked" in entry["accepted_evidence_status"]
        )

    blockers = "\n".join(matrix["strict_blockers"])
    assert "No accepted industrial multi-seed baseline table yet." in blockers
    assert "No SOTA claim is allowed from this matrix alone." in blockers


def test_fuzzy_xfd_baseline_ablation_matrix_is_command_bound_not_ready() -> None:
    assert PAPER05_MATRIX.exists()
    matrix = yaml.safe_load(PAPER05_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert matrix["evidence_level"] == "config-target validated only"
    assert len(matrix["baselines"]) >= 6
    assert len(matrix["ablations"]) >= 6
    assert "pass in LQ_signal" in matrix["proposed"]["dummy_smoke_status"]
    assert (
        sum("pass in LQ_signal" in entry.get("dummy_smoke_status", "") for entry in matrix["baselines"])
        >= 6
    )
    assert (
        sum(
            "pass in LQ_signal" in entry.get("dummy_smoke_status", "")
            or "same run as B01" in entry.get("dummy_smoke_status", "")
            for entry in matrix["ablations"]
        )
        >= 6
    )

    main_py_baselines = [
        entry
        for entry in matrix["baselines"]
        if "python main.py --config" in entry["command"]
    ]
    assert len(main_py_baselines) >= 6

    for entry in matrix["baselines"] + matrix["ablations"]:
        assert entry["config_target_validated"] is True
        assert "CUDA_VISIBLE_DEVICES=0" in entry["command"]
        assert (
            "pending" in entry["accepted_evidence_status"]
            or "blocked" in entry["accepted_evidence_status"]
        )

    blockers = "\n".join(matrix["strict_blockers"])
    assert "No accepted CWRU/XJTU or industrial multi-seed baseline table yet." in blockers
    assert "Hard-threshold, safety-fallback, and no-rule-output" in blockers
    assert "No SOTA claim is allowed from this matrix alone." in blockers


def test_moe_baseline_matrix_records_ablation_blockers_not_ready() -> None:
    assert PAPER04_MATRIX.exists()
    matrix = yaml.safe_load(PAPER04_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert matrix["evidence_level"] == "baseline config-target validated; MoE ablation evidence partial"
    assert len(matrix["local_moe_evidence"]) >= 4
    assert len(matrix["baselines"]) >= 6
    assert len(matrix["ablations"]) >= 6
    assert "pass in LQ_signal" in matrix["proposed"]["dummy_smoke_status"]
    assert all(
        "pass in LQ_signal" in entry.get("dummy_smoke_status", "")
        for entry in matrix["baselines"]
    )

    for entry in matrix["baselines"]:
        assert entry["config_target_validated"] is True
        assert "CUDA_VISIBLE_DEVICES=0" in entry["command"]
        assert "python main.py --config" in entry["command"]
        assert "pending" in entry["accepted_evidence_status"]

    bound_ablations = [
        entry for entry in matrix["ablations"] if entry["config_target_validated"] is True
    ]
    blocked_ablations = [
        entry for entry in matrix["ablations"] if entry["config_target_validated"] is False
    ]
    assert len(bound_ablations) == 1
    assert len(blocked_ablations) >= 5
    assert "run_expert_ablation_probe.py" in bound_ablations[0]["command"]
    assert all(entry["command"].startswith("blocked:") for entry in blocked_ablations)

    blockers = "\n".join(matrix["strict_blockers"])
    assert "Only one MoE-specific ablation command is currently bound" in blockers
    assert "No accepted TOP representative command/log/artifact mapping yet." in blockers
    assert "No SOTA claim is allowed from this matrix alone." in blockers


def test_toolkit_baseline_matrix_records_ablation_blockers_not_ready() -> None:
    assert PAPER01_MATRIX.exists()
    matrix = yaml.safe_load(PAPER01_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert (
        matrix["evidence_level"]
        == "baseline config-target validated; Toolkit ablation evidence mostly blocked"
    )
    assert len(matrix["existing_toolkit_evidence"]) >= 4
    assert len(matrix["baselines"]) >= 6
    assert len(matrix["ablations"]) >= 6
    assert "pass in LQ_signal" in matrix["proposed"]["dummy_smoke_status"]
    assert all(
        "pass in LQ_signal" in entry.get("dummy_smoke_status", "")
        or "same run as P00" in entry.get("dummy_smoke_status", "")
        for entry in matrix["baselines"]
    )

    for entry in matrix["baselines"]:
        assert entry["config_target_validated"] is True
        assert "CUDA_VISIBLE_DEVICES=0" in entry["command"]
        assert "python main.py --config" in entry["command"]
        assert "pending" in entry["accepted_evidence_status"]

    bound_ablations = [
        entry for entry in matrix["ablations"] if entry["config_target_validated"] is True
    ]
    blocked_ablations = [
        entry for entry in matrix["ablations"] if entry["config_target_validated"] is False
    ]
    assert len(bound_ablations) == 1
    assert len(blocked_ablations) >= 5
    assert "trainer.extensions.explain.enable=false" in bound_ablations[0]["command"]
    assert all(entry["command"].startswith("blocked:") for entry in blocked_ablations)

    blockers = "\n".join(matrix["strict_blockers"])
    assert "Only one Toolkit-specific ablation command is currently bound" in blockers
    assert "No accepted TOP representative command/log/artifact mapping yet." in blockers
    assert "No SOTA or submission-ready infrastructure claim" in blockers


def test_readiness_matrix_tracks_baseline_ablation_and_sota_status() -> None:
    text = (GOAL_DIR / "99_submission_readiness_matrix.md").read_text(encoding="utf-8")

    assert "6+ Baselines" in text
    assert "TOP Recent Work" in text
    assert "Runnable TOP Baseline" in text
    assert "Compute Budget" in text
    assert "GPU Feasible" in text
    assert "Ablations" in text
    assert "SOTA Gate" in text
    assert "Citation README" in text
    assert "Run Evidence" in text
