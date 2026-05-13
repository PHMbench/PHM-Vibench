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
EXECUTION_QUEUE = GOAL_DIR / "09_gpu_execution_queue.yaml"
PAPER07_MATRIX = Path(
    "paper/UXFD_paper/TII_operator_attention/submission_prep/"
    "baseline_ablation_matrix.yaml"
)
PAPER07_REJECTION_CONTRACT = Path(
    "paper/UXFD_paper/TII_operator_attention/submission_prep/"
    "rejection_recovery_contract.md"
)
PAPER07_REVIEWER_TRACE = Path(
    "paper/UXFD_paper/TII_operator_attention/submission_prep/"
    "reviewer_traceability_matrix.md"
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
PAPER02_MATRIX = Path(
    "paper/UXFD_paper/1D-2D_fusion_explainable/submission_prep/"
    "baseline_ablation_matrix.yaml"
)
PAPER03_MATRIX = Path(
    "paper/UXFD_paper/LLM_Explainable_FD_Toolkit/submission_prep/"
    "baseline_ablation_matrix.yaml"
)
PAPER03_LLM_EVIDENCE_CONTRACT = Path(
    "paper/UXFD_paper/LLM_Explainable_FD_Toolkit/submission_prep/"
    "llm_evidence_package_contract.md"
)
PAPER06_MATRIX = Path(
    "paper/UXFD_paper/Neuralsymbolic_theory/submission_prep/"
    "baseline_ablation_matrix.yaml"
)

PAPER_MATRICES = {
    "Explainable_FD_Toolkit": PAPER01_MATRIX,
    "1D-2D_fusion_explainable": PAPER02_MATRIX,
    "LLM_Explainable_FD_Toolkit": PAPER03_MATRIX,
    "MOE_explainable": PAPER04_MATRIX,
    "Paper_fuzzy_XFD": PAPER05_MATRIX,
    "Neuralsymbolic_theory": PAPER06_MATRIX,
    "TII_operator_attention": PAPER07_MATRIX,
}

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

REQUIRED_2026_TOP_IDS = (
    "RWTOP2026-TIMESEG",
    "RWTOP2026-TIMESLIVER",
    "RWTOP2026-PGRFNET",
    "RWTOP2026-GTM",
    "RWTOP2026-CSLSTM",
    "RWTOP2026-TSPULSE",
)

PAPER07_REQUIRED_TOP_IDS = (
    "RWTOP2024-TIMEMIXER",
    "RWTOP2024-SARAD",
    "RWTOP2025-CATCH",
    "RWTOP2025-DADA",
    "RWTOP2026-PGRFNET",
    "RWTOP2026-GTM",
    "RWTOP2026-CSLSTM",
    "RWTOP2026-TSPULSE",
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


def test_all_seven_paper_matrices_exist_with_baseline_ablation_and_blocker_state() -> None:
    assert len(PAPER_MATRICES) == 7

    for paper_id, matrix_path in PAPER_MATRICES.items():
        assert matrix_path.exists(), paper_id
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))

        assert matrix["paper_id"] == paper_id
        assert matrix["submission_ready"] is False
        assert len(matrix.get("baselines", [])) >= 6
        assert len(matrix.get("ablations", [])) >= 6
        assert matrix.get("strict_blockers"), paper_id
        assert "No SOTA" in "\n".join(matrix["strict_blockers"])

        common_policy = matrix.get("common_policy", {})
        assert "4090" in common_policy.get("devices", ""), paper_id
        assert "CUDA_VISIBLE_DEVICES=0" in common_policy.get("default_binding", ""), paper_id
        assert any(
            "CUDA_VISIBLE_DEVICES" in item
            for item in common_policy.get("required_metadata", [])
        ), paper_id


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
    assert "ICLR 2026 Poster" in accepted_pool
    for top_id in REQUIRED_2026_TOP_IDS:
        assert top_id in accepted_pool
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
        assert "RWTOP2026-" in quota_section, goal_file
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


def test_readiness_matrix_records_gpu_preflight_blocker_and_execution_queue() -> None:
    text = (GOAL_DIR / "99_submission_readiness_matrix.md").read_text(encoding="utf-8")
    resource_section = _section(text, "Resource Check")
    queue_section = _section(text, "Immediate Execution Queue")

    assert "nvidia-smi -L" in resource_section
    assert "couldn't communicate with the NVIDIA driver" in resource_section
    assert "torch.cuda.is_available() == True" in resource_section
    assert "torch.cuda.device_count() == 2" in resource_section
    assert "no accepted GPU evidence" in resource_section
    assert "CUDA_VISIBLE_DEVICES=0" in resource_section

    for step in ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"):
        assert f"| {step} |" in queue_section
    assert "SOTA wording remains blocked" in queue_section
    assert "09_gpu_execution_queue.yaml" in queue_section


def test_gpu_execution_queue_covers_all_papers_and_keeps_sota_blocked() -> None:
    assert EXECUTION_QUEUE.exists()
    queue = yaml.safe_load(EXECUTION_QUEUE.read_text(encoding="utf-8"))

    assert queue["status"] == "blocked_resource_preflight"
    preflight = queue["resource_preflight"]
    assert preflight["required_devices"] == ["0", "1"]
    assert preflight["required_gpu_class"] == "RTX 4090"
    assert preflight["current_session_result"]["torch_cuda_available"] is False
    assert preflight["current_session_result"]["torch_cuda_device_count"] == 0
    assert "blocked" in preflight["current_session_result"]["verdict"]
    assert any("nvidia-smi -L" in item["command"] for item in preflight["required_commands"])
    assert any(
        "torch.cuda.device_count" in item["command"]
        for item in preflight["required_commands"]
    )

    scheduler = queue["scheduler"]
    assert scheduler["default_devices"] == ["0", "1"]
    assert scheduler["max_concurrent_single_gpu_jobs"] == 2
    assert "CUDA_VISIBLE_DEVICES=0,1" in scheduler["multi_gpu_rule"]

    metadata = queue["accepted_run_metadata_required"]
    for required in (
        "CUDA_VISIBLE_DEVICES",
        "GPU model",
        "GPU count",
        "seed",
        "runtime",
        "metrics path",
        "OOM or failure reason if any",
    ):
        assert required in metadata

    bindings = queue["top_representative_bindings"]
    assert len(bindings) >= 7
    assert {binding["paper_id"] for binding in bindings} == set(PAPER_MATRICES)
    for binding in bindings:
        assert binding["external_work_id"].startswith("RWTOP2026-")
        assert binding["status"] == "pending_gpu_and_artifacts"
        assert "not exact" in binding["exact_reproduction_status"] or (
            "evaluation protocol only" in binding["exact_reproduction_status"]
        )
        assert binding["local_proxy_matrix_entries"], binding["binding_id"]
        assert "baseline_ablation_matrix.yaml" in binding["command_source"]
        assert "run_meta.yaml" in binding["artifact_requirement"]
        assert (
            "metrics" in binding["artifact_requirement"]
            or "metrics.json" in binding["artifact_requirement"]
        )

    sota_contract = queue["sota_comparison_contract"]
    assert "single run" in sota_contract["single_run_rule"]
    assert "proposed method" in sota_contract["same_protocol_population"]
    assert "every declared baseline" in sota_contract["same_protocol_population"]
    assert "runnable TOP representative" in sota_contract["same_protocol_population"]
    assert "matched seed set" in sota_contract["seed_protocol"]
    assert "minimum_seeds" in sota_contract["seed_protocol"]
    assert "95% confidence interval" in sota_contract["aggregate_statistics"]
    assert "effect size" in sota_contract["aggregate_statistics"]
    assert "failure_record" in sota_contract["seed_protocol"]
    assert "representative TOP proxy" in sota_contract["top_scope"]
    assert "exact external-method SOTA" in sota_contract["top_scope"]

    paper_queue = queue["paper_queue"]
    assert len(paper_queue) == 7
    assert [item["queue_id"] for item in paper_queue] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
        "Q7",
    ]
    assert {item["paper_id"] for item in paper_queue} == set(PAPER_MATRICES)

    for item in paper_queue:
        matrix_path = Path(item["matrix_path"])
        assert matrix_path == PAPER_MATRICES[item["paper_id"]]
        assert matrix_path.exists(), item["paper_id"]
        assert Path(item["goal_file"]).exists(), item["paper_id"]
        assert Path(item["base_config"]).exists(), item["paper_id"]
        assert item["minimum_seeds"] >= 3
        assert "CUDA_VISIBLE_DEVICES=0" in item["device_binding"]
        assert item["required_phases"] == [
            "proposed",
            "baselines",
            "ablations",
            "top_representatives",
        ]

    gate = queue["cross_paper_gate"]
    assert gate["queue_id"] == "Q8"
    assert "blocked" in gate["status"]
    assert "No SOTA wording is allowed" in gate["sota_rule"]
    assert "multi-seed aggregate evidence" in gate["sota_rule"]
    assert "submission_ready: true" in gate["submission_rule"]


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


def test_operator_attention_rejection_recovery_contract_blocks_unproven_claims() -> None:
    assert PAPER07_REJECTION_CONTRACT.exists()
    text = PAPER07_REJECTION_CONTRACT.read_text(encoding="utf-8")
    matrix = yaml.safe_load(PAPER07_MATRIX.read_text(encoding="utf-8"))

    for phrase in (
        "It is not accepted experiment evidence",
        "must not use SOTA",
        "paper remains not submission-ready",
        "paper/UXFD_paper/results/accepted_runs/TII_operator_attention/",
        "Q0 preflight",
        "Stop SOTA wording",
        "achieved=false",
    ):
        assert phrase in text

    top_ids = {entry["id"] for entry in matrix["top_recent_work"]}
    assert set(PAPER07_REQUIRED_TOP_IDS) <= top_ids
    assert "2024-2026 TOP representative" in "\n".join(matrix["strict_blockers"])


def test_operator_attention_reviewer_traceability_matrix_blocks_overclaims() -> None:
    assert PAPER07_REVIEWER_TRACE.exists()
    text = PAPER07_REVIEWER_TRACE.read_text(encoding="utf-8")
    contract_text = PAPER07_REJECTION_CONTRACT.read_text(encoding="utf-8")
    readiness_text = (
        PAPER07_MATRIX.parents[1] / "submission_prep/ieee_trans_readiness.md"
    ).read_text(encoding="utf-8")
    parent_matrix_text = (GOAL_DIR / "99_submission_readiness_matrix.md").read_text(
        encoding="utf-8"
    )

    for phrase in (
        "not accepted experiment evidence",
        "Weak industrial performance",
        "Theory-experiment mismatch",
        "Unclear innovation",
        "Insufficient recent/SOTA baselines",
        "DSOA v2",
        "OAS, OSS, and OCS",
        "must not claim",
        "parent objective audit is not achieved",
        "accepted_runs/TII_operator_attention",
    ):
        assert phrase in text

    assert "reviewer_traceability_matrix.md" in contract_text
    assert "reviewer_traceability_matrix.md" in readiness_text
    assert "reviewer_traceability_matrix.md" in parent_matrix_text


def test_fuzzy_xfd_baseline_ablation_matrix_is_command_bound_not_ready() -> None:
    assert PAPER05_MATRIX.exists()
    matrix = yaml.safe_load(PAPER05_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert (
        matrix["evidence_level"]
        == "config-target validated; reviewer-ablation smoke runner and manuscript checkpoint bound"
    )
    assert matrix["manuscript"]["entrypoint"] == "manuscript/final_tex/main.tex"
    assert "pdflatex" in matrix["manuscript"]["compile_command"]
    assert "pass" in matrix["manuscript"]["compile_status"]
    assert "evidence-snapshot only" in matrix["manuscript"]["evidence_status"]
    assert len(matrix["baselines"]) >= 6
    assert len(matrix["ablations"]) >= 6
    assert len(matrix["reviewer_requested_ablations"]) == 3
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

    for entry in matrix["reviewer_requested_ablations"]:
        assert entry["config_target_validated"] is True
        assert "run_reviewer_ablation_smoke.py" in entry["command"]
        assert "accepted_evidence=false" in entry["evidence_status"]
        assert "pending same-protocol" in entry["accepted_evidence_status"]

    blockers = "\n".join(matrix["strict_blockers"])
    assert "No accepted CWRU/XJTU or industrial multi-seed baseline table yet." in blockers
    assert "Hard-threshold, safety-fallback, and no-rule-output" not in blockers
    assert "No SOTA claim is allowed from this matrix alone." in blockers


def test_moe_baseline_matrix_records_ablation_blockers_not_ready() -> None:
    assert PAPER04_MATRIX.exists()
    matrix = yaml.safe_load(PAPER04_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert (
        matrix["evidence_level"]
        == "baseline config-target validated; MoE ablation smoke runner bound"
    )
    assert len(matrix["local_moe_evidence"]) >= 5
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
    assert len(bound_ablations) >= 6
    assert len(blocked_ablations) == 0
    assert any("run_expert_ablation_probe.py" in entry["command"] for entry in bound_ablations)
    assert any(
        entry["artifact"] == "scripts/run_moe_ablation_smoke.py"
        and "non-accepted smoke runner" in entry["status"]
        for entry in matrix["local_moe_evidence"]
    )
    assert not any("run_moe_ablation_smoke.py" in entry["command"] for entry in bound_ablations)

    blockers = "\n".join(matrix["strict_blockers"])
    assert "Only smoke MoE ablation runner artifacts exist" in blockers
    assert "No accepted TOP representative command/log/artifact mapping yet." in blockers
    assert "No SOTA claim is allowed from this matrix alone." in blockers


def test_toolkit_baseline_matrix_records_ablation_blockers_not_ready() -> None:
    assert PAPER01_MATRIX.exists()
    matrix = yaml.safe_load(PAPER01_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert (
        matrix["evidence_level"]
        == "baseline config-target validated; Toolkit ablation smoke and manuscript checkpoint bound"
    )
    assert matrix["manuscript"]["entrypoint"] == "manuscript/final_tex/main.tex"
    assert "pdflatex" in matrix["manuscript"]["compile_command"]
    assert "pass" in matrix["manuscript"]["compile_status"]
    assert "evidence checkpoint only" in matrix["manuscript"]["evidence_status"]
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
    assert len(bound_ablations) >= 6
    assert len(blocked_ablations) == 0
    assert any("trainer.extensions.explain.enable=false" in entry["command"] for entry in bound_ablations)
    assert not any("run_toolkit_ablations.py" in entry["command"] for entry in bound_ablations)

    readiness_text = (
        PAPER01_MATRIX.parents[1] / "submission_prep/ieee_trans_readiness.md"
    ).read_text(encoding="utf-8")
    assert "run_toolkit_ablations.py --condition all" in readiness_text
    assert "per-condition `run_meta.yaml` and `metrics.json`" in readiness_text
    assert "accepted_evidence: false" in readiness_text

    blockers = "\n".join(matrix["strict_blockers"])
    assert "Only smoke Toolkit ablation runner artifacts exist" in blockers
    assert "No accepted TOP representative command/log/artifact mapping yet." in blockers
    assert "No SOTA or submission-ready infrastructure claim" in blockers

    main_tex = PAPER01_MATRIX.parents[1] / "manuscript/final_tex/main.tex"
    manuscript_text = main_tex.read_text(encoding="utf-8")
    assert "\\documentclass[journal]{IEEEtran}" in manuscript_text
    assert "../../figures/example.pdf" not in manuscript_text
    assert "[论文标题]" not in manuscript_text
    assert "[请在此处" not in manuscript_text
    assert "overall_scores_comparison.png" in manuscript_text
    assert "not a final submission-ready manuscript" in manuscript_text


def test_1d2d_fusion_matrix_records_dummy_only_and_ablation_blockers() -> None:
    assert PAPER02_MATRIX.exists()
    matrix = yaml.safe_load(PAPER02_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert (
        matrix["evidence_level"]
        == "baseline config-target validated; fusion ablation smoke runner bound"
    )
    assert "pass in LQ_signal" in matrix["proposed"]["dummy_smoke_status"]
    assert "test_accuracy=0.39" in matrix["paper_local_demo"]["dummy_smoke_metric"]
    assert "PHM-Vibench HDF5" in matrix["paper_local_demo"]["real_h5_smoke_status"]
    assert "Target 8 is out of bounds" in matrix["paper_local_demo"]["failed_sanity_check"]
    assert len(matrix["baselines"]) >= 6
    assert len(matrix["ablations"]) >= 6
    assert all(
        "pass in LQ_signal" in entry.get("dummy_smoke_status", "")
        for entry in matrix["baselines"]
    )

    for entry in matrix["baselines"]:
        assert entry["config_target_validated"] is True
        assert "CUDA_VISIBLE_DEVICES=0" in entry["command"]
        assert "python main.py --config" in entry["command"]
        assert "pending" in entry["accepted_evidence_status"]

    passing_ablations = [
        entry
        for entry in matrix["ablations"]
        if "pass in LQ_signal" in entry.get("dummy_smoke_status", "")
        or "same run as B01" in entry.get("dummy_smoke_status", "")
    ]
    blocked_ablations = [
        entry for entry in matrix["ablations"] if entry["config_target_validated"] is False
    ]
    assert len(passing_ablations) >= 4
    assert len(blocked_ablations) == 0
    assert any(
        "FFT-only forward now passes" in entry.get("evidence_status", "")
        for entry in matrix["ablations"]
    )
    assert not any(
        "run_fusion_ablation_smoke.py" in entry["command"]
        for entry in matrix["ablations"]
    )
    readiness_text = (
        PAPER02_MATRIX.parents[1] / "submission_prep/ieee_trans_readiness.md"
    ).read_text(encoding="utf-8")
    assert "scripts/run_fusion_ablation_smoke.py" in readiness_text

    blockers = "\n".join(matrix["strict_blockers"])
    assert "FFT-only signal-layer ablation currently fails" not in blockers
    assert "Legacy ablation runner assumes GPU 2" not in blockers
    assert "Paper-local demo falls back" not in blockers
    assert "NatureMi" not in blockers
    assert "No SOTA claim is allowed from this matrix alone." in blockers


def test_llm_toolkit_matrix_records_package_gate_and_evidence_blockers() -> None:
    assert PAPER03_MATRIX.exists()
    matrix = yaml.safe_load(PAPER03_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert (
        matrix["evidence_level"]
        == "baseline config-target validated; LLM package import gate fixed; accepted evidence still blocked"
    )
    assert "pass in LQ_signal" in matrix["proposed"]["dummy_smoke_status"]
    assert len(matrix["llm_demo_evidence"]) >= 4
    assert len(matrix["baselines"]) >= 6
    assert len(matrix["ablations"]) >= 6
    assert any("standalone template LLM" in entry["label"] for entry in matrix["baselines"])
    assert any("package-based template LLM pipeline" in entry["label"] for entry in matrix["llm_demo_evidence"])
    assert any("14 passed" in entry["status"] for entry in matrix["llm_demo_evidence"])
    assert any(
        "accepted_evidence=false" in entry["status"]
        for entry in matrix["llm_demo_evidence"]
    )

    main_py_baselines = [
        entry
        for entry in matrix["baselines"]
        if "python main.py --config" in entry["command"]
    ]
    assert len(main_py_baselines) >= 6
    for entry in main_py_baselines:
        assert entry["config_target_validated"] is True
        assert "CUDA_VISIBLE_DEVICES=0" in entry["command"]
        assert "pending" in entry["accepted_evidence_status"]

    bound_ablations = [
        entry for entry in matrix["ablations"] if entry["config_target_validated"] is True
    ]
    blocked_ablations = [
        entry for entry in matrix["ablations"] if entry["config_target_validated"] is False
    ]
    assert len(bound_ablations) >= 7
    assert len(blocked_ablations) == 0
    assert any("package-based template pipeline" in entry["label"] for entry in bound_ablations)
    assert any(
        "package-based template pipeline" in entry["label"]
        and "accepted_evidence=false" in entry.get("evidence_status", "")
        for entry in bound_ablations
    )
    assert any("core toolkit unit-test gate" in entry["label"] for entry in bound_ablations)
    assert not any(
        "run_llm_evidence_smoke.py" in entry["command"]
        for entry in bound_ablations
    )
    readiness_text = (
        PAPER03_MATRIX.parents[1] / "submission_prep/ieee_trans_readiness.md"
    ).read_text(encoding="utf-8")
    assert "run_llm_evidence_smoke.py --condition all" in readiness_text
    assert "no-checker" in readiness_text
    assert "llm_evidence_package_contract.md" in readiness_text

    assert PAPER03_LLM_EVIDENCE_CONTRACT.exists()
    contract_text = PAPER03_LLM_EVIDENCE_CONTRACT.read_text(encoding="utf-8")
    for phrase in (
        "not accepted experiment evidence",
        "accepted_same_protocol",
        "unsupported-claim rate",
        "latency p50 and p95",
        "prompt_set.json",
        "responses.jsonl",
        "accepted_evidence=false",
        "TOP-Q7-TIMESEG",
        "No SOTA claim is allowed",
    ):
        assert phrase in contract_text

    parent_matrix_text = (GOAL_DIR / "99_submission_readiness_matrix.md").read_text(
        encoding="utf-8"
    )
    assert "LLM Explainable FD Toolkit" in parent_matrix_text
    assert "submission_prep/llm_evidence_package_contract.md" in parent_matrix_text
    assert "LLM evidence package contract checkpoint" in parent_matrix_text

    blockers = "\n".join(matrix["strict_blockers"])
    assert "The manuscript/ieee_tii/main.tex entrypoint is a conservative compile checkpoint" in blockers
    assert "No final IEEE TeX entrypoint exists" not in blockers
    assert "run_meta.yaml,metrics.json" in blockers
    assert "not accepted LLM evidence packages" in blockers
    assert "Only smoke hallucination-checker, context-removal, and latency-sweep runners exist" in blockers
    assert "No SOTA or human-centered decision-support claim" in blockers


def test_neuralsymbolic_matrix_records_proposition_blockers_not_ready() -> None:
    assert PAPER06_MATRIX.exists()
    matrix = yaml.safe_load(PAPER06_MATRIX.read_text(encoding="utf-8"))

    assert matrix["submission_ready"] is False
    assert (
        matrix["evidence_level"]
        == "baseline config-target validated; proposition evidence partial; source-backed mapping and manuscript checkpoint bound"
    )
    assert matrix["manuscript"]["entrypoint"] == "manuscript/final_tex/main.tex"
    assert "pdflatex" in matrix["manuscript"]["compile_command"]
    assert "pass" in matrix["manuscript"]["compile_status"]
    assert "not final submission-ready text" in matrix["manuscript"]["evidence_status"]
    assert len(matrix["proposition_evidence"]) >= 7
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
    assert len(bound_ablations) >= 7
    assert len(blocked_ablations) == 0
    assert any("logit_scale=0.1" in entry["command"] for entry in bound_ablations)
    assert any("logit_scale=1.0" in entry["command"] for entry in bound_ablations)
    assert any(
        entry["id"] == "MAP-ABL"
        and "run_mapping_ablation_smoke.py --condition no_mapping" in entry["command"]
        for entry in matrix["proposition_evidence"]
    )
    assert any(
        entry["id"] == "MAP-SRC"
        and entry["artifact"] == "report/source_backed_mapping_report.json"
        and "accepted_evidence=false" in entry["current_result"]
        for entry in matrix["proposition_evidence"]
    )
    assert any(
        entry["id"] == "A06"
        and "build_source_backed_mapping.py" in entry["command"]
        and "source_backed=true" in entry["evidence_status"]
        for entry in bound_ablations
    )

    p2_entries = [
        entry
        for entry in matrix["proposition_evidence"]
        if entry["id"].startswith("P2")
    ]
    assert len(p2_entries) >= 2
    assert any("proposition_2_verified=false" in entry["current_result"] for entry in p2_entries)
    assert any("does not override" in entry["accepted_evidence_status"] for entry in p2_entries)
    assert any(
        "p2_evidence_contract.md" in entry["accepted_evidence_status"]
        for entry in p2_entries
    )

    blockers = "\n".join(matrix["strict_blockers"])
    assert "P2 has only scope-limited synthetic hooks" in blockers
    assert "no accepted real-data robustness protocol supports final P2 yet" in blockers
    assert "Cross-method mapping report is scripted" not in blockers
    assert "Manuscript entrypoint remains placeholder-heavy" not in blockers
    assert "No SOTA claim is allowed from this matrix alone." in blockers

    main_tex = PAPER06_MATRIX.parents[1] / "manuscript/final_tex/main.tex"
    manuscript_text = main_tex.read_text(encoding="utf-8")
    assert "\\documentclass[journal]{IEEEtran}" in manuscript_text
    assert "../../figures/example.pdf" not in manuscript_text
    assert "[论文标题]" not in manuscript_text
    assert "[请在此处" not in manuscript_text
    assert "mapping_validation.png" in manuscript_text


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
