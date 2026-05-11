from pathlib import Path


PAPER02_ROOT = Path("paper/UXFD_paper/1D-2D_fusion_explainable")
CURRENT_EXEC_ROOT = "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix"


def _read(relative_path: str) -> str:
    return (PAPER02_ROOT / relative_path).read_text(encoding="utf-8")


def test_paper02_control_docs_do_not_use_stale_execution_entries() -> None:
    control_docs = (
        "README.md",
        "CORE.md",
        "program.md",
        "paper_blueprint.md",
        "innovation_contract.md",
    )

    for relative_path in control_docs:
        text = _read(relative_path)
        assert "PHM-Vibench copy 2" not in text
        assert "main.py --config_dir" not in text
        assert "configs/unified_baseline" not in text
        assert "Paper/1D-2D_fusion_explainable/configs" not in text
        assert "CUDA_VISIBLE_DEVICES=2" not in text
        assert "GPU_ID = 2" not in text

    assert CURRENT_EXEC_ROOT in _read("README.md")
    assert CURRENT_EXEC_ROOT in _read("CORE.md")
    assert CURRENT_EXEC_ROOT in _read("program.md")


def test_paper02_innovation_contract_is_bound_to_authority_docs() -> None:
    contract = PAPER02_ROOT / "innovation_contract.md"
    assert contract.exists()

    for relative_path in ("README.md", "CORE.md", "paper_blueprint.md"):
        assert "innovation_contract.md" in _read(relative_path)

    contract_text = contract.read_text(encoding="utf-8")
    for required_text in (
        "Physical-Semantic-Geometric Three-Layer Alignment",
        "Interpretable 1D-2D Fusion Network",
        "High-Accuracy Multi-Dataset Diagnosis",
        "CWRU",
        "XJTU",
        "THU_006",
        "MoE",
        "TSPN",
        "OperatorAttention",
        ">= 0.98",
    ):
        assert required_text in contract_text
