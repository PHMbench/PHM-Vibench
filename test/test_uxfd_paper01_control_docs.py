from pathlib import Path


PAPER01_ROOT = Path("paper/UXFD_paper/Explainable_FD_Toolkit")
CURRENT_EXEC_ROOT = "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix"


def _read(relative_path: str) -> str:
    return (PAPER01_ROOT / relative_path).read_text(encoding="utf-8")


def test_paper01_control_docs_do_not_use_stale_execution_roots() -> None:
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
        assert "python Paper/Explainable_FD_Toolkit" not in text
        assert "Paper/Explainable_FD_Toolkit/scripts" not in text

    assert CURRENT_EXEC_ROOT in _read("README.md")
    assert CURRENT_EXEC_ROOT in _read("CORE.md")
    assert CURRENT_EXEC_ROOT in _read("program.md")


def test_paper01_innovation_contract_is_bound_to_authority_docs() -> None:
    contract = PAPER01_ROOT / "innovation_contract.md"
    assert contract.exists()

    for relative_path in ("README.md", "CORE.md", "paper_blueprint.md"):
        assert "innovation_contract.md" in _read(relative_path)

    contract_text = contract.read_text(encoding="utf-8")
    for required_text in (
        "Explainability OS For Fault Diagnosis",
        "Captum",
        "SHAP",
        "LIME",
        "CWRU",
        "XJTU",
        "THU_018_basic",
        "accepted",
    ):
        assert required_text in contract_text
