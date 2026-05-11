import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


PAPER02_ROOT = Path("paper/UXFD_paper/1D-2D_fusion_explainable")
PAPER02_TEX = PAPER02_ROOT / "paper_draft/NMI_Paper1_Fusion1D2D.tex"
PAPER02_MATRIX = PAPER02_ROOT / "submission_prep/baseline_ablation_matrix.yaml"


def test_paper02_canonical_tex_uses_ieee_template() -> None:
    text = PAPER02_TEX.read_text(encoding="utf-8")
    matrix = yaml.safe_load(PAPER02_MATRIX.read_text(encoding="utf-8"))

    assert r"\documentclass[journal]{IEEEtran}" in text
    assert "NatureMi" not in text
    assert r"\bibliographystyle{IEEEtranN}" in text
    assert r"\bibliography{paper_draft/references}" in text
    assert "NatureMi" not in "\n".join(matrix["strict_blockers"])


def test_paper02_canonical_tex_compiles_with_ieee_template(tmp_path: Path) -> None:
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    result = subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={tmp_path}",
            str(PAPER02_TEX.relative_to(PAPER02_ROOT)),
        ],
        cwd=PAPER02_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "NMI_Paper1_Fusion1D2D.pdf").exists()
    log_text = (tmp_path / "NMI_Paper1_Fusion1D2D.log").read_text(encoding="utf-8")
    assert "undefined citations" not in log_text.lower()
    assert "undefined references" not in log_text.lower()
    assert "rerun to get citations correct" not in log_text.lower()
