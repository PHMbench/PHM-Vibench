import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER06_ROOT = REPO_ROOT / "paper/UXFD_paper/Neuralsymbolic_theory"
SCRIPT = PAPER06_ROOT / "scripts/build_source_backed_mapping.py"
EXPECTED_PAPERS = {
    "1D-2D_fusion_explainable",
    "MOE_explainable",
    "Paper_fuzzy_XFD",
    "Explainable_FD_Toolkit",
    "LLM_Explainable_FD_Toolkit",
    "TII_operator_attention",
}


def test_source_backed_mapping_script_generates_nonaccepted_report(tmp_path: Path) -> None:
    output_json = tmp_path / "source_backed_mapping_report.json"
    output_md = tmp_path / "source_backed_mapping_report.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "source_backed=true" in completed.stdout
    assert output_json.exists()
    assert output_md.exists()

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["report_id"] == "paper06_source_backed_cross_method_mapping"
    assert report["source_backed"] is True
    assert report["accepted_evidence"] is False
    assert report["paper_count"] == 6

    papers = {paper["paper_id"]: paper for paper in report["papers"]}
    assert set(papers) == EXPECTED_PAPERS
    for paper_id, paper in papers.items():
        assert paper["source_backed"] is True, paper_id
        assert paper["accepted_evidence"] is False, paper_id
        assert not paper["missing_required_terms"], paper_id
        assert not paper["missing_layers"], paper_id
        assert paper["evidence"], paper_id
        for record in paper["evidence"]:
            assert record["exists"] is True, (paper_id, record["path"])
            source_path = (PAPER06_ROOT / record["path"]).resolve()
            assert source_path.exists(), source_path
            assert record["matched_terms"], (paper_id, record["path"])

    markdown = output_md.read_text(encoding="utf-8")
    assert "Accepted evidence: `false`" in markdown
    assert "It does not prove model performance" in markdown
