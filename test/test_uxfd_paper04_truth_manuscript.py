from pathlib import Path


PAPER04_ROOT = Path("paper/UXFD_paper/MOE_explainable")
FINAL_TEX = PAPER04_ROOT / "manuscript" / "final_tex" / "main.tex"
DRAFT_MD = PAPER04_ROOT / "manuscript" / "draft_md" / "draft.md"
SYNC_SCRIPT = PAPER04_ROOT / "scripts" / "sync_truth_first_manuscript.py"


def test_paper04_truth_manuscript_has_clean_tex_boundary() -> None:
    text = FINAL_TEX.read_text(encoding="utf-8")

    assert text.count("\\end{document}") == 1
    after_end = text.split("\\end{document}", 1)[1].strip()
    assert after_end == ""
    assert "## 202603" not in text
    assert "accepted: `True`" not in text
    assert "PHM-Vibench copy 2" not in text
    assert "main.py --config_dir" not in text


def test_paper04_truth_manuscript_stays_non_submission_ready() -> None:
    draft = DRAFT_MD.read_text(encoding="utf-8")
    tex = FINAL_TEX.read_text(encoding="utf-8")
    script = SYNC_SCRIPT.read_text(encoding="utf-8")

    for text in (draft, tex, script):
        assert "submission-ready" not in text
        assert "SOTA" not in text
        assert "state-of-the-art" not in text.lower()

    assert "External submission readiness is still governed by the parent UXFD submission gate" in tex
    assert "外部投稿 readiness 仍以父仓库 UXFD submission gate 为准" in draft
