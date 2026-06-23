from __future__ import annotations

from pathlib import Path


MAINTAINED_DOCS = [
    Path("README.md"),
    Path("AGENTS.md"),
    Path("configs/README.md"),
    Path("docs/README.md"),
    Path("docs/REPO_INDEX.md"),
    Path("docs/app_usage.md"),
    Path("frontend/README.md"),
]


def test_maintained_docs_point_to_frontend_streamlit_entrypoint() -> None:
    expected = "streamlit run frontend/streamlit_app.py"
    required_docs = [
        Path("README.md"),
        Path("AGENTS.md"),
        Path("docs/app_usage.md"),
        Path("frontend/README.md"),
    ]

    for path in required_docs:
        assert expected in path.read_text(encoding="utf-8"), path


def test_maintained_docs_do_not_reference_legacy_streamlit_paths() -> None:
    forbidden = [
        "streamlit run streamlit_app.py",
        "streamlit run app/",
        "app/gui.py",
        "app/gui_refactored.py",
    ]

    hits = []
    for path in MAINTAINED_DOCS:
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden:
            if snippet in text:
                hits.append(f"{path}: {snippet}")

    assert hits == []
