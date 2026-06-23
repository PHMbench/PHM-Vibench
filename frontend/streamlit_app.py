"""PHMfactory Streamlit entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "streamlit is required to run the research console. Install dependencies from requirements.txt first."
    ) from exc

from frontend.console.pages.workbench import render_workbench
from frontend.console.theme import inject_theme, setup_page


def _render_workbench_page() -> None:
    setup_page("PHMfactory | Workbench")
    inject_theme()
    render_workbench()


PAGES = [
    (_render_workbench_page, "Workbench", ":material/tune:"),
    ("pages/02_Compose.py", "Compose", ":material/edit_note:"),
    ("pages/03_Runs.py", "Runs", ":material/monitoring:"),
    ("pages/04_Compare.py", "Compare", ":material/balance:"),
    ("pages/05_Registry.py", "Registry", ":material/widgets:"),
    ("pages/06_Artifacts.py", "Artifacts", ":material/folder_open:"),
]


def main() -> None:
    """Run the Streamlit research console."""
    if hasattr(st, "Page") and hasattr(st, "navigation"):
        pages = [st.Page(path, title=title, icon=icon) for path, title, icon in PAGES]
        navigator = st.navigation(pages, position="top")
        navigator.run()
        return

    _render_workbench_page()
    st.sidebar.info("Legacy Streamlit fallback detected. The Workbench is rendered here; other pages may appear in the sidebar if your Streamlit version supports the pages/ directory.")


if __name__ == "__main__":
    main()
