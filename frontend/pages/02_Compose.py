"""Compose page."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend.console.pages.compose import render_compose
from frontend.console.theme import inject_theme, setup_page

setup_page("PHMfactory | Compose")
inject_theme()
render_compose()
