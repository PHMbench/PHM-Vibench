"""Theme helpers for the Streamlit research console."""

from __future__ import annotations

import html
from typing import Optional

import streamlit as st


THEME_CSS = """
<style>
:root {
  --phm-bg: #0b0d10;
  --phm-panel: #12161b;
  --phm-panel-strong: #0f1317;
  --phm-text: #e9e0d4;
  --phm-muted: #a39586;
  --phm-accent: #c69461;
  --phm-accent-strong: #dfb179;
  --phm-trace: #7fb0ad;
  --phm-good: #8bb39f;
  --phm-warn: #d39f5e;
  --phm-danger: #c96f6f;
  --phm-line: rgba(198, 148, 97, 0.20);
  --phm-radius: 8px;
  --phm-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
}

html, body, [class*="css"] {
  font-family: Inter, "Segoe UI", Arial, sans-serif;
}

.stApp {
  background: linear-gradient(180deg, #080a0d 0%, #0c1115 100%);
  color: var(--phm-text);
}

.block-container {
  padding-top: 1.5rem;
  padding-bottom: 2.5rem;
}

[data-testid="stSidebar"] {
  background: rgba(12, 16, 20, 0.94);
  border-right: 1px solid var(--phm-line);
}

h1, h2, h3 {
  font-family: Georgia, "Times New Roman", serif;
  letter-spacing: 0;
}

.phm-hero,
.phm-card,
.phm-metric,
.phm-artifact {
  border: 1px solid var(--phm-line);
  border-radius: var(--phm-radius);
  background: linear-gradient(180deg, rgba(20, 24, 29, 0.90), rgba(12, 16, 20, 0.96));
  box-shadow: var(--phm-shadow);
}

.phm-hero {
  padding: 1.4rem 1.5rem;
  margin-bottom: 1rem;
}

.phm-card,
.phm-artifact {
  padding: 1rem 1.05rem;
  margin-bottom: 0.85rem;
}

.phm-metric {
  padding: 1rem 1.05rem;
  min-height: 110px;
}

.phm-eyebrow {
  color: var(--phm-accent-strong);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
  margin-bottom: 0.35rem;
}

.phm-subtle {
  color: var(--phm-muted);
  line-height: 1.6;
}

.phm-metric-label {
  color: var(--phm-muted);
  font-size: 0.9rem;
}

.phm-metric-value {
  display: block;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 2rem;
  margin: 0.35rem 0;
}

.phm-mono {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

.phm-code-hint {
  color: var(--phm-muted);
  margin: 0.4rem 0 0.1rem 0;
}

div[data-testid="stMetric"],
.stDataFrame,
.stTable {
  border-radius: var(--phm-radius);
}

.stButton > button,
.stDownloadButton > button {
  border-radius: var(--phm-radius);
  border: 1px solid var(--phm-line);
  background: linear-gradient(180deg, rgba(24, 28, 33, 0.96), rgba(16, 19, 24, 0.96));
  color: var(--phm-text);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #d3a06a, #ba8758);
  color: #111;
  border: 0;
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
  border-radius: var(--phm-radius);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.35rem;
}

.stTabs [data-baseweb="tab"] {
  border-radius: var(--phm-radius);
}

.phm-status-good { color: var(--phm-good); }
.phm-status-warn { color: var(--phm-warn); }
.phm-status-danger { color: var(--phm-danger); }
</style>
"""


def setup_page(page_title: str) -> None:
    """Configure the current Streamlit page."""
    st.set_page_config(
        page_title=page_title,
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_theme() -> None:
    """Inject the shared CSS theme."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def hero(title: str, eyebrow: str, subtitle: str) -> None:
    """Render a page hero."""
    st.markdown(
        (
            '<div class="phm-hero">'
            f'<div class="phm-eyebrow">{html.escape(eyebrow)}</div>'
            f"<h1>{html.escape(title)}</h1>"
            f'<div class="phm-subtle">{html.escape(subtitle)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def card(title: str, body: str, eyebrow: Optional[str] = None) -> None:
    """Render a themed content card."""
    eyebrow_html = (
        f'<div class="phm-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    )
    body_html = html.escape(body).replace("\n", "<br/>")
    st.markdown(
        (
            '<div class="phm-card">'
            f"{eyebrow_html}"
            f"<h3>{html.escape(title)}</h3>"
            f'<div class="phm-subtle">{body_html}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> None:
    """Render a themed metric card."""
    st.markdown(
        (
            '<div class="phm-metric">'
            f'<div class="phm-metric-label">{html.escape(label)}</div>'
            f'<span class="phm-metric-value">{html.escape(value)}</span>'
            f'<div class="phm-subtle">{html.escape(note)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
