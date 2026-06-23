"""Shared helpers for Streamlit console pages."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd
import streamlit as st

from frontend.console.adapters.configuration import ConfigCatalogEntry, load_config_catalog
from frontend.console.adapters.runs import RunSummary
from frontend.console.state import (
    COMPARE_BASELINE_RUN_KEY,
    COMPARE_SELECTED_RUNS_KEY,
    LAST_LAUNCH_KEY,
    LAST_PREFLIGHT_KEY,
    RUN_FILTER_EVIDENCE_KEY,
    RUN_FILTER_QUERY_KEY,
    SELECTED_CONFIG_KEY,
    SELECTED_RUN_KEY,
)


def page_link(label: str, page: str, help_text: str = "") -> None:
    """Render a page link when the Streamlit runtime supports it."""
    if hasattr(st, "page_link"):
        st.page_link(page, label=label, help=help_text)
        return
    st.caption(f"{label}: {page}")


def catalog_entries() -> list[ConfigCatalogEntry]:
    """Return maintained config entries for UI use."""
    return load_config_catalog()


def catalog_df() -> pd.DataFrame:
    """Return the maintained config catalog as a dataframe."""
    rows = [
        {
            "category": entry.category,
            "config": entry.path,
            "pipeline": entry.pipeline or "Pipeline_01_default",
            "status": entry.status or "/",
            "description": entry.description,
        }
        for entry in catalog_entries()
    ]
    return pd.DataFrame(rows)


def default_config_path() -> str:
    """Return the first demo config, or the first maintained config."""
    entries = catalog_entries()
    if not entries:
        return "configs/demo/00_smoke/dummy_dg.yaml"
    demos = [entry.path for entry in entries if entry.category == "demo"]
    return demos[0] if demos else entries[0].path


def selected_catalog_entry() -> Optional[ConfigCatalogEntry]:
    """Return the currently selected config when it is present in the catalog."""
    current_path = st.session_state.get(SELECTED_CONFIG_KEY, default_config_path())
    for entry in catalog_entries():
        if entry.path == current_path:
            return entry
    return None


def catalog_entry_for_path(path: str) -> Optional[ConfigCatalogEntry]:
    """Resolve a maintained catalog entry by path."""
    for entry in catalog_entries():
        if entry.path == path:
            return entry
    return None


def current_protocol_context() -> tuple[Optional[ConfigCatalogEntry], str, str]:
    """Return the current protocol entry, path, and source label."""
    last_preflight = st.session_state.get(LAST_PREFLIGHT_KEY)
    if isinstance(last_preflight, dict) and last_preflight.get("config_path"):
        path = str(last_preflight["config_path"])
        return catalog_entry_for_path(path), path, "last preflight"

    last_launch = st.session_state.get(LAST_LAUNCH_KEY)
    if isinstance(last_launch, dict) and last_launch.get("config_path"):
        path = str(last_launch["config_path"])
        return catalog_entry_for_path(path), path, "last launch"

    selected = st.session_state.get(SELECTED_CONFIG_KEY)
    if isinstance(selected, str) and selected:
        return catalog_entry_for_path(selected), selected, "draft selection"

    default_path = default_config_path()
    return catalog_entry_for_path(default_path), default_path, "default maintained config"


def config_selectbox(label: str, key: str) -> str:
    """Render the maintained config selectbox."""
    entries = catalog_entries()
    paths = [entry.path for entry in entries]
    current = st.session_state.get(SELECTED_CONFIG_KEY, default_config_path())
    if current not in paths and paths:
        current = paths[0]
    labels = {
        entry.path: f"[{entry.category}] {entry.path} - {entry.description}"
        for entry in entries
    }
    index = paths.index(current) if current in paths else 0
    selected = st.selectbox(
        label,
        options=paths,
        index=index,
        format_func=lambda value: labels.get(value, value),
        key=key,
    )
    st.session_state[SELECTED_CONFIG_KEY] = selected
    return selected


def set_selected_run(run_id: str) -> None:
    """Persist the currently selected run across pages."""
    st.session_state[SELECTED_RUN_KEY] = run_id


def set_compare_selection(run_ids: Sequence[str], baseline_run_id: Optional[str] = None) -> None:
    """Persist compare selection state across pages."""
    chosen = list(dict.fromkeys(run_ids))
    st.session_state[COMPARE_SELECTED_RUNS_KEY] = chosen
    if baseline_run_id:
        st.session_state[COMPARE_BASELINE_RUN_KEY] = baseline_run_id


def go_to_page(page: str, state_updates: Optional[dict[str, Any]] = None) -> None:
    """Persist state updates and navigate when Streamlit supports page switching."""
    if state_updates:
        for key, value in state_updates.items():
            st.session_state[key] = value
    if hasattr(st, "switch_page"):
        st.switch_page(page)
        return
    st.info(f"Selection saved. Open {page} to continue.")


def filter_run_summaries(summaries: Sequence[RunSummary]) -> list[RunSummary]:
    """Filter run summaries by query and evidence state."""
    if not summaries:
        return []

    all_states = ["complete", "partial", "minimal"]
    controls_left, controls_right = st.columns([0.62, 0.38])
    with controls_left:
        query = st.text_input(
            "Find runs",
            value=st.session_state.get(RUN_FILTER_QUERY_KEY, ""),
            key=RUN_FILTER_QUERY_KEY,
            placeholder="Filter by run_id, timestamp, evidence, or path",
        ).strip()
    with controls_right:
        selected_states = st.multiselect(
            "Evidence state",
            options=all_states,
            default=st.session_state.get(RUN_FILTER_EVIDENCE_KEY, all_states),
            key=RUN_FILTER_EVIDENCE_KEY,
        )

    filtered: list[RunSummary] = []
    for summary in summaries:
        haystack = " ".join(
            [
                summary.run_id,
                summary.timestamp,
                summary.evidence_state,
                str(summary.run_dir),
            ]
        ).lower()
        if query and query.lower() not in haystack:
            continue
        if selected_states and summary.evidence_state not in selected_states:
            continue
        filtered.append(summary)
    return filtered


def run_selectbox(
    summaries: Sequence[RunSummary],
    key: str,
    label: str = "Run",
) -> Optional[RunSummary]:
    """Render a run selectbox and return the selected summary."""
    if not summaries:
        return None
    run_ids = [summary.run_id for summary in summaries]
    current = st.session_state.get(SELECTED_RUN_KEY, run_ids[0])
    if current not in run_ids:
        current = run_ids[0]
    selected_run_id = st.selectbox(label, run_ids, index=run_ids.index(current), key=key)
    st.session_state[SELECTED_RUN_KEY] = selected_run_id
    for summary in summaries:
        if summary.run_id == selected_run_id:
            return summary
    return summaries[0]
