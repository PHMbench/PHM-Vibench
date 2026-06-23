"""Workbench page renderer."""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd
import streamlit as st

from frontend.console.adapters.runs import RunSummary, discover_recent_runs, list_artifacts
from frontend.console.pages.shared import current_protocol_context, go_to_page, page_link
from frontend.console.state import (
    COMPARE_BASELINE_RUN_KEY,
    COMPARE_SELECTED_RUNS_KEY,
    LAST_LAUNCH_KEY,
    SELECTED_RUN_KEY,
)
from frontend.console.theme import card, hero, metric_card


def _evidence_counts(records: Sequence[RunSummary]) -> Dict[str, int]:
    counts = {"complete": 0, "partial": 0, "minimal": 0}
    for record in records:
        counts[record.evidence_state] = counts.get(record.evidence_state, 0) + 1
    return counts


def render_workbench() -> None:
    """Render the research workbench homepage."""
    hero(
        "PHMfactory Workbench",
        "research control plane",
        "Start from protocol, recent runs, and evidence. Keep the CLI authoritative.",
    )

    runs = discover_recent_runs(limit=10)
    evidence = _evidence_counts(runs)
    current_protocol, protocol_path, protocol_source = current_protocol_context()

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Current Protocol", current_protocol.path if current_protocol else protocol_path, protocol_source)
    with col2:
        metric_card("Recent Runs", str(len(runs)), "Resolved from manifest evidence only")
    with col3:
        metric_card(
            "Evidence Complete",
            str(evidence.get("complete", 0)),
            "Runs with config, metrics, manifest, and figures",
        )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Current Protocol")
        protocol_df = pd.DataFrame(
            [
                {
                    "config": current_protocol.path if current_protocol else protocol_path,
                    "pipeline": current_protocol.pipeline if current_protocol else "",
                    "status": current_protocol.status if current_protocol else "",
                    "description": current_protocol.description if current_protocol else "Unregistered config path in session state.",
                    "source": protocol_source,
                }
            ]
        )
        st.dataframe(protocol_df, use_container_width=True, hide_index=True)

        st.subheader("Recent Runs")
        recent_rows = [
            {
                "run_id": record.run_id,
                "timestamp": record.timestamp,
                "evidence": record.evidence_state,
                "missing": sum(not item.exists for item in list_artifacts(record)),
                "run_dir": str(record.run_dir),
            }
            for record in runs
        ]
        if recent_rows:
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No run evidence discovered under results/ or save/.")

        st.subheader("Quick Actions")
        quick1, quick2, quick3, quick4 = st.columns(4)
        with quick1:
            page_link("Open Compose", "pages/02_Compose.py", "Prepare and launch a config")
        with quick2:
            disabled = not runs
            if st.button("Inspect Latest Run", use_container_width=True, disabled=disabled):
                latest_run_id = runs[0].run_id
                go_to_page("pages/03_Runs.py", state_updates={SELECTED_RUN_KEY: latest_run_id})
        with quick3:
            disabled = not runs
            if st.button("Open Latest Artifacts", use_container_width=True, disabled=disabled):
                latest_run_id = runs[0].run_id
                go_to_page("pages/06_Artifacts.py", state_updates={SELECTED_RUN_KEY: latest_run_id})
        with quick4:
            disabled = len(runs) < 2
            if st.button("Compare Latest Pair", use_container_width=True, disabled=disabled):
                compare_ids = [runs[0].run_id, runs[1].run_id]
                go_to_page(
                    "pages/04_Compare.py",
                    state_updates={
                        COMPARE_BASELINE_RUN_KEY: runs[0].run_id,
                        COMPARE_SELECTED_RUNS_KEY: compare_ids,
                    },
                )
    with right:
        last_launch = st.session_state.get(LAST_LAUNCH_KEY)
        if last_launch:
            card(
                "Last Launch",
                (
                    f"Config: {last_launch.get('config_path', '')}\n"
                    f"Return code: {last_launch.get('returncode', '')}\n"
                    f"Command: {last_launch.get('shell_command', '')}"
                ),
                eyebrow="session",
            )
        else:
            card(
                "Launch Surface",
                "No launch executed in this browser session yet. Compose always shows the exact CLI command first.",
                eyebrow="session",
            )

        artifact_note = "No artifacts discovered yet."
        if runs:
            latest = runs[0]
            missing = [item.label for item in list_artifacts(latest) if not item.exists]
            artifact_note = (
                f"run_id: {latest.run_id}\n"
                f"timestamp: {latest.timestamp}\n"
                f"evidence: {latest.evidence_state}\n"
                f"manifest: {latest.manifest_path}\n"
                f"missing: {', '.join(missing) if missing else 'none'}"
            )
        card("Latest Artifact", artifact_note, eyebrow="evidence")

        card(
            "Artifact Status",
            (
                f"complete: {evidence.get('complete', 0)}\n"
                f"partial: {evidence.get('partial', 0)}\n"
                f"minimal: {evidence.get('minimal', 0)}"
            ),
            eyebrow="inventory",
        )
