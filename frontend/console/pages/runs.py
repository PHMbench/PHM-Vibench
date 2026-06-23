"""Runs page renderer."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml

from frontend.console.adapters.runs import (
    discover_recent_runs,
    figure_files,
    hydrate_run_record,
    list_artifacts,
    load_config_snapshot,
    load_metrics,
    load_metrics_history,
)
from frontend.console.pages.shared import filter_run_summaries, go_to_page, run_selectbox
from frontend.console.state import SELECTED_RUN_KEY
from frontend.console.theme import hero


def render_runs() -> None:
    """Render recent run discovery and previews."""
    hero(
        "Run Monitor",
        "runs / trace / evidence",
        "Inspect repo-native run evidence: manifest, config snapshot, metrics CSV, figures, predictions, and checkpoints.",
    )
    summaries = discover_recent_runs(limit=100)
    if not summaries:
        st.warning("No runs discovered under results/ or save/.")
        return
    summaries = filter_run_summaries(summaries)
    if not summaries:
        st.info("No runs match the current filters.")
        return

    summary_rows: List[Dict[str, Any]] = [
        {
            "run_id": summary.run_id,
            "timestamp": summary.timestamp,
            "evidence": summary.evidence_state,
            "missing": sum(not item.exists for item in list_artifacts(summary)),
            "run_dir": str(summary.run_dir),
        }
        for summary in summaries
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    selected_summary = run_selectbox(summaries, key="runs_selected_run")
    if selected_summary is None:
        return

    selected = hydrate_run_record(selected_summary)
    artifacts = list_artifacts(selected)
    present = [item for item in artifacts if item.exists]
    missing = [item.label for item in artifacts if not item.exists]
    overview_tab, metrics_tab, config_tab, figures_tab, artifacts_tab = st.tabs(
        ["Overview", "Metrics", "Config", "Figures", "Artifacts"]
    )

    with overview_tab:
        st.subheader("Selected Run")
        st.code(str(selected.run_dir), language="text")
        st.caption(f"Evidence state: {selected.evidence_state} | Signature: {selected.protocol_signature.summary}")
        summary_df = pd.DataFrame(
            [
                {"field": "run_id", "value": selected.run_id},
                {"field": "timestamp", "value": selected.timestamp},
                {"field": "present_artifacts", "value": len(present)},
                {"field": "missing_artifacts", "value": len(missing)},
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        if missing:
            st.warning(f"Missing artifacts: {', '.join(missing)}")
        action_left, action_right = st.columns(2)
        with action_left:
            if st.button("Open Selected Artifacts", use_container_width=True):
                go_to_page("pages/06_Artifacts.py", state_updates={SELECTED_RUN_KEY: selected.run_id})
        with action_right:
            st.caption("Artifacts page gives the same run a dedicated evidence-first view.")

    with metrics_tab:
        st.subheader("Metrics Snapshot")
        metrics = load_metrics(selected)
        if metrics:
            st.json(metrics, expanded=False)
        else:
            st.info("No metrics payload discovered.")
        st.subheader("Metrics History")
        history = load_metrics_history(selected)
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        else:
            st.info("No logs/**/metrics.csv found.")

    with config_tab:
        st.subheader("Config Snapshot")
        config = load_config_snapshot(selected)
        if config:
            st.code(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), language="yaml")
            st.caption(
                json.dumps(
                    {
                        "pipeline": config.get("pipeline", ""),
                        "task": config.get("task", {}),
                        "model": config.get("model", {}),
                    },
                    ensure_ascii=False,
                )
            )
        else:
            st.info("config_snapshot.yaml not present.")

    with figures_tab:
        st.subheader("Figures")
        figures = figure_files(selected)
        if figures:
            previewable = [path for path in figures if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if previewable:
                st.image(
                    [str(path) for path in previewable],
                    caption=[path.name for path in previewable],
                    use_column_width=True,
                )
            st.caption("\n".join(str(path) for path in figures))
        else:
            st.info("No figures directory found.")

    with artifacts_tab:
        st.subheader("Artifact Inventory")
        artifact_df = pd.DataFrame(
            [
                {
                    "artifact": item.label,
                    "exists": item.exists,
                    "kind": item.kind,
                    "path": str(item.path) if item.path else "",
                }
                for item in artifacts
            ]
        )
        st.dataframe(artifact_df, use_container_width=True, hide_index=True)
