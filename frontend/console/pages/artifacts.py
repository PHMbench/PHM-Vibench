"""Artifacts page renderer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.console.adapters.runs import (
    discover_recent_runs,
    figure_files,
    hydrate_run_record,
    list_artifacts,
    load_metrics_history,
    preview_predictions,
    preview_text,
)
from frontend.console.pages.shared import filter_run_summaries, run_selectbox
from frontend.console.theme import hero


def _preview_artifact(record, label: str) -> None:
    if label == "config_snapshot.yaml":
        st.code(preview_text(record.config_snapshot), language="yaml")
        return
    if label == "test_result_*.csv":
        if record.metrics_path.exists():
            st.dataframe(pd.read_csv(record.metrics_path), use_container_width=True, hide_index=True)
        else:
            st.info("Artifact not present.")
        return
    if label == "artifacts/manifest.json":
        st.json(record.manifest, expanded=False)
        return
    if label == "figures/":
        figures = figure_files(record)
        if not figures:
            st.info("Artifact not present.")
            return
        previewable = [path for path in figures if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if previewable:
            st.image(
                [str(path) for path in previewable],
                caption=[path.name for path in previewable],
                use_column_width=True,
            )
        st.caption("\n".join(str(path) for path in figures))
        return
    if label == "logs/**/metrics.csv":
        history = load_metrics_history(record)
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        else:
            st.info("Artifact not present.")
        return
    if label == "artifacts/predictions.npz":
        summary = preview_predictions(record.predictions_path)
        if summary:
            st.json(summary, expanded=False)
        else:
            st.info("Artifact not present.")
        return
    target = next((item for item in list_artifacts(record) if item.label == label), None)
    if target and target.path.exists():
        st.code(str(target.path), language="text")
    else:
        st.info("Artifact not present.")


def render_artifacts() -> None:
    """Render the artifact-first explorer."""
    hero(
        "Artifact Surface",
        "artifacts / preview / export",
        "Surface config snapshots, metric CSVs, manifests, figures, predictions, and checkpoints as first-class evidence.",
    )
    summaries = discover_recent_runs(limit=100)
    if not summaries:
        st.warning("No artifacts discovered under results/ or save/.")
        return
    summaries = filter_run_summaries(summaries)
    if not summaries:
        st.info("No runs match the current filters.")
        return

    selected_summary = run_selectbox(summaries, key="artifacts_selected_run")
    if selected_summary is None:
        return

    selected = hydrate_run_record(selected_summary)
    inventory = list_artifacts(selected)
    present = [item for item in inventory if item.exists]
    missing = [item for item in inventory if not item.exists]
    labels = [item.label for item in inventory]
    default_label = present[0].label if present else labels[0]
    selected_label = st.selectbox("Artifact", labels, index=labels.index(default_label))
    present_df = pd.DataFrame(
        [
            {
                "artifact": item.label,
                "kind": item.kind,
                "exists": item.exists,
                "path": str(item.path) if item.path else "",
            }
            for item in present
        ]
    )
    missing_df = pd.DataFrame(
        [
            {
                "artifact": item.label,
                "kind": item.kind,
                "exists": item.exists,
                "path": str(item.path) if item.path else "",
            }
            for item in missing
        ]
    )
    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.caption(f"Present artifacts: {len(present)}")
        if not present_df.empty:
            st.dataframe(present_df, use_container_width=True, hide_index=True)
        else:
            st.info("No present artifacts discovered for this run.")
    with summary_right:
        st.caption(f"Missing artifacts: {len(missing)}")
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.info("No missing artifacts for this run.")
    st.subheader("Preview")
    _preview_artifact(selected, selected_label)
