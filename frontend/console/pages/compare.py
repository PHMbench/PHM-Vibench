"""Compare page renderer."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from frontend.console.adapters.runs import (
    compare_protocols,
    discover_recent_runs,
    hydrate_run_record,
    load_metrics,
)
from frontend.console.pages.shared import filter_run_summaries
from frontend.console.state import COMPARE_BASELINE_RUN_KEY, COMPARE_SELECTED_RUNS_KEY
from frontend.console.theme import hero


def render_compare() -> None:
    """Render compare guard rails and metric tables."""
    hero(
        "Fair Benchmark Compare",
        "compare / guard rails",
        "Compare only when pipeline and split-defining config fields line up. Missing evidence is surfaced, not hidden.",
    )
    summaries = discover_recent_runs(limit=100)
    if len(summaries) < 2:
        st.warning("Need at least two discovered runs to compare.")
        return
    summaries = filter_run_summaries(summaries)
    if len(summaries) < 2:
        st.info("Need at least two filtered runs to compare.")
        return

    run_ids = [summary.run_id for summary in summaries]
    baseline_default = st.session_state.get(COMPARE_BASELINE_RUN_KEY, run_ids[0])
    if baseline_default not in run_ids:
        baseline_default = run_ids[0]
    selected_default = st.session_state.get(COMPARE_SELECTED_RUNS_KEY, run_ids[:2])
    selected_default = [run_id for run_id in selected_default if run_id in run_ids]
    if len(selected_default) < 2:
        selected_default = run_ids[:2]

    baseline_id = st.selectbox(
        "Baseline run",
        run_ids,
        index=run_ids.index(baseline_default),
        key=COMPARE_BASELINE_RUN_KEY,
    )
    selected_ids = st.multiselect(
        "Runs to compare",
        run_ids,
        default=selected_default,
        key=COMPARE_SELECTED_RUNS_KEY,
    )
    if baseline_id not in selected_ids:
        selected_ids = [baseline_id, *selected_ids]
        selected_ids = list(dict.fromkeys(selected_ids))
        st.info("Baseline run was automatically included in the compare set.")
    if len(selected_ids) < 2:
        st.info("Select at least two runs.")
        return

    summary_map = {summary.run_id: summary for summary in summaries}
    selected_records = [
        hydrate_run_record(summary_map[run_id])
        for run_id in selected_ids
        if run_id in summary_map
    ]

    st.caption(f"Baseline: {baseline_id}")
    incomplete = [record.run_id for record in selected_records if record.evidence_state != "complete"]
    if incomplete:
        st.warning(
            "Selected runs have incomplete evidence: "
            + ", ".join(incomplete)
            + ". Compare results should be read with caution."
        )

    report = compare_protocols(selected_records)
    if report.compatible:
        st.success("Protocol signature is compatible across the selected runs.")
    else:
        st.error("Protocol signature mismatch detected. Direct comparison should be treated as unfair.")

    if report.hard_mismatches:
        st.subheader("Hard Mismatches")
        st.dataframe(pd.DataFrame(report.hard_mismatches), use_container_width=True, hide_index=True)
    if report.soft_mismatches:
        st.subheader("Soft Differences")
        st.dataframe(pd.DataFrame(report.soft_mismatches), use_container_width=True, hide_index=True)

    st.subheader("Metric Comparison")
    metric_rows: List[Dict[str, Any]] = []
    for record in selected_records:
        row = {
            "run_id": record.run_id,
            "baseline": record.run_id == baseline_id,
            "evidence": record.evidence_state,
            "signature": record.protocol_signature.summary,
        }
        row.update(load_metrics(record))
        metric_rows.append(row)
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    st.caption("Hard fields include pipeline, task family, data factory, metadata source, and split-defining task fields.")
