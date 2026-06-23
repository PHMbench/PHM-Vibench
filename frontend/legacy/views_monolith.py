"""Streamlit page renderers for the PHMfactory console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from app.frontend.configuration import (
    build_launch_request,
    build_preflight_result,
    load_config_catalog,
    parse_override_text,
    run_launch_request,
)
from app.frontend.registry import (
    load_data_factories,
    load_model_registry,
    load_task_registry,
    load_trainer_presets,
    load_trainer_registry,
)
from app.frontend.runs import (
    RunRecord,
    compare_protocols,
    discover_run_records,
    figure_files,
    list_artifacts,
    load_config_snapshot,
    load_metrics,
    load_metrics_history,
    preview_predictions,
    preview_text,
)
from app.frontend.state import (
    LAST_LAUNCH_KEY,
    LAST_PREFLIGHT_KEY,
    SELECTED_CONFIG_KEY,
    SELECTED_RUN_KEY,
)
from app.frontend.theme import card, hero, metric_card


def _page_link(label: str, page: str, help_text: str = "") -> None:
    if hasattr(st, "page_link"):
        st.page_link(page, label=label, help=help_text)
    else:
        st.caption(f"{label}: {page}")


def _catalog_df() -> pd.DataFrame:
    entries = load_config_catalog()
    rows = [
        {
            "category": entry.category,
            "config": entry.path,
            "pipeline": entry.pipeline or "Pipeline_01_default",
            "status": entry.status or "/",
            "description": entry.description,
        }
        for entry in entries
    ]
    return pd.DataFrame(rows)


def _default_config_path() -> str:
    entries = load_config_catalog()
    if not entries:
        return "configs/demo/00_smoke/dummy_dg.yaml"
    demos = [entry.path for entry in entries if entry.category == "demo"]
    return demos[0] if demos else entries[0].path


def _config_selectbox(label: str, key: str) -> str:
    entries = load_config_catalog()
    paths = [entry.path for entry in entries]
    current = st.session_state.get(SELECTED_CONFIG_KEY, _default_config_path())
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


def _status_icon(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _evidence_counts(records: Sequence[RunRecord]) -> Dict[str, int]:
    counts = {"complete": 0, "partial": 0, "minimal": 0}
    for record in records:
        counts[record.evidence_state] = counts.get(record.evidence_state, 0) + 1
    return counts


def render_workbench() -> None:
    """Render the default home page."""
    hero(
        "PHMfactory Workbench",
        "research control plane",
        "Start from protocol, recent runs, and evidence. Keep the CLI authoritative.",
    )

    catalog_df = _catalog_df()
    runs = discover_run_records(limit=10)
    evidence = _evidence_counts(runs)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Maintained Configs", str(len(catalog_df.index)), "Registry-backed demos and bases")
    with col2:
        metric_card("Recent Runs", str(len(runs)), "Discovered from artifacts/manifest.json")
    with col3:
        metric_card(
            "Evidence Complete",
            str(evidence.get("complete", 0)),
            "Runs with config, metrics, manifest, and figures",
        )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Recommended Starts")
        demo_df = catalog_df[catalog_df["category"] == "demo"].head(6)
        st.dataframe(demo_df, use_container_width=True, hide_index=True)
        st.caption("Workbench surfaces maintained demos first; compose and launch still resolve through the CLI path.")

        st.subheader("Quick Actions")
        quick1, quick2, quick3 = st.columns(3)
        with quick1:
            _page_link("Open Compose", "pages/02_Compose.py", "Prepare and launch a config")
        with quick2:
            _page_link("Open Runs", "pages/03_Runs.py", "Inspect recent run evidence")
        with quick3:
            _page_link("Open Compare", "pages/04_Compare.py", "Check protocol compatibility")
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
                "No launch executed in this browser session yet. Compose will always show the exact CLI command first.",
                eyebrow="session",
            )

        if runs:
            recent = runs[0]
            card(
                "Latest Evidence",
                (
                    f"run_id: {recent.run_id}\n"
                    f"timestamp: {recent.timestamp}\n"
                    f"evidence: {recent.evidence_state}\n"
                    f"run_dir: {recent.run_dir}"
                ),
                eyebrow="latest run",
            )


def render_compose() -> None:
    """Render config compose, preflight, and launch."""
    hero(
        "Compose Experiment",
        "compose / inspect / launch",
        "Build a run from maintained configs, inspect the resolved YAML, then launch through the exact CLI command.",
    )

    selected_config = _config_selectbox("Config", key="compose_config_select")
    notes = st.text_input("Notes", value="", help="Passed through to main.py --notes.")
    default_overrides = "trainer.num_epochs=1\ndata.num_workers=0"
    override_text = st.text_area(
        "Overrides",
        value=st.session_state.get("compose_override_text", default_overrides),
        height=140,
        help="One key=value override per line.",
        key="compose_override_text",
    )
    overrides = parse_override_text(override_text)

    preflight = None
    preflight_error: Optional[str] = None
    try:
        preflight = build_preflight_result(selected_config, overrides, notes=notes)
    except Exception as exc:
        preflight_error = str(exc)

    if preflight is not None:
        st.session_state[LAST_PREFLIGHT_KEY] = {
            "config_path": selected_config,
            "shell_command": preflight.shell_command,
            "output_preview": preflight.output_preview,
        }
        st.subheader("Command Preview")
        st.code(preflight.shell_command, language="bash")
        st.caption(f"Predicted output pattern: {preflight.output_preview}")
    else:
        st.subheader("Command Preview")
        st.code(build_launch_request(selected_config, overrides, notes=notes).shell_command, language="bash")
        st.error(f"Preflight failed: {preflight_error}")

    launch_col, inspect_col = st.columns([0.24, 0.76])
    with launch_col:
        launch_clicked = st.button(
            "Launch via CLI",
            type="primary",
            use_container_width=True,
            disabled=preflight is None,
        )
    with inspect_col:
        st.caption("The UI never bypasses the CLI contract. It builds the exact command above and runs it as a subprocess.")

    if launch_clicked:
        request = build_launch_request(selected_config, overrides, notes=notes)
        status = st.status("Launching through main.py", expanded=True)
        output_box = st.empty()
        streamed_lines: List[str] = []

        def _on_output(line: str) -> None:
            streamed_lines.append(line)
            output_box.code("".join(streamed_lines)[-8000:], language="text")

        result = run_launch_request(request, on_output=_on_output)
        st.session_state[LAST_LAUNCH_KEY] = {
            "config_path": selected_config,
            "shell_command": request.shell_command,
            "returncode": result.returncode,
            "output": result.output[-8000:],
        }
        if result.returncode == 0:
            status.update(label="Launch completed", state="complete", expanded=False)
            st.success("CLI run completed.")
        else:
            status.update(label="Launch failed", state="error", expanded=True)
            st.error(f"CLI exited with code {result.returncode}.")

    left, right = st.columns([1.0, 1.0])
    with left:
        st.subheader("Preflight Sanity")
        if preflight is None:
            st.info("Resolve the config successfully to inspect sanity checks.")
        else:
            sanity_rows = [
                {
                    "status": _status_icon(bool(item.get("ok"))),
                    "check": item.get("check", ""),
                    "message": item.get("message", ""),
                    "fix": item.get("fix", ""),
                }
                for item in preflight.sanity
            ]
            st.dataframe(pd.DataFrame(sanity_rows), use_container_width=True, hide_index=True)

            st.subheader("Resolved YAML")
            st.code(preflight.resolved_yaml, language="yaml")
    with right:
        st.subheader("Field Sources")
        if preflight is None:
            st.info("Field sources and targets appear after a successful preflight.")
        else:
            st.dataframe(pd.DataFrame(preflight.sources), use_container_width=True, hide_index=True)
            st.subheader("Instantiation Targets")
            st.json(preflight.targets, expanded=False)


def _run_options(records: Sequence[RunRecord]) -> List[str]:
    return [record.run_id for record in records]


def _select_run(records: Sequence[RunRecord], key: str) -> Optional[RunRecord]:
    if not records:
        return None
    run_ids = _run_options(records)
    current = st.session_state.get(SELECTED_RUN_KEY, run_ids[0])
    if current not in run_ids:
        current = run_ids[0]
    selected_run_id = st.selectbox("Run", run_ids, index=run_ids.index(current), key=key)
    st.session_state[SELECTED_RUN_KEY] = selected_run_id
    for record in records:
        if record.run_id == selected_run_id:
            return record
    return records[0]


def _run_summary_rows(records: Sequence[RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        config = load_config_snapshot(record)
        rows.append(
            {
                "run_id": record.run_id,
                "timestamp": record.timestamp,
                "evidence": record.evidence_state,
                "pipeline": config.get("pipeline", ""),
                "task": f"{config.get('task', {}).get('type', '')}/{config.get('task', {}).get('name', '')}",
                "model": config.get("model", {}).get("name", ""),
                "run_dir": str(record.run_dir),
            }
        )
    return rows


def render_runs() -> None:
    """Render recent run discovery and previews."""
    hero(
        "Run Monitor",
        "runs / trace / evidence",
        "Inspect repo-native run evidence: manifest, config snapshot, metrics CSV, figures, predictions, and checkpoints.",
    )
    records = discover_run_records(limit=100)
    if not records:
        st.warning("No runs discovered under results/ or save/.")
        return

    st.dataframe(pd.DataFrame(_run_summary_rows(records)), use_container_width=True, hide_index=True)
    selected = _select_run(records, key="runs_selected_run")
    if selected is None:
        return

    st.subheader("Selected Run")
    st.code(str(selected.run_dir), language="text")
    st.caption(f"Evidence state: {selected.evidence_state} | Signature: {selected.protocol_signature.summary}")

    artifacts = list_artifacts(selected)
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

    left, right = st.columns([1.0, 1.0])
    with left:
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
    with right:
        st.subheader("Config Snapshot")
        config = load_config_snapshot(selected)
        if config:
            st.code(
                json.dumps(config, ensure_ascii=False, indent=2),
                language="json",
            )
        else:
            st.info("config_snapshot.yaml not present.")

    st.subheader("Figures")
    figures = figure_files(selected)
    if figures:
        previewable = [path for path in figures if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if previewable:
            st.image([str(path) for path in previewable], caption=[path.name for path in previewable], use_column_width=True)
        st.caption("\n".join(str(path) for path in figures))
    else:
        st.info("No figures directory found.")


def render_compare() -> None:
    """Render compare guard rails and metric tables."""
    hero(
        "Fair Benchmark Compare",
        "compare / guard rails",
        "Compare only when pipeline and split-defining config fields line up. Missing evidence is surfaced, not hidden.",
    )
    records = discover_run_records(limit=100)
    if len(records) < 2:
        st.warning("Need at least two discovered runs to compare.")
        return

    run_ids = _run_options(records)
    selected_ids = st.multiselect("Runs to compare", run_ids, default=run_ids[:2])
    selected_records = [record for record in records if record.run_id in selected_ids]
    if len(selected_records) < 2:
        st.info("Select at least two runs.")
        return

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
            "evidence": record.evidence_state,
            "signature": record.protocol_signature.summary,
        }
        row.update(load_metrics(record))
        metric_rows.append(row)
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    st.subheader("Compare Notes")
    st.caption("Hard fields: pipeline, task family, data factory, metadata source, and split-defining task fields.")


def render_registry() -> None:
    """Render registry explorers and shipped presets."""
    hero(
        "Registry Explorer",
        "registry / presets / extension points",
        "See what the maintained system already wires before you open source files or invent new names.",
    )
    data_tab, model_tab, task_tab, trainer_tab, config_tab = st.tabs(
        ["Data", "Models", "Tasks", "Trainers", "Configs"]
    )
    with data_tab:
        st.dataframe(pd.DataFrame(load_data_factories()), use_container_width=True, hide_index=True)
    with model_tab:
        st.dataframe(pd.DataFrame(load_model_registry()), use_container_width=True, hide_index=True)
    with task_tab:
        st.dataframe(pd.DataFrame(load_task_registry()), use_container_width=True, hide_index=True)
    with trainer_tab:
        st.markdown("**Registered Trainers**")
        st.dataframe(pd.DataFrame(load_trainer_registry()), use_container_width=True, hide_index=True)
        st.markdown("**Trainer Presets**")
        st.dataframe(pd.DataFrame(load_trainer_presets()), use_container_width=True, hide_index=True)
    with config_tab:
        st.dataframe(_catalog_df(), use_container_width=True, hide_index=True)


def _preview_artifact(record: RunRecord, label: str) -> None:
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
            st.image([str(path) for path in previewable], caption=[path.name for path in previewable], use_column_width=True)
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
    records = discover_run_records(limit=100)
    if not records:
        st.warning("No artifacts discovered under results/ or save/.")
        return

    selected = _select_run(records, key="artifacts_selected_run")
    if selected is None:
        return

    inventory = list_artifacts(selected)
    labels = [item.label for item in inventory]
    selected_label = st.selectbox("Artifact", labels, index=0)
    inventory_df = pd.DataFrame(
        [
            {
                "artifact": item.label,
                "kind": item.kind,
                "exists": item.exists,
                "path": str(item.path) if item.path else "",
            }
            for item in inventory
        ]
    )
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)
    st.subheader("Preview")
    _preview_artifact(selected, selected_label)
