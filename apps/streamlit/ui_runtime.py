"""Live run, result, artifact, and log rendering."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Optional

import streamlit as st

try:
    from .config_service import format_command
    from .result_service import (
        artifact_groups,
        discover_results,
        format_bytes,
        headline_metrics,
    )
    from .run_service import (
        RunRecord,
        RunServiceError,
        cancel_run,
        elapsed_seconds,
        get_run,
        list_runs,
        read_log_tail,
        restart_run,
    )
    from .ui_theme import _render_error
except ImportError:  # pragma: no cover
    from config_service import format_command  # type: ignore
    from result_service import (  # type: ignore
        artifact_groups,
        discover_results,
        format_bytes,
        headline_metrics,
    )
    from run_service import (  # type: ignore
        RunRecord,
        RunServiceError,
        cancel_run,
        elapsed_seconds,
        get_run,
        list_runs,
        read_log_tail,
        restart_run,
    )
    from ui_theme import _render_error  # type: ignore

_STATUS_ICON = {
    "starting": "◌",
    "running": "●",
    "cancelling": "◐",
    "succeeded": "✓",
    "failed": "×",
    "cancelled": "■",
    "orphaned": "!",
    "detached": "↗",
}


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def _status_label(record: RunRecord) -> str:
    return f"{_STATUS_ICON.get(record.status, '•')} {record.status.upper()}"


def _render_run_selector(repo_root: Path) -> Optional[str]:
    try:
        runs = list_runs(repo_root, limit=20)
    except RunServiceError:
        return None
    if not runs:
        st.sidebar.caption("No experiment runs yet.")
        return None
    ids = tuple(item.run_id for item in runs)
    preferred = st.session_state.selected_run_id or st.session_state.active_run_id
    if preferred not in ids:
        preferred = ids[0]
    lookup = {item.run_id: item for item in runs}
    selected = st.sidebar.selectbox(
        "Recent runs",
        ids,
        index=ids.index(preferred),
        format_func=lambda run_id: (
            f"{_STATUS_ICON.get(lookup[run_id].status, '•')} "
            f"{lookup[run_id].template_id or run_id} · {run_id[-8:]}"
        ),
        key=f"run-selector::{preferred}",
    )
    st.session_state.selected_run_id = selected
    return selected


def _render_overview(record: RunRecord, bundle: Any) -> None:
    st.markdown("**Actual reproduction command**")
    st.code(format_command(record.command), language="bash")
    left, right = st.columns(2)
    with left:
        st.markdown("**Run directory**")
        st.code(str(record.run_dir), language="text")
        st.markdown("**Output root**")
        st.code(record.output_root or "save", language="text")
    with right:
        st.markdown("**Template / mode**")
        st.write(f"{record.template_id or 'custom'} · {record.mode or 'unknown'}")
        st.markdown("**Discovered roots**")
        st.code("\n".join(str(path) for path in bundle.roots) or "None", language="text")
    if record.error:
        st.warning(record.error)
    if record.metadata:
        with st.expander("Run metadata"):
            st.json(dict(record.metadata))


def _render_metrics(bundle: Any) -> None:
    for warning in bundle.warnings:
        st.warning(warning)
    headlines = headline_metrics(bundle.metrics)
    if headlines:
        cols = st.columns(len(headlines))
        for col, (name, value) in zip(cols, headlines):
            col.metric(name, f"{value:.5g}" if isinstance(value, float) else value)
    if not bundle.metrics:
        st.info(
            "No structured metrics artifact was found. The run and raw logs remain available."
        )
        return
    for table in bundle.metrics:
        st.markdown(f"**{table.source.name}**")
        if table.warning:
            st.warning(table.warning)
        if table.rows:
            st.dataframe(list(table.rows), use_container_width=True, hide_index=True)


def _render_artifacts(bundle: Any, run_id: str) -> None:
    groups = artifact_groups(bundle)
    images = groups.get("image", ())
    if images:
        st.markdown("**Visual artifacts**")
        columns = st.columns(min(3, len(images)))
        for index, artifact in enumerate(images):
            with columns[index % len(columns)]:
                try:
                    st.image(
                        str(artifact.path),
                        caption=artifact.relative_path,
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(f"Could not render {artifact.relative_path}: {exc}")
    rows = [
        {
            "type": artifact.kind,
            "file": artifact.relative_path,
            "size": format_bytes(artifact.size_bytes),
            "modified": artifact.modified_at,
            "root": str(artifact.root),
        }
        for artifact in bundle.artifacts
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No artifacts have been discovered yet.")

    downloadable = [
        item
        for item in bundle.artifacts
        if item.path.is_file() and item.size_bytes <= 5_000_000
    ][:12]
    if downloadable:
        with st.expander("Download small artifacts"):
            for index, artifact in enumerate(downloadable):
                try:
                    data = artifact.path.read_bytes()
                except OSError:
                    continue
                st.download_button(
                    f"Download {artifact.relative_path}",
                    data=data,
                    file_name=artifact.path.name,
                    key=f"artifact::{run_id}::{index}",
                    use_container_width=True,
                )


def _render_logs(record: RunRecord) -> None:
    log = read_log_tail(record)
    st.code(log or "No log output yet.", language="text")
    log_path = record.run_dir / "run.log"
    if log_path.is_file() and log_path.stat().st_size <= 10_000_000:
        st.download_button(
            "Download full log",
            data=log_path.read_bytes(),
            file_name=f"{record.run_id}.log",
            key=f"download-log::{record.run_id}",
            use_container_width=True,
        )


def _render_run_actions(repo_root: Path, record: RunRecord) -> None:
    left, middle, right = st.columns(3)
    if record.is_active and record.status != "detached":
        if left.button(
            "Cancel run",
            type="secondary",
            key=f"cancel::{record.run_id}",
            use_container_width=True,
        ):
            try:
                cancel_run(repo_root, record.run_id)
                st.toast("Cancellation requested.")
                st.rerun()
            except RunServiceError as exc:
                _render_error("The run could not be cancelled.", exc)
    else:
        left.button(
            "Cancel run",
            disabled=True,
            key=f"cancel-disabled::{record.run_id}",
            use_container_width=True,
        )

    if record.is_terminal:
        if middle.button(
            "Restart same run",
            type="primary",
            key=f"restart::{record.run_id}",
            use_container_width=True,
        ):
            try:
                restarted = restart_run(repo_root, record.run_id)
                st.session_state.active_run_id = restarted.run_id
                st.session_state.selected_run_id = restarted.run_id
                st.toast("Experiment restarted from its immutable snapshot.")
                st.rerun()
            except RunServiceError as exc:
                _render_error("The run could not be restarted.", exc)
    else:
        middle.button(
            "Restart same run",
            disabled=True,
            key=f"restart-disabled::{record.run_id}",
            use_container_width=True,
        )

    manifest = record.run_dir / "run.json"
    if manifest.is_file():
        right.download_button(
            "Download run manifest",
            data=manifest.read_bytes(),
            file_name=f"{record.run_id}.json",
            key=f"manifest::{record.run_id}",
            use_container_width=True,
        )


@st.fragment(run_every="2s")
def _render_live_run(repo_root_text: str, run_id: str) -> None:
    repo_root = Path(repo_root_text)
    try:
        record = get_run(repo_root, run_id)
    except RunServiceError as exc:
        _render_error("The selected run could not be loaded.", exc)
        return

    st.markdown(
        f'<span class="phm-status">{html.escape(_status_label(record))}</span> '
        f'<span class="phm-muted">{html.escape(record.run_id)}</span>',
        unsafe_allow_html=True,
    )
    status_cols = st.columns(4)
    status_cols[0].metric("Status", record.status)
    status_cols[1].metric("Elapsed", _format_duration(elapsed_seconds(record)))
    status_cols[2].metric("PID", record.pid if record.pid is not None else "—")
    status_cols[3].metric(
        "Exit code",
        record.exit_code if record.exit_code is not None else "—",
    )
    _render_run_actions(repo_root, record)

    try:
        bundle = discover_results(repo_root, record)
    except Exception as exc:  # keep logs/actions available if artifact parsing fails.
        st.warning(f"Result discovery failed without affecting the run: {exc}")
        bundle = None

    tabs = st.tabs(("Overview", "Metrics", "Artifacts", "Logs"))
    with tabs[0]:
        if bundle is not None:
            _render_overview(record, bundle)
    with tabs[1]:
        if bundle is not None:
            _render_metrics(bundle)
    with tabs[2]:
        if bundle is not None:
            _render_artifacts(bundle, record.run_id)
    with tabs[3]:
        _render_logs(record)
