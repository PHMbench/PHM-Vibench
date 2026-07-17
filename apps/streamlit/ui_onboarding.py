"""Streamlit components for first-run readiness and template guidance."""

from __future__ import annotations

from typing import Optional

import streamlit as st

try:
    from .onboarding import ReadinessReport, TemplateDataStatus, TemplateProfile
except ImportError:  # pragma: no cover - Streamlit executes app.py as a script.
    from onboarding import ReadinessReport, TemplateDataStatus, TemplateProfile  # type: ignore


_STATUS_ICON = {"ready": "✓", "warning": "!", "blocked": "×"}


def template_option_label(template_id: str, profile: TemplateProfile) -> str:
    """Keep the select box readable while exposing the experiment intent."""

    return f"{profile.title} — {template_id}"


def render_readiness_sidebar(report: ReadinessReport) -> None:
    st.sidebar.markdown("### First-run readiness")
    if report.can_execute and not report.warnings:
        st.sidebar.success("CPU smoke prerequisites look ready.")
    elif report.can_execute:
        st.sidebar.warning("Ready with a machine-local configuration warning.")
    else:
        st.sidebar.error(f"{len(report.blocked)} prerequisite(s) need attention.")

    with st.sidebar.expander("Environment checks", expanded=not report.can_execute):
        for item in report.checks:
            icon = _STATUS_ICON.get(item.status, "•")
            st.markdown(f"**{icon} {item.label}**")
            st.caption(item.detail)
            if item.action:
                st.code(item.action, language="text")


def render_readiness_banner(report: ReadinessReport) -> None:
    """Show only information that changes the user's next action."""

    if report.blocked:
        st.error(
            "The workspace can be explored, but experiment launch is disabled until "
            "the blocked first-run checks are fixed."
        )
    elif report.warnings:
        st.warning(
            "The environment is runnable, but configs/local/local.yaml is active. "
            "Review it before treating the smoke run as a clean baseline."
        )
    else:
        st.success("Environment ready for the repository-shipped CPU smoke experiment.")


def render_template_profile(profile: TemplateProfile) -> None:
    with st.container(border=True):
        st.markdown(f"#### {profile.title}")
        st.caption(profile.summary)
        columns = st.columns(4)
        columns[0].metric("Difficulty", profile.difficulty)
        columns[1].metric("Data", profile.data_label)
        columns[2].metric("Recommended device", profile.device_label)
        columns[3].metric("Time", profile.estimated_time)
        if profile.badges:
            st.caption(" · ".join(profile.badges))
        if profile.next_step:
            st.info(profile.next_step)


def render_template_data_status(status: TemplateDataStatus) -> None:
    if status.ready:
        st.success(status.detail)
    else:
        st.error(status.detail)
        if status.action:
            st.markdown(f"**Next action:** {status.action}")
    if status.data_root or status.metadata_path:
        with st.expander("Resolved data paths"):
            if status.data_root:
                st.code(status.data_root, language="text")
            if status.metadata_path:
                st.code(status.metadata_path, language="text")


def render_launch_blockers(
    readiness: ReadinessReport,
    data_status: Optional[TemplateDataStatus],
) -> None:
    reasons = [item.label for item in readiness.blocked]
    if data_status is not None and not data_status.ready:
        reasons.append("Selected template data")
    if reasons:
        st.warning(
            "Run is disabled until these checks pass: " + ", ".join(reasons) + "."
        )
