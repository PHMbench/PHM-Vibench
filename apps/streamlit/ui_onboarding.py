"""Streamlit components for first-run readiness and template guidance."""

from __future__ import annotations

from typing import Optional
from pathlib import Path

import streamlit as st

try:
    from .config_service import format_command
    from .onboarding import DoctorReport, ReadinessReport, TemplateDataStatus, TemplateProfile, run_public_doctor
except ImportError:  # pragma: no cover - Streamlit executes app.py as a script.
    from config_service import format_command  # type: ignore
    from onboarding import DoctorReport, ReadinessReport, TemplateDataStatus, TemplateProfile, run_public_doctor  # type: ignore


_STATUS_ICON = {"ready": "✓", "warning": "!", "blocked": "×"}


def template_option_label(template_id: str, profile: TemplateProfile) -> str:
    """Keep the select box readable while exposing experiment intent."""

    return f"{profile.title} — {template_id}"


def render_readiness_sidebar(report: ReadinessReport) -> None:
    st.sidebar.markdown("### First-run readiness")
    if report.can_execute and not report.warnings:
        st.sidebar.success("Basic CPU smoke prerequisites were discovered.")
    elif report.can_execute:
        st.sidebar.warning("Basic prerequisites found; review non-blocking warnings.")
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
            "Basic prerequisites were found, but there are warnings. Doctor can "
            "check real imports before the selected experiment is validated."
        )
    else:
        st.info("Basic smoke prerequisites found. Use Doctor to verify real imports; validation and preflight still apply.")



def render_public_doctor(repo_root: Path) -> None:
    """Offer real import diagnostics without blocking the UI or starting training."""

    st.sidebar.caption(
        "Presence checks do not import Torch or Lightning. Use the existing "
        "PHMFactory doctor to verify the active Python environment."
    )
    if st.sidebar.button(
        "Verify runtime imports (doctor)",
        use_container_width=True,
        key="run-phmfactory-doctor",
    ):
        with st.spinner("Running phmfactory doctor without training..."):
            st.session_state["phmfactory_doctor_report"] = run_public_doctor(repo_root)

    report = st.session_state.get("phmfactory_doctor_report")
    if not isinstance(report, DoctorReport):
        return
    if report.passed:
        st.sidebar.success(
            "Doctor passed the installed core-runtime checks. "
            "Selected experiment validation is still required."
        )
    else:
        st.sidebar.error(
            "Doctor could not verify the runtime. Fix the reported dependency or "
            "environment problem and run Doctor again."
        )
    with st.sidebar.expander("Doctor details", expanded=not report.passed):
        st.code(format_command(report.command), language="bash")
        if report.error:
            st.error(report.error)
        if report.stdout:
            st.code(report.stdout, language="text")
        if report.stderr:
            st.code(report.stderr, language="text")


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
