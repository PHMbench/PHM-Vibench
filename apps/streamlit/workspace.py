"""Application orchestration for the PHM-Vibench Streamlit workspace."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import streamlit as st

try:
    from .config_service import (
        Catalog,
        ConfigServiceError,
        RegistryEntry,
        ValidationReport,
        apply_overrides,
        build_main_command,
        dump_yaml,
        entry_by_id,
        find_repo_root,
        format_command,
        get_nested,
        group_entries,
        inspect_config,
        load_catalog,
        load_registry,
        normalize_overrides,
        parse_override_lines,
        parse_yaml_text,
        resolve_repo_path,
    )
    from .onboarding import (
        OnboardingError,
        apply_safe_defaults,
        assess_template_data,
        collect_environment_readiness,
        load_template_profiles,
        profile_for,
    )
    from .run_service import RunConflictError, RunRequest, RunServiceError, start_run
    from .runtime_policy import inspect_execution_yaml, inspect_portable_config
    from .ui_onboarding import (
        render_launch_blockers,
        render_readiness_banner,
        render_readiness_sidebar,
        render_template_data_status,
        render_template_profile,
        template_option_label,
    )
    from .ui_runtime import _render_live_run, _render_run_selector
    from .ui_theme import (
        _ensure_advanced_yaml,
        _group_label,
        _inject_style,
        _render_diff,
        _render_error,
        _render_fields,
        _render_hero,
        _render_template_summary,
        _render_validation,
    )
except ImportError:  # pragma: no cover
    from config_service import (  # type: ignore
        Catalog,
        ConfigServiceError,
        RegistryEntry,
        ValidationReport,
        apply_overrides,
        build_main_command,
        dump_yaml,
        entry_by_id,
        find_repo_root,
        format_command,
        get_nested,
        group_entries,
        inspect_config,
        load_catalog,
        load_registry,
        normalize_overrides,
        parse_override_lines,
        parse_yaml_text,
        resolve_repo_path,
    )
    from onboarding import (  # type: ignore
        OnboardingError,
        apply_safe_defaults,
        assess_template_data,
        collect_environment_readiness,
        load_template_profiles,
        profile_for,
    )
    from run_service import (  # type: ignore
        RunConflictError,
        RunRequest,
        RunServiceError,
        start_run,
    )
    from runtime_policy import inspect_execution_yaml, inspect_portable_config  # type: ignore
    from ui_onboarding import (  # type: ignore
        render_launch_blockers,
        render_readiness_banner,
        render_readiness_sidebar,
        render_template_data_status,
        render_template_profile,
        template_option_label,
    )
    from ui_runtime import _render_live_run, _render_run_selector  # type: ignore
    from ui_theme import (  # type: ignore
        _ensure_advanced_yaml,
        _group_label,
        _inject_style,
        _render_diff,
        _render_error,
        _render_fields,
        _render_hero,
        _render_template_summary,
        _render_validation,
    )

APP_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False, ttl=5)
def _cached_registry(repo_root: str) -> Tuple[RegistryEntry, ...]:
    return load_registry(Path(repo_root))


@st.cache_data(show_spinner=False, ttl=5)
def _cached_catalog(path: str) -> Catalog:
    return load_catalog(Path(path))


@st.cache_data(show_spinner=False, ttl=5)
def _cached_profiles(path: str):
    return load_template_profiles(Path(path))


@st.cache_data(show_spinner=False, ttl=5)
def _cached_inspection(
    repo_root: str,
    config_path: str,
    overrides: Tuple[Tuple[str, Any], ...],
    apply_local: bool,
) -> ValidationReport:
    if apply_local:
        return inspect_config(Path(repo_root), Path(config_path), overrides)
    return inspect_portable_config(Path(repo_root), Path(config_path), overrides)


def _signature(mode: str, source: str, overrides: Sequence[Tuple[str, Any]]) -> str:
    payload = repr((mode, source, tuple(overrides))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _local_config_fingerprint(repo_root: Path) -> str:
    """Invalidate preflight when the machine-local configuration changes."""

    path = repo_root / "configs" / "local" / "local.yaml"
    if not path.is_file():
        return "missing"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"unreadable:{type(exc).__name__}"


def _initialize_state() -> None:
    defaults = {
        "ui_mode": "Quick Start",
        "template_group": "quick_start",
        "selected_template_id": "",
        "advanced_yaml_template_id": "",
        "advanced_yaml_text": "",
        "advanced_override_text": "",
        "validation_report": None,
        "validation_signature": "",
        "active_run_id": "",
        "selected_run_id": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_smoke_reset(default_template_id: str) -> None:
    apply_safe_defaults(st.session_state, default_template_id)
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="PHM-Vibench Experiment Workspace",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialize_state()
    _inject_style()
    _render_hero()

    try:
        repo_root = find_repo_root(APP_DIR)
        catalog = _cached_catalog(str(APP_DIR / "field_catalog.yaml"))
        profiles = _cached_profiles(str(APP_DIR / "template_profiles.yaml"))
        registry = _cached_registry(str(repo_root))
    except (ConfigServiceError, OnboardingError) as exc:
        _render_error("The application contract could not be loaded.", exc)
        st.stop()

    default_profile = profile_for(profiles, catalog.default_template_id)
    readiness = collect_environment_readiness(repo_root, default_profile)
    render_readiness_banner(readiness)

    st.sidebar.markdown("### First run")
    if st.sidebar.button(
        "Use safe CPU smoke defaults",
        type="primary",
        use_container_width=True,
        help="Return to Quick Start, the bundled dummy template, CPU, and one epoch.",
    ):
        _safe_smoke_reset(catalog.default_template_id)
    render_readiness_sidebar(readiness)
    st.sidebar.divider()

    st.sidebar.markdown("### Experiment workspace")
    mode = st.sidebar.radio(
        "Experience level",
        ("Quick Start", "Advanced"),
        key="ui_mode",
        help=(
            "Quick Start exposes the smallest safe surface. Advanced adds "
            "full YAML and raw overrides."
        ),
    )
    selected_run = _render_run_selector(repo_root)

    st.header("1. 选择实验模板 | Select a validated template")
    group_keys = tuple(catalog.template_groups.keys())
    if not group_keys:
        st.error("No template groups are defined in field_catalog.yaml.")
        st.stop()
    if st.session_state.template_group not in group_keys:
        st.session_state.template_group = group_keys[0]
    group_key = st.selectbox(
        "Template group",
        group_keys,
        format_func=lambda value: _group_label(catalog, value),
        key="template_group",
    )
    available = group_entries(registry, catalog, group_key)
    if not available:
        st.info("No maintained template is registered in this category.")
        st.stop()

    available_ids = tuple(entry.id for entry in available)
    preferred_id = st.session_state.selected_template_id or catalog.default_template_id
    if preferred_id not in available_ids:
        preferred_id = available_ids[0]
    selected_id = st.selectbox(
        "Experiment template",
        available_ids,
        index=available_ids.index(preferred_id),
        format_func=lambda value: template_option_label(
            value, profile_for(profiles, value)
        ),
    )
    st.session_state.selected_template_id = selected_id
    entry = entry_by_id(available, selected_id)
    profile = profile_for(profiles, selected_id)
    _render_template_summary(entry)
    render_template_profile(profile)

    try:
        config_path = resolve_repo_path(repo_root, entry.path, yaml_only=True)
    except (ConfigServiceError, OSError) as exc:
        _render_error("The selected registry entry is not usable.", exc)
        st.stop()

    with st.spinner("Resolving template through the repository inspector..."):
        runtime_report = _cached_inspection(str(repo_root), str(config_path), (), True)
    if not runtime_report.resolved:
        _render_error(
            "The template could not be fully resolved.",
            RuntimeError(runtime_report.error or "Unknown inspector failure."),
            details=runtime_report.stderr,
        )
        st.stop()

    with st.spinner("Preparing a portable standalone YAML before local overrides..."):
        portable_report = _cached_inspection(str(repo_root), str(config_path), (), False)
    if not portable_report.resolved:
        _render_error(
            "The portable source configuration could not be resolved.",
            RuntimeError(portable_report.error or "Unknown inspector failure."),
            details=portable_report.stderr,
        )
        st.stop()

    baseline_resolved: Mapping[str, Any] = (
        portable_report.resolved if mode == "Advanced" else runtime_report.resolved
    )
    portable_yaml_text = dump_yaml(portable_report.resolved)

    st.header("2. 修改参数 | Configure safely")
    advanced_yaml_text = ""
    execution_yaml_text = portable_yaml_text
    configuration_has_error = False
    if mode == "Quick Start":
        st.info("Quick Start exposes only catalog-approved onboarding fields.")
        overrides = _render_fields(
            baseline_resolved,
            catalog,
            selected_id,
            quick_only=True,
        )
        source_for_signature = portable_yaml_text
    else:
        _ensure_advanced_yaml(selected_id, baseline_resolved)
        tabs = st.tabs(("Safe fields", "Full YAML", "Raw overrides"))
        with tabs[0]:
            safe_overrides = _render_fields(
                baseline_resolved,
                catalog,
                selected_id,
                quick_only=False,
            )
            st.caption(
                "Field aliases are declarative; no model-specific UI branches "
                "are required."
            )
        with tabs[1]:
            advanced_yaml_text = st.text_area(
                "Portable standalone YAML",
                height=520,
                key="advanced_yaml_text",
                help=(
                    "Machine-local configuration is applied exactly once during "
                    "validation and execution."
                ),
            )
            _render_diff(dump_yaml(baseline_resolved), advanced_yaml_text)
        with tabs[2]:
            override_text = st.text_area(
                "Additional overrides (one key=value per line)",
                key="advanced_override_text",
                placeholder="data.data_dir=/path/to/data\ntrainer.num_epochs=1",
            )
            st.caption(
                "Raw overrides have highest precedence and remain argv tokens, "
                "never shell text."
            )
        try:
            raw_overrides = parse_override_lines(override_text)
            overrides = normalize_overrides((*safe_overrides, *raw_overrides))
        except ConfigServiceError as exc:
            _render_error("Raw overrides are invalid.", exc)
            overrides = safe_overrides
            configuration_has_error = True
        source_for_signature = advanced_yaml_text
        execution_yaml_text = advanced_yaml_text

    st.header("3. 验证并运行 | Validate and launch")
    if mode == "Quick Start":
        command = build_main_command(repo_root, config_path, overrides)
        preview_config = apply_overrides(baseline_resolved, overrides)
    else:
        command = build_main_command(
            repo_root,
            repo_root / "outputs" / "streamlit" / "<run-id>" / "execution.yaml",
            overrides,
        )
        try:
            preview_config = apply_overrides(parse_yaml_text(advanced_yaml_text), overrides)
        except ConfigServiceError as exc:
            st.error(str(exc))
            preview_config = {}
            configuration_has_error = True

    data_status = assess_template_data(repo_root, preview_config, profile)
    render_template_data_status(data_status)
    if not data_status.ready and selected_id != catalog.default_template_id:
        if st.button(
            "Switch to the offline CPU smoke template",
            type="secondary",
            use_container_width=True,
            key="switch-to-offline-smoke",
        ):
            _safe_smoke_reset(catalog.default_template_id)

    output_dir = get_nested(preview_config, "environment.output_dir", "save")
    device = get_nested(preview_config, "trainer.device", "unspecified")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Device", str(device))
    summary_cols[1].metric("Output root", str(output_dir))
    summary_cols[2].metric("Overrides", len(overrides))
    summary_cols[3].metric("Template status", entry.status or "unspecified")
    st.markdown("**Planned reproduction command**")
    st.code(format_command(command), language="bash")
    with st.expander("Execution source preview"):
        if preview_config:
            st.code(dump_yaml(preview_config), language="yaml")
        else:
            st.warning("Fix the configuration before validation.")

    signature_source = (
        source_for_signature
        + "\n# local-config-sha256="
        + _local_config_fingerprint(repo_root)
    )
    current_signature = _signature(mode, signature_source, overrides)
    validate_col, download_col, run_col = st.columns(3)
    if validate_col.button(
        "Validate configuration",
        type="secondary",
        disabled=configuration_has_error,
        use_container_width=True,
    ):
        try:
            with st.spinner("Running repository config inspection..."):
                report = (
                    inspect_config(repo_root, config_path, overrides)
                    if mode == "Quick Start"
                    else inspect_execution_yaml(repo_root, advanced_yaml_text, overrides)
                )
            st.session_state.validation_report = report
            st.session_state.validation_signature = current_signature
        except ConfigServiceError as exc:
            st.session_state.validation_report = None
            st.session_state.validation_signature = ""
            _render_error("Validation could not start.", exc)

    report = st.session_state.validation_report
    report_is_current = (
        isinstance(report, ValidationReport)
        and st.session_state.validation_signature == current_signature
    )
    if report_is_current:
        _render_validation(report)
    elif isinstance(report, ValidationReport):
        st.warning(
            "The configuration changed after validation. Validate it again before running."
        )

    can_download = bool(
        report_is_current and report.ok and not configuration_has_error
    )
    can_run = bool(can_download and readiness.can_execute and data_status.ready)
    render_launch_blockers(readiness, data_status)

    if can_download:
        download_col.download_button(
            "Download execution YAML",
            data=execution_yaml_text,
            file_name="phm_vibench_config.yaml",
            mime="application/x-yaml",
            use_container_width=True,
        )
    else:
        download_col.button(
            "Download execution YAML",
            disabled=True,
            use_container_width=True,
        )

    if run_col.button(
        "Run experiment",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
        help=(
            "Runs the public main.py --config contract after environment, data, "
            "and configuration checks pass."
        ),
    ):
        assert isinstance(report, ValidationReport)
        resolved_output = get_nested(report.resolved, "environment.output_dir", output_dir)
        try:
            launched = start_run(
                RunRequest(
                    repo_root=repo_root,
                    template_id=selected_id,
                    mode=mode,
                    config_yaml=execution_yaml_text,
                    overrides=overrides,
                    output_root=str(resolved_output or "save"),
                    validation_signature=current_signature,
                    metadata={
                        "registry_path": entry.path,
                        "pipeline": entry.pipeline or "Pipeline_01_default",
                        "description": entry.description,
                        "difficulty": profile.difficulty,
                        "data_requirement": profile.data_label,
                        "recommended_device": profile.device_label,
                    },
                )
            )
            st.session_state.active_run_id = launched.run_id
            st.session_state.selected_run_id = launched.run_id
            st.toast("Experiment started. Live logs are available below.")
            st.rerun()
        except (RunServiceError, RunConflictError) as exc:
            _render_error("The experiment could not start.", exc)

    st.header("4. 运行与结果 | Live run and evidence")
    run_id = (
        selected_run
        or st.session_state.selected_run_id
        or st.session_state.active_run_id
    )
    if run_id:
        _render_live_run(str(repo_root), run_id)
    else:
        st.info(
            "Validate the CPU smoke template and start an experiment. This area "
            "will show live logs, headline metrics, images, artifacts, and the "
            "immutable reproduction command."
        )
