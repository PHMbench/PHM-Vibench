"""Experimental, configuration-first Streamlit UI for PHM-Vibench.

Launch from the repository root:

    streamlit run apps/streamlit/app.py

The application never imports or calls a Pipeline function directly. Validation
is delegated to ``python -m scripts.config_inspect`` and experiment execution is
added by the run-service layer in the next stacked change.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import streamlit as st

try:  # Supports both ``streamlit run`` and package imports in tests/tools.
    from .config_service import (
        Catalog,
        ConfigServiceError,
        FieldSpec,
        RegistryEntry,
        ValidationReport,
        apply_overrides,
        build_field_overrides,
        build_main_command,
        dump_yaml,
        entry_by_id,
        field_value,
        find_repo_root,
        format_command,
        get_nested,
        group_entries,
        inspect_config,
        inspect_yaml_text,
        load_catalog,
        load_registry,
        normalize_overrides,
        parse_override_lines,
        parse_yaml_text,
        resolve_repo_path,
    )
except ImportError:  # pragma: no cover - Streamlit executes this file as a script.
    from config_service import (  # type: ignore
        Catalog,
        ConfigServiceError,
        FieldSpec,
        RegistryEntry,
        ValidationReport,
        apply_overrides,
        build_field_overrides,
        build_main_command,
        dump_yaml,
        entry_by_id,
        field_value,
        find_repo_root,
        format_command,
        get_nested,
        group_entries,
        inspect_config,
        inspect_yaml_text,
        load_catalog,
        load_registry,
        normalize_overrides,
        parse_override_lines,
        parse_yaml_text,
        resolve_repo_path,
    )


APP_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def _cached_registry(repo_root: str) -> Tuple[RegistryEntry, ...]:
    return load_registry(Path(repo_root))


@st.cache_data(show_spinner=False)
def _cached_catalog(path: str) -> Catalog:
    return load_catalog(Path(path))


@st.cache_data(show_spinner=False)
def _cached_inspection(
    repo_root: str,
    config_path: str,
    overrides: Tuple[Tuple[str, Any], ...],
) -> ValidationReport:
    return inspect_config(Path(repo_root), Path(config_path), overrides)


def _signature(
    mode: str,
    source: str,
    overrides: Sequence[Tuple[str, Any]],
) -> str:
    payload = repr((mode, source, tuple(overrides))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_error(title: str, error: BaseException, *, details: str = "") -> None:
    st.error(f"**{title}**\n\n{error}")
    with st.expander("排查建议 | Troubleshooting", expanded=True):
        st.markdown(
            "1. 从仓库根目录启动应用。\n"
            "2. 确认核心依赖和可选前端依赖均已安装。\n"
            "3. 使用 CPU smoke 模板排除数据集和 GPU 环境问题。\n"
            "4. 复制页面中的 CLI 命令在终端复现。"
        )
        if details:
            st.code(details, language="text")


def _group_options(catalog: Catalog) -> Tuple[str, ...]:
    return tuple(catalog.template_groups.keys())


def _group_label(catalog: Catalog, key: str) -> str:
    group = catalog.template_groups.get(key, {})
    return str(group.get("label") or key)


def _entry_label(entry: RegistryEntry) -> str:
    return f"{entry.id} — {entry.description or entry.path}"


def _render_template_summary(entry: RegistryEntry) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Pipeline", entry.pipeline or "Pipeline_01_default")
    col2.metric("Registry status", entry.status or "unspecified")
    col3.metric("Category", entry.category or "unspecified")
    st.caption(entry.description)
    with st.expander("模板契约 | Template contract"):
        st.code(entry.path, language="text")
        if entry.minimal_run:
            st.markdown("**Registry minimal run**")
            st.code(entry.minimal_run, language="bash")
        if entry.outputs:
            st.markdown("**Expected output pattern**")
            st.code(entry.outputs, language="text")
        if entry.related_docs:
            st.markdown("**Related repository documents**")
            st.code(entry.related_docs.replace(";", "\n"), language="text")


def _number_widget(spec: FieldSpec, current: Any, key: str) -> Any:
    if current is None:
        current = spec.default
    if current is None:
        current = 0
    is_integer = isinstance(current, int) and not isinstance(current, bool)
    minimum = spec.minimum
    maximum = spec.maximum
    step = spec.step

    if is_integer or (step is not None and float(step).is_integer()):
        return st.number_input(
            spec.label,
            min_value=int(minimum) if minimum is not None else None,
            max_value=int(maximum) if maximum is not None else None,
            value=int(current),
            step=int(step) if step is not None else 1,
            help=spec.help or None,
            key=key,
        )
    return st.number_input(
        spec.label,
        min_value=float(minimum) if minimum is not None else None,
        max_value=float(maximum) if maximum is not None else None,
        value=float(current),
        step=float(step) if step is not None else 0.0001,
        format="%.8f",
        help=spec.help or None,
        key=key,
    )


def _render_field(spec: FieldSpec, current: Any, template_id: str) -> Any:
    widget_key = f"field::{template_id}::{spec.key}"
    if spec.widget == "select":
        options = list(spec.options)
        if current not in options:
            options.insert(0, current)
        index = options.index(current)
        return st.selectbox(
            spec.label,
            options,
            index=index,
            help=spec.help or None,
            key=widget_key,
        )
    if spec.widget == "number":
        return _number_widget(spec, current, widget_key)
    if spec.widget == "checkbox":
        return st.checkbox(
            spec.label,
            value=bool(current),
            help=spec.help or None,
            key=widget_key,
        )
    return st.text_input(
        spec.label,
        value="" if current is None else str(current),
        help=spec.help or None,
        key=widget_key,
    )


def _render_quick_fields(
    resolved: Mapping[str, Any],
    catalog: Catalog,
    template_id: str,
) -> Tuple[Tuple[str, Any], ...]:
    values: Dict[str, Any] = {}
    quick_specs = tuple(spec for spec in catalog.fields if spec.quick_start)
    columns = st.columns(max(1, len(quick_specs)))
    for column, spec in zip(columns, quick_specs):
        with column:
            current = field_value(resolved, spec)
            values[spec.key] = _render_field(spec, current, template_id)
    return build_field_overrides(resolved, catalog, values, quick_start_only=True)


def _render_safe_fields(
    resolved: Mapping[str, Any],
    catalog: Catalog,
    template_id: str,
) -> Tuple[Tuple[str, Any], ...]:
    values: Dict[str, Any] = {}
    for index, spec in enumerate(catalog.fields):
        if index % 2 == 0:
            left, right = st.columns(2)
        column = left if index % 2 == 0 else right
        with column:
            current = field_value(resolved, spec)
            values[spec.key] = _render_field(spec, current, template_id)
    return build_field_overrides(resolved, catalog, values)


def _ensure_advanced_yaml(template_id: str, resolved: Mapping[str, Any]) -> None:
    if st.session_state.advanced_yaml_template_id != template_id:
        st.session_state.advanced_yaml_template_id = template_id
        st.session_state.advanced_yaml_text = dump_yaml(resolved)
        st.session_state.advanced_override_text = ""
        st.session_state.validation_report = None
        st.session_state.validation_signature = ""


def _render_diff(before: str, after: str) -> None:
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="registry-resolved.yaml",
            tofile="edited-config.yaml",
        )
    )
    with st.expander("配置差异 | Configuration diff"):
        if diff:
            st.code(diff, language="diff")
        else:
            st.info("No YAML changes relative to the resolved registry template.")


def _render_validation(report: ValidationReport) -> None:
    if report.ok:
        st.success("Configuration validation passed.")
    else:
        st.error(report.error or "Configuration validation failed.")

    if report.sanity:
        rows = [
            {
                "check": item.get("check", ""),
                "status": "PASS" if item.get("ok") else "FAIL",
                "message": item.get("message", ""),
                "fix": item.get("fix", ""),
            }
            for item in report.sanity
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    if report.stderr:
        with st.expander("Validator stderr"):
            st.code(report.stderr, language="text")


def _download_final_config(report: ValidationReport, *, key: str) -> None:
    if not report.resolved:
        return
    st.download_button(
        "Download validated YAML",
        data=dump_yaml(report.resolved),
        file_name="phm_vibench_config.yaml",
        mime="application/x-yaml",
        key=key,
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="PHM-Vibench Experiment Console",
        page_icon="🧪",
        layout="wide",
    )
    _initialize_state()

    st.title("PHM-Vibench Experiment Console")
    st.caption(
        "Experimental optional UI: template selection → safe overrides → "
        "CLI validation → results. The core command remains "
        "`python main.py --config ...`."
    )

    try:
        repo_root = find_repo_root(APP_DIR)
        catalog = _cached_catalog(str(APP_DIR / "field_catalog.yaml"))
        registry = _cached_registry(str(repo_root))
    except ConfigServiceError as exc:
        _render_error("The application contract could not be loaded.", exc)
        st.stop()

    st.sidebar.header("Workflow")
    mode = st.sidebar.radio(
        "Mode",
        ("Quick Start", "Advanced"),
        key="ui_mode",
        help=(
            "Quick Start exposes only device and epoch count. Advanced allows "
            "safe fields, full YAML, and raw overrides."
        ),
    )

    st.header("1. 选择实验模板 | Select a validated template")
    group_keys = _group_options(catalog)
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
        st.info(
            "The current registry has no maintained template in this category. "
            "Choose another category or add a sanity_ok registry entry."
        )
        st.stop()

    available_ids = tuple(entry.id for entry in available)
    preferred_id = st.session_state.selected_template_id or catalog.default_template_id
    if preferred_id not in available_ids:
        preferred_id = available_ids[0]
    selected_id = st.selectbox(
        "Experiment template",
        available_ids,
        index=available_ids.index(preferred_id),
        format_func=lambda value: _entry_label(entry_by_id(available, value)),
    )
    st.session_state.selected_template_id = selected_id
    entry = entry_by_id(available, selected_id)
    _render_template_summary(entry)

    try:
        config_path = resolve_repo_path(repo_root, entry.path, yaml_only=True)
    except ConfigServiceError as exc:
        _render_error("The selected registry entry is not usable.", exc)
        st.stop()

    with st.spinner("Resolving the template through scripts.config_inspect..."):
        base_report = _cached_inspection(str(repo_root), str(config_path), ())
    if not base_report.resolved:
        _render_error(
            "The template could not be fully resolved.",
            RuntimeError(base_report.error or "Unknown inspector failure."),
            details=base_report.stderr,
        )
        st.stop()
    baseline_resolved: Mapping[str, Any] = base_report.resolved

    st.header("2. 修改参数 | Configure")
    advanced_yaml_text = ""
    overrides: Tuple[Tuple[str, Any], ...]
    source_for_signature: str

    if mode == "Quick Start":
        st.info("Quick Start changes only the device and number of epochs.")
        overrides = _render_quick_fields(baseline_resolved, catalog, selected_id)
        source_for_signature = str(config_path)
    else:
        _ensure_advanced_yaml(selected_id, baseline_resolved)
        tabs = st.tabs(("Safe fields", "Full YAML", "Raw overrides"))
        with tabs[0]:
            safe_overrides = _render_safe_fields(
                baseline_resolved,
                catalog,
                selected_id,
            )
            st.caption(
                "Safe fields become CLI overrides. Alias resolution is "
                "catalog-driven, so maintained and legacy key locations can "
                "coexist without model-specific UI branches."
            )
        with tabs[1]:
            advanced_yaml_text = st.text_area(
                "Standalone resolved YAML",
                height=520,
                key="advanced_yaml_text",
                help=(
                    "This is a standalone five-block config. Safe fields and "
                    "raw overrides are applied after this YAML."
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
                "Raw overrides have the highest precedence. They are passed as "
                "argv tokens and are never concatenated into a shell command."
            )
        try:
            raw_overrides = parse_override_lines(override_text)
            overrides = normalize_overrides((*safe_overrides, *raw_overrides))
        except ConfigServiceError as exc:
            _render_error("Raw overrides are invalid.", exc)
            overrides = safe_overrides
        source_for_signature = advanced_yaml_text

    st.header("3. 运行前检查 | Preflight")
    if mode == "Quick Start":
        command = build_main_command(repo_root, config_path, overrides)
        preview_config = apply_overrides(baseline_resolved, overrides)
    else:
        command = build_main_command(
            repo_root,
            repo_root / "phm_vibench_config.yaml",
            overrides,
        )
        try:
            preview_config = apply_overrides(
                parse_yaml_text(advanced_yaml_text),
                overrides,
            )
        except ConfigServiceError:
            preview_config = {}

    output_dir = get_nested(preview_config, "environment.output_dir", "save")
    device = get_nested(preview_config, "trainer.device", "unspecified")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Expected device", str(device))
    summary_cols[1].metric("Output root", str(output_dir))
    summary_cols[2].metric("Override count", len(overrides))

    st.markdown("**Reproduction command**")
    st.code(format_command(command), language="bash")
    with st.expander("Final configuration preview"):
        if preview_config:
            st.code(dump_yaml(preview_config), language="yaml")
        else:
            st.warning("Fix the edited YAML before validation.")

    current_signature = _signature(mode, source_for_signature, overrides)
    validate_col, download_col, run_col = st.columns(3)
    validate_clicked = validate_col.button(
        "Validate",
        type="primary",
        use_container_width=True,
    )
    if validate_clicked:
        try:
            with st.spinner("Running repository config inspection..."):
                if mode == "Quick Start":
                    report = inspect_config(repo_root, config_path, overrides)
                else:
                    report = inspect_yaml_text(repo_root, advanced_yaml_text, overrides)
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
        with download_col:
            _download_final_config(report, key="download_validated_config")
    else:
        download_col.button(
            "Download validated YAML",
            disabled=True,
            use_container_width=True,
            help="Validate the current configuration first.",
        )
        if isinstance(report, ValidationReport):
            st.warning(
                "The configuration changed after validation. Validate again "
                "before running."
            )

    run_col.button(
        "Run Experiment",
        disabled=True,
        use_container_width=True,
        help=(
            "Execution, cancellation, live logs, and result discovery are "
            "introduced in the stacked PR-S2."
        ),
    )

    st.header("4. 查看结果 | Results")
    st.info(
        "PR-S1 intentionally stops at a validated, downloadable, reproducible "
        "configuration. The stacked PR-S2 adds subprocess execution and result "
        "discovery without changing main.py."
    )


if __name__ == "__main__":
    main()
