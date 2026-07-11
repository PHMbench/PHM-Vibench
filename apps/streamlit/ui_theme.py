"""Visual components and configuration widgets for the Streamlit workspace."""

from __future__ import annotations

import difflib
from typing import Any, Dict, Mapping, Tuple

import streamlit as st

try:
    from .config_service import (
        Catalog,
        FieldSpec,
        RegistryEntry,
        ValidationReport,
        build_field_overrides,
        dump_yaml,
        field_value,
    )
except ImportError:  # pragma: no cover
    from config_service import (  # type: ignore
        Catalog,
        FieldSpec,
        RegistryEntry,
        ValidationReport,
        build_field_overrides,
        dump_yaml,
        field_value,
    )


def _inject_style() -> None:
    st.markdown(
        """
<style>
.block-container {max-width: 1480px; padding-top: 1.5rem; padding-bottom: 4rem;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
.phm-hero {
  padding: 1.7rem 1.9rem; border-radius: 22px; margin-bottom: 1.2rem;
  background:
    radial-gradient(circle at 88% 18%, rgba(107,114,255,.24), transparent 30%),
    linear-gradient(135deg, rgba(20,25,45,.98), rgba(38,44,78,.94));
  color: #f7f8ff; border: 1px solid rgba(255,255,255,.10);
  box-shadow: 0 18px 55px rgba(15,20,45,.18);
}
.phm-hero .eyebrow {
  font-size:.76rem; letter-spacing:.14em; opacity:.72; font-weight:700;
}
.phm-hero h1 {
  font-size:clamp(1.75rem,3vw,3rem); line-height:1.08; margin:.45rem 0 .6rem;
}
.phm-hero p {max-width:850px; font-size:1.02rem; opacity:.82; margin:0;}
.phm-chips {display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.1rem;}
.phm-chip {
  padding:.36rem .68rem; border-radius:999px;
  border:1px solid rgba(255,255,255,.16);
  background:rgba(255,255,255,.08); font-size:.82rem;
}
.phm-steps {
  display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
  gap:.65rem; margin:1.05rem 0 1.45rem;
}
.phm-step {
  padding:.72rem .8rem; border:1px solid rgba(128,128,128,.22);
  border-radius:14px; background:rgba(128,128,128,.055); font-size:.86rem;
}
.phm-step b {
  display:inline-grid; place-items:center; width:1.55rem; height:1.55rem;
  border-radius:50%; margin-right:.42rem; background:rgba(99,102,241,.15);
}
.phm-status {
  display:inline-flex; align-items:center; gap:.42rem; padding:.34rem .62rem;
  border-radius:999px; background:rgba(128,128,128,.10);
  border:1px solid rgba(128,128,128,.22); font-weight:650;
}
.phm-muted {opacity:.67; font-size:.88rem;}
@media (max-width: 800px) {
  .phm-steps {grid-template-columns:1fr 1fr;}
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
<div class="phm-hero">
  <div class="eyebrow">PHM-VIBENCH · EXPERIMENT WORKSPACE</div>
  <h1>从可信配置，到可复现实验证据。</h1>
  <p>选择已验证模板，安全修改参数，一键运行核心 CLI，并在同一界面查看指标、图像与完整日志。</p>
  <div class="phm-chips">
    <span class="phm-chip">Config-first</span>
    <span class="phm-chip">CPU smoke ready</span>
    <span class="phm-chip">No shell execution</span>
    <span class="phm-chip">Reproducible run manifest</span>
  </div>
</div>
<div class="phm-steps">
  <div class="phm-step"><b>1</b>选择模板</div>
  <div class="phm-step"><b>2</b>调整参数</div>
  <div class="phm-step"><b>3</b>验证与运行</div>
  <div class="phm-step"><b>4</b>指标与产物</div>
</div>
""",
        unsafe_allow_html=True,
    )


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


def _group_label(catalog: Catalog, key: str) -> str:
    return str(catalog.template_groups.get(key, {}).get("label") or key)


def _entry_label(entry: RegistryEntry) -> str:
    return f"{entry.id} — {entry.description or entry.path}"


def _render_template_summary(entry: RegistryEntry) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pipeline", entry.pipeline or "Pipeline_01_default")
    col2.metric("Readiness", entry.status or "unspecified")
    col3.metric("Category", entry.category or "unspecified")
    col4.metric("Overrides", len([v for v in entry.common_overrides.split(";") if v]))
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
    if is_integer or (spec.step is not None and float(spec.step).is_integer()):
        return st.number_input(
            spec.label,
            min_value=int(spec.minimum) if spec.minimum is not None else None,
            max_value=int(spec.maximum) if spec.maximum is not None else None,
            value=int(current),
            step=int(spec.step) if spec.step is not None else 1,
            help=spec.help or None,
            key=key,
        )
    return st.number_input(
        spec.label,
        min_value=float(spec.minimum) if spec.minimum is not None else None,
        max_value=float(spec.maximum) if spec.maximum is not None else None,
        value=float(current),
        step=float(spec.step) if spec.step is not None else 0.0001,
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
        return st.selectbox(
            spec.label,
            options,
            index=options.index(current),
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


def _render_fields(
    resolved: Mapping[str, Any],
    catalog: Catalog,
    template_id: str,
    *,
    quick_only: bool,
) -> Tuple[Tuple[str, Any], ...]:
    specs = tuple(spec for spec in catalog.fields if spec.quick_start or not quick_only)
    values: Dict[str, Any] = {}
    columns = st.columns(min(4, max(1, len(specs))))
    for index, spec in enumerate(specs):
        with columns[index % len(columns)]:
            values[spec.key] = _render_field(
                spec,
                field_value(resolved, spec),
                template_id,
            )
    return build_field_overrides(
        resolved,
        catalog,
        values,
        quick_start_only=quick_only,
    )


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
        st.code(diff, language="diff") if diff else st.info("No YAML changes.")


def _render_validation(report: ValidationReport) -> None:
    st.success("Configuration validation passed.") if report.ok else st.error(
        report.error or "Configuration validation failed."
    )
    if report.sanity:
        st.dataframe(
            [
                {
                    "check": item.get("check", ""),
                    "status": "PASS" if item.get("ok") else "FAIL",
                    "message": item.get("message", ""),
                    "fix": item.get("fix", ""),
                }
                for item in report.sanity
            ],
            use_container_width=True,
            hide_index=True,
        )
    if report.resolved:
        with st.expander("Final resolved configuration"):
            st.code(dump_yaml(report.resolved), language="yaml")
    if report.stderr:
        with st.expander("Validator stderr"):
            st.code(report.stderr, language="text")
