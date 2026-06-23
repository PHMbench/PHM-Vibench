"""Compose page renderer."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from frontend.console.adapters.configuration import (
    PreflightResult,
    build_launch_request,
    build_preflight_result,
    parse_override_text,
    run_launch_request,
)
from frontend.console.pages.shared import config_selectbox
from frontend.console.state import (
    COMPOSE_INPUT_FINGERPRINT_KEY,
    COMPOSE_NOTES_KEY,
    COMPOSE_OVERRIDE_TEXT_KEY,
    COMPOSE_PREFLIGHT_ERROR_KEY,
    COMPOSE_PREFLIGHT_RESULT_KEY,
    LAST_LAUNCH_KEY,
    LAST_PREFLIGHT_KEY,
)
from frontend.console.theme import hero


def _status_icon(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _fingerprint(config_path: str, overrides: List[str], notes: str) -> Tuple[str, Tuple[str, ...], str]:
    return (config_path, tuple(overrides), notes.strip())


def _active_preflight(fingerprint: Tuple[str, Tuple[str, ...], str]) -> Optional[PreflightResult]:
    stored_fingerprint = st.session_state.get(COMPOSE_INPUT_FINGERPRINT_KEY)
    stored_result = st.session_state.get(COMPOSE_PREFLIGHT_RESULT_KEY)
    if stored_fingerprint == fingerprint and isinstance(stored_result, PreflightResult):
        return stored_result
    return None


def _preflight_status(
    fingerprint: Tuple[str, Tuple[str, ...], str],
) -> tuple[str, str]:
    """Return the current preflight status key and user-facing message."""
    stored_fingerprint = st.session_state.get(COMPOSE_INPUT_FINGERPRINT_KEY)
    stored_error = st.session_state.get(COMPOSE_PREFLIGHT_ERROR_KEY, "")
    if _active_preflight(fingerprint) is not None:
        return "success", "Resolved for the current inputs. Launch is available."
    if stored_fingerprint == fingerprint and stored_error:
        return "error", "Resolve failed for the current inputs. Fix the issue and run preflight again."
    if stored_fingerprint is not None and stored_fingerprint != fingerprint:
        return "stale", "Inputs changed. Run Resolve YAML / Preflight again before launch."
    return "idle", "Resolve YAML / Preflight to inspect the merged config before launch."


def _preflight_error_help(error_text: str) -> str:
    """Return an actionable preflight error hint."""
    lowered = error_text.lower()
    if "override" in lowered:
        return "Check each override line for `key=value` format and confirm the key exists in the config tree."
    if "not found" in lowered or "no such file" in lowered:
        return "Check the config path, referenced base config files, and any local data paths used by the config."
    if "import" in lowered or "module" in lowered:
        return "Check that the selected pipeline, task, model, and trainer targets are importable in the current environment."
    return "Check the config path, override names, and importable components, then run preflight again."


def _set_preflight_state(
    fingerprint: Tuple[str, Tuple[str, ...], str],
    result: Optional[PreflightResult],
    error: str = "",
) -> None:
    st.session_state[COMPOSE_INPUT_FINGERPRINT_KEY] = fingerprint
    st.session_state[COMPOSE_PREFLIGHT_RESULT_KEY] = result
    st.session_state[COMPOSE_PREFLIGHT_ERROR_KEY] = error
    if result is not None:
        st.session_state[LAST_PREFLIGHT_KEY] = {
            "config_path": result.config_path,
            "shell_command": result.shell_command,
            "output_preview": result.output_preview,
        }


def render_compose() -> None:
    """Render config compose, explicit preflight, and launch."""
    hero(
        "Compose Experiment",
        "compose / inspect / launch",
        "Build a run from maintained configs, inspect the resolved YAML, then launch through the exact CLI command.",
    )

    selected_config = config_selectbox("Config", key="compose_config_select")
    notes = st.text_input(
        "Notes",
        value=st.session_state.get(COMPOSE_NOTES_KEY, ""),
        help="Passed through to main.py --notes.",
        key=COMPOSE_NOTES_KEY,
    )
    override_text = st.text_area(
        "Overrides",
        value=st.session_state.get(COMPOSE_OVERRIDE_TEXT_KEY, "trainer.num_epochs=1\ndata.num_workers=0"),
        height=140,
        help="One key=value override per line.",
        key=COMPOSE_OVERRIDE_TEXT_KEY,
    )
    overrides = parse_override_text(override_text)
    request = build_launch_request(selected_config, overrides, notes=notes)
    fingerprint = _fingerprint(selected_config, overrides, notes)
    preflight = _active_preflight(fingerprint)
    preflight_error = st.session_state.get(COMPOSE_PREFLIGHT_ERROR_KEY, "")
    preflight_state, preflight_message = _preflight_status(fingerprint)

    st.subheader("Command Preview")
    st.code(request.shell_command, language="bash")
    if preflight_state == "success" and preflight is not None:
        st.caption(f"Predicted output pattern: {preflight.output_preview}")
        st.success(preflight_message)
    elif preflight_state == "stale":
        st.warning(preflight_message)
    elif preflight_state == "error":
        st.error(preflight_message)
    else:
        st.info(preflight_message)
    st.caption("Launch stays locked until the current inputs complete a successful preflight.")

    resolve_col, launch_col = st.columns([0.36, 0.24])
    with resolve_col:
        resolve_clicked = st.button("Resolve YAML / Preflight", use_container_width=True)
    with launch_col:
        launch_clicked = st.button(
            "Launch via CLI",
            type="primary",
            use_container_width=True,
            disabled=preflight is None,
        )

    if resolve_clicked:
        try:
            preflight = build_preflight_result(selected_config, overrides, notes=notes)
            _set_preflight_state(fingerprint, preflight)
            preflight_error = ""
            st.rerun()
        except Exception as exc:
            preflight = None
            preflight_error = str(exc)
            _set_preflight_state(fingerprint, None, error=preflight_error)

    if preflight_error and preflight is None:
        st.error(_preflight_error_help(preflight_error))
        with st.expander("Preflight error details"):
            st.code(preflight_error, language="text")

    if launch_clicked and preflight is not None:
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
            st.info("Run Resolve YAML / Preflight to inspect sanity checks.")
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
            st.info("Field sources and instantiation targets appear after a successful preflight.")
        else:
            st.dataframe(pd.DataFrame(preflight.sources), use_container_width=True, hide_index=True)
            st.subheader("Instantiation Targets")
            st.json(preflight.targets, expanded=False)
