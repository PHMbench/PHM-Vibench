"""Main entry point for the Streamlit application."""
from __future__ import annotations

import streamlit as st

from . import state
from . import layout
from .pipeline import display_output


CONFIG_PATH = "configs/demo/ID/id_demo.yaml"


def display_signal_if_available() -> None:
    """Draw the raw signal of the selected sample, if possible."""
    selected_id = st.session_state.get("test_id")
    if selected_id and st.session_state.data_dir:
        signal = layout.load_signal(st.session_state.data_dir, selected_id)
        if signal is not None:
            layout.plot_signal(signal)


def load_data_step() -> None:
    """Handle metadata upload and preview."""
    st.header("\u6b65\u9aa4 1: \u52a0\u8f7d\u6570\u636e")
    upload = st.file_uploader("\u9009\u62e9\u5143\u6570\u636e\u6587\u4ef6", type=["csv", "xlsx"])
    if upload is not None:
        df = layout.load_metadata(upload)
        if df is not None:
            ids = layout.available_ids(df)
            st.session_state["available_ids"] = ids
            st.success("\u8bfb\u53d6\u6210\u529f")
            layout.preview_metadata()


def parameter_step() -> None:
    """Render parameter input sections."""
    st.header("\u6b65\u9aa4 2: \u914d\u7f6e\u53c2\u6570")
    ids = st.session_state.get("available_ids", [])
    cols = st.columns(4)
    with cols[0]:
        layout.data_section(ids)
    with cols[1]:
        layout.model_section()
    with cols[2]:
        layout.task_section()
    with cols[3]:
        layout.trainer_section()


def run_step() -> None:
    """Start the experiment and show output."""
    st.header("\u6b65\u9aa4 3: \u8fd0\u884c\u5e76\u67e5\u770b\u7ed3\u679c")
    layout.run_controls(CONFIG_PATH)
    display_output()


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="PHM-Vibench", layout="wide")
    state.init_state()

    st.title("PHM-Vibench")
    load_data_step()
    display_signal_if_available()
    parameter_step()
    run_step()


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
