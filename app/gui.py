"""Streamlit interface for PHM-Vibench."""

import os
import streamlit as st

from . import utils


def step_load_data() -> None:
    """Step 1: load metadata and preview it."""
    st.header("\u6b65\u9aa4 1: \u52a0\u8f7d\u6570\u636e")
    upload = st.file_uploader("\u9009\u62e9\u5143\u6570\u636e\u6587\u4ef6", type=["csv", "xlsx"])
    if upload is not None:
        df = utils.load_metadata(upload)
        if df is not None:
            ids = utils.available_ids(df)
            st.session_state["available_ids"] = ids
            st.success("\u8bfb\u53d6\u6210\u529f")
            utils.preview_metadata()

    selected_id = st.session_state.get("test_id")
    if selected_id and st.session_state.data_dir:
        signal = utils.load_signal(st.session_state.data_dir, selected_id)
        if signal is not None:
            utils.plot_signal(signal)


def _data_params(col) -> None:
    """Render dataset related parameters."""
    st.session_state.data_dir = col.text_input(
        "HDF5 Data Directory",
        value=st.session_state.data_dir,
        key="data_dir_input",
    )
    if st.session_state.data_dir:
        exists = os.path.exists(st.session_state.data_dir)
        if exists:
            col.success("\u8def\u5f84\u6b63\u786e")
        else:
            col.error("\u8def\u5f84\u65e0\u6548")
    ids = st.session_state.get("available_ids", [])
    st.session_state.train_ids = col.multiselect(
        "Train IDs", ids, default=st.session_state.train_ids
    )
    st.session_state.val_ids = col.multiselect(
        "Val IDs", ids, default=st.session_state.val_ids
    )
    st.session_state.test_id = (
        col.selectbox(
            "Test ID",
            ids,
            index=ids.index(st.session_state.test_id)
            if st.session_state.test_id in ids
            else 0,
        )
        if ids
        else None
    )


def _model_params(col) -> None:
    """Render model parameter inputs."""
    st.session_state.learning_rate = col.number_input(
        "Learning Rate",
        min_value=1e-6,
        max_value=1.0,
        value=float(st.session_state.learning_rate),
        step=1e-4,
        format="%f",
        help="\u8bad\u7ec3\u7684\u5b66\u4e60\u7387",
    )


def _task_params(col) -> None:
    """Render task parameter inputs."""
    st.session_state.task_type = col.text_input(
        "Task Type",
        value=st.session_state.task_type,
        help="\u4efb\u52a1\u7c7b\u578b\uff0c\u5982\u5206\u7c7b",
    )


def _trainer_params(col) -> None:
    """Render trainer parameter placeholder."""
    col.write("...")


def step_configure() -> None:
    """Step 2: parameter configuration."""
    st.header("\u6b65\u9aa4 2: \u914d\u7f6e\u53c2\u6570")
    cols = st.columns(4)
    with cols[0].expander("Data", expanded=False):
        _data_params(cols[0])
    with cols[1].expander("Model", expanded=False):
        _model_params(cols[1])
    with cols[2].expander("Task", expanded=False):
        _task_params(cols[2])
    with cols[3].expander("Trainer", expanded=False):
        _trainer_params(cols[3])


def step_run() -> None:
    """Step 3: run or pause the experiment."""
    st.header("\u6b65\u9aa4 3: \u8fd0\u884c\u5e76\u67e5\u770b\u7ed3\u679c")
    run_col, pause_col = st.columns(2)
    if run_col.button("Start Experiment"):
        if st.session_state.test_id is None:
            st.warning("\u8bf7\u9009\u62e9 Test ID")
        elif not os.path.exists(st.session_state.data_dir):
            st.warning("\u8bf7\u6307\u5b9a\u6709\u6548\u7684 HDF5 \u76ee\u5f55")
        else:
            utils.start_pipeline("configs/demo/ID/id_demo.yaml")
    if pause_col.button("Pause/Resume") and st.session_state.process:
        utils.toggle_pause()

    utils.display_output()


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="PHM-Vibench", layout="wide")
    utils.init_session_state()
    st.title("PHM-Vibench")
    step_load_data()
    step_configure()
    step_run()


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
