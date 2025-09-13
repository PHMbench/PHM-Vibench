"""Streamlit interface for PHM-Vibench.

This app guides users through loading metadata, configuring parameters, and running pipeline experiments with real-time output.
"""

import os
import signal
import subprocess
import threading
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Session State Utilities
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize default values in ``st.session_state``."""
    defaults = {
        "metadata_df": None,
        "train_ids": [],
        "val_ids": [],
        "test_id": None,
        "data_dir": "",
        "learning_rate": 0.001,
        "task_type": "classification",
        "process": None,
        "output_lines": [],
        "paused": False,
        "experiment_run": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

def load_metadata(file) -> Optional[pd.DataFrame]:
    """Load metadata from ``file`` with error handling."""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as exc:  # pragma: no cover - UI feedback
        st.error(f"\u4e0a\u4f20\u5931\u8d25: {exc}")
        return None
    st.session_state.metadata_df = df
    return df


def preview_metadata() -> None:
    """Display the first few rows of the loaded metadata."""
    df = st.session_state.metadata_df
    if df is not None:
        with st.expander("\u5148\u770b\u5143\u6570\u636e", expanded=False):
            st.dataframe(df.head())


def available_ids(df: pd.DataFrame) -> List[str]:
    """Return a list of sample IDs from ``df`` if present."""
    for col in ("id", "ID", "sample_id"):
        if col in df.columns:
            return df[col].astype(str).unique().tolist()
    return []


def load_signal(data_dir: str, sample_id: str) -> Optional[List[float]]:
    """Load raw signal for ``sample_id`` from ``data_dir``.

    This expects a file named ``<sample_id>.h5`` containing a ``signal`` dataset.
    """
    path = os.path.join(data_dir, f"{sample_id}.h5")
    try:
        with h5py.File(path, "r") as hf:
            signal = hf["signal"][:]
    except Exception as exc:  # pragma: no cover - UI feedback
        st.error(f"\u65ad\u7247\u52a0\u8f7d\u9519\u8bef: {exc}")
        return None
    return signal


def plot_signal(signal: List[float]) -> None:
    """Plot ``signal`` on the main page."""
    fig, ax = plt.subplots()
    ax.plot(signal)
    ax.set_title("\u539f\u59cb\u4fe1\u53f7")
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Subprocess Management
# ---------------------------------------------------------------------------

def _reader_thread(process: subprocess.Popen) -> None:
    """Collect output from ``process`` into the session state."""
    for line in iter(process.stdout.readline, b""):
        st.session_state.output_lines.append(line.decode())
        # Force a UI update
        st.experimental_rerun()
    process.wait()
    st.session_state.process = None
    st.session_state.paused = False


def start_pipeline(config_path: str) -> None:
    """Launch the training pipeline as a subprocess."""
    if st.session_state.process:
        st.warning("\u5df2\u7ecf\u5b58\u5728\u6b63\u5728\u8fd0\u884c\u7684\u5b50\u8fdb\u7a0b")
        return
    cmd = ["python", "main.py", "--config_path", config_path]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    st.session_state.process = process
    st.session_state.output_lines = []
    st.session_state.experiment_run = True
    thread = threading.Thread(target=_reader_thread, args=(process,), daemon=True)
    thread.start()


def toggle_pause() -> None:
    """Pause or resume the running subprocess."""
    process = st.session_state.process
    if not process:
        return
    if st.session_state.paused:
        os.kill(process.pid, signal.SIGCONT)
        st.session_state.paused = False
    else:
        os.kill(process.pid, signal.SIGSTOP)
        st.session_state.paused = True


def display_output() -> None:
    """Stream captured output to the UI."""
    if st.session_state.output_lines:
        st.text_area(
            "\u8fd0\u884c\u8f93\u51fa",
            value="".join(st.session_state.output_lines),
            height=200,
        )

# ---------------------------------------------------------------------------
# UI Section Helpers
# ---------------------------------------------------------------------------

def step_load_metadata() -> None:
    """Handle file upload and metadata preview."""
    st.header("\u6b65\u9aa4 1: \u52a0\u8f7d\u6570\u636e")
    upload = st.file_uploader("\u9009\u62e9\u5143\u6570\u636e\u6587\u4ef6", type=["csv", "xlsx"])
    if upload is not None:
        df = load_metadata(upload)
        if df is not None:
            st.session_state["available_ids"] = available_ids(df)
            st.success("\u8bfb\u53d6\u6210\u529f")
            preview_metadata()


def preview_selected_signal() -> None:
    """Visualize the raw signal for the selected test sample."""
    selected_id = st.session_state.get("test_id")
    if selected_id and st.session_state.data_dir:
        signal = load_signal(st.session_state.data_dir, selected_id)
        if signal is not None:
            plot_signal(signal)


def _data_params(col) -> None:
    """Input fields for data-related parameters."""
    with col.expander("Data", expanded=False):
        st.session_state.data_dir = st.text_input(
            "HDF5 Data Directory",
            value=st.session_state.data_dir,
            key="data_dir_input",
        )
        if st.session_state.data_dir:
            exists = os.path.exists(st.session_state.data_dir)
            if exists:
                st.success("\u8def\u5f84\u6b63\u786e")
            else:
                st.error("\u8def\u5f84\u65e0\u6548")
        ids = st.session_state.get("available_ids", [])
        st.session_state.train_ids = st.multiselect(
            "Train IDs",
            ids,
            default=st.session_state.train_ids,
        )
        st.session_state.val_ids = st.multiselect(
            "Val IDs",
            ids,
            default=st.session_state.val_ids,
        )
        st.session_state.test_id = st.selectbox(
            "Test ID",
            ids,
            index=ids.index(st.session_state.test_id) if st.session_state.test_id in ids else 0,
        ) if ids else None
        if st.button("Update Data", key="update_data"):
            st.success("Data parameters updated")


def _model_params(col) -> None:
    """Input fields for model configuration."""
    with col.expander("Model", expanded=False):
        st.session_state.learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1.0,
            value=float(st.session_state.learning_rate),
            step=1e-4,
            format="%f",
            help="\u8bad\u7ec3\u7684\u5b66\u4e60\u7387",
        )
        if st.button("Update Model", key="update_model"):
            st.success("Model parameters updated")


def _task_params(col) -> None:
    """Input fields for task configuration."""
    with col.expander("Task", expanded=False):
        st.session_state.task_type = st.text_input(
            "Task Type",
            value=st.session_state.task_type,
            help="\u4efb\u52a1\u7c7b\u578b\uff0c\u5982\u5206\u7c7b",
        )
        if st.button("Update Task", key="update_task"):
            st.success("Task parameters updated")


def _trainer_params(col) -> None:
    """Input fields for trainer configuration."""
    with col.expander("Trainer", expanded=False):
        st.write("...")  # Placeholder for trainer parameters
        if st.button("Update Trainer", key="update_trainer"):
            st.success("Trainer parameters updated")


def step_configure_parameters() -> None:
    """Display parameter sections for data, model, task and trainer."""
    st.header("\u6b65\u9aa4 2: \u914d\u7f6e\u53c2\u6570")
    cols = st.columns(4)
    _data_params(cols[0])
    _model_params(cols[1])
    _task_params(cols[2])
    _trainer_params(cols[3])
    preview_selected_signal()


def step_run_experiment() -> None:
    """Controls to run, pause and show experiment output."""
    st.header("\u6b65\u9aa4 3: \u8fd0\u884c\u5e76\u67e5\u770b\u7ed3\u679c")
    run_col, pause_col = st.columns(2)
    if run_col.button("Start Experiment"):
        if st.session_state.test_id is None:
            st.warning("\u8bf7\u9009\u62e9 Test ID")
        elif not os.path.exists(st.session_state.data_dir):
            st.warning("\u8bf7\u6307\u5b9a\u6709\u6548\u7684 HDF5 \u76ee\u5f55")
        else:
            start_pipeline("configs/demo/ID/id_demo.yaml")
    if pause_col.button("Pause/Resume") and st.session_state.process:
        toggle_pause()

    display_output()


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="PHM-Vibench", layout="wide")
    init_session_state()

    st.title("PHM-Vibench")
    step_load_metadata()
    step_configure_parameters()
    step_run_experiment()


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
