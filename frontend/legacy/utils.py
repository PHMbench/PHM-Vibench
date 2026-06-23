"""Utility functions for the Streamlit GUI."""

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
    """Initialize default values in ``st.session_state``.

    The GUI relies heavily on session state to persist user choices and the
    running process between reruns triggered by widget interaction.
    """
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
    """Load metadata from ``file`` with basic error handling."""
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

    The function expects each sample to be stored as ``<id>.h5`` with a
    ``signal`` dataset. Errors are reported via ``st.error`` and ``None`` is
    returned on failure.
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
        st.experimental_rerun()  # force UI update
    process.wait()
    st.session_state.process = None
    st.session_state.paused = False


def start_pipeline(config_path: str) -> None:
    """Launch the training pipeline as a subprocess."""
    if st.session_state.process:
        st.warning("\u5df2\u7ecf\u6709\u6b63\u5728\u8fd0\u884c\u7684\u4efb\u52a1")
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

