"""Streamlit UI layout helpers."""
from __future__ import annotations

import os
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def _show_update_message(section: str) -> None:
    """Display a brief toast indicating that *section* parameters were saved."""
    st.success(f"{section} parameters updated", icon="✅")


def _validate_path(path: str) -> None:
    """Show visual feedback for ``path`` existence."""
    if os.path.exists(path):
        st.success("\u8def\u5f84\u6b63\u786e")
    else:
        st.error("\u8def\u5f84\u65e0\u6548")


def _id_selectors(ids: List[str]) -> None:
    """Update train/val/test ID selections in session state."""
    st.session_state.train_ids = st.multiselect(
        "Train IDs", ids, default=st.session_state.train_ids
    )
    st.session_state.val_ids = st.multiselect(
        "Val IDs", ids, default=st.session_state.val_ids
    )
    st.session_state.test_id = (
        st.selectbox(
            "Test ID",
            ids,
            index=ids.index(st.session_state.test_id)
            if st.session_state.test_id in ids
            else 0,
        )
        if ids
        else None
    )


# ---------------------------- Data utilities ----------------------------

def load_metadata(file) -> Optional[pd.DataFrame]:
    """Load metadata file and store result in session state."""
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
    """Show the first few rows of the loaded metadata."""
    df = st.session_state.metadata_df
    if df is not None:
        with st.expander("\u5148\u770b\u5143\u6570\u636e", expanded=False):
            st.dataframe(df.head())


def available_ids(df: pd.DataFrame) -> List[str]:
    """Return a list of ID strings from the metadata DataFrame."""
    for col in ("id", "ID", "sample_id"):
        if col in df.columns:
            return df[col].astype(str).unique().tolist()
    return []


def load_signal(data_dir: str, sample_id: str) -> Optional[List[float]]:
    """Load raw signal for ``sample_id`` stored under ``data_dir``."""
    path = os.path.join(data_dir, f"{sample_id}.h5")
    try:
        with h5py.File(path, "r") as hf:
            signal = hf["signal"][:]
    except Exception as exc:  # pragma: no cover - UI feedback
        st.error(f"\u65ad\u7247\u52a0\u8f7d\u9519\u8bef: {exc}")
        return None
    return signal


def plot_signal(signal: List[float]) -> None:
    """Plot a waveform in the main area."""
    fig, ax = plt.subplots()
    ax.plot(signal)
    ax.set_title("\u539f\u59cb\u4fe1\u53f7")
    st.pyplot(fig)


# ---------------------------- Layout sections ---------------------------

def data_section(ids: List[str]) -> None:
    """Render inputs related to dataset selection."""
    with st.expander("Data", expanded=False):
        st.session_state.data_dir = st.text_input(
            "HDF5 Data Directory",
            value=st.session_state.data_dir,
            key="data_dir_input",
        )
        if st.session_state.data_dir:
            _validate_path(st.session_state.data_dir)
        _id_selectors(ids)
        st.button("Update Data", on_click=lambda: _show_update_message("Data"))


def model_section() -> None:
    """Render model parameter inputs."""
    with st.expander("Model", expanded=False):
        st.session_state.learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1.0,
            value=float(st.session_state.learning_rate),
            step=1e-4,
            format="%f",
            help="\u8bad\u7ec3\u7684\u5b66\u4e60\u7387",
        )
        st.button(
            "Update Model", on_click=lambda: _show_update_message("Model")
        )


def task_section() -> None:
    """Render task parameter inputs."""
    with st.expander("Task", expanded=False):
        st.session_state.task_type = st.text_input(
            "Task Type",
            value=st.session_state.task_type,
            help="\u4efb\u52a1\u7c7b\u578b\uff0c\u5982\u5206\u7c7b",
        )
        st.button("Update Task", on_click=lambda: _show_update_message("Task"))


def trainer_section() -> None:
    """Render trainer parameter inputs."""
    with st.expander("Trainer", expanded=False):
        st.write("...")
        st.button(
            "Update Trainer",
            on_click=lambda: _show_update_message("Trainer"),
        )


def run_controls(config_path: str) -> None:
    """Render start and pause buttons."""
    from .pipeline import start_pipeline, toggle_pause

    run_col, pause_col = st.columns(2)
    if run_col.button("Start Experiment"):
        if st.session_state.test_id is None:
            st.warning("\u8bf7\u9009\u62e9 Test ID")
        elif not os.path.exists(st.session_state.data_dir):
            st.warning("\u8bf7\u6307\u5b9a\u6709\u6548\u7684 HDF5 \u76ee\u5f55")
        else:
            start_pipeline(config_path)
            _show_update_message("Run")
    if pause_col.button("Pause/Resume") and st.session_state.process:
        toggle_pause()
        _show_update_message("Pause")
