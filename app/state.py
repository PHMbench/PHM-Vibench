"""Utilities for managing Streamlit session state."""
from __future__ import annotations

import streamlit as st


def init_state() -> None:
    """Initialize default variables in ``st.session_state``.

    This prevents losing data between reruns triggered by widget
    interaction. Call this function once at the start of ``main``.
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
