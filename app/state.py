"""Utilities for managing ``st.session_state`` variables."""
from __future__ import annotations

import streamlit as st


DEFAULT_STATE = {
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


def init_state() -> None:
    """Populate missing ``st.session_state`` keys with defaults."""
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value
