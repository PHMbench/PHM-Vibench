"""Registry page renderer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.console.adapters.registry import (
    load_data_factories,
    load_model_registry,
    load_task_registry,
    load_trainer_presets,
    load_trainer_registry,
)
from frontend.console.pages.shared import catalog_df
from frontend.console.theme import hero


def render_registry() -> None:
    """Render registry explorers and shipped presets."""
    hero(
        "Registry Explorer",
        "registry / presets / extension points",
        "See what the maintained system already wires before you open source files or invent new names.",
    )
    data_tab, model_tab, task_tab, trainer_tab, config_tab = st.tabs(
        ["Data", "Models", "Tasks", "Trainers", "Configs"]
    )
    with data_tab:
        st.dataframe(pd.DataFrame(load_data_factories()), use_container_width=True, hide_index=True)
    with model_tab:
        st.dataframe(pd.DataFrame(load_model_registry()), use_container_width=True, hide_index=True)
    with task_tab:
        st.dataframe(pd.DataFrame(load_task_registry()), use_container_width=True, hide_index=True)
    with trainer_tab:
        st.markdown("**Registered Trainers**")
        st.dataframe(pd.DataFrame(load_trainer_registry()), use_container_width=True, hide_index=True)
        st.markdown("**Trainer Presets**")
        st.dataframe(pd.DataFrame(load_trainer_presets()), use_container_width=True, hide_index=True)
    with config_tab:
        st.dataframe(catalog_df(), use_container_width=True, hide_index=True)
