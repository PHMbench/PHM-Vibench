"""Finite batch controls; rendering never schedules the next experiment."""
from pathlib import Path

import streamlit as st
import yaml

from .batch_service import BatchPlanError, plan_grid
from .config_service import ConfigServiceError, parse_yaml_text
from .run_service import (
    RunRequest, RunServiceError, cancel_batch, continue_batch, get_batch,
    list_batches, start_batch,
)

# Only these existing user-selected fields can vary in the first batch UI.
# In particular, planning does not change data selection, split or model identity.
GRID_PATHS = (
    "task.lr", "data.batch_size", "trainer.num_epochs",
    "environment.seed", "environment.iterations",
)


def _batch_status(repo_root: Path, record) -> None:
    st.write(f"Batch {record.batch_id}: **{record.status}**")
    st.caption(f"{len(record.trials)} planned CLI calls; {record.total_fits} planned fits. "
               "Trial status is process status, not benchmark validation.")
    st.dataframe([
        {key: trial[key] for key in ("index", "status", "fit_count", "run_id", "error")}
        for trial in record.trials
    ], hide_index=True, use_container_width=True)
    if record.error:
        st.warning(record.error)
    left, right = st.columns(2)
    if left.button("Continue remaining trials", key=f"continue-batch::{record.batch_id}",
                   disabled=record.status != "paused"):
        try:
            continue_batch(repo_root, record.batch_id)
            st.rerun()
        except RunServiceError as error:
            st.error(str(error))
    if right.button("Cancel batch", key=f"cancel-batch::{record.batch_id}",
                    disabled=record.is_terminal):
        cancel_batch(repo_root, record.batch_id)
        st.rerun()
    st.caption("Continuing runs only pending trials. Failed trials remain failed and are not retried.")
    for trial in record.trials:
        if trial["run_id"] and st.button(f"View trial {trial['index']}",
                                        key=f"batch-run::{record.batch_id}::{trial['index']}"):
            st.session_state.selected_run_id = trial["run_id"]
            st.session_state.active_run_id = trial["run_id"]
            st.rerun()


@st.fragment(run_every="2s")
def _active_batch(repo_root_text: str, batch_id: str) -> None:
    record = get_batch(Path(repo_root_text), batch_id)
    if record.status not in {"running", "cancelling"}:
        st.rerun()
    _batch_status(Path(repo_root_text), record)


def render_batch_controls(repo_root: Path, approved_yaml: str, *, template_id: str, mode: str) -> None:
    st.subheader("有限批次 | Finite batch")
    st.caption("Preview a fixed grid, then explicitly run it. One trial at a time; "
               "the first failed or cancelled trial pauses pending work.")
    with st.expander("Plan parameter combinations"):
        st.caption("Allowed existing paths: " + ", ".join(GRID_PATHS))
        grid_text = st.text_area("Parameter grid (YAML lists)",
                                value="task.lr: [0.001, 0.0005]", key="batch_grid_text")
        left, right = st.columns(2)
        max_trials = left.number_input("Maximum CLI calls", min_value=1, max_value=16,
                                      value=16, step=1, key="batch_max_trials")
        max_fits = right.number_input("Maximum fits", min_value=1, max_value=64,
                                       value=64, step=1, key="batch_max_fits")
        inputs = (approved_yaml, grid_text, max_trials, max_fits, template_id, mode)
        if st.button("Preview batch", disabled=not approved_yaml):
            try:
                plan = plan_grid(parse_yaml_text(approved_yaml), yaml.safe_load(grid_text),
                                 allowed_paths=GRID_PATHS, max_trials=max_trials, max_fits=max_fits)
                st.session_state.batch_plan = plan
                st.session_state.batch_plan_inputs = inputs
                st.session_state.batch_plan_submitted = False
            except (BatchPlanError, ConfigServiceError, yaml.YAMLError) as error:
                st.session_state.batch_plan = None
                st.session_state.batch_plan_inputs = None
                st.error(str(error))
        plan = st.session_state.get("batch_plan")
        current = plan is not None and st.session_state.get("batch_plan_inputs") == inputs
        if plan is not None and not current:
            st.warning("Batch inputs changed. Preview the new plan before running.")
        if current:
            st.write(f"**{len(plan.trials)} CLI calls / {plan.total_fits} fits**")
            st.dataframe([
                {"trial": trial.trial_id, "fits": trial.fit_count,
                 **{path: repr(value) for path, value in trial.overrides}}
                for trial in plan.trials
            ], hide_index=True, use_container_width=True)
            st.caption("Seed and iterations are preserved. A seed grid does not reset iterations to one.")
            selected_trial = st.selectbox("Inspect planned YAML",
                                             range(len(plan.trials)),
                                             format_func=lambda i: plan.trials[i].trial_id)
            st.code(plan.trials[selected_trial].config_yaml, language="yaml")
        batches = list_batches(repo_root)
        reserved = any(not batch.is_terminal for batch in batches)
        submitted = st.session_state.get("batch_plan_submitted", False)
        if st.button("Run batch", disabled=not current or reserved or submitted, type="primary"):
            try:
                with st.spinner("Checking all trial snapshots before submitting the batch..."):
                    batch = start_batch(
                        RunRequest(repo_root=repo_root, template_id=template_id, mode=mode,
                                   config_yaml=approved_yaml), plan)
                st.session_state.selected_batch_id = batch.batch_id
                # Keep the inspected YAML visible; require a fresh Preview before resubmission.
                st.session_state.batch_plan_submitted = True
                st.rerun()
            except (RunServiceError, BatchPlanError, ConfigServiceError) as error:
                st.error(str(error))
        if submitted:
            st.caption("This preview has been submitted. Preview a new plan to submit again.")
        if reserved:
            st.info("A batch owns this worker. Continue or cancel paused work before another submission.")
    batches = list_batches(repo_root)
    if not batches:
        return
    ids = tuple(batch.batch_id for batch in batches)
    preferred = st.session_state.get("selected_batch_id", ids[0])
    selected = st.selectbox("Recent batches", ids,
                            index=ids.index(preferred) if preferred in ids else 0)
    st.session_state.selected_batch_id = selected
    record = get_batch(repo_root, selected)
    if record.status in {"running", "cancelling"}:
        _active_batch(str(repo_root), selected)
    else:
        _batch_status(repo_root, record)
