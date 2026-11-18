# Projected windows moved onto the registered dataset path

The private projection-windowing core that `tii_data` carried is now the
registered dataset adapter `src/data_factory/dataset_task/DG/tii_joint_dataset.py`
(`set_dataset` hook, listed in `dataset_task_mapping.csv`). Whole records are
read through the maintained H5 reader — either from the scratch link farm
(`from_inventory`, source inventories) or directly from the immutable source
root (`from_root`, fixed target episodes) — projected onto the orthogonal
physical bands, and sliced into deterministic whole-duration windows in one
shared implementation. `initialize_tii_data` and `load_tii_target_data` now
compose this class; the duplicated private loop is deleted. The historical
reading order (file-name groups, table order, left-to-right windows) is
preserved exactly.

Equivalence evidence: on the archived twelve single/joint S/U study
configurations, old and new data construction agree with `np.array_equal` on
the window inventory and **bitwise** equality on every waveform, mask and
label tensor and on the fitted source RMS (stronger than the required
`np.allclose(atol=1e-6)`).

`tii_joint` additionally gains two opt-in audits, both purely additive and off
by default (`task.branch_audit`, `task.optimizer_audit`):
`branch_rms.csv` records per-validation-epoch common/incremental feature RMS
and their ratio through the inference-mode frozen-feature path, and
`optimizer_trace.csv` records per-step learning rate with global and, for the
matched tokenizer, per-branch gradient norms read from the already-computed
gradients. Neither audit consumes training randomness, adds a backward pass,
or changes any existing column or file.
