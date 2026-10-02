# P08 physical conditioning

Run from the PHMFactory root with the existing `LQ_signal` environment. The dedicated
research entrypoint uses the public `analyze_config` and Model Factory. The ordinary
generic pipeline CLI does not execute this dedicated protocol. It does not
download data, infer missing physical metadata, or replace the model after a failure.

```bash
conda run --no-capture-output -n LQ_signal python -m scripts.p08_physical data-check --config configs/experiments/p08/physical_conditioning.yaml --local-config /absolute/path/p08.local.yaml
conda run --no-capture-output -n LQ_signal python -m scripts.p08_physical smoke --config configs/experiments/p08/physical_conditioning.yaml --local-config /absolute/path/p08.local.yaml --target 1 --seed 42 --arms B1,P0
conda run --no-capture-output -n LQ_signal python -m scripts.p08_physical tune --config configs/experiments/p08/physical_conditioning.yaml --local-config /absolute/path/p08.local.yaml --target 1 --arms B1,LATE,TOKEN,P0
```

`B0/B1/F01/P0` are index/physical coordinates crossed with neutral/FiLM conditioning.
`TOKEN` and `LATE` receive exactly the P0 condition encoding, but fuse before the token
processor or after pooling, respectively. Late fusion is a task-specific adaptation of
the feature-fusion comparison in [Domain Generalization: A Tale of Two ERMs,
Appendix D](https://arxiv.org/html/2510.04441v1), not an official DI-ERM reproduction.
The [FiLM source](https://arxiv.org/abs/1709.07871) establishes the inherited affine
conditioning mechanism; its vision-task training recipe is not a PHM tuning result.

The model is selectively reused from PHMFactory commit `05fe45519616edc214386c226301f8c667146ecb`.
The older P4 delivery was not recovered from the searched local paths, P08 branch
history, releases, or issues. This implementation does not claim byte identity with
that attachment. It uses the maintained real `E_01_HSE`, with no imitation or fallback.

## Required local data

`record_inventory` is a CSV with `record_id,system_id,physical_unit_id,raw_label,
sampling_rate,signal_path,signal_key,channel` plus declared physical-condition columns.
Paths identify immutable PHMFactory H5 records; channel and physical unit must be
documented. A recording with changing operating conditions must be segmented with a
documented synchronized condition, not assigned an unexplained average. The record
index is a local derived artifact, never an uploaded dataset.

`ontology_file` maps `system_id,raw_label,label,definition,source`. Integer equality is
not a physical mapping. `qualification_file` records one row per system with
`approved,ontology_source,group_source,channel_source,condition_source`. An unqualified
system stops the industrial run. The three-class value and speed/load field names in
the reference config are prerequisites to verify, not fabricated dataset properties.
If their documented common ontology or measured units differ, resolve and freeze the
local mapping before loading signals or inspecting any target outcome.

Verify raw recording ownership locally, including H5 hard-link aliases or different keys
that describe segments of one recording; metadata checks cannot establish those facts.
Source physical units are partitioned before windowing. Condition statistics and
vocabularies fit source training only. Per-window standardization is an explicitly
declared deterministic sample-wise transform. Target data never enters fitting,
hyperparameter selection, checkpoint selection, or condition statistics.

## Budgets and outputs

Each tuned arm receives the same six AdamW candidates (three learning rates, two weight
decays), with 40 epochs × 40 updates per candidate. Source-validation record Brier loss
includes the one-half factor and weights systems and physical units equally. Each arm may choose different optimizer
settings. The matched factorial comparison instead shares the selected B1 settings.
Five predeclared seeds describe optimization variability; they are not five machines.

Retain configuration, commands/code commit, source encoder state, logs, selected
checkpoints, record predictions, metrics, summaries and failed-run reasons. Existing
run directories are not overwritten. Report target systems individually; no automatic
positive paper claim follows from a run completing. Practical superiority requires a
separately justified diagnostic threshold, never the retired universal 0.01 rule.

Software check (no industrial signal access):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 conda run --no-capture-output -n LQ_signal python -m pytest -q test/test_p08_model.py test/test_p08_data.py test/test_p08_execution.py
```
