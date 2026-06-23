# Data Model: PHM Task Experiment Matrix

## Task Family

Represents a PHM learning problem addressable through the task factory.

**Fields**:

- `task_type`: registry task type, such as `DG`, `CDDG`, `FS`, `GFS`, `pretrain`,
  or `Default_task`.
- `task_name`: registry task name, such as `classification` or `hse_contrastive`.
- `registry_path`: source path recorded in the task registry.
- `dataset_path`: dataset task wrapper recorded in the task registry.
- `batch_format`: expected batch keys or semantics recorded in the task registry.
- `notes`: registry notes used for human interpretation.
- `status`: one Support Status value.

**Validation rules**:

- `task_type` and `task_name` together must be unique in the task registry.
- A runnable task family must resolve to an implementation path and dataset path.
- Unregistered task modules are not supported until a registry row is added.

## Matrix Entry

Represents one task/config combination in the experiment matrix.

**Fields**:

- `entry_id`: stable identifier derived from the config registry row or focused
  test name.
- `config_path`: maintained config path when the entry is config-backed.
- `task_type`: resolved task type from the config.
- `task_name`: resolved task name from the config.
- `mode`: `smoke`, `full`, or `focused-test`.
- `data_requirement`: `offline` or `real-data-root`.
- `command`: command used to verify the entry.
- `expected_artifacts`: Slice 1 artifact expectations for completed runs.
- `status`: one Support Status value.
- `evidence`: latest Matrix Evidence.

**Validation rules**:

- A config-backed entry must appear in `configs/config_registry.csv` when it is a
  maintained demo or Hydra matrix entry.
- `task_type` and `task_name` must exist in the task registry.
- `full` entries must require an explicit real-data root.
- `smoke` entries must not require private raw data.

## Support Status

Auditable task-family status.

**Values**:

- `smoke-tested`: at least one offline command or focused test passes.
- `real-data-ready`: full matrix command is defined and real-data gate has recorded
  evidence when data is available.
- `unverified`: registry/config surface exists but no passing smoke/full evidence is
  recorded.
- `unsupported`: entry is absent from source-of-truth registries or intentionally
  out of scope with a reason.

**Validation rules**:

- Every registry-backed task family has exactly one status.
- A status other than `unsupported` needs command or test evidence.
- Unsupported and unverified entries must not silently run as another task.

## Task/Data Compatibility Contract

Defines the minimum compatibility expectations between task and data.

**Fields**:

- `required_batch_keys`: keys or semantics the task needs from a batch.
- `required_metadata_fields`: fields such as system id, domain id, class label, or
  file id.
- `class_constraints`: few-shot and generalized few-shot class/sample constraints.
- `domain_constraints`: DG/CDDG system and domain requirements.
- `pretrain_constraints`: objective-specific fields for classification,
  contrastive, reconstruction, or prediction pretraining.

**Validation rules**:

- Missing registry `batch_format` information is a matrix gap.
- Domain, system, class, and shot constraints must fail explicitly when they can be
  checked from config or metadata.
- Runtime-only compatibility failures must identify the matrix entry and task family.

## Matrix Evidence

Represents a checked result for a matrix entry.

**Fields**:

- `command`: exact command run.
- `result`: `pass`, `fail`, or `skipped`.
- `reason`: required for failures and skips.
- `artifact_paths`: run artifacts or logs produced by a passing entry.
- `timestamp`: time evidence was recorded.

**Validation rules**:

- Do not mark an entry verified without a command result.
- Skipped full-matrix validation must state whether real data was unavailable.

## Validation Result

Summarizes matrix validation.

**Fields**:

- `registry_consistency`: task registry and config registry cross-check result.
- `smoke_matrix`: offline matrix result.
- `full_matrix`: real-data matrix result or explicit skip reason.
- `atlas_sync`: generated atlas diff status.
- `open_gaps`: unsupported or unverified entries requiring follow-up.

**Validation rules**:

- Any failed gate must include the command and concrete failure.
- Atlas sync must be checked after maintained registry changes.
