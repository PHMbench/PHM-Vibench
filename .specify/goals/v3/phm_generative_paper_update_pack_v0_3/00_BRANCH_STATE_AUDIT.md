# 00. Branch State Audit: `Feature_factory-update`

## Current high-level state

The branch has grown into a paper-oriented PHM generative benchmark branch.
It is ahead of `main` by 33 commits and includes a large cleanup/removal of
`.claude` / `.codex` skill artifacts, a new `.specify` goal queue, specs under
`specs/002-phm-genbench-frontier/`, multiple generative configs, runtime tasks,
metrics, and paper scripts.

## Positive changes already present

### 1. Entry-path safety

`main.py` now has:

```text
ALLOWED_PIPELINES
validate_pipeline_name()
_load_yaml_probe()
--preflight-only
pydantic ExperimentConfig validation
```

This is a major improvement over the earlier state where malformed YAML could
silently fall back.  Keep this.

### 2. Generative runtime exists

`Pipeline_06_generative.py` supports:

```text
mode=train
mode=sample
mode=eval
```

It now records normalization artifacts, hashes config/protocol/dependency-lock
inputs, writes sample payloads with conditions, and computes eval metrics.

### 3. Task registry is broader

The task registry now names:

```text
conditional_flow_matching
rectified_flow
ddpm_epsilon
score_sde
meanflow
drifting_flow
transition_flow_matching
ot_nfm
```

The last four must remain exploratory until promoted.

### 4. Model factory has more backbones

Current generative backbones include:

```text
phm_cfm_mlp1d
phm_unet1d
phm_dit1d
mamba1d_backbone
```

The `mamba1d_backbone` should be treated as SSM-style placeholder unless a real
Mamba/selective-SSM implementation is added and tested.

### 5. Synthetic manifest improved

The manifest now rejects forbidden source splits, tracks normalization evidence,
config/protocol/dependency hashes, condition sampling evidence, leakage checks,
and missing evidence.

### 6. Paper-side scripts exist

Current branch includes:

```text
scripts/generative_benchmark_effect.py
scripts/generative_sweep.py
scripts/paperpack_generative.py
scripts/generative_submission_draft.py
configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml
```

This is the right direction for a paper repository.

## High-risk gaps

### Gap A: sample/eval/paperpack path coherence

The benchmark-effect dry-run planner writes train/sample/eval/paperpack stages
with `<experiment_name>` placeholders.  Paperpack scans the eval run directory
for `synthetic_data_manifest.json`, but sample manifests may live in a sibling
sample run directory.  Without an execution ledger or artifact resolver, the
paperpack can miss manifests.

Required fix:

```text
Add artifact_linker.json or run_stage_ledger.csv:
train checkpoint -> sample generated_path/manifest -> eval metrics -> paperpack.
```

### Gap B: metric-status evidence is not promotion-ready

`synthetic_data_manifest.py` requires `metric_status_reason_recorded=True` for
benchmark-ready status, but sample-time manifests are written before eval-time
metric statuses exist.  This is good for safety but means no sample manifest
can be benchmark-valid without a later promotion/eval manifest.

Required fix:

```text
Add eval_evidence_manifest.json or promoted_synthetic_data_manifest.json.
```

### Gap C: train_distribution condition sampling can use all metadata

The `train_distribution` selector samples metadata rows with `split=train` when
available, but if metadata rows have no split field it can include all rows.
That must be explicit in the manifest and must not be benchmark-valid.

Required fix:

```text
If no split field exists, mark condition_sampling_policy evidence as
train_distribution_unverified and keep exploratory.
```

### Gap D: pipeline `_to_ncl` is permissive

The pipeline-level `_to_ncl()` returns contiguous data even if channel inference
fails.  The task-level `_to_ncl()` raises an error.  The pipeline helper should
match the stricter behavior to prevent silent metric shape errors.

### Gap E: TSTR is only a nearest-centroid probe

The current `tstr_metrics()` is useful as a smoke metric, but it is not enough
for paper utility claims.  Paper results need either:
- fixed downstream classifier protocol, or
- explicitly named `nearest_centroid_probe_tstr`, not full TSTR.

### Gap F: exploratory methods are runnable too early

MeanFlow, drifting, transition flow, and OT-NFM currently inherit the rectified
flow velocity contract.  This is acceptable only if they remain labeled
`experimental=true`, `num_steps=1`, and never benchmark-valid before promotion.

### Gap G: tool/process artifact cleanup is too broad

The branch removes large `.claude` and `.codex` tool artifacts.  That may be
intentional, but it should be isolated from runtime generative PRs.  Keep
process artifacts under `.specify` / `specs/002-phm-genbench-frontier/`, but do
not mix repo hygiene removals with model/task/metric changes in one merge.
