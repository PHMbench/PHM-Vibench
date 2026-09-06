# GFS Task Compatibility Path

`GFS` currently selects a hierarchical sampler and the ordinary
`GFS/classification` cross-entropy task. The maintained path is not generalized
few-shot learning.

## Current Surface

| Task type | Task name | Module | Current status |
|---|---|---|---|
| `GFS` | `classification` | `classification.py` | Execution-smoke path; sampled CE classification |
| `GFS` | `matching` | `matching.py` | Registered historical implementation; no maintained protocol |

Maintained compatibility config:

- `configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml`

The current sampler groups indices by system, domain and label and selects K+Q samples
per selected label. DataLoader then emits one ordinary `x/y/file_id` batch. The task
applies standard cross-entropy to all samples.

## What Is Not Implemented

```text
support/query markers
base/novel class definitions
episode-local adaptation
generalized query evaluation
base accuracy / novel accuracy / harmonic mean
```

`num_support` and `num_query` currently influence the number of sampled items, not a
support/query learning objective. `num_systems` is fixed to `1` because the current
sampler selects one system per episode.

A registered implementation or one successful smoke run must not be presented as a
generalized few-shot benchmark.
