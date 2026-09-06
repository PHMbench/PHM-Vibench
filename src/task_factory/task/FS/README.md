# FS Task Compatibility Path

`FS` is currently a compatibility selector used by the existing data adapter and split
path. The maintained `FS/classification` task is ordinary supervised classification.
It does not construct N-way K-shot episodes.

## Current Surface

| Task type | Task name | Module | Current status |
|---|---|---|---|
| `FS` | `classification` | `classification.py` | Execution-smoke path; non-episodic CE |
| `FS` | `prototypical_network` | `prototypical_network.py` | Registered historical implementation; no maintained batch contract |
| `FS` | `matching_network` | `matching_network.py` | Registered historical implementation; no maintained batch contract |
| `FS` | `knn_feature` | `knn_feature.py` | Registered historical implementation |
| `FS` | `finetuning` | `finetuning.py` | Registered historical implementation |

Maintained compatibility config:

- `configs/demo/03_fewshot/cwru_protonet.yaml`

Despite the filename, the config resolves to `FS/classification` with cross-entropy. It
contains no support/query tensors, prototype computation, episode-local labels, or
query-only objective.

## Configuration

```yaml
task:
  type: FS
  name: classification
  loss: CE
  target_system_id: [1]
```

The removed `n_way`, `k_shot`, `q_query`, and `episodes_per_epoch` fields were not
consumed by the maintained sampler or task and therefore did not affect computation.

## Boundary for a Real ProtoNet Path

A future maintained ProtoNet path must provide together:

```text
N-way K-shot sampler
+ explicit support/query batch fields
+ episode-local label mapping
+ prototype computation
+ query loss
+ episodic metrics
```

Registration or a Python module alone is not scientific or release support.
