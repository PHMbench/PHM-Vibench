# X_model (Explainability / Auxiliary Models)

`X_model` contains explainability and auxiliary model entries that can be loaded by:

```yaml
model:
  type: "X_model"
  name: "<MODEL_NAME>"
```

## Directory map

- Stable entries used by `model_factory`:
  - `TSPN.py`, `TSPN_UXFD.py`, `NSN.py`, `MWA_CNN.py`, `BASE_ExplainableCNN.py`
  - `CI_GNN.py`, `GradCAM_XFD.py`, `Physics_informed_PDN.py`
  - `Resnet.py`, `Sincnet.py`, `WKN.py`, `EELM.py`, `MCN.py`, `TFN.py`, `F_EQL.py`
- Legacy source archive (migrated from
  `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model_collection`):
  - `legacy_collection/`
  - `legacy_collection/MCN/`
  - `legacy_collection/TFN/`
- Reusable UXFD modules:
  - `UXFD/`
- Baselines:
  - `baselines/`

## Factory-ready model names

`model.name` options under `type: "X_model"`:

- Existing:
  - `MWA_CNN`, `TSPN`, `TSPN_UXFD`, `NSN`, `BASE_ExplainableCNN`
- Added legacy wrappers:
  - `CI_GNN` (requires `torch_geometric`)
  - `GradCAM_XFD`
  - `Physics_informed_PDN`
  - `Resnet`
  - `Sincnet`
  - `WKN`
  - `EELM`
  - `MCN` (default: `MultiChannel_MCN_GFK`)
  - `TFN` (default: `TFN_Morlet`)
  - `F_EQL` (compatibility runnable implementation)

## Utility-only archived files

These are kept in `legacy_collection/` for traceability and reference, but are not registered as model entries:

- `legacy_collection/base_explainable.py`
- `legacy_collection/test_models.py`

## MWA_CNN compatibility

- `MWA_CNN.py` remains the single public factory entry (`model.name: "MWA_CNN"`).
- Source-style construction is kept through `Huan_net(input_size=..., num_class=...)` compatibility alias.

## Dependencies

For the migrated wrappers, ensure these packages exist in your environment:

- `torch_geometric` (required by `CI_GNN`)
- `pytorch_wavelets`
- `ptwt`

## Minimal YAML examples

### CI_GNN
```yaml
model:
  type: "X_model"
  name: "CI_GNN"
  in_channels: 8
  num_sensors: 8
  hidden_dim: 128
  num_layers: 3
  num_classes: 10
```

### GradCAM_XFD
```yaml
model:
  type: "X_model"
  name: "GradCAM_XFD"
  in_channels: 1
  seq_length: 4096
  dropout: 0.2
  num_classes: 10
```

### Physics_informed_PDN
```yaml
model:
  type: "X_model"
  name: "Physics_informed_PDN"
  input_dim: 4096
  hidden_dim: 128
  num_samples: 10
  num_classes: 10
```

### MCN (default GFK)
```yaml
model:
  type: "X_model"
  name: "MCN"
  mode: "gfk"
  in_channels: 2
  in_dim: 1024
  num_mfks: 8
  num_classes: 10
```

### TFN (default Morlet)
```yaml
model:
  type: "X_model"
  name: "TFN"
  variant: "morlet"
  in_channels: 2
  mid_channel: 16
  num_classes: 10
```
