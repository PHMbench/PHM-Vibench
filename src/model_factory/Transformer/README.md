# Transformer Models

This directory provides transformer-based architectures for long-range temporal modeling.

## Available models (`model.type = "Transformer"`)

Config pattern:

```yaml
model:
  type: "Transformer"
  name: "PatchTST"        # or Autoformer / Informer / Linformer / ConvTransformer / Transformer_Dummy
  # transformer-specific hyperparameters...
```

Supported `model.name` values:
- `PatchTST`
- `Autoformer`
- `Informer`
- `Linformer`
- `ConvTransformer`
- `Transformer_Dummy`

For non-ISFM models, `model.embedding`, `model.backbone`, and `model.task_head` are **not used** and should be recorded as `/` in the CSV registry.

## Common hyperparameters

Most transformer models share the following configuration fields (exact names may vary per file):

| Field       | Description                               |
|------------|-------------------------------------------|
| `input_dim`| input feature dimension                   |
| `d_model`  | model dimension                           |
| `n_heads`  | number of attention heads                 |
| `num_layers` / `e_layers` / `d_layers` | encoder/decoder depth |
| `d_ff`     | feed-forward dimension                    |
| `dropout`  | dropout probability                       |
| `seq_len`  | input sequence length (if used)           |
| `pred_len` | prediction length (forecasting models)    |

Please consult each model file for the exact argument list.

## Example: PatchTST configuration

PatchTST (`PatchTST.py`) expects at least:

- `input_dim`
- optionally: `patch_size`, `stride`, `d_model`, `n_heads`, `num_layers`, `d_ff`, `dropout`, `num_classes` or `output_dim`.

Classification-style usage:

```yaml
model:
  type: "Transformer"
  name: "PatchTST"

  input_dim: 3          # number of channels
  patch_size: 16
  stride: 8
  d_model: 256
  n_heads: 8
  num_layers: 6
  d_ff: 512
  dropout: 0.1
  num_classes: 4
```

Forecasting-style usage would instead set `output_dim` and omit `num_classes`.

