# CNN Models

This directory contains 1D convolutional architectures for industrial time-series and vibration signals.

## Available models (`model.type = "CNN"`)

Set in config:

```yaml
model:
  type: "CNN"
  name: "ResNet1D"        # or AttentionCNN / MobileNet1D / MultiScaleCNN / TCN
  # other hyperparameters...
```

Supported `model.name` values:
- `ResNet1D`
- `AttentionCNN`
- `MobileNet1D`
- `MultiScaleCNN`
- `TCN`

For non-ISFM models, `model.embedding`, `model.backbone`, and `model.task_head` are **not used** and should be recorded as `/` in the CSV registry.

## Common hyperparameters

Typical fields (names may vary slightly per file):

| Field             | Description                                   |
|-------------------|-----------------------------------------------|
| `input_dim`       | input feature dimension (channels)            |
| `num_classes`     | number of classes (classification)            |
| `output_dim`      | output feature dimension (regression)         |
| `layers` / depth  | number of blocks per stage (e.g. ResNet1D)   |
| `initial_channels`| base number of convolution channels          |
| `kernel_size`     | convolution kernel size (e.g. for TCN)       |
| `stride`, `dilation` | temporal stride / dilation               |
| `dropout`         | dropout ratio (if used)                       |

Please see each `.py` file for exact argument names.

## Example: ResNet1D configuration

From `ResNet1D.py`, the `Model` class expects at least:

- `input_dim`
- optionally: `block_type`, `layers`, `initial_channels`, `num_classes` or `output_dim`.

Classification example:

```yaml
model:
  type: "CNN"
  name: "ResNet1D"

  input_dim: 3           # e.g. 3-axis vibration
  block_type: "basic"    # or "bottleneck"
  layers: [2, 2, 2, 2]
  initial_channels: 64
  num_classes: 4         # fault classes
```

For regression / forecasting, you would instead set `output_dim` and omit `num_classes`.

