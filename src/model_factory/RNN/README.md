# RNN Models

This directory contains recurrent neural networks for sequential modeling of industrial sensor data.

## Available models (`model.type = "RNN"`)

Example config:

```yaml
model:
  type: "RNN"
  name: "AttentionLSTM"   # or AttentionGRU / ConvLSTM / ResidualRNN / TransformerRNN
  # other hyperparameters...
```

Supported `model.name` values:
- `AttentionLSTM`
- `AttentionGRU`
- `ConvLSTM`
- `ResidualRNN`
- `TransformerRNN`

For non-ISFM models, `model.embedding`, `model.backbone`, and `model.task_head` are **not used** and should be set to `not_applicable` in the CSV registry.

Typical hyperparameters:
- `input_dim`, `hidden_dim`
- `num_layers`
- `bidirectional`
- task-specific outputs: `num_classes` or `output_dim`

Please consult the implementation files for detailed signatures, and extend this README with parameter tables and YAML examples as the interfaces stabilize.

