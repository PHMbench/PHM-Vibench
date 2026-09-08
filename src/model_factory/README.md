# Model Factory

Select a model through the resolved `model` configuration. The public Factory imports
`src.model_factory.<type>.<name>` and constructs `Model(args_model, metadata)`. It returns
a PyTorch module; the Trainer owns device placement, not this constructor.

## Select an implementation

The following is a model fragment, not a complete experiment:

```yaml
model:
  type: ISFM
  name: M_01_ISFM
  embedding: E_01_HSE
  backbone: B_04_Dlinear
  task_head: H_01_Linear_cla
```

Use a complete maintained configuration to supply the remaining parameters and metadata.
`model.type` is the directory and `model.name` is the module exporting `Model`; neither is
an arbitrary class description. The public Pipeline receives an already analyzed
configuration and must not parse or merge YAML again inside this Factory.

Implementation families:
[CNN](CNN/README.md), [RNN](RNN/README.md), [MLP](MLP/README.md),
[Transformer](Transformer/README.md), [neural operators](NO/README.md),
[ISFM](ISFM/README.md), [ISFM Prompt](ISFM_Prompt/README.md), and
[explainability/auxiliary models](X_model/README.md).

## Catalogue is not support

[model_registry.csv](model_registry.csv) indexes module paths, typical arguments and
recorded test notes. It is not a list of scientifically validated Data × Model × Task
combinations. Check the selected implementation, actual tests, complete configuration,
and [supported combinations](../../SUPPORTED_COMBINATIONS.md).

Changing a model can change input layout, output meaning, sequence-length requirements,
metadata needs or dependencies. Verify those boundaries rather than silently resizing
an input, remapping labels or replacing the task to make a forward pass work.

## Explicit checkpoint loading

```yaml
model:
  weights_path: /path/to/checkpoint.ckpt
  weights_strict: true
```

This fragment extends the selected model, not its identity. `weights_strict` must be a
YAML boolean, not a quoted string. Strict loading is the default. Intentional non-strict
transfer still requires at least one matching parameter name and shape; zero matches
must fail rather than silently use a random model. Construction/loading failures remain
errors, not a signal to select another model.

## Add a compatible model

Place it in the appropriate family and expose `Model(args_model, metadata)`. State input,
output, dtype and metadata requirements, then add a focused test and minimal configuration.
Use the existing Factory resolution; do not edit the CLI or add another registry.
Update catalogue/navigation when promoting a maintained implementation, not as a
substitute for execution. Preserve caller configuration when changing constructors;
existing inferred-field mutation is not a design pattern to copy.

Read [contributing.md](contributing.md) for the contribution boundary. For a model change,
run the affected tests and a compatible maintained smoke; a directory-wide model list is
not a requirement to execute every research model.
