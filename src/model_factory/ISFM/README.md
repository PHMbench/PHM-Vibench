# Industrial Signal Foundation Models

ISFM composes an embedding, a backbone and a task head. Their roles are distinct:

```text
signal + per-sample metadata
→ embedding / patch representation
→ backbone
→ task-specific head
```

## Select a model and its components

A selection fragment matching the maintained family is:

```yaml
model:
  type: ISFM
  name: M_01_ISFM
  embedding: E_01_HSE
  backbone: B_04_Dlinear
  task_head: H_01_Linear_cla
```

Use a complete configuration from [the demo index](../../../configs/demo/README.md) to
supply component parameters. Other family modules include `M_02_ISFM`,
`M_02_ISFM_heterogeneous_batch` and `M_03_ISFM`; inspect their own forward and metadata
contracts rather than assuming they accept every interchangeable component combination.

[isfm_components.csv](isfm_components.csv) indexes embedding/backbone/head modules and
arguments. It is a component catalogue, not evidence of all combinations being tested.
The parent [Model Factory guide](../README.md) defines top-level construction and weights.

## Representation and shape boundaries

`E_01_HSE` uses `patch_size_L`, `patch_size_C`, `num_patches` and `output_dim`. Its forward
receives per-sample sampling-rate information, rather than a dataset-wide rate guessed
from the model name. Existing metadata compatibility accepts `Sample_rate` and the
historical `Sample_Rate` spelling; preserve actual per-sample values.

Patch length and channel count must fit the declared signal. Do not silently shrink
patches or broadcast an unrelated sample's metadata. Training-time sampling and
evaluation-time sampling must be distinguished and covered by the embedding tests.
Ablation modules have their own arguments and need their own comparison evidence.

For `B_04_Dlinear`, `num_patches` and `output_dim` determine the embedded sequence and
feature dimensions. For `H_01_Linear_cla`, the feature width and system/class-count mapping
must match the selected head; keep the batch system identity intact. Do not infer a new
ontology from a partial batch or remap unknown systems to a known head.

## Component navigation

Use [embedding/](embedding/), [backbone/](backbone/), and [task_head/](task_head/) for the
implementations. The catalogue retains exact IDs such as `E_03_Patch`, `B_08_PatchTST`,
`H_03_Linear_pred`, and `H_10_ProjectionHead`; their presence does not establish a complete
prediction or contrastive-learning protocol.

A component change should document its actual tensor/metadata contract and test the
affected complete configuration. Do not edit the CLI or add another configuration
loader. Parameter efficiency, cross-system generalization and diagnostic accuracy require
matched experiments and are not guaranteed by selecting the ISFM family.
