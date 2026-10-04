# Explainability and auxiliary models

This folder holds reusable signal-processing, explanation and auxiliary implementations.
Not every helper is a top-level Factory model. Select an exact model module, for example:

```yaml
model:
  type: X_model
  name: TSPN_UXFD
```

This is a fragment; use the corresponding complete configuration for operator and shape
settings. Catalogue entries include `MWA_CNN`, `TSPN`, `TSPN_UXFD`, `XOANOperatorPath` and
`BASE_ExplainableCNN`. Consult [model_registry.csv](../model_registry.csv) and the selected
source, rather than assuming `Feature_extract.py` or every helper exposes `Model`.

## Boundaries

Model assembly follows the parent [Factory contract](../README.md). Keep reusable code
here; paper-specific methods, figures and results belong to their research repository.
Use [paper/project/README.md](../../../paper/project/README.md) for migrated source
locations, not the removed historical paper-submodule paths.

The [LLM explanation integration guide](../../../docs/LLM_EXPLANATION_INTEGRATION.md)
describes adapters for model traces. A valid trace or citation reference does not by
itself prove causal, physical or natural-language faithfulness. Preserve the actual
forward branch, intervention settings and active masks when reporting explanations.

For a change, use the relevant assembly/trace tests and exact configuration. Do not infer
universal model support, numerical performance or a paper claim from helper imports.

## MWA-CNN-6 industrial comparator

Install the optional wavelet dependency with `python -m pip install -e '.[mwa]'`.
The extra bounds setuptools below 82 because `pytorch-wavelets==1.3.0` imports
`pkg_resources`, which [setuptools removed in version 82](https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html).
Select the six-stage model explicitly:

```yaml
model:
  type: X_model
  name: MWA_CNN
  depth: 6
  in_channels: 1
  num_classes: 3
```

This extends the repository's existing four-stage implementation to the architecture
in the [authors' MWA-CNN-6 source](https://github.com/PHM-Code/MWA-CNN/blob/692020e1dc9d41131844b77935fa0318af709b49/Network_Code/MWA-CNN-6.ipynb).
The architecture has six db16 DWT stages with zero boundary extension, initial width
12, GroupNorm with six groups, five residual channel-attention blocks and dropout
0.1, followed by global average pooling and a linear classifier. Input is `(B,L,C)`;
class count and input channels are explicit task adaptations. No notebook is vendored.
The measured window must have at least 32 samples, one db16 filter length; deeper
coefficients still use the declared zero extension. Training batches require at least
two observations because attention applies BatchNorm after temporal pooling.

Omitted `depth` keeps the existing four-stage model and checkpoint names for the
maintained P07 configuration; it is not MWA-CNN-6. New six-stage experiments must set
`depth: 6`. Device placement belongs to the Trainer. The model performs no data
normalization and does not read metadata, labels or domain IDs.

The authors' training example uses CE, Adam at 1e-4, batch size 64 and 150 epochs.
Those facts do not validate a new DG training budget. PHMFactory's experiment protocol
owns shared preprocessing, group sampling, augmentation, optimizer search and
source-only checkpoint selection. The notebook's per-epoch test evaluation and
maximum-test-accuracy selection are not reproduced. Comparing this model under a
new DG protocol is an architecture reproduction, not reproduction of published scores.

`python -m pytest test/test_mwa_cnn.py -q` checks the six-stage parameter dimensions,
wavelet values against PyWavelets, gradients, CPU construction and strict checkpoint
restoration on synthetic inputs. Industrial performance and CUDA execution remain
separate validation tasks.
