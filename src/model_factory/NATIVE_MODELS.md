# Native time-series classifiers

These are native supervised classification ports, selected by the existing Model Factory.
They do not wrap an external training runner or download weights. The existing
`DG.classification` Task owns cross entropy, metrics and optimization.

| Model selector | Upstream source | Complete CPU example |
| --- | --- | --- |
| `Transformer.iTransformer` | [Time-Series-Library iTransformer](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/models/iTransformer.py) | [iTransformer](../../configs/experiments/model_integration/itransformer_dummy.yaml) |
| `CNN.TimesNet` | [Time-Series-Library TimesNet](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/models/TimesNet.py) | [TimesNet](../../configs/experiments/model_integration/timesnet_dummy.yaml) |
| `CNN.TSLANet` | [Official TSLANet classification](https://github.com/emadeldeen24/TSLANet/blob/ca0e88416d3ae49fd50e399c44ae94868378a94d/Classification/TSLANet_classification.py) | [TSLANet](../../configs/experiments/model_integration/tslanet_dummy.yaml) |

## Run

From the repository root, after the usual PHMFactory installation:

```bash
phmfactory preflight --config configs/experiments/model_integration/itransformer_dummy.yaml
phmfactory --config configs/experiments/model_integration/itransformer_dummy.yaml
```

Substitute the corresponding complete YAML for TimesNet or TSLANet. These examples use
bundled Dummy CSV inputs, one CPU epoch, the existing selected-checkpoint lifecycle and
both acc and f1. The Dummy reader consumes `ch1, ch2`; no channels are synthesized to
match illustrative metadata counts. Results are software smoke evidence, not diagnostic
performance or an independent held-out protocol claim.

## Interface

`Model(args_model, metadata)` returns a torch module. It consumes fixed-length, fully
observed float32 `[B, L, C]` signals. `seq_len=L`, `input_dim=C`, and `num_classes` are
explicit. One integer class count or one-entry class-count mapping is accepted; multiple
independent label ontologies require a different head, not silently pooled logits.

`forward(x, file_id=None, task_id=None, return_feature=False)` returns `[B, K]` logits;
`return_feature=True` returns `(logits, features)`. `task_id` is absent or `classification`.
There is no forecasting, imputation, probabilistic or pretraining Task hidden behind these
names. Construction does not move devices or alter caller configuration.

Shared architecture fields are `d_model`, `e_layers`, `dropout`. Other required fields:

- iTransformer: `n_heads`, `d_ff`, `activation` (`gelu` or `relu`). Width is divisible by
  heads. Variate tokens and the native GELU/dropout/flatten classifier are retained;
  the forecasting branch's normalization is not applied to classification.
- TimesNet: `d_ff`, `top_k`, `num_kernels`; width is even. The block selects only positive
  FFT bins, fixing the upstream DC tie case on constant signals. Internal zero padding
  builds a periodic convolution grid and is cropped back to the original length; it is
  part of the algorithm, not input resampling. Calendar embeddings and unused tasks are
  omitted. Frequency selection depends on the batch, as in the upstream method.
- TSLANet: `patch_size`, and explicit booleans `adaptive_filter`, `apply_asb`, `apply_icb`.
  Stride remains half the patch length. The supported configuration covers the full
  sequence, rejects silent tail truncation, and requires ASB or ICB. The original
  straight-through spectral threshold and interactive convolutions are retained.
  Global argparse flags become per-instance values. Masked pretraining and its unused
  classifier input projection are not imported.

The port has its own parameter names; an upstream checkpoint is not silently remapped.
PHMFactory-created checkpoints are restored strictly. Matching seed values alone do not
imply identical initial weights to an upstream training script with unused parameters.

## Validation

```bash
python -m pytest test/test_native_model_ports.py -q
python -m pytest test/test_native_model_public_path.py -q
```

The first suite exercises actual torch forwards, cross-entropy gradients, parameter
updates, strict save/load, eval RNG behavior, parameter/input failures and separate
reference equations for attention, periodic convolution and spectral gating. These
reference equations are not a rerun of the original published training experiments.
The second suite executes three real public CLI Dummy fits with their exact configurations;
it is not a mocked Pipeline. The dedicated CI runs both files. Existing package/user-path
checks remain enabled.

## Sources and licensing

The iTransformer and TimesNet adaptations derive from Time-Series-Library at
`4e938a1767106324dd753b2a44832bf870a0252e` (MIT, Copyright 2021 THUML @ Tsinghua University).
TSLANet derives from its official repository at `ca0e88416d3ae49fd50e399c44ae94868378a94d`
(MIT, Copyright 2024 Emadeldeen Eldele). The full notices accompany the installed Python
modules in [_native_classification.py](_native_classification.py). No third-party
checkpoint or dataset is redistributed.

## Catalogue integration boundary

The supplied 187-row spreadsheet is a candidate list, not 187 completed implementations.
These three additions do not close the full catalogue. Existing models must be compared
before replacement; forecasting models need an actual forecasting Task; foundation
checkpoints and closed services need their own executable evaluation. Missing dependencies,
weights, credentials, licenses or upstream implementations must stay explicit rather than
be replaced by another network or counted as passed tests.
