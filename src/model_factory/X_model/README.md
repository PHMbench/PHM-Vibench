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

## TSPN configurations for paper comparisons

TON, TIFN and DEN comparisons use the common `model.type: X_model`,
`model.name: TSPN` assembly with an explicitly reviewed configuration; a paper label
does not select a separate model or establish a faithful reproduction. In particular,
an unverified logic block cannot stand for DEN logic, and a deterministic view of one
waveform cannot stand for TIFN's independently measured sensor channels.

The shared TSPN model accepts these optional settings under `model`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `classifier_hidden_dims` | `[128]` | Hidden widths; `[]` gives one linear map. `[4]` gives two linear maps through width 4. |
| `classifier_activation` | `relu` | `relu` or `identity` between linear maps; no output activation. |
| `classifier_bias` | `true` | Bias in every classifier linear map. |
| `gate_parameterization` | `softmax` | `softmax` across output channels (`dim=0`) or unrestricted `raw` mixing weights. |
| `gate_temperature` | `0.1` | Finite positive temperature used for softmax mixing. |
| `gate_bias`, `skip_bias` | `true` | Bias in signal-layer mixing and optional residual skip maps. |
| `feature_mixing` | `shared` | One shared pre-statistic map or a separate map for each statistic (`per_feature`). |
| `feature_mixing_bias` | `true` | Bias in those pre-statistic maps. |
| `feature_definitions` | `{}` | Explicit definitions for active `Entropy` and `Kurtosis`; omitted entries keep historical definitions. |
| `feature_epsilon` | `1e-12` | Positive finite constant for the new absolute statistic and population-variance floor; not used by legacy definitions. |

`Entropy: legacy_softmax_weighted` computes
`mean(x * log_softmax(x))` along time. This is the original author-TSPN statistic,
now evaluated without `log(softmax(x))` underflow; it is **not Shannon entropy**.
Its availability does not resolve a paper's unspecified domain for `x * log(x)`.
`Entropy: absolute_mean_xlogx` computes `mean(abs(x) * log(abs(x) + feature_epsilon))`;
zero magnitude contributes zero. This explicitly declared adaptation averages along
time and also is not probability entropy. It does not silently replace the historical
feature or claim to reproduce an unspecified paper formula.
`Kurtosis: legacy_sample_variance` divides the fourth central population moment by
the square of sample variance (Bessel correction), retaining its undefined result for
a constant signal. `population_moment` computes `m4 / max(m2, feature_epsilon)^2`
using centered population moments; its explicit floor maps constant signals to zero.
Neither subtracts three. Record the chosen definition and epsilon in the run config;
these are protocol choices, not quantities tuned using target data.

For a three-layer width-four WF configuration, use `out_channels: 1`, `scale: 4`
and one `WF` per layer. `scale` determines the number of independently learned WF
centres/bandwidths; `out_channels: 4`, `scale: 1` instead shares one filter across
four mixed channels. The WF inverse transform preserves even and odd window lengths;
odd windows use their actual rFFT bin frequencies and have no Nyquist bin.

Omitted options preserve the original module names, parameter shapes, initialization
order and default computation, so existing p0 checkpoints load strictly. Opt-in
topology changes require their matching config for strict restoration. Feature
normalization updates running statistics on training batches only, stores them as
detached buffers, and uses those frozen values in evaluation. Switching an exported
model to training mode on target data violates that evaluation contract.

`python -m pytest test/test_tspn_paper_configs.py -q` checks the historical forward
equations/checkpoint keys, explicit parameter composition, odd/even lengths, gradients,
feature formulas and source-statistics batch invariance. Matching a paper's total
parameter count is only a consistency check: entropy semantics, classifier topology,
sensor meaning and source-only optimization still require independent verification.

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
