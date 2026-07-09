# UXFD Operator Catalog (SSOT)

This catalog enumerates the currently available operators/features that can be composed by `TSPN` / `TSPN_UXFD`.

## Signal Processing Operators (`ALL_SP`)

Defined in `src/model_factory/X_model/TSPN.py` (mapping keys → operator modules).

### Unary operators

- `FFT`
- `HT`
- `WF`
- `I`
- `LNO`
- `RWF`
- `LWF`
- `CWF`
- `MWF`
- `Morlet`
- `Laplace`
- `Order1MAFilter`
- `Order2MAFilter`
- `Order1DFFilter`
- `Order2DFFilter`
- `Log`
- `Squ`
- `Sin`

### Binary operators (2-arity)

- `Add`
- `Mul`
- `Div`

Configuration mapping:
- `model.signal_processing_configs.layer1: ["I", "FFT", ...]`

## Feature Extractors (`ALL_FE`)

Defined in `src/model_factory/X_model/TSPN.py` (mapping keys → feature modules).

- `Mean`
- `Std`
- `Var`
- `Entropy`
- `Max`
- `Min`
- `AbsMean`
- `Kurtosis`
- `RMS`
- `CrestFactor`
- `Skewness`
- `ClearanceFactor`
- `ShapeFactor`

Configuration mapping:
- `model.feature_extractor_configs: ["Mean", "Std", ...]`

## UXFD-specific Modules (assembled by `TSPN_UXFD`)

Implementation: `src/model_factory/X_model/TSPN_UXFD.py`

- SP2D: `src/model_factory/X_model/UXFD/signal_processing_2d/`
  - knobs: `model.uxfd.enable_sp2d`, `model.uxfd.sp2d.*`
- Fusion: `src/model_factory/X_model/UXFD/fusion/`
  - knobs: `model.uxfd.fusion.type` (`concat|sum|gated`)
- Fuzzy: `src/model_factory/X_model/UXFD/fuzzy/`
  - knobs: `model.uxfd.fuzzy.enable`, `model.uxfd.fuzzy.logit_scale`
- Operator Attention: `src/model_factory/X_model/UXFD/operator_attention/`
  - knobs: `model.uxfd.operator_attention.enable`, `model.uxfd.operator_attention.operators`
- Logic (neuro-symbolic): `src/model_factory/X_model/UXFD/neurosymbolic/`
  - knobs: `model.uxfd.logic.enable`, `model.uxfd.logic.logit_scale`
