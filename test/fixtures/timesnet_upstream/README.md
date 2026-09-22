# TimesNet numerical reference (test only)

Source: THUML Time-Series-Library at `4e938a1767106324dd753b2a44832bf870a0252e`.
MIT license: [LICENSE](LICENSE). Not part of the installed PHMFactory wheel.

- `TimesNet.py`: upstream `models/TimesNet.py`; only the two imports are relocated.
- `Embed.py`: upstream `layers/Embed.py`, unchanged definitions through `DataEmbedding`.
- `Conv_Blocks.py`: upstream `layers/Conv_Blocks.py`, unchanged `Inception_Block_V1`.
- Unused trailing embedding/block classes are omitted. No numerical behavior is patched.

The test initializes the real upstream classifier, copies all active weights into the
port, and compares full logits, cross entropy, input gradients and every active parameter
gradient. The only unused parameter is the upstream calendar embedding, which classification
never calls. The fixed positional buffer is sliced to the configured length explicitly.

The supported comparison uses fully valid, fixed-length sequences (all-ones mask), CPU
float32, evaluation and training with matched RNG. The DC zero-spectrum defect is reproduced
separately; the corrected bin selection is not claimed to be bitwise upstream-equivalent
on that defect. Forecasting, imputation, missingness and original benchmark results are
outside the claim. Tests do not download code, weights or data.
