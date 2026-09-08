# ISFM Prompt research models

The catalogue entry is:

```yaml
model:
  type: ISFM_Prompt
  name: M_02_ISFM_Prompt
```

This is only a selection fragment. Inspect [M_02_ISFM_Prompt.py](M_02_ISFM_Prompt.py) for
its current prompt parameters, component assembly, input metadata and model branch.
`name: ISFM_Prompt` is not this module's identifier. Some prompt implementations assemble
components internally; do not assume all standard ISFM component selectors are exposed.

Prompt-only tuning requires an explicit frozen/trainable parameter selection and a
measured parameter count. Do not repeat a less-than-one-percent training claim without
its actual configuration and measurement. Pretraining, finetuning and adaptation need
separate data and evaluation boundaries; a prompt parameter alone defines none of them.

Preserve dataset/system identity. Do not copy truncation, modulo remapping, random-arm or
signal-only fallback recipes to handle incompatible metadata. Check both the selected
arm and failure behavior with focused tests before advertising a maintained example.

See [Model Factory](../README.md), [ISFM](../ISFM/README.md) and the
[model catalogue](../model_registry.csv). This is a research interface, not a claim that
all documented prompt variants are validated on the current software.
