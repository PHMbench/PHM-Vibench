# Multi-Task Pretrain/Fine-Tune Pipeline — Historical Status

This path is retained as a compatibility and provenance marker. The former guide
described an unverified two-stage Pipeline 03 workflow and is **not** current
PHM-Vibench operating guidance.

## Current status

`Pipeline_03_Multitask_Pretraining_Finetuning.py` exists as research-oriented code, but
Pipeline 03 is outside the current release-supported surface. The maintained
public entrypoint remains:

```bash
python main.py --config <yaml> [--override key=value ...]
```

The previous page used direct script commands, proposed configuration keys, and
paths such as:

```text
configs/multitask_pretrain_finetune_config.yaml
```

without a maintained registry entry and release-gate evidence. It also included
example accuracy, RUL, anomaly, convergence, statistical-significance, and
performance-improvement numbers that were not linked to reproducible repository
artifacts. Those values must not be treated as benchmark results.

## Use maintained documentation

- [Quickstart](quickstart.md)
- [Configuration system](../configs/README.md)
- [Supported components](../SUPPORTED_COMPONENTS.md)
- [Supported combinations](../SUPPORTED_COMBINATIONS.md)
- [Known limitations](../KNOWN_LIMITATIONS.md)
- [Testing and evidence](testing.md)

The current maintained pretraining surface is limited to the configurations and
single-stage behavior explicitly listed in the support documents.

## Requirements for a future maintained Pipeline 03 workflow

A promotion PR must provide:

- a valid config-first `main.py --config` entry;
- explicit stage configuration and state transitions;
- checkpoint production, selection, loading, and resume rules;
- model/task/data compatibility constraints;
- train-only normalization and split/leakage controls;
- focused tests for each stage and stage handoff;
- a minimum pretrain → fine-tune → evaluate closure;
- artifact provenance and exact reproduction commands;
- multi-seed evidence before numerical performance claims;
- updated registry, atlas, support, migration, and limitation documents.

Until then, Pipeline 03 work should remain under an explicitly experimental
configuration and must not be represented as a released benchmark capability.

The former detailed guide remains available in Git history for research
provenance, but its commands and numerical examples are intentionally removed from
the maintained documentation surface.
