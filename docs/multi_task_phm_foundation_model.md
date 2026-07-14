# Multi-Task PHM Model — Experimental Status Note

This path is retained because it previously described a proposed multi-task
Prognostics and Health Management (PHM) model. It is **not** a maintained usage
guide and is outside the current release-supported surface.

## Why the former guide was retired

The previous page mixed implemented modules, missing paths, proposed
configurations, and example metrics without reproducible evidence. In particular,
it referenced:

```text
src/task_factory/multi_task_lightning.py
configs/multi_task_config.yaml
```

which are not current maintained entrypoints. It also described classification,
Remaining Useful Life (RUL), anomaly-detection, optimizer, scheduler, and metric
behavior as a complete supported system without a maintained config-first smoke
path.

The repository contains research-oriented multi-task components, including an
ISFM multi-task head, but component presence does not prove end-to-end task,
dataset, loss, metric, checkpoint, or benchmark support.

## Current authority

Use these maintained pages instead:

- [Supported components](../SUPPORTED_COMPONENTS.md)
- [Supported combinations](../SUPPORTED_COMBINATIONS.md)
- [Known limitations](../KNOWN_LIMITATIONS.md)
- [Developer guide](developer_guide.md)
- [Task contribution guide](../src/task_factory/contributing.md)
- [Model contribution guide](../src/model_factory/contributing.md)

## Promotion requirements

A future multi-task capability should be documented as maintained only after a
focused PR provides:

- a valid five-block config selected through `main.py --config`;
- an explicit batch and model-output contract for every task;
- loss weighting and metric semantics;
- data split and leakage controls;
- compatible sampler, model, task, and trainer rules;
- focused positive and negative tests;
- checkpoint and artifact behavior;
- a minimum end-to-end smoke run;
- reproducible experiment evidence for any numerical claim.

Until those gates exist, treat multi-task files as experimental research code and
refer to the exact commit when discussing them.

The former detailed page remains recoverable from Git history before this status
note; it is not copied into the maintained documentation because its commands and
metrics were not validated.
