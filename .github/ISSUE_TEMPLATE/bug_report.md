---
name: Bug report
about: Report a reproducible PHM-Vibench defect
title: "[BUG] "
labels: bug
assignees: ''
---

## Prerequisites

- [ ] I searched existing [issues](https://github.com/PHMbench/PHM-Vibench/issues).
- [ ] I reproduced the problem on a current commit or stated release tag.
- [ ] I read the [contributor guide](../../CONTRIBUTING.md).
- [ ] This is not a security vulnerability. Security reports follow [SECURITY.md](../../SECURITY.md).

## Problem

Describe the defect and its impact. State whether it affects configuration,
data, model, task, trainer, CLI, Streamlit, checkpoint, or artifact behavior.

## Reproduction

1. Repository commit or tag:
2. Config file:
3. CLI overrides:
4. Data source/fixture:
5. Exact command:
6. Stable reproduction steps:

Use the maintained entrypoint, for example:

```bash
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Attach the smallest legal config/data fixture that reproduces the problem.
Remove credentials, private data, and machine-specific secrets.

## Expected behavior

Describe the expected output, state transition, error, metric, or artifact.

## Actual behavior

Include the exit code and complete traceback/log.

```text
paste log here
```

## Environment

```text
Operating system:
CPU/GPU:
Python:
PyTorch:
CUDA runtime/driver:
PyTorch Lightning:
Other relevant packages:
```

Helpful commands:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
```

## Additional evidence

Include screenshots, output-tree excerpts, checkpoints, or related issues only
when they help reproduce or isolate the defect.
