---
name: Bug report
about: Report a reproducible PHM-Vibench defect
title: "[BUG] "
labels: bug
assignees: ''
---

## Before submitting

- [ ] I searched existing issues and pull requests.
- [ ] I reproduced the problem on a current `main` checkout or identified the exact release/commit affected.
- [ ] I read [CONTRIBUTING.md](https://github.com/PHMbench/PHM-Vibench/blob/main/CONTRIBUTING.md).
- [ ] This report does not contain credentials, private data, or an undisclosed security vulnerability.

For security-sensitive reports, stop and use [SECURITY.md](https://github.com/PHMbench/PHM-Vibench/blob/main/SECURITY.md).

## Problem

Describe the failure and its impact. Distinguish a code defect from a missing optional dependency, unavailable external dataset, unsupported combination, or documentation problem.

## Minimal reproduction

**Repository commit or tag:**

```text
<git rev-parse HEAD>
```

**Configuration:**

```text
<path under configs/ or attach a minimal YAML>
```

**Command:**

```bash
python main.py --config <yaml> [--override key=value ...]
```

**Steps:**

1.
2.
3.

## Expected behavior

What should have happened?

## Actual behavior

What happened instead? Include the command exit code.

```text
<paste the complete traceback or log as text>
```

## Environment

```text
Operating system:
Python version:
PyTorch version:
PyTorch Lightning version:
CPU/GPU:
CUDA version, if relevant:
Installation method:
```

Helpful commands:

```bash
python --version
python -m pip freeze
```

Attach environment output as a file when it is long. Remove secrets and private paths where possible.

## Data and artifacts

```text
Data source: repository dummy data | external data
Metadata file:
Relevant input shape:
Output or checkpoint path:
```

Do not upload data or model artifacts unless their license permits it. Prefer a small legal fixture or synthetic reproduction.

## Additional context

List workarounds tried, related issues, suspected files, or the last known working commit. Do not describe the issue as a performance regression without comparable commands, data, seeds, and environment evidence.
