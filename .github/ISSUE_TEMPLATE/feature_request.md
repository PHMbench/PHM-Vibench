---
name: Feature request
about: Propose a scoped PHM-Vibench capability or improvement
title: "[FEAT] "
labels: enhancement
assignees: ''
---

## Before submitting

- [ ] I searched existing issues and pull requests.
- [ ] I read [CONTRIBUTING.md](https://github.com/PHMbench/PHM-Vibench/blob/main/CONTRIBUTING.md) and the current support boundaries.
- [ ] I considered whether this belongs in a local experiment, external package, or PHM-Vibench core.

## User or research scenario

Who needs this capability, and what concrete task are they trying to complete?

## Current limitation

What cannot be done with the current config-first entrypoint, factories, maintained configs, or documented extension points?

Include a minimal current command or configuration when possible:

```bash
python main.py --config <yaml> [--override key=value ...]
```

## Proposed behavior

Describe observable behavior rather than only an implementation idea.

```text
Inputs:
Outputs:
Failure behavior:
Configuration keys:
```

State the intended maturity:

- [ ] Maintained public capability
- [ ] Experimental capability
- [ ] Research-only prototype
- [ ] Documentation or tooling improvement

## Simpler alternatives

What smaller workaround, configuration change, factory extension, or external tool did you consider? Why is it insufficient?

## Architecture and compatibility impact

```text
Affected factory or module:
New dependencies:
CLI/config compatibility:
Checkpoint/data compatibility:
CPU/GPU implications:
Migration or deprecation needs:
```

Do not propose component-specific branches in `main.py` when an existing factory can express the capability.

## Evidence and validation plan

What tests, fixtures, configs, smoke commands, artifacts, or benchmark protocol would demonstrate that the feature works?

For a model or algorithm proposal, link a primary paper or stable specification and identify any source-code license. A paper reference alone is not runtime evidence.

## Maintenance cost and risks

Describe likely ownership, optional dependency handling, data or licensing constraints, failure modes, and long-term documentation/test burden.

## Additional context

Add diagrams, examples, or related projects only when they clarify the user problem or compatibility boundary.
