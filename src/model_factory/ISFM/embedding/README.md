# ISFM Embedding Components

This directory contains the embedding stage used by Industrial Signal Foundation Model (ISFM) configurations. The embedding converts input signals into the representation consumed by an ISFM backbone.

## Source of truth

Use these files as the maintained references:

- [`../README.md`](../README.md): ISFM configuration and component overview.
- [`../isfm_components.csv`](../isfm_components.csv): machine-readable component inventory, module paths, key arguments, and recorded test status.

The presence of a Python file alone does not imply release support. Maintained combinations are defined by the repository's public configs and release documentation.

## Registered embedding IDs

| Component ID | Implementation | Purpose |
| --- | --- | --- |
| `E_01_HSE` | `E_01_HSE.py` | Hierarchical signal embedding for time-domain patches. |
| `E_01_HSE_abalation` | `E_01_HSE.py` | Configurable HSE ablation variant. |
| `E_01_HTFE` | `E_01_HTFE.py` | Time-frequency embedding. |
| `E_02_HSE_v2` | `E_02_HSE_rec.py` | Reconstruction-capable, system-aware HSE variant. |
| `E_03_Patch` | `E_03_Patch.py` | Basic patch embedding baseline. |
| `E_com_00_PE` | `E_com_00_PE.py` | Common positional or temporal embedding utility. |

Exact constructor arguments and module paths are recorded in `../isfm_components.csv`. Runtime metadata requirements, such as sampling-rate or system identifiers, are documented in `../README.md` and the selected configuration.

## HSE runtime contract

`E_01_HSE` consumes signals with shape `[B, L, C]`.

- Training mode samples valid patch starts randomly.
- Evaluation mode uses a deterministic, evenly spaced patch grid.
- Explicit `start_indices_L` and `start_indices_C` may be supplied together by a controlled evaluator.
- `patch_size_L > L` is invalid and fails before feature construction.
- `patch_size_C > C` is invalid and fails before feature construction.
- HSE never repeats or pads the time axis and never duplicates or pads channels to satisfy a patch request.

This distinction keeps training stochastic while ensuring repeated validation and test passes evaluate the same finite input representation.

## Adding or changing an embedding

A public embedding change should keep the following surfaces synchronized:

1. implementation under this directory;
2. component entry in `../isfm_components.csv`;
3. the embedding section in `../README.md`;
4. a focused import, assembly, or forward test when runtime behavior changes;
5. any maintained config or generated documentation that exposes the component.

Documentation-only edits should run the repository documentation and configuration validators. Runtime edits should additionally run the focused model tests and an applicable maintained smoke configuration.

## Related documentation

- [`../README.md`](../README.md) — ISFM overview
- [`../backbone/README.md`](../backbone/README.md) — backbone components
- [`../task_head/README.md`](../task_head/README.md) — task-head components
- [`../../contributing.md`](../../contributing.md) — model-factory contribution guide
