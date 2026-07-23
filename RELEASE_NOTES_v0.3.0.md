# PHMFactory v0.3.0 Release Notes

> **Draft — unreleased.**
>
> This document describes the intended v0.3.0 release surface. It is not release
> evidence and must not be used to infer that the package, repository rename,
> public data pins, tag, or artifacts have been published.

## Release purpose

v0.3.0 changes PHM-Vibench into PHMFactory as a public, configuration-first PHM
framework while preserving the established runtime core.

The release is primarily about:

- repository ownership boundaries;
- one public package and CLI;
- descriptive Pipeline names;
- one public configuration resolver;
- a bounded dual-provider CWRU demo contract;
- optional dependency ownership;
- removal of duplicate UI and non-framework workspaces;
- repository portability and release governance.

It is not a wholesale reader, model, task, trainer, or Pipeline-algorithm rewrite.

## Public interface

The distribution, import namespace, and executable are all named `phmfactory`:

```python
import phmfactory
```

```bash
python main.py --config <yaml>
python -m phmfactory --config <yaml>
phmfactory --config <yaml>
```

All three command forms use the same parser and dispatcher. Root `main.py`
remains supported.

## Pipeline names

The six established Pipeline files use descriptive canonical names:

```text
Pipeline_01_Fault_Diagnosis
Pipeline_02_Pretraining_Few_Shot
Pipeline_03_Multitask_Pretraining_Finetuning
Pipeline_04_Unified_Evaluation
Pipeline_05_Explainable_Fault_Diagnosis
Pipeline_06_Generative_Modeling
```

Legacy configuration identifiers remain explicit aliases. Direct imports of
old module filenames are a breaking change because no six-file wrapper layer is
introduced.

## Configuration

`phmfactory.config.resolve_config()` is the public configuration entrypoint. It
provides:

- maintained preset and YAML-path resolution;
- ordered recursive `base_configs` composition;
- typed dotted overrides;
- canonical Pipeline selection;
- cycle and missing-source errors;
- resolved path and override provenance.

The mature internal configuration implementation remains in place for v0.3
compatibility. Physical internal consolidation is deferred.

## Data and CWRU quickstart

The maintained CWRU demo bundle contains:

```text
metadata.xlsx       required
RM_001_CWRU.h5      required
corpus.xlsx         optional
```

Provider adapters selectively download only the declared files from Hugging
Face or ModelScope, verify the bundle, and pass a local directory to the
established Data Factory.

The release does not claim new raw-MAT reader validation. The prebuilt HDF5 path
and fully offline `Dummy_Data` smoke are separate validation surfaces.

## Dependency ownership

The root `requirements.txt` remains the core installation authority. Optional
increments live with the subsystem that owns them, including:

```text
apps/streamlit/requirements.txt
phmfactory/data_sources/modelscope/requirements.txt
plot/requirements.txt
test/requirements.txt
```

This prevents Streamlit, ModelScope, test, and plotting dependencies from being
silently treated as mandatory core packages.

## UI consolidation

The maintained optional web workspace is:

```text
apps/streamlit/
```

The legacy duplicate `app/` tree and root `streamlit_app.py` launcher are removed
after exact preservation outside the public framework. The maintained UI calls
the public CLI and does not become a second training implementation.

## Repository ownership cleanup

Personal Agent tooling, development scratchpads, generated result placeholders,
personal submodules, and historical duplicate paths are moved or removed only
after archive and integrity evidence.

Paper submodules are not removed merely because similarly named destination
repositories exist. Each requires content-level mapping and review.

The framework has no runtime, build, test, data, or release dependency on the
personal archive or paper repositories.

## Cross-platform repository layout

Case-colliding compatibility paths are removed in favor of canonical spellings,
and a repository-layout gate rejects future case-insensitive path collisions.
This protects Windows and default macOS checkouts.

## Preserved runtime boundary

v0.3 preserves the mature runtime under `src/`, including the existing dataset
reader path:

```text
src/data_factory/reader/
```

The migration does not intentionally change:

- reader parsing or signatures;
- channel order, shape, or dtype;
- model/task/trainer algorithms;
- data splits or seeds;
- metrics or checkpoint formats.

## Breaking changes

- Project and public Python naming converges on PHMFactory / `phmfactory`.
- Direct imports of the six old Pipeline module filenames must be updated.
- The duplicate root Streamlit launcher and historical `app/` package are removed.
- Personal, Agent, and paper/result workspaces are no longer public framework
  runtime surfaces.
- Case-colliding lowercase compatibility files are removed.

## Migration guide

See [MIGRATION_v0.2_to_v0.3.md](MIGRATION_v0.2_to_v0.3.md).

## Release blockers still open

v0.3.0 must not be released until all of the following are resolved:

1. package and module versions are finalized from `0.3.0.dev0` to `0.3.0`;
2. README and citation metadata use final PHMFactory branding and repository identity;
3. the GitHub repository rename and redirect are verified;
4. Hugging Face and ModelScope bundle revisions are immutable;
5. required CWRU bundle SHA-256 values are populated and cross-provider parity passes;
6. v0.2 release/RC provenance is explicitly resolved;
7. the stacked PR sequence is reviewed and integrated in dependency order;
8. final wheel, source distribution, cross-platform imports, offline smoke, and
   provider release gates pass;
9. the governed optional `phm-data-factory` backend disposition is recorded;
10. the v0.3.0 tag and release artifacts are created only from the approved final commit.

## Evidence boundary

Passing unit, contract, or one-epoch smoke tests demonstrates software wiring. It
does not establish benchmark superiority, universal component compatibility,
or permission to redistribute external datasets.
