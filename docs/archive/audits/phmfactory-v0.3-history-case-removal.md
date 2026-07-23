# PHMFactory v0.3 historical-path and case-collision removal

## Completed scope

This audit covers removal of the following historical or case-colliding public paths:

```text
docs/past/
docs/v0.1.0/
citation.cff
contributing.md
configs/readme.md
src/readme.md
src/task_factory/readme.md
src/trainer_factory/readme.md
src/utils/readme.md
```

The atomic cleanup commit is:

```text
f87a925ddb2517a547a7c1c810413e927f4fe68b
```

Canonical paths retained:

```text
CITATION.cff
CONTRIBUTING.md
configs/README.md
src/README.md
src/task_factory/README.md
src/trainer_factory/README.md
src/utils/README.md
```

## Private preservation

Exact source bytes are stored in the approved personal fork:

```text
repository: liq22/PHM-Vibench
branch:     archive/phmfactory-v0.3.0-removals
path:       upstream-archive/phmfactory-v0.3.0/history-case/
source:     PHMbench/PHM-Vibench@f22409521da2a6001dd620efd3e21180130d1b52
```

The primary archive contains 26 verified candidate blobs. Three supplemental
lowercase README files were copied and independently verified, for 29 archived blobs
in total. `SOURCE_BLOB_MANIFEST.tsv`, `ARCHIVED_FILE_MANIFEST.tsv`,
`SUPPLEMENTAL_CASE_COLLISION_MANIFEST.tsv`, and the recorded Git blob identities
provide restoration evidence.

The previous long-form `src/configs/README.md` was also preserved separately before
its public compatibility documentation was reduced:

```text
source blob: 3b7c1f414507a750c862c98fd561bbc4a57304a5
verification: PASS
```

## Findings

The inventory identified seven case-insensitive path-collision groups:

```text
CITATION.cff / citation.cff
CONTRIBUTING.md / contributing.md
configs/README.md / configs/readme.md
src/README.md / src/readme.md
src/task_factory/README.md / src/task_factory/readme.md
src/trainer_factory/README.md / src/trainer_factory/readme.md
src/utils/README.md / src/utils/readme.md
```

The lowercase member of each group was removed, leaving one canonical path per
group. It also recorded:

```text
external reference lines requiring classification: 53
configs/v0.0.9 reference lines:                    161
```

Historical audit records and the immutable v0.3 baseline generator may continue to
name removed paths as evidence. Maintained navigation and compatibility documentation
were updated to canonical public locations.

## Explicit retention

`configs/v0.0.9/` is not removed in this PR. The protected compatibility loader and
historical presets still reference it. Physical deletion requires a separate change
that reduces runtime references to zero and provides downstream compatibility
evidence.

## Canonical documentation changes

- `src/README.md` now contains the accurate runtime architecture and extension map;
- `src/configs/README.md` now documents the legacy compatibility boundary and points
  new integrations to `phmfactory.config` and `configs/README.md`;
- `docs/index.md` and `docs/archive/README.md` point historical governance to
  `docs/archive/`, Git history, and the approved private archive;
- `SUPPORTED_COMPONENTS.md` no longer names a removed public history directory;
- `configs/v0.0.9/local/README.md` links to the canonical configuration authority.

## Protected boundary

No reader, Data Factory, model, task, trainer, Pipeline algorithm, configuration
runtime, CWRU bundle, dependency owner, Streamlit implementation, report, plotting
tool, or paper submodule is changed by this cleanup.

## Rollback

A normal revert restores every removed path and prior documentation. Exact source
content is independently recoverable from public Git history and the verified
personal-fork archive.
