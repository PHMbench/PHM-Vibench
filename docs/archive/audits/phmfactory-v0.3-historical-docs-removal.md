# PHMFactory v0.3 historical documentation migration

## Completed scope

The following public historical documentation trees are removed from the maintained
framework branch:

```text
docs/past/
docs/v0.1.0/
```

The atomic migration commit is:

```text
5726364936cdd75b8b02da5be55cc93ef208df29
```

They contain 22 tracked documents from earlier development, migration, and release
planning cycles. They are valuable as provenance but are not current installation,
configuration, extension, or support guidance.

## Preservation before removal

Exact source bytes were copied to the approved personal-fork archive before public
removal:

```text
repository: liq22/PHM-Vibench
branch:     archive/phmfactory-v0.3.0-removals
path:       upstream-archive/phmfactory-v0.3.0/history-case/tree/
source:     PHMbench/PHM-Vibench@f22409521da2a6001dd620efd3e21180130d1b52
```

`SOURCE_BLOB_MANIFEST.tsv` and `ARCHIVED_FILE_MANIFEST.tsv` record each source path,
Git blob SHA, byte count, and SHA-256. The private archive has no runtime, build,
test, data, or release role in PHMFactory.

The original long-form `src/configs/README.md` was also preserved separately before
its maintained public compatibility documentation was reduced:

```text
source blob: 3b7c1f414507a750c862c98fd561bbc4a57304a5
verification: PASS
```

## Canonical documentation retained and updated

- `docs/index.md` remains the maintained documentation entrypoint;
- `docs/archive/README.md` defines the public provenance and audit policy;
- `src/README.md` absorbs the still-accurate runtime architecture and extension map;
- `src/configs/README.md` documents the protected compatibility layer without stale
  v0.1 paths or unsupported performance claims;
- `SUPPORTED_COMPONENTS.md` continues to define evidence-backed support boundaries;
- `configs/v0.0.9/local/README.md` links to canonical configuration guidance.

Historical audit files under `docs/archive/audits/` may continue to name the removed
paths as immutable evidence. They are not current user guidance.

## Explicit retention blocker

```text
configs/v0.0.9/
```

is not removed. The inventory found 161 reference lines, including active mappings in
the protected compatibility loader. Physical deletion requires a separate migration
that reduces runtime references to zero and supplies downstream compatibility
evidence.

The following are also outside this PR:

```text
reports/
plot/
eight paper/research submodules
```

## Protected boundary

No dataset reader, Data Factory implementation, model, task, trainer, Pipeline
algorithm, configuration runtime, CWRU bundle, dependency owner, Streamlit runtime,
report, plotting utility, or paper gitlink is changed.

## Rollback

A normal revert restores both historical trees and the prior canonical documentation.
Exact source files remain independently recoverable from public Git history and the
verified personal-fork archive.
