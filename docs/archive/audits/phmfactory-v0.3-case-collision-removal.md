# PHMFactory v0.3 case-collision removal audit

## Decision

PHMFactory keeps one canonical spelling for tracked authority and module-document
paths so Linux, macOS, and Windows checkouts resolve the same files.

Canonical public files:

```text
CITATION.cff
CONTRIBUTING.md
configs/README.md
src/README.md
src/task_factory/README.md
src/trainer_factory/README.md
src/utils/README.md
```

Removed lowercase duplicate or historical compatibility files:

```text
citation.cff
contributing.md
configs/readme.md
src/readme.md
src/task_factory/readme.md
src/trainer_factory/readme.md
src/utils/readme.md
```

## Preservation before removal

Exact source Git blobs were copied to the approved personal-fork archive:

```text
repository:            liq22/PHM-Vibench
branch:                archive/phmfactory-v0.3.0-removals
archive update commit: a8bcee98bddcc2ef5bfbeddfafc82af665dc623b
path:                  upstream-archive/phmfactory-v0.3.0/case-collisions/tree/
source:                f22409521da2a6001dd620efd3e21180130d1b52
```

| Removed path | Source and archive Git blob |
| --- | --- |
| `citation.cff` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |
| `contributing.md` | `20759cbb8a7fdb5f73ac1d42162e149b8fb3c7bc` |
| `configs/readme.md` | `21b60cc2ec53a00d2eb4b7b8983c9a089702d289` |
| `src/readme.md` | `dfe9d6f442bb55627acf1a50c46dba77ed6b0ab7` |
| `src/task_factory/readme.md` | `35edc18aff65ff0415836f19afaf7c4e799a6816` |
| `src/trainer_factory/readme.md` | `e7d3c72a64310754b7d492959f7299605f66ce8c` |
| `src/utils/readme.md` | `84c5269f619d5d0c9daa23c941c098d731c00ed2` |

`SOURCE_BLOB_MANIFEST.tsv` in the private archive records the same identities. The
lowercase citation file was empty. The lowercase contribution file and the three
nested module files were compatibility redirects. The lowercase configuration and
source README files contained historical notes; their exact contents remain
recoverable from the personal archive and immutable public Git history.

PHMFactory has no runtime, build, test, data, or release dependency on the private
archive.

## Regression prevention

`tools/repo/check_case_collisions.py` examines every tracked file and directory prefix
using Unicode NFC normalization plus case folding.
`.github/workflows/repository-layout.yml` runs this contract on pull requests and
pushes to `main`.

The check fails when two tracked paths would become ambiguous on a case-insensitive
filesystem. Focused tests cover file-name, directory-prefix, Unicode-normalization,
and non-colliding cases.

## Historical directories intentionally retained

This change does not remove or reinterpret:

```text
docs/past/
docs/v0.1.0/
configs/v0.0.9/
reports/
plot/
```

Those paths require separate reference, provenance, and ownership review. Removing a
case duplicate does not authorize deletion of a historical directory.

## Protected boundary

No dataset reader, Data Factory, model, task, trainer, Pipeline algorithm,
configuration behavior, CWRU bundle, dependency set, or Streamlit runtime is changed.

## Rollback

A normal revert restores the lowercase paths. Exact Git blobs remain in both Git
history and the personal-fork archive.
