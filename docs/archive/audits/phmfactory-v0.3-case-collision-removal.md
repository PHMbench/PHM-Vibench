# PHMFactory v0.3 case-collision removal audit

## Decision

PHMFactory keeps one canonical spelling for repository-root and authority files so
Linux, macOS, and Windows checkouts resolve the same paths.

The canonical public files are:

```text
CITATION.cff
CONTRIBUTING.md
configs/README.md
src/README.md
```

The following lowercase duplicates or historical compatibility files were removed:

```text
citation.cff
contributing.md
configs/readme.md
src/readme.md
```

## Preservation before removal

Exact source Git blobs were copied to the approved personal-fork archive:

```text
repository:           liq22/PHM-Vibench
branch:               archive/phmfactory-v0.3.0-removals
archive object commit: 05e56ef93c6dd4e5f4c53a95a9f0a1362a38e1dd
path:                 upstream-archive/phmfactory-v0.3.0/case-collisions/tree/
source:               f22409521da2a6001dd620efd3e21180130d1b52
```

| Removed path | Source and archive Git blob |
| --- | --- |
| `citation.cff` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |
| `contributing.md` | `20759cbb8a7fdb5f73ac1d42162e149b8fb3c7bc` |
| `configs/readme.md` | `21b60cc2ec53a00d2eb4b7b8983c9a089702d289` |
| `src/readme.md` | `dfe9d6f442bb55627acf1a50c46dba77ed6b0ab7` |

`SOURCE_BLOB_MANIFEST.tsv` in the private archive records the same identities. The
lowercase citation file was empty. The lowercase contribution file was a compatibility
redirect. The two lowercase README files contained historical notes; their exact
contents remain recoverable from the personal archive and immutable public Git history.

PHMFactory has no runtime, build, test, data, or release dependency on the private
archive.

## Regression prevention

`tools/repo/check_case_collisions.py` examines every tracked file and directory prefix
using Unicode normalization plus case folding. `.github/workflows/repository-layout.yml`
runs this contract on pull requests and pushes to `main`.

The check fails when two tracked paths would become ambiguous on a case-insensitive
filesystem.

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
