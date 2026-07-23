# Data Factory Readers

Dataset reader modules convert one metadata-addressed raw file into a NumPy
signal array. This page documents the behavior implemented by the current data
factory; it is not a redesign of the reader runtime.

## Current runtime contract

The maintained data factory reads each metadata row's `Name` field and imports:

```text
src.data_factory.reader.<Name>
```

The imported module must expose a callable compatible with:

```python
def read(file_path, args_data):
    """Return one sample as a NumPy array."""
```

Several historical readers accept `*args` rather than naming `args_data`, but
the current factory calls every reader with both `file_path` and `args_data`.
A new reader should therefore accept those two arguments explicitly unless a
reviewed compatibility reason requires a broader signature.

Reader modules are resolved by module name. They are **not** currently
registered as `BaseReader` subclasses, and `reader/__init__.py` is not a reader
registry.

## Metadata and raw-file resolution

The factory requires each selected metadata row to provide at least:

```text
Id
Name
File
```

It constructs the raw path as:

```text
<data.data_dir>/raw/<Name>/<File>
```

Example:

```text
metadata Name: RM_001_CWRU
metadata File: 98.mat
resolved path: <data_dir>/raw/RM_001_CWRU/98.mat
reader module: src.data_factory.reader.RM_001_CWRU
```

Other metadata fields such as labels, domains, sampling rate, sample length,
channel count, and task flags are consumed by metadata selection and task
adapters rather than being returned in a reader dictionary.

## Signal return value

A reader returns a NumPy signal array, normally with shape:

```text
(L, C)
```

where `L` is signal length and `C` is channel count. The current data factory
expands a two-dimensional reader result before writing its runtime cache, so
reader implementations must not add an extra singleton dimension merely to
anticipate that factory behavior.

Document for every maintained reader:

- accepted source format and required source keys or columns;
- returned shape and channel ordering;
- dtype and physical units when known;
- truncation, alignment, resampling, normalization, or byte-order handling;
- missing-channel and malformed-input behavior.

Do not change an existing reader's channel order, shape, dtype handling, or
numeric preprocessing in a repository-cleanup PR.

## Cache flow

The current factory uses two cache levels:

```text
raw/<Name>/<File>
        ↓ reader.read(...)
<Name>.h5, keyed by Id
        ↓ consolidation
cache.h5, keyed by Id
```

If an `Id` already exists in `<Name>.h5`, the raw reader can be skipped. The
final `cache.h5` contains the IDs selected for the run and is exposed through
`H5DataDict`.

Reader code must not assume that every training run will execute the raw-file
conversion path; a compatible prebuilt `<Name>.h5` cache may be used instead.

## Existing readers and support status

Files under this directory include dataset-specific readers such as
`RM_001_CWRU.py`, `RM_002_XJTU.py`, and `RM_003_FEMTO.py`, plus the offline
`Dummy_Data.py` reader. File presence alone does not establish maintained or
benchmark-supported status. Support claims must be backed by the maintained
configuration registry, focused tests, a runnable demo where applicable, and
explicit limitations.

`Dummy_Data` can generate deterministic synthetic signals when raw files are
absent. It remains the fully offline smoke path used by:

```text
configs/demo/00_smoke/dummy_dg.yaml
```

Historical or placeholder files are retained until a separate inventory proves
their consumers and disposition; they must not be deleted merely because their
names overlap with another reader.

## Adding a reader

1. Choose a stable metadata `Name`.
2. Add `src/data_factory/reader/<Name>.py`.
3. Implement `read(file_path, args_data) -> np.ndarray`.
4. Place local raw files under `<data_dir>/raw/<Name>/<File>`.
5. Add or validate metadata rows using the same `Name` and source `File`.
6. Document source provenance, license, signal contract, and preprocessing.
7. Add focused parsing, shape, dtype, channel-order, and malformed-input tests.
8. Run the relevant config inspection and smoke gates before claiming support.

Do not hard-code a personal data directory in the reader's maintained execution
path. Local paths belong in `configs/local/local.yaml` or CLI overrides.

See [the data-factory contribution guide](../contributing.md) for evidence,
licensing, configuration, and test requirements.

## PHMFactory v0.3 preservation boundary

The v0.3 repository cleanup preserves reader module paths and runtime behavior.
The protected contract is recorded in
[`docs/PHMFACTORY_V0_3_READER_PRESERVATION.md`](../../../docs/PHMFACTORY_V0_3_READER_PRESERVATION.md).

Changes to reader parsing or numerical behavior require a separate bugfix or
feature PR with before/after evidence. Documentation cleanup does not authorize
runtime refactoring.

## Related documentation

- [Data directory and external-data boundary](../../../data/README.md)
- [Data factory overview](../README.md)
- [Data factory contribution guide](../contributing.md)
- [Configuration guide](../../../configs/README.md)
