# Data Factory Readers

A reader converts one metadata-addressed raw file into a signal array. Reader output is
scientific input: a read failure remains a failure, and an invalid return is rejected
before any HDF5 cache is published.

## Runtime resolution

Each selected metadata row requires:

```text
Id
Name
File
```

The public Data Factory resolves:

```text
raw file:      <data.data_dir>/raw/<Name>/<File>
reader module: src.data_factory.reader.<Name>
```

The module must expose:

```python
def read(file_path, args_data) -> np.ndarray:
    ...
```

Historical readers may accept `*args`, but the factory supplies the resolved raw path and
the current `data` configuration.

## Failure contract

```text
missing raw file
→ FileNotFoundError with Id and resolved path

reader raises a source-format or domain error
→ preserve the same exception type and traceback

any reader failure
→ do not publish a new cache
```

Readers should fail where source fields, channel order, units, or numeric assumptions are
violated. They must not return `None`, an empty array, or substitute data to keep the run
alive.

## Return contract

A successful reader returns a `numpy.ndarray` with:

```text
rank ∈ {1, 2, 3}
all dimensions > 0
real numeric dtype
all values finite
```

New maintained readers should normally return `(L, C)`, where `L` is signal length and
`C` is channel count. Rank-1 and rank-3 arrays remain accepted for compatibility only when
the documented source format requires them.

The Data Factory does not:

- convert lists or arbitrary objects into arrays;
- coerce string/object samples to numbers;
- discard NaN or Inf;
- take real parts of complex signals;
- pad, repeat, copy, or guess channels;
- squeeze an unexpected rank into a supported one.

The historical cache representation remains:

```text
reader rank 1 → cached unchanged
reader rank 2 → singleton axis appended before caching
reader rank 3 → cached unchanged
```

The Dataset layer validates the signal again before windowing. Reader validation prevents
invalid derived data from being published; it does not replace Dataset-level checks.

## Documenting a reader

Record:

- accepted source format and required fields or columns;
- returned shape and channel order;
- dtype and physical units when known;
- truncation, alignment, resampling, normalization, or byte-order handling;
- malformed-input and missing-channel behavior.

Do not change an existing reader's channel order, shape, dtype handling, or numerical
preprocessing in a cleanup PR. Such changes require a focused scientific bug fix with
before/after tests.

## Derived cache flow

```text
raw/<Name>/<File>
        ↓ reader.read(...)
<cache_dir or data_dir>/<Name>.h5, keyed by Id
        ↓ consolidation
<cache_dir or data_dir>/cache.h5, keyed by Id
```

Cache reuse is explicit:

```text
data.use_cache omitted or false
→ execute current readers and rebuild selected data

data.use_cache true
→ reuse complete existing HDF5 entries by selected Id
```

Use `data.use_cache: true` only when raw files, reader code, and reader-relevant
configuration are intentionally unchanged. Matching Id keys alone do not establish that
two cached datasets have the same scientific meaning.

`data.cache_dir` changes only the derived HDF5 location. Metadata and raw files still
resolve from `data.data_dir`.

## Maintained examples

- `Dummy_Data.py` reads the repository-shipped `ch1` and `ch2` columns. Missing or
  malformed fixtures fail; no synthetic substitute is generated.
- `CSV_Signal.py` requires explicit `data.csv_signal_columns` and rejects guessed,
  non-numeric, empty, or non-finite columns.
- `RM_007_MFPT.py` validates the MFPT signal and physical metadata used by the current
  real-data candidate.

File presence alone is not a support claim. Check `SUPPORTED_COMBINATIONS.md` and the
configuration registry for exact maintained combinations.

## Adding a reader

1. Choose a stable metadata `Name`.
2. Add `src/data_factory/reader/<Name>.py`.
3. Implement `read(file_path, args_data) -> np.ndarray`.
4. Place raw files under `<data_dir>/raw/<Name>/<File>`.
5. Add metadata rows using the same `Name` and source `File`.
6. Document source provenance, license, shape, channel order, and preprocessing.
7. Add parsing, output-contract, malformed-input, and channel-order tests.
8. Run the relevant public config and Data Factory tests before claiming support.

Do not hard-code a personal path or modify Model, Task, Trainer, Pipeline, or CLI code to
add a compatible reader.

## Related documentation

- [Data base configuration](../../../configs/base/data/README.md)
- [Data directory and external-source boundary](../../../data/README.md)
- [Data Factory overview](../README.md)
- [Data Factory contribution guide](../contributing.md)
