# Data Factory Readers

Dataset reader modules convert one metadata-addressed raw file into a signal array. The
maintained public Data Factory treats the reader result as scientific input: a reader
failure remains a failure, and an invalid successful return is rejected before any HDF5
cache is published.

## Runtime resolution

For each selected metadata row, the public Data Factory requires:

```text
Id
Name
File
```

It resolves:

```text
raw file:
<data.data_dir>/raw/<Name>/<File>

reader module:
src.data_factory.reader.<Name>
```

The imported module must expose a callable compatible with:

```python
def read(file_path, args_data) -> np.ndarray:
    ...
```

Historical readers may accept `*args`, but the factory always supplies both the resolved
raw path and the current `data` configuration.

## Failure contract

The maintained public path does not replace reader exceptions:

```text
missing declared raw file
→ FileNotFoundError with Id and resolved path

reader raises ValueError / FloatingPointError / domain-specific error
→ the same exception type and traceback escape Data Factory

any reader failure
→ no new dataset cache is published
```

Readers should raise at the point where source format, required fields, channel order,
units, or numeric assumptions are violated. They must not return `None`, an empty signal,
or a substitute signal to keep the experiment running.

## Successful return contract

A successful reader must return a `numpy.ndarray` satisfying:

```text
rank ∈ {1, 2, 3}
all dimensions > 0
real numeric dtype
all values finite
```

New maintained readers should normally return:

```text
(L, C)
```

where `L` is signal length and `C` is channel count. Rank-1 and rank-3 arrays remain
accepted for existing reader compatibility, but a new reader should use them only when
its documented source semantics require that representation.

The Data Factory does not:

- convert Python lists or arbitrary objects into arrays;
- coerce string/object samples to numbers;
- discard NaN or Inf;
- take real parts of complex signals;
- pad, repeat, copy, or guess channels;
- squeeze an unexpected rank into a supported one.

The historical cache representation is preserved:

```text
reader rank 1 → cached unchanged
reader rank 2 → singleton axis appended before caching
reader rank 3 → cached unchanged
```

The Dataset layer later validates the signal again before windowing. Early reader-output
validation exists to prevent invalid derived data from being published, not to replace
Dataset-level window and preprocessing checks.

## Reader documentation

Document for every maintained reader:

- accepted source format and required source fields or columns;
- returned shape and channel ordering;
- dtype and physical units when known;
- truncation, alignment, resampling, normalization, or byte-order handling;
- malformed-input and missing-channel behavior.

Do not change an existing reader's channel order, shape, dtype handling, or numerical
preprocessing in a cleanup PR. Such changes require a focused scientific bugfix with
before/after tests.

## Derived cache flow

The public path uses two derived HDF5 levels:

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
→ execute the current readers and rebuild selected data

data.use_cache true
→ reuse complete existing HDF5 entries by selected Id
```

Set `data.use_cache: true` only when raw files, reader code, and reader-relevant
configuration are intentionally unchanged. The factory does not infer semantic
equivalence from matching Id keys and does not add cache hashes or provenance machinery.

`data.cache_dir` changes only the location of derived HDF5 files. Metadata and raw files
continue to resolve from `data.data_dir`.

## Current maintained examples

- `Dummy_Data.py` strictly reads the repository-shipped `ch1` and `ch2` columns. Missing
  or malformed fixtures fail; no synthetic fallback is generated.
- `CSV_Signal.py` requires explicit `data.csv_signal_columns` and rejects guessed,
  non-numeric, empty, or non-finite columns.
- `RM_007_MFPT.py` validates the public MFPT signal and physical metadata used by the
  current real-data reference.

File presence alone is not a support claim. Consult `SUPPORTED_COMBINATIONS.md` and the
configuration registry for the exact execution-verified or baseline-valid combinations.

## Adding a reader

1. Choose a stable metadata `Name`.
2. Add `src/data_factory/reader/<Name>.py`.
3. Implement `read(file_path, args_data) -> np.ndarray`.
4. Place local raw files under `<data_dir>/raw/<Name>/<File>`.
5. Add metadata rows using the same `Name` and source `File`.
6. Document source provenance, license, signal contract, channel order, and
   preprocessing.
7. Add focused parsing, output-contract, malformed-input, and channel-order tests.
8. Run the relevant public config and Data Factory gates before making a support claim.

Do not hard-code a personal path or modify Model, Task, Trainer, Pipeline, or CLI code to
add a compatible reader.

## Related documentation

- [Data base configuration](../../../configs/base/data/README.md)
- [Data directory and external-source boundary](../../../data/README.md)
- [Data Factory overview](../README.md)
- [Data Factory contribution guide](../contributing.md)
- [v0.3 reader preservation boundary](../../../docs/PHMFACTORY_V0_3_READER_PRESERVATION.md)
