# PHMFactory v0.3 CWRU Demo Contract

## Scope

The v0.3 quickstart uses a prebuilt, provider-neutral local bundle and then hands
its directory to the existing Data Factory. It does not modify
`src/data_factory/reader/RM_001_CWRU.py`, rebuild raw MAT files, or replace the
existing `<Name>.h5 -> cache.h5` runtime path.

## Bundle files

Required:

```text
metadata.xlsx
RM_001_CWRU.h5
```

Optional:

```text
corpus.xlsx
```

The metadata and HDF5 files are joined by `Id`. For rows whose `Name` is
`RM_001_CWRU`, every metadata Id must exist as an HDF5 key. Each signal dataset
must be two-dimensional with shape `(L, C)`, matching the accepted metadata
aliases for sample length and channel count.

`corpus.xlsx` is not required for the v0.3 fault-diagnosis demo. When present,
its Id values must be a subset of metadata Id values. Explainable or generative
workflows that require text remain responsible for rejecting a missing corpus.

## Public providers

The bundled manifest declares two explicit sources:

- Hugging Face: `PHMbench/PHM-Vibench`
- ModelScope: `PHMbench/PHM-Vibench`

The provider layer downloads only the declared filenames. It does not download
the whole dataset repository and does not perform network access from a DataLoader
or reader.

Development commands currently use the provider branch names recorded in the
manifest. These floating revisions are not sufficient for a v0.3.0 release. The
release gate must publish the same bundle to both services, replace both revisions
with immutable revisions, populate expected SHA-256 values, and prove provider
parity.

## Installation

The root requirements include the default Hugging Face provider:

```bash
python -m pip install -r requirements.txt
```

ModelScope is optional and is installed from its owning subsystem:

```bash
python -m pip install -r requirements.txt
python -m pip install -r phmfactory/data_sources/modelscope/requirements.txt
```

## CLI

Hugging Face:

```bash
python main.py data download \
  --bundle cwru-demo-v1 \
  --source huggingface
```

ModelScope:

```bash
python main.py data download \
  --bundle cwru-demo-v1 \
  --source modelscope
```

Validate an existing local bundle:

```bash
python main.py data validate \
  --bundle cwru-demo-v1 \
  --path ~/.cache/phmfactory/cwru-demo-v1
```

Compare independently downloaded provider materializations:

```bash
python main.py data compare \
  --left /tmp/cwru-huggingface \
  --right /tmp/cwru-modelscope
```

All commands are also available through `python -m phmfactory` and the installed
`phmfactory` executable.

## Minimal experiment

```bash
python examples/cwru_quickstart.py --source huggingface
```

The example downloads and validates the bundle, then runs the maintained CWRU
cross-domain configuration for one CPU epoch with a small window count. Switching
providers changes only `--source` after the corresponding provider dependency is
installed.

## Validation levels

Every pull request runs offline contract tests with a generated metadata workbook
and HDF5 cache. Those tests cover:

- required files;
- optional corpus behavior;
- metadata/HDF5 Id integrity;
- `(L, C)` shape and metadata consistency;
- provider command construction;
- local provider-hash parity;
- `phmfactory data validate`.

Online Hugging Face and ModelScope downloads are a nightly/release gate after the
versioned remote bundle is published. The prebuilt-HDF5 path does not claim to
revalidate raw MAT conversion in v0.3.0.
