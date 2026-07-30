# PHMFactory v0.3 Legacy UI Removal Audit

## Decision

PHMFactory v0.3 maintains one optional web workspace:

```text
apps/streamlit/
```

The historical `app/` prototype and root `streamlit_app.py` launcher are removed
from the public upstream after exact preservation in the user's personal fork.

## Public source

```text
repository: PHMbench/PHM-Vibench
branch:     agent/v030-ui-consolidation-pr09
source SHA: 00ec30bab7ed92a5142d7954e999bbe654e31fa5
```

Source files and Git blobs:

| Source path | Git blob |
| --- | --- |
| `app/README.md` | `2d30302ba6a61bdcaefea26a6294801f8de994d5` |
| `app/__init__.py` | `3628a6c14cc25e45d21efb0ed59193bce15a97f9` |
| `app/gui.py` | `bcccc44b3afa2fd9fc04e29cdcf679333d1a1090` |
| `app/gui_refactored.py` | `3cc0aad2c30d4cd3ff20d2d511a2687e4ef2f11e` |
| `app/layout.py` | `3df02d45ed68b2de79e3e352cac8ce647ef9e4a4` |
| `app/pipeline.py` | `5aa2b6b918ea0ddceba86f72361e2d27a2b06d3c` |
| `app/requirements_gui.txt` | `32cba6fcb6ea89d12b0f9a64d20071824ffafd49` |
| `app/state.py` | `7c8742e84d936c8a46e9e2a417708e4f824ed3ac` |
| `app/utils.py` | `e839ed9be3ffcbf7edd3b289cdca2cede5cfd1c0` |
| `streamlit_app.py` | `2a8361efa7a984fabacd0008b768af07f7f2b1ec` |

## Personal-fork preservation

```text
repository:   liq22/PHM-Vibench
branch:       archive/phmfactory-v0.3.0-removals
verified head: cabd9948ccb531ef993fa81f5c14463063aa307a
archive root: upstream-archive/phmfactory-v0.3.0/legacy-ui/
```

The durable private archive contains:

```text
ARCHIVE_README.md
SOURCE_BLOB_MANIFEST.tsv
tree/app/**
tree/streamlit_app.py
```

A one-shot private workflow fetched the fixed public source commit, reconstructed
all ten files with `git archive`, recalculated each Git blob, required 10/10 exact
matches, committed only the archive root, and removed itself.

Manual connector reads subsequently confirmed all ten destination blob SHAs match
the source table above.

## Supplementary public artifact

Before deletion, a public repository workflow also exported the exact source tree,
capability inventory, reference inventory, and blob manifest:

```text
artifact: phmfactory-v030-legacy-ui
artifact id: 8554784189
size: 31792 bytes
SHA-256: 4cf870585588c6f5af4c2a71cb95849b19a1d335ae2dd018f3aa040233246897
```

The artifact is supplementary and expires. The personal Git archive and source
history are the durable recovery mechanisms.

## Capability review

The removed `app/` tree is an independent prototype implementation with direct
Streamlit, data loading, plotting, process control, and large experimental GUI
modules. The maintained workspace already owns the supported architecture:

```text
apps/streamlit/app.py
apps/streamlit/config_service.py
apps/streamlit/run_service.py
apps/streamlit/result_service.py
apps/streamlit/runtime_policy.py
apps/streamlit/onboarding.py
```

The maintained workspace delegates experiments to the public CLI rather than
keeping a second training implementation. No function or module from `app/` is
imported into the maintained runtime.

## Public changes

The removal PR:

- deletes `app/**`;
- deletes root `streamlit_app.py`;
- keeps `apps/streamlit/app.py` as the sole deployment entrypoint;
- updates Linux/Windows Streamlit CI;
- updates the UI import contract;
- updates maintained Streamlit documentation and dependency ownership;
- removes temporary archive/deletion workflows from the final tree.

## Explicit non-impact

No reader, Data Factory, model, task, trainer, Pipeline algorithm, configuration
schema, CWRU bundle, paper/result workspace, or submodule is changed.

## Rollback

A normal revert restores the public files. Exact copies remain in the personal
fork archive and the immutable public Git history.
