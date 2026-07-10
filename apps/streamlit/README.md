# PHM-Vibench Streamlit Experiment Console

This directory contains an **experimental, optional** user interface for the
configuration-first PHM-Vibench workflow.

The UI is not a second training framework. Its boundary is:

```text
validated template selection
+ safe CLI overrides
+ core CLI execution
+ result discovery
```

The only experiment entry point remains:

```bash
python main.py --config <yaml> [--override key=value ...]
```

The application does not import or call a Pipeline function directly, does not
modify `main.py`, and does not change the five-block configuration schema.

## Install

Install the repository's core environment first, then the optional UI layer:

```bash
pip install -r requirements.txt
pip install -r apps/streamlit/requirements.txt
```

The app requires Streamlit 1.37 or newer because the run-monitoring layer uses
`st.fragment` for independent status refreshes. Streamlit 1.37 made this API
generally available.

## Start

Run from the repository root:

```bash
streamlit run apps/streamlit/app.py
```

For the first run, keep the defaults:

```text
Template: demo_00_smoke_dummy_dg
Device:   cpu
Epochs:   1
```

The corresponding core command is:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.device=cpu \
  --override trainer.num_epochs=1
```

The UI delegates authoritative resolution and validation to:

```bash
python -m scripts.config_inspect --config <yaml> --dump all --format json
```

## Modes

### Quick Start

Exposes only the selected registry template, device, and epoch count. This is the
recommended onboarding path.

### Advanced

Provides:

- catalog-defined safe fields with legacy key aliases;
- an editable, portable YAML resolved before machine-local overrides;
- one `key=value` CLI override per line;
- a unified configuration diff;
- the exact reproduction command.

Raw overrides are passed as `subprocess` argv elements. They are never composed
into a `shell=True` command.

The execution precedence is explicit:

```text
portable YAML
< configs/local/local.yaml (when present)
< catalog-safe CLI overrides
< raw CLI overrides
```

The portable YAML never bakes in machine-local values. Normal validation and
execution therefore let the core apply the local layer exactly once.

## Compatibility boundary

- The registry is the source of template identity and status.
- `field_catalog.yaml` is the source of editable keys, aliases, widget metadata,
  and template groups.
- New registry columns are retained as metadata without requiring an immediate
  code change.
- Key migrations should normally add a path alias in `field_catalog.yaml` rather
  than add model-specific conditionals to `app.py`.
- Machine-specific paths should be provided through advanced overrides or
  `configs/local/local.yaml`; they must not be committed into maintained demos.

## Current staged scope

PR-S1 provides template browsing, safe editing, config inspection, YAML download,
and CLI preview. The stacked PR-S2 adds:

- start/cancel/restart controls;
- cross-platform process-group termination;
- live log polling through `st.fragment`;
- run manifests and bounded result discovery;
- CSV/JSON metrics and image artifact display.

## Validation

```bash
python -m py_compile apps/streamlit/*.py
python -m pytest test/test_streamlit_config_service.py
python -m scripts.validate_configs
python -m scripts.validate_docs
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

## Troubleshooting

### Inspector reports a missing Python module

The UI dependencies do not replace the core benchmark dependencies. Install the
core requirements and confirm the smoke command works in a terminal.

### Data path does not exist

Choose the repository-shipped dummy smoke template, edit `data.data_dir` in
Advanced mode, or configure the machine-local override file.

### GPU run fails

Return to `trainer.device=cpu` and validate the environment independently before
selecting CUDA.

### No structured metrics are found

A completed core run can still be inspected through its raw log and output
folder. Result discovery is tolerant of missing optional artifacts.
