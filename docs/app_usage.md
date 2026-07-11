# Streamlit Experiment Workspace

PHM-Vibench provides an optional browser-based workspace for users who prefer a
guided experiment flow over editing YAML and invoking the CLI manually.

The workspace does not replace the maintained command:

```bash
python main.py --config <yaml> [--override key=value ...]
```

It selects a validated template, creates a portable YAML snapshot, launches that
command, and reads the resulting logs and artifacts.

## Install

```bash
pip install -r requirements.txt
pip install -r apps/streamlit/requirements.txt
```

## Start

Run from the repository root:

```bash
streamlit run apps/streamlit/app.py
```

The historical command remains compatible and routes to the same workspace:

```bash
streamlit run streamlit_app.py
```

For the first experiment, use:

```text
Template: demo_00_smoke_dummy_dg
Device: cpu
Epochs: 1
```

This template uses repository-shipped dummy data and requires no dataset download.

## Workflow

### 1. Select a template

Templates come from `configs/config_registry.csv`. The UI shows only entries
selected by the declarative groups in `apps/streamlit/field_catalog.yaml`.

### 2. Configure

Quick Start exposes a minimal safe parameter surface. Advanced mode adds the
portable YAML editor and typed CLI overrides.

### 3. Validate and run

The UI delegates validation to:

```bash
python -m scripts.config_inspect --config <yaml> --dump all --format json
```

The Run button is enabled only for the exact configuration signature that passed
validation. Execution uses an argv list and `shell=False`.

### 4. Inspect evidence

The live run area contains:

- process status, PID, elapsed time, and exit code;
- cancel and immutable restart controls;
- actual reproduction command;
- headline metrics and CSV/JSON tables;
- image and file artifacts;
- live log tail and full log download.

Every run writes:

```text
outputs/streamlit/<run_id>/execution.yaml
outputs/streamlit/<run_id>/run.json
outputs/streamlit/<run_id>/run.log
```

## Configuration precedence

```text
portable YAML
< configs/local/local.yaml
< safe UI overrides
< raw Advanced overrides
```

Machine-specific paths should remain in `configs/local/local.yaml` or Advanced
overrides, not in maintained demo configurations.

## Operational limits

The optional UI is intentionally a local single-worker experiment console, not a
multi-user scheduler. It manages one active process at a time, does not implement
pause/resume, and applies bounded artifact scanning.

See `apps/streamlit/README.md` for architecture, compatibility, testing, and
troubleshooting details.
