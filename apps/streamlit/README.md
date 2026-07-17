# PHM-Vibench Streamlit Experiment Workspace

This directory contains the optional, configuration-first web workspace for
PHM-Vibench. It is a user-facing adapter around the maintained CLI, not a second
training framework.

```text
validated registry template
+ first-run readiness checks
+ portable YAML snapshot
+ typed CLI overrides
+ managed main.py process
+ bounded result discovery
```

The experiment contract remains:

```bash
python main.py --config <yaml> [--override key=value ...]
```

The application never imports a Pipeline function, does not modify `main.py`, and
does not change the five-block configuration schema.

## Install and start

Install the core environment first, then the optional UI layer:

```bash
pip install -r requirements.txt
pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

Run the command from the repository root. Streamlit 1.37 or newer is required for
independent live-log refresh through `st.fragment`.

## First experiment

Use the sidebar action **Use safe CPU smoke defaults**. It resets only the
configuration editor; existing run history is preserved.

The safe defaults are:

```text
Template: demo_00_smoke_dummy_dg
Mode:     Quick Start
Device:   cpu
Epochs:   1
```

The template uses repository-shipped dummy data and is the recommended way to
verify the environment before selecting an external dataset or GPU.

The workspace now checks, before launch:

- repository entrypoint and config directory;
- Streamlit, PyYAML, PyTorch, and Lightning availability;
- repository-shipped smoke assets;
- write access for `outputs/streamlit/`;
- whether `configs/local/local.yaml` is changing machine-local defaults;
- whether the selected template's data directory and metadata file exist.

A failed readiness check does not prevent users from inspecting or downloading a
valid configuration, but the **Run experiment** action stays disabled until the
environment and selected data are ready.

The workspace guides the user through four steps:

1. select a maintained registry template;
2. adjust a small catalog-approved parameter surface;
3. validate and launch the public CLI;
4. inspect live logs, metrics, images, artifacts, and the immutable command.

## Template guidance

User-facing template guidance is declarative in `template_profiles.yaml`. Each
maintained demo can describe:

- difficulty;
- bundled or external data requirements;
- recommended device;
- honest runtime guidance;
- onboarding badges and the next action.

Adding a new template profile must not add model-specific branches to
`workspace.py`. Unknown registry templates receive a conservative generic profile.

## Experience modes

### Quick Start

Quick Start exposes only onboarding-safe fields. The selected template is resolved
into a portable standalone YAML before execution. Machine-local configuration is
not baked into that snapshot.

### Advanced

Advanced adds:

- catalog-defined safe fields and ordered legacy aliases;
- a portable full-YAML editor;
- one typed `key=value` override per line;
- a configuration diff;
- the exact planned and actual reproduction commands.

The precedence model is explicit:

```text
portable YAML
< configs/local/local.yaml (when present)
< catalog-safe CLI overrides
< raw CLI overrides
```

Overrides are passed as argv elements. The application never builds a
`shell=True` command.

## Run lifecycle

Each run creates a durable workspace:

```text
outputs/streamlit/<run_id>/
├── execution.yaml
├── run.json
└── run.log
```

`run.json` is written atomically and records the template, mode, command,
overrides, PID, timestamps, exit status, validation signature, output root, and
restart ancestry.

The UI supports:

- **Run** — start one managed experiment process;
- **Cancel** — terminate the process group, then force-kill after a grace period;
- **Restart same run** — launch from the immutable YAML and override snapshot.

Pause/resume is intentionally not implemented. It is not portable across Windows,
CUDA workers, and data-loader subprocesses.

A single Streamlit worker manages one active experiment at a time. This keeps the
optional UI simple and prevents it from becoming an implicit scheduler.

## Results

Result discovery is bounded by directory depth, entry count, file count, metric
file size, and metric row count. Symbolic links are skipped.

The result tabs provide:

- **Overview** — actual command, output roots, run metadata;
- **Metrics** — headline values plus CSV/JSON tables;
- **Artifacts** — images, file inventory, small-file downloads;
- **Logs** — live tail and full-log download.

Missing or malformed optional artifacts never invalidate a completed run. The user
still receives the process status, command, manifest, and raw log.

## Compatibility and extension boundary

- `configs/config_registry.csv` remains the source of template identity/status.
- `field_catalog.yaml` remains the source of editable fields, aliases, widgets,
  and template groups.
- `template_profiles.yaml` remains the source of user-facing difficulty, data,
  device, and first-action guidance.
- Unknown future registry columns are retained as metadata.
- Key migrations should add an alias in `field_catalog.yaml`, not a model-specific
  conditional in `app.py`.
- Process lifecycle is isolated in `run_service.py`.
- Artifact parsing is isolated in `result_service.py`.
- Runtime/local-config precedence is isolated in `runtime_policy.py`.
- First-run checks and template data resolution are isolated in `onboarding.py`.
- Visual components and live-run views are isolated in `ui_theme.py`,
  `ui_onboarding.py`, and `ui_runtime.py`.

New metric formats should be added to `result_service.py`; new safe user-facing
fields should be added to `field_catalog.yaml`; new user guidance should be added
to `template_profiles.yaml`.

## Validation

```bash
python -m py_compile apps/streamlit/*.py streamlit_app.py
python -m pytest \
  test/test_streamlit_config_service.py \
  test/test_streamlit_runtime_policy.py \
  test/test_streamlit_onboarding.py \
  test/test_streamlit_run_service.py \
  test/test_streamlit_result_service.py \
  test/test_streamlit_ui_imports.py
python -m scripts.validate_configs
python -m scripts.validate_docs
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

The focused Streamlit service tests run on Linux and Windows through
`.github/workflows/streamlit-quality-gates.yml`.

## Troubleshooting

### The inspector reports a missing module

Install the repository's core dependencies. The optional UI requirements do not
replace the training environment. The readiness panel lists the missing module and
the relevant installation action.

### A data path does not exist

The selected template card displays the resolved data root and metadata path. Use
the offline CPU smoke template, edit `data.data_dir` in Advanced mode, or put the
machine-specific path in `configs/local/local.yaml`.

### A GPU run fails

Return to **Use safe CPU smoke defaults**, verify the offline run, then validate
CUDA and PyTorch independently.

### The worker restarted during a run

On POSIX, a still-live process is marked `detached` and automatic cancellation is
disabled to avoid killing a reused PID. Use the operating-system process manager,
then start a new run.

### No structured metrics appear

The run can still be inspected through `run.log`, `run.json`, and the output file
inventory. Optional artifact parsing is deliberately failure-tolerant.
