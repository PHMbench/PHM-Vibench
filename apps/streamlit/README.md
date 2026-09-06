# PHMFactory Streamlit Experiment Workspace

This optional browser workspace helps users select a maintained template, edit a bounded
parameter surface, validate the exact configuration, launch the public CLI, and inspect
logs and artifacts.

It is an adapter around PHMFactory—not a second configuration parser, scheduler, or
training framework.

## Install and start

Install the core source checkout first, then the optional UI dependencies:

```bash
python -m pip install -e .
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

Run the command from the repository root. `apps/streamlit/app.py` is the only maintained
web entrypoint.

## First experiment

Use **Use safe CPU smoke defaults** in the sidebar. It selects:

```text
Template: demo_00_smoke_dummy_dg
Mode:     Quick Start
Device:   cpu
Epochs:   1
Data:     repository-shipped Dummy files
```

The workspace then guides the user through four steps:

1. select a maintained template;
2. change only the parameters needed for this run;
3. validate the exact effective configuration and launch it;
4. inspect live logs, metrics, files, and the reproduction command.

## One configuration truth

Streamlit delegates composition and validation to the same public inspector used by:

```text
phmfactory preflight
phmfactory run
scripts.validate_configs
scripts.config_inspect
```

The UI does not implement `base_configs` merging, Pipeline canonicalization, or hidden
machine overrides itself.

The public precedence is:

```text
base_configs
< selected experiment YAML
< explicit local config, only when supplied by a CLI user
< explicit overrides
```

The current UI intentionally does not auto-discover or silently apply
`configs/local/local.yaml`. Quick Start and Advanced mode therefore have no invisible
machine-local layer. Machine-specific values are edited in the standalone YAML or entered
as explicit overrides. The planned command shown by the UI is the command that is
launched.

A successful validation report carries the same `effective_config_sha256` that CLI
preflight and the final run manifest record. If the visible YAML or overrides change, the
validation becomes stale and the Run button is disabled until validation runs again.

## Quick Start mode

Quick Start exposes only catalog-approved fields. The selected template is resolved once
through the public inspector. UI values are converted into typed `key=value` argv tokens.

Use this mode for:

- the first offline smoke;
- common epoch, device, worker, seed, or data-path changes;
- users who do not need to edit the full YAML.

## Advanced mode

Advanced mode adds:

- the same safe field catalog;
- a standalone effective-YAML editor;
- one typed override per line;
- a configuration diff;
- exact planned and actual reproduction commands.

The YAML shown in this mode already contains the fully resolved base configuration. It
has no hidden local layer. Raw overrides remain highest precedence and are passed as argv
elements; the UI never builds a `shell=True` command.

## Readiness and launch blockers

Before launch, the workspace checks:

- the repository entrypoint and configuration inventory;
- required Python imports;
- repository-shipped smoke assets;
- output-directory writability;
- selected template data and metadata availability;
- public config inspection and sanity checks.

A failed readiness check does not prevent users from inspecting the template or editing
YAML, but execution remains disabled until the relevant environment, data, or config
problem is fixed.

## Run lifecycle

Each UI run creates a managed workspace:

```text
outputs/streamlit/<run-id>/
├── execution.yaml
├── run.json
└── run.log
```

The process then invokes the public command contract. PHMFactory's own runtime writes the
invocation manifest below the configured experiment output directory:

```text
<environment.output_dir>/.phmfactory/runs/<run-id>/run_manifest.json
```

The UI supports:

- **Run** — start one experiment process;
- **Cancel** — terminate the process group, then force-kill after a grace period;
- **Restart same run** — reuse the immutable YAML and override snapshot.

Pause/resume is intentionally absent because it is not portable across Windows, CUDA,
and data-loader subprocesses. One Streamlit worker manages one active experiment at a
time; the workspace is not an implicit cluster scheduler.

## Results

Result discovery is bounded by directory depth, entry count, file count, metric file
size, and row count. Symbolic links are skipped.

The result views provide:

- **Overview** — actual command, output roots, run metadata;
- **Metrics** — headline values plus small CSV/JSON tables;
- **Artifacts** — images, file inventory, and small-file downloads;
- **Logs** — live tail and full-log download.

Malformed optional artifacts do not erase the process status, command, run manifest, or
raw log.

## Extension boundaries

Use declarative files rather than model-specific UI branches:

- `configs/config_registry.csv` — maintained template identity and status;
- `field_catalog.yaml` — editable fields, aliases, widgets, and template groups;
- `template_profiles.yaml` — difficulty, data, device, and first-action guidance;
- `config_service.py` — UI-safe serialization and public-inspector adapter;
- `runtime_policy.py` — temporary standalone YAML inspection;
- `run_service.py` — process lifecycle;
- `result_service.py` — bounded artifact parsing;
- `onboarding.py` — environment and data readiness;
- `ui_*.py` — visual components only.

New config semantics belong in `phmfactory.config`, not in Streamlit. New metrics formats
belong in `result_service.py`. New safe fields belong in `field_catalog.yaml`.

## Validation

```bash
python -m py_compile apps/streamlit/*.py
python -m pytest \
  test/test_streamlit_config_service.py \
  test/test_streamlit_runtime_policy.py \
  test/test_streamlit_onboarding.py \
  test/test_streamlit_run_service.py \
  test/test_streamlit_result_service.py \
  test/test_streamlit_ui_imports.py
python -m scripts.validate_configs
python -m scripts.validate_docs
phmfactory preflight --config smoke
phmfactory demo
```

Focused UI tests run on both Linux and Windows through
`.github/workflows/streamlit-quality-gates.yml`.

## Troubleshooting

### Validation and the CLI disagree

Copy the planned command from the UI and run it in the same environment. Report:

```text
exact command
UI effective_config_sha256
CLI preflight effective_config_sha256
complete stderr
```

Different hashes for the same visible YAML and overrides are a configuration-parity bug.

### A local path is missing

Use the offline smoke template to separate installation from external data availability.
In Advanced mode, edit `data.data_dir` in the standalone YAML or add an explicit raw
override. The UI does not read a hidden local file.

### A dependency import fails

Run `phmfactory doctor` in the same environment. The optional UI requirements do not
replace the core training environment.

### CUDA initialization fails

Return to CPU smoke, then verify the installed PyTorch build, driver, and device outside
PHMFactory before changing experiment code.

### No structured metrics appear

Inspect `run.log`, `run.json`, and the PHMFactory `run_manifest.json`. Optional parser
failure does not invalidate those primary records.
