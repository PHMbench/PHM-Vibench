# PHMFactory Streamlit Experiment Workspace

Select an experiment, edit its parameters, inspect the configuration, and launch the
existing PHMFactory CLI from a browser. Training stays in a separate process; the UI
never calls a Factory or Trainer directly.

## Install and start

Use the same Python environment for PHMFactory and Streamlit. From the checkout root:

```bash
python -m pip install -e .
python -m pip install -r apps/streamlit/requirements.txt
python -m streamlit run apps/streamlit/app.py
```

`apps/streamlit/app.py` is the maintained web entrypoint. Data paths refer to the machine
running Streamlit, not to another computer opening the browser.

## First experiment

1. Select **Use safe CPU smoke defaults**. This chooses the bundled Dummy template,
   CPU, one epoch, and zero data workers.
2. Review the visible parameters and click **Validate configuration**.
3. Click **Run experiment** and inspect the process status and log below.

The public inspector remains the configuration authority. The UI accepts its current
`resolved`, `sources`, `targets`, `sanity`, and `local_config_path` output; no configuration
digest is required. Validation stores the submitted YAML text and typed override argv
for direct comparison. Editing either makes the old validation stale and disables Run
until the new request is checked. Hidden machine-local files do not participate.

Inspection is not a promise that training will succeed. It does not construct the training
stack or perform every public preflight check. A missing dependency or unavailable device
must remain a visible failure, not trigger a different experiment.

## Edit a configuration

**Quick Start** exposes common fields from `field_catalog.yaml`. **Advanced** also offers
standalone YAML and one `key=value` override per line. Raw overrides have highest priority;
the displayed command repeats `--override` for each value and is launched with `shell=False`.

The backend owns composition, strict types, Pipeline selection, and explicit local config.
The UI does not auto-discover `configs/local/local.yaml`. Repository templates remain
unchanged when a run creates its own `execution.yaml`.

Current limitation: downloaded YAML does not yet fold in separate overrides. Preserve
both the configuration and the displayed override arguments when reproducing a run.
A single checked/downloaded/executed snapshot is the next configuration improvement.

## Run and inspect

Each UI run has a small process workspace:

```text
outputs/streamlit/<run-id>/
├── execution.yaml
├── run.json
└── run.log
```

`run.json` records process state, command and timestamps. It is not a scientific evaluation
or an integrity record. PHMFactory returns its own result paths in `run.log`:

```text
result_dir=...
best_checkpoint=...
test_metrics=...
run_summary=...
primary_metrics=...
run=completed
```

Use those exact paths to confirm the current run. A training-only request does not have
test metrics or a test summary. A non-zero exit remains failed even if partial files exist.

The workspace can cancel its active process and repeat a prior configuration in a new run.
One Streamlit worker manages one active experiment. Browser refresh does not submit a new
run; a server restart may leave the child process detached and must not trigger automatic
resubmission. Pause/resume and cluster scheduling are not supported.

Current limitation: the Metrics/Artifacts views still discover files below the configured
output root. Until direct-path result binding is implemented, do not use a shared-output
view to attribute results from concurrent experiments. The per-run CLI log is authoritative.

## Troubleshooting

A rejected configuration shows the inspector's original stderr. Copy the visible command
and run it with the same Python and working directory. Use `phmfactory doctor` for environment
issues, and `phmfactory preflight --config <yaml>` with the same overrides for public
preflight checks. Missing local data requires correcting the requested path; switching to
the offline example is an explicit user action, never an automatic fallback.

## Development and tests

Keep changes within the existing services: `config_service.py` adapts the public inspector,
`run_service.py` manages subprocesses, and `result_service.py` displays results. Field and
template catalogues are UI metadata, not new experiment schemas. There is no experiment
Agent or autonomous parameter search in this version.

```bash
python -m pytest -q \
  test/test_streamlit_config_service.py \
  test/test_streamlit_runtime_policy.py \
  test/test_streamlit_onboarding.py \
  test/test_streamlit_run_service.py \
  test/test_streamlit_result_service.py \
  test/test_streamlit_ui_imports.py

# Full PHMFactory environment; actual inspector and Dummy CLI subprocess.
python -m pytest test/test_streamlit_public_contract.py -q

# Run separately from optional-import stub tests; requires real Streamlit.
python -m pytest test/test_streamlit_app.py -q
python -m scripts.validate_docs
```

The existing Streamlit workflow runs lightweight service tests on Linux and Windows. A
separate Ubuntu integration job uses real Streamlit, the public inspector and the actual
Dummy lifecycle. AppTest verifies page behavior, not browser disconnection or cancellation
of every operating-system child process.
