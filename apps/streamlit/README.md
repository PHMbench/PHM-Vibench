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

Inspection resolves the requested experiment. Run then materializes that approved mapping as
one `execution.yaml` and invokes the existing public preflight against that exact file before
training starts. A missing dependency or unavailable device remains a visible failure and
does not trigger a different experiment.

## Edit a configuration

**Quick Start** exposes common fields from `field_catalog.yaml`. **Advanced** also offers
standalone YAML and one `key=value` override per line. Raw overrides have highest priority
while the request is being inspected.

The backend owns composition, strict types, Pipeline selection, and explicit local config.
The UI does not auto-discover `configs/local/local.yaml`. Repository templates remain
unchanged when a run creates its own `execution.yaml`.

After validation, all safe-field and raw override edits are folded into one resolved YAML.
Download, public preflight, execution, and restart use that same snapshot without a second
`--override` layer. Repository templates remain unchanged.

## Run and inspect

Each UI run has a small process workspace:

```text
outputs/streamlit/<run-id>/
├── execution.yaml
├── run.json
└── run.log
```

`run.json` records process state, command and timestamps. It is not a scientific evaluation
or an integrity record. PHMFactory returns its scientific result paths in the final CLI
trailer written to `run.log`:

```text
result_dir=...
best_checkpoint=...
test_metrics=...
run_summary=...
primary_metrics=...
run=completed
```

The Metrics and Artifacts views bind to that exact `result_dir`. `test_metrics` and
`run_summary` are consumed only when the reported files are inside the reported result
directory. Headline cards come from the CLI `primary_metrics` summary, not from an arbitrary
CSV row. Files elsewhere under the configured output root are never attributed to the run by
mtime or filename. The Streamlit process directory remains available for its own YAML, log,
and run record.

A training-only request does not require test metrics or a test summary. A non-zero exit,
cancelled run, or successful process without the final CLI trailer does not trigger a search
for substitute results. The page keeps the process status and logs available and reports that
direct scientific results are unavailable.

The workspace can cancel its active process and repeat a prior configuration in a new run.
One Streamlit worker manages one active experiment. Browser refresh does not submit a new
run; a server restart may leave the child process detached and must not trigger automatic
resubmission. Pause/resume and cluster scheduling are not supported.

## Troubleshooting

A rejected configuration shows the inspector's original stderr. Copy the visible command
and run it with the same Python and working directory. Use `phmfactory doctor` for environment
issues. The Run action already executes `phmfactory preflight --config <execution.yaml>`
before training; copy that saved YAML to reproduce the same preflight manually. Missing local
data requires correcting the requested path; switching to the offline example is an explicit
user action, never an automatic fallback.

If the process exits successfully but the page reports no direct results, inspect the full
per-run `run.log`. The UI requires the public final trailer shown above and does not guess a
result directory from adjacent files.

## Development and tests

Keep changes within the existing services: `config_service.py` adapts the public inspector,
`run_service.py` manages subprocesses, and `result_service.py` consumes direct CLI results.
Field and template catalogues are UI metadata, not new experiment schemas. There is no
experiment Agent or autonomous parameter search in this version.

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
