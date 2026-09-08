# Streamlit accepts the current public inspector

## User-visible correction

The workspace no longer rejects a valid inspector response because it lacks a
configuration digest. It compares the submitted YAML and typed override arguments directly
to the inputs last checked. A changed request requires another check before Run is enabled.
The unused validation signature has been removed from the UI process records as well.

The app entrypoint now imports the workspace by its package path. Launch from the checkout
root with `python -m streamlit run apps/streamlit/app.py`; the entrypoint no longer retries
an ambiguous top-level import when executed outside package context.

A regression uses the real `scripts.config_inspect` process rather than a mock that supplies
a retired field. The integration checks also exercise the actual Dummy CLI through the
existing run service and the page's validate/edit/revalidate sequence through AppTest.
The Streamlit workflow triggers on both inspector/configuration dependencies and the
CLI, runtime, factories and bundled data exercised by that integration. Its CPU environment
installs matching Torch and torchvision binaries before normal dependency resolution.

## Scope and limits

Only the optional frontend, its tests, workflow and documentation change. No backend
configuration, Pipeline, Factory, model, split, objective, metric or checkpoint behavior
changes. No THU or other external-data experiment is required.

This closes the configuration-inspection blocker, not the whole frontend plan. Folding
all edits into one downloaded/executed configuration, public preflight at the submission
boundary, exact result-path binding, batch execution and Agent tools remain separate work.
