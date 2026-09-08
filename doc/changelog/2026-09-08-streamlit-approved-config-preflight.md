# Streamlit uses one approved execution configuration

## User-visible change

After validation, Streamlit materializes the public inspector's resolved mapping as one
`execution.yaml`. Download, public preflight, the real CLI process, and restart consume
that snapshot. Safe-field and raw override values are folded into the YAML during approval;
the training command no longer receives a second override list.

The run service repeats public config inspection for programmatic callers, then runs the
existing `phmfactory preflight` against the exact saved file before starting training. A
preflight failure removes the unused UI run directory and propagates the real diagnostic.
No alternative device, data source, Pipeline, or parameter value is selected.

## Scope

This is a Streamlit adapter correction. It does not modify PHMFactory configuration
composition, Factory/Pipeline code, datasets, split logic, objectives, metrics, checkpoint
selection, or release claims. Result binding remains separate work: Metrics/Artifacts may
still scan the configured output root, while direct paths in `run.log` remain authoritative
until U3.
