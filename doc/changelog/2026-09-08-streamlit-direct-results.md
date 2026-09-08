# Streamlit binds results to the selected CLI run

## User-visible change

The Streamlit Metrics and Artifacts views no longer scan the configured shared output root
or use file modification time to infer which experiment produced a result. After a successful
process, the UI reads the final public CLI trailer from that run's own `run.log` and accepts
only the reported `result_dir`, `best_checkpoint`, `test_metrics`, `run_summary`, and
`primary_metrics`.

The reported metric and summary files must resolve inside the reported `result_dir` before
the UI reads them. Headline metric cards use `primary_metrics` returned by PHMFactory rather
than selecting numeric columns from a discovered table. The exact result directory may still
be browsed for images and small artifacts; the Streamlit process directory remains separately
available for `execution.yaml`, `run.json`, and `run.log`.

A failed/cancelled process, a successful process without the final `run=completed` trailer,
or a direct path outside `result_dir` does not trigger a fallback search. Training-only runs
are identified from the approved `trainer.test_after_fit` value and do not require test files.

## Scope

Only the optional Streamlit result adapter, UI rendering, focused tests, integration test,
and documentation change. PHMFactory result generation, metric mathematics, run summary,
checkpoint selection, Factory/Pipeline behavior, and experiment configuration are unchanged.
Batch experiments and Agent execution remain separate later work.
