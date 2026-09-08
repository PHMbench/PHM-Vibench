# Streamlit refreshes only active experiments

Date: 2026-09-08

## User-visible change

The Streamlit workspace now reserves the two-second fragment refresh for active experiment processes. While a run is starting, running, cancelling, or detached, the page refreshes process status and the current log only. It does not repeatedly discover metrics or artifacts before PHMFactory reports exact result paths.

When the selected process reaches a terminal state, the timed fragment triggers one normal app rerun. The terminal view then binds and renders exact PHMFactory results without another periodic timer.

This reduces repeated file parsing and image/table work on completed runs and avoids pretending that in-progress files are scientific results.

## Scientific boundary

The frontend still treats the public CLI trailer as result authority. It does not compute metrics, scan a shared output root for the newest run, or promote process completion into benchmark validity. Training-only, failed, cancelled, and incomplete-result cases retain their existing semantics.

No PHMFactory backend configuration, Factory, Pipeline, runtime, split, objective, metric, checkpoint selection, THU/MFPT experiment, or Agent execution logic changes.

## Validation

Focused policy tests verify that active rendering never calls `discover_results`, terminal transitions leave the timed fragment, the outer renderer uses the timer only for active records, and a terminal render performs one result discovery per page rerun. Existing Streamlit public integration and AppTest continue to exercise the real inspector, page, and Dummy CLI lifecycle.
