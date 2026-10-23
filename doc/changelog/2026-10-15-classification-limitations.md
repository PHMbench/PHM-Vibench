# Classification runtime limitations

Correct two outdated statements in `KNOWN_LIMITATIONS.md`: the maintained classification
runtime already keeps all iterations and aggregate outputs under one invocation root,
and rejects results missing a declared metric for an expected test dataset.

These corrections describe existing behavior in `src/runtime/classification.py` and
its `test/test_run_summary.py` regressions. They do not extend the guarantee to research
or legacy runtimes or change real-data qualification or release readiness.
