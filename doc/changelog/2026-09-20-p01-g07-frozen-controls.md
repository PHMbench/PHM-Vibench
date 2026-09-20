# P01 G07 frozen-output controls

`python -m experiments.p01.frozen_controls --run-root ORIGINAL_RUN
--recorded-root ORIGINAL_ABSOLUTE_PREFIX --output NEW_SIBLING_DIRECTORY` uses
saved predictions only. Set `CUDA_VISIBLE_DEVICES=''` and use the existing CPU
Python environment. Explicit root rebinding refers to a known copy, never a
search for interchangeable results.

The reference/protocol gate precedes source-fixed constant estimation. A failed
gate records failure and does not run controls. Existing parameter/buffer guards
and saved restoration outputs are distinguished from a new tensor-level audit.
The original group estimator and paired bootstrap are reused, with constants
fixed in every draw and p0 represented once. Pipeline terms are risk accounting,
not causal effects. All results are post-hoc; completion seals these test bearings.

The entrypoint rejects existing output directories, writes inside the original
run, H5/checkpoint opens, and nonempty GPU visibility. It neither loads models nor
changes D1 recipes, labels, predictions, decisions, or manuscript claims.

Validation: targeted constructed-array tests; actual-run evidence is recorded
separately and must not be inferred from software tests.
