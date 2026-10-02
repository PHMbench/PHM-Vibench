# P08 execution authority

The active implementation is `scripts.p08_physical`, configured by
`configs/experiments/p08/physical_conditioning.yaml`. The executable protocol,
input contracts, budgets, and commands are maintained in the
[experiment guide](../configs/experiments/p08/README.md). The paper's
[Local Codex SOP](https://github.com/AI4Engineering-L/P08-HSE-Prompt-CDDG/blob/dev/local-execution-v5/START_AGENT.md)
defines the local handoff.

## Archived alternate protocol

Commit `838ad9407b0af5d9e086c5ac47ddd8065f6e1d8c` contains the concurrently
published native Data/Task/Trainer prototype in full. This merge retains that
commit as a parent; no result or implementation is lost from Git history.
Its `source_only.yaml`, `DG/p08` task, and separate fit/evaluate schema are
retired from the active tree because their selection, factorial contrasts,
metadata schema, and metric population differ from the approved scientific
frame. The former `scripts.p08_experiments` entrypoint reports that retirement
and never silently invokes a different experiment.

The active model retains the alternate work's explicit component-name checks
and fixed-patch positive overfit test. The shared metadata reader retains its
optional dtype argument. Native lifecycle integration remains recoverable from
the archived commit for a separately reviewed migration, rather than a second
current protocol.

## Scope of target exclusion

The active data preflight may inspect the combined inventory, including target
metadata and ontology completeness. Only source-training records fit condition
statistics and model parameters, and only source-validation records select
hyperparameters and checkpoints. This is not a claim that target inventory is
never opened. Target labels do not fit or select a model; dataset qualification
is documented before formal comparison. Synthetic checks do not qualify
industrial datasets or establish diagnostic performance.
