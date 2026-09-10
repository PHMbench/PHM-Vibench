# Finite Streamlit batches execute through the existing CLI

After validating one experiment, users can preview a finite parameter grid, inspect its
YAMLs and fit count, then submit it explicitly. The current UI varies learning rate,
batch size, epochs, seed or iterations and caps the plan at 16 calls and 64 fits.

The existing run service owns all processes. A batch reserves its single-run slot across
trials, resolves all configurations before launch, and uses the same per-trial public
preflight, CLI and cancellation path as an individual experiment. A failed trial pauses
pending work; explicit continuation never retries or erases that failure. Cancellation
stops the current managed child and cancels unstarted items. No scientific result is
recomputed or inferred from shared directories.

Scheduling state survives page refresh through a small batch record. Server-process
restart does not resume execution: the batch becomes interrupted and pending work remains
unsubmitted. This is a single-control-process feature, not a distributed scheduler or a
promise of crash recovery. Agent proposals and adaptive search remain separate work.

Validation covers real child-process ordering, mutual exclusion, budgets, failure pause,
explicit continuation, cancellation during preflight/running/paused states, and lost
ownership. Existing public integration tests add two real CPU Dummy trials with independent
CLI result paths. AppTest checks preview/edit invalidation without running experiments.

Changes are limited to the optional frontend, its tests, existing UI workflow selection
and documentation. Data, model, Task, Trainer, split, estimator and checkpoint selection
implementations remain unchanged. No external-data download or benchmark promotion is
part of this change.
