# Opt-in warmup-cosine schedule, gradient clipping, and the declared stabilized TII recipe

`configure_optimizers` gains a `warmup_cosine` scheduler: a step-interval
linear warmup (default five percent of the declared budget, explicit
`warmup_steps` or `warmup_fraction`) chained into cosine annealing through
`SequentialLR`. The total update budget is declared explicitly via
`options.total_steps`, which serves fixed-round single-epoch protocols whose
budget is not derivable from epochs. All options fail closed.

The trainer forwards the new `trainer.gradient_clip_val` (default `None`,
behavior unchanged) into `pl.Trainer`.

`tii_joint` now accepts exactly one declared stabilized recipe:
`task.scheduler = {name: warmup_cosine, options: {total_steps: <rounds>, ...}}`
with explicit positive `lr` and nonnegative `weight_decay`. `total_steps`
must equal the declared round budget. The legacy recipe remains enforced
unchanged: without the schedule, `lr=.001`/`weight_decay=.0001` is still the
only accepted optimizer setting and every other scheduler name is rejected.

Tests pin the LR trajectory (linear rise to peak at the warmup boundary,
cosine decay afterwards), the five-percent default, fail-closed option
validation, trainer wiring, and the tii_joint accept/reject matrix.
