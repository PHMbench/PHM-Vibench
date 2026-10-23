# Bounded per-source fitting diagnostics

Add `--fit95` to the existing `scripts.tii_one_model` and
`scripts/run_tii_fit95.sh`. One shared S/DLinear network is fitted once for
10,000 joint updates, seed 0, physical GPU0. Existing source eligibility,
source-only RMS, local heads, optimizer and validation-NLL selection remain
unchanged. No retry-until-success loop, extra source, or backbone switch.

A read-only callback records complete train/validation predictions at fixed
updates. The same validation-selected checkpoint must reach both group-balanced
and window accuracy of 95% in every source. Terminal performance and majority
references are diagnostic, not alternative selection criteria. Below-target
runs retain their observations and the shell exits 3 after exporting plots.
Figures only read retained CSVs; export recovery never repeats fit.

The old 20-update acceptance mode and its saved-report schema remain valid.
Tests cover invalid roles/populations, source rather than pooled thresholds,
checkpoint identity, unchanged test-network optimization, and a full native
CPU lifecycle on explicitly generated sine fixtures. These are software tests,
not industrial fitting results or natural-acquisition qualification.

See [execution instructions](../../configs/experiments/tii/FIT95.md).
Rollback: revert this feature's squash commit; retain historical run artifacts.
