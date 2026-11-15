# Bounded constructed input-organization and readout simulation

Adds `python -m scripts.tii_motivation_simulation --output <new-directory>` for
the frozen [simulation protocol](../../reports/tii_motivation_simulation/protocol.md).
The public CLI fixes two tasks, three paired data/optimizer seeds, two native S/U
arms, 300 encoder updates per fit, and original/scaled target readouts. It requires
`CUDA_VISIBLE_DEVICES=''`, uses CPU only, and refuses existing output directories.

The runner reuses native whole-window projection, source sampling/RMS, M_01_ISFM
with DLinear, tii_joint loss/AdamW, source-validation estimation, frozen encoding,
episode selection, pooling and convex target-head fitting. It retains record roles,
matched initialization, rank diagnostics, source curves, selected checkpoints,
features, head solutions/query logits, paired seed differences and resources.
Failed fits preserve their errors and partial outputs rather than changing the
experiment or skipping a selected record.

These analytically constructed fixtures check finite-model behavior and readout
coordinate sensitivity. They do not qualify industrial acquisition, establish
competitive performance, estimate a source-pooling interaction, or close T2.
The target comprises independent records from the full source's generating
distribution; this does not test an out-of-distribution shift. The stored branch
weight stack describes U's first affine input map; for S it omits the projections.
Focused tests include a two-update native software smoke and failure retention;
those small test fixtures are not scientific repetitions.

Validation: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run
-n LQ_signal --no-capture-output python -m pytest
test/test_tii_motivation_simulation.py -q` — **PASS**, 8 tests in 5.48 seconds,
including fixed-episode isolation, head scaling/solver checks, the native smoke,
earliest-checkpoint ties/deep copies, output refusal, and error retention.
