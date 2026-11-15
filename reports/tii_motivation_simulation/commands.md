# Actual execution and checks

Date: 2026-10-17. Worktree: `PHMFactory-tii-simulation-20261017`, branch
`research/tii-simulation-20261017`, based on child dev `43a1f12e`. Code and
pre-result protocol were committed as `150d6976f3fd5b35f4da8c92dc5330b5859fe43f`
before the scientific run. The paper working base is `0efd0ae4be4510a8617fcd90022f0a3b09664506`.
Its gitlink `540b0fa1` and original child checkout `4d718fc2` remain unchanged.
No raw industrial data, H5, metadata, dependency or installed skill was changed.

## Software tests

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n LQ_signal --no-capture-output python -m pytest test/test_tii_motivation_simulation.py -q
```

PASS: 8 tests, 5.48 s. The two-update native smoke, checkpoint tie test and
deliberate failure checks are software fixtures, not additional scientific
populations. An initial smoke exposed a floating-path issue in an overly strict
restore check after freezing parameters; verification was moved before freezing,
without changing the selection tolerance, solver or experiment semantics.

## One frozen scientific pass

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/time -v -o reports/tii_motivation_simulation/resources.txt conda run -n LQ_signal --no-capture-output python -m scripts.tii_motivation_simulation --output reports/tii_motivation_simulation/run_v1 > reports/tii_motivation_simulation/run.log 2>&1
```

Exit 0; 12 encoder fits, each 300 native joint updates, and 24 convex head fits.
No retries, HPO or extra scientific fits followed. Actual interpreter, package
versions, command arguments, generator/model/optimizer settings, explicit CPU
selection and code revision are in `run_v1/config.json`; measured resources are
in `run_v1/status.json` and `resources.txt`. Logs retain every predeclared cell.

## Output inspection without retraining

Using NumPy/pandas/SciPy in LQ_signal, inspected all 12 cells and recomputed
NLL from the saved query logits. Verified 320 unique generated records per
task/repetition, independent source roles and target groups, 10 target support
records and 118 query records, saved features/head parameters/logit agreement,
and source-only checkpoint selection from steps0/50/100/150/200/250/300.
All checks passed. The maximum inverse-weight logit difference was exactly0;
all selected U first-affine ranks were16; all heads met the existing solver
criterion (maximum gradient_inf 1.5119082956947805e-8). See the retained arrays,
per-record predictions and source curves rather than treating these checks as
industrial validity evidence.

Full repository tests were not rerun: the prior complete-suite attempt in this
same LQ_signal environment failed at collection due to missing optional
`streamlit`, and no dependency was installed. No full-suite pass is claimed.

## Documentation check

```bash
CUDA_VISIBLE_DEVICES='' conda run -n LQ_signal --no-capture-output python -m scripts.validate_docs > reports/tii_motivation_simulation/docs_check.txt 2>&1
```

PASS: 95 files scanned, exit0; result retained in `docs_check.txt`. Git whitespace
inspection also passed. These software/document checks do not upgrade any
industrial evidence claim.
