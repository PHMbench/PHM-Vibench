# Goal07 actual commands and provenance

Date: 2026-10-16. All Python commands used conda `LQ_signal`; all computations
were on CPU with `CUDA_VISIBLE_DEVICES=''`. No training, HPO, target episode draw,
H5 comparison, raw-data modification or package installation was performed.

The paper working base was `8c9507c9cafa4b028cb6d23b4b9d6da0f7fa3e8b`.
Its recorded child remains `540b0fa1b9bed78193049a23566caa941cfe23c3`, while
the existing child checkout remains `4d718fc22c6267623b6d2325d9c425554b387c90`.
Neither checkout nor gitlink was changed. The implementation uses an isolated
worktree of the same PHMFactory repository, branch
`feat/tii-pu-observations-20261016`, from current dev `c47f976e`.
The real observation code was committed as `c7f2fbc3` before the full run.

## Focused software check

```bash
CUDA_VISIBLE_DEVICES='' conda run -n LQ_signal --no-capture-output python -m pytest test/test_tii_pu_observations.py -q
```

PASS: 18 tests in 2.89 s. These use constructed MAT files to check association,
invalid XIndex/raster/length, nonfinite or nonmonotonic time, retained failure
rows, coverage, zero-valued identifiers, and refusal to overwrite outputs.

## One full original-PU observation pass

From the isolated PHMFactory worktree:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/time -v -o reports/tii_pu_observation_followup/resources.txt conda run -n LQ_signal --no-capture-output python -m scripts.tii_pu_observations --data-root /home/user/data/PHMbenchdata/PHM-Vibench --inventory reports/tii_local_j1/RECORD_INVENTORY.csv --output-dir reports/tii_pu_observation_followup --code-commit c7f2fbc3 > reports/tii_pu_observation_followup/run.log 2>&1
```

The native metadata reader verifies the selected IDs/files/rates/lengths against
the existing inventory before reading original MAT records. This is a new PU
channel/time-field measurement, not another 18-H5 qualification scan. Actual
terminal state, counts, software versions and in-process resources are in
`observation_summary.json`; process wall time/peak RSS are in `resources.txt`.
Parsing failures, if any, remain explicit rows. Scientific eligibility is
UNRESOLVED regardless of a successful program exit.

Actual result: exit 0; 2,559/2,559 records, zero parsing failures, 32 bearings and
four operating conditions. `resources.txt` reports 188.06 s wall and 616,244 KiB
peak RSS; the program reports 183.641 s inspection/output time. The final output
was inspected using pandas in LQ_signal: metadata IDs are unique, all rows are
`observed`, and the 128 group-summary rows sum to all 2,559 records. All linked
vectors have zero duplicate/reversed steps; all signal/time Unit fields are
empty; 1,517 lengths still disagree with metadata. The per-stored-time-unit rate
range is 63,998.934772764085–72,961.54681087175. These are stored-axis summaries,
not physical-clock qualification or independent-device inference.

## Sentinel metadata and documented replacement pair

The following additional bounded original-MAT checks were executed interactively
with `conda run -n LQ_signal --no-capture-output python -` and SciPy `loadmat`.
They inspect no private `Description.General.User` field.

```python
from pathlib import Path
import numpy as np
from scipy.io import loadmat, whosmat
base = Path('/home/user/data/PHMbenchdata/PHM-Vibench/raw/RM_027_PU')
records = []
for rel in ('KA04/N09_M07_F10_KA04_17.mat', 'KA04/N09_M07_F10_KA04_18.mat'):
    path = base / rel
    print(rel, whosmat(path))
    original = loadmat(path, squeeze_me=True, struct_as_record=False)[path.stem]
    signal = next(y for y in np.atleast_1d(original.Y) if y.Name == 'vibration_1')
    axis = np.atleast_1d(original.X)[int(signal.XIndex) - 1]
    records.append((np.asarray(signal.Data), np.asarray(axis.Data)))
print(np.array_equal(records[0][0], records[1][0]))
print(np.array_equal(records[0][1], records[1][1]))
path = base / 'K001/N09_M07_F10_K001_1.mat'
original = loadmat(path, squeeze_me=True, struct_as_record=False)[path.stem]
for name in ('Recording', 'Measurement'):
    block = getattr(original.Description, name)
    print(name, {key: str(getattr(block, key)) for key in block._fieldnames})
print(original.Description.General.Origin)
```

Observed: each replacement-pair file contains its correctly named root variable;
both raw `vibration_1` vectors and their linked time vectors compare exactly equal.
Existing inventory IDs are 48223 and 48224, both physical bearing KA04.
`documented_replacement_check.csv` preserves this result for the pair identified
in its original measuring log; no whole-dataset duplicate search was performed.

For sentinel Id47567, `Recording.StopCondition` is `Time limit: 4.000000`,
`Measurement.Length=4`, `XAxisOffset=0`, `StartTimestamp=0.9924019165442665`,
and `StopTimestamp=4.992401916544266`. `General.Origin` names
`modell_autass_05_14.sdf` and `modell_autass_05_14.x86`. No actual model/export
configuration was inspected. These fields support duration provenance without
establishing the physical sample clock or a missing unit conversion.

## Complete-suite attempt

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n LQ_signal --no-capture-output python -m pytest test/ -q > reports/tii_pu_observation_followup/full_suite.log 2>&1
```

FAIL during collection: `test/test_streamlit_app.py` imports unavailable
`streamlit`; 11.14 s, exit 2. The log is retained. No complete-suite pass is
claimed, no dependency was installed, and the successful focused tests are
reported separately. Repository documentation checks are recorded after the
complete report is ready. No paid API compute or GPU was used; monetary local
CPU cost was not measured.
