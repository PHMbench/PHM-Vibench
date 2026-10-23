# PU Goal07 outcome: UNRESOLVED

The bounded observation task completed on 2026-10-16, exit 0: all 2,559
predeclared PU records were inspected once with no parsing failures. This
completes Goal07's measurement, not PU acquisition qualification or a training run.

## New observed evidence

| Measurement | Actual result | Interpretation |
| --- | --- | --- |
| Coverage | 2,559 unique metadata IDs; 32 bearing IDs; four operating conditions; 128 bearing/condition summaries | Exactly the existing PU inventory, not a new source selection |
| Vibration/time association | Every record has unique `vibration_1` at Y[7], `XIndex=2`, matching X[2]/Y `Raster=HostService`, and equal signal/time lengths | Resolves which stored time vector is associated with the retained vibration channel |
| Time order | Zero duplicate and zero reversed steps in all linked vectors | Stored vectors are strictly increasing; this does not establish a uniform physical sampling clock |
| Stored increments | All 2,559 vectors have nonconstant exact floating-point increments | A descriptive property of the export; no clock tolerance or physical-clock interpretation was inferred |
| Units | All 2,559 vibration `Unit` and linked time `Unit` fields are empty | No numerical acceleration conversion or absolute clock unit can be recovered from these fields alone |
| Metadata length | 1,517 original vibration lengths differ from listed metadata length | The known conflict remains; no trimming or metadata repair was performed |
| Documented replacement | Id48223/48224 (KA04 files 17/18) have exactly equal vibration and associated time arrays | Confirms the original log's replacement remark for this pair only; it does not establish whole-PU duplicate coverage |

The span-average cadence ranges from 63,998.934772764085 to 72,961.54681087175
samples per stored time unit. The CSV's Hz columns are explicitly conditional on
`X.Data` being seconds; they are not certified native or effective sampling rates.
Signal lengths range from 249,940 to 299,038. No waveform or time vector was
replaced by an assumed uniform grid.

Sentinel Id47567 has 256,823 linked signal/time entries, first stored time
`-3.4217894419041883e-6`, last `4.0`, and conditional span-average cadence
`64205.44507562148`. Its minimum/median/maximum stored increments are
`4.277236802074924e-6`, `1.5398052489068448e-5`, and
`1.5825776169542394e-5`. Its listed length is smaller by 822. The original
metadata-rate claim remains 64,000 Hz; no corrected rate is proposed.

## H1 verdict and remaining requirements

**H1 verdict: UNRESOLVED.** The channel-to-time association is now directly
established across the fixed inventory. Original documents independently support
nominal 64 kHz digitization, four-second recordings, acceleration sensing with
336C04 and 100 mV/g sensitivity, and nominal 5015(A) second-order 30 kHz filtering.
They do not determine the stored MAT amplitude scale, actual gain/ADC/export
conversion, complete calibrated frequency response/transition guards, or the
meaning and accuracy of the HostService timestamps.

Therefore the stored time variation does not by itself prove incompatible
physical sampling. Conversely, neither a near-64-kHz average nor a nominal
30-kHz low-pass establishes the frozen binary-support observation model.
The documented replacement pair must not be described as two distinct original
acquisitions. Its existing bearing grouping is retained; no file, label, group,
support record or query episode was changed.

J1 remains NO-GO. The retained qualification table has zero qualified corpora and
zero eligible target folds; this task does not requalify other datasets. No
industrial encoder was initialized or trained, no target head was fitted, and
no tokenizer effect or 95% fitting result is estimated. The existing S/U
Method, fitting budget and natural-acquisition protocol are unchanged.

## Static stage eligibility (C5/C6/C7)

This is a **read-only reassessment** of the retained `reports/tii_local_j1/DATASET_QUALIFICATION.csv` and `FOLD_ELIGIBILITY.csv` at the PR's base `dev=c47f976e`. The original 19 dataset/task identities contain **zero qualified independent industrial datasets** and **zero eligible target folds**. The PU episode alone is PASS, not acquisition-qualified. This follow-up has not requalified other sources, redrawn episodes, or enumerated inner-alpha tasks.

| Stage | Necessary condition | Evidence now available | Execution verdict |
| --- | --- | --- | --- |
| C5 source fitting | At least two qualified independent sources, with group-separated train/val and valid physical input support | 0 qualified datasets; PU remains UNRESOLVED | **BLOCKED**; fit95 NOT RUN |
| C6 held-out transfer | At least three qualified datasets, at least two valid sources after each target exclusion, a fixed target episode and a nonempty admissible source-only inner-alpha set | 0 qualified datasets and 0 eligible folds; PU's episode PASS does not satisfy the acquisition gate; inner-alpha feasibility NOT EVALUATED after failed prerequisite | **BLOCKED**; no industrial SC/UU target effects |
| C7 pooling interaction | C6 prerequisites plus SC/UU × single/joint four cells, a predefined fully observed single source, fixed full-pool geometry and matched adaptation | C6 BLOCKED; no qualified single source or complete industrial four-cell results | **BLOCKED**; no interaction estimate |

### Separate acquisition-dimension decisions

| Dimension | Verdict | Exact boundary |
| --- | --- | --- |
| Raw record and stored-axis mapping | **resolved** (structural mapping only) | All selected metadata IDs link to a named original MAT and the unique vibration channel's matching XIndex/raster/length; KA04 replacement IDs 48223/48224 are not independent acquisitions |
| Time | **unresolved** | Stored vectors are ordered, but HostService semantics, clock accuracy and the relation of stored increments to physical sampling are unknown; conditional cadence is not a certified Hz rate |
| Unit | **unresolved** | MAT unit fields are empty and end-to-end conditioner/ADC/export scaling cannot be established from the nominal 100 mV/g sensitivity |
| Frequency response | **unresolved** | Nominal second-order 30 kHz low-pass does not provide measured full-chain passband, attenuation tolerance, transition guards or anti-alias qualification |
| Physical grouping | **resolved for bearing-level grouping; incomplete for individual acquisition independence** | 32 documented bearing codes are retained as groups; the one log-identified duplicate pair is confirmed; no exhaustive duplicate-acquisition proof was performed |

**Overall: UNRESOLVED.** No newly eligible source or fold is established. These dimensions cannot be averaged into an acquisition PASS; further progress requires original calibration/scaling, clock/export and full-chain response documents. Do not restart training or alter the frozen protocol.

## Artifacts and checks

- `record_observations.csv`: complete record-level structural/time/unit observations.
- `group_summary.csv`: all 128 bearing/condition descriptive summaries.
- `observation_summary.json`, `run.log`, `resources.txt`: actual input scope,
  environment, code revision, exit and measured resources.
- `acquisition_evidence.md`: original local document pages and inference limits.
- `documented_replacement_check.csv`, `commands.md`: the bounded pair check and
  reproducible commands, including sentinel duration/export metadata.

The program inspected records in 183.641 s. Measured end-to-end wall time was
188.06 s and peak RSS 616,244 KiB (about 602 MiB); CPU only, LQ_signal. Monetary
local CPU cost was not measured. Output inspection confirmed 2,559 distinct IDs,
all statuses observed, and group-summary record totals equal to 2,559. Eighteen
focused synthetic tests passed in 2.89 s. The complete-suite attempt failed at
collection because LQ_signal lacks streamlit; its log is retained and no
complete-suite pass is claimed.

Blocked action: the one-model fit95 remains NOT RUN because acquisition eligibility and a real multi-source configuration are incomplete.
Minimum new evidence: record-associated acquisition/export scaling and gain/ADC calibration, HostService timing semantics and clock tolerance, and the applicable full-chain usable response/guards.
Stop here without new source evidence; do not repeat the same scan, relax the protocol, repair raw data, or launch J2/N/H/J3.
