# Local data qualification — 2026-09-19

The supplied 18 H5 containers contain 19 metadata dataset identities: SEU contains
both Dataset_id 9 (bearings) and 15 (gears). All candidates remain in the tables.
There are **0 qualified datasets and 0 eligible held-out folds** under the frozen
natural-support protocol. This data decision alone prevents J1 Go and J2; software
fixture tests cannot replace it.

The checks inspected every H5 key, shape, dtype and attribute and all 49,855 selected
metadata rows. Every selected metadata ID exists in its H5 and its original raw
path exists. No duplicate `(Name, File)`, resolved raw path or raw inode was found.
This checks declared local file identity, not equality of separately copied signals
or independent acquisition. Physical independence comes from original acquisition
documents; neither row IDs nor operating-condition `Domain_id` were used as groups.

`RECORD_INVENTORY.csv` records the exact Id → H5 key → original File relationship,
documented physical groups and shape facts. No experimental windows were generated.
A future admitted window would additionally require channel/start/end coordinates.
`metadata_sample_rates` reports the metadata claims; `effective_rate` separately
records known conflicts and raw sampling evidence.

The native readers were exercised on 30 whole recordings, covering every candidate
Dataset_id and each observed channel/rate/modality category. There were 26 exact
raw/cache matches, 3 reader/data failures and 1 mismatch. These are **sampled**
preprocessing observations, not a full signal comparison of the 83+ GiB collection.
The separate MFPT check compared all 23 original `gs` arrays against their complete
H5 arrays and read every embedded `sr`; all 23 arrays matched. No raw data was changed.

| Observed defect or qualification limit | Direct evidence |
| --- | --- |
| CWRU has 155 metadata rows but 163 H5 keys; keys 156–163 have no current metadata row; five rows lack labels/rates. | `DATA_ANOMALIES.csv`, `RECORD_INVENTORY.csv` |
| FEMTO includes 2,020 zero-channel records. Bearing1_4 has 1,665 nonnumeric cached records: 1,428 two-channel acceleration records and 237 single-channel temperature records. Missing/stage/bearing-specific labels also prevent fault-classification admission. | All H5 shapes/dtypes and raw names; `DATA_ANOMALIES.csv`, `RECORD_INVENTORY.csv` |
| THU cached two-channel records do not match the current native raw reader; the representative cache has 1,228,800 rows versus 1,236,992 raw-reader rows. | `RAW_CACHE_COMPARISON.csv`, original vibration/voltage instructions |
| MFPT native reader requires BPFO absent from the supplied laboratory files. OilPump Id46303 is 24,414 Hz and Planet Id46304 is 6,104 Hz, despite metadata 48,828 Hz. The cache did not resample them. | `RAW_CACHE_COMPARISON.csv`, `RAW_ACQUISITION_CHECKS.csv` |
| DIRG has reliable acceleration units, hardware-response evidence and seven original bearings, but class 0 has only one bearing. The 65 endurance records are 819,200 points, versus metadata 819,600. C4A and E4A are the same bearing. | Original DIRG paper; manufacturer link in qualification CSV; `DATA_ANOMALIES.csv` |
| JUST has seven channels (six acceleration m/s² and one AE dB), while all 180 metadata rows claim three channels and an incompatible length. | Original CSV headers/paper; `DATA_ANOMALIES.csv` |
| PU has 32 documented bearings and a feasible fixed episode, but 1,517 H5 lengths differ from metadata. Id47567 has 256,823 samples, nonuniform original time increments and average rate 64,205.445 Hz; vibration Unit is empty. | Original measuring log/MAT; `RAW_ACQUISITION_CHECKS.csv`, `DATA_ANOMALIES.csv` |
| Ottawa23 has documented m/s², 42 kHz and 20 bearings. Its exact conditioner/filter settings are absent; sensor response alone does not establish the full acquisition passband. | Original paper §2.1/§3.4–3.6 and device-source links in qualification CSV |

All numerical support/query counts use the authoritative
`tii_target_head.make_episode(shots=5, seed=1729)`. Its failed fixed draws are retained
through `FixedEpisodeError`; no draw was repeated. Known insufficient recording
budgets fail before drawing; unknown physical groups or invalid task labels remain
explicitly unexecuted. Fold states are 13 `failure` and 6 `ineligible`.

| Dataset | Support records/class | Support groups/class | Query groups/class | Fixed episode |
| --- | --- | --- | --- | --- |
| Ottawa23 labels 0–4 | 5,5,5,5,5 | 5,4,4,4,4 | 3,1,1,0,1 | FAIL |
| UNSW labels 0–3 | 5,5,5,5 | 1,1,1,1 | 0,0,0,0 | FAIL |
| DIRG labels 0–2 | 5,5,5 | 1,3,3 | 0,0,0 | FAIL |
| PU labels 0–2 | 5,5,5 | 3,5,4 | 3,8,9 | PASS; dataset remains ineligible |

No common/increment bands were inferred from Nyquist or target signal spectra.
DIRG's documented sensor/DAQ passband is positive evidence, but response outside that
band is not automatically “unavailable.” Other channel, unit, physical-group,
sampling and end-to-end response gaps are identified separately for every dataset.
Original licenses and source restrictions are reported separately from the
collection's Apache-2.0 declaration.

Reproduce from the PHMFactory repository:

```bash
/usr/bin/time -v -o reports/tii_local_j1/qualification_resources.txt \
  conda run -n LQ_signal --no-capture-output python -m scripts.tii_qualify_data \
  > reports/tii_local_j1/qualification.log 2>&1
```

Final execution: exit 0; wall time 46.34 s; script inspection/output time 41.206 s;
maximum resident set 2,328,460 KiB (approximately 2.22 GiB); no GPU or training.
Output coverage checks passed for all 18 files, all 19 Dataset_ids, all 49,855 rows,
original-path/inode uniqueness, per-dataset native-reader probes and zero admitted
folds. Reader failures remain scientific input failures, not successful integration.
