# 2026-10-26 — KAIST/HUST23 source-typed MAT reader candidates

Former `RM_025_KAIST.py` and `RM_026_HUST23.py` were non-executable TODOs.
Implement source-grounded *bounded* numerical readers: KAIST constant-load
vibration four-axis MAT Signal layout, and HUST23 single-channel steady-state
MAT data. They fail on unsupported sources, rather than inferring a modality,
using a fault-coded filename, or conflating HUST23 shaft-frequency `fs`
with the 51.2 kHz acquisition rate. Add five synthetic positive/negative
reader tests. No data population, split, cache key, trained model, real
signal/diagnosis or release support claim is changed. Actual original
MAT/TDMS import qualification remains pending.
