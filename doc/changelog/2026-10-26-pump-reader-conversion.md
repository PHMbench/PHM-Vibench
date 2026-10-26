# 2026-10-26 — Pump reader ndarray conversion

A legacy `RM_011_pump` XLSX reader obtained `pandas.DataFrame.values`
(a NumPy array) and then called `.float()`, which NumPy arrays do not provide.
Convert the same row-filtered array to finite-compatible float64 using NumPy,
retaining last-sheet selection, first-row exclusion, output shape and channel
order. A synthetic XLSX test exercises correct rows and invalid numeric input.
No raw pump asset was accessed. This is a reader software correction, not
physical-channel/roller-geometry qualification or a real-data PHM baseline.
