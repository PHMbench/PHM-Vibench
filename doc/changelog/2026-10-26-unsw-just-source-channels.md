# 2026-10-26 — UNSW and JUST source-channel contracts

Previously, UNSW silently omitted missing MATLAB keys and shifted later
channel indices; the Reader supported only v2/v3 `loadCell` despite the
published v1 `load` variable. The source-backed Reader now requires six
aligned channels, normalizing only that documented name change, and rejects
missing/ambiguous/misaligned/nonfinite input. It verifies the MAT `Fs`
sampling-rate scalar without exposing filename-derived labels.

JUST's publisher describes seven **measured** columns (six acceleration,
one acoustic emission). The historical Reader dropped the first channel
unconditionally. For the specifically declared v1 numeric seven-column
layout the correction preserves all seven and the first data row. Possible
header/time/index variants fail until verified from a real source CSV.

Synthetic test-only fixtures verify version aliases, ordering, rejection
and unmodified numeric output. These changes alter affected H5 bytes and
require deliberate cache regeneration; they do not establish real-data
import/calibration or any bearing/diagnostic benchmark validity.
