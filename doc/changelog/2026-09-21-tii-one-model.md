# Bounded TII single-model command

`scripts/tii_one_model.py` adds one concrete local action: jointly train one native
support-conditioned model on 2–5 qualified industrial sources, seed 0 and physical
GPU0. It never expands a source list into separate fits or an experiment sweep.
The native config compiler, Data/Model/Task/Trainer and checkpoint authority remain
unchanged. Source-local heads belong to the same shared model.

Missing local configuration reports the existing five-candidate admission facts
without rescanning raw data. Current saved admission remains No-Go. Configured runs
reject unqualified sources, fixtures, multiple fits/devices, extra datasets and reused
output folders. Native failure/interrupt logs are retained; no fallback or retry.
The output is training-only, not an S/U effect, target transfer or benchmark result.

Focused launcher tests include native-compiler rejection of the shipped fixture;
subprocess-stub tests check orchestration only. No new industrial training result is
asserted. Usage: `configs/experiments/tii/ONE_MODEL.md`.
