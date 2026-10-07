# P01 experiment control over existing DG execution

`experiments/p01/control.py` plans and invokes the existing `multiview_dg` stage
commands; the paper exposes it through its existing `fusion.sh dg plan|run|status`.
No model, loss, data protocol, search grid, qualification rule or metric is changed.
Manual task/arm/final-seed selection applies only to source HPO/fitting. Complete
source qualification and the original all-task/all-arm/all-seed freeze still gate
any target release; the controller additionally requires explicit target permission.

Process records and per-stage logs support stop/resume and fail-fast behavior. A
per-root lock prevents concurrent controllers. Signals reach the stage process group.
The existing trainers validate exact completed work; HPO also rechecks every recorded
trial before accepting an existing winner. Explicit interrupted-trial restart keeps
old outputs and restarts the identical configuration/seed from initialization, not
from an incomplete optimizer state. Numerical failures are never silently retried.
Some non-transactional stages still require inspection after interruption; no cleanup
or target re-selection is automated. Documentation states these recovery boundaries.

Clean installation exposed an unrelated dependency incompatibility before numerical
execution: unbounded Transformers resolved to 5.19.0 and its accelerator probe raised
on the declared PyTorch 2.6 CPU build during Lightning/TorchMetrics import. The
runtime requirements now pin Transformers 4.57.1. No device API monkeypatch, skipped
import, replacement model or CPU/GPU fallback is used. Existing installations must
resolve the declared dependencies before a new study, not change versions mid-run.

Focused tests cover read-only planning, filters and unchanged study population,
parameter/target guards, logs/exit codes, locks, process-group termination, original
constructed data admission and interrupted-versus-failed run handling. Software
verification does not admit a real dataset or establish industrial performance.
