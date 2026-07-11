# Streamlit compatibility and extensibility review

The frontend was reviewed in two architectural slices: configuration/preflight and
process/results.

## Configuration findings accepted

1. Maintained keys do not always live at idealized paths. Learning rate currently
   lives at `task.lr`, while batch size and worker count normally live in `data.*`.
   Ordered aliases therefore belong in `field_catalog.yaml`, not UI conditionals.
2. Portable YAML must be resolved without machine-local values. Normal validation
   and execution then apply `configs/local/local.yaml` exactly once.
3. Edited YAML must use operating-system temporary directories during inspection,
   so validation does not dirty the checkout.
4. Precedence is explicit: portable YAML, local config, safe overrides, then raw
   overrides.
5. Unknown registry columns remain available as metadata for future UI features.
6. Registry paths are contained under `configs/`; traversal and symlink escapes are
   rejected.

## Process findings accepted

1. The service launches only the public `main.py --config` contract and always uses
   `shell=False`.
2. POSIX runs use a new session; Windows runs use a new process group. Cancellation
   targets the group, waits for a grace period, then force-kills.
3. Pause/resume was rejected because `SIGSTOP`/`SIGCONT` is not portable and can
   leave CUDA/data-loader workers inconsistent.
4. Process handles remain in an in-memory service registry, while durable state is
   stored in an atomic `run.json` manifest.
5. A detached POSIX process blocks a second run. It cannot be cancelled from a new
   worker because safe process ownership cannot be re-established.
6. Restart uses the immutable `execution.yaml` and override snapshot rather than
   the current UI state.

## Result findings accepted

1. Result discovery scans only the run directory and configured output root.
2. Repository roots, filesystem roots, home directories, and repository parents
   are rejected as overly broad scan targets.
3. Directory depth, entry count, file count, metric bytes, and metric rows are
   bounded.
4. Symbolic links are skipped to prevent artifact-path escape.
5. Missing, malformed, or oversized optional metric files become warnings and do
   not hide logs or process status.
6. New artifact/metric formats are isolated in `result_service.py`, preserving a
   stable UI and process layer.

## UI findings accepted

1. Visual styling is isolated from orchestration in `ui_theme.py`.
2. Live refresh and artifact rendering are isolated in `ui_runtime.py`.
3. The official entry point remains small and stable: `apps/streamlit/app.py`.
4. The legacy root command remains a compatibility shim instead of maintaining a
   second implementation.
5. The Run button is enabled only for the exact signature that passed repository
   inspection.

## Deferred

- remote execution and multi-user scheduling;
- persistent job queues or databases;
- online mutation of a running experiment;
- deletion of legacy `app/` prototype modules.
