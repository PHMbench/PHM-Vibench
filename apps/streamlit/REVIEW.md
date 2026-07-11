# Reviewer notes

This staged change received a compatibility and extensibility review before the
pull request was opened and again before the execution layer was stacked.

## Accepted findings

1. **Do not assume idealized key paths.** Maintained configs currently place the
   learning rate at `task.lr`, while batch size and worker count normally live in
   `data.*`. The field catalog therefore uses ordered aliases rather than UI
   conditionals.
2. **Do not dirty the checkout during validation.** Edited YAML is written to an
   operating-system temporary directory, not a repository-local cache folder.
3. **Keep machine-local config as a runtime layer.** Advanced YAML is resolved
   with an explicit empty local override so the editable document stays portable.
   Normal validation and execution then allow `configs/local/local.yaml` to be
   merged exactly once by the core loader.
4. **Make precedence explicit.** Portable YAML is the base, machine-local config
   is applied next, catalog-safe CLI overrides follow, and raw CLI overrides have
   the highest precedence.
5. **Preserve registry evolution.** Unknown registry columns remain available as
   metadata; template groups and field aliases remain declarative.
6. **Keep execution isolated.** This first PR does not start training and never
   imports a Pipeline. It validates only through the existing public CLI.
7. **Bound cache staleness.** Registry, catalog, and inspection cache entries use
   a short TTL so edits to configs or local overrides are not hidden indefinitely.

## Deferred to stacked changes

- process lifecycle, cross-platform cancellation, and live log polling;
- bounded artifact discovery and run manifests;
- legacy entry-point deprecation and root dependency cleanup.
