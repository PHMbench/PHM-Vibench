# Reviewer notes

This staged change received a compatibility and extensibility review before the
pull request was opened.

## Accepted findings

1. **Do not assume idealized key paths.** Maintained configs currently place the
   learning rate at `task.lr`, while batch size and worker count normally live in
   `data.*`. The field catalog therefore uses ordered aliases rather than UI
   conditionals.
2. **Do not dirty the checkout during validation.** Edited YAML is written to an
   operating-system temporary directory, not a repository-local cache folder.
3. **Do not apply machine-local config twice.** Advanced YAML starts from the
   resolved configuration. Validation supplies an explicit empty local override
   so `configs/local/local.yaml` is not merged a second time.
4. **Make override precedence explicit.** Full YAML is the base, safe fields are
   CLI overrides, and raw overrides have the highest precedence.
5. **Preserve registry evolution.** Unknown registry columns remain available as
   metadata; template groups and field aliases remain declarative.
6. **Keep execution isolated.** This first PR does not start training and never
   imports a Pipeline. It validates only through the existing public CLI.

## Deferred to stacked changes

- process lifecycle, cross-platform cancellation, and live log polling;
- bounded artifact discovery and run manifests;
- legacy entry-point deprecation and root dependency cleanup.
