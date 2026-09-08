# Historical reference configurations

These files are not maintained onboarding templates. Use [configs/demo](../demo/README.md)
for a new experiment. A historical file may require explicit migration before it is
accepted by the current schema or executable by the selected components.

Inspect an explicitly selected file before attempting execution:

```bash
phmfactory preflight --config <historical-config.yaml>
```

Preflight is an early check, not proof of runtime compatibility or scientific validity.
Do not restore a legacy loader, fallback or implicit default to make an old example pass.
Retain useful research choices in an explicitly migrated experiment; do not treat old
migration suggestions as an automatic task queue.
