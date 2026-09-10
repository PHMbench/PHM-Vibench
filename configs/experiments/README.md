# Research experiment configurations

Start from the nearest example in [configs/demo](../demo/README.md), then copy it into a
named subdirectory here. Record the actual data, objective and evaluation behavior; a
research filename or directory is not evidence that its named method is implemented.

Inspect the exact complete configuration before running:

```bash
phmfactory preflight --config configs/experiments/<name>/exp.yaml
```

For machine-specific paths, pass `--local-config <local.yaml>` explicitly. Keep credentials
and personal absolute paths out of shared files. Composition and override order are in
[the configuration guide](../README.md).

A private prototype need not be registered. To promote a configuration into a maintained
demo, provide its real execution and protocol boundary, then update the existing registry
and generated navigation. `python -m scripts.validate_configs` validates its selected
maintained set; it is not proof that every unregistered local file here was tested.
