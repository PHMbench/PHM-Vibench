# Runtime control plane

PHMFactory's maintained entrypoints compile user intent before protected runtime code is
imported:

```text
config or preset + CLI overrides
  -> phmfactory.config.resolve_config
  -> phmfactory.runtime.CompiledRunSpec
  -> canonical Pipeline adapter
  -> protected src runtime
```

`CompiledRunSpec` is the hand-off boundary for the ongoing runtime consolidation. It
contains the fully composed configuration, canonical Pipeline identifier, explicit
overrides, and a deterministic semantic SHA-256. Its hash excludes the absolute
installation path, so an identical packaged preset has the same identity outside a
repository checkout.

Protected Pipeline code must gradually consume `compiled_run_spec.runtime_config()`
instead of reparsing the source YAML or automatically discovering machine-local files.
Until a Pipeline is migrated, the legacy loader remains a compatibility implementation,
not a second public configuration authority.

The following invariants govern later refactors:

1. the public config is compiled exactly once;
2. local configuration is an explicit input;
3. the selected Pipeline and executed configuration share one contract;
4. runtime code receives a mutable copy and cannot mutate the compiled contract;
5. failures must return a non-zero process outcome;
6. successful runs must produce a mandatory minimal attestation.
