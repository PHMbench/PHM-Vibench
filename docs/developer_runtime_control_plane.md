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

Protected Pipeline code must consume `compiled_run_spec.runtime_config()` instead of
reparsing the source YAML or automatically discovering machine-local files. Until a
Pipeline is migrated, the legacy loader remains a compatibility implementation, not a
second public configuration authority.

## Shared classification runtime

Pipeline 01 and Pipeline 05 use one lifecycle implementation under
`src.runtime.classification`:

```text
load compiled config
  -> validate required sections
  -> construct data/model/task/trainer
  -> fit
  -> restore best checkpoint
  -> test and write result CSV
  -> close data and logging resources in finally
```

The Pipeline modules remain intentionally thin. Pipeline 01 selects the default shared
runtime. Pipeline 05 selects the same runtime plus `ExplainabilityHooks`, which own only
the UXFD-specific configuration snapshot, metadata snapshot, eligibility, and manifest
refresh. Hooks must represent genuine extensions; they must not duplicate configuration
loading, factory construction, training, testing, or cleanup.

Direct imports of the old Pipeline functions retain an explicit compatibility fallback
to the legacy config loader. The maintained public CLI path always uses the compiled
contract and therefore applies base configs and CLI overrides exactly once.

The following invariants govern later refactors:

1. the public config is compiled exactly once;
2. local configuration is an explicit input;
3. the selected Pipeline and executed configuration share one contract;
4. runtime code receives a mutable copy and cannot mutate the compiled contract;
5. missing sections, invalid iterations, invalid factory results, and execution failures
   raise rather than printing success;
6. data and logging resources are closed through a shared `finally` boundary;
7. successful runs must produce a mandatory minimal attestation.
