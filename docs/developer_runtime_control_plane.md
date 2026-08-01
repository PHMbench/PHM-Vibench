# Runtime control plane

PHMFactory's maintained entrypoints compile user intent before protected runtime code is
imported:

```text
config or preset + CLI overrides
  -> phmfactory.config.resolve_config
  -> phmfactory.runtime.CompiledRunSpec
  -> phmfactory.runtime.ExecutionEnvelope
  -> phmfactory.runtime.RunAttestation (pending)
  -> canonical Pipeline adapter
  -> protected src runtime
  -> RunAttestation (succeeded or failed)
```

`CompiledRunSpec` contains the fully composed configuration, canonical Pipeline
identifier, explicit overrides, and a deterministic semantic SHA-256. Its hash excludes
the absolute installation path, so an identical packaged preset has the same identity
outside a repository checkout.

`ExecutionEnvelope` is the public invocation boundary. It records pending, running,
succeeded, or failed state and rejects ambiguous outcomes. A Pipeline module must expose
a callable `pipeline(args)` and a successful invocation must return an explicit result.
Returning `None`, omitting the entrypoint, or attempting to execute the same envelope
twice is a contract error. Exceptions retain their original traceback while the envelope
records the failure stage, type, and message.

The public CLI prints the completion message only after the envelope reaches
`succeeded`. A failed Pipeline therefore produces a non-zero process outcome rather than
`print + return` followed by apparent success.

## Mandatory invocation manifest

After configuration compilation and before Pipeline import, the CLI creates:

```text
<environment.output_dir>/.phmfactory/runs/<run_id>/run_manifest.json
```

The first atomic version has `status: pending`. The same path is atomically replaced with
`succeeded` or `failed` after execution. Each manifest contains the run ID, run-spec hash,
canonical Pipeline and module, config source, explicit overrides, code revision when
available, Python/platform summary, execution timestamps, and structured failure data.

Manifest writes use a same-directory temporary file, flush and `fsync`, followed by
`os.replace`. If the pending manifest cannot be created, the Pipeline is not imported. If
the final succeeded manifest cannot be written, the public invocation fails and no
completion message is printed. This makes the minimum attestation mandatory rather than
best effort.

The first schema intentionally keeps `data`, `protocol`, `seed`, and `environment`
evidence as nested extension points. Pipeline-specific evidence such as the Pipeline 06
stage ledger or UXFD explainability artifacts should be referenced from this manifest in
later bounded PRs; they must not create another top-level run identity.

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

## Pipeline 02 mode selection

Pipeline 02 has exactly three recognizable inputs and never changes mode after an error:

```text
compiled config without stages -> shared single-stage classification runtime
compiled config with stages    -> one unified multi-stage orchestrator
explicit legacy_dual_yaml       -> isolated dual-YAML adapter + one orchestrator
```

The maintained public demo currently has no `stages` list and therefore uses the same
classification lifecycle as Pipeline 01/05. A non-empty `stages` list selects the
orchestrator before execution. Orchestrator errors are propagated; they do not activate a
manual pretrain/few-shot implementation or another algorithm.

The historical `fs_config_path` input is rejected unless direct legacy callers also set
`pipeline_mode=legacy_dual_yaml`. The public CLI does not expose this mode. Manual
`run_stage`, `run_pretraining_stage`, `run_fewshot_stage`, and duplicate single-stage
functions are no longer maintained entrypoints.

The following invariants govern later refactors:

1. the public config is compiled exactly once;
2. local configuration is an explicit input;
3. the selected Pipeline and executed configuration share one contract;
4. runtime code receives a mutable copy and cannot mutate the compiled contract;
5. missing sections, invalid iterations, invalid factory results, `None` Pipeline
   results, and execution failures raise rather than printing success;
6. data and logging resources are closed through a shared `finally` boundary;
7. every compiled invocation creates one mandatory minimal attestation;
8. Pipeline-specific evidence extends the invocation manifest instead of redefining it;
9. an exception never changes the selected execution mode or activates a fallback.
