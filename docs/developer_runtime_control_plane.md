# Runtime control plane

PHMFactory separates configuration authority, scientific execution, and diagnostic run
records. The central invariant is:

```text
requested experiment = executed experiment
```

A Pipeline failure must propagate from its source. A diagnostic-file failure must not
rewrite a completed scientific run as failed.

## Maintained public path

```text
preset or YAML
+ optional explicit --local-config
+ CLI --override values
  -> phmfactory.config.analyze_config
  -> ConfigAnalysis
  -> CompiledRunSpec
  -> ExecutionEnvelope
  -> optional pending diagnostic manifest
  -> Pipeline maturity gate
  -> canonical Pipeline module or narrow adapter
  -> protected src runtime
  -> optional evidence indexing
  -> optional terminal diagnostic manifest
```

Configuration composition has one authority. Validators, inspectors, preflight, the CLI,
support generation, Pipeline adapters, and the optional Streamlit workspace consume the
same effective mapping.

## `ConfigAnalysis`: one configuration truth

`ConfigAnalysis` records:

```text
requested source or preset
resolved YAML path
fully effective configuration
canonical Pipeline
explicit overrides
optional explicit local-config path
ordered source files
last source of each leaf field
diagnostics
effective_config_sha256
```

The maintained precedence is:

```text
base_configs
< selected experiment YAML
< explicit --local-config YAML
< CLI --override values
```

No public component automatically searches for `configs/local/local.yaml`. Hidden files
would make the same visible command execute different experiments on different machines.

The effective configuration hash remains a diagnostic identity for config-parity checks.
It is not a security claim and does not determine whether training, checkpoint selection,
evaluation, or metrics succeeded.

## `CompiledRunSpec`: configuration-to-runtime boundary

`CompiledRunSpec` contains a deep-copied runtime mapping. Runtime adapters call:

```python
compiled_run_spec.runtime_config()
```

They must not re-read source YAML, discover a local file, or reapply CLI overrides. The
existing spec hash is retained for compatibility and config-parity tooling; it is not a
run-success gate.

## Scientific execution authority

`ExecutionEnvelope` records:

```text
pending -> running -> succeeded
                   -> failed
```

A Pipeline module must expose `pipeline(args)` and return an explicit result. Returning
`None`, omitting the callable, or executing one envelope twice is a contract error.
Exceptions preserve their original traceback while the envelope records failure stage,
type, and message.

For maintained classification paths, success is governed by the scientific lifecycle:

```text
fit completed
+ best checkpoint exists and loads
+ evaluation completed
+ returned metrics are non-empty and finite
```

The public completion message is printed only after the Pipeline returns successfully.
No evidence adapter, artifact index, digest, receipt, ledger index, or diagnostic manifest
may convert that result into a failed run.

## Diagnostic run manifest during v0.3 migration

The historical writer may create:

```text
<environment.output_dir>/.phmfactory/runs/<run-id>/run_manifest.json
```

It is now optional diagnostic output rather than an execution authority:

- preparation failure emits a warning and execution continues;
- Pipeline-specific evidence-indexing failure emits a warning and execution continues;
- terminal write failure emits a warning and preserves the Pipeline result;
- a Pipeline exception remains authoritative and is never replaced by a manifest-write
  exception.

When available, the compatibility manifest can still contain config identities, indexed
artifacts, and evidence sections. Those fields are not proof of scientific correctness.
The maintained results remain the actual checkpoints, test metrics, split information,
and aggregate run summary produced by the scientific path.

A later bounded cleanup may replace this compatibility file with a smaller `run_status`
record. That change must not introduce a second runtime or make status-file writing a
success condition.

## Shared classification runtime

Pipeline 01 and Pipeline 05 use one lifecycle under `src.runtime.classification`:

```text
consume compiled config
-> validate required blocks
-> build data/model/task/trainer
-> fit
-> restore best checkpoint
-> test and write metrics
-> aggregate repeated seeds
-> close data and logging resources in finally
```

Pipeline 01 is the default adapter. Pipeline 05 adds explainability hooks. Hooks must not
duplicate config loading, factory construction, training, testing, or cleanup.

Pipeline 02 chooses exactly one mode before execution:

```text
compiled config without stages -> shared classification runtime
compiled config with stages    -> unified multi-stage orchestrator
explicit legacy_dual_yaml       -> compatibility adapter + orchestrator
```

An exception never changes the selected algorithm.

## Pipeline 06 boundary

The train/sample/eval implementation remains in:

```text
src.Pipeline_06_Generative_Modeling
```

The public descriptor imports the narrow adapter:

```text
phmfactory.runtime.pipeline06_adapter
```

The adapter consumes the compiled configuration and dispatches the already selected
stage. Pipeline 06 may define internal scientific or stage-completion requirements.
However, after the Pipeline explicitly returns success, failure of the optional CLI
evidence index to find or register a stage ledger no longer changes the invocation to
failed.

## Streamlit boundary

The optional UI edits values and calls the public config/runtime entrypoint. It does not
merge base configs independently, discover local YAML, construct a Trainer, or create a
second success definition.

## Compatibility evidence API

`RunAttestation` and `phmfactory.runtime.evidence` remain importable during migration for
legacy tests and direct research tooling. They are not called as mandatory authorities by
the public scientific result.

New maintained code should not add:

```text
hash chains
receipts
ledgers as global success gates
artifact-integrity auditing
new evidence registries
```

Tests should instead protect data population, split disjointness, model/task identity,
objective participation, checkpoint restoration, estimator definition, and finite metrics.

## Pipeline maturity

`phmfactory.pipelines.PIPELINE_DESCRIPTORS` separates discoverability, opt-in execution,
and maintained support. Pipeline 03 and Pipeline 04 require explicit opt-in. The flag
acknowledges maturity; it does not promote scientific support.

Exact execution and protocol claims are configuration-specific and listed in:

- `SUPPORTED_COMPONENTS.md`
- `SUPPORTED_COMBINATIONS.md`
- `configs/config_registry.csv`

## Invariants for future changes

1. Public configuration composition has one implementation.
2. Machine-local YAML is an explicit input.
3. Overrides are applied exactly once.
4. Runtime code receives a copy and cannot mutate the compiled contract.
5. Errors propagate from their source; no exception activates another algorithm.
6. `None` is not a successful Pipeline result.
7. Resources close through `finally` boundaries.
8. Pipeline success is determined by scientific execution, not diagnostic finalization.
9. Diagnostic write failure never replaces the original Pipeline exception.
10. Discoverable, runnable, execution-verified, and baseline-valid remain distinct claims.
