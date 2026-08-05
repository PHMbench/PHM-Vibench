# Runtime control plane

PHMFactory separates user intent, execution, and run records so one command cannot be
silently reinterpreted by a downstream Pipeline.

## Maintained public path

```text
preset or YAML
+ optional explicit --local-config
+ CLI --override values
  -> phmfactory.config.analyze_config
  -> ConfigAnalysis
  -> CompiledRunSpec
  -> RunAttestation (pending)
  -> Pipeline maturity gate
  -> canonical Pipeline module or narrow adapter
  -> protected src runtime
  -> Pipeline-specific evidence registration
  -> RunAttestation (succeeded or failed)
```

The public path has one configuration authority. Validators, inspectors, preflight, the
CLI, support generation, Pipeline 06, and the optional Streamlit workspace all consume the
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

The effective hash includes only the canonical effective mapping. It deliberately excludes
preset spelling, source path, installation path, and provenance metadata. Consequently:

```text
--config smoke
```

and:

```text
--config configs/demo/00_smoke/dummy_dg.yaml
```

share `effective_config_sha256` when their final mappings are equal.

## `CompiledRunSpec`: runtime and invocation identity

`CompiledRunSpec` contains a deep-copied runtime mapping and two hashes:

```text
effective_config_sha256
  identifies the scientific configuration semantics

sha256 / run_spec_sha256
  identifies this invocation, including requested source and explicit overrides
```

Runtime adapters call `compiled_run_spec.runtime_config()` to obtain a mutable copy. They
must not re-read the source YAML or reapply CLI overrides.

The absolute resolved config path remains available for diagnostics but is excluded from
both identities. An installed preset therefore behaves consistently across checkout and
wheel locations.

## Execution boundary

`ExecutionEnvelope` records the finite states:

```text
pending -> running -> succeeded
                   -> failed
```

A Pipeline module must expose `pipeline(args)` and return an explicit result. Returning
`None`, omitting the callable, or executing one envelope twice is a contract error.
Exceptions keep their traceback while the envelope records failure stage, type, and
message.

The public process prints the completion message only after execution, evidence
registration, and final manifest writing succeed. A printed warning followed by `return
None` cannot become a successful command.

## Mandatory run manifest

Before Pipeline import, PHMFactory creates:

```text
<environment.output_dir>/.phmfactory/runs/<run-id>/run_manifest.json
```

The pending file is atomically replaced with the terminal state. The manifest includes:

```text
run ID and status
run_spec_sha256
effective_config_sha256
canonical Pipeline and imported module
requested source, resolved path, and overrides
code revision when available
Python/platform summary
execution timestamps
structured failure information
indexed artifacts and evidence sections
```

Writes use a same-directory temporary file, flush, `fsync`, and `os.replace`. Failure to
create the pending manifest prevents Pipeline import. Failure to write the final success
state prevents a success claim.

## Shared classification runtime

Pipeline 01 and Pipeline 05 use one lifecycle under `src.runtime.classification`:

```text
consume compiled config
-> validate required blocks
-> build data/model/task/trainer
-> fit
-> restore best checkpoint
-> test and write metrics
-> close data and logging resources in finally
```

Pipeline 01 is a thin default adapter. Pipeline 05 adds only explainability hooks. Hooks
must not duplicate config loading, factory construction, training, testing, or cleanup.

Pipeline 02 chooses exactly one mode before execution:

```text
compiled config without stages -> shared classification runtime
compiled config with stages    -> unified multi-stage orchestrator
explicit legacy_dual_yaml       -> compatibility adapter + orchestrator
```

An exception never changes the selected algorithm.

## Pipeline 06 compiled-config adapter

The train/sample/eval science remains in:

```text
src.Pipeline_06_Generative_Modeling
```

The public descriptor imports the narrow adapter:

```text
phmfactory.runtime.pipeline06_adapter
```

Its only responsibilities are:

1. require the compiled Pipeline to be `Pipeline_06_Generative_Modeling`;
2. convert `compiled_run_spec.runtime_config()` to the namespace shape expected by the
   protected implementation;
3. dispatch the already selected `train`, `sample`, or `eval` stage;
4. preserve stage-ledger failure handling.

It does not re-read YAML, apply overrides a second time, discover local files, or alter
the generative algorithm. Direct imports of the historical `src` function keep an
explicit compatibility loader, but that path is not the maintained CLI authority.

## Streamlit boundary

The optional UI edits values and calls the public inspector. It does not merge base
configs in Python UI code, discover a local YAML, import a Pipeline, or construct a
Trainer.

The UI receives the same `effective_config_sha256` as CLI preflight. The displayed
reproduction command is the command passed to the public runtime. Edited Advanced YAML is
a standalone effective config and contains no invisible local layer.

## Evidence convergence

`RunAttestation` is the only invocation-level run identity. Metrics, checkpoints,
explainability files, stage ledgers, synthetic manifests, and evaluation reports remain
separate artifacts indexed through:

```python
run_attestation.register_artifact(role=..., path=..., sha256=..., metadata=...)
run_attestation.set_evidence(section, value)
run_attestation.append_evidence(section, value)
```

A Pipeline-specific file extends the run; it does not create another top-level run
identity. Missing required evidence changes the invocation to failed even when model code
completed.

## Pipeline maturity

`phmfactory.pipelines.PIPELINE_DESCRIPTORS` separates discoverability, opt-in execution,
and release support. Pipeline 03 and Pipeline 04 require:

```bash
phmfactory --config <yaml> --allow-experimental
```

The flag acknowledges maturity; it does not promote support. Pipeline 01 and the
maintained single-stage Pipeline 02 path remain the release-supported surface. Pipeline
05, Pipeline 06, and Pipeline_ID remain outside the maintained combination table unless
current evidence promotes an exact configuration.

## Invariants for future changes

1. Public configuration composition has one implementation.
2. Machine-local YAML is an explicit input.
3. Run, preflight, inspect, validate, UI, and Pipeline 06 share the effective hash.
4. Overrides are applied exactly once.
5. Runtime code receives a copy and cannot mutate the compiled contract.
6. Errors propagate from their source; no exception activates another algorithm.
7. `None` is not a successful Pipeline result.
8. Resources close through `finally` boundaries.
9. Every public run creates one mandatory terminal manifest.
10. Discoverable, runnable, supported, and benchmark-valid remain distinct claims.
