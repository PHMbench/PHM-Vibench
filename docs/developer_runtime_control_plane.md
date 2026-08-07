# Runtime control plane

PHMFactory separates configuration resolution, execution, and minimal run recording so a
user command cannot be silently reinterpreted by a downstream Pipeline.

## Maintained public path

```text
preset or YAML
+ optional explicit --local-config
+ CLI --override values
  -> phmfactory.config.analyze_config
  -> ConfigAnalysis
  -> CompiledRunSpec
  -> Pipeline maturity gate
  -> canonical Pipeline module or narrow adapter
  -> protected src runtime
  -> run_manifest.json
```

The public path has one configuration authority. Run, preflight, inspection, validation,
support generation, Pipeline 06, and the optional Streamlit workspace consume the same
effective mapping.

## `ConfigAnalysis`: one configuration truth

`ConfigAnalysis` resolves:

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
```

The maintained precedence is:

```text
base_configs
< selected experiment YAML
< explicit --local-config YAML
< CLI --override values
```

No public component automatically searches for `configs/local/local.yaml`. Hidden local
files would make the same visible command execute different experiments on different
machines.

Correctness is checked by comparing the resolved configuration itself and the behavior it
produces. Runtime correctness does not depend on a configuration digest.

## `CompiledRunSpec`: configuration-to-runtime handoff

`CompiledRunSpec` owns a deep-copied runtime mapping plus the request information needed
for execution. Runtime adapters call:

```python
compiled_run_spec.runtime_config()
```

They must not re-read source YAML, rediscover local configuration, or reapply CLI
overrides.

The important contract is therefore:

```text
configuration resolution ends
-> CompiledRunSpec
-> execution consumes that configuration
```

not a second identity or verification system.

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

No exception may switch to another Pipeline, model, dataset adapter, task, or objective.

## Minimal run manifest

Each public experiment writes:

```text
<environment.output_dir>/.phmfactory/runs/<run-id>/run_manifest.json
```

The manifest is deliberately small. It records only information useful for understanding
whether the requested run completed:

```text
run ID and status
canonical Pipeline and imported module
requested config and resolved config path
explicit overrides
execution timestamps
structured failure information
```

Metrics, checkpoints, figures, generated signals, and other scientific outputs stay in
their owning experiment directories. The runtime does not build a second artifact index,
hash chain, evidence ledger, or attestation hierarchy around them.

A successful experiment is determined by the actual Pipeline lifecycle and evaluation,
not by post-hoc evidence registration.

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

## Pipeline 02

Pipeline 02 chooses exactly one mode before execution:

```text
compiled config without stages -> shared classification runtime
compiled config with stages    -> unified multi-stage orchestrator
explicit legacy_dual_yaml       -> compatibility adapter + orchestrator
```

An exception never changes the selected mode or algorithm. A completed stage must have a
valid checkpoint and a valid evaluation result; evaluation failure must propagate rather
than becoming an empty-metrics success.

## Pipeline 06 compiled-config adapter

The train/sample/eval science remains in:

```text
src.Pipeline_06_Generative_Modeling
```

The public descriptor imports the narrow adapter:

```text
phmfactory.runtime.pipeline06_adapter
```

Its responsibilities are limited to:

1. require the compiled Pipeline to be `Pipeline_06_Generative_Modeling`;
2. convert `compiled_run_spec.runtime_config()` to the namespace shape expected by the
   protected implementation;
3. dispatch the already selected `train`, `sample`, or `eval` stage;
4. preserve the Pipeline's own stage failure handling.

The public runtime does not re-index Pipeline 06 outputs into a second evidence system.

## Streamlit boundary

The optional UI edits values and calls the public configuration tools. It does not merge
base configs independently, discover a local YAML, import a Pipeline, or construct a
Trainer.

The displayed reproduction command is the command passed to the public runtime. Edited
Advanced YAML is a standalone effective config and contains no invisible local layer.

## Pipeline maturity

`phmfactory.pipelines.PIPELINE_DESCRIPTORS` separates discoverability, opt-in execution,
and release support. Pipeline 03 and Pipeline 04 require:

```bash
phmfactory --config <yaml> --allow-experimental
```

The flag acknowledges maturity; it does not promote support. Execution smoke and
scientific protocol validity remain separate claims.

## Invariants for future changes

1. Public configuration composition has one implementation.
2. Machine-local YAML is an explicit input.
3. Run, preflight, inspect, validate, UI, and Pipeline adapters use the same effective
   configuration, not separate hash-based identities.
4. Overrides are applied exactly once.
5. Runtime code receives a copy and cannot mutate the compiled contract.
6. Errors propagate from their source; no exception activates another algorithm.
7. `None` is not a successful Pipeline result.
8. Resources close through `finally` boundaries.
9. Each public run keeps one minimal terminal run manifest.
10. Discoverable, runnable, supported, baseline-valid, and benchmark-ready remain distinct
    claims.
