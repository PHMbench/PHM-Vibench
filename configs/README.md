# Configuration System

This page is the maintained authority for PHMFactory configuration composition,
precedence, inspection, and registry maintenance.

## Public contract

Use a maintained preset or YAML file through the same public interface:

```bash
phmfactory preflight --config <preset-or-yaml> [--override key=value ...]
phmfactory --config <preset-or-yaml> [--override key=value ...]
```

The compatibility forms `python -m phmfactory` and `python main.py` use the same
configuration analysis. For installation and the first offline run, start with the
[installation guide](../docs/installation.md) and [quickstart](../docs/quickstart.md).

## Five-block model

Maintained configurations contain five logical mappings:

```yaml
environment: {}  # seed, iterations, output location, process settings
data: {}         # metadata, raw data, windowing, workers, sampling inputs
model: {}        # model family, components, and model hyperparameters
task: {}         # diagnosis/pretraining objective and task-specific protocol fields
trainer: {}      # device, epochs, precision, logging, checkpoint behavior
```

A top-level `pipeline` selects orchestration. New datasets, models, tasks, and trainers
should normally extend their factory rather than add a special branch to `main.py`.

## One configuration truth

Every maintained public consumer uses `phmfactory.config.analyze_config`:

```text
run
preflight
scripts.validate_configs
scripts.config_inspect
support generation
Pipeline 06 public adapter
Streamlit validation and launch
```

For the same source and explicit inputs, these consumers must obtain the same effective
configuration and `effective_config_sha256`.

The precedence order, from lowest to highest, is:

1. YAML files listed by `base_configs`, in declared order;
2. values in the selected experiment YAML;
3. an **explicitly supplied** machine-local YAML, when `--local-config` is present;
4. repeatable CLI `--override key=value` arguments.

There is no automatic search for `configs/local/local.yaml`. Hidden machine state would
make two apparently identical commands execute different experiments, so a local file is
an input only when it is visible in the command:

```bash
phmfactory preflight \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml \
  --override trainer.num_epochs=1
```

Equivalent `--local_config` spelling remains accepted for compatibility, but new
documentation uses `--local-config`.

## Effective identity and invocation identity

PHMFactory records two related hashes:

```text
effective_config_sha256
  = hash of the canonical fully resolved configuration

run_spec_sha256
  = identity of this invocation, including requested source and explicit overrides
```

Two commands that produce the same effective configuration share
`effective_config_sha256`, even when one uses a preset and another uses an explicit path.
The invocation hash may differ because it preserves how the user requested the run.

Use the effective hash for semantic comparisons. Use the run-spec hash and full command
for provenance and debugging.

## Maintained directories

- `configs/base/` — reusable environment, data, model, task, and trainer blocks;
- `configs/demo/` — maintained user-facing examples;
- `configs/experiments/` — local or research variants not automatically supported;
- `configs/local/` — optional untracked machine files that must be supplied explicitly;
- `configs/reference/` and versioned historical directories — reference material, not
  the maintained quickstart surface.

Start a local experiment by copying the nearest maintained demo into
`configs/experiments/` and changing only behavior-affecting values required by the
experiment.

## Preflight before execution

```bash
phmfactory preflight \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

The report includes:

```text
status=passed
effective_config_sha256=<64-hex>
run_spec_sha256=<64-hex>
pipeline=<canonical-name>
output_dir=<resolved-path>
```

Preflight does not import or execute the training Pipeline, construct factories, allocate
a GPU, create the configured output directory, or start a run.

## Inspect values, sources, and targets

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

Useful views:

```bash
python -m scripts.config_inspect --config <yaml> --dump resolved --format yaml
python -m scripts.config_inspect --config <yaml> --dump sources --format md
python -m scripts.config_inspect --config <yaml> --dump targets --format json
```

`resolved` is the exact effective mapping used by the runtime. `sources` identifies the
last source of each leaf field. `targets` reports the Pipeline and factory modules that
would be selected without constructing them. The inspector exits non-zero when a sanity
check fails.

## Validate maintained configurations

```bash
python -m scripts.validate_configs
```

The validator first uses the same public analysis, then applies the Pydantic experiment
schema. A config cannot pass validation through a different legacy merge path than the
one used for execution.

## Registry and generated documentation

The maintained config inventory is:

```text
configs/config_registry.csv
```

Related files:

- field reference: `docs/config_registry_schema.md`;
- generated human-readable inventory: `docs/CONFIG_ATLAS.md`;
- evidence-derived support boundaries: `SUPPORTED_COMPONENTS.md` and
  `SUPPORTED_COMBINATIONS.md`.

After an intentional registry change:

```bash
python -m scripts.gen_config_atlas
python -m scripts.validate_configs
python -m scripts.gen_support_matrix
git diff --exit-code \
  docs/CONFIG_ATLAS.md \
  SUPPORTED_COMPONENTS.md \
  SUPPORTED_COMBINATIONS.md
```

Do not hand-edit generated output to hide source drift. A registry row proves inventory,
not runtime or release support.

## Common explicit overrides

Run one epoch:

```bash
phmfactory --config <yaml> --override trainer.num_epochs=1
```

Use local data without editing a maintained demo:

```bash
phmfactory --config <yaml> \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx
```

Select a bounded CPU smoke setup:

```bash
phmfactory --config <yaml> \
  --override trainer.device=cpu \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Promoting a configuration

A config should enter the maintained registry only when its PR includes:

- a clear purpose and intended audience;
- valid five-block composition;
- no committed personal absolute paths;
- public analysis, schema validation, and inspection success;
- the smallest applicable runtime evidence;
- synchronized registry and generated documents;
- updated support or limitation documentation when the public surface changes.

Use `needs_smoke` or another explicit non-supported status until runtime evidence exists.
Do not label a config `sanity_ok` based only on YAML parsing.

## Read next

- Base blocks: `configs/base/README.md`
- Maintained demos: `configs/demo/README.md`
- Experiment variants: `configs/experiments/README.md`
- Explicit local overrides: `configs/local/README.md`
- Documentation index: `docs/index.md`
- Testing and evidence: `docs/testing.md`
