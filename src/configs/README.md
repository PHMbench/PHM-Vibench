# Legacy configuration compatibility layer

`src/configs/` is the protected compatibility implementation used by direct historical
Pipeline calls and older scripts. It is **not** a second public configuration authority.

New code must use:

```python
from phmfactory.config import analyze_config
```

or a public command:

```bash
phmfactory preflight --config <preset-or-yaml>
phmfactory --config <preset-or-yaml>
```

The maintained semantics are documented in [`configs/README.md`](../../configs/README.md).

## Compatibility API

The historical API remains available:

```python
from src.configs import load_config
```

`load_config(...)` accepts legacy input forms and returns `ConfigWrapper`, which supports
attribute and mapping-style access. Keep it only where a protected direct-call path still
requires that interface.

Do not use this loader in new validators, inspectors, UI code, support generation, or
public Pipeline adapters. Those paths must consume `ConfigAnalysis` or
`CompiledRunSpec.runtime_config()` so base configs and overrides are not applied twice.

## Historical presets

`PRESET_TEMPLATES` still maps these names to `configs/v0.0.9/` for historical-script
compatibility:

```text
quickstart
basic
isfm
gfs
pretrain
id
```

They are not the PHMFactory v0.3 maintained presets. New callers should use public
presets such as `smoke` or a maintained YAML path through `phmfactory.config`.

`configs/v0.0.9/` remains while compatibility references exist. Removing it requires a
separate migration with zero runtime references and downstream validation.

## Machine-local values

Do not add personal paths or credentials to this compatibility package. Public machine
inputs are explicit:

```bash
phmfactory \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml
```

No public command automatically reads `configs/local/local.yaml`. CLI overrides remain
the highest-precedence input.

## Validation

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config <yaml>
python -m pytest test/test_config_analysis_parity.py -q
```

These commands use the public resolver, not this compatibility loader.
