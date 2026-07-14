# Internal Configuration Utilities (`src/configs/`)

This package contains the Python configuration-loading and experiment-helper
implementation. It is not the primary user documentation.

Use these maintained entrypoints instead:

- configuration composition and CLI usage: [`configs/README.md`](../../configs/README.md);
- first successful run: [`docs/quickstart.md`](../../docs/quickstart.md);
- generated shipped-config reference: [`docs/CONFIG_ATLAS.md`](../../docs/CONFIG_ATLAS.md);
- registry column contract: [`docs/config_registry_schema.md`](../../docs/config_registry_schema.md).

The public runtime command remains:

```bash
python main.py --config <yaml> [--override key=value ...]
```

## Package responsibilities

| Module | Responsibility |
|---|---|
| `config_utils.py` | Load YAML/config objects, merge values, apply local overrides, and expose `ConfigWrapper` utilities |
| `ablation_helper.py` | Generate local ablation/grid-search variants |
| `contrastive_config.py` | Compatibility helpers for contrastive-learning configuration |
| `deprecated/` | Historical implementations; not current extension targets |

Public exports are defined in `src/configs/__init__.py`. Read the implementation
and tests before relying on an export that is not used by the maintained CLI.

## Internal API examples

Load a file:

```python
from src.configs import load_config

config = load_config("configs/demo/00_smoke/dummy_dg.yaml")
```

Save a resolved config:

```python
from src.configs import save_config

save_config(config, "resolved.yaml")
```

Programmatic helpers are secondary to the config-first CLI. Do not introduce a
new loader or precedence model for one pipeline or component.

## Precedence and compatibility

The maintained user-facing precedence is documented in
[`configs/README.md`](../../configs/README.md). Internal code should preserve that
contract:

```text
base config
< selected YAML overrides
< optional configs/local/local.yaml
< CLI overrides
```

Legacy preset names and compatibility helpers may still exist in code. Their
presence does not make the old `configs/v0.0.9/` examples part of the current
release-supported surface.

## Contributor rules

- Add public configuration behavior through the existing loader and schema path.
- Keep the five sections `environment/data/model/task/trainer` stable.
- Add focused tests for precedence, type conversion, aliases, and invalid values.
- Prevent unknown keys or misspellings from being silently accepted when the
  schema can reject them.
- Do not hard-code personal paths, hostnames, or credentials.
- Put local/research configs under `configs/experiments/` first.
- Update `configs/config_registry.csv` and regenerate `docs/CONFIG_ATLAS.md` only
  for shipped configs.
- Treat code under `deprecated/` and old presets as compatibility/history, not
  examples for new development.

See [CONTRIBUTING.md](../../CONTRIBUTING.md) and
[Testing and evidence](../../docs/testing.md).
