# Legacy configuration compatibility layer

`src/configs/` is the protected v0.3 compatibility implementation used by existing
Pipelines and historical scripts. It is not the primary public configuration API.

Use the maintained public surfaces for new integrations:

```python
from phmfactory.config import resolve_config
```

```bash
python main.py --config <yaml> [--override key=value ...]
```

The authoritative configuration documentation is
[`configs/README.md`](../../configs/README.md). Maintained examples live under
`configs/demo/`, compose reusable blocks through `base_configs`, and are inventoried
in `configs/config_registry.csv`.

## Compatibility API

The existing internal API remains available to the protected runtime:

```python
from src.configs import load_config
```

`load_config(...)` accepts the legacy input forms implemented by
`src/configs/config_utils.py` and returns a `ConfigWrapper` compatible with current
Pipeline code. Do not remove or redesign this layer as part of repository cleanup.

## Historical presets

`PRESET_TEMPLATES` still maps the following names to files under
`configs/v0.0.9/` for historical-script compatibility:

```text
quickstart
basic
isfm
gfs
pretrain
id
```

Those presets are not the maintained PHMFactory v0.3 quickstart surface. New code
should select a maintained YAML file or public preset through `phmfactory.config`.

`configs/v0.0.9/` must remain while these compatibility mappings exist. Removing it
requires a separate migration with zero runtime references and explicit downstream
evidence.

## Scope boundary

Do not add new public presets, machine-specific paths, or credentials here. New
configuration behavior belongs in the public resolver and maintained root
configuration tree; machine-specific values belong in `configs/local/local.yaml` or
CLI overrides.

Validation and inspection commands:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config <yaml>
python -m scripts.gen_config_atlas
```

The previous long-form v5 compatibility guide is preserved in immutable Git history
and in the approved personal-fork archive used for the v0.3 repository migration.
