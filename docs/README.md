# PHM-Vibench Documentation

The maintained documentation entrypoint is [docs/index.md](index.md).

Use it to find installation, quickstart, configuration, data, testing,
development, contribution, Streamlit, release, and historical documentation.

Key maintenance commands:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
```

`docs/CONFIG_ATLAS.md` is generated from `configs/config_registry.csv`; do not
edit the atlas manually.
