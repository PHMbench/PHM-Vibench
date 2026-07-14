## Problem and rationale

<!-- What concrete problem does this PR solve? Why does it belong in PHM-Vibench? -->

## Scope

### Included

- 

### Explicit non-goals

- 

## Change type

- [ ] Bug or compatibility fix
- [ ] User-visible feature
- [ ] Configuration or registry change
- [ ] Dataset, reader, model, task, trainer, or pipeline contribution
- [ ] Test or CI change
- [ ] Documentation-only change
- [ ] Cleanup or removal
- [ ] Release preparation

## Public behavior and compatibility

<!-- Describe CLI, YAML, factory, tensor/batch, checkpoint, data, or artifact behavior that changes. Write "No public behavior change" when accurate. -->

```text
Migration required:
Deprecated behavior:
Compatibility layer:
Known incompatible cases:
```

The maintained entrypoint must remain:

```bash
python main.py --config <yaml> [--override key=value ...]
```

## Validation evidence

Use `PASS`, `FAIL`, `EXPECTED FAILURE`, or `NOT EXECUTED — <reason>`. Local results are not CI results.

| Command | Result | Environment or evidence link |
| --- | --- | --- |
| `python -m scripts.validate_docs` |  |  |
| `python -m scripts.validate_configs` |  |  |
| `python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md` |  |  |
| Focused tests |  |  |
| `python -m pytest test/ -q` |  |  |
| Offline dummy smoke |  |  |

Offline smoke command when applicable:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Tests

<!-- List new or changed tests and what behavior they assert. Do not only state that no exception was raised. -->

- 

## Documentation and sources of truth

- [ ] Updated the authoritative document rather than copying existing guidance.
- [ ] Updated `configs/config_registry.csv` when the maintained config inventory changed.
- [ ] Regenerated `docs/CONFIG_ATLAS.md` from the registry; did not hand-edit it.
- [ ] Updated `SUPPORTED_COMPONENTS.md`, `SUPPORTED_COMBINATIONS.md`, or `KNOWN_LIMITATIONS.md` when the release-supported surface changed.
- [ ] Added new maintained pages to `docs/index.md`.
- [ ] Commands, paths, links, configuration keys, and filenames were checked.

## Data, model, paper, and license review

<!-- Complete when external data, code, weights, papers, or model implementations are added or adapted. -->

```text
Source:
License:
Download or citation:
Redistribution constraints:
Preprocessing or conversion:
Reproducibility notes:
```

## Risk and rollback

```text
Primary risks:
Failure signals:
Rollback method:
Follow-up work:
```

## Final checklist

- [ ] The diff has one coherent primary purpose.
- [ ] No unrelated formatting or generated noise is included.
- [ ] No credentials, private data, personal absolute paths, raw datasets, caches, logs, local goal packs, or machine-only agent settings are included.
- [ ] Invalid inputs fail explicitly rather than through an unexplained deep runtime error.
- [ ] Test skips, mocks, exception catches, or tolerance changes do not hide the behavior under review.
- [ ] Performance, compatibility, maturity, and dataset claims are supported by linked evidence or clearly marked as unverified.
- [ ] A rollback is possible without losing unique branch or research history.
