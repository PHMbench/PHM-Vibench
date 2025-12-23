# trainer_factory

PyTorch Lightning trainer construction and orchestration.

This directory already has `src/trainer_factory/readme.md` and `src/trainer_factory/CLAUDE.md` with detailed guidance.
This `README.md` exists to satisfy the repo merge rule: whenever we change a folder, keep a top-level `README.md`
updated with the current responsibilities.

UXFD merge additions:
- `src/trainer_factory/extensions/manifest.py`: writes `artifacts/manifest.json` at the end of fit/test.

