# Contributing Trainers

The trainer factory creates PyTorch Lightning trainers used by PHMbench. Use this guide when adding a new training strategy.

## Workflow
1. Fork the repository and create a branch `feature/trainer-<name>`.
2. Implement the trainer class in `src/trainer_factory/<TrainerName>.py` and inherit from an existing base trainer.
3. Register the trainer in `src/trainer_factory/trainer_factory.py` so it can be selected via configuration.
4. Provide tests under `test/` to verify training works as expected.
5. Document usage in the README or within this directory.

## Required Interface
Your trainer should expose at least:
```python
class YourTrainer(LightningTrainer):
    def __init__(self, config):
        super().__init__(config)
    # override fit/validate if necessary
```

## Contribution Checklist
- [ ] Trainer file added and registered.
- [ ] Example config or docs updated.
- [ ] Tests or minimal run succeed.
- [ ] Tests pass or example run succeeds.
- [ ] Documentation updated if necessary.
- [ ] Code follows PEP8 and includes docstrings.


Questions can be raised through GitHub issues.
