# Contributing Models

This document describes how to add new model implementations to **PHMbench**.

## Workflow
1. Fork the repository and create a branch such as `feature/model-<name>`.
2. Place your model code under `src/model_factory/<Family>/<ModelName>.py`.
3. Register the model in `src/model_factory/model_factory.py` so it can be instantiated via the factory.
4. Provide an example configuration file in `configs/` demonstrating how to train the model.
5. Add tests or run `main_dummy.py` to ensure the model integrates correctly.
6. Follow PEP&nbsp;8 style and document your classes and functions.

## Example Skeleton
```python
class YourModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # build layers

    def forward(self, x):
        return self.layers(x)
```

## Contribution Checklist
- [ ] Model file added under `src/model_factory`.
- [ ] Factory registration updated.
- [ ] Example config or documentation provided.
- [ ] Tests pass or example run succeeds.

For any questions, open an issue or discussion thread.
