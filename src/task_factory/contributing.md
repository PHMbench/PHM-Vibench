# Contributing Tasks

Tasks define how models are trained and evaluated in **PHMbench**. This guide explains how to add a new task or task component.

## Workflow
1. Fork the repository and create a branch `feature/task-<name>`.
2. Implement your task under `src/task_factory/<TaskName>/` from `Default_task` or as a `pytorch_lightning.LightningModule` subclass.
3. Register the task in `src/task_factory/task_factory.py` so it can be instantiated by name.
4. Reuse components from `src/task_factory/Components/` when possible. Add new losses or metrics there if needed.
5. Provide an example configuration in `configs/` and add tests exercising training and evaluation.
6. Ensure your code follows PEP8 and includes docstrings.

## Basic Structure
```python
class YourTask(pl.LightningModule):
    def __init__(self, network, cfg):
        super().__init__()
        self.network = network
        # initialise losses or metrics

    def training_step(self, batch, batch_idx):
        # implement training logic

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.lr)
```

## Contribution Checklist
- [ ] Task implemented under `src/task_factory`.
- [ ] Task registered in `task_factory.py`.
- [ ] Optional components added or reused.
- [ ] Example config and tests provided.
- [ ] Tests pass or example run succeeds.
- [ ] Documentation updated if necessary.
- [ ] Code follows PEP8 and includes docstrings.

Please open an issue if you have questions about the design or implementation.
