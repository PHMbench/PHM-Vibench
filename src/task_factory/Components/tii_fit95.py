"""Read-only source-fitting observations inside the existing native fit.

No new runner, data loader, optimizer, checkpoint selection or target adapter.
The native source-validation monitor remains the only selection criterion.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import pytorch_lightning as pl

from src.task_factory.Components.tii_evaluation import evaluate_source_predictions

FIT95_STEPS = (0, 20, 100, 500, 1000, 2500, 5000, 10000)


def retain_predictions(path: Path, table: pd.DataFrame) -> None:
    """Retain observations on recovery; never overwrite a disagreement."""
    import numpy as np
    if not path.exists():
        table.to_csv(path, index=False)
        return
    old = pd.read_csv(path, dtype=str, keep_default_na=False)
    new = table.reset_index(drop=True).astype(str)
    vectors = ['logits', 'probabilities']
    pd.testing.assert_frame_equal(old.drop(columns=vectors), new.drop(columns=vectors))
    for column in vectors:
        for before, after in zip(old[column], new[column]):
            a, b = np.asarray(json.loads(before)), np.asarray(json.loads(after))
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-7):
                raise ValueError(f'restored predictions disagree with retained {path.name}:{column}')


class SourceFit95(pl.Callback):
    """Observe the same training trajectory at predeclared updates only."""
    def __init__(self, factory, output: Path):
        self.factory = factory
        self.output = Path(output)
        self.seen = set()

    def _observe(self, trainer, task):
        step = int(trainer.global_step)
        if step not in FIT95_STEPS or step in self.seen:
            return
        directory = self.output / 'fit95'
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / f'step_{step:05d}.ckpt'
        if checkpoint.exists():
            raise FileExistsError(f'will not replace an observed checkpoint: {checkpoint}')
        trainer.save_checkpoint(str(checkpoint))
        classes = json.loads((self.output / 'local_class_map.json').read_text())
        was_training = task.training
        devices = [task.device.index] if task.device.type == 'cuda' else []
        rows = []
        try:
            # An observational forward must not consume training randomness or
            # change dropout/batch-normalization state.
            with torch.random.fork_rng(devices=devices):
                task.eval()
                for role in ('source_train', 'source_val'):
                    path = directory / f'step_{step:05d}_{role}.csv'
                    predictions = task.source_predictions(self.factory, checkpoint, role=role)
                    retain_predictions(path, predictions)
                    _, _, metrics = evaluate_source_predictions(path,
                        expected_windows_file=self.output / 'expected_source_windows.csv',
                        local_class_map=classes, checkpoint=str(checkpoint),
                        intervention='full', role=role)
                    metrics['global_step'] = step
                    metrics['role'] = role
                    rows.append(metrics)
        finally:
            task.train(was_training)
        curve = self.output / 'source_fit_curve.csv'
        table = pd.concat(rows, ignore_index=True)
        table.to_csv(curve, mode='a', header=not curve.exists(), index=False)
        self.seen.add(step)

    def on_train_start(self, trainer, pl_module):
        self._observe(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._observe(trainer, pl_module)
