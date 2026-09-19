"""Source checkpoint selection tests without model training or serialization."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from src.trainer_factory.Default_trainer import ToleranceModelCheckpoint, call_backs
from test.test_tii_joint import inventory, make_task


@pytest.fixture
def checkpoint_and_trainer(tmp_path):
    checkpoint = ToleranceModelCheckpoint(
        monitor="val_group_nll",
        mode="min",
        min_delta=1e-8,
        save_top_k=1,
        dirpath=tmp_path,
        filename="model-{epoch:02d}-{step}",
    )
    trainer = SimpleNamespace(
        global_step=0,
        ckpt_path=None,
        is_global_zero=False,
        strategy=SimpleNamespace(broadcast=lambda value: value, remove_checkpoint=Mock()),
        save_checkpoint=Mock(),
    )
    return checkpoint, trainer


def select(checkpoint, trainer, *, round_index, nll):
    trainer.global_step = round_index
    checkpoint._save_topk_checkpoint(
        trainer,
        {
            "epoch": torch.tensor(round_index),
            "step": torch.tensor(round_index),
            # Float64 distinguishes improvements smaller than the frozen tolerance.
            "val_group_nll": torch.tensor(nll, dtype=torch.float64),
        },
    )


@pytest.mark.parametrize("improvement", [0.0, 5e-9, 1e-8])
def test_tie_within_tolerance_keeps_earlier_checkpoint(checkpoint_and_trainer, improvement):
    checkpoint, trainer = checkpoint_and_trainer
    select(checkpoint, trainer, round_index=1, nll=0.5)
    earlier_path = checkpoint.best_model_path

    select(checkpoint, trainer, round_index=2, nll=0.5 - improvement)

    assert checkpoint.best_model_path == earlier_path
    assert checkpoint.best_model_score.item() == 0.5
    assert list(checkpoint.best_k_models) == [earlier_path]
    trainer.save_checkpoint.assert_called_once_with(earlier_path, False)
    trainer.strategy.remove_checkpoint.assert_not_called()


def test_improvement_beyond_tolerance_updates_checkpoint(checkpoint_and_trainer):
    checkpoint, trainer = checkpoint_and_trainer
    select(checkpoint, trainer, round_index=1, nll=0.5)
    earlier_path = checkpoint.best_model_path
    improved_nll = 0.5 - 2e-8

    select(checkpoint, trainer, round_index=2, nll=improved_nll)

    assert checkpoint.best_model_path != earlier_path
    assert checkpoint.best_model_score.item() == improved_nll
    assert list(checkpoint.best_k_models) == [checkpoint.best_model_path]
    assert trainer.save_checkpoint.call_count == 2
    trainer.save_checkpoint.assert_called_with(checkpoint.best_model_path, False)
    trainer.strategy.remove_checkpoint.assert_called_once_with(earlier_path)


def test_frozen_source_validation_callback_configuration(tmp_path):
    callbacks = call_backs(
        SimpleNamespace(
            monitor="val_group_nll",
            monitor_mode="min",
            checkpoint_min_delta=1e-8,
            save_top_k=1,
            early_stopping=False,
            pruning=0.0,
        ),
        str(tmp_path),
    )

    assert len(callbacks) == 1
    checkpoint = callbacks[0]
    assert isinstance(checkpoint, ToleranceModelCheckpoint)
    assert checkpoint.monitor == "val_group_nll"
    assert checkpoint.mode == "min"
    assert checkpoint.min_delta == 1e-8
    assert checkpoint.save_top_k == 1
    assert checkpoint.dirpath == str(tmp_path)


def test_task_computes_validation_nll_before_float32_rounding():
    task = make_task().eval()
    task.on_validation_epoch_start()
    logits = torch.tensor([[.1, .9], [1., .2], [.7, .5], [.2, 1.]], dtype=torch.float32)
    for source, batch in inventory().items():
        batch['source'] = source
        batch['role'] = ['source_val'] * 4
        with patch.object(task, 'forward', return_value=logits):
            task.validation_step(batch, 0)
    losses = torch.nn.functional.cross_entropy(logits.double(), batch['y'], reduction='none')
    expected = ((losses[0].item() + losses[2].item()) / 2
                + (losses[1].item() + losses[3].item()) / 2) / 2
    assert task.validation_risk()[0] == expected


def test_task_logged_nll_preserves_improvement_and_tie(checkpoint_and_trainer):
    task = make_task().eval()
    checkpoint, trainer = checkpoint_and_trainer
    paths = []
    for round_index, nll in enumerate((1.00000008, 1.00000006, 1.000000059), start=1):
        task.on_validation_epoch_start()
        task.validation_groups = {(source, 0): [nll, 1., 1] for source in task.sources}
        with patch.object(task, 'log') as log:
            task.on_validation_epoch_end()
        logged = {call.args[0]: call.args[1] for call in log.call_args_list}
        metric = logged['val_group_nll']
        assert torch.is_tensor(metric)
        assert metric.dtype == torch.float64
        assert metric.item() == nll
        trainer.global_step = round_index
        checkpoint._save_topk_checkpoint(trainer, {
            'epoch': torch.tensor(round_index), 'step': torch.tensor(round_index),
            'val_group_nll': metric,
        })
        paths.append(checkpoint.best_model_path)
    assert paths[0] != paths[1]
    assert paths[1] == paths[2]
    assert checkpoint.best_model_score.item() == 1.00000006
    assert trainer.save_checkpoint.call_count == 2
