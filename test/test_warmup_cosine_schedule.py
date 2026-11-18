"""Warmup-cosine schedule, gradient-clip wiring, and the tii_joint stabilized recipe contract."""
import inspect
import math
from types import SimpleNamespace

import torch
import unittest

import src.trainer_factory.Default_trainer as trainer_module
from src.task_factory.Default_task import Default_task
from src.task_factory.task.DG.tii_joint import task as tii_joint_task


class _Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class _Host:
    """Duck-typed host: configure_optimizers only reads args_task/trainer/parameters."""

    def __init__(self, args_task):
        self.args_task = args_task
        self.trainer = None
        self._net = torch.nn.Linear(2, 2)

    def parameters(self, recurse=True):
        return self._net.parameters(recurse)


def _build(scheduler, base_lr=0.01):
    host = _Host(SimpleNamespace(optimizer='adamw', lr=base_lr, weight_decay=0.0,
                                 scheduler=scheduler))
    return Default_task.configure_optimizers(host)


class WarmupCosineTests(unittest.TestCase):
    def test_lr_rises_then_cosine_decays_at_step_interval(self):
        total, warm = 1000, 50
        result = _build({'name': 'warmup_cosine',
                         'options': {'total_steps': total, 'warmup_steps': warm}})
        optimizers, scheduler_specs = result
        spec = scheduler_specs[0]
        self.assertEqual(spec['interval'], 'step')
        scheduler, optimizer = spec['scheduler'], optimizers[0]
        peak = optimizer.param_groups[0]['initial_lr']
        lrs = []
        for _ in range(total):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(lrs[0], peak * 1e-3, delta=peak * 2e-3)
        self.assertAlmostEqual(lrs[warm], peak, delta=peak * 5e-3)
        self.assertLess(lrs[-1], peak * 0.05)
        midpoint = total // 2 + warm
        expected = peak * (1 + math.cos(math.pi * (midpoint - warm) / (total - warm))) / 2
        self.assertAlmostEqual(lrs[midpoint], expected, delta=peak * 0.05)

    def test_default_five_percent_warmup(self):
        total, warm = 4000, 200
        result = _build({'name': 'warmup_cosine', 'options': {'total_steps': total}})
        optimizers, scheduler_specs = result
        scheduler, optimizer = scheduler_specs[0]['scheduler'], optimizers[0]
        peak = optimizer.param_groups[0]['initial_lr']
        lrs = []
        for _ in range(total):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        self.assertLess(lrs[warm - 1], peak)
        self.assertAlmostEqual(lrs[warm], peak, delta=peak * 5e-3)

    def test_invalid_options_fail_closed(self):
        for options in ({'total_steps': 1}, {'total_steps': 100, 'warmup_fraction': 0},
                        {'total_steps': 100, 'warmup_fraction': 1.5},
                        {'total_steps': 100, 'warmup_steps': 100},
                        {'total_steps': 100, 'warmup_steps': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                _build({'name': 'warmup_cosine', 'options': options})


class TrainerClipWiringTests(unittest.TestCase):
    def test_gradient_clip_val_read_from_args(self):
        source = inspect.getsource(trainer_module)
        self.assertIn('gradient_clip_val=getattr(args_t, "gradient_clip_val", None)', source)


class TiiJointStabilizedRecipeTests(unittest.TestCase):
    def _args(self, scheduler, lr, weight_decay):
        return dict(
            network=None,
            args_data=SimpleNamespace(rounds=1000, source_batch_size=4),
            args_model=None,
            args_task=SimpleNamespace(loss='CE', optimizer='adamw', lambda_common=0,
                                      lambda_private=0, scheduler=scheduler, metrics=['acc'],
                                      lr=lr, weight_decay=weight_decay,
                                      source_system_ids=[1, 2], sampling_seed=0),
            args_trainer=SimpleNamespace(num_epochs=1, early_stopping=False, devices=1,
                                         monitor='val_group_nll', monitor_mode='min',
                                         checkpoint_min_delta=1e-8, num_sanity_val_steps=0,
                                         val_check_interval=100),
            args_environment=None, metadata=None,
        )

    def test_legacy_wrong_lr_rejected(self):
        with self.assertRaisesRegex(ValueError, 'lr=.001'):
            tii_joint_task(**self._args(None, 0.01, 0.0001))

    def test_foreign_scheduler_rejected(self):
        with self.assertRaisesRegex(ValueError, 'warmup_cosine'):
            tii_joint_task(**self._args({'name': 'step', 'options': {}}, 0.001, 0.0001))

    def test_stabilized_recipe_requires_matching_total_steps(self):
        scheduler = {'name': 'warmup_cosine', 'options': {'total_steps': 999}}
        with self.assertRaisesRegex(ValueError, 'total_steps'):
            tii_joint_task(**self._args(scheduler, 0.01, 0.0))

    def test_stabilized_recipe_accepts_declared_lr(self):
        scheduler = {'name': 'warmup_cosine',
                     'options': {'total_steps': 1000, 'warmup_fraction': 0.05}}
        # Recipe checks pass; construction proceeds until the absent network wiring fails.
        with self.assertRaises(AttributeError):
            tii_joint_task(**self._args(scheduler, 0.01, 0.0))

    def test_stabilized_recipe_rejects_bad_lr_type(self):
        scheduler = {'name': 'warmup_cosine', 'options': {'total_steps': 1000}}
        with self.assertRaisesRegex(ValueError, 'explicit positive lr'):
            tii_joint_task(**self._args(scheduler, 1, 0.0))


if __name__ == '__main__':
    unittest.main()
