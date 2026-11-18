"""Generated tensor tests of the declared shared-update measure."""
import copy
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data_factory.tii_sampling import SourceRounds, fit_source_rms, assign_physical_groups
from src.task_factory import build_task
from test.test_tii_native_model import make_model


def inventory():
    torch.manual_seed(21)
    return {d: dict(x=torch.randn(4, 3, 4), incremental=torch.randn(4, 3, 4),
                    availability=torch.tensor([0, 0, 1, 1]), y=torch.tensor([0, 1, 0, 1]),
                    file_id=[offset, offset+1, offset, offset+1],
                    recording_id=[offset, offset+1, offset, offset+1],
                    group=[0, 1, 0, 1], role=['source_train']*4)
            for d, offset in ((1, 0), (2, 2))}


def make_task(scheduler=None, fixed_rounds=False):
    m = make_model()
    data = NS(source_batch_size=32, **({'rounds': 6} if fixed_rounds else {}))
    trainer = NS(device='cpu', num_epochs=1, early_stopping=False, devices=1,
                 monitor='val_group_nll', monitor_mode='min', checkpoint_min_delta=1e-8,
                 num_sanity_val_steps=0, val_check_interval=2)
    return build_task(
        args_task=NS(type='DG', name='tii_joint', source_system_ids=[1, 2], loss='CE',
                     metrics=['acc'], optimizer='adamw', lr=.001, weight_decay=.0001,
                     lambda_common=0, lambda_private=0, scheduler=scheduler),
        network=m, args_data=data, args_model=m.args_m,
        args_trainer=trainer, args_environment=NS(seed=0), metadata=m.metadata)


class JointTests(unittest.TestCase):
    def test_lightning_optimizer_steps_follow_declared_schedule_and_legacy_lr(self):
        class Trace(pl.Callback):
            def __init__(self):
                self.lrs = []

            def on_before_optimizer_step(self, trainer, module, optimizer):
                self.lrs.append(optimizer.param_groups[0]['lr'])

        for scheduled in (False, True):
            with self.subTest(scheduled=scheduled):
                scheduler = ({'name': 'warmup_cosine',
                              'options': {'total_steps': 6, 'warmup_steps': 2}}
                             if scheduled else None)
                task = make_task(scheduler=scheduler, fixed_rounds=True)
                trace = Trace()
                data = SourceRounds(inventory(), rounds=6, batch_size=32, seed=0)
                loader = DataLoader(data, batch_size=None, num_workers=0)
                trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1,
                                     logger=False, enable_checkpointing=False,
                                     enable_model_summary=False, enable_progress_bar=False,
                                     limit_val_batches=0, callbacks=[trace])
                before = next(task.network.parameters()).detach().clone()
                trainer.fit(task, train_dataloaders=loader)
                self.assertEqual(trainer.global_step, 6)
                self.assertFalse(torch.equal(before, next(task.network.parameters())))
                self.assertEqual(len(trace.lrs), 6)
                if scheduled:
                    self.assertAlmostEqual(trace.lrs[0], 0.001 * 0.001)
                    self.assertLess(trace.lrs[0], trace.lrs[1])
                    self.assertAlmostEqual(trace.lrs[2], 0.001)
                    self.assertGreater(trace.lrs[2], trace.lrs[3])
                    self.assertGreater(trace.lrs[3], trace.lrs[5])
                    self.assertEqual(trainer.lr_scheduler_configs[0].interval, 'step')
                else:
                    self.assertEqual(trace.lrs, [0.001] * 6)
                    self.assertEqual(trainer.lr_scheduler_configs, [])

    def test_invalid_warmup_options_fail_at_optimizer_setup(self):
        task = make_task()
        task.args_task.scheduler = {'name': 'warmup_cosine',
                                    'options': {'total_steps': 6, 'warmup_steps': 6}}
        with self.assertRaisesRegex(ValueError, 'warmup_steps'):
            task.configure_optimizers()

    def test_physical_group_assignment_is_order_invariant_and_disjoint(self):
        groups = [g for g in range(12) for _ in range(2)]
        allocations = [('source_train', 7), ('source_val', 3), ('target', 2)]
        roles = assign_physical_groups(groups, allocations, seed=7)
        reversed_roles = assign_physical_groups(groups[::-1], allocations, seed=7)
        self.assertEqual(roles, reversed_roles[::-1])
        split = [{g for g, r in zip(groups, roles) if r == role} for role, _ in allocations]
        self.assertEqual([len(s) for s in split], [7, 3, 2])
        self.assertTrue(all(a.isdisjoint(b) for i, a in enumerate(split) for b in split[i+1:]))
        with self.assertRaisesRegex(ValueError, 'exact coverage'):
            assign_physical_groups(groups, [('source_train', 7), ('source_val', 3)], seed=7)

    def test_schedule_is_reproducible_and_samples_groups_before_windows(self):
        sources = inventory()
        # A group with 100 windows must not outweigh a group with one window.
        for data in sources.values():
            ids = [0]*100+[1]
            for key, val in list(data.items()):
                data[key] = val[ids] if torch.is_tensor(val) else [val[i] for i in ids]
        rounds = SourceRounds(sources, rounds=200, batch_size=32, seed=0)
        counts = {1: 0, 2: 0}
        for i in range(len(rounds)):
            batch = rounds[i]
            self.assertEqual(set(batch), {1, 2})
            for source in counts:
                self.assertEqual(len(batch[source]['y']), 32)
                counts[source] += batch[source]['group'].count(1)
        for count in counts.values(): self.assertTrue(2900 < count < 3500, count)
        torch.testing.assert_close(rounds[11][1]['x'], rounds[11][1]['x'], rtol=0, atol=0)
        self.assertEqual(rounds[11][1]['window_index'], SourceRounds(sources, rounds=200, batch_size=32, seed=0)[11][1]['window_index'])

    def test_rms_group_weight_and_query_rejection(self):
        sources = inventory()
        for d, data in sources.items():
            data['x'] = torch.full((4, 3, 4), float(d))
            data['incremental'].zero_()
        self.assertAlmostEqual(fit_source_rms(sources), np.sqrt(2.5))
        repeated = copy.deepcopy(sources)
        ids = [0, 2, 0, 2, 1, 3]
        for key, val in list(repeated[1].items()):
            repeated[1][key] = val[ids] if torch.is_tensor(val) else [val[i] for i in ids]
        self.assertEqual(fit_source_rms(sources), fit_source_rms(repeated))
        sources[1]['role'][0] = 'query'
        with self.assertRaisesRegex(ValueError, 'source_train'): fit_source_rms(sources)

    def test_both_sources_backpropagate_to_same_encoder_and_isolate_heads(self):
        task = make_task()
        batch = SourceRounds(inventory(), rounds=1, batch_size=32, seed=0)[0]
        optimizer = task.configure_optimizers()
        # Prime optimizer state for both heads before testing an unselected head.
        optimizer.zero_grad(set_to_none=True)
        task.joint_loss(batch).backward()
        optimizer.step()
        shared = task.network.embedding.common_branch[0].weight
        for selected in (1, 2):
            other = task.network.task_head.mutiple_fc[str(3-selected)]
            params = [p.detach().clone() for p in other.parameters()]
            states = [copy.deepcopy(optimizer.state[p]) for p in other.parameters()]
            task.optimizer_zero_grad(0, 0, optimizer)
            task.source_loss(selected, batch[selected]).backward()
            self.assertIsNotNone(shared.grad)
            self.assertGreater(shared.grad.norm().item(), 0)
            self.assertTrue(all(p.grad is None for p in other.parameters()))
            optimizer.step()
            for p, old, state in zip(other.parameters(), params, states):
                torch.testing.assert_close(p, old, rtol=0, atol=0)
                for key in state:
                    torch.testing.assert_close(optimizer.state[p][key], state[key], rtol=0, atol=0)
        expected = torch.stack([task.source_loss(d, batch[d]) for d in (1, 2)]).mean()
        torch.testing.assert_close(task.joint_loss(batch), expected)
        with self.assertRaisesRegex(ValueError, 'every declared source'):
            task.joint_loss({1: batch[1]})

    def test_source_validation_estimator_and_role_gate(self):
        task = make_task().eval()
        data = inventory()
        task.on_validation_epoch_start()
        expected = []
        for d, batch in data.items():
            batch['source'] = d
            batch['role'] = ['source_val']*4
            logits = task(batch)
            expected.append(torch.nn.functional.cross_entropy(logits, batch['y']).item())
            task.validation_step(batch, 0)
        self.assertAlmostEqual(task.validation_risk()[0], sum(expected)/2, places=6)
        data[1]['role'][0] = 'query'
        with self.assertRaisesRegex(ValueError, 'source_val'):
            task.validation_step(data[1], 0)
        data[1]['role'] = ['source_val']
        with self.assertRaisesRegex(ValueError, 'per source window'):
            task.validation_step(data[1], 0)


if __name__ == '__main__': unittest.main()
