"""Generated tensor tests of the declared shared-update measure."""
import copy
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch

from src.data_factory.tii_sampling import SourceRounds, fit_source_rms
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


def make_task():
    m = make_model()
    return build_task(
        args_task=NS(type='DG', name='tii_joint', source_system_ids=[1, 2], loss='CE',
                     metrics=['acc'], optimizer='adamw', lr=.001, weight_decay=.0001,
                     lambda_common=0, lambda_private=0, scheduler=None),
        network=m, args_data=NS(source_batch_size=32), args_model=m.args_m,
        args_trainer=NS(device='cpu'), args_environment=NS(seed=0), metadata=m.metadata)


class JointTests(unittest.TestCase):
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
