"""The bounded P07 method preserves executable semantics and counts all extraction work."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from src.model_factory.X_model.P07OperatorPath import Model, OperatorNet


class BoundedP07Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        self.x = torch.linspace(-2, 2, 64, dtype=torch.double).reshape(2, 32, 1)
        self.model = OperatorNet(1, 3).double().eval()

    def test_factory_fixed_input_trace_and_checkpoint_parity(self):
        factory = Model(SimpleNamespace(in_channels=1, num_classes=3)).double().eval()
        factory.load_state_dict(self.model.state_dict(), strict=True)
        logits, trace = self.model(self.x)
        torch.testing.assert_close(factory(self.x), logits, rtol=0, atol=0)
        direct, direct_trace = OperatorNet.forward(factory, self.x)
        for expected, actual in zip(trace.states + trace.weights, direct_trace.states + direct_trace.weights):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        modified, _ = self.model(self.x, frozen=trace.weights, intervention=(1, 2, 'ZERO'))
        other, _ = OperatorNet.forward(factory, self.x, frozen=direct_trace.weights, intervention=(1, 2, 'ZERO'))
        torch.testing.assert_close(modified, other, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            Model(SimpleNamespace(in_channels=1, num_classes=3, residual_weight=.1))

    def test_regularizer_has_gradient_and_does_not_change_model_capacity(self):
        _, trace = self.model(self.x)
        loss = self.model.routing_concentration_loss(trace)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.gates.parameters()))
        self.assertEqual(set(self.model.state_dict()), set(OperatorNet(1, 3).state_dict()))

    def test_extract_counts_reference_argmax_ranking_candidate_and_replay(self):
        # Make acceptance non-vacuous without claiming task accuracy.
        with torch.no_grad():
            self.model.head.weight.mul_(.01)
            self.model.head.bias.copy_(torch.tensor([10., 0., -1.], dtype=torch.double))
        for strategy in ('cost', 'perturbation'):
            out = self.model.extract(self.x[:1], budget=238, strategy=strategy)
            self.assertTrue(out['accepted'])
            self.assertEqual(out['checked_argmax_accepted'], out['argmax_gap'] <= .49 * out['margin'])
            self.assertEqual(out['argmax_total_queries'], 2)
            self.assertLessEqual(out['total_queries'], 238)
            self.assertEqual(out['total_queries'], sum(out[k] for k in (
                'reference_queries', 'argmax_queries', 'ranking_queries', 'candidate_queries', 'replay_queries')))
            self.assertEqual(out['ranking_queries'], 18 if strategy == 'perturbation' else 0)
            self.assertLessEqual(out['exact_gap'], .49 * out['margin'])
            self.assertGreaterEqual(out['extraction_seconds'], out['replay_seconds'])
        minimal = self.model.extract(self.x[:1], budget=2)
        self.assertTrue(minimal['accepted'])
        self.assertEqual(minimal['accepted_source'], 'cached_argmax')
        self.assertFalse(minimal['minimum_declared_cost'])
        self.assertEqual(minimal['total_queries'], 2)
        with self.assertRaises(ValueError):
            self.model.extract(self.x[:1], budget=19, strategy='perturbation')

    def test_minimum_extraction_budgets_match_executed_calls(self):
        for strategy, budget in (('cost', 2), ('perturbation', 2 + 6 * self.model.stages)):
            with self.subTest(strategy=strategy):
                with patch.object(OperatorNet, 'forward', autospec=True,
                                  side_effect=OperatorNet.forward) as forward_call, \
                     patch.object(OperatorNet, 'discrete', autospec=True,
                                  side_effect=OperatorNet.discrete) as discrete_call:
                    out = self.model.extract(self.x[:1], budget=budget, strategy=strategy)
                self.assertEqual(forward_call.call_count, budget - 1)
                self.assertEqual(discrete_call.call_count, 1)
                executed_calls = forward_call.call_count + discrete_call.call_count
                self.assertEqual(out['total_queries'], executed_calls)
                self.assertLessEqual(executed_calls, budget)

    def test_budget_exhaustion_and_zero_margin_abstain(self):
        with torch.no_grad():
            self.model.head.weight.zero_()
            self.model.head.bias.zero_()
        out = self.model.extract(self.x[:1], budget=2)
        self.assertFalse(out['accepted'])
        self.assertFalse(out['checked_argmax_accepted'])
        self.assertEqual(out['total_queries'], 2)
        self.assertFalse(out['search_complete'])


if __name__ == '__main__':
    unittest.main()
