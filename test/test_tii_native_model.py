"""Tensor fixtures only: native model semantics, not industrial accuracy."""
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from phmfactory.config import analyze_config
from src.data_factory.data_utils import MetadataAccessor
from src.model_factory import build_model


def make_model(arm='support', embedding='SupportConditionedTokenizer'):
    resolved = analyze_config('smoke', override_values=[
        f'model.embedding={embedding}', f'model.token_organization={arm}',
        'model.patch_size_L=4', 'model.patch_size_C=1', 'model.num_patches=3',
        'model.output_dim=8', 'model.d_model=6', 'model.source_rms=2.5',
    ])
    metadata = MetadataAccessor(pd.DataFrame([
        dict(Id=i, Dataset_id=d, Label=c, Sample_rate=12000, Name=f'fixture_{d}')
        for i, (d, c) in enumerate(((1, 0), (1, 1), (2, 0), (2, 1)))
    ]))
    return build_model(SimpleNamespace(**resolved.runtime_config()['model']), metadata)


class NativeModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.c = torch.randn(2, 3, 4)
        self.p = torch.randn_like(self.c)

    def paired(self):
        torch.manual_seed(11)
        s = make_model()
        torch.manual_seed(11)
        u = make_model('ordinary')
        return s, u

    def test_parameter_shapes_both_branches_and_common_only_update_null(self):
        s, u = self.paired()
        self.assertEqual({k: v.shape for k, v in s.state_dict().items()},
                         {k: v.shape for k, v in u.state_dict().items()})
        optimizers = [torch.optim.AdamW(m.parameters(), lr=.001) for m in (s, u)]
        outputs = []
        for m, optimizer in zip((s, u), optimizers):
            calls = []
            handles = [branch.register_forward_hook(lambda *args: calls.append(1))
                       for branch in (m.embedding.common_branch, m.embedding.increment_branch)]
            tokens = m.embedding(self.c, self.p, torch.zeros(2))
            self.assertEqual(tokens.shape, (2, 3, 8))
            self.assertEqual(len(calls), 2)
            for h in handles: h.remove()
            logits = m(self.c, file_id=[0, 1], task_id='classification',
                       incremental=self.p, availability=torch.zeros(2))
            outputs.append(logits.detach())
            logits.square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for a, b in zip(s.parameters(), u.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_increment_intervention_changes_branch_inputs(self):
        s, u = self.paired()
        observed = []
        for m in (s, u):
            inputs = []
            handles = [b.register_forward_pre_hook(lambda _, args: inputs.append(args[0].clone()))
                       for b in (m.embedding.common_branch, m.embedding.increment_branch)]
            m.embedding(self.c, self.p, torch.ones(2))
            for h in handles: h.remove()
            observed.append(inputs)
        torch.testing.assert_close(observed[0][0], self.c/2.5)
        torch.testing.assert_close(observed[0][1], self.p/2.5)
        torch.testing.assert_close(observed[1][0], (self.c+self.p)/2.5)
        torch.testing.assert_close(observed[1][0], observed[1][1])
        self.assertFalse(torch.equal(observed[0][0], observed[1][0]))

    def test_missing_input_isolation_and_zero_gradient_through_logits(self):
        for m in self.paired():
            p = self.p.clone().requires_grad_()
            gate = torch.tensor([0, 1])
            a = m(self.c, file_id=[0, 1], task_id='classification', incremental=p, availability=gate)
            changed = p.detach().clone()
            changed[0] = float('nan')
            b = m(self.c, file_id=[0, 1], task_id='classification', incremental=changed, availability=gate)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            a.sum().backward()
            self.assertEqual(torch.count_nonzero(p.grad[0]).item(), 0)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(torch.count_nonzero(p.grad[1]).item(), 0)

    def test_pure_target_encoding_never_accesses_metadata_or_head(self):
        m = make_model().eval()
        keys = tuple(m.task_head.mutiple_fc)
        with patch.object(m, '_head', side_effect=AssertionError('source head accessed')):
            with patch('src.model_factory.ISFM.M_01_ISFM.resolve_batch_metadata',
                       side_effect=AssertionError('source metadata accessed')):
                features = m.encode(self.c, incremental=self.p, availability=torch.ones(2))
        self.assertEqual(features.shape, (2, 3, 8))
        self.assertEqual(keys, tuple(m.task_head.mutiple_fc))
        self.assertEqual(set(keys), {'1', '2'})

    def test_native_hse_pure_encoding_preserves_rates_and_explicit_starts(self):
        m = make_model(embedding='E_01_HSE').eval()
        x = torch.randn(2, 16, 1)
        rates = torch.tensor([12000., 24000.])
        starts = torch.tensor([[0, 4, 8], [1, 5, 9]])
        channels = torch.zeros_like(starts)
        expected = m.backbone(m.embedding(x, rates, start_indices_L=starts, start_indices_C=channels))
        with patch.object(m, '_head', side_effect=AssertionError('head accessed')):
            actual = m.encode(x, sample_rates=rates, start_indices_L=starts, start_indices_C=channels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpoint_restores_tokens_features_logits_and_rms(self):
        for arm in ('support', 'ordinary'):
            m = make_model(arm).eval()
            with torch.no_grad(): m.embedding.source_rms.fill_(3.25)
            gate = torch.tensor([0, 1])
            def predictions(model):
                return (model.embedding(self.c, self.p, gate),
                        model.encode(self.c, incremental=self.p, availability=gate),
                        model(self.c, file_id=[0, 1], task_id='classification', incremental=self.p, availability=gate))
            before = predictions(m)
            stream = io.BytesIO()
            torch.save(m.state_dict(), stream)
            stream.seek(0)
            restored = make_model(arm).eval()
            restored.load_state_dict(torch.load(stream, weights_only=True))
            for a, b in zip(before, predictions(restored)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(restored.embedding.source_rms.item(), 3.25)


if __name__ == '__main__': unittest.main()
