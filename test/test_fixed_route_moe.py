from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from src.model_factory import build_model


def model(arm='aligned'):
    torch.manual_seed(20)
    return build_model(SimpleNamespace(type='MoE', name='M_05_FixedRouteMoE',
                                      input_dim=4, num_experts=3, num_classes=2,
                                      width=5, arm=arm))


@pytest.mark.parametrize('arm', ['aligned', 'generic', 'shuffled', 'learned_physics', 'uniform', 'no_balance'])
def test_factory_model_and_exact_single_slot_replacement(arm):
    network = model(arm)
    torch.manual_seed(2)
    views, raw, cues = torch.randn(2, 3, 4), torch.randn(2, 4), torch.zeros(2, 3)
    logits, gates, clean = network(views, raw, cues)
    changed_views, changed_raw = views + .2, raw - .1
    base, replaced, response = network.fixed_route_replacements(changed_views, changed_raw, gates, clean)
    changed = network.encode(changed_views, changed_raw)
    explicit = []
    for target in range(3):
        intervention = changed.clone()
        intervention[:, target] = clean[:, target]
        explicit.append(network.head(torch.sum(gates[..., None] * intervention, dim=1)).softmax(-1))
    torch.testing.assert_close(replaced, torch.stack(explicit, dim=1))
    torch.testing.assert_close(base, network.head((gates[..., None] * changed).sum(1)).softmax(-1))
    assert response.shape == (2, 3)
    logits.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())


def test_invalid_physical_view_axes_fail_instead_of_reshaping():
    with pytest.raises(ValueError, match='raw'):
        model()(torch.zeros(2, 4, 3), torch.zeros(2, 4), torch.zeros(2, 3))


def test_invalid_route_cannot_enter_intervention():
    network = model()
    with pytest.raises(ValueError, match='sum to one'):
        network.fixed_route_replacements(torch.zeros(2, 3, 4), torch.zeros(2, 4),
                                         torch.ones(2, 3), torch.zeros(2, 3, 5))


def test_explicit_width_factory_does_not_import_training_stack():
    code = '''
import sys
from types import SimpleNamespace
from src.model_factory import build_model
build_model(SimpleNamespace(type="MoE", name="M_05_FixedRouteMoE", input_dim=4,
                            num_experts=3, num_classes=2, width=5, arm="aligned"))
assert "pytorch_lightning" not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True)
