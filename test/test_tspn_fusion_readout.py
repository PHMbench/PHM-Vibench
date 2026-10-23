"""Readout behavior on an explicit small interface fixture.

Actual TSPN/factory/H5 restoration is tested by the P01 integration suite.
This fixture only isolates head equations, gradients and frozen buffers.
"""
import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.model_factory.X_model.TSPN_fusion import TSPNFusion


class Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.args=SimpleNamespace(num_classes=2,in_channels=1)
        self.channel_for_classifier=4
        self.signal_processing_layers=nn.ModuleList([nn.Flatten(1)])
        self.feature_extractor_layers=nn.BatchNorm1d(4)
        self.clf=nn.Linear(4,2)


def model(kind='linear',width=16,cap=5.):
    return TSPNFusion(Reference(),in_channels=1,num_classes=2,branches=[],
                      head_type=kind,head_hidden_dim=width,head_frobenius_cap=cap)


def test_default_linear_reconstruction_and_state_keys():
    torch.manual_seed(4);a=model().eval();b=model('linear').eval()
    b.load_state_dict(a.state_dict(),strict=True)
    assert not any('candidate_hidden' in k for k in a.state_dict())
    x=torch.randn(5,4,1)
    torch.testing.assert_close(a.forward_details(x)['candidate_probs'],b.forward_details(x)['candidate_probs'],rtol=0,atol=0)


def test_mlp_updates_both_layers_without_changing_reference():
    torch.manual_seed(7);a=model('mlp').train();x=torch.randn(8,4,1);y=torch.arange(8)%2
    frozen={k:v.clone() for k,v in a.reference.state_dict().items()}
    before={k:v.clone() for k,v in a.state_dict().items() if k.startswith('candidate_')}
    opt=torch.optim.SGD([p for p in a.parameters() if p.requires_grad],lr=.1)
    out=a.forward_details(x);opt.zero_grad()
    nn.functional.cross_entropy(out['candidate_logits'],y).backward()
    for layer in (a.candidate_hidden,a.candidate_head):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum()>0
    opt.step()
    assert all(torch.equal(v,a.reference.state_dict()[k]) for k,v in frozen.items())
    assert not a.reference.training and all(p.grad is None for p in a.reference.parameters())
    assert any(not torch.equal(v,a.state_dict()[k]) for k,v in before.items())
    a.eval();a.set_alpha(0)
    torch.testing.assert_close(a.predict_proba(x),out['raw_probs'],rtol=0,atol=0)


def test_mlp_effective_norm_product_and_probability_sensitivity():
    torch.manual_seed(8);a=model('mlp',cap=5.).double()
    with torch.no_grad():
        a.candidate_hidden.weight.mul_(100);a.candidate_head.weight.mul_(100)
    w=a.candidate_hidden.weight;cap=a.head_frobenius_cap.sqrt()
    w1=w/(torch.linalg.vector_norm(w)/cap).clamp_min(1.)
    w2=a.effective_head_weight()
    assert float(torch.linalg.matrix_norm(w1,2)*torch.linalg.matrix_norm(w2,2))<=5.+1e-10
    z=torch.randn(8,4,dtype=torch.double);zp=z+.02*torch.randn_like(z)
    def predict(t):
        return nn.functional.linear(nn.functional.relu(nn.functional.linear(t,w1,a.candidate_hidden.bias)),w2,a.candidate_head.bias).softmax(-1)
    assert torch.all((predict(zp)-predict(z)).norm(dim=-1)<=2.5*(zp-z).norm(dim=-1)+1e-10)


def test_mlp_strict_checkpoint_roundtrip():
    torch.manual_seed(9);a=model('mlp',width=7).eval();a.set_alpha(.37)
    state=copy.deepcopy(a.state_dict());b=model('mlp',width=7).eval();b.load_state_dict(state,strict=True)
    x=torch.randn(4,4,1)
    torch.testing.assert_close(a.predict_proba(x),b.predict_proba(x),rtol=0,atol=0)
    with pytest.raises(RuntimeError):model('linear').load_state_dict(state,strict=True)
    with pytest.raises(RuntimeError):model('mlp',width=8).load_state_dict(state,strict=True)


@pytest.mark.parametrize('width',[0,-1,2.5,True])
def test_mlp_width_must_be_explicit_positive_integer(width):
    with pytest.raises(ValueError,match='hidden width'):model('mlp',width=width)


def test_unknown_head_is_not_a_fallback():
    with pytest.raises(ValueError,match='head_type'):model('transformer')
