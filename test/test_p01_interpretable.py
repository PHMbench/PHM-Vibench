"""Operator and statistical-unit tests; constructed inputs are not PHM evidence."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.model_factory.X_model.TSPN_fusion import (
    EnvelopeBranch, OperatorResidualHead, TSPNFusion, analytic_envelope)
from experiments.p01.diagnostic_gain import report


def bands():
    return dict(centers=[.2], widths=[.03], center_bounds=[[.18,.22]], width_bounds=[[.02,.04]])


def modulation():
    return dict(centers=[.08], widths=[.01], center_bounds=[[.07,.09]], width_bounds=[[.008,.014]])


def envelope(diagnostics=True):
    result = dict(name='envelope', type='envelope', carrier=bands(), modulation=modulation())
    if diagnostics:
        result['diagnostics'] = dict(lags=[1,4,8], epsilon=1e-8)
    return result


def test_diagnostic_features_match_explicit_signal_formulas():
    branch = EnvelopeBranch(1, bands(), modulation(), dict(lags=[1,4],epsilon=1e-8)).double()
    x = torch.randn(2,256,1,dtype=torch.float64,requires_grad=True)
    f = torch.fft.rfftfreq(256,dtype=x.dtype)
    filtered = torch.fft.irfft(torch.fft.rfft(x.transpose(1,2),norm='ortho')[:,:,None,:] * branch.carrier(f)[None,None],n=256,norm='ortho')
    e = analytic_envelope(filtered);m=e.mean(-1,keepdim=True); ec=e-m
    power = torch.fft.rfft(ec,norm='ortho').abs().square()
    w=branch.modulation(f);w=w/w.sum(-1,keepdim=True)
    expected=[torch.log1p(m),torch.log1p(torch.einsum('bckf,mf->bckm',power,w))]
    yc=filtered-filtered.mean(-1,keepdim=True); var=yc.square().mean(-1,keepdim=True)
    expected += [torch.log1p(yc.pow(4).mean(-1,keepdim=True)/(var+1e-8).square()),
                 torch.log1p(ec.square().mean(-1,keepdim=True)/(m.square()+1e-8)),
                 torch.asinh((filtered[...,1:-1].square()-filtered[...,:-2]*filtered[...,2:]).mean(-1,keepdim=True)/(filtered.square().mean(-1,keepdim=True)+1e-8))]
    for lag in (1,4):
        a,b=ec[...,:-lag],ec[...,lag:]
        expected += [(a*b).mean(-1,keepdim=True)/((a.square().mean(-1,keepdim=True)+1e-8)*(b.square().mean(-1,keepdim=True)+1e-8)).sqrt()]
    actual=branch(x)
    torch.testing.assert_close(actual,torch.cat(expected,-1).flatten(1))
    assert actual.shape[-1] == len(branch.feature_names) == branch.output_dim == 7
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    # Zero signal has well-defined regularized values and gradients.
    zero=torch.zeros_like(x,requires_grad=True);branch(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_optional_diagnostics_preserve_old_coordinates_and_state():
    old = EnvelopeBranch(2,bands(),modulation())
    new = EnvelopeBranch(2,bands(),modulation(),dict(lags=[1],epsilon=1e-8))
    new.load_state_dict(old.state_dict(),strict=True)
    x=torch.randn(3,256,2)
    torch.testing.assert_close(new(x).view(3,2,1,6)[...,:2].flatten(1),old(x),rtol=0,atol=0)
    assert set(old.state_dict()) == set(new.state_dict())


@pytest.mark.parametrize('diagnostics', [dict(lags=[0],epsilon=1e-8),dict(lags=[True],epsilon=1e-8),
    dict(lags=[2,2],epsilon=1e-8),dict(lags=[2],epsilon=0),dict(lags=[2],epsilon=True),
    dict(lags=[2],epsilon=float('nan')),dict(lags=[2]),dict(lags=[2],epsilon=1e-8,auto=True)])
def test_invalid_physical_support_is_not_repaired(diagnostics):
    with pytest.raises(ValueError): EnvelopeBranch(1,bands(),modulation(),diagnostics)


def test_short_lag_support_is_rejected():
    branch=EnvelopeBranch(1,bands(),modulation(),dict(lags=[255],epsilon=1e-8))
    with pytest.raises(ValueError,match='two observed pairs'): branch(torch.randn(1,256,1))


def test_branch_head_zero_anchor_initialization_and_sensitivity():
    head=OperatorResidualHead(7,16,3).double();cap=torch.tensor(5.,dtype=torch.float64)
    z=torch.randn(9,7,dtype=torch.float64)
    assert torch.equal(head(z,cap),torch.zeros(9,3,dtype=z.dtype))
    with torch.no_grad():
        head.output.weight.normal_(0,2);head.hidden.weight.mul_(30);head.hidden.bias.fill_(.7)
    assert torch.equal(head(torch.zeros_like(z),cap),torch.zeros(9,3,dtype=z.dtype))
    w1,w2=head.effective_weights(cap)
    assert w1.norm()<=cap.sqrt()+1e-12 and w2.norm()<=cap.sqrt()+1e-12
    zz=torch.randn_like(z)
    assert torch.all((head(z,cap)-head(zz,cap)).norm(dim=-1)<=cap*(z-zz).norm(dim=-1)+1e-12)
    torch.testing.assert_close(head(z,cap).sum(-1),torch.zeros(9,dtype=z.dtype),atol=1e-12,rtol=0)


class _Stats(nn.Module):
    def forward(self,x):
        return torch.cat([x.mean(1),x.square().mean(1).sqrt(),x.std(1)],-1)


class _Reference(nn.Module):
    """Explicit component fixture. Full original-TSPN coverage is in runtime tests."""
    def __init__(self):
        super().__init__();self.args=SimpleNamespace(num_classes=3,in_channels=1)
        self.channel_for_classifier=3;self.signal_processing_layers=nn.ModuleList()
        self.feature_extractor_layers=_Stats();self.clf=nn.Linear(3,3)


def test_residual_keeps_complete_reference_and_has_exact_branch_margins():
    torch.manual_seed(4)
    reference=_Reference()
    model=TSPNFusion(reference,in_channels=1,num_classes=3,branches=[envelope()],
                     head_type='operator_residual',reference_temperature=1.7)
    baseline={key:value.clone() for key,value in reference.state_dict().items()}
    x=torch.randn(8,256,1);initial=model.forward_details(x)
    torch.testing.assert_close(initial['candidate_probs'],initial['raw_probs'],rtol=0,atol=0)
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.03)
    for _ in range(2):
        opt.zero_grad();out=model.forward_details(x)
        nn.functional.cross_entropy(out['candidate_logits'],torch.arange(8)%3).backward();opt.step()
    assert all(torch.equal(baseline[k],v) for k,v in reference.state_dict().items())
    assert all(p.grad is None for p in reference.parameters())
    out=model.forward_details(x)
    parts=out['branch_logit_contributions']; total=torch.stack(list(parts.values())).sum(0)
    torch.testing.assert_close(out['candidate_logits'],out['raw_logits']/1.7+total)
    rawlp=(out['raw_logits']/1.7).log_softmax(-1);qlp=out['candidate_logits'].log_softmax(-1)
    torch.testing.assert_close((qlp[:,0]-qlp[:,1])-(rawlp[:,0]-rawlp[:,1]),total[:,0]-total[:,1],atol=2e-7,rtol=1e-5)
    model.eval();model.set_alpha(0)
    torch.testing.assert_close(model.predict_proba(x),out['raw_probs'])
    model.set_alpha(1);torch.testing.assert_close(model.predict_proba(x),out['candidate_probs'])
    assert not reference.training


def test_legacy_head_state_keys_have_no_residual_parameters():
    for kind in ('linear','mlp'):
        model=TSPNFusion(_Reference(),in_channels=1,num_classes=3,branches=[],head_type=kind)
        assert not any(k.startswith('operator_heads.') for k in model.state_dict())
        assert 'candidate_head.weight' in model.state_dict()
        assert 'branch_logit_contributions' not in model.forward_details(torch.randn(2,256,1))


def _prediction_fixture(n=10):
    y=np.arange(n)%2
    raw=np.eye(2)[y]*.6+.2
    q=raw.copy()
    ids=np.array([str(i) for i in range(n)])
    return raw,q,y,ids,ids,np.repeat('condition',n)


def test_accuracy_qualification_is_not_gain_evidence():
    data=_prediction_fixture()
    dev=report(*data,stage='development')
    assert dev['development_competence_pass']
    assert dev['by_domain'][0]['baseline_accuracy_lower'] is None
    assert not dev['all_candidate_gains_supported']
    same=report(*data,stage='confirmation',independent_groups=True)
    assert same['by_domain'][0]['candidates'][0]['accuracy_gain']==0
    assert not same['all_candidate_gains_supported']
    with pytest.raises(ValueError,match='sampling declaration'): report(*data,stage='confirmation')


def test_group_accuracy_not_window_accuracy():
    # A has two acquisitions (one right), B one acquisition right. Group accuracy .75;
    # acquisition accuracy 2/3; duplicated windows in A do not change its group weight.
    raw=np.array([[.9,.1],[.9,.1],[.1,.9],[.1,.9]])
    y=np.array([0,0,0,1]);group=np.array(['A','A','A','B']);acq=np.array(['a','a','b','c']);d=np.repeat('d',4)
    r=report(raw,raw,y,group,acq,d,stage='development')
    assert r['by_domain'][0]['baseline_accuracy']==.75
    assert not r['development_competence_pass']


def test_gain_report_can_pass_with_enough_independent_groups():
    raw,q,y,g,a,d=_prediction_fixture(10000)
    raw[::10]=raw[::10,::-1]
    r=report(raw,q,y,g,a,d,stage='confirmation',independent_groups=True)
    c=r['by_domain'][0]['candidates'][0]
    assert c['accuracy_gain']==pytest.approx(.1)
    assert c['brier_excess']<0 and c['accuracy_gain_lower']>0
    assert r['all_candidate_gains_supported']


def test_no_positive_claim_from_brier_without_accuracy_gain():
    raw,q,y,g,a,d=_prediction_fixture()
    q=np.eye(2)[y]*.8+.1
    r=report(raw,q,y,g,a,d,stage='confirmation',independent_groups=True)
    c=r['by_domain'][0]['candidates'][0]
    assert c['brier_excess']<0 and c['accuracy_gain']==0 and not c['gain_supported']


@pytest.mark.parametrize('kwargs',[dict(competence=.79),dict(minimum_gain=-.1),dict(failure=0),dict(method='auto')])
def test_gain_report_rejects_weakened_or_invalid_contract(kwargs):
    with pytest.raises(ValueError):report(*_prediction_fixture(),stage='development',**kwargs)
