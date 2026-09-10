"""Finite operator/gradient checks. ReferenceFixture is explicitly not real TSPN."""
from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from src.model_factory.X_model.TSPN_fusion import TSPNFusion, mixture_log_probs
from src.model_factory.X_model.TSPN_tf_operators import STFTMap, CWTMap, StockwellMap, ChirpletMap, TimeFrequencyReadout, TimeFrequencyBranch
from src.task_factory.Components.tspn_fusion_loss import TSPNFusionLoss, domain_means

torch.set_num_threads(1)

class Moments(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.BatchNorm1d(4)
    def forward(self,x):
        x=x[:,:,0]
        return self.norm(torch.stack([x.mean(1),x.std(1,unbiased=False),x[:,0],x[:,-1]],1))

class ReferenceFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.args=SimpleNamespace(num_classes=3,in_channels=1)
        self.signal_processing_layers=nn.ModuleList([nn.Identity()])
        self.feature_extractor_layers=Moments();self.clf=nn.Linear(4,3);self.channel_for_classifier=4
    def forward(self,x):
        return self.clf(self.feature_extractor_layers(x))


def config(all_ops=False):
    branches=[
      {'name':'envelope','type':'envelope',
       'carrier':{'centers':[.19,.32],'widths':[.025,.025],
                  'center_bounds':[[.16,.22],[.29,.35]],'width_bounds':[[.015,.035],[.015,.035]]},
       'modulation':{'centers':[.025,.06,.1],'widths':[.0045,.0045,.0045],
                     'center_bounds':[[.021,.029],[.056,.064],[.096,.104]],
                     'width_bounds':[[.003,.006],[.003,.006],[.003,.006]]}},
      {'name':'stft_short','type':'stft','transform':{'win_length':64,'n_fft':128,'hop_length':16},
       'readout':{'row_groups':8,'time_bins':4,'log_floor':.001}},
      {'name':'morlet','type':'cwt','transform':{'scales':[4.,6.,9.,14.],'wavelet':'morlet','omega0':6.,'truncate':4.,'stride':4},
       'readout':{'time_bins':4,'log_floor':.001}},
      {'name':'stft_long','type':'stft','transform':{'win_length':128,'n_fft':128,'hop_length':16},
       'readout':{'row_groups':8,'time_bins':4,'log_floor':.001}},
      {'name':'mexican_hat','type':'cwt','transform':{'scales':[2.,4.,6.,9.],'wavelet':'mexican_hat','truncate':4.,'stride':4},
       'readout':{'time_bins':4,'log_floor':.001}},
      {'name':'stockwell','type':'stockwell','transform':{'frequencies':[.05,.09,.15,.23,.3],'width_factor':1.,'truncate':4.,'stride':4},
       'readout':{'time_bins':4,'log_floor':.001}},
      {'name':'chirplet_up','type':'chirplet','transform':{'frequencies':[.08,.14,.22,.3],'window_sigma':10.,'chirp_rate':.0002,'truncate':4.,'stride':4},
       'readout':{'time_bins':4,'log_floor':.001}}]
    return {'model':{'branches':branches if all_ops else branches[:3]}}


def model(all_ops=False,branches=None):
    return TSPNFusion(ReferenceFixture(),in_channels=1,num_classes=3,
                     branches=config(all_ops)['model']['branches'] if branches is None else branches)


def output(logits,raw=None):
    if raw is None: raw=torch.zeros_like(logits)
    return {'candidate_logits':logits,'raw_logits':raw,'candidate_probs':logits.softmax(-1),
            'raw_probs':raw.softmax(-1),'reference_temperature':1.}


@pytest.mark.parametrize('nfft',[32,64])
def test_stft_direct_sum(nfft):
    t=STFTMap(32,nfft,8).double();x=torch.randn(2,96,1,dtype=torch.double)
    actual=t.coefficients(x);frame=x[:,16:48,0]
    n=torch.arange(32,dtype=torch.double);f=torch.arange(nfft//2+1,dtype=torch.double)/nfft
    expected=torch.einsum('bn,fn->bf',(frame*t.window).to(torch.cdouble),torch.exp(-2j*math.pi*f[:,None]*n))
    torch.testing.assert_close(actual[:,0,:,2],expected,atol=1e-11,rtol=1e-11)


def test_padding_does_not_change_enbw():
    a,b=STFTMap(32,32,8),STFTMap(32,128,8)
    assert a.enbw()==pytest.approx(1.5/32) and b.enbw()==pytest.approx(a.enbw())
    assert len(a.frequency)!=len(b.frequency)


@pytest.mark.parametrize('wavelet',['morlet','mexican_hat'])
def test_cwt_independent_sum(wavelet):
    t=CWTMap([5.,8.],wavelet=wavelet,truncate=4.,stride=2).double()
    x=torch.randn(1,128,1,dtype=torch.double);actual=t.coefficients(x)
    b=int(t.times(128)[4]);scale=5.;lag=np.arange(-t.radius,t.radius+1);u=lag/scale
    if wavelet=='morlet': psi=np.pi**(-.25)*(np.exp(6j*u)-np.exp(-18))*np.exp(-u*u/2)
    else: psi=2/(np.sqrt(3)*np.pi**.25)*(1-u*u)*np.exp(-u*u/2)
    expected=np.sum(x[0,b-t.radius:b+t.radius+1,0].numpy()*np.conj(psi)/np.sqrt(scale))
    np.testing.assert_allclose(actual[0,0,0,4].item(),expected,atol=1e-11,rtol=1e-11)


def test_stockwell_absolute_phase():
    t=StockwellMap([.1,.2],stride=3).double();x=torch.randn(1,160,1,dtype=torch.double)
    actual=t.coefficients(x);center=int(t.times(160)[7]);f=.1;sigma=1/f
    n=np.arange(center-t.radius,center+t.radius+1)
    expected=np.sum(x[0,n,0].numpy()*np.exp(-(n-center)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)*np.exp(-2j*np.pi*f*n))
    np.testing.assert_allclose(actual[0,0,0,7].item(),expected,atol=1e-11,rtol=1e-11)


def test_chirplet_independent_sum():
    t=ChirpletMap([.1,.2],10.,chirp_rate=.0002,stride=2).double()
    x=torch.randn(1,160,1,dtype=torch.double);actual=t.coefficients(x);b=int(t.times(160)[3])
    r=np.arange(-t.radius,t.radius+1)
    atom=np.pi**(-.25)/np.sqrt(10)*np.exp(-r*r/200)*np.exp(-2j*np.pi*(.1*r+.0002*r*r/2))
    expected=np.sum(x[0,b-t.radius:b+t.radius+1,0].numpy()*atom)
    np.testing.assert_allclose(actual[0,0,0,3].item(),expected,atol=1e-11,rtol=1e-11)


def test_chirplet_rate_changes_representation():
    a=ChirpletMap([.1,.2],10.,chirp_rate=.0003);b=ChirpletMap([.1,.2],10.,chirp_rate=-.0003)
    x=torch.randn(1,200,1)
    assert not torch.allclose(a.coefficients(x).abs(),b.coefficients(x).abs())


@pytest.mark.parametrize('tf',[STFTMap(32,64,8),CWTMap([4.,6.]),StockwellMap([.1,.2]),ChirpletMap([.1,.2],8.)])
def test_transform_input_gradient(tf):
    x=torch.randn(2,160,1,requires_grad=True)
    tf.coefficients(x).abs().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_wavelet_scale_support_rejected():
    with pytest.raises(ValueError,match='Nyquist'):CWTMap([1.,2.])


def test_chirplet_support_rejected():
    with pytest.raises(ValueError,match='crosses'):ChirpletMap([.48],10.,chirp_rate=.001)


def test_insufficient_measured_support_rejected():
    with pytest.raises(ValueError,match='Measured signal'):StockwellMap([.02]).coefficients(torch.randn(1,128,1))


def test_no_empty_readout_bins():
    with pytest.raises(ValueError,match='time positions'):
        TimeFrequencyReadout(4,1,4,.01)(torch.ones(1,1,4,2,dtype=torch.complex64))


def test_readout_zero_signal_finite_gradients():
    x=torch.zeros(2,1,4,32,dtype=torch.cfloat,requires_grad=True)
    y=TimeFrequencyReadout(4,1,4,.01)(x);y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


def test_all_configured_branches_and_reference_frozen():
    m=model(True).train();snapshot={k:v.clone() for k,v in m.reference.state_dict().items()}
    out=m.forward_details(torch.randn(9,512,1))
    assert list(out['branch_features'])==list(m.feature_dims) and len(m.branches)==7
    loss=TSPNFusionLoss(lambda_delta=0)(out,torch.arange(9)%3,torch.arange(9),torch.arange(9)//3)
    loss['loss'].backward()
    assert not m.reference.training and all(p.grad is None for p in m.reference.parameters())
    assert all(torch.equal(snapshot[k],v) for k,v in m.reference.state_dict().items())
    assert m.candidate_head.weight.grad.abs().sum()>0
    assert m.branches['envelope'].carrier.center_parameter.grad.abs().sum()>0
    for i,(name,dim) in enumerate(m.feature_dims.items()):
        start=sum(list(m.feature_dims.values())[:i])
        assert m.candidate_head.weight.grad[:,start:start+dim].abs().sum()>0,name


def test_each_tf_branch_reaches_head():
    m=model(True).train();original=m.forward_details(torch.randn(6,512,1))
    base=original['candidate_logits']
    for name in m.branches:
        with torch.no_grad():
            start=0
            for key,dim in m.feature_dims.items():
                if key==name:break
                start+=dim
            contribution=F.linear(original['branch_features'][name]/math.sqrt(dim),m.effective_head_weight()[:,start:start+dim])
            assert contribution.abs().sum()>0 and not torch.allclose(base-contribution,base)


def test_refit_head_control_has_no_side_operator():
    m=model(branches=[])
    assert len(m.branches)==0 and m.forward_details(torch.randn(2,512,1))['candidate_logits'].shape==(2,3)


def test_unknown_and_duplicate_operators_fail():
    with pytest.raises(ValueError,match='Unknown'):TimeFrequencyBranch('fake_sst',1,{},dict(time_bins=4,log_floor=.01))
    c=config()['model']['branches']
    with pytest.raises(ValueError,match='unique'):model(branches=[c[1],c[1]])


def test_deployment_alpha_zero_does_not_evaluate_operators():
    m=model().eval();x=torch.randn(2,512,1)
    def fail(*args):raise AssertionError('a side operator ran at alpha=0')
    m.branches['stft_short'].forward=fail
    torch.testing.assert_close(m.predict_proba(x),m.reference(x).softmax(-1),rtol=0,atol=0)


def test_probability_mixture_not_logits():
    a=torch.tensor([[8.,0.,0.]]);b=torch.tensor([[0.,2.,0.]])
    actual=mixture_log_probs(a,b,.3).exp()
    torch.testing.assert_close(actual,.7*a.softmax(-1)+.3*b.softmax(-1))
    assert not torch.allclose(actual,(.7*a+.3*b).softmax(-1))


def test_checkpoint_complete_restore(tmp_path):
    m=model(True).eval();m.set_alpha(.3);x=torch.randn(2,512,1)
    pred=m.predict_proba(x);torch.save(m.state_dict(),tmp_path/'m.pt')
    n=model(True).eval();n.load_state_dict(torch.load(tmp_path/'m.pt',weights_only=True))
    torch.testing.assert_close(n.predict_proba(x),pred,rtol=0,atol=0)


def test_training_deployment_forward_rejected():
    with pytest.raises(RuntimeError,match='forward_details'):model().train()(torch.randn(2,512,1))


def test_domain_means_not_window_weighted():
    value=torch.tensor([1.,1.,3.,8.,10.]);units=torch.tensor([0,0,1,2,3]);dom=torch.tensor([0,0,0,1,1])
    _,v=domain_means(value,units,dom);torch.testing.assert_close(v,torch.tensor([2.,9.]))


def test_duplicate_windows_do_not_change_domain_loss():
    logits=torch.randn(6,3);raw=torch.randn(6,3);y=torch.arange(6)%3;u=torch.arange(6);d=torch.arange(6)//3
    f=TSPNFusionLoss(lambda_delta=0);a=f(output(logits,raw),y,u,d)['loss']
    ix=torch.tensor([0,0,1,2,3,4,5]);b=f(output(logits[ix],raw[ix]),y[ix],u[ix],d[ix])['loss']
    torch.testing.assert_close(a,b)


def test_worst_source_emphasizes_larger_relative_harm():
    logits=torch.tensor([[4.,0.],[4.,0.],[0.,4.],[0.,4.]],requires_grad=True)
    raw=torch.zeros_like(logits);y=torch.zeros(4,dtype=torch.long);u=torch.arange(4);d=torch.tensor([0,0,1,1])
    r=TSPNFusionLoss(lambda_delta=0)(output(logits,raw),y,u,d)
    assert r['domain_weights'][1]>r['domain_weights'][0] and r['source_envelope']>=r['max_source_excess']


def test_tau_one_composite_identity():
    logits=torch.randn(6,3);raw=torch.randn(6,3);y=torch.arange(6)%3;d=torch.arange(6)//3
    beta=.25;r=TSPNFusionLoss(lambda_delta=0,reduction='mean_source')(output(logits,raw),y,torch.arange(6),d)
    eye=F.one_hot(y,3)
    expected=F.cross_entropy(logits,y)-F.cross_entropy(raw,y)+beta*((logits.softmax(-1)-eye).square().sum(-1)-(raw.softmax(-1)-eye).square().sum(-1)).mean()
    torch.testing.assert_close(r['loss'],expected)


def test_zero_training_tau_rejected():
    with pytest.raises(ValueError):TSPNFusionLoss(tau=0)


def test_worst_source_one_domain_rejected():
    with pytest.raises(ValueError,match='two source domains'):
        TSPNFusionLoss(lambda_delta=0)(output(torch.randn(3,3)),torch.arange(3),torch.arange(3),torch.zeros(3,dtype=torch.long))


def test_pairs_supervised_even_when_consistency_zero():
    a,b=output(torch.randn(6,3)),output(torch.randn(6,3));y=torch.arange(6)%3;u=torch.arange(6);d=u//3
    f=TSPNFusionLoss(lambda_delta=0,reduction='mean_source')
    r=f(a,y,u,d,paired=b,paired_target=y,sample_ids=u,paired_sample_ids=u)
    torch.testing.assert_close(r['loss'],.5*(f(a,y,u,d)['loss']+f(b,y,u,d)['loss']))


def test_correction_invariance_not_raw_imitation():
    a=output(torch.randn(6,3),torch.randn(6,3));y=torch.arange(6)%3;u=torch.arange(6);d=u//3
    r=TSPNFusionLoss(lambda_delta=.1)(a,y,u,d,paired=a,paired_target=y,sample_ids=u,paired_sample_ids=u)
    assert float(r['correction_consistency'])==0 and (a['candidate_probs']-a['raw_probs']).abs().sum()>0


def test_ce_prevents_wrong_confident_brier_gradient_saturation():
    logits=torch.tensor([[10.,-10.]],requires_grad=True);y=torch.tensor([1])
    b=(logits.softmax(-1)-F.one_hot(y,2)).square().sum();gb=torch.autograd.grad(b,logits,retain_graph=True)[0].norm()
    gc=torch.autograd.grad(F.cross_entropy(logits,y),logits)[0].norm()
    assert gc>1. and gb<1e-6


def test_finite_extreme_logits_training():
    logits=torch.tensor([[1000.,-1000.],[-1000.,1000.]],requires_grad=True)
    y=torch.tensor([1,0]);u=torch.arange(2);d=u.clone()
    r=TSPNFusionLoss(lambda_delta=0)(output(logits),y,u,d)
    r['loss'].backward();assert torch.isfinite(r['loss']) and torch.isfinite(logits.grad).all()


def test_no_accuracy_guarantee_from_brier():
    raw=torch.tensor([[.51,.49],[.51,.49]]);new=torch.tensor([[.49,.51],[.99,.01]])
    y=torch.zeros(2,dtype=torch.long);one=F.one_hot(y,2)
    assert (new-one).square().sum(-1).mean()<(raw-one).square().sum(-1).mean()
    assert (new.argmax(-1)==y).float().mean()<(raw.argmax(-1)==y).float().mean()


def test_short_optimization_updates_candidate_not_reference():
    torch.manual_seed(2026)
    m=model().train();x=torch.randn(9,512,1);y=torch.arange(9)%3;u=torch.arange(9);d=u//3
    before={k:v.clone() for k,v in m.reference.state_dict().items()}
    objective=TSPNFusionLoss(lambda_delta=0)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    initial=float(objective(m.forward_details(x),y,u,d)['loss'].detach())
    for _ in range(12):
        opt.zero_grad(set_to_none=True);objective(m.forward_details(x),y,u,d)['loss'].backward();opt.step()
    final=float(objective(m.forward_details(x),y,u,d)['loss'].detach())
    assert final<initial and all(torch.equal(before[k],v) for k,v in m.reference.state_dict().items())


def test_same_label_permuted_pairs_are_rejected():
    a=output(torch.randn(4,2));labels=torch.zeros(4,dtype=torch.long);ids=torch.arange(4);domains=ids//2
    with pytest.raises(ValueError,match='row correspondence'):
        TSPNFusionLoss()(a,labels,ids,domains,paired=a,paired_target=labels,
                         sample_ids=ids,paired_sample_ids=ids.flip(0))


def test_pairing_requires_sample_ids_not_just_labels():
    a=output(torch.randn(4,2));labels=torch.zeros(4,dtype=torch.long);ids=torch.arange(4);domains=ids//2
    with pytest.raises(ValueError,match='sample identities'):
        TSPNFusionLoss()(a,labels,ids,domains,paired=a,paired_target=labels)
