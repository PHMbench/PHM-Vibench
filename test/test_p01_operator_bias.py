from pathlib import Path
import copy
from types import SimpleNamespace
import sys
import pytest
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.model_factory.X_model.P01OperatorBias import Model,PhysicalBands,envelope
from src.task_factory.Components.p01_bias_losses import covariance_loss,retention_loss,unit_mean,path_margins,replay_der_losses


def model(**kwargs):
    return Model(SimpleNamespace(num_classes=3,in_channels=1,**kwargs))


def data(n=257,b=4):
    return torch.randn(b,n,1),{'sample_rate_hz':torch.full((b,),2048.),'rotation_speed_rpm':torch.tensor([480.,600.,840.,1080.])[:b]}


def test_same_forward_reconstruction_and_gradients():
    torch.manual_seed(7); m=model();x,meta=data()
    output=m.forward_details(x,physical_metadata=meta)
    assert output['path_names']==('bias','raw','periodic','envelope','stft')
    torch.testing.assert_close(output['logits'],output['contributions'].sum(1),rtol=0,atol=0)
    torch.testing.assert_close(output['contributions'].mean(-1),torch.zeros(4,5),atol=1e-6,rtol=0)
    loss=F.cross_entropy(output['logits'],torch.tensor([0,1,2,1]));loss.backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    for p in m.readouts.values(): assert p.weight.grad.abs().sum()>0


def test_physical_laws_per_sample():
    m=model();fs=torch.tensor([2048.,2048.]);rho=torch.tensor([8.,16.])
    fc,bw=m.bands['periodic'].physical(fs,rho)
    torch.testing.assert_close(fc[1],2*fc[0]);torch.testing.assert_close(bw[1],2*bw[0])
    fc,bw=m.carrier.physical(fs,rho)
    torch.testing.assert_close(fc[0],fc[1]);torch.testing.assert_close(bw[0],bw[1])


def test_no_metadata_broadcast_or_guessed_speed():
    m=model();x,meta=data()
    with pytest.raises(ValueError):m(x)
    meta['rotation_speed_rpm']=torch.tensor([600.])
    with pytest.raises(ValueError):m(x,physical_metadata=meta)


def test_metadata_rows_preserve_file_identity():
    rows={11:{'sample_rate_hz':2048.,'rotation_speed_rpm':480.},27:{'sample_rate_hz':2048.,'rotation_speed_rpm':960.}}
    m=Model(SimpleNamespace(num_classes=3,in_channels=1),rows);x=torch.randn(2,256,1)
    fs,rho=m._physical_metadata(x,torch.tensor([27,11]),None)
    torch.testing.assert_close(rho,torch.tensor([16.,8.]))


@pytest.mark.parametrize('n',[255,256,257])
def test_odd_even_signal_shape(n):
    x,meta=data(n=n)
    assert envelope(x).shape==x.shape
    assert model()(x,physical_metadata=meta).shape==(4,3)


def test_band_support_fails_without_clipping():
    m=model();x,meta=data();meta['sample_rate_hz']=torch.full((4,),100.)
    with pytest.raises(ValueError,match='observable support'):m(x,physical_metadata=meta)


def test_stft_does_not_erase_frequency_distribution():
    m=model(paths=['stft']);t=torch.arange(1024)/2048
    x=torch.stack([torch.sin(2*torch.pi*30*t),torch.sin(2*torch.pi*140*t)])[:,:,None]
    meta={'sample_rate_hz':torch.full((2,),2048.),'rotation_speed_rpm':torch.full((2,),600.)}
    z=m.forward_details(x,physical_metadata=meta)['features']['stft']
    assert not torch.allclose(z[0],z[1],atol=1e-3)


def test_state_unchanged_during_eval_and_checkpoint_roundtrip(tmp_path):
    m=model().eval();x,meta=data();state=copy.deepcopy(m.state_dict());expected=m(x,physical_metadata=meta)
    for key,value in state.items():torch.testing.assert_close(value,m.state_dict()[key],rtol=0,atol=0)
    torch.save(m.state_dict(),tmp_path/'m.pt');n=model().eval();n.load_state_dict(torch.load(tmp_path/'m.pt',weights_only=True))
    torch.testing.assert_close(expected,n(x,physical_metadata=meta))


def test_teacher_stops_gradients_and_bias_is_constrained():
    current=torch.randn(3,5,3,requires_grad=True);teacher=torch.randn(3,5,3,requires_grad=True)
    y=torch.tensor([0,1,2]);ids=torch.arange(3)
    loss=retention_loss(current,teacher,y,ids);loss.backward()
    assert teacher.grad is None and current.grad is not None
    assert current.grad[:,0].abs().sum()>0


def test_one_sided_allows_improvement_not_deterioration():
    old=torch.zeros(2,3,3);new=old.clone();y=torch.tensor([0,1]);ids=torch.arange(2)
    new[0,:,0]=1;new[1,:,1]=1
    assert retention_loss(new,old,y,ids).item()==0
    assert retention_loss(-new,old,y,ids).item()>0
    assert retention_loss(new,old,y,ids,'path_symmetric').item()>0


def test_path_cancellation_is_visible_but_total_margin_is_not():
    old=torch.zeros(1,3,3);new=old.clone();new[0,1,0]=1;new[0,2,0]=-1
    y=torch.tensor([0]);ids=torch.tensor([0])
    assert retention_loss(new,old,y,ids,'total_one_sided').item()==0
    assert retention_loss(new,old,y,ids).item()>0


def test_unit_weights_not_window_weights():
    values=torch.tensor([1.,1.,1.,5.]);ids=torch.tensor([0,0,0,1])
    assert unit_mean(values,ids).item()==3


def test_covariance_uses_same_path_and_trainable_gradient():
    a=torch.randn(2,4,3,requires_grad=True);b=torch.randn(2,4,3,requires_grad=True);ids=torch.arange(2)
    assert covariance_loss(a,a,ids).item()==0
    covariance_loss(a,b,ids).backward()
    assert a.grad[:,1:].abs().sum()>0 and b.grad[:,1:].abs().sum()>0
    assert a.grad[:,0].abs().sum()==0


def test_retention_ce_upper_bound():
    torch.manual_seed(10);old=torch.randn(8,5,4);new=torch.randn(8,5,4);y=torch.arange(8)%4
    decline=(path_margins(old,y)-path_margins(new,y)).relu()
    bound=(5*decline.square().sum((1,2))).sqrt()
    increase=F.cross_entropy(new.sum(1),y,reduction='none')-F.cross_entropy(old.sum(1),y,reduction='none')
    assert (increase<=bound+1e-6).all()


def test_der_uses_given_historical_logits():
    x=torch.randn(2,3,requires_grad=True);old=torch.randn(2,3,requires_grad=True);y=torch.tensor([0,1]);ids=torch.arange(2)
    ce,mse=replay_der_losses(x,old,y,ids)
    (ce+mse).backward();assert old.grad is None
