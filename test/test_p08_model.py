"""Actual maintained HSE, not an imitation; constructed tensors are software tests."""
from types import SimpleNamespace
import pytest
import torch
from src.model_factory.ISFM.M_P08_PhysicalConditioning import Model


def args(coordinates='physical', fusion='film'):
    return SimpleNamespace(coordinates=coordinates, fusion=fusion, output_dim=8,
        num_classes=3, condition_dim=8, patch_size_L=8, num_patches=4, nhead=2)


def model(coordinates='physical', fusion='film'):
    torch.manual_seed(3)
    return Model(args(coordinates, fusion)).eval()


def inputs():
    torch.manual_seed(7)
    return torch.randn(4, 32, 1), torch.randn(4, 8)


def test_index_coordinates_ignore_rate_but_real_hse_uses_it():
    x, p = inputs()
    index = model('index', 'none')
    assert torch.equal(index(x, fs=1., condition=p), index(x, fs=100., condition=p))
    physical = model('physical', 'none')
    assert not torch.allclose(physical(x, fs=1., condition=p), physical(x, fs=100., condition=p))


def test_shared_initialization_and_active_parameter_disclosure():
    reference = model(fusion='none').state_dict()
    for fusion in ('film', 'token_concat', 'late_concat'):
        m = model(fusion=fusion)
        assert all(torch.equal(v, m.state_dict()[key]) for key, v in reference.items())
        counts = m.parameter_counts()
        assert counts['stored_parameters'] > counts['active_parameters'] > counts['active_conditional_parameters'] > 0


@pytest.mark.parametrize('fusion', ['film', 'token_concat', 'late_concat'])
def test_condition_changes_output_but_detached_path_does_not(fusion):
    x, p = inputs(); m = model(fusion=fusion)
    p.requires_grad_()
    out = m(x, fs=10., condition=p)
    out.square().sum().backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert not torch.allclose(out, m(x, fs=10., condition=p+1))
    assert torch.equal(m(x, fs=10., condition=p, detach_condition=True),
                       m(x, fs=10., condition=p+1, detach_condition=True))


def test_checkpoint_and_no_id_lookup():
    x, p = inputs(); m = model()
    clone = model(); clone.load_state_dict(m.state_dict(), strict=True)
    assert torch.equal(m(x, file_id=['source']*4, fs=10., condition=p),
                       clone(x, file_id=['unseen']*4, fs=10., condition=p))
    with pytest.raises(ValueError): m(x, task_id='another-head', fs=10., condition=p)
    with pytest.raises(ValueError): m(x, fs=10., condition=torch.zeros(4, 7))
    with pytest.raises(ValueError): m(x[:, :4], fs=10., condition=p)


def test_explicit_patch_coordinates_are_shared_and_validated():
    x, p = inputs(); m = model()
    starts = torch.tensor([[0, 8, 16, 24]]).expand(4, -1)
    channels = torch.zeros_like(starts)
    a = m(x, fs=10., condition=p, start_indices_L=starts, start_indices_C=channels)
    b = m(x, fs=10., condition=p, start_indices_L=starts, start_indices_C=channels)
    assert torch.equal(a, b)
    with pytest.raises(ValueError): m(x, fs=10., condition=p, start_indices_L=starts)


def test_factory_forward_finite_gradient_and_feature_interface():
    from src.model_factory import build_model
    configured = args()
    configured.type = 'ISFM'
    configured.name = 'M_P08_PhysicalConditioning'
    m = build_model(configured, metadata=None).eval()
    x, p = inputs()
    logits, z = m(x, fs=10., condition=p, return_features=True)
    assert logits.shape == (4, 3) and z.shape == (4, 8)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
    for module in (m.embedding, m.norm, m.backbone, m.head, m.film):
        for parameter in module.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    with pytest.raises(ValueError):
        m(x, fs=10., condition=torch.full_like(p, float('nan')), detach_condition=True)


def test_small_batch_is_trainable_with_fixed_real_hse_patches():
    x, p = inputs(); m = model(fusion='none').train()
    labels = torch.tensor([0,1,0,1]); starts=torch.tensor([[0,8,16,24]]).expand(4,-1)
    channels=torch.zeros_like(starts)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.02)
    losses=[]
    for _ in range(60):
        optimizer.zero_grad()
        logits=m(x,fs=10.,condition=p,start_indices_L=starts,start_indices_C=channels)
        loss=torch.nn.functional.cross_entropy(logits,labels)
        assert torch.isfinite(loss)
        losses.append(float(loss.detach())); loss.backward(); optimizer.step()
    assert losses[-1] < .15 and losses[-1] < losses[0]/2


@pytest.mark.parametrize('key', ['embedding', 'backbone', 'task_head'])
def test_rejects_unexecuted_component_names(key):
    configured = args()
    setattr(configured, key, 'not-the-implemented-component')
    with pytest.raises(ValueError, match=f'P08 {key} must explicitly name'):
        Model(configured)
