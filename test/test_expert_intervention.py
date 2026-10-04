"""Scientific-operator fixtures, not fault-diagnosis performance evidence."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.model_factory.MoE.M_05_FixedRouteMoE import Model
from src.task_factory.task.DG.expert_intervention import (
    bounded_radius, brier, execute, fit_roles, fixed_route_deletions,
    load_pack, paired_edits, paired_interventions, training_objective, validate_dg,
)
from src.task_factory.task.DG.expert_intervention.audit import audit
from src.task_factory.task.DG.expert_intervention.preparation import read_signal
from src.task_factory.task.DG.expert_intervention.training import load_checkpoint


@pytest.fixture
def feature_pack(tmp_path):
    rng = np.random.default_rng(2)
    n, k, d = 32, 2, 4
    views = rng.normal(size=(n,k,d)).astype('float32')
    raw = rng.normal(size=(n,d)).astype('float32')
    fields = dict(views=views, raw=raw, compatibility=np.zeros((n,k), dtype='float32'),
        labels=np.tile([0,0,1,1],8), group=np.repeat(np.arange(16).astype(str),2),
        specimen=np.repeat(np.arange(16).astype(str),2),
        split=np.repeat(['train','val','match','test'],8),
        domain=np.array(['a']*4+['b']*4+['a']*8+['b']*8+['target']*8),
        probe_views=np.repeat(views[:,None,:,:],k,axis=1)+.05,
        control_views=np.repeat(views[:,None,:,:],k,axis=1)-.05,
        probe_raw=np.repeat(raw[:,None,:],k,axis=1)+.05,
        control_raw=np.repeat(raw[:,None,:],k,axis=1)-.05)
    fields.update(role_names=np.array(['low_order','high_order']), class_names=np.array(['normal','fault']))
    path = tmp_path/'fixture.npz'
    np.savez(path,**fields)
    return path, fields


def separated_packs(fields, directory):
    paths = []
    for name, parts in (('source', ('train','val')), ('audit', ('match','test'))):
        mask = np.isin(fields['split'], parts)
        values = {key: value if key in {'role_names','class_names'} else value[mask]
                  for key,value in fields.items()}
        path = directory/f'{name}.npz'
        np.savez(path,**values)
        paths.append(path)
    return paths


@pytest.fixture
def config():
    return dict(trainer=dict(device='cpu'), task=dict(expert_intervention=dict(arm='aligned',
        seed=20, epochs=2, width=4, batch_size=4, lr=.001, weight_decay=.0001,
        balance=.01, dg=True, alpha=.05, independent_groups=False, intervention='replacement')))


def test_objective_value_and_gradient_keep_balancing_semantics():
    torch.manual_seed(1)
    logits = torch.randn(5,3,requires_grad=True)
    route_logits = torch.randn(5,2,requires_grad=True)
    gates = route_logits.softmax(-1)
    labels = torch.tensor([0,1,2,0,1])
    observed = training_objective(logits,gates,labels,.1,'aligned')
    reference = torch.nn.functional.cross_entropy(logits,labels)+.1*2*(gates.mean(0)-.5).square().sum()
    assert torch.equal(observed,reference)
    observed_grad = torch.autograd.grad(observed,(logits,route_logits),retain_graph=True)
    reference_grad = torch.autograd.grad(reference,(logits,route_logits),retain_graph=True)
    for actual, expected in zip(observed_grad,reference_grad):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    unbalanced = training_objective(logits,gates,labels,.1,'no_balance')
    torch.testing.assert_close(unbalanced,torch.nn.functional.cross_entropy(logits,labels))


def test_role_fit_excludes_losses_and_is_slot_permutation_equivariant():
    responses = np.tile(np.eye(3)[None,:,:],(4,1,1))
    groups = np.array(['a','a','b','b'])
    perm = np.array([2,0,1])
    np.testing.assert_array_equal(perm[fit_roles(responses[:,:,perm],groups)],fit_roles(responses,groups))
    # Duplicating every window of one group does not reweight that group.
    repeated = np.concatenate([responses[:2], responses[2:], responses[2:]])
    np.testing.assert_array_equal(fit_roles(repeated,np.array(['a','a','b','b','b','b'])),fit_roles(responses,groups))


def test_negative_replacement_contrast_can_restore_useful_expert():
    probabilities = np.array([[.8,.2],[.35,.65],[.8,.2],[.8,.2]])
    losses = brier(probabilities,np.zeros(4,dtype=int))
    assert losses[0]-losses[1]-(losses[2]-losses[3]) == pytest.approx(-.3825)


def test_spectral_probe_and_control_have_equal_l2_size():
    n = 128
    t = np.arange(n)/n
    x = 2*np.sin(2*np.pi*5*t)+np.cos(2*np.pi*15*t)
    bins = np.arange(n//2+1)
    probe, control = paired_edits(x,bins==5,bins==15,.1)
    assert np.linalg.norm(probe-x) == pytest.approx(np.linalg.norm(control-x))
    np.testing.assert_allclose(np.fft.rfft(probe-x)[bins!=5],0,atol=1e-12)
    np.testing.assert_allclose(np.fft.rfft(control-x)[bins!=15],0,atol=1e-12)


def test_normalization_is_source_only_and_checkpoint_scaling_is_reused(feature_pack,tmp_path):
    path, fields = feature_pack
    initial = load_pack(path)
    changed = dict(fields)
    changed['raw'] = fields['raw'].copy()
    changed['raw'][fields['split']=='test'] += 1000
    changed_path = tmp_path/'changed.npz'
    np.savez(changed_path,**changed)
    test_changed = load_pack(changed_path)
    np.testing.assert_array_equal(initial['raw_mean'],test_changed['raw_mean'])
    normalization = {key:initial[key] for key in ('raw_mean','raw_scale','views_mean','views_scale')}
    changed['raw'][fields['split']=='train'] += 1000
    np.savez(changed_path,**changed)
    restored = load_pack(changed_path,normalization=normalization)
    np.testing.assert_array_equal(initial['raw_mean'],restored['raw_mean'])


def test_physical_specimen_cannot_cross_partitions_even_with_distinct_groups(feature_pack,tmp_path):
    _, fields = feature_pack
    fields['specimen'] = fields['specimen'].copy()
    fields['specimen'][24:26] = fields['specimen'][0]
    path=tmp_path/'leak.npz'
    np.savez(path,**fields)
    with pytest.raises(ValueError,match='physical identity crosses partitions'):
        load_pack(path)


def test_source_domains_and_target_are_disjoint(feature_pack):
    path, _ = feature_pack
    data = load_pack(path)
    validate_dg(data)
    data['domain'][data['split']=='match'] = 'target'
    with pytest.raises(ValueError,match='training source domains'):
        validate_dg(data)


def test_fixed_route_deletion_equals_explicit_remaining_mixture():
    torch.manual_seed(2)
    model = Model(SimpleNamespace(input_dim=4,num_experts=3,num_classes=2,width=3,arm='aligned'))
    views, raw = torch.randn(2,3,4), torch.randn(2,4)
    gates = torch.tensor([[.2,.3,.5],[.4,.6,0.]])
    _, deleted = fixed_route_deletions(model,views,raw,gates)
    features = model.encode(views,raw)
    for index in range(3):
        keep = [slot for slot in range(3) if slot != index]
        remaining = (features[:,keep]*gates[:,keep,None]).sum(1)/(1-gates[:,index,None])
        torch.testing.assert_close(deleted[:,index],model.head(remaining).softmax(-1))
    with pytest.raises(ValueError,match='mass one'):
        fixed_route_deletions(model,views,raw,torch.tensor([[1.,0.,0.],[.4,.6,0.]]))


def test_same_clean_route_is_used_for_both_paired_edits(feature_pack):
    path, _ = feature_pack
    data = load_pack(path)
    torch.manual_seed(2)
    model = Model(SimpleNamespace(input_dim=4,num_experts=2,num_classes=2,width=4,arm='aligned'))
    ids = np.flatnonzero(data['split']=='test')
    observed = paired_interventions(model,data,ids,torch.device('cpu'))
    raw = torch.as_tensor(data['raw'][ids])
    views = torch.as_tensor(data['views'][ids])
    _, gates, clean = model(views,raw,torch.as_tensor(data['compatibility'][ids]))
    for role in range(2):
        for pair,prefix in enumerate(('probe','control')):
            base,replaced,_ = model.fixed_route_replacements(torch.as_tensor(data[f'{prefix}_views'][ids,role]),
                torch.as_tensor(data[f'{prefix}_raw'][ids,role]),gates,clean)
            np.testing.assert_allclose(observed['base'][:,role,pair],base.detach().numpy(),rtol=0,atol=0)
            np.testing.assert_allclose(observed['replaced'][:,role,:,pair],replaced.detach().numpy(),rtol=0,atol=0)


def test_explicit_checkpoint_roundtrip_train_does_not_evaluate(feature_pack,config,tmp_path):
    _, fields = feature_pack
    source_path, audit_path = separated_packs(fields,tmp_path)
    train = execute(config,phase='train',data=source_path,output=tmp_path/'train')
    assert train['test_metrics'] == {}
    assert not (tmp_path/'train'/'audit_input.npz').exists()
    saved_path = Path(train['best_checkpoint'])
    evaluation = execute(config,phase='evaluate',data=audit_path,checkpoint=saved_path,output=tmp_path/'evaluate')
    replay = execute(config,phase='evaluate',data=audit_path,checkpoint=saved_path,output=tmp_path/'replay')
    assert evaluation['test_metrics'] == replay['test_metrics']
    with np.load(tmp_path/'evaluate'/'audit_input.npz') as first, np.load(tmp_path/'replay'/'audit_input.npz') as second:
        assert first['base'].shape == (16,2,2,2)
        assert first['replaced'].shape == (16,2,2,2,2)
        for key in first.files:
            np.testing.assert_array_equal(first[key],second[key])
    model, saved = load_checkpoint(saved_path,torch.device('cpu'))
    assert not model.training
    assert saved['selected_validation_ce'] == train['selected_validation_ce']
    with pytest.raises(ValueError,match='checkpoint'):
        execute(config,phase='evaluate',data=audit_path,output=tmp_path/'missing_checkpoint')
    assert not (tmp_path/'missing_checkpoint').exists()
    with np.load(audit_path) as data:
        changed = {key:data[key] for key in data.files}
    changed['specimen'][changed['specimen'] == changed['specimen'][0]] = '0'
    overlap = tmp_path/'overlap.npz'
    np.savez(overlap,**changed)
    with pytest.raises(ValueError,match='overlaps the checkpoint'):
        execute(config,phase='evaluate',data=overlap,checkpoint=saved_path,output=tmp_path/'leak')
    assert not (tmp_path/'leak').exists()
    with np.load(audit_path) as data:
        changed = {key:data[key] for key in data.files}
    changed['role_names'] = changed['role_names'][::-1]
    permuted = tmp_path/'permuted.npz'
    np.savez(permuted,**changed)
    with pytest.raises(ValueError,match='role_names differ'):
        execute(config,phase='evaluate',data=permuted,checkpoint=saved_path,output=tmp_path/'bad_schema')


def test_source_training_rejects_target_containing_pack_before_fit(feature_pack,config,tmp_path):
    path, _ = feature_pack
    with pytest.raises(ValueError,match='separate source and audit packs'):
        execute(config,phase='train',data=path,output=tmp_path/'invalid_train')
    assert not (tmp_path/'invalid_train').exists()


def test_tuning_validates_every_candidate_before_training(feature_pack,config,tmp_path):
    _, fields = feature_pack
    source, _ = separated_packs(fields,tmp_path)
    config['task']['expert_intervention'].update(search_arms=['generic','aligned','shuffled'],
        learning_rates=[.001,-1],weight_decays=[.0001,.01])
    with pytest.raises(ValueError,match='lr must be finite'):
        execute(config,phase='tune',data=source,output=tmp_path/'bad_search')
    assert not (tmp_path/'bad_search').exists()


def test_duplicate_raw_acquisition_is_rejected_before_read(tmp_path):
    import csv
    import json
    from src.task_factory.task.DG.expert_intervention import prepare
    records = [dict(file='missing.mat',reader='mat',signal_key='x',channel='0',fs_hz='100',rpm='60',
                    group=group,specimen=group,split=split,domain=domain,label='0')
               for group,split,domain in [('a','train','source'),('b','test','target')]]
    manifest=tmp_path/'manifest.csv'
    with manifest.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    roles=tmp_path/'roles.json'
    roles.write_text(json.dumps([dict(name=name,target_orders=[1,3],control_orders=[4,6]) for name in ('a','b')]))
    with pytest.raises(ValueError,match='Duplicate declared acquisition'):
        prepare(manifest,tmp_path,roles,tmp_path/'pack.npz',16,.1)
    assert not (tmp_path/'pack.npz').exists()


def test_renamed_visibility_cannot_select_physical_gpu_two(config,tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2')
    with pytest.raises(ValueError,match='Physical GPU 2'):
        execute(config,phase='train',data=tmp_path/'unused.npz',device='cuda:0',output=tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


def test_invalid_data_fails_before_creating_output(config,tmp_path):
    with pytest.raises(FileNotFoundError):
        execute(config,phase='train',data=tmp_path/'missing.npz',output=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_confidence_bounds_cannot_count_duplicate_specimens_as_independent(tmp_path):
    n,r,k,c = 8,2,2,2
    base=np.full((n,r,2,c),.5)
    replaced=np.full((n,r,k,2,c),.5)
    fields=dict(base=base,replaced=replaced,response=np.tile(np.eye(2)[None],(n,1,1)),
        labels=np.zeros(n,dtype=int),group=np.repeat(['a','b','c','d'],2),
        split=np.repeat(['match','test'],4),domain=np.repeat('one',n),
        specimen=np.repeat(['first','first','second','second'],2),model='aligned',seed=0)
    path=tmp_path/'responses.npz'
    np.savez(path,**fields)
    with pytest.raises(ValueError,match='duplicate a specimen'):
        audit(path,tmp_path/'audit',independent_groups=True)
    assert bounded_radius(4,2,.05) > bounded_radius(40,2,.05)


def test_declared_h5_reader_uses_channel_without_flattening(tmp_path):
    import h5py
    signal=np.arange(60,dtype=float).reshape(20,3,1)
    path=tmp_path/'signals.h5'
    with h5py.File(path,'w') as stream:
        stream['recording']=signal
    actual=read_signal(dict(file=path.name,reader='h5',signal_key='recording',channel='2'),tmp_path)
    np.testing.assert_array_equal(actual,signal[:,2,0])
    with pytest.raises(ValueError,match='valid channel'):
        read_signal(dict(file=path.name,reader='h5',signal_key='recording',channel='3'),tmp_path)
