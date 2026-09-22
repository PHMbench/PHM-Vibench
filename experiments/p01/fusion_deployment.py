"""Frozen H5 predictions -> existing assessment -> self-contained predictor -> test.

Neural candidates reuse TSPN_fusion and its strict loader. Temperature is applied
to the original logits without changing the reference temperature. A declared
source-mixture assessment is not a guarantee for an arbitrary new condition.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.p01.fusion_data import read_records, summarize_rows


def local_path(value,base):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base/p).resolve()


def load_model(checkpoint,device):
    from src.model_factory.model_factory import model_factory
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('kind')=='classifier':
        model=FrozenClassifier(saved,device).eval()
        return model,saved
    settings=copy.deepcopy(saved['model'])
    settings.update(checkpoint_kind='fusion',checkpoint_path=str(checkpoint),device=device)
    model=model_factory(SimpleNamespace(**settings),metadata=None).eval()
    return model,saved


class FrozenClassifier(torch.nn.Module):
    """Existing factory classifier paired with the same complete frozen reference."""
    def __init__(self,saved,device):
        super().__init__()
        from src.model_factory.model_factory import model_factory
        candidate=dict(saved['model'],device=device)
        reference=dict(saved['reference_model'],device=device)
        self.reference_settings=copy.deepcopy(saved['reference_model'])
        self.candidate=model_factory(SimpleNamespace(**candidate),metadata=None)
        self.reference=model_factory(SimpleNamespace(**reference),metadata=None)
        self.candidate.load_state_dict(saved['state_dict'],strict=True)
        self.reference.load_state_dict(saved['reference_state_dict'],strict=True)
        self.num_classes=int(candidate['num_classes'])
        if self.num_classes!=int(reference['num_classes']):
            raise ValueError('Baseline and frozen reference class spaces differ.')
        self.register_buffer('reference_temperature',torch.tensor(float(saved['reference_temperature']),dtype=torch.float64))
        self.to(device).eval().requires_grad_(False)

    def forward_details(self,x):
        return dict(raw_logits=self.reference(x),candidate_logits=self.candidate(x))

    def forward(self,x):
        return self.candidate(x)


def log_predictions(model,x,kind,temperature,alpha,contributions=None):
    from src.model_factory.X_model.TSPN_fusion import mixture_log_probs
    if kind=='model':
        out=model.forward_details(x);raw=out['raw_logits'];candidate=out['candidate_logits']
        if contributions is not None:
            contributions.update(out.get('branch_logit_contributions',{}))
    elif kind=='temperature':
        raw=model.reference(x)
        candidate=raw/(float(model.reference_temperature)*temperature)
    else:raise ValueError('Candidate kind must be model or temperature.')
    t0=float(model.reference_temperature)
    raw_lp=F.log_softmax(raw/t0,-1);q_lp=F.log_softmax(candidate,-1)
    final_lp=mixture_log_probs(raw,candidate,alpha,t0)
    return raw_lp,q_lp,final_lp


@torch.no_grad()
def predict_records(model,records,dataset,data,classes,device,kind='model',temperature=1.,alpha=1.):
    from experiments.p01.window_io import window_record
    if not records:raise ValueError('No acquisitions in the requested partition.')
    if not math.isfinite(temperature) or temperature<=0 or not 0<=alpha<=1:
        raise ValueError('Positive temperature and coefficient in [0,1] required.')
    if len(classes)!=model.num_classes or len(set(classes))!=len(classes):
        raise ValueError('Declare the ordered class names once.')
    arrays={k:[] for k in ('raw_probs','candidate_probs','deployed_probs','raw_log_probs','candidate_log_probs','deployed_log_probs',
                           'labels','group_ids','acquisition_ids','window_ids','domains')}
    for record in records:
        x=window_record(record,dataset,data['data']).to(device)
        contributions={}
        lp0,lpq,lpf=log_predictions(model,x,kind,temperature,alpha,contributions)
        # These explain the DIRECT candidate's log-odds correction. A probability
        # mixture at intermediate alpha is not additive in these logit terms.
        for name,value in contributions.items():
            arrays.setdefault('logit_contribution__'+name,[]).append(value.cpu().numpy())
        for prefix,lp in zip(('raw','candidate','deployed'),(lp0,lpq,lpf)):
            if not torch.isfinite(lp).all():raise FloatingPointError('Nonfinite model log probabilities.')
            arrays[prefix+'_log_probs'].append(lp.cpu().numpy())
            arrays[prefix+'_probs'].append(lp.exp().cpu().numpy())
        n=len(x)
        arrays['labels'].append(np.full(n,record['label'],dtype=np.int64))
        for field,value in [('group_ids',record['unit_id']),('acquisition_ids',record['acquisition_id']),('domains',record['domain'])]:
            arrays[field].append(np.repeat(str(value),n))
        arrays['window_ids'].append(np.asarray([str(i) for i in range(n)]))
    result={k:np.concatenate(v) for k,v in arrays.items()}
    result['raw_class_names']=np.asarray(classes,dtype=str)
    result['candidate_class_names']=np.asarray(classes,dtype=str)
    return result


def acquisition_rows(predictions):
    p=predictions;keys=list(zip(p['domains'],p['group_ids'],p['acquisition_ids']));buckets={}
    for i,key in enumerate(keys):buckets.setdefault(key,[]).append(i)
    rows=[];classes=p['raw_probs'].shape[1]
    for (domain,group,acquisition),idx in buckets.items():
        y=p['labels'][idx]
        if len(set(y.tolist()))!=1:raise ValueError('Class probability aggregation needs a constant-label acquisition.')
        for name in ('raw','candidate','deployed'):
            prob=p[name+'_probs'][idx];lp=p[name+'_log_probs'][idx]
            rows.append(dict(domain=domain,unit_id=group,acquisition_id=acquisition,label=int(y[0]),predictor=name,
                             prediction=int(prob.mean(0).argmax()),windows=len(idx),
                             ce=float(-lp[np.arange(len(idx)),y].mean()),
                             brier=float(((prob-np.eye(classes)[y])**2).sum(1).mean()),
                             mean_probabilities=json.dumps(prob.mean(0).tolist())))
    return rows


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save_bundle(model,model_settings,path,*,kind,temperature,alpha,classes,input_data,sampling_rate,scope):
    if not math.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('A deployment coefficient must lie in [0,1].')
    settings=copy.deepcopy(model_settings)
    if isinstance(model,FrozenClassifier):
        bundle=dict(kind='classifier',model=settings,
                    state_dict={k:v.detach().cpu().clone() for k,v in model.candidate.state_dict().items()},
                    reference_model=copy.deepcopy(model.reference_settings),
                    reference_state_dict={k:v.detach().cpu().clone() for k,v in model.reference.state_dict().items()},
                    reference_temperature=float(model.reference_temperature))
    else:
        settings.update(checkpoint_kind='fusion',checkpoint_path=str(Path(path).resolve()),device='cpu')
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        # Serialization must not mutate the predictor being compared or its buffers.
        state['alpha']=torch.tensor(float(alpha),dtype=model.alpha.dtype)
        bundle=dict(model=settings,state_dict=state)
    bundle['deployment']=dict(kind=kind,temperature=float(temperature),alpha=float(alpha),
                                reference_temperature=float(model.reference_temperature),class_names=list(classes),
                                probability_order='temperature_per_window_then_convex_mixture_then_acquisition_mean',
                                data=copy.deepcopy(input_data),sample_rate_hz=float(sampling_rate),scope=scope)
    torch.save(bundle,path)


def verify_vectors(expected,actual):
    for key in ('labels','group_ids','acquisition_ids','window_ids','domains','raw_class_names','candidate_class_names'):
        if not np.array_equal(expected[key],actual[key]):raise AssertionError(f'Restored {key} differ.')
    for key in ('raw_probs','candidate_probs','deployed_probs','raw_log_probs','candidate_log_probs','deployed_log_probs'):
        np.testing.assert_allclose(actual[key],expected[key],atol=1e-7,rtol=1e-6,err_msg=f'Restored {key}')
    names={key for key in expected if key.startswith('logit_contribution__')}
    if names!={key for key in actual if key.startswith('logit_contribution__')}:
        raise AssertionError('Restored explanation branches differ.')
    for key in names:
        np.testing.assert_allclose(actual[key],expected[key],atol=1e-7,rtol=1e-6,err_msg=f'Restored {key}')


def selection_risk(p):
    """Temperature selection minimizes absolute worst-source candidate Brier."""
    summary=summarize_rows(acquisition_rows(p),len(p['raw_class_names']))
    return max(row['brier'] for row in summary if row['predictor']=='candidate')


def selection_excess(p,alpha):
    q=dict(p)
    q['deployed_probs']=(1-alpha)*p['raw_probs']+alpha*p['candidate_probs']
    if alpha == 0:q['deployed_log_probs']=p['raw_log_probs']
    elif alpha == 1:q['deployed_log_probs']=p['candidate_log_probs']
    else:q['deployed_log_probs']=np.logaddexp(math.log1p(-alpha)+p['raw_log_probs'],math.log(alpha)+p['candidate_log_probs'])
    summary=summarize_rows(acquisition_rows(q),len(p['raw_class_names']))
    return max(next(r['brier'] for r in summary if r['domain']==d and r['predictor']=='deployed')-
               next(r['brier'] for r in summary if r['domain']==d and r['predictor']=='raw') for d in set(p['domains']))


def run(plan_path,output,device):
    from experiments.p01.calibrate_tspn_fusion import assess_plan
    from experiments.p01.fusion_assessment import same_reference
    plan_path=Path(plan_path).resolve();plan=yaml.safe_load(plan_path.read_text());base=plan_path.parent
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    dpath=local_path(plan['data_config'],base);data=yaml.safe_load(dpath.read_text())
    dataset=next(d for d in data['datasets'] if d['name']==plan['dataset'])
    records=read_records(dataset,data);sources=list(map(str,dataset['source_domains']));classes=list(plan['class_names'])
    if len(classes)!=int(data['model']['num_classes']):raise ValueError('Class-order length differs from the data.')
    if len({r['sample_rate_hz'] for r in records})!=1:raise ValueError('Current normalized-frequency model needs one fixed sampling rate.')
    mode=plan['mode']
    if mode not in {'independent','empirical'}:raise ValueError('Choose independent or empirical explicitly.')
    val=[r for r in records if r['domain'] in sources and r['split']=='validation']
    assessment=[r for r in records if r['domain'] in sources and r['split']=='assessment']
    observed=assessment if mode=='independent' else val
    grid=sorted(set(float(a) for a in plan['selection_alpha_grid']))
    if not grid or grid[0]!=0 or grid[-1]>1 or any(not math.isfinite(a) or a<0 for a in grid):
        raise ValueError('Predeclare a coefficient grid in [0,1] including the raw model.')
    names=[c['name'] for c in plan['candidates']]
    if not names or len(set(names))!=len(names):raise ValueError('Candidate names must be unique.')
    predictions=[];selected_specs=[];selection_rows=[];bank=[]
    torch.set_num_threads(1)
    for i,item in enumerate(plan['candidates']):
        checkpoint=local_path(item['checkpoint'],base);model,saved=load_model(checkpoint,device)
        if int(saved['model']['reference_config']['in_dim'])!=int(data['data']['window_size']):
            raise ValueError('Declared window differs from the original reference interval.')
        kind=item['kind'];temperatures=item.get('temperature_grid',[1.]) if kind=='temperature' else [1.]
        trials=[]
        for temp in temperatures:
            p=predict_records(model,val,dataset,data,classes,device,kind,float(temp),1.)
            trials.append((selection_risk(p),float(temp),p))
        _,temp,p=min(trials,key=lambda t:(t[0],t[1]))
        score,alpha=min((selection_excess(p,a),a) for a in grid)
        selected=dict(name=item['name'],kind=kind,checkpoint=str(checkpoint),temperature=temp,fixed_alpha=alpha)
        selected_specs.append(selected);selection_rows.append(dict(candidate=item['name'],temperature=temp,alpha=alpha,worst_source_brier_excess=score))
        p=predict_records(model,observed,dataset,data,classes,device,kind,temp,1.)
        if predictions:same_reference(predictions[0]['raw_probs'],classes,p['raw_probs'],classes)
        predictions.append(p);path=root/f'frozen_candidate_{i}.npz';np.savez(path,**p)
        # Temperature is already applied to logits. Assess that fixed exported
        # function once, rather than applying calibration a second time.
        bank.append(dict(name=item['name'],kind='model',predictions=str(path),fixed_alpha=alpha))
    write_csv(root/'source_selection.csv',selection_rows)
    (root/'plan.yaml').write_text(yaml.safe_dump(plan,sort_keys=False))
    if mode=='independent':
        history=[str(local_path(p,base)) for p in plan['development_group_files']]
        if not history:raise ValueError('Independent assessment needs the actual reference and candidate development groups; otherwise choose empirical mode.')
        current=root/'current_development_groups.csv'
        write_csv(current,[dict(group_id=g) for g in sorted({r['unit_id'] for r in records if r['domain'] in sources and r['split'] in {'update','validation'}})])
        rule=plan['rule']
        if rule not in {'moments','paired'}:raise ValueError('Choose one primary assessment rule.')
        assessment_plan=dict(scope=plan['scope'],bound=plan['bound'],delta_total=plan['delta_total'],delta_shift=plan['delta_shift'],
                             rule_budgets={rule:float(plan['delta_total'])-float(plan['delta_shift'])},
                             candidates=bank,conditions=sources,development_group_files=history+[str(current)])
        for key in ('response_kind','response'):
            if key in plan:assessment_plan[key]=plan[key]
        apath=root/'assessment_plan.yaml';apath.write_text(yaml.safe_dump(assessment_plan,sort_keys=False))
        result=assess_plan(apath,root/'decision.json');chosen=result['rules'][rule]['selected'];index=chosen['candidate'];alpha=chosen['alpha']
        scope=dict(mode=mode,assessment_scope=plan['scope'],rule=rule,
                   external_test='empirical evaluation; a source-mixture statement does not cover new physical conditions')
    else:
        index=min(range(len(selection_rows)),key=lambda i:(selection_rows[i]['worst_source_brier_excess'],selection_rows[i]['alpha']))
        alpha=selection_rows[index]['alpha'];scope=dict(mode=mode,assessment_scope='none',rule='source_selection',
                external_test='empirical; complete original-model development history not established')
        (root/'decision.json').write_text(json.dumps(dict(selected=index,alpha=alpha,scope=scope),indent=2))
    spec=selected_specs[index];model,saved=load_model(spec['checkpoint'],device)
    expected=predict_records(model,observed,dataset,data,classes,device,spec['kind'],spec['temperature'],alpha)
    bundle=root/'deployment.pt'
    save_bundle(model,saved['model'],bundle,kind=spec['kind'],temperature=spec['temperature'],alpha=alpha,classes=classes,
                input_data=data['data'],sampling_rate=records[0]['sample_rate_hz'],scope=scope)
    split='assessment' if mode=='independent' else 'validation'
    common=[sys.executable,str(Path(__file__).resolve()),'predict','--bundle',str(bundle),'--data-config',str(dpath),
            '--dataset',plan['dataset'],'--device',device]
    subprocess.run(common+['--split',split,'--sources-only','--output',str(root/'restored_check')],check=True)
    with np.load(root/'restored_check/predictions.npz',allow_pickle=False) as restored:verify_vectors(expected,restored)
    # Test follows source selection, assessment, saving and separate-process
    # restoration. No parameter is recomputed from these test predictions.
    subprocess.run(common+['--split','test','--output',str(root/'test')],check=True)
    report=dict(dataset=plan['dataset'],selected_candidate=spec['name'],candidate_kind=spec['kind'],alpha=alpha,
                reference_temperature=float(model.reference_temperature),candidate_temperature=spec['temperature'],
                separate_process_probability_check=True,scope=scope,
                independent_groups={s:len({r['unit_id'] for r in records if r['split']==s}) for s in ('update','validation','assessment','test')})
    (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def predict(bundle,data_path,dataset_name,split,output,device,sources_only=False):
    model,saved=load_model(Path(bundle).resolve(),device);spec=saved['deployment']
    data=yaml.safe_load(Path(data_path).read_text());dataset=next(d for d in data['datasets'] if d['name']==dataset_name)
    if data['data']!=spec['data']:raise ValueError('Restoration cannot change the stored observation/preprocessing rule.')
    records=read_records(dataset,data)
    if any(r['sample_rate_hz']!=spec['sample_rate_hz'] for r in records):raise ValueError('Sampling convention changed.')
    selected=[r for r in records if r['split']==split and (not sources_only or r['domain'] in list(map(str,dataset['source_domains'])))]
    p=predict_records(model,selected,dataset,data,spec['class_names'],device,spec['kind'],spec['temperature'],spec['alpha'])
    root=Path(output);root.mkdir(parents=True,exist_ok=False);np.savez(root/'predictions.npz',**p)
    rows=acquisition_rows(p);summary=summarize_rows(rows,len(spec['class_names']))
    write_csv(root/'acquisitions.csv',rows);write_csv(root/'conditions.csv',summary)
    changes=[]
    for raw in (r for r in rows if r['predictor']=='raw'):
        deployed=next(r for r in rows if r['predictor']=='deployed' and (r['domain'],r['unit_id'],r['acquisition_id'])==(raw['domain'],raw['unit_id'],raw['acquisition_id']))
        changes.append(dict(domain=raw['domain'],unit_id=raw['unit_id'],acquisition_id=raw['acquisition_id'],label=raw['label'],
                            raw_prediction=raw['prediction'],deployed_prediction=deployed['prediction'],
                            repaired=int(raw['prediction']!=raw['label'] and deployed['prediction']==raw['label']),
                            damaged=int(raw['prediction']==raw['label'] and deployed['prediction']!=raw['label']),
                            brier_change=deployed['brier']-raw['brier']))
    write_csv(root/'decision_changes.csv',changes)
    (root/'scope.json').write_text(json.dumps(spec,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='action',required=True)
    runp=commands.add_parser('run');runp.add_argument('--plan',required=True);runp.add_argument('--output',required=True);runp.add_argument('--device',default='cpu')
    pred=commands.add_parser('predict')
    for key in ('bundle','data-config','dataset','output'):pred.add_argument('--'+key,required=True)
    pred.add_argument('--split',choices=['validation','assessment','test'],required=True);pred.add_argument('--device',default='cpu');pred.add_argument('--sources-only',action='store_true')
    a=p.parse_args()
    if a.action=='run':run(a.plan,a.output,a.device)
    else:predict(a.bundle,a.data_config,a.dataset,a.split,a.output,a.device,a.sources_only)


if __name__=='__main__':main()
