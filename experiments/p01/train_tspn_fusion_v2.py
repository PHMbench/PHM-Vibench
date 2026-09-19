"""Fit a candidate on existing H5: group sampling and independent source selection.

Window losses are averaged within acquisition and then within physical group.
Assessment is never materialized here. Deployment alpha stays zero.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.p01.fusion_data import read_records,group_pools,sample_units,summarize_rows


def write_csv(path,rows):
    if not rows:raise ValueError('No result rows to write.')
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def f1(labels,predictions,classes):
    matrix=np.zeros((classes,classes),int);np.add.at(matrix,(labels,predictions),1)
    den=matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2*np.diag(matrix),den,out=np.zeros(classes),where=den!=0).mean())


def selection_score(summary,brier_weight=.25,predictor='candidate'):
    if not math.isfinite(brier_weight) or brier_weight<0 or predictor not in {'candidate','training_tau_mixture'}:
        raise ValueError('Invalid selection weight or predictor.')
    excess=[]
    for domain in sorted({r['domain'] for r in summary}):
        raw=next(r for r in summary if r['domain']==domain and r['predictor']=='raw')
        new=next(r for r in summary if r['domain']==domain and r['predictor']==predictor)
        excess.append(new['ce']-raw['ce']+brier_weight*(new['brier']-raw['brier']))
    if not excess or not np.isfinite(excess).all():raise ValueError('No finite source validation scores.')
    return float(max(excess))


def feature_trace(model,out,epoch,step):
    """Actual normalized inputs and raw-parameter gradient blocks, before update.

    These are sampled source-training trajectory diagnostics, not final-model
    population estimates and not a rule for rescaling any branch.
    """
    first=model.candidate_hidden if model.head_type=='mlp' else model.candidate_head
    gradient=first.weight.grad
    if gradient is None:raise RuntimeError('No candidate input-layer gradient.')
    cap=model.head_frobenius_cap.sqrt() if model.head_type=='mlp' else model.head_frobenius_cap
    effective=first.weight.detach()/(first.weight.detach().norm()/cap).clamp_min(1.)
    rows=[];offset=0
    for name,z in out['branch_features'].items():
        d=model.feature_dims[name];norm=(z.detach()/math.sqrt(d)).norm(dim=-1)
        rows.append(dict(epoch=epoch,step=step,branch=name,feature_dim=d,sampled_windows=len(norm),
                         mean_scaled_squared_norm=float(norm.square().mean()),
                         median_scaled_norm=float(torch.quantile(norm,.5)),
                         input_weight_gradient_norm=float(gradient[:,offset:offset+d].norm()),
                         input_weight_parameters=int(first.weight[:,offset:offset+d].numel()),
                         raw_input_weight_norm=float(first.weight[:,offset:offset+d].detach().norm()),
                         effective_input_weight_norm=float(effective[:,offset:offset+d].norm()),
                         effective_output_weight_norm=float(model.effective_head_weight().detach().norm())))
        offset+=d
    return rows


@torch.no_grad()
def evaluate(model,units,device,tau,beta,pack_batch,selection_predictor='candidate',prediction_arrays=None):
    from src.model_factory.X_model.TSPN_fusion import mixture_log_probs
    model.eval();rows=[]
    for u in units:
        x,y,_,_=pack_batch([u],device);out=model.forward_details(x)
        lp0=F.log_softmax(out['raw_logits']/float(model.reference_temperature),-1)
        lpq=F.log_softmax(out['candidate_logits'],-1)
        lpm=mixture_log_probs(out['raw_logits'],out['candidate_logits'],tau,float(model.reference_temperature))
        if prediction_arrays is not None:
            for name,lp in [('raw',lp0),('candidate',lpq),('training_tau_mixture',lpm)]:
                prediction_arrays.setdefault(name+'_log_probs',[]).append(lp.cpu().numpy())
                prediction_arrays.setdefault(name+'_probs',[]).append(lp.exp().cpu().numpy())
            for key,value in [('group_ids',u['unit_id']),('acquisition_ids',u.get('acquisition_id',u['unit_id'])),('domains',u['domain'])]:
                prediction_arrays.setdefault(key,[]).append(np.repeat(str(value),len(y)))
            prediction_arrays.setdefault('labels',[]).append(y.cpu().numpy())
            prediction_arrays.setdefault('window_ids',[]).append(np.asarray([str(i) for i in range(len(y))]))
        for name,lp in [('raw',lp0),('candidate',lpq),('training_tau_mixture',lpm)]:
            if not torch.isfinite(lp).all():raise FloatingPointError('Nonfinite validation prediction.')
            prob=lp.exp();one=F.one_hot(y,model.num_classes)
            rows.append(dict(unit_id=u['unit_id'],acquisition_id=u.get('acquisition_id',u['unit_id']),
                             domain=u['domain'],label=u['label'],predictor=name,
                             prediction=int(prob.mean(0).argmax()),ce=float(F.nll_loss(lp,y)),
                             brier=float((prob-one).square().sum(-1).mean()),windows=len(y),
                             mean_probabilities=json.dumps(prob.mean(0).cpu().tolist())))
    summary=summarize_rows(rows,model.num_classes)
    return selection_score(summary,beta,selection_predictor),rows,summary


@torch.no_grad()
def response_trace(out,paired,y,uids,dids,sources,epoch,step):
    """Source-trajectory moments, not independent assessment or tuning criteria."""
    from src.task_factory.Components.tspn_fusion_loss import domain_means
    p0=out['raw_probs'].detach();q=out['candidate_probs'].detach();v=q-p0
    one=F.one_hot(y,q.shape[-1]).to(q.dtype)
    moments={'A':v.square().sum(-1),'b':((one-p0)*v).sum(-1)};differences={}
    if paired is not None:
        p0p=paired['raw_probs'].detach();qp=paired['candidate_probs'].detach();vp=qp-p0p
        moments['A']=.5*(moments['A']+vp.square().sum(-1))
        moments['b']=.5*(moments['b']+((one-p0p)*vp).sum(-1))
        for name,difference in [('delta_p0',p0p-p0),('delta_q',qp-q),('delta_v',vp-v)]:
            moments[name+'_squared_norm']=difference.square().sum(-1)
            differences[name]=torch.stack([domain_means(difference[:,i],uids,dids)[1]
                                           for i in range(difference.shape[-1])],dim=-1).cpu().tolist()
    values={name:domain_means(value,uids,dids)[1].tolist() for name,value in moments.items()}
    rows=[]
    for i,d in enumerate(torch.unique(dids,sorted=True).tolist()):
        row=dict(epoch=epoch,step=step,domain=sources[d],**{name:v[i] for name,v in values.items()})
        row['sqrt_A']=math.sqrt(row['A'])
        row['b_over_sqrt_A']=row['b']/row['sqrt_A'] if row['A']>0 else None
        row.update({name+'_mean_vector':json.dumps(value[i]) for name,value in differences.items()})
        rows.append(row)
    return rows


def population_counts(units):
    return dict(groups=len({u['unit_id'] for u in units}),acquisitions=len(units),
                windows=sum(len(u['x']) for u in units),labelled_acquisitions=len(units))


def synchronized_time(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    return time.perf_counter()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model-config','data-config','dataset','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--steps-per-epoch',type=int,default=50)
    p.add_argument('--units-per-domain',type=int,default=2);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--pair-shift',type=int,default=0,help='Circular time-origin change, NOT speed augmentation.')
    p.add_argument('--selection-brier-weight',type=float,default=.25)
    p.add_argument('--selection-predictor',choices=['candidate','training_tau_mixture'],default='candidate')
    p.add_argument('--arm',default='candidate',help='Declared comparison arm; recorded without changing the model.')
    p.add_argument('--evaluate-test',action='store_true',help='Empirical candidate comparison after source choices are fixed; not assessed deployment.')
    args=p.parse_args()
    if min(args.epochs,args.steps_per_epoch,args.units_per_domain)<1 or not math.isfinite(args.lr) or args.lr<=0 or args.pair_shift<0:
        raise ValueError('Positive training budget and nonnegative pair shift required.')
    if not math.isfinite(args.selection_brier_weight) or args.selection_brier_weight<0:
        raise ValueError('Invalid independent selection weight.')
    from experiments.p01.window_io import materialize,pack_batch
    from src.model_factory.model_factory import model_factory
    from src.task_factory.Components.tspn_fusion_loss import TSPNFusionLoss
    cfg=yaml.safe_load(Path(args.model_config).read_text());data=yaml.safe_load(Path(args.data_config).read_text())
    dataset=next(d for d in data['datasets'] if d['name']==args.dataset)
    if cfg['model']['checkpoint_kind']!='reference':raise ValueError('Start from the declared frozen raw checkpoint.')
    classes=int(cfg['model']['reference_config']['num_classes'])
    if cfg['model']['num_classes']!=classes or data['model']['num_classes']!=classes:raise ValueError('Model, reference and dataset class spaces disagree.')
    if int(data['data']['window_size'])!=int(cfg['model']['reference_config']['in_dim']):raise ValueError('Window must match the reference interval.')
    if args.pair_shift>=int(data['data']['window_size']):raise ValueError('pair-shift must be shorter than the window.')
    if args.device.startswith('cuda'):
        import os
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not visible or ',' in visible or visible=='2' or not torch.cuda.is_available():raise ValueError('Select one available non-2 physical GPU explicitly.')
    torch.manual_seed(args.seed);np.random.seed(args.seed);rng=random.Random(args.seed)
    pair_rng=random.Random(args.seed+10000);torch.set_num_threads(1)
    cfg['model']['device']=args.device
    model=model_factory(SimpleNamespace(**cfg['model']),metadata=None).to(args.device)
    objective=TSPNFusionLoss(**cfg['loss'])
    if objective.lambda_delta>0 and args.pair_shift<1:raise ValueError('Supply a justified --pair-shift or set lambda_delta=0.')
    records=read_records(dataset,data);sources=list(map(str,dataset['source_domains']))
    if len({r['sample_rate_hz'] for r in records})!=1:raise ValueError('Normalized-frequency model requires one sampling rate.')
    load_started=time.perf_counter()
    train=materialize([r for r in records if r['domain'] in sources and r['split']=='update'],dataset,data)
    val=materialize([r for r in records if r['domain'] in sources and r['split']=='validation'],dataset,data)
    materialization_seconds=time.perf_counter()-load_started
    pools=group_pools(train,sources)
    if any(len(v)<args.units_per_domain for v in pools.values()):raise ValueError('Not enough independent source groups per domain.')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    (output/'model_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (output/'data_config.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    benchmark_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    command=dict(vars(args),argv=sys.argv,python=sys.executable,torch_version=torch.__version__,benchmark_commit=benchmark_commit,
                 pair_seed=args.seed+10000,pair_rng='random.Random; randint(1,pair_shift) once per batch')
    (output/'command.json').write_text(json.dumps(command,indent=2))
    write_csv(output/'development_groups.csv',[dict(group_id=g) for g in sorted({u['unit_id'] for u in train+val})])
    baseline={k:v.detach().clone() for k,v in model.reference.state_dict().items()}
    torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('reference.')},
               output/'initial_candidate_state.pt')
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    best=float('inf');log=[];domain_log=[];feature_log=[];batch_log=[];sampling_log=[];prior_log=[];response_log=[]
    training_seconds=0.;selection_seconds=0.;selected_epoch=None
    diagnostic_keys=('diagnostic_excess','max_source_excess','source_envelope','correction_consistency',
                     'risk_objective','pair_penalty','candidate_output_consistency','reference_output_consistency')
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    started=synchronized_time(args.device)
    for epoch in range(args.epochs):
        train_started=synchronized_time(args.device)
        model.train();losses=[];diagnostics=[]
        for step in range(args.steps_per_epoch):
            units=sample_units(pools,args.units_per_domain,rng)
            x,y,uids,_=pack_batch(units,torch.device(args.device))
            dids=torch.cat([torch.full((len(u['x']),),sources.index(u['domain']),dtype=torch.long) for u in units]).to(args.device)
            sample_ids=torch.arange(len(y),device=y.device,dtype=torch.long)
            out=model.forward_details(x);paired=None;shift=0
            if args.pair_shift>0:
                shift=pair_rng.randint(1,args.pair_shift)
                paired=model.forward_details(x.roll(shift,dims=1))
            for index,u in enumerate(units):
                sampling_log.append(dict(epoch=epoch,step=step,draw=index,domain=u['domain'],group_id=u['unit_id'],
                    acquisition_id=u.get('acquisition_id',u['unit_id']),windows=len(u['x']),
                    window_ids=';'.join(map(str,range(len(u['x'])))),pair_shift=shift))
            terms=objective(out,y,uids,dids,paired=paired,paired_target=y if paired is not None else None,
                            sample_ids=sample_ids,paired_sample_ids=sample_ids if paired is not None else None)
            if not torch.isfinite(terms['loss']):raise FloatingPointError('Nonfinite training objective.')
            optimizer.zero_grad(set_to_none=True);terms['loss'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite candidate gradient.')
            if model.reference.training or any(p.grad is not None for p in model.reference.parameters()):
                raise AssertionError('Frozen reference entered training mode or acquired a gradient.')
            feature_log.extend(feature_trace(model,out,epoch,step))
            response_log.extend(response_trace(out,paired,y,uids,dids,sources,epoch,step))
            optimizer.step();losses.append(float(terms['loss'].detach()))
            values={key:float(terms[key].detach()) for key in diagnostic_keys}
            diagnostics.append([values[key] for key in diagnostic_keys])
            batch_log.append(dict(epoch=epoch,step=step,loss=losses[-1],**values))
            domain_ids=terms['domains'].tolist();reference_risks=terms['domain_reference_risk'].tolist()
            for d,risk,reference_risk,excess,weight in zip(domain_ids,terms['domain_candidate_risk'].tolist(),reference_risks,
                    terms['domain_excess'].tolist(),terms['domain_weights'].tolist()):
                domain_log.append(dict(epoch=epoch,step=step,domain=sources[d],candidate_risk=risk,
                    reference_risk=reference_risk,excess=excess,weight=weight))
            for i,d in enumerate(domain_ids):
                for j,other in enumerate(domain_ids):
                    if i<j:prior_log.append(dict(epoch=epoch,step=step,domain_i=sources[d],domain_j=sources[other],
                        reference_risk_difference_over_rho=(reference_risks[i]-reference_risks[j])/objective.domain_temperature))
        training_seconds+=synchronized_time(args.device)-train_started
        selection_started=synchronized_time(args.device);prediction_arrays={}
        score,val_rows,summary=evaluate(model,val,args.device,objective.tau,args.selection_brier_weight,pack_batch,args.selection_predictor,prediction_arrays)
        selection_seconds+=synchronized_time(args.device)-selection_started
        means=np.mean(diagnostics,axis=0)
        log.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),worst_validation_relative_score=score,
                        **{('batch_mean_'+key if key in {'diagnostic_excess','max_source_excess','source_envelope'} else key):
                           float(value) for key,value in zip(diagnostic_keys,means)}))
        write_csv(output/'training.csv',log);write_csv(output/'training_domains.csv',domain_log)
        write_csv(output/'training_features.csv',feature_log)
        write_csv(output/'training_batches.csv',batch_log);write_csv(output/'training_responses.csv',response_log)
        write_csv(output/'sampling.csv',sampling_log)
        if prior_log:write_csv(output/'training_reference_prior.csv',prior_log)
        if score<best:
            best=score;selected_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'model':cfg['model'],'epoch':epoch,'arm':args.arm,'seed':args.seed,
                        'benchmark_commit':benchmark_commit},output/'selected_candidate.pt')
            write_csv(output/'selected_source_validation.csv',summary);write_csv(output/'selected_source_validation_units.csv',val_rows)
            arrays={key:np.concatenate(values) for key,values in prediction_arrays.items()}
            class_names=data['model'].get('class_names')
            if class_names is not None:
                if len(class_names)!=classes or len(set(class_names))!=classes:raise ValueError('Invalid ordered class names.')
                arrays.update(raw_class_names=np.asarray(class_names,dtype=str),candidate_class_names=np.asarray(class_names,dtype=str))
            arrays.update(arm=np.asarray(args.arm),seed=np.asarray(args.seed),checkpoint=np.asarray(str(output/'selected_candidate.pt')),
                          benchmark_commit=np.asarray(benchmark_commit),model_config=np.asarray(str(output/'model_config.yaml')))
            np.savez_compressed(output/'selected_source_validation_windows.npz',**arrays)
        print(log[-1],flush=True)
    if args.device.startswith('cuda'):torch.cuda.synchronize()
    total_seconds=time.perf_counter()-started
    if not all(torch.equal(baseline[k],v) for k,v in model.reference.state_dict().items()):raise AssertionError('Frozen raw state changed.')
    selected=torch.load(output/'selected_candidate.pt',map_location=args.device,weights_only=True)
    model.load_state_dict(selected['state_dict'],strict=True);model.eval()
    if args.evaluate_test:
        test=materialize([r for r in records if r['split']=='test'],dataset,data)
        _,rows,summary=evaluate(model,test,args.device,objective.tau,args.selection_brier_weight,pack_batch,args.selection_predictor)
        write_csv(output/'test_units.csv',rows);write_csv(output/'test_domains.csv',summary)
    note=dict(status='candidate_fitted',deployment_alpha=float(model.alpha),independent_assessment_performed=False,
              reference_state_unchanged=True,selection_brier_weight=args.selection_brier_weight,selection_predictor=args.selection_predictor,
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),branch_feature_dimensions=model.feature_dims,
              total_parameters=sum(p.numel() for p in model.parameters()),reference_parameters=sum(p.numel() for p in model.reference.parameters()),
              head_type=model.head_type,training_and_selection_seconds=training_seconds+selection_seconds,
              training_seconds=training_seconds,source_selection_seconds=selection_seconds,total_run_seconds=total_seconds,
              source_materialization_seconds=materialization_seconds,fit_access=population_counts(train),selection_access=population_counts(val),
              sampled_acquisitions=len(sampling_log),sampled_windows=sum(r['windows'] for r in sampling_log),
              supervised_endpoints=2 if args.pair_shift>0 else 1,optimizer_steps=args.epochs*args.steps_per_epoch,
              selected_epoch=selected_epoch,selection_tie_break='earliest epoch (strictly smaller score only)',
              risk_reference=objective.risk_reference,consistency_target=objective.consistency_target,
              peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
              permanent_test_predicted=args.evaluate_test,
              aggregation='windows/acquisition; equal acquisitions/physical group; equal groups/condition',
              classification='group-balanced acquisition accuracy and macro-F1 from a weighted confusion matrix',
              feature_diagnostics='sampled source-training inputs after dimension scaling; not frozen population moments',
              interpretation='training_tau_mixture is not independently assessed deployment')
    (output/'result_scope.json').write_text(json.dumps(note,indent=2));print(output)


if __name__=='__main__':main()
