"""Bounded P08 experiments using PHMFactory's native data/model/task/trainer owners.

fit/tune never open a target inventory. evaluate requires a frozen checkpoint and
an independently supplied target inventory. compare/plot read predictions only.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
import numpy as np


def _plain(x):
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k):_plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [_plain(v) for v in x]
    if isinstance(x, SimpleNamespace) or hasattr(x, '__dict__'): return _plain(vars(x))
    if hasattr(x,'item'): return x.item()
    return x


def write(path, data):
    Path(path).write_text(json.dumps(_plain(data), indent=2, allow_nan=False)+'\n')


def configuration(args, overrides=()):
    from phmfactory.config import analyze_config
    analysis = analyze_config(args.config, local_config=args.local_config,
                              override_values=list(args.override)+list(overrides))
    c = analysis.effective_config
    if c['data']['factory_name'] != 'p08' or (c['task']['type'],c['task']['name']) != ('DG','p08'):
        raise ValueError('Select the native P08 data and task owners')
    if c['model']['name'] != 'M_P08_PhysicalConditioning' or c['trainer']['test_after_fit']:
        raise ValueError('Source fitting requires the declared P08 model and test_after_fit=false')
    if c['environment']['iterations'] != 1:
        raise ValueError('Use explicit repeated source seeds, not an implicit seed sequence')
    return analysis


def preflight(args, out):
    from src.configs.config_utils import dict_to_namespace
    from src.data_factory import build_data
    analysis = configuration(args)
    c = analysis.effective_config
    write(out/'config.json', c)
    factory = build_data(dict_to_namespace(c['data']), dict_to_namespace(c['task']))
    write(out/'source_contract.json', factory.metadata.p08_contract)
    summary = {'source_records':len(factory.metadata.df),'train_windows':len(factory.train_dataset),
               'validation_windows':len(factory.val_dataset),'target_opened':False, 'condition_dim':factory.train_dataset.conditions.shape[1]}
    write(out/'preflight.json', summary)
    return summary


def fit_once(args, out, overrides=()):
    import torch
    from src.runtime import run_classification_pipeline
    analysis = configuration(args, overrides)
    write(out/'config.json', analysis.effective_config)
    # The same compiled config and lifecycle used by the public PHMFactory path.
    invocation = SimpleNamespace(config_path=str(args.config), compiled_run_spec=analysis,
        resolved_pipeline=analysis.pipeline, local_config=args.local_config, override=[], notes='P08 source-only fit')
    started = time.monotonic()
    result = run_classification_pipeline(invocation)
    checkpoint = torch.load(result['best_checkpoint'],map_location='cpu',weights_only=False)
    score = checkpoint['source_unit_brier']
    if score is None or not np.isfinite(score): raise ValueError('No finite source validation selection metric')
    selected = {'checkpoint':result['best_checkpoint'],'source_unit_brier':float(score),
                'fit_seconds':time.monotonic()-started,'target_opened':False,
                'config':analysis.effective_config,'native_result':result}
    write(out/'fit.json', selected)
    return selected


def tune(args, out):
    cfg = configuration(args).effective_config
    search = cfg['p08_search']
    if set(search) != {'arms','learning_rates','weight_decays'}: raise ValueError('Unrecognized search controls')
    arms = search['arms']; rates=search['learning_rates']; decays=search['weight_decays']
    if any(not np.isfinite(x) or x <= 0 for x in rates) or any(not np.isfinite(x) or x < 0 for x in decays):
        raise ValueError('Invalid explicit search space')
    if len(set(arms)) != len(arms) or any(a not in {'none','film','late_concat','token_concat'} for a in arms):
        raise ValueError('Declare distinct supported fusion arms')
    trials = list(itertools.product(arms,rates,decays))
    if not trials or args.max_fits is None or len(trials)>args.max_fits:
        raise ValueError(f'{len(trials)} fits planned; pass an explicit sufficient --max-fits budget')
    write(out/'search.json',{'search':search,'planned_fits':len(trials),'maximum_fits':args.max_fits,
                           'selection':'minimum source_unit_brier; first trial wins exact ties','target_opened':False})
    selected = {}; results=[]
    for i, (arm,lr,decay) in enumerate(trials):
        directory = out/f'trial_{i:03d}'; directory.mkdir()
        current = fit_once(args,directory,[f'model.fusion={arm}', f'task.lr={lr}',f'task.weight_decay={decay}'])
        current['trial']=i; current['arm']=arm; results.append(current)
        if arm not in selected or current['source_unit_brier'] < selected[arm]['source_unit_brier']:
            selected[arm]=current
        write(out/'completed_trials.json',results)
    write(out/'selected.json', selected)
    return {'completed_fits':len(results),'target_opened':False}


def evaluate(args, out):
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from src.configs.config_utils import dict_to_namespace
    from src.data_factory.p08_data import validate_inventory, encode_conditions, read_records, read_inventory, P08Windows
    from src.model_factory import build_model
    from src.task_factory.task.DG.p08 import prediction_rows, record_metrics
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    contract = checkpoint['p08_contract']
    # All preprocessing and architecture come from the source-selected checkpoint.
    data = dict_to_namespace(contract['data']); data.data_dir = str(args.data_root)
    frame = read_inventory(args.inventory,data.condition_fields)
    frame = validate_inventory(frame,data,evaluation=True,source_groups=contract['source_groups'])
    if set(frame.record_id.astype(str)) & set(contract['source_records']): raise ValueError('Target original record occurs in sources')
    condition_frame = frame
    if args.conditions is not None:
        alternate = read_inventory(args.conditions,data.condition_fields)
        expected = ['record_id']+[s['name'] for s in contract['condition_schema']]
        if set(alternate) != set(expected) or alternate.record_id.duplicated().any():
            raise ValueError('Alternative conditions require only record_id and the same physical fields')
        alternate.record_id=alternate.record_id.astype(str)
        if set(alternate.record_id) != set(frame.record_id.astype(str)): raise ValueError('Alternate population differs')
        condition_frame=alternate.set_index('record_id').loc[frame.record_id.astype(str)].reset_index()
    conditions = encode_conditions(condition_frame,contract['condition_schema'])
    dataset = P08Windows(frame,read_records(frame,data),conditions,data)
    model_args = dict_to_namespace(contract['model'])
    if getattr(model_args,'weights_path',None): raise ValueError('Reinitialization weights must not override the frozen checkpoint')
    model = build_model(model_args,metadata=None)
    model.load_state_dict({k.removeprefix('network.'):v for k,v in checkpoint['state_dict'].items()
                           if k.startswith('network.')},strict=True)
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('Requested GPU unavailable; no CPU fallback')
    if args.device not in {'cpu','cuda'}: raise ValueError('Explicit cpu or cuda device required')
    model.to(args.device).eval(); rows=[]
    write(out/'evaluation.json',{'checkpoint':str(args.checkpoint),'inventory':str(args.inventory),
        'conditions':str(args.conditions) if args.conditions else None,'detach_condition':args.detach,
        'source_contract':contract,'device':args.device,'parameter_counts':model.parameter_counts()})
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=data.batch_size,shuffle=False,num_workers=0):
            probabilities=model(batch['x'].to(args.device),fs=batch['fs'].to(args.device),
                condition=batch['condition'].to(args.device),detach_condition=args.detach).softmax(dim=-1)
            rows.extend(prediction_rows(batch, probabilities))
    records, metrics = record_metrics(rows,model.num_classes,dataset.expected)
    pd.DataFrame(rows).to_csv(out/'window_predictions.csv',index=False)
    records.to_csv(out/'predictions.csv',index=False)
    write(out/'metrics.json',metrics)
    return metrics


def plot_existing(args,out):
    import pandas as pd
    import matplotlib.pyplot as plt
    tables=[]
    for path in args.predictions:
        table=pd.read_csv(path,dtype={'record_id':'string','unit':'string','system':'string'}); required={'record_id','system','unit','label','prediction'}
        if not required<=set(table) or table.record_id.duplicated().any(): raise ValueError('Expected record-level predictions')
        from sklearn.metrics import f1_score
        for system, rows in table.groupby('system'):
            classes=[int(c[1:]) for c in table if c.startswith('p') and c[1:].isdigit()]
            if not classes: raise ValueError('Missing class probability columns')
            tables.append({'run':str(path.parent),'system':system,'macro_f1':f1_score(rows.label,rows.prediction,
                labels=sorted(classes),average='macro',zero_division=0),'records':len(rows),'units':rows.unit.nunique()})
    summary=pd.DataFrame(tables); summary.to_csv(out/'summary.csv',index=False)
    for index, (system, rows) in enumerate(summary.groupby('system')):
        figure, axis=plt.subplots(figsize=(7,4)); axis.bar(range(len(rows)),rows.macro_f1)
        axis.set_xticks(range(len(rows)),[Path(p).name for p in rows.run],rotation=25,ha='right')
        axis.set_ylabel('Record-level macro-F1'); axis.set_title(str(system)); figure.tight_layout()
        figure.savefig(out/f'system_{index}.svg'); plt.close(figure)
    return {'scope':'descriptive scores only; no significance or cross-system independence assumed'}


def compare_existing(args, out):
    import pandas as pd
    from sklearn.metrics import f1_score
    if len(args.predictions) != 2: raise ValueError('compare takes baseline then method record predictions')
    baseline, method = [pd.read_csv(path,dtype={'record_id':'string','unit':'string','system':'string'}) for path in args.predictions]
    for table in (baseline, method):
        if table.record_id.duplicated().any(): raise ValueError('Record predictions are duplicated')
    baseline=baseline.set_index('record_id').sort_index(); method=method.set_index('record_id').sort_index()
    if not baseline.index.equals(method.index) or not baseline[['system','unit','label']].equals(method[['system','unit','label']]):
        raise ValueError('Comparisons require identical records, physical units, systems and labels')
    classes=sorted(int(c[1:]) for c in baseline if c.startswith('p') and c[1:].isdigit())
    if not classes or [f'p{c}' for c in classes] != [c for c in method if c.startswith('p') and c[1:].isdigit()]:
        raise ValueError('Class probability columns differ')
    if not 1 <= args.bootstrap_repeats <= 10000: raise ValueError('Bound the analysis resampling count')
    rng=np.random.default_rng(args.analysis_seed); output=[]
    for system, left in baseline.groupby('system'):
        right=method.loc[left.index]
        y=left.label.to_numpy(); a=left.prediction.to_numpy(); b=right.prediction.to_numpy()
        score=lambda truth,pred: float(f1_score(truth,pred,labels=classes,average='macro',zero_division=0))
        effect=score(y,b)-score(y,a); units=left.unit.unique(); draws=[]
        if len(units)>1:
            positions={u:np.flatnonzero(left.unit.to_numpy()==u) for u in units}
            for _ in range(args.bootstrap_repeats):
                idx=np.concatenate([positions[u] for u in rng.choice(units,size=len(units),replace=True)])
                draws.append(score(y[idx],b[idx])-score(y[idx],a[idx]))
        interval=np.quantile(draws,[.025,.975]).tolist() if draws else [None,None]
        output.append({'system':str(system),'record_macro_f1_difference':effect,'interval_95':interval,
                       'physical_units':len(units),'records':len(y)})
    result={'comparisons':output,'resampling':'paired physical-unit percentile bootstrap, conditional on the fitted checkpoints',
            'exclusions':'not an interval for new systems or retraining; no LOSO-fold independence assumption',
            'analysis_seed':args.analysis_seed,'bootstrap_repeats':args.bootstrap_repeats}
    write(out/'comparison.json',result); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','fit','tune','evaluate','compare','plot'])
    parser.add_argument('--config',type=Path); parser.add_argument('--local-config',type=Path)
    parser.add_argument('--override',action='append',default=[]); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-fits',type=int); parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--inventory',type=Path); parser.add_argument('--data-root',type=Path)
    parser.add_argument('--conditions',type=Path); parser.add_argument('--detach',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--predictions',type=Path,nargs='+')
    parser.add_argument('--bootstrap-repeats',type=int,default=1000)
    parser.add_argument('--analysis-seed',type=int,default=0)
    args=parser.parse_args()
    if args.stage in {'preflight','fit','tune'} and args.config is None: parser.error('--config is required')
    if args.stage=='evaluate' and any(x is None for x in (args.checkpoint,args.inventory,args.data_root)):
        parser.error('evaluate requires checkpoint, inventory, data-root')
    if args.stage in {'plot','compare'} and not args.predictions: parser.error('compare/plot require record predictions')
    args.out.mkdir(parents=True,exist_ok=False)
    write(args.out/'invocation.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        result={'preflight':preflight,'fit':fit_once,'tune':tune,'evaluate':evaluate,'plot':plot_existing,'compare':compare_existing}[args.stage](args,args.out)
        write(args.out/'completion.json',result)
    except Exception:
        (args.out/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__': main()
