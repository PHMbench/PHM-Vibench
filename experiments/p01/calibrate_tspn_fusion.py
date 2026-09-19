"""Independent raw-reference risk assessment; labels determine the deployment decision.

Inputs are frozen probabilities and physical IDs, not another signal dataset.
Neither the complete development history nor response assumptions follow from arrays.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Assessment:
    alpha: float
    independent_units: int
    candidate_count: int
    delta: float
    b_hat: float
    a_hat: float
    b_lower: float
    a_upper: float
    upper_excess_risk: float
    target_scope: str
    epsilon_raw: float
    epsilon_candidate: float


def _probability_check(p: Tensor) -> None:
    if p.ndim != 2 or p.shape[-1] < 2 or not torch.isfinite(p).all():
        raise ValueError("Expected finite N,C probability arrays, C >= 2.")
    if not torch.all((p >= 0) & (p <= 1)) or not torch.allclose(p.sum(-1), torch.ones_like(p[:, 0]), atol=1e-6, rtol=1e-6):
        raise ValueError("Input must be probabilities, not logits or silently normalized scores.")


def calibrate(raw: Tensor, candidate: Tensor, labels: Tensor, unit_ids: Tensor, *,
              target_scope: str, delta: float = .05, candidate_count: int = 1,
              epsilon_raw: float | None = None,
              epsilon_candidate: float | None = None) -> Assessment:
    """Retained legacy single-population NR05 API; TF06 uses assess_plan."""
    raw, candidate = raw.detach().cpu().double(), candidate.detach().cpu().double()
    labels, unit_ids = labels.detach().cpu(), unit_ids.detach().cpu()
    _probability_check(raw); _probability_check(candidate)
    nrows = len(raw)
    if raw.shape != candidate.shape or labels.shape != (nrows,) or unit_ids.shape != (nrows,) or nrows == 0:
        raise ValueError("Probabilities, labels and units must refer to identical nonempty rows.")
    if labels.dtype != torch.long or unit_ids.dtype != torch.long or not torch.all((labels >= 0) & (labels < raw.shape[1])):
        raise ValueError("Labels and unit IDs must be explicit integer arrays with valid classes.")
    if not 0 < delta < 1 or candidate_count < 1 or int(candidate_count) != candidate_count:
        raise ValueError("Invalid confidence level or prespecified candidate count.")
    if target_scope == "same_distribution":
        if epsilon_raw is not None or epsilon_candidate is not None:
            raise ValueError("Use bounded_shift when specifying movement bounds.")
        e0, eq = 0., 0.
    elif target_scope == "bounded_shift":
        if epsilon_raw is None or epsilon_candidate is None:
            raise ValueError("DG requires both independently justified final-predictor movement bounds.")
        e0, eq = float(epsilon_raw), float(epsilon_candidate)
        if not all(math.isfinite(e) and 0 <= e <= math.sqrt(2) for e in (e0, eq)):
            raise ValueError("Movement bounds must lie in [0,sqrt(2)] for probability vectors.")
    else:
        raise ValueError("Declare target_scope as same_distribution or bounded_shift.")
    _, inverse, counts = torch.unique(unit_ids, sorted=True, return_inverse=True, return_counts=True)
    n = len(counts)
    def average_by_unit(x):
        return float((x.new_zeros(n).scatter_add_(0, inverse, x) / counts).mean())
    v = candidate-raw
    onehot = torch.nn.functional.one_hot(labels, raw.shape[1]).double()
    b = average_by_unit(((onehot-raw)*v).sum(-1))
    a = average_by_unit(v.square().sum(-1))
    t = math.sqrt(math.log(2*candidate_count/delta)/(2*n))
    lower = b - 2.5*t - math.sqrt(2)*(eq + 2*e0)
    upper = min(2., a + 2*t + 2*math.sqrt(2)*(eq+e0))
    alpha = max(0., min(1., lower/upper))
    risk_upper = upper*alpha**2-2*lower*alpha
    return Assessment(alpha,n,int(candidate_count),delta,b,a,lower,upper,risk_upper,target_scope,e0,eq)


def assess_plan(plan_path, output):
    """Assess one frozen bank, using literal group IDs and complete raw vectors."""
    import csv
    import yaml
    from experiments.p01.fusion_assessment import (grouped_moments, summarize, assess, same_reference,
                                   literal_ids, temperature_candidate)
    plan_path=Path(plan_path).resolve()
    plan=yaml.safe_load(plan_path.read_text(encoding='utf-8'))
    def local(path):
        p=Path(path).expanduser()
        return p if p.is_absolute() else plan_path.parent/p
    banks=plan['candidates'];names=[r['name'] for r in banks]
    if not banks or len(set(names))!=len(names): raise ValueError('Declare a unique fixed candidate bank.')
    rules=plan['rule_budgets']
    if not rules or set(rules)-{'moments','paired'}: raise ValueError('Unknown rule family.')
    total=float(plan['delta_total']);shift=float(plan['delta_shift'])
    if not 0<total<1 or shift<0 or not all(0<float(v)<1 for v in rules.values()):
        raise ValueError('Invalid error probability allocation.')
    if sum(map(float,rules.values()))+shift>total+1e-12:
        raise ValueError('Rule and response failures exceed the total budget.')
    scope=plan['scope'];response=None
    if scope=='source_mixture':
        if shift!=0 or plan.get('response') is not None:
            raise ValueError('Identity source-mixture coupling takes no supplied shift constants.')
    elif scope=='bounded_shift':
        kind=plan['response_kind']
        if kind not in {'deterministic','statistical'} or (kind=='deterministic' and shift!=0) or (kind=='statistical' and shift<=0):
            raise ValueError('Distinguish deterministic response bounds from a statistical response event.')
        response={d:(r['epsilon_raw'],np.asarray([r['zeta'][name] for name in names]))
                  for d,r in plan['response'].items()}
    else: raise ValueError('Declare source_mixture or bounded_shift; no arbitrary OOD claim.')
    used=set()
    for path in plan['development_group_files']:
        with local(path).open(newline='',encoding='utf-8') as f:
            rows=csv.DictReader(f)
            if 'group_id' not in (rows.fieldnames or []): raise ValueError('Development lists need literal group_id.')
            for row in rows:
                if not row['group_id']: raise ValueError('Empty development identity.')
                used.add(row['group_id'])
    if not used: raise ValueError('Provide actual raw/candidate fitting and selection group lists.')
    common=None;candidates=[];alphas=[]
    for item in banks:
        with np.load(local(item['predictions']),allow_pickle=False) as archive:
            data={key:archive[key] for key in archive.files}
        raw=data['raw_probs'];n=len(raw)
        ids=[literal_ids(data[key],n,key) for key in ('domains','group_ids','acquisition_ids','window_ids')]
        keys=list(zip(*ids))
        if len(set(keys))!=n: raise ValueError('Repeated acquisition/window row; not another independent sample.')
        order=np.array(sorted(range(n),key=lambda i:keys[i]))
        keys=[keys[i] for i in order]
        raw_classes=data['raw_class_names'];candidate_classes=data['candidate_class_names']
        data={key:data[key][order] for key in ('raw_probs','candidate_probs','labels')}
        domain,group,acquisition,window=[v[order] for v in ids]
        if raw_classes.shape!=(raw.shape[1],) or not np.array_equal(raw_classes,candidate_classes):
            raise ValueError('Candidate and raw class names/order differ.')
        if len(set(raw_classes.tolist()))!=len(raw_classes): raise ValueError('Repeated class name.')
        if used.intersection(group.tolist()): raise ValueError('Assessment group used to fit/select a raw or candidate model.')
        if common is None:
            common=(keys,data['raw_probs'],data['labels'],raw_classes,group,acquisition,domain)
        else:
            if common[0]!=keys or not np.array_equal(common[2],data['labels']):
                raise ValueError('Candidate files describe different assessment observations or labels.')
            same_reference(common[1],common[3],data['raw_probs'],raw_classes)
        if item.get('kind','model')=='temperature':
            candidate=temperature_candidate(data['raw_probs'],float(item['temperature']))
        elif item.get('kind','model')=='model': candidate=data['candidate_probs']
        else: raise ValueError('Unknown fixed candidate kind.')
        candidates.append(candidate)
        if 'paired' in rules: alphas.append(float(item['fixed_alpha']))
    grouped=grouped_moments(common[1],np.stack(candidates),common[2],common[4],common[5],common[6])
    if set(map(str,plan['conditions']))!=set(grouped): raise ValueError('Assessment domains differ from the predeclared conditions.')
    summaries={d:summarize(*values) for d,values in grouped.items()}
    results={rule:assess(summaries,method=plan['bound'],rule=rule,failure=float(budget),
                        fixed_alpha=alphas if rule=='paired' else None,response=response)
             for rule,budget in rules.items()}
    result=dict(scope=scope,delta_total=total,delta_shift=shift,candidate_names=names,rules=results,
                aggregation='equal windows/acquisition; equal acquisitions/group; equal groups/condition',
                declared_label_budget=plan.get('label_budget'),
                status='conditional_frozen_probability_assessment_not_a_model_deployment',
                assumptions='Complete development histories, fixed candidates/coefficients, iid groups per condition and valid response event remain required.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--plan',help='Predeclared candidate bank and error budgets; grouped multi-source assessment.')
    inputs.add_argument('--predictions',help='Legacy single-population NR05 probability NPZ.')
    parser.add_argument('--scope',choices=['same_distribution','bounded_shift'])
    parser.add_argument('--epsilon-raw',type=float);parser.add_argument('--epsilon-candidate',type=float)
    parser.add_argument('--delta',type=float);parser.add_argument('--candidate-count',type=int)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.plan:
        if args.scope or args.epsilon_raw is not None or args.epsilon_candidate is not None or args.delta is not None or args.candidate_count is not None:
            raise ValueError('The plan owns scope/bounds; do not combine it with legacy flags.')
        assess_plan(args.plan,args.output);return
    if not args.scope: parser.error('Legacy input requires --scope; TF06 requires --plan.')
    with np.load(args.predictions,allow_pickle=False) as data:
        p0,q,y=[torch.from_numpy(data[key]) for key in ('raw_probs','candidate_probs','labels')]
        ids=data['unit_ids']
        if ids.dtype.kind not in 'USiu': raise ValueError('Preserve literal or integer group IDs, not float-coerced identities.')
        _,inverse=np.unique(ids,return_inverse=True)
    result=calibrate(p0,q,y,torch.as_tensor(inverse,dtype=torch.long),target_scope=args.scope,
                     delta=.05 if args.delta is None else args.delta,
                     candidate_count=1 if args.candidate_count is None else args.candidate_count,
                     epsilon_raw=args.epsilon_raw,epsilon_candidate=args.epsilon_candidate)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle: json.dump(asdict(result),handle,indent=2)
    print(json.dumps(asdict(result),indent=2))


if __name__=='__main__': main()
