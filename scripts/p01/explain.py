"""Frozen-model path deletion and matched contribution replacement.

These are controlled prediction-dependence tests, not physical interventions.
Target labels define evaluation pairs only and never update the model.
"""
import argparse
import json
from pathlib import Path
import random
import torch
import torch.nn.functional as F
import yaml
from run_experiments import build_model, read_records, materialize, pack_batch, dump_csv


def score(contributions,label):
    logits=contributions.sum(1)
    p=logits.softmax(-1)
    y=torch.full((len(p),),label,dtype=torch.long,device=p.device)
    return {'brier':float((p-F.one_hot(y,p.shape[-1])).square().sum(-1).mean()),
            'prediction':int(p.mean(0).argmax())}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--config',required=True);a.add_argument('--checkpoint',required=True)
    a.add_argument('--dataset',required=True);a.add_argument('--output',required=True)
    a.add_argument('--pair-seed',type=int,default=412)
    args=a.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    dataset=next(d for d in cfg['datasets'] if d['name']==args.dataset)
    records=[r for r in read_records(dataset,cfg) if r['split']=='test']
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=build_model(checkpoint['model'],cfg['engine']).eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    traces=[]
    with torch.no_grad():
        for u in materialize(records,dataset,cfg):
            x,y,ids,meta=pack_batch([u],torch.device('cpu'))
            result=model.forward_details(x,physical_metadata=meta)
            if 'contributions' not in result:
                raise ValueError('path tests require an additive model; no post-hoc MLP decomposition')
            traces.append(dict(u,contributions=result['contributions'],names=result['path_names']))
    rows=[]; rng=random.Random(args.pair_seed); excluded=0
    for anchor in traces:
        original=score(anchor['contributions'],anchor['label'])
        for k,name in enumerate(anchor['names']):
            if k==0: continue
            deleted=anchor['contributions'].clone();deleted[:,k]=0
            changed=score(deleted,anchor['label'])
            rows.append({'anchor':anchor['unit_id'],'domain':anchor['domain'],'label':anchor['label'],
                         'condition':'delete','path':name,'donor':'','donor_domain':'',
                         'brier_change':changed['brier']-original['brier'],
                         'prediction_flip':int(changed['prediction']!=original['prediction'])})
        candidates=[]
        for domain in sorted({r['domain'] for r in traces}-{anchor['domain']}):
            eligible=[r for r in traces if r['domain']==domain and r['unit_id']!=anchor['unit_id']]
            same=[r for r in eligible if r['label']==anchor['label']]
            different=[r for r in eligible if r['label']!=anchor['label']]
            if same and different: candidates.append((domain,same,different))
        if not candidates:
            excluded+=1; continue
        domain,same,different=rng.choice(candidates)
        # Identical anchor and donor-domain choice in both replacement conditions.
        for condition,donor in [('same_class',rng.choice(same)),('different_class',rng.choice(different))]:
            if donor['names']!=anchor['names'] or donor['contributions'].shape!=anchor['contributions'].shape:
                raise ValueError('replacement needs the same path order and windows-per-unit')
            for k,name in enumerate(anchor['names']):
                if name in {'bias','raw'}: continue
                changed=anchor['contributions'].clone();changed[:,k]=donor['contributions'][:,k]
                result=score(changed,anchor['label'])
                rows.append({'anchor':anchor['unit_id'],'domain':anchor['domain'],'label':anchor['label'],
                             'condition':condition,'path':name,'donor':donor['unit_id'],'donor_domain':domain,
                             'brier_change':result['brier']-original['brier'],
                             'prediction_flip':int(result['prediction']!=original['prediction'])})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    dump_csv(out/'path_tests.csv',rows)
    (out/'pairing.json').write_text(json.dumps({'pair_seed':args.pair_seed,'anchors':len(traces),
        'replacement_anchors':len(traces)-excluded,'excluded_no_matched_pair':excluded,
        'meaning':'path-contribution replacement, including its metadata dependence; not raw-waveform causality'},indent=2))
    print(out)


if __name__=='__main__': main()
