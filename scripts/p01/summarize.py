"""Combine actual results. Seeds remain optimization repeats, not physical samples."""
from pathlib import Path
import argparse
import json
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); root=Path(args.root); out=Path(args.output)
    failures=list(root.rglob('failure.txt'))
    if failures:
        raise RuntimeError('incomplete runs exist: '+', '.join(map(str,failures)))
    frames=[]; costs=[]
    for path in sorted(root.rglob('summary.json')):
        summary=json.loads(path.read_text())
        if summary['status']!='completed': raise ValueError(f'incomplete {path}')
        frame=pd.read_csv(path.parent/'metrics.csv')
        for key in ['dataset','arm','mode','seed','run_kind']:
            frame[key]=summary[key]
        frames.append(frame);costs.append(summary)
    if not frames: raise ValueError('no completed runs found')
    out.mkdir(parents=True,exist_ok=False)
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(out/'domain_stage_metrics.csv',index=False)
    pd.DataFrame(costs).to_csv(out/'costs.csv',index=False)
    metrics=['ce','brier','brier_class_balanced','accuracy','macro_f1']
    grouped=data.groupby(['dataset','mode','arm','stage','timing','domain','run_kind'])[metrics].agg(['mean','std','count'])
    grouped.columns=['_'.join(c) for c in grouped.columns]
    grouped.to_csv(out/'seed_summary.csv')
    # Unbiased sample std is blank for one seed. Do not replace it by zero.
    print(out)


if __name__=='__main__': main()
