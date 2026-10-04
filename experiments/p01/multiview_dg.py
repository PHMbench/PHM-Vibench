"""Source-only DG orchestration over the existing P01 model, trainers and evaluator.

Task assignment is administrative: it contains physical IDs and domains, not a
learned split. Only the final `test` command reads held-out labels or waveforms.
This module neither defines another network nor reimplements the risk estimator.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.p01.fusion_data import read_records, summarize_rows
from experiments.p01.fusion_deployment import (
    FrozenClassifier, load_model, predict_records, save_bundle, verify_vectors,
    acquisition_rows, write_csv,
)
from experiments.p01.window_io import window_record, _h5_metadata_frame, _vibench_id
from experiments.p01 import analyze_d1
from src.model_factory.model_factory import model_factory

ROOT = Path(__file__).resolve().parents[2]
CORE = ('I', 'I-F', 'MLP16', 'I-single', 'Dense', 'I-base')
CONTRASTS = {f'I-{arm}': ('I', arm) for arm in ('p0', 'I-F', 'MLP16', 'I-single', 'Dense', 'I-base')}
SUPPORTED = {('CNN','ResNet1D'), ('CNN','TCN'), ('Transformer','PatchTST'),
             ('X_model','BASE_ExplainableCNN')}


def read(path):
    path=Path(path)
    text=path.read_text(encoding='utf-8')
    return json.loads(text) if path.suffix=='.json' else yaml.safe_load(text)


def dump(path, obj):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def resolve(value, base):
    value = Path(os.path.expandvars(str(value))).expanduser()
    return value.resolve() if value.is_absolute() else (base/value).resolve()


def validate_study(study):
    if study['schema_version'] != 1:
        raise ValueError('Unsupported DG study schema.')
    seeds = study['seeds']
    if study['min_datasets']<2 or study['min_splits_per_dataset']<2:
        raise ValueError('The industrial suite needs multiple datasets and multiple splits per dataset.')
    if len(seeds)<2 or any(type(x) is not int for x in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('At least two unique explicit final seeds are required.')
    if study['hpo_seed'] in seeds:
        raise ValueError('HPO seed and final reporting seeds must be distinct.')
    if study['reference_min_accuracy'] != .8:
        raise ValueError('The current study retains its prespecified .8 reference criterion.')
    if min(study[k] for k in ('epochs','steps_per_epoch','units_per_domain','overfit_steps')) < 1:
        raise ValueError('Positive training and smoke budgets are required.')
    if len(study['trials']) < 2:
        raise ValueError('Freeze more than one source-only HPO trial.')
    for trial in study['trials']:
        if set(trial)!={'lr','weight_decay','scheduler'} or not math.isfinite(trial['lr']) or trial['lr']<=0:
            raise ValueError('Each trial declares lr, weight_decay and scheduler.')
        if trial['scheduler'] not in {'none','cosine'} or not math.isfinite(trial['weight_decay']) or trial['weight_decay']<0:
            raise ValueError('Invalid optimizer trial.')
    if set(study['baselines']) & set((*CORE, 'p0')):
        raise ValueError('Baseline aliases may not shadow a core model identity.')
    if not study['baselines'] or any((m['type'],m['name']) not in SUPPORTED for m in study['baselines'].values()):
        raise ValueError('Use only inspected classification models at the recorded runtime revision.')
    names = [b['name'] for b in study['fusion']['model']['branches']]
    if len(names)<2 or len(names)!=len(set(names)) or len(study['trials'])<len(names):
        raise ValueError('Require multiple unique views and enough total single-view trials to cover them.')
    loss=study['fusion']['loss']
    if (loss['tau']!=1 or loss['brier_weight']!=.25 or loss['lambda_delta']!=0 or loss['reduction']!='mean_source'):
        raise ValueError('The primary DG model retains ordinary paired CE+.25 Brier, no new alignment or routing loss.')
    if study['fusion']['model']['head_type']!='operator_residual':
        raise ValueError('Do not replace the current proposed method.')


def _metadata_rows(dataset, selected_ids):
    """Read labels only for explicitly allowed acquisition rows (CSV or XLSX).

    The first pass requests the Id column only; skipped rows are not returned to
    fitting or selection. Original target labels are never copied to source.csv.
    """
    path=Path(dataset['metadata_file']); idcol=dataset['columns']['id']
    reader=pd.read_excel if path.suffix.lower()=='.xlsx' else pd.read_csv
    ids=reader(path,usecols=[idcol],dtype=str,keep_default_na=False)[idcol].map(_vibench_id)
