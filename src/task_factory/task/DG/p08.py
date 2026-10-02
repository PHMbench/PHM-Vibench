"""P08 source-only objective and original-record evaluation in the Task owner."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score
from src.task_factory import register_task


def record_metrics(window_rows, classes, expected_counts):
    frame = pd.DataFrame(window_rows)
    probability = [f'p{j}' for j in range(classes)]
    needed = {'record_id','unit','system','label','window_start',*probability}
    if frame.empty or not needed <= set(frame): raise ValueError('Incomplete window predictions')
    if frame[['record_id','window_start']].duplicated().any(): raise ValueError('Duplicate evaluated window')
    if set(frame.record_id) != set(expected_counts): raise ValueError('Predicted population differs from declared records')
    p = frame[probability].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1),1.,atol=1e-6):
        raise ValueError('Invalid probability vector')
    for rid, values in frame.groupby('record_id'):
        if len(values) != expected_counts[rid] or any(values[key].nunique() != 1 for key in ('unit','system','label')):
            raise ValueError('Incomplete record or inconsistent record identity/label')
    records = frame.groupby(['record_id','unit','system','label'], sort=False)[probability].mean().reset_index()
    y = records.label.to_numpy()
    if not np.isin(y, np.arange(classes)).all(): raise ValueError('Prediction label outside frozen ontology')
    pred = records[probability].to_numpy().argmax(axis=1)
    brier = .5*np.sum((records[probability].to_numpy()-np.eye(classes)[y.astype(int)])**2,axis=1)
    records['brier'] = brier; records['prediction'] = pred
    # Each unit, then each system, contributes equally to model selection.
    units = records.groupby(['system','unit']).brier.mean()
    selection = float(units.groupby(level='system').mean().mean())
    per_system = []
    for system, rows in records.groupby('system'):
        per_system.append({'system':system, 'record_macro_f1':float(f1_score(rows.label, rows.prediction,
            labels=list(range(classes)), average='macro',zero_division=0)),
            'record_balanced_accuracy':float(balanced_accuracy_score(rows.label,rows.prediction)),
            'record_count':len(rows),'unit_count':rows.unit.nunique()})
    return records, {'unit_balanced_brier':selection,'systems':per_system}


def prediction_rows(batch, probabilities):
    p = probabilities.detach().cpu().numpy()
    return [{'record_id':batch['record_id'][i], 'unit':batch['unit'][i], 'system':batch['system'][i],
             'label':int(batch['y'][i]), 'window_start':int(batch['window_start'][i]),
             **{f'p{j}':float(value) for j,value in enumerate(p[i])}} for i in range(len(p))]


@register_task('DG','p08')
class task(pl.LightningModule):
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__()
        if args_trainer.test_after_fit or args_trainer.monitor != 'source_unit_brier' or args_trainer.monitor_mode != 'min':
            raise ValueError('P08 requires source-only fitting and source_unit_brier/min checkpoint selection')
        if args_trainer.devices != 1 or args_trainer.num_sanity_val_steps != 0:
            raise ValueError('P08 uses one device and a complete validation population (no truncated sanity pass)')
        if args_task.loss != 'CE' or set(args_task.metrics) != {'record_macro_f1','record_balanced_accuracy','unit_balanced_brier'}:
            raise ValueError('P08 objective is CE; metrics must specify the original-record estimators')
        self.network = network
        self.contract = deepcopy(metadata.p08_contract)
        self.contract['model'] = vars(args_model).copy()
        self.contract['data'] = vars(args_data).copy()
        self.contract['task'] = vars(args_task).copy()
        self.contract['seed'] = args_environment.seed
        self.lr = args_task.lr; self.weight_decay = args_task.weight_decay
        if not np.isfinite(self.lr) or self.lr <= 0 or not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid optimizer settings')
        from src.data_factory.p08_data import encode_conditions
        width = encode_conditions(metadata.df.iloc[:1], self.contract['condition_schema']).shape[1]
        if width != network.condition_dim or len(self.contract['label_names']) != network.num_classes:
            raise ValueError('Model output/condition width disagrees with the frozen source schema')
        self.validation_rows = []; self.last_source_brier = None

    def forward(self, batch):
        return self.network(batch['x'], fs=batch['fs'], condition=batch['condition'])

    def training_step(self, batch, batch_idx):
        loss = F.cross_entropy(self(batch), batch['y'])
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite source training loss')
        self.log('train_ce',loss,on_step=False,on_epoch=True,batch_size=len(batch['y']))
        return loss

    def on_validation_epoch_start(self): self.validation_rows = []

    def validation_step(self, batch, batch_idx):
        self.validation_rows.extend(prediction_rows(batch, self(batch).softmax(dim=-1)))

    def on_validation_epoch_end(self):
        dataset = self.trainer.val_dataloaders.dataset
        _, metrics = record_metrics(self.validation_rows, self.network.num_classes, dataset.expected)
        self.last_source_brier = metrics['unit_balanced_brier']
        self.log('source_unit_brier',self.last_source_brier,prog_bar=True,on_epoch=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['p08_contract'] = deepcopy(self.contract)
        checkpoint['source_unit_brier'] = self.last_source_brier

    def configure_optimizers(self):
        return torch.optim.AdamW(self.network.parameters(),lr=self.lr,weight_decay=self.weight_decay)
