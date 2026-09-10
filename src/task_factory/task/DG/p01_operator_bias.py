"""Native PHMFactory DG task for P01 source-paired contribution consistency.

The ordinary pipeline handles data, optimizer, selected checkpoint and metrics.
Continual unit replay lives in the explicit paper runner, not in validation/test.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from ...Default_task import Default_task
from ...Components.p01_bias_losses import covariance_loss, unit_mean


class task(Default_task):
    def forward(self, batch):
        metadata={key:batch[key] for key in ['sample_rate_hz','rotation_speed_rpm'] if key in batch}
        self._p01_output=self.network.forward_details(
            batch['x'],batch['file_id'],self._resolve_model_task_id(batch),
            physical_metadata=metadata or None)
        return self._p01_output['logits']

    def _shared_step(self,batch,stage,task_id=False):
        metrics=super()._shared_step(batch,stage,task_id)
        weight=float(getattr(self.args_task,'p01_cov_weight',0.0))
        paired_supervision=bool(getattr(self.args_task,'p01_paired_supervision',False))
        if stage!='train' or not (weight or paired_supervision):
            return metrics
        if self.loss_name!='CE':
            raise ValueError('P01 paired supervision requires task.loss=CE')
        if 'unit_id' not in batch:
            raise ValueError('P01 consistency needs per-window unit_id from a unit-balanced sampler')
        ids=batch['unit_id']
        if not isinstance(ids,torch.Tensor):
            # Equality grouping only: no unit identity is inferred from labels.
            distinct=list(dict.fromkeys(ids))
            ids=torch.tensor([distinct.index(u) for u in ids],device=batch['x'].device)
        original=self._p01_output
        shift=int(getattr(self.args_task,'p01_shift_samples',0))
        if shift<=0:
            raise ValueError('declare a positive p01_shift_samples for circular time-origin pairing')
        x=batch['x']; meta={key:batch[key] for key in ['sample_rate_hz','rotation_speed_rpm'] if key in batch}
        offset=int(torch.randint(1,shift+1,(),device=x.device))
        paired=self.network.forward_details(x.roll(offset,1),batch['file_id'],
            self._resolve_model_task_id(batch),physical_metadata=meta or None)
        # Retain existing regularization while replacing CE by paired mean CE.
        ce=unit_mean(F.cross_entropy(original['logits'],batch['y'],reduction='none'),ids)
        paired_ce=unit_mean(F.cross_entropy(paired['logits'],batch['y'],reduction='none'),ids)
        total=metrics['train_total_loss']-metrics['train_loss']+(ce+paired_ce)/2
        if weight:
            cov=covariance_loss(original['contributions'],paired['contributions'],ids)
            total=total+weight*cov
            metrics['train_p01_cov_loss']=cov
        metrics['train_total_loss']=total
        self._p01_output=None
        return metrics
