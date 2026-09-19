"""Source-balanced shared-encoder training with source-local labels."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from src.task_factory import register_task
from src.utils.identifiers import validate_identifiers


@register_task('DG', 'tii_joint')
class task(pl.LightningModule):
    def __init__(self, network, args_data, args_model, args_task, args_trainer,
                 args_environment, metadata):
        super().__init__()
        if args_task.loss != 'CE' or args_task.optimizer.lower() != 'adamw':
            raise ValueError('TII primary requires CE and AdamW')
        if args_task.lambda_common != 0 or args_task.lambda_private != 0:
            raise ValueError('TII primary requires lambda_common=lambda_private=0')
        if getattr(args_task, 'scheduler', None):
            raise ValueError('TII primary has no scheduler')
        if list(args_task.metrics) != ['acc']:
            raise ValueError('tii_joint currently reports NLL and acc; task.metrics must be [acc]')
        if hasattr(args_data, 'rounds'):
            if args_trainer.num_epochs != 1 or args_trainer.early_stopping or args_trainer.devices != 1:
                raise ValueError('tii_joint requires one fixed-round epoch, no early stopping and one device')
            if args_trainer.monitor != 'val_group_nll' or args_trainer.monitor_mode != 'min':
                raise ValueError('checkpoint selection must minimize source val_group_nll')
            if args_trainer.checkpoint_min_delta != 1e-8 or args_trainer.num_sanity_val_steps != 0:
                raise ValueError('TII requires 1e-8 checkpoint ties and complete validation passes')
            if args_data.rounds % args_trainer.val_check_interval:
                raise ValueError('round budget must finish on a source-validation boundary')
            if args_task.lr != .001 or args_task.weight_decay != .0001:
                raise ValueError('primary optimizer requires lr=.001 and weight_decay=.0001')
        self.sources = tuple(validate_identifiers(args_task.source_system_ids, 'dataset'))
        if len(set(self.sources)) != len(self.sources):
            raise ValueError('duplicate source_system_ids')
        if set(map(str, self.sources)) != set(network.task_head.mutiple_fc):
            raise ValueError('model must contain exactly source heads, no target heads')
        self.network = network
        self.args_task = args_task
        self.metadata = metadata
        self.batch_size_per_source = args_data.source_batch_size
        self.save_hyperparameters({
            name: vars(value) for name, value in (
                ('data', args_data), ('model', args_model), ('task', args_task),
                ('trainer', args_trainer), ('environment', args_environment))})
        self.validation_groups = {}

    def forward(self, batch):
        return self.network(batch['x'], file_id=batch['file_id'], task_id='classification',
                            incremental=batch['incremental'], availability=batch['availability'])

    def _validate_source_batch(self, source, batch):
        if source not in self.sources:
            raise ValueError('unknown source')
        n = len(batch['y'])
        if n < 1 or any(len(batch[field]) != n for field in
                        ('x', 'incremental', 'availability', 'group', 'recording_id', 'file_id', 'role')):
            raise ValueError('one identity, role and input per source window required')
        validate_identifiers(batch['group'], 'group')
        validate_identifiers(batch['recording_id'], 'recording_id')
        for fid in validate_identifiers(batch['file_id'], 'file_id'):
            if self.metadata[fid]['Dataset_id'] != source:
                raise ValueError('source batch uses another dataset head')

    def source_loss(self, source, batch):
        self._validate_source_batch(source, batch)
        return F.cross_entropy(self(batch), batch['y'])

    def joint_loss(self, batch):
        if set(batch) != set(self.sources):
            raise ValueError('each joint round must include every declared source exactly once')
        losses = []
        for source in self.sources:
            item = batch[source]
            if len(item['y']) != self.batch_size_per_source:
                raise ValueError('wrong windows/source/round')
            if any(role != 'source_train' for role in item['role']):
                raise ValueError('training may only consume source_train windows')
            losses.append(self.source_loss(source, item))
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        loss = self.joint_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False,
                 batch_size=self.batch_size_per_source*len(self.sources))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.network.parameters(), lr=self.args_task.lr,
                                 betas=(.9, .999), eps=1e-8,
                                 weight_decay=self.args_task.weight_decay)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        # Zero tensors would allow AdamW momentum/decay to move an unused head.
        optimizer.zero_grad(set_to_none=True)

    def on_validation_epoch_start(self):
        self.validation_groups = {}

    def validation_step(self, batch, batch_idx):
        source = batch['source']
        self._validate_source_batch(source, batch)
        if source not in self.sources or any(r != 'source_val' for r in batch['role']):
            raise ValueError('checkpoint selection requires source_val only')
        groups = validate_identifiers(batch['group'], 'group')
        logits = self(batch)
        loss = F.cross_entropy(logits.double(), batch['y'], reduction='none').detach().cpu()
        correct = (logits.argmax(-1) == batch['y']).detach().double().cpu()
        for group, nll, acc in zip(groups, loss.tolist(), correct.tolist()):
            values = self.validation_groups.setdefault((source, group), [0., 0., 0])
            values[0] += nll
            values[1] += acc
            values[2] += 1

    def validation_risk(self):
        observed = {source for source, _ in self.validation_groups}
        if observed != set(self.sources):
            raise ValueError('validation must include every source')
        result = []
        for source in self.sources:
            groups = [v for (d, _), v in self.validation_groups.items() if d == source]
            result.append([sum(v[i]/v[2] for v in groups)/len(groups) for i in (0, 1)])
        return tuple(sum(row[i] for row in result)/len(result) for i in (0, 1))

    def on_validation_epoch_end(self):
        nll, acc = self.validation_risk()
        # The frozen 1e-8 tie tolerance is smaller than float32 spacing near 1.
        self.log('val_group_nll', torch.tensor(nll, dtype=torch.float64, device=self.device),
                 on_epoch=True, batch_size=1)
        self.log('val_group_acc', acc, on_epoch=True, batch_size=1)

    def test_step(self, batch, batch_idx):
        raise RuntimeError('tii_joint trains sources only; frozen target query evaluation is separate')
