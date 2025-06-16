import torch
from torch import nn
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
from ...Default_task import Default_task
from ...Components.prediction_loss import Signal_mask_Loss
from ....model_factory.ISFM.task_head.H_02_distance_cla import H_02_distance_cla


class task(Default_task):
    """Standard classification task, often used for pretraining a backbone."""
    def __init__(self,
                 network: nn.Module,
                 args_data: Any,
                 args_model: Any,
                 args_task: Any,
                 args_trainer: Any,
                 args_environment: Any,
                 metadata: Any):
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)
        # 初始化 prediction head 和 prediction loss
        num_classes = getattr(self.args_task, 'n_classes', {})
        self.pred_head = H_02_distance_cla(self.args_model, num_classes)
        pred_cfg = getattr(self.args_task, 'pred_cfg', self.args_task)
        self.pred_loss_fn = Signal_mask_Loss(pred_cfg)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch['x']
        system_id = batch.get('Id', [None])[0].item() if 'Id' in batch else None
        task_id = batch.get('Task_id', None)
        return self.network(x, system_id, task_id)

    def forward_feature(self, batch: dict) -> torch.Tensor:
        x = batch['x']
        system_id = batch.get('Id', [None])[0].item() if 'Id' in batch else None
        task_id = batch.get('Task_id', None)
        # 返回特征
        return self.network(x, system_id, task_id, return_feature=True)

    # 计算 prediction loss
    def _compute_prediction_loss(self, signal: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 调用 Signal_mask_Loss: 返回 (loss, stats_dict)
        return self.pred_loss_fn(self.pred_head, signal)

    def _shared_step(self, batch: dict, stage: str, Task_id: Any = False) -> Dict[str, torch.Tensor]:
        # 配置 Task_id 和 Data_id
        batch['Task_id'] = Task_id
        idx = batch.get('Id', [0])[0].item()
        data_name = self.metadata[idx]['Name']
        batch['Data_id'] = idx

        # 1. forward
        y_hat = self.forward(batch)
        feature = self.forward_feature(batch)
        y = batch['y']

        # 2. 计算 CE 损失
        ce_loss = self._compute_loss(y_hat, y)

        # 3. 计算 Prediction 损失
        signal = batch.get('signal', None)
        if signal is not None:
            pred_loss, pred_stats = self._compute_prediction_loss(signal)
        else:
            pred_loss = torch.tensor(0.0, device=ce_loss.device)
            pred_stats = {}

        # 4. 计算 metric loss
        metric_loss = self._compute_metric_loss(feature, y)

        # 5. 记录各项指标
        step_metrics: Dict[str, torch.Tensor] = {
            f"{stage}_ce_loss": ce_loss,
            f"{stage}_pred_loss": pred_loss,
            f"{stage}_metric_loss": metric_loss
        }
        # 记录 prediction stats
        for k, v in pred_stats.items():
            step_metrics[f"{stage}_{k}"] = torch.tensor(v, device=ce_loss.device)

        # 6. 记录分类指标
        y_pred_label = torch.argmax(y_hat, dim=1)
        step_metrics.update(self._compute_metrics(y_pred_label, y, data_name, stage))

        # 7. 正则化
        reg_dict = self._compute_regularization()
        for reg_type, reg_val in reg_dict.items():
            if reg_type != 'total':
                step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_val

        # 8. 总损失：CE + pred + reg_total
        total = ce_loss + pred_loss + reg_dict.get('total', torch.tensor(0.0, device=ce_loss.device))
        step_metrics[f"{stage}_total_loss"] = total

        return step_metrics

    def training_step(self, batch: dict, *args, **kwargs) -> torch.Tensor:
        metrics = self._shared_step(batch, 'train')
        self._log_metrics(metrics, 'train')
        return metrics['train_total_loss']

    def validation_step(self, batch: dict, *args, **kwargs) -> None:
        metrics = self._shared_step(batch, 'val')
        self._log_metrics(metrics, 'val')

    def test_step(self, batch: dict, *args, **kwargs) -> None:
        metrics = self._shared_step(batch, 'test')
        self._log_metrics(metrics, 'test')
