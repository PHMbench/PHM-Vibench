import torch
from torch import nn
import pytorch_lightning as pl  # Import PyTorch Lightning
from ...Default_task import Default_task  # Corrected import path
from typing import Dict, List, Optional, Any, Tuple
from ...Components.loss import get_loss_fn
class task(Default_task):
    """Standard classification task, often used for pretraining a backbone."""
    def __init__(self, network, args_data, args_model, args_task,
                  args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task,
                          args_trainer, args_environment, metadata)
        
        self.metric_loss = get_loss_fn(self.args_task.metric_loss)(args_task)

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算任务损失"""
        # 确保 y 是 long 类型用于分类损失        
        return self.loss_fn(y_hat, y.long() if y.dtype != torch.long else y)

    def _compute_metric_loss(self, feature: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.metric_loss(feature, y)

    def forward(self, batch):
        """模型前向传播"""
        x = batch['x']
        file_id = batch['file_id']

        task_id = batch['task_id'] if 'task_id' in batch else None
        return self.network(x, file_id, task_id)
    def forward_feature(self, batch):
        """模型前向传播，返回特征"""
        x = batch['x']
        file_id = batch['file_id']
        task_id = batch['task_id'] if 'task_id' in batch else None
        return self.network(x, file_id, task_id, return_feature=True)
    
    def _shared_step(self, batch: Tuple,
                     stage: str,
                     task_id = False):
        """
        通用处理步骤 (已重构)
        期望 batch 格式: ((x, y), data_name)
        """
        try:
            # x, y, id = batch['x'], batch['y'], batch['id']
            batch.update({'task_id': 'classification'})  # 确保 task_id 是分类任务
            file_id = batch['file_id'][0].item()  # 确保 id 是字符串 TODO @liq22 sample 1 id rather than tensor
            data_name = self.metadata[file_id]['Name']# .values
            # dataset_id = self.metadata[id]['Dataset_id'].item() 
            batch.update({'file_id': file_id})
        except (ValueError, TypeError) as e:
            raise ValueError(f" Error: {e}")

        # 1. 前向传播
        y_hat = self.forward(batch)

        # 2. 计算任务损失
        y = batch['y']
        loss = self._compute_loss(y_hat, y)
        y_argmax = torch.argmax(y_hat, dim=1) if y_hat.ndim > 1 else y_hat
        # 2.5 计算预测损失
        step_metrics = {f"{stage}_loss": loss}
        step_metrics[f"{stage}_{data_name}_loss"] = loss # 记录特定数据集的损失
        # 2.5 计算度量损失
        if stage == 'train':
            feature = self.forward_feature(batch)
            _metric_loss, metric_acc = self._compute_metric_loss(feature, y)
            step_metrics[f"{stage}_{data_name}_metric_loss"] = _metric_loss # 记录特定数据集的度量损失
        metric_values = self._compute_metrics(y_argmax, y, data_name, stage)
        step_metrics.update(metric_values)

        # 4. 计算正则化损失
        if stage == 'train':
            reg_dict = self._compute_regularization()
            for reg_type, reg_loss_val in reg_dict.items():
                if reg_type != 'total':
                    step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_loss_val
            total_loss = loss + reg_dict.get('total', torch.tensor(0.0, device=loss.device))
        # 5. 计算总损失
        else:
            total_loss = loss   

        step_metrics[f"{stage}_total_loss"] = total_loss

        # 添加 batch size 用于日志记录
        # step_metrics[f"{stage}_batch_size"] = torch.tensor(x.shape[0], dtype=torch.float, device=loss.device)

        return step_metrics


# 