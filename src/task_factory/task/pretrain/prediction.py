import torch
from torch import nn
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
from ...Default_task import Default_task
from ...Components.prediction_loss import Signal_mask_Loss


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
        super().__init__(network,
                         args_data,
                         args_model,
                         args_task,
                         args_trainer,
                         args_environment,
                         metadata)
        # 初始化 prediction head 和 prediction loss

        pred_cfg = getattr(self.args_task, 'pred_cfg', self.args_task)
        self.pred_loss_fn = Signal_mask_Loss(pred_cfg)



    # 计算 prediction loss
    def _compute_prediction_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 调用 Signal_mask_Loss: 返回 (loss, stats_dict)
        return self.pred_loss_fn(self.network, batch) # 

    def _shared_step(self, batch: Tuple,
                     stage: str,
                     task_id = False) -> Dict[str, torch.Tensor]:
        """
        通用处理步骤 (已重构)
        期望 batch 格式: ((x, y), data_name)
        """
        try:
            file_id = batch['file_id'][0].item()  # 确保 id 是字符串 TODO @liq22 sample 1 id rather than tensor
            data_name = self.metadata[file_id]['Name']# .values
            # dataset_id = self.metadata[file_id]['Dataset_id'].item()
            batch.update({'file_id': file_id})
        except (ValueError, TypeError) as e:
            raise ValueError(f" Error: {e}")

        # # 1. forward
        # batch.update({'task_id': 'classification'})
        # y_hat = self.forward(batch)

        # # 2. 计算任务损失
        # y = batch['y']
        # loss = self._compute_loss(y_hat, y)
        # y_argmax = torch.argmax(y_hat, dim=1) if y_hat.ndim > 1 else y_hat
        # 计算 prediction loss
        pred_loss = self._compute_prediction_loss(batch)
        # 3. 计算和记录指标
        # step_metrics = {f"{stage}_loss": loss}
        step_metrics    = {}
        step_metrics[f"{stage}_{data_name}_pred_loss"] = pred_loss

        # step_metrics[f"{stage}_{data_name}_loss"] = loss # 记录特定数据集的损失

        # metric_values = self._compute_metrics(y_argmax, y, data_name, stage)

        step_metrics.update(metric_values)

        # 4. 计算正则化损失
        reg_dict = self._compute_regularization()
        for reg_type, reg_loss_val in reg_dict.items():
            if reg_type != 'total':
                step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_loss_val

        # 5. 计算总损失
        total_loss = self.args_task.alpha_prediction * pred_loss + \
                reg_dict.get('total', torch.tensor(0.0, device=pred_loss.device))

        step_metrics[f"{stage}_total_loss"] = total_loss

        # 添加 batch size 用于日志记录
        # step_metrics[f"{stage}_batch_size"] = torch.tensor(x.shape[0], dtype=torch.float, device=loss.device)

        return step_metrics

    # def training_step(self, batch: dict, *args, **kwargs) -> torch.Tensor:
    #     metrics = self._shared_step(batch, 'train')
    #     self._log_metrics(metrics, 'train')
    #     return metrics['train_total_loss']

    # def validation_step(self, batch: dict, *args, **kwargs) -> None:
    #     metrics = self._shared_step(batch, 'val')
    #     self._log_metrics(metrics, 'val')

    # def test_step(self, batch: dict, *args, **kwargs) -> None:
    #     metrics = self._shared_step(batch, 'test')
    #     self._log_metrics(metrics, 'test')
