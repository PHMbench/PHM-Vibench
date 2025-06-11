import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# 导入解耦后的组件
from .Components.loss import get_loss_fn
from .Components.metrics import get_metrics
from .Components.regularization import calculate_regularization


class Default_task(pl.LightningModule):
    """
    通用 PyTorch Lightning 任务模块 (已重构)

    Features:
    - 通过组件配置损失函数和评估指标
    - 通过组件配置正则化方法
    - 灵活的优化器和调度器配置
    - 期望 batch 格式为 ((x, y), data_name)
    """

    def __init__(
        self,
        network: nn.Module,
        args_data: Any,  # Data args (Namespace)
        args_model: Any,  # Model args (Namespace)
        args_task: Any,  # Training args (Namespace)
        args_trainer: Any,  # Trainer args (Namespace)
        args_environment: Any,  # Environment args (Namespace)
        metadata: Any # Metadata object/dict
    ):
        """
        初始化训练模块

        :param network: 待训练的主干网络
        :param args_t: 训练参数配置对象 (Namespace)
        :param args_m: 模型参数配置对象 (Namespace)
        :param args_d: 数据参数配置对象 (Namespace)
        :param metadata: 数据元信息
        """
        super().__init__()
        self.network = network.cuda() if args_trainer.gpus else network  # 确保网络在正确的设备上
        self.args_task = args_task
        self.args_model = args_model
        self.args_data = args_data
        self.metadata = metadata # 存储 metadata
        self.args_trainer = args_trainer
        self.args_environment = args_environment

        # 使用组件配置损失和指标
        self.loss_fn = get_loss_fn(self.args_task.loss)
        # 假设 get_metrics 需要数据配置来确定任务类型和类别数
        self.metrics = get_metrics(self.args_task.metrics, self.metadata)

        # 保存超参数 (确保 Namespace 可以转换为字典)
        hparams_dict = {**vars(self.args_task),
                            **vars(self.args_model),
                            **vars(self.args_data),
                            **vars(self.args_trainer),
                            **vars(self.args_environment),
                            # metadata 可能包含复杂对象，选择性保存或忽略
                            # 'metadata': metadata
                            }
        self.save_hyperparameters(hparams_dict, ignore=['network', 'metadata'])


    def forward(self, batch):
        """模型前向传播"""
        x = batch['x']
        id = batch['Id'][0].item() if 'Id' in batch else None

        task_id = batch['Task_id'] if 'Task_id' in batch else None
        return self.network(x, id, task_id)

    def _forward_pass(self, batch) -> torch.Tensor:
        """执行前向传播"""
        return self(batch)

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算任务损失"""
        # 确保 y 是 long 类型用于分类损失        
        return self.loss_fn(y_hat, y.long() if y.dtype != torch.long else y)

    def _compute_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, data_name: str, stage: str) -> Dict[str, torch.Tensor]:
        """计算并更新评估指标"""
        metric_values = {}
        # print(f"计算 {stage} 阶段的指标: {data_name}")
        if data_name in self.metrics:
            for metric_key, metric_fn in self.metrics[data_name].items():
                if metric_key.startswith(stage):
                    # metric_fn 是 torchmetrics 对象，调用会更新内部状态并返回值
                    value = metric_fn(y_hat, y)
                    # 记录当前 step 的值 (注意：torchmetrics 通常在 epoch 结束时计算最终值)
                    # 为了日志记录，我们可能需要记录瞬时值或累计值
                    # 这里记录瞬时值，log_dict 会在 epoch 结束时聚合
                    metric_values[f"{metric_key}_{data_name}"] = value
        else:
            # 仅在第一次遇到未知 data_name 时打印警告，避免刷屏
            if not hasattr(self, '_warned_missing_metrics') or data_name not in self._warned_missing_metrics:
                 print(f"警告: 在 metrics 中未找到数据名称 '{data_name}' 的指标配置。")
                 if not hasattr(self, '_warned_missing_metrics'):
                     self._warned_missing_metrics = set()
                 self._warned_missing_metrics.add(data_name)

        return metric_values

    def _compute_regularization(self) -> Dict[str, torch.Tensor]:
        """计算正则化损失"""
        return calculate_regularization(
            getattr(self.args_task, 'regularization', {}),
            self.parameters() # 只对当前 LightningModule 的参数计算正则化
        )

    def _shared_step(self, batch: Tuple,
                     stage: str,
                     Task_id = False) -> Dict[str, torch.Tensor]:
        """
        通用处理步骤 (已重构)
        期望 batch 格式: ((x, y), data_name)
        """
        try:
            # x, y, id = batch['x'], batch['y'], batch['id']
            batch.update({'Task_id': Task_id})
            id = batch['Id'][0].item()  # 确保 id 是字符串 TODO @liq22 sample 1 id rather than tensor
            data_name = self.metadata[id]['Name']# .values
            # dataset_id = self.metadata[id]['Dataset_id'].item() 
            batch.update({'Data_id': id})
        except (ValueError, TypeError) as e:
            raise ValueError(f" Error: {e}")

        # 1. 前向传播
        y_hat = self._forward_pass(batch)

        # 2. 计算任务损失
        y = batch['y']
        loss = self._compute_loss(y_hat, y)
        y_argmax = torch.argmax(y_hat, dim=1) if y_hat.ndim > 1 else y_hat

        # 3. 计算和记录指标
        step_metrics = {f"{stage}_loss": loss}
        step_metrics[f"{stage}_{data_name}_loss"] = loss # 记录特定数据集的损失
        metric_values = self._compute_metrics(y_argmax, y, data_name, stage)
        step_metrics.update(metric_values)

        # 4. 计算正则化损失
        reg_dict = self._compute_regularization()
        for reg_type, reg_loss_val in reg_dict.items():
            if reg_type != 'total':
                step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_loss_val

        # 5. 计算总损失
        total_loss = loss + reg_dict.get('total', torch.tensor(0.0, device=loss.device))
        step_metrics[f"{stage}_total_loss"] = total_loss

        # 添加 batch size 用于日志记录
        # step_metrics[f"{stage}_batch_size"] = torch.tensor(x.shape[0], dtype=torch.float, device=loss.device)

        return step_metrics

    def training_step(self, batch: dict, *args, **kwargs) -> torch.Tensor:
        """训练步骤"""
        metrics = self._shared_step(batch, "train")
        # 使用 _log_metrics 记录 (确保 batch_size 传递正确)
      
        self._log_metrics(metrics, "train")
        # 返回用于反向传播的总损失
        return metrics["train_total_loss"]

    def validation_step(self, batch: dict, *args, **kwargs) -> None:
        """验证步骤"""
        metrics = self._shared_step(batch, "val")
      
        self._log_metrics(metrics, "val")
        # validation_step 通常不返回损失

    def test_step(self, batch: dict, *args, **kwargs) -> None:
        """测试步骤"""
        metrics = self._shared_step(batch, "test")
        
        self._log_metrics(metrics, "test")
        # test_step 通常不返回损失

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], stage: str) -> None:
        """统一日志记录"""
        log_dict = {}
        prog_bar_metrics = {}
        for k, v in metrics.items():
            # 过滤掉非当前阶段或 batch_size 的指标
            if k.startswith(stage) and "batch_size" not in k:
                log_dict[k] = v
                # 选择要在进度条上显示的指标
                if any(prog_key in k for prog_key in ['loss', 'acc', 'f1']): # 简化进度条显示
                    # 只显示不带数据集名称的总指标或第一个数据集的指标
                    if f"{stage}_loss" == k or f"{stage}_acc_" in k or f"{stage}_f1_" in k:
                         prog_bar_metrics[k.replace(f"_{stage}", "")] = v # 简化显示名称


        self.log_dict(
            log_dict,
            on_step=(stage == "train"), # 训练时可以记录 step 级别的 loss
            on_epoch=True,
            prog_bar=False, # 单独控制进度条
            logger=True,
            sync_dist=True,
        )
        # 单独记录需要在进度条显示的指标 (只在 epoch 结束时显示聚合值)
        # self.log_dict(
        #     prog_bar_metrics,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=False, # 避免重复记录
        #     sync_dist=True,
        # )

    def configure_optimizers(self):
        """配置优化器和学习率调度器 (保持不变或根据需要调整)"""
        optimizer_name = self.args_task.optimizer.lower()
        lr = self.args_task.lr
        weight_decay = getattr(self.args_task, 'weight_decay', 0.0) # 提供默认值

        # 选择优化器
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = getattr(self.args_task, 'momentum', 0.9) # SGD momentum
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        # 配置学习率调度器 (如果指定)
        scheduler_config = getattr(self.args_task, 'scheduler', None)
        if not scheduler_config or not isinstance(scheduler_config, dict) or not scheduler_config.get('name'):
            return optimizer # 只返回优化器

        scheduler_name = scheduler_config['name'].lower()
        scheduler_options = scheduler_config.get('options', {}) # 获取调度器特定参数

        if scheduler_name == 'reduceonplateau':
            # 确保 monitor 指标存在
            monitor_metric = getattr(self.args_task, 'monitor', 'val_total_loss')
            # 可以在这里添加检查，确保 monitor_metric 会被记录
            # if monitor_metric not in self.metrics... (但这比较复杂，因为指标是动态生成的)
            patience = scheduler_options.get('patience', getattr(self.args_task, 'patience', 10) // 2 if hasattr(self.args_task, 'patience') else 5)
            factor = scheduler_options.get('factor', 0.1)
            mode = scheduler_options.get('mode', 'min')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': monitor_metric, # 指定监控的指标
                    'interval': 'epoch', # 通常在 epoch 结束时调整
                    'frequency': 1
                }
            }
        elif scheduler_name == 'cosine':
            # 尝试从 trainer 获取 max_epochs，否则从 args_task 获取
            max_epochs = getattr(self.trainer, 'max_epochs', None) or getattr(self.args_task, 'max_epochs', 100)
            t_max = scheduler_options.get('T_max', max_epochs)
            eta_min = scheduler_options.get('eta_min', 0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        elif scheduler_name == 'step':
            step_size = scheduler_options.get('step_size', 10)
            gamma = scheduler_options.get('gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"不支持的调度器: {scheduler_name}")

        # 对于非 ReduceLROnPlateau 的调度器，返回列表形式
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}]


# __main__ 部分需要更新以匹配新的配置结构和组件用法
if __name__ == '__main__':
    from dataclasses import dataclass, field
    from argparse import Namespace

    # 模拟配置类 (使用 Namespace 模拟)
    train_args = Namespace(
        optimizer='adam',
        lr=1e-3,
        weight_decay=0.0,
        monitor="val_total_loss",
        patience=10,
        cla_loss="CE",
        metrics=["acc", "f1"],
        regularization={'flag': True, 'method': {'l2': 0.001}},
        scheduler={'name': 'reduceonplateau', 'options': {'patience': 5}},
        max_epochs=50
    )

    model_args = Namespace(
        input_dim=128,
        name='DummyModel',
        type='SimpleFC'
    )

    data_args = Namespace(
        task={'mydataset': {'n_classes': 10, 'path': '/path/to/data'}},
        batch_size=32
    )

    # 模拟 Metadata
    metadata = {'info': 'some metadata'}

    # 创建模拟网络
    class DummyModel(nn.Module):
        def __init__(self, args_m, args_d):
            super().__init__()
            first_task_name = list(args_d.task.keys())[0]
            n_classes = args_d.task[first_task_name]['n_classes']
            self.fc = nn.Linear(args_m.input_dim, n_classes)
        def forward(self, x): return self.fc(x)

    # 初始化模型和任务模块
    dummy_network = DummyModel(model_args, data_args)
    task_model = Default_task( # 使用 Default_task 类名
        network=dummy_network,
        args_trainer=Namespace(),  # 模拟训练参数
        args_task=train_args,
        args_model=model_args,
        args_data=data_args,
        args_environment=Namespace(WANDB_MODE=False),  # 模拟环境参数
        metadata=metadata # 传入 metadata
    )

    # 模拟数据 (符合 ((x, y), data_name) 格式)
    batch_with_name = ((torch.randn(16, 128), torch.randint(0, 10, (16,))), 'mydataset')

    print("\n测试训练步骤:")
    # 模拟 trainer 环境以避免 configure_optimizers 中的 trainer 访问错误
    # task_model.trainer = pl.Trainer(max_epochs=train_args.max_epochs)
    train_loss_tensor = task_model.training_step(batch_with_name, 0)
    print(f"训练总损失 (Tensor): {train_loss_tensor}")

    print("\n测试验证步骤:")
    task_model.validation_step(batch_with_name)

    print("\n测试前向传播:")
    output = task_model(torch.randn(5, 128))
    print("输出形状:", output.shape)

    print("\n测试指标配置:")
    print(task_model.metrics)

    print("\n测试 Metadata:")
    print(task_model.metadata)


