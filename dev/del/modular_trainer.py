"""
模块化训练器实现，提供高度可定制的训练流程
"""
import os
import pandas as pd
import torch
import pytorch_lightning as pl
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.trainer_factory import register_trainer
from src.trainer_factory.base_trainer import BaseTrainer
from src.utils.config_utils import makedir
from src.utils.loss_utils import get_loss_function
from src.utils.metrics_utils import compute_metrics
from src.utils.model_selection import ModelValidator


class ModularTrainer(BaseTrainer):
    """模块化训练器基类，高度可定制的训练流程
    
    特点：
    1. 支持多种损失函数和评估指标
    2. 支持K折交叉验证和模型选择
    3. 支持早停和检查点保存
    4. 支持日志记录和结果可视化
    5. 功能组件化，易于扩展
    """
    
    def __init__(self, **kwargs):
        """初始化训练器
        
        Args:
            **kwargs: 训练相关参数，可以包括：
                - task_type: 任务类型，可选'classification', 'anomaly', 'rul'
                - loss_fn: 损失函数名称或损失函数对象
                - metrics: 要计算的指标列表
                - epochs: 训练轮数
                - batch_size: 批量大小
                - optimizer: 优化器名称或配置
                - lr: 学习率
                - weight_decay: 权重衰减
                - scheduler: 是否使用学习率调度器
                - scheduler_type: 调度器类型
                - early_stopping: 是否使用早停
                - es_patience: 早停耐心值
                - cv_folds: K折交叉验证折数，0表示不使用交叉验证
                - save_checkpoints: 是否保存检查点
                - use_wandb: 是否使用WandB记录实验
                - project: WandB项目名称
                - experiment_name: 实验名称
                - random_state: 随机种子
        """
        super().__init__(**kwargs)
        
        # 训练相关配置
        self.task_type = kwargs.get('task_type', 'classification')
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)
        self.optimizer_config = kwargs.get('optimizer', {'name': 'adam', 'lr': 0.001})
        self.lr = kwargs.get('lr', 0.001)
        self.weight_decay = kwargs.get('weight_decay', 0)
        
        # 学习率调度器配置
        self.use_scheduler = kwargs.get('scheduler', False)
        self.scheduler_type = kwargs.get('scheduler_type', 'step')
        self.scheduler_config = kwargs.get('scheduler_config', {})
        
        # 早停配置
        self.early_stopping = kwargs.get('early_stopping', True)
        self.es_patience = kwargs.get('es_patience', 10)
        self.es_monitor = kwargs.get('es_monitor', 'val_loss')
        
        # 交叉验证配置
        self.cv_folds = kwargs.get('cv_folds', 0)
        self.stratify = kwargs.get('stratify', True)
        
        # 检查点保存配置
        self.save_checkpoints = kwargs.get('save_checkpoints', True)
        self.save_best_only = kwargs.get('save_best_only', True)
        
        # 实验记录配置
        self.use_wandb = kwargs.get('use_wandb', False)
        self.project = kwargs.get('project', 'vbench')
        self.experiment_name = kwargs.get('experiment_name', f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # 随机种子
        self.random_state = kwargs.get('random_state', 42)
        pl.seed_everything(self.random_state)
        
        # 损失函数和评估指标
        self.loss_fn_name = kwargs.get('loss_fn', None)
        self.loss_fn_config = kwargs.get('loss_fn_config', {})
        self.metrics = kwargs.get('metrics', None)
        
        # 解析损失函数
        self._loss_fn = None
        if self.loss_fn_name is not None:
            if isinstance(self.loss_fn_name, str):
                self._loss_fn = get_loss_function(self.loss_fn_name, **self.loss_fn_config)
            else:
                self._loss_fn = self.loss_fn_name
    
    def get_loss_fn(self):
        """获取损失函数
        
        Returns:
            损失函数实例
        """
        return self._loss_fn
        
    def _configure_callbacks(self, save_path):
        """配置回调函数
        
        Args:
            save_path: 保存路径
            
        Returns:
            回调函数列表
        """
        callbacks = []
        
        # 添加检查点回调
        if self.save_checkpoints:
            checkpoint_dir = os.path.join(save_path, 'checkpoints')
            makedir(checkpoint_dir)
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='best-{epoch:02d}-{val_loss:.4f}',
                monitor=self.es_monitor,
                mode='min' if 'loss' in self.es_monitor else 'max',
                save_top_k=1 if self.save_best_only else 3,
                save_last=True,
            )
            callbacks.append(checkpoint_callback)
        
        # 添加早停回调
        if self.early_stopping:
            early_stopping = EarlyStopping(
                monitor=self.es_monitor,
                patience=self.es_patience,
                mode='min' if 'loss' in self.es_monitor else 'max'
            )
            callbacks.append(early_stopping)
        
        return callbacks
    
    def _configure_trainer(self, save_path):
        """配置Lightning训练器
        
        Args:
            save_path: 保存路径
            
        Returns:
            配置好的训练器
        """
        # 配置回调
        callbacks = self._configure_callbacks(save_path)
        
        # 配置日志记录
        logger = None
        if self.use_wandb:
            logger = WandbLogger(
                project=self.project,
                name=self.experiment_name,
                save_dir=save_path
            )
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator='auto',
            devices=1 if torch.cuda.is_available() else None,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            default_root_dir=save_path
        )
        
        return trainer
    
    def _create_lightning_module(self, model, task=None):
        """创建Lightning模块
        
        Args:
            model: 模型实例
            task: 任务实例
            
        Returns:
            Lightning模块
        """
        from src.trainer_factory.lightning_module import TaskLightningModule
        
        return TaskLightningModule(
            model=model,
            task=task,
            loss_fn=self.get_loss_fn(),
            task_type=self.task_type,
            metrics=self.metrics,
            optimizer_config={
                'name': self.optimizer_config.get('name', 'adam'),
                'lr': self.lr,
                'weight_decay': self.weight_decay
            },
            scheduler_config={
                'use_scheduler': self.use_scheduler,
                'scheduler_type': self.scheduler_type,
                **self.scheduler_config
            }
        )
    
    def train_and_evaluate(
        self,
        model,
        task,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        save_path
    ):
        """训练并评估模型
        
        Args:
            model: 模型实例
            task: 任务实例
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            test_dataloader: 测试数据加载器
            save_path: 保存路径
            
        Returns:
            测试结果字典
        """
        # 创建Lightning模块
        lightning_module = self._create_lightning_module(model, task)
        
        # 配置训练器
        trainer = self._configure_trainer(save_path)
        
        # 训练模型
        trainer.fit(lightning_module, train_dataloader, val_dataloader)
        
        # 加载最佳模型
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback is not None:
            if hasattr(trainer.checkpoint_callback, 'best_model_path'):
                best_model_path = trainer.checkpoint_callback.best_model_path
                if os.path.exists(best_model_path):
                    print(f"[INFO] 加载最佳模型: {best_model_path}")
                    lightning_module = type(lightning_module).load_from_checkpoint(best_model_path)
        
        # 测试模型
        test_results = trainer.test(lightning_module, test_dataloader)
        
        # 保存预测结果
        test_predictions = trainer.predict(lightning_module, test_dataloader)
        if test_predictions:
            # 收集预测结果和真实标签
            y_pred = torch.cat([p[0] for p in test_predictions]).cpu().numpy()
            y_true = torch.cat([p[1] for p in test_predictions]).cpu().numpy()
            y_probs = torch.cat([p[2] for p in test_predictions]).cpu().numpy() if len(test_predictions[0]) > 2 else None
            
            # 计算详细指标
            if y_probs is not None:
                detailed_metrics = compute_metrics(
                    self.task_type, y_true, y_pred, 
                    y_prob=y_probs if self.task_type == 'classification' else None,
                    anomaly_score=y_probs if self.task_type == 'anomaly' else None
                )
            else:
                detailed_metrics = compute_metrics(self.task_type, y_true, y_pred)
            
            # 更新测试结果
            test_results[0].update(detailed_metrics)
            
            # 保存预测结果
            predictions_df = pd.DataFrame({
                'y_true': y_true.flatten(),
                'y_pred': y_pred.flatten()
            })
            if y_probs is not None and y_probs.ndim <= 2:
                if y_probs.ndim == 1 or y_probs.shape[1] == 1:
                    predictions_df['y_prob'] = y_probs.flatten()
                elif y_probs.shape[1] > 1:
                    # 多类别概率
                    for i in range(y_probs.shape[1]):
                        predictions_df[f'y_prob_class_{i}'] = y_probs[:, i]
            
            predictions_df.to_csv(os.path.join(save_path, 'test_predictions.csv'), index=False)
        
        return test_results[0] if test_results else {}
    
    def __call__(self, 
                dataset=None,
                model=None, 
                task=None,
                train_loader=None,
                val_loader=None,
                test_loader=None,
                configs=None, 
                args_t=None,
                args_m=None,
                args_d=None,
                args_task=None,
                save_path=None, 
                iteration=0):
        """执行训练和评估流程
        
        Args:
            dataset: 数据集实例
            model: 模型实例
            task: 任务实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            configs: 完整配置字典
            args_t: 训练器参数命名空间
            args_m: 模型参数命名空间
            args_d: 数据集参数命名空间
            args_task: 任务参数命名空间
            save_path: 保存路径
            iteration: 当前迭代次数
            
        Returns:
            评估结果字典
        """
        # 设置实验名称
        if model and dataset:
            model_name = model.__class__.__name__
            dataset_name = dataset.__class__.__name__
            self.experiment_name = f"{model_name}_{dataset_name}_{iteration}"
        
        # 如果使用交叉验证
        if self.cv_folds > 0:
            print(f"[INFO] 使用 {self.cv_folds} 折交叉验证")
            
            # 创建验证器
            validator = ModelValidator(
                trainer_factory=lambda _: self,
                n_splits=self.cv_folds,
                stratify=self.stratify,
                random_state=self.random_state,
                save_path=os.path.join(save_path, 'cv')
            )
            
            # 执行交叉验证
            cv_results = validator.cross_validate(configs, dataset, self.task_type)
            
            # 返回汇总结果
            return cv_results['summary']
        
        # 否则进行常规训练和评估
        else:
            print(f"[INFO] 开始常规训练和评估")
            results = self.train_and_evaluate(
                model, task, train_loader, val_loader, test_loader, save_path
            )
            
            print(f"[INFO] 评估结果: {results}")
            return results


@register_trainer('ModularTrainer')
class DefaultModularTrainer(ModularTrainer):
    """默认模块化训练器，提供适用于各种任务的预配置
    """
    
    def __init__(self, **kwargs):
        """初始化默认训练器
        
        根据任务类型设置默认损失函数和评估指标
        """
        # 获取任务类型
        task_type = kwargs.get('task_type', 'classification')
        
        # 设置默认损失函数和评估指标
        if task_type == 'classification':
            default_loss = 'cross_entropy'
            default_metrics = ['accuracy', 'precision', 'recall', 'f1']
        elif task_type == 'anomaly':
            default_loss = 'asymmetric'
            default_metrics = ['precision', 'recall', 'f1', 'auc_roc']
        elif task_type == 'rul':
            default_loss = 'rul'
            default_metrics = ['mae', 'mse', 'rmse', 'r2']
        else:
            default_loss = 'mse'
            default_metrics = ['mae', 'mse']
        
        # 使用默认值，除非在kwargs中指定
        if 'loss_fn' not in kwargs:
            kwargs['loss_fn'] = default_loss
        if 'metrics' not in kwargs:
            kwargs['metrics'] = default_metrics
        
        # 调用父类的初始化方法
        super().__init__(**kwargs)