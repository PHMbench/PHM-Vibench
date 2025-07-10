import os
import pandas as pd
import torch
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.trainer_factory import register_trainer
from src.trainer_factory.base_trainer import BaseTrainer
from src.utils.config_utils import makedir, transfer_namespace
from src.task_factory import build_task

# Lightning 模型包装器类
class LightningModelWrapper(pl.LightningModule):
    """PyTorch Lightning 模型包装器，便于统一训练接口"""
    
    def __init__(self, model: torch.nn.Module, task=None, args_t=None, args_m=None, args_d=None):
        """初始化 Lightning 模型包装器
        
        Args:
            model: 基础模型
            task: 任务实例，用于获取损失函数和评估方法
            args_t: 训练相关参数
            args_m: 模型相关参数
            args_d: 数据集相关参数
        """
        super().__init__()
        self.model = model
        self.task = task
        self.args_t = args_t
        self.args_m = args_m
        self.args_d = args_d
        
        # 保存超参数
        self.save_hyperparameters(ignore=["model", "task"])
        
        # 获取损失函数，优先使用任务实例提供的损失函数
        if task and hasattr(task, 'get_loss_fn'):
            self.loss_fn = task.get_loss_fn()
        else:
            # 从训练参数获取损失函数，默认为 MSE
            loss_fn_name = getattr(self.args_t, "loss_fn", "mse")
            if loss_fn_name == "mse":
                self.loss_fn = torch.nn.MSELoss()
            elif loss_fn_name == "cross_entropy":
                self.loss_fn = torch.nn.CrossEntropyLoss()
            else:
                self.loss_fn = torch.nn.MSELoss()
        
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss}
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_fn(y_hat, y)
        
        # 记录测试指标
        self.log("test_loss", test_loss)
        
        # 如果任务实例定义了评估方法，使用它来计算指标
        if self.task and hasattr(self.task, 'calculate_accuracy'):
            accuracy = self.task.calculate_accuracy(y_hat, y)
            self.log("test_accuracy", accuracy)
        
        # 从训练参数获取指标列表
        metrics = {}
        if hasattr(self.args_t, "metrics") and self.args_t.metrics:
            for metric_name in self.args_t.metrics:
                # 默认情况下使用损失作为指标
                metric_value = test_loss
                
                # 如果任务定义了相应的指标计算方法，则调用它
                if self.task and hasattr(self.task, f'calculate_{metric_name}'):
                    metric_func = getattr(self.task, f'calculate_{metric_name}')
                    metric_value = metric_func(y_hat, y)
                
                metrics[f"test_{metric_name}"] = metric_value
                self.log(f"test_{metric_name}", metric_value)
        
        return {"test_loss": test_loss, **metrics}
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        lr = getattr(self.args_t, "lr", 0.001)
        weight_decay = getattr(self.args_t, "weight_decay", 0)
        
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # 配置学习率调度器
        if hasattr(self.args_t, "scheduler") and self.args_t.scheduler:
            scheduler_type = getattr(self.args_t, "scheduler_type", "step")
            
            if scheduler_type == "step":
                step_size = getattr(self.args_t, "step_size", 10)
                gamma = getattr(self.args_t, "gamma", 0.5)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=step_size, 
                    gamma=gamma
                )
            elif scheduler_type == "cosine":
                t_max = getattr(self.args_t, "t_max", 50)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=t_max
                )
            elif scheduler_type == "plateau":
                patience = getattr(self.args_t, "patience", 5)
                factor = getattr(self.args_t, "factor", 0.5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode="min", 
                    factor=factor, 
                    patience=patience
                )
                # ReduceLROnPlateau 需要特殊配置
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                    }
                }
            else:
                return optimizer
                
            # 其他调度器的返回格式
            return [optimizer], [scheduler]
            
        # 如果没有调度器，只返回优化器
        return optimizer


@register_trainer('DefaultTrainer')
class DefaultTrainer(BaseTrainer):
    """默认训练器，使用 PyTorch Lightning 执行训练和评估"""
    
    def setup_trainer(self, args_t: Any, save_path: str) -> pl.Trainer:
        """设置 PyTorch Lightning Trainer
        
        Args:
            args_t: 训练相关参数
            save_path: 结果保存路径
            
        Returns:
            配置好的 Lightning Trainer
        """
        # 回调函数
        callbacks = []
        
        # 检查点回调
        checkpoint_dir = os.path.join(save_path, 'checkpoints')
        makedir(checkpoint_dir)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
        
        # 提前停止回调
        if getattr(args_t, "early_stopping", False):
            patience = getattr(args_t, "es_patience", 10)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                mode='min'
            )
            callbacks.append(early_stopping)
        
        # WandB 日志记录
        logger = None
        if getattr(args_t, "wandb", False) and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            project = getattr(args_t, "project", "vbench")
            name = getattr(args_t, "name", "experiment")
            tags = getattr(args_t, "tags", [])
            logger = WandbLogger(project=project, name=name, tags=tags)
        
        # 训练器配置
        epochs = getattr(args_t, "epochs", 100)
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            default_root_dir=save_path
        )
        
        return trainer
    
    def train_and_evaluate(self, 
                          model_pl,
                          train_dataloader, 
                          val_dataloader, 
                          test_dataloader, 
                          args_t, 
                          save_path) -> Dict[str, Any]:
        """训练并评估模型
        
        Args:
            model_pl: Lightning 模型实例
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            test_dataloader: 测试数据加载器
            args_t: 训练相关参数
            save_path: 保存路径
            
        Returns:
            评估结果字典
        """
        # 设置训练器
        trainer = self.setup_trainer(args_t, save_path)
        
        # 训练模型
        trainer.fit(model_pl, train_dataloader, val_dataloader)
        
        # 加载最佳模型
        best_model_path = trainer.checkpoint_callback.best_model_path
        if os.path.exists(best_model_path):
            print(f"[INFO] 加载最佳模型: {best_model_path}")
            model_pl = model_pl.load_from_checkpoint(best_model_path)
        
        # 测试模型
        test_results = trainer.test(model_pl, test_dataloader)
        
        # Lightning 的 test() 返回一个列表，我们获取第一个元素
        return test_results[0] if test_results else {}
    
    def __call__(self, configs, save_path, iteration=0) -> Dict[str, Any]:
        """调用训练器执行训练和评估
        
        Args:
            configs: 配置字典
            save_path: 保存路径
            iteration: 当前迭代次数
            
        Returns:
            评估结果
        """
        from src.data_factory import build_dataset
        from src.model_factory import build_model
        from src.task_factory import build_task
        
        # 准备命名空间参数
        args_t = transfer_namespace(configs['trainer'].get('args', {}))
        args_m = transfer_namespace(configs['model'].get('args', {}))
        args_d = transfer_namespace(configs['dataset'].get('args', {}))
        
        # 构建数据集
        print(f"[INFO] 构建数据集: {configs['dataset']['name']}")
        dataset = build_dataset(configs['dataset'])
        
        # 获取数据加载器
        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_val_loader()
        test_loader = dataset.get_test_loader()
        
        # 构建模型
        print(f"[INFO] 构建模型: {configs['model']['name']}")
        model = build_model(configs['model'])
        
        # 构建任务
        print(f"[INFO] 构建任务: {configs['task']['name']}")
        # 将模型实例传递给任务
        task_config = configs['task'].copy()
        task_config['args'] = task_config.get('args', {})
        task_config['args']['model'] = model
        task = build_task(task_config)
        
        # 构建 Lightning 模型
        model_pl = LightningModelWrapper(
            model=model, 
            task=task,
            args_t=args_t, 
            args_m=args_m, 
            args_d=args_d
        )
        print(f"[INFO] 模型结构:\n{model}")
        
        # 训练和评估模型
        print(f"[INFO] 开始训练 (迭代 {iteration+1})")
        results = self.train_and_evaluate(
            model_pl, train_loader, val_loader, test_loader, args_t, save_path
        )
        
        # 保存结果
        result_df = pd.DataFrame([results])
        result_df.to_csv(os.path.join(save_path, f'test_result_{iteration}.csv'), index=False)
        
        print(f"[INFO] 评估结果: {results}")
        return results


if __name__ == '__main__':
    """测试入口点"""
    import argparse
    from src.utils.config_utils import load_config, path_name
    
    # 解析参数
    parser = argparse.ArgumentParser(description='DefaultTrainer 测试')
    parser.add_argument('--config_path', type=str, default='configs/demo/basic.yaml')
    args = parser.parse_args()
    
    # 加载配置
    configs = load_config(args.config_path)
    
    # 设置保存路径
    save_path = 'results/trainer_test'
    os.makedirs(save_path, exist_ok=True)
    
    # 创建训练器
    trainer = DefaultTrainer()
    
    # 执行训练和评估
    results = trainer(configs, save_path)
    
    print("\n训练器测试成功!")