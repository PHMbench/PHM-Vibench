import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import Dict, Any, Optional

from src.task_factory import register_task
from Vbench.src.task_factory.task_factory import BaseTask

@register_task('DummyClassificationTask')  # 修改注册名称
class DummyClassificationTask(BaseTask):  # 修改类名
    """分类任务类，处理分类相关的训练和评估
    
    用于测试框架的示例分类任务类，支持多类分类
    """
    
    def __init__(self, model=None, loss_fn="cross_entropy", metrics=None, **kwargs):
        """初始化分类任务
        
        Args:
            model: 模型实例，通常由model_factory构建并传入
            loss_fn: 损失函数名称或实例
            metrics: 评估指标列表
            **kwargs: 其他参数
        """
        super().__init__(model=model, **kwargs)
        self.metrics = metrics if metrics is not None else ["accuracy"]
        
        # 设置损失函数
        if loss_fn == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print(f"[WARN] 未知的损失函数 '{loss_fn}'，使用默认的交叉熵损失")
    
    def train(self, 
             train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None,
             epochs: int = 10,
             lr: float = 0.001,
             save_path: Optional[str] = None,
             **kwargs) -> Dict[str, Any]:
        """
        训练模型（简化版，实际应使用LightningModelWrapper）
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            save_path: 模型保存路径
            **kwargs: 其他参数
            
        Returns:
            训练历史记录字典
        """
        print("[WARN] 该方法仅用于演示，实际应使用Lightning训练流程")
        return {"message": "使用Lightning训练流程替代直接调用任务的train方法"}
    
    def evaluate(self, test_loader: DataLoader, **kwargs) -> Dict[str, Any]:
        """
        评估模型（简化版，实际应使用LightningModelWrapper）
        
        Args:
            test_loader: 测试数据加载器
            **kwargs: 其他参数
            
        Returns:
            包含评估指标的字典
        """
        print("[WARN] 该方法仅用于演示，实际应使用Lightning评估流程")
        
        # 返回一些示例指标
        metrics_results = {
            "accuracy": 0.85,
            "f1_score": 0.83,
            "precision": 0.86,
            "recall": 0.81
        }
        
        # 只返回请求的指标
        if self.metrics and self.metrics != "all":
            return {k: v for k, v in metrics_results.items() if k in self.metrics}
        
        return metrics_results
    
    @staticmethod
    def calculate_accuracy(y_pred, y_true):
        """计算准确率
        
        Args:
            y_pred: 预测值
            y_true: 真实值
            
        Returns:
            准确率
        """
        if y_pred.shape != y_true.shape:
            # 如果形状不同，假设是多分类问题
            _, predicted = torch.max(y_pred, 1)
            correct = (predicted == y_true).sum().item()
        else:
            # 二分类问题
            predicted = (torch.sigmoid(y_pred) > 0.5).float()
            correct = (predicted == y_true).sum().item()
        
        total = y_true.size(0)
        return correct / total
        
    def get_loss_fn(self):
        """获取损失函数
        
        Returns:
            损失函数实例
        """
        return self.loss_fn


if __name__ == '__main__':
    """测试入口点"""
    # 导入必要的模块
    from Vbench.src.model_factory.CNN.cnnmodel import DummyCNNModel  # 更新导入
    
    # 创建示例模型
    model = DummyCNNModel(input_channels=1, output_dim=10)  # 更新类名
    
    # 创建任务实例
    task = DummyClassificationTask(  # 更新类名
        model=model,
        loss_fn="cross_entropy",
        metrics=["accuracy", "f1_score"]
    )
    
    # 测试评估方法
    dummy_loader = "dummy_loader"  # 实际使用时会是 DataLoader 实例
    results = task.evaluate(dummy_loader)
    
    print("\n评估结果:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n任务测试成功!")