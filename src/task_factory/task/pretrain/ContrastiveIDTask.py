"""
长信号对比学习预训练任务
基于BaseIDTask扩展，利用多窗口机制构建对比学习
保持简洁实用，避免过度复杂
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import logging
import numpy as np

from ...ID_task import BaseIDTask
from ... import register_task

logger = logging.getLogger(__name__)


@register_task("contrastive_id", "pretrain")
class ContrastiveIDTask(BaseIDTask):
    """
    长信号对比学习任务
    继承BaseIDTask的所有功能，专注于对比学习逻辑
    """
    
    def __init__(self, network, args_data, args_model, args_task, 
                 args_trainer, args_environment, metadata):
        """初始化对比学习任务"""
        super().__init__(
            network, args_data, args_model, args_task,
            args_trainer, args_environment, metadata
        )
        
        # 对比学习参数
        self.temperature = getattr(args_task, 'temperature', 0.07)
        
        logger.info(f"ContrastiveIDTask initialized with temperature={self.temperature}")

    def prepare_batch(self, batch_data: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """
        为对比学习准备批次数据
        每个ID生成2个窗口作为正样本对
        """
        anchors, positives, ids = [], [], []
        
        for sample_id, data_array, metadata in batch_data:
            try:
                # 1. 数据预处理
                processed_data = self.process_sample(data_array, metadata)
                
                # 2. 生成2个随机窗口
                windows = self.create_windows(
                    processed_data,
                    strategy='random',
                    num_window=2
                )
                
                if len(windows) < 2:
                    logger.warning(f"Sample {sample_id} has insufficient windows: {len(windows)}")
                    continue
                    
                # 3. 选择正样本对
                anchor_window = windows[0]
                positive_window = windows[1]
                
                # 4. 转换为张量
                anchors.append(torch.tensor(anchor_window, dtype=torch.float32))
                positives.append(torch.tensor(positive_window, dtype=torch.float32))
                ids.append(sample_id)
                
            except Exception as e:
                logger.error(f"Failed to process sample {sample_id}: {e}")
                continue
        
        # 5. 检查批次有效性
        if len(anchors) == 0:
            logger.warning("Empty batch after processing")
            return self._empty_batch()
            
        return {
            'anchor': torch.stack(anchors),
            'positive': torch.stack(positives),
            'ids': ids
        }
    
    def _empty_batch(self) -> Dict[str, torch.Tensor]:
        """返回空批次"""
        return {
            'anchor': torch.empty(0, self.args_data.window_size, 1),
            'positive': torch.empty(0, self.args_data.window_size, 1),
            'ids': []
        }
    
    def _shared_step(self, batch: Dict[str, Any], stage: str, task_id: bool = False) -> Dict[str, torch.Tensor]:
        """对比学习训练步骤"""
        # 1. 预处理原始批次（如果需要）
        if 'anchor' not in batch:
            batch = self._preprocess_raw_batch(batch)
            
        if len(batch['ids']) == 0:
            return {'loss': torch.tensor(0.0, requires_grad=True)}
        
        # 2. 前向传播
        z_anchor = self.network(batch['anchor'])      # [B, D]
        z_positive = self.network(batch['positive'])   # [B, D]
        
        # 3. 计算InfoNCE损失
        loss = self.infonce_loss(z_anchor, z_positive)
        
        # 4. 计算准确率
        with torch.no_grad():
            accuracy = self.compute_accuracy(z_anchor, z_positive)
        
        # 5. 日志记录
        self.log(f'{stage}_contrastive_loss', loss, prog_bar=True)
        self.log(f'{stage}_contrastive_acc', accuracy, prog_bar=True)
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def infonce_loss(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """InfoNCE对比损失函数"""
        # L2归一化
        z_anchor = F.normalize(z_anchor, dim=1)
        z_positive = F.normalize(z_positive, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(z_anchor, z_positive.t()) / self.temperature
        
        # 正样本在对角线上
        positive_samples = torch.diag(similarity_matrix)
        
        # InfoNCE损失
        logsumexp = torch.logsumexp(similarity_matrix, dim=1)
        loss = -positive_samples + logsumexp
        
        return loss.mean()
    
    def compute_accuracy(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """计算对比学习准确率"""
        with torch.no_grad():
            # L2归一化
            z_anchor = F.normalize(z_anchor, dim=1)
            z_positive = F.normalize(z_positive, dim=1)
            
            # 计算相似度矩阵
            similarity_matrix = torch.mm(z_anchor, z_positive.t())
            
            # 找到每行最大值的索引
            _, predicted = torch.max(similarity_matrix, dim=1)
            
            # 正确的匹配在对角线上
            correct = torch.arange(similarity_matrix.shape[0], device=predicted.device)
            
            # 计算准确率
            accuracy = (predicted == correct).float().mean()
            
        return accuracy