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

        # 初始化对比学习损失函数
        from ...Components.contrastive_loss import InfoNCELoss
        self.contrastive_loss_fn = InfoNCELoss(
            temperature=self.temperature,
            reduction='mean',
            symmetric=False  # 使用非对称版本以匹配原始实现
        )

        # 初始化对比学习准确率指标
        from ...Components.metrics import ContrastiveAccuracy
        self.contrastive_acc_fn = ContrastiveAccuracy()

        # 验证元数据完整性
        self._validate_metadata(metadata)

        # 初始化对比学习统计信息
        self.contrastive_stats = {
            'total_positive_pairs': 0,
            'successful_windows': 0,
            'failed_windows': 0,
            'average_similarity': 0.0
        }

        logger.info(f"ContrastiveIDTask initialized with temperature={self.temperature}")
        logger.info(f"Metadata contains {len(metadata) if hasattr(metadata, '__len__') else 'unknown'} entries")

    def _validate_metadata(self, metadata):
        """验证元数据必需字段"""
        try:
            if metadata is None:
                logger.warning("Metadata is None")
                return

            # 检查元数据是否为数据访问器
            if hasattr(metadata, '__getitem__') and hasattr(metadata, 'keys'):
                # 检查关键字段（如果可用）
                required_fields = ['ID', 'Sample_length', 'Channel']
                available_fields = []

                # 尝试获取第一个样本的字段
                try:
                    if hasattr(metadata, 'keys'):
                        sample_keys = list(metadata.keys())[:1]
                        if sample_keys:
                            first_sample = metadata[sample_keys[0]]
                            if hasattr(first_sample, 'keys'):
                                available_fields = list(first_sample.keys())
                            elif isinstance(first_sample, dict):
                                available_fields = list(first_sample.keys())
                except:
                    pass

                missing_fields = [field for field in required_fields if field not in available_fields]
                if missing_fields and available_fields:
                    logger.warning(f"Missing metadata fields: {missing_fields}")
                    logger.info(f"Available fields: {available_fields[:10]}...")  # 显示前10个字段
                elif available_fields:
                    logger.info(f"Metadata validation passed. Available fields: {len(available_fields)}")

            else:
                logger.info(f"Metadata type: {type(metadata)}, attempting direct validation")

        except Exception as e:
            logger.warning(f"Metadata validation failed: {e}")
            # 不阻止任务继续，只记录警告

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

    def _preprocess_raw_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        预处理标准批次格式为对比学习格式

        处理三种可能的输入格式：
        1. 标准格式: ((x, y), data_name) 来自Default_task
        2. ID格式: {'x': tensor, 'file_id': list} 来自id_data_factory
        3. 已处理格式: {'anchor': tensor, 'positive': tensor} 来自prepare_batch
        """
        try:
            # 情况1: 标准格式 ((x, y), data_name)
            if isinstance(batch, tuple) and len(batch) == 2:
                (x, y), data_name = batch
                file_ids = data_name if isinstance(data_name, list) else [data_name]
                return self._convert_standard_batch(x, file_ids)

            # 情况2: ID格式字典
            elif isinstance(batch, dict):
                if 'x' in batch and 'file_id' in batch:
                    x = batch['x']
                    file_ids = batch['file_id']
                    return self._convert_standard_batch(x, file_ids)
                elif 'anchor' in batch and 'positive' in batch:
                    # 已经是对比学习格式
                    return batch
                else:
                    logger.warning(f"Unknown batch format with keys: {batch.keys()}")
                    return self._empty_batch()

            else:
                logger.warning(f"Unsupported batch format: {type(batch)}")
                return self._empty_batch()

        except Exception as e:
            logger.error(f"Error in _preprocess_raw_batch: {e}")
            return self._empty_batch()

    def _convert_standard_batch(self, x: torch.Tensor, file_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        将标准批次转换为对比学习批次

        Args:
            x: [B, seq_len, channels] 输入张量
            file_ids: [B] 文件ID列表

        Returns:
            对比学习批次格式
        """
        anchors, positives, valid_ids = [], [], []

        for i, file_id in enumerate(file_ids):
            try:
                signal = x[i].cpu().numpy()  # [seq_len, channels]

                # 使用窗口创建方法生成正样本对
                windows = self.create_windows(
                    signal,
                    strategy=getattr(self.args_data, 'window_sampling_strategy', 'random'),
                    num_window=2
                )

                if len(windows) >= 2:
                    anchor = torch.tensor(windows[0], dtype=torch.float32, device=x.device)
                    positive = torch.tensor(windows[1], dtype=torch.float32, device=x.device)

                    anchors.append(anchor)
                    positives.append(positive)
                    valid_ids.append(file_id)
                else:
                    logger.warning(f"Insufficient windows for sample {file_id}: {len(windows)}")

            except Exception as e:
                logger.error(f"Failed to process sample {file_id}: {e}")
                continue

        if len(anchors) == 0:
            return self._empty_batch()

        return {
            'anchor': torch.stack(anchors),
            'positive': torch.stack(positives),
            'ids': valid_ids
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
        """InfoNCE对比损失函数 - 使用组件化损失函数"""
        return self.contrastive_loss_fn(z_anchor, z_positive)
    
    def compute_accuracy(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """计算对比学习准确率 - 使用组件化指标"""
        # 更新指标状态
        self.contrastive_acc_fn.update(z_anchor, z_positive)
        # 计算当前准确率
        return self.contrastive_acc_fn.compute()