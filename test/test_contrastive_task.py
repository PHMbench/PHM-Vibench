#!/usr/bin/env python3
"""
ContrastiveIDTask单元测试套件
测试对比学习预训练任务的核心功能
"""

import pytest
import torch
import numpy as np
import warnings
from unittest.mock import Mock, patch
from argparse import Namespace
from typing import Dict, List, Tuple, Any

# 添加项目路径
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


class TestContrastiveIDTask:
    """ContrastiveIDTask单元测试类"""
    
    @pytest.fixture
    def mock_config(self):
        """创建模拟配置"""
        args_data = Namespace(
            factory_name='id',
            batch_size=4,
            window_size=128,
            stride=64,
            num_window=2,
            window_sampling_strategy='random',
            normalization=True,
            truncate_length=1024
        )
        
        args_task = Namespace(
            type='pretrain',
            name='contrastive_id',
            lr=1e-3,
            weight_decay=1e-4,
            temperature=0.07,
            loss='CE',
            metrics=['acc']
        )
        
        args_model = Namespace(
            name='M_01_ISFM',
            backbone='B_08_PatchTST',
            d_model=64
        )
        
        args_trainer = Namespace(
            epochs=10,
            accelerator='cpu',
            devices=1
        )
        
        args_environment = Namespace(
            save_dir='./test_results',
            experiment_name='test'
        )
        
        metadata = {
            1: {'Label': 0, 'Name': 'test_data'},
            2: {'Label': 1, 'Name': 'test_data'},
            3: {'Label': 2, 'Name': 'test_data'}
        }
        
        return {
            'args_data': args_data,
            'args_task': args_task,
            'args_model': args_model,
            'args_trainer': args_trainer,
            'args_environment': args_environment,
            'metadata': metadata
        }
    
    @pytest.fixture
    def mock_network(self):
        """创建模拟网络"""
        return torch.nn.Linear(128, 64)
    
    @pytest.fixture
    def contrastive_task(self, mock_config, mock_network):
        """创建ContrastiveIDTask实例"""
        with patch.object(ContrastiveIDTask, '__init__', lambda x, *args, **kwargs: None):
            task = ContrastiveIDTask.__new__(ContrastiveIDTask)
            
            # 手动初始化必要属性
            task.network = mock_network
            task.args_data = mock_config['args_data']
            task.args_task = mock_config['args_task']
            task.args_model = mock_config['args_model']
            task.args_trainer = mock_config['args_trainer']
            task.args_environment = mock_config['args_environment']
            task.metadata = mock_config['metadata']
            task.temperature = 0.07
            
            # 模拟BaseIDTask的方法
            task.process_sample = Mock(side_effect=lambda data, metadata: data)
            task.create_windows = Mock(side_effect=lambda data, strategy, num_window: [
                np.random.randn(128, 2), 
                np.random.randn(128, 2)
            ])
            task.log = Mock()
            
            return task
    
    def create_mock_batch_data(self, num_samples=4):
        """创建模拟批次数据"""
        batch_data = []
        for i in range(num_samples):
            sample_id = f"sample_{i}"
            # 创建足够长的信号以支持窗口化
            data_array = np.random.randn(256, 2)  # 256时间步，2通道
            metadata = {'Label': i % 3, 'Domain_id': 1}
            batch_data.append((sample_id, data_array, metadata))
        return batch_data
    
    def test_task_initialization(self, mock_config, mock_network):
        """测试任务初始化"""
        # 测试正常初始化
        with patch('src.task_factory.task.pretrain.ContrastiveIDTask.BaseIDTask.__init__'):
            task = ContrastiveIDTask(
                network=mock_network,
                args_data=mock_config['args_data'],
                args_model=mock_config['args_model'],
                args_task=mock_config['args_task'],
                args_trainer=mock_config['args_trainer'],
                args_environment=mock_config['args_environment'],
                metadata=mock_config['metadata']
            )
            
            # 验证温度参数
            assert task.temperature == 0.07
    
    def test_prepare_batch_normal(self, contrastive_task):
        """测试正常批次准备"""
        batch_data = self.create_mock_batch_data(4)
        
        result = contrastive_task.prepare_batch(batch_data)
        
        # 验证返回结构
        assert 'anchor' in result
        assert 'positive' in result
        assert 'ids' in result
        
        # 验证张量形状
        assert result['anchor'].shape[0] == 4  # 批量大小
        assert result['positive'].shape[0] == 4
        assert result['anchor'].shape[1:] == (128, 2)  # 窗口形状
        assert result['positive'].shape[1:] == (128, 2)
        
        # 验证ID列表
        assert len(result['ids']) == 4
        assert all(id.startswith('sample_') for id in result['ids'])
        
        # 验证调用了BaseIDTask的方法
        assert contrastive_task.process_sample.call_count == 4
        assert contrastive_task.create_windows.call_count == 4
    
    def test_prepare_batch_insufficient_windows(self, contrastive_task):
        """测试窗口数量不足的情况"""
        # 模拟create_windows返回不足的窗口
        contrastive_task.create_windows = Mock(return_value=[np.random.randn(128, 2)])
        
        batch_data = self.create_mock_batch_data(2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略警告
            result = contrastive_task.prepare_batch(batch_data)
        
        # 应该跳过窗口不足的样本
        assert len(result['ids']) == 0  # 所有样本都被跳过
        assert result['anchor'].shape[0] == 0
        assert result['positive'].shape[0] == 0
    
    def test_prepare_batch_empty_input(self, contrastive_task):
        """测试空输入"""
        result = contrastive_task.prepare_batch([])
        
        # 验证返回空批次
        assert len(result['ids']) == 0
        assert result['anchor'].shape[0] == 0
        assert result['positive'].shape[0] == 0
    
    def test_prepare_batch_processing_error(self, contrastive_task):
        """测试数据处理错误"""
        # 模拟process_sample抛出异常
        contrastive_task.process_sample = Mock(side_effect=RuntimeError("Processing error"))
        
        batch_data = self.create_mock_batch_data(2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略警告
            result = contrastive_task.prepare_batch(batch_data)
        
        # 所有样本都应该被跳过
        assert len(result['ids']) == 0
        assert result['anchor'].shape[0] == 0
        assert result['positive'].shape[0] == 0
    
    def test_empty_batch(self, contrastive_task):
        """测试空批次生成"""
        result = contrastive_task._empty_batch()
        
        assert 'anchor' in result
        assert 'positive' in result  
        assert 'ids' in result
        
        # 验证空张量形状
        assert result['anchor'].shape == (0, 128, 1)
        assert result['positive'].shape == (0, 128, 1)
        assert result['ids'] == []
    
    def test_infonce_loss_computation(self, contrastive_task):
        """测试InfoNCE损失计算"""
        batch_size = 4
        feature_dim = 64
        
        # 创建模拟特征
        z_anchor = torch.randn(batch_size, feature_dim)
        z_positive = torch.randn(batch_size, feature_dim)
        
        loss = contrastive_task.infonce_loss(z_anchor, z_positive)
        
        # 验证损失是标量
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0  # InfoNCE损失应该非负
    
    def test_infonce_loss_identical_features(self, contrastive_task):
        """测试相同特征的InfoNCE损失"""
        batch_size = 4
        feature_dim = 64
        
        # 创建相同的特征
        z_anchor = torch.randn(batch_size, feature_dim)
        z_positive = z_anchor.clone()
        
        loss = contrastive_task.infonce_loss(z_anchor, z_positive)
        
        # 相同特征的损失应该接近0
        assert loss.item() < 1e-5
    
    def test_infonce_loss_temperature_effect(self, contrastive_task):
        """测试温度参数对损失的影响"""
        batch_size = 4
        feature_dim = 64
        
        z_anchor = torch.randn(batch_size, feature_dim)
        z_positive = torch.randn(batch_size, feature_dim)
        
        # 测试不同温度
        original_temp = contrastive_task.temperature
        
        contrastive_task.temperature = 0.01  # 低温度
        loss_low_temp = contrastive_task.infonce_loss(z_anchor, z_positive)
        
        contrastive_task.temperature = 1.0   # 高温度
        loss_high_temp = contrastive_task.infonce_loss(z_anchor, z_positive)
        
        # 温度较低时损失通常更大
        assert loss_low_temp.item() > loss_high_temp.item()
        
        # 恢复原始温度
        contrastive_task.temperature = original_temp
    
    def test_compute_accuracy(self, contrastive_task):
        """测试准确率计算"""
        batch_size = 4
        feature_dim = 64
        
        z_anchor = torch.randn(batch_size, feature_dim)
        z_positive = torch.randn(batch_size, feature_dim)
        
        accuracy = contrastive_task.compute_accuracy(z_anchor, z_positive)
        
        # 验证准确率在[0, 1]范围内
        assert 0 <= accuracy.item() <= 1
        assert accuracy.dim() == 0
    
    def test_compute_accuracy_perfect_match(self, contrastive_task):
        """测试完美匹配的准确率"""
        batch_size = 4
        feature_dim = 64
        
        # 创建完美匹配的特征（相同特征）
        z_anchor = torch.randn(batch_size, feature_dim)
        z_positive = z_anchor.clone()
        
        accuracy = contrastive_task.compute_accuracy(z_anchor, z_positive)
        
        # 完美匹配应该得到100%准确率
        assert abs(accuracy.item() - 1.0) < 1e-6
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA不可用")
    def test_gpu_compatibility(self, contrastive_task):
        """测试GPU兼容性"""
        batch_size = 4
        feature_dim = 64
        
        # 将特征移动到GPU
        z_anchor = torch.randn(batch_size, feature_dim).cuda()
        z_positive = torch.randn(batch_size, feature_dim).cuda()
        
        # 测试InfoNCE损失
        loss = contrastive_task.infonce_loss(z_anchor, z_positive)
        assert loss.device.type == 'cuda'
        
        # 测试准确率计算
        accuracy = contrastive_task.compute_accuracy(z_anchor, z_positive)
        assert accuracy.device.type == 'cuda'
    
    def test_shared_step_normal(self, contrastive_task):
        """测试正常的共享步骤"""
        # 创建模拟批次
        batch = {
            'anchor': torch.randn(4, 128, 2),
            'positive': torch.randn(4, 128, 2),
            'ids': ['sample_0', 'sample_1', 'sample_2', 'sample_3']
        }
        
        # 模拟网络输出
        contrastive_task.network = Mock(return_value=torch.randn(4, 64))
        
        result = contrastive_task._shared_step(batch, 'train')
        
        # 验证返回结果
        assert 'loss' in result
        assert 'accuracy' in result
        assert torch.isfinite(result['loss'])
        assert 0 <= result['accuracy'].item() <= 1
        
        # 验证日志记录被调用
        assert contrastive_task.log.call_count == 2
    
    def test_shared_step_empty_batch(self, contrastive_task):
        """测试空批次的共享步骤"""
        # 空批次
        batch = {
            'anchor': torch.empty(0, 128, 2),
            'positive': torch.empty(0, 128, 2),
            'ids': []
        }
        
        result = contrastive_task._shared_step(batch, 'train')
        
        # 应该返回零损失
        assert result['loss'].item() == 0.0
        assert result['loss'].requires_grad  # 确保梯度计算
    
    def test_shared_step_raw_batch_preprocessing(self, contrastive_task):
        """测试原始批次预处理"""
        # 模拟原始批次（不包含anchor/positive）
        raw_batch = [
            ('sample_0', np.random.randn(256, 2), {'Label': 0}),
            ('sample_1', np.random.randn(256, 2), {'Label': 1})
        ]
        
        # 模拟_preprocess_raw_batch方法
        contrastive_task._preprocess_raw_batch = Mock(return_value={
            'anchor': torch.randn(2, 128, 2),
            'positive': torch.randn(2, 128, 2), 
            'ids': ['sample_0', 'sample_1']
        })
        
        contrastive_task.network = Mock(return_value=torch.randn(2, 64))
        
        result = contrastive_task._shared_step(raw_batch, 'train')
        
        # 验证预处理被调用
        contrastive_task._preprocess_raw_batch.assert_called_once_with(raw_batch)
        
        # 验证返回结果
        assert 'loss' in result
        assert 'accuracy' in result
    
    def test_batch_size_robustness(self, contrastive_task):
        """测试不同批量大小的鲁棒性"""
        batch_sizes = [1, 2, 8, 16]
        
        for batch_size in batch_sizes:
            batch_data = self.create_mock_batch_data(batch_size)
            result = contrastive_task.prepare_batch(batch_data)
            
            # 验证输出形状
            assert result['anchor'].shape[0] == batch_size
            assert result['positive'].shape[0] == batch_size
            assert len(result['ids']) == batch_size
            
            # 测试损失计算
            features = torch.randn(batch_size, 64)
            loss = contrastive_task.infonce_loss(features, features)
            assert torch.isfinite(loss)
    
    def test_edge_cases_single_sample(self, contrastive_task):
        """测试单样本边界情况"""
        batch_data = self.create_mock_batch_data(1)
        result = contrastive_task.prepare_batch(batch_data)
        
        # 单样本应该也能正常处理
        assert result['anchor'].shape[0] == 1
        assert result['positive'].shape[0] == 1
        assert len(result['ids']) == 1
        
        # 测试单样本损失计算
        features = torch.randn(1, 64)
        loss = contrastive_task.infonce_loss(features, features)
        accuracy = contrastive_task.compute_accuracy(features, features)
        
        assert torch.isfinite(loss)
        assert 0 <= accuracy.item() <= 1
    
    def test_normalization_effect(self, contrastive_task):
        """测试L2归一化的效果"""
        batch_size = 4
        feature_dim = 64
        
        # 创建未归一化的特征（大幅度值）
        z_anchor = torch.randn(batch_size, feature_dim) * 100
        z_positive = torch.randn(batch_size, feature_dim) * 100
        
        loss = contrastive_task.infonce_loss(z_anchor, z_positive)
        
        # 即使输入幅度很大，损失仍应该有界
        assert torch.isfinite(loss)
        assert loss.item() < 100  # 应该在合理范围内
    
    def test_numerical_stability(self, contrastive_task):
        """测试数值稳定性"""
        batch_size = 4
        feature_dim = 64
        
        # 测试极小值
        z_anchor = torch.randn(batch_size, feature_dim) * 1e-6
        z_positive = torch.randn(batch_size, feature_dim) * 1e-6
        
        loss_small = contrastive_task.infonce_loss(z_anchor, z_positive)
        acc_small = contrastive_task.compute_accuracy(z_anchor, z_positive)
        
        assert torch.isfinite(loss_small)
        assert torch.isfinite(acc_small)
        
        # 测试极大值
        z_anchor = torch.randn(batch_size, feature_dim) * 1e6
        z_positive = torch.randn(batch_size, feature_dim) * 1e6
        
        loss_large = contrastive_task.infonce_loss(z_anchor, z_positive)
        acc_large = contrastive_task.compute_accuracy(z_anchor, z_positive)
        
        assert torch.isfinite(loss_large)
        assert torch.isfinite(acc_large)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])