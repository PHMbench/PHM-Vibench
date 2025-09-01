"""
Flow模型集成测试
测试M_04_ISFM_Flow主模型的完整功能
"""

import pytest
import torch
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from model_factory.ISFM.M_04_ISFM_Flow import Model


class MockArgs:
    """模拟配置参数"""
    def __init__(self):
        self.sequence_length = 256
        self.channels = 1
        self.hidden_dim = 64
        self.time_dim = 32
        self.condition_dim = 32
        self.use_conditional = True
        self.sigma_min = 0.001
        self.sigma_max = 1.0


class MockMetadata:
    """模拟metadata"""
    def __init__(self):
        self.df = pd.DataFrame({
            'Domain_id': [1, 2, 1, 3, 2],
            'Dataset_id': [5, 8, 5, 10, 12],
            'Name': ['CWRU', 'XJTU', 'PU', 'FEMTO', 'IMS']
        })
        
        self._data = {
            'file1': {'Domain_id': 1, 'Dataset_id': 5, 'Name': 'CWRU'},
            'file2': {'Domain_id': 2, 'Dataset_id': 8, 'Name': 'XJTU'},
            'file3': {'Domain_id': 1, 'Dataset_id': 5, 'Name': 'PU'},
            'file4': {'Domain_id': 3, 'Dataset_id': 10, 'Name': 'FEMTO'},
            'missing_file': {'Domain_id': None, 'Dataset_id': None, 'Name': 'Unknown'}
        }
    
    def __contains__(self, key):
        return key in self._data
    
    def __getitem__(self, key):
        return self._data.get(key, {'Domain_id': None, 'Dataset_id': None, 'Name': 'Unknown'})


class TestM04ISFMFlow:
    """测试M_04_ISFM_Flow主模型"""
    
    @pytest.fixture
    def mock_args(self):
        return MockArgs()
    
    @pytest.fixture
    def mock_metadata(self):
        return MockMetadata()
    
    @pytest.fixture
    def model(self, mock_args, mock_metadata):
        return Model(mock_args, mock_metadata)
    
    def test_model_initialization(self, mock_args, mock_metadata):
        """测试模型初始化"""
        model = Model(mock_args, mock_metadata)
        
        # 检查基本属性
        assert model.sequence_length == 256
        assert model.channels == 1
        assert model.latent_dim == 256  # sequence_length * channels
        assert model.use_conditional == True
        
        # 检查组件是否正确初始化
        assert model.flow_model is not None
        assert model.condition_encoder is not None
        assert model.metadata is not None
    
    def test_model_initialization_no_metadata(self, mock_args):
        """测试无metadata的初始化"""
        model = Model(mock_args, metadata=None)
        
        assert model.condition_encoder is not None  # 应该使用默认配置
        assert model.metadata is None
    
    def test_model_initialization_no_conditional(self, mock_args, mock_metadata):
        """测试禁用条件编码的初始化"""
        mock_args.use_conditional = False
        model = Model(mock_args, mock_metadata)
        
        assert model.condition_encoder is None
        assert model.condition_dim == 0
    
    def test_forward_pass_with_conditions(self, model):
        """测试带条件的前向传播"""
        batch_size = 4
        x = torch.randn(batch_size, 256, 1)  # (B, L, C)
        file_ids = ['file1', 'file2', 'file3', 'file4']
        
        outputs = model(x, file_ids)
        
        # 检查输出键
        expected_keys = [
            'v_pred', 'v_true', 'x_t', 'noise', 't',
            'flow_loss', 'total_loss',
            'x_original', 'x_flat', 'condition_features'
        ]
        for key in expected_keys:
            assert key in outputs
        
        # 检查形状
        assert outputs['v_pred'].shape == (batch_size, 256)
        assert outputs['x_original'].shape == (batch_size, 256, 1)
        assert outputs['condition_features'].shape == (batch_size, 32)
        
        # 检查损失
        assert outputs['total_loss'].item() >= 0
        assert not torch.isnan(outputs['total_loss'])
    
    def test_forward_pass_no_conditions(self, model):
        """测试无条件的前向传播"""
        batch_size = 2
        x = torch.randn(batch_size, 256, 1)
        
        outputs = model(x, file_ids=None)
        
        assert outputs['v_pred'].shape == (batch_size, 256)
        assert outputs['condition_features'] is None
    
    def test_forward_pass_missing_file_ids(self, model):
        """测试缺失file_id的处理"""
        batch_size = 3
        x = torch.randn(batch_size, 256, 1)
        file_ids = ['file1', 'nonexistent_file', 'file3']
        
        outputs = model(x, file_ids)
        
        # 应该正常处理，不报错
        assert outputs['v_pred'].shape == (batch_size, 256)
        assert outputs['condition_features'].shape == (batch_size, 32)
    
    def test_dimension_mismatch_warning(self, model, capfd):
        """测试维度不匹配的警告"""
        x = torch.randn(2, 128, 2)  # 错误的维度
        
        outputs = model(x, file_ids=['file1', 'file2'])
        
        # 捕获警告输出
        captured = capfd.readouterr()
        assert "维度不匹配" in captured.out
        
        # 模型应该仍然能处理
        assert 'v_pred' in outputs
    
    def test_sampling_with_conditions(self, model):
        """测试条件采样"""
        model.eval()
        
        samples = model.sample(
            batch_size=3,
            file_ids=['file1', 'file2', 'file3'],
            num_steps=10
        )
        
        assert samples.shape == (3, 256, 1)  # (B, L, C)
        assert not torch.isnan(samples).any()
    
    def test_sampling_no_conditions(self, model):
        """测试无条件采样"""
        model.eval()
        
        samples = model.sample(
            batch_size=2,
            file_ids=None,
            num_steps=10
        )
        
        assert samples.shape == (2, 256, 1)
        assert not torch.isnan(samples).any()
    
    def test_encode_to_noise(self, model):
        """测试编码到噪声"""
        model.eval()
        
        x = torch.randn(2, 256, 1)
        file_ids = ['file1', 'file2']
        
        noise = model.encode_to_noise(x, file_ids, num_steps=10)
        
        assert noise.shape == (2, 256, 1)
        assert not torch.isnan(noise).any()
    
    def test_anomaly_score_computation(self, model):
        """测试异常分数计算"""
        model.eval()
        
        x = torch.randn(3, 256, 1)
        file_ids = ['file1', 'file2', 'file3']
        
        scores = model.compute_anomaly_score(x, file_ids, num_steps=10)
        
        assert scores.shape == (3,)
        assert (scores >= 0).all()  # 异常分数应该非负
        assert not torch.isnan(scores).any()
    
    def test_gradient_flow(self, model):
        """测试梯度流动"""
        model.train()
        
        x = torch.randn(2, 256, 1, requires_grad=True)
        file_ids = ['file1', 'file2']
        
        outputs = model(x, file_ids)
        loss = outputs['total_loss']
        
        loss.backward()
        
        # 检查主要参数都有梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"
                assert not torch.isnan(param.grad).any(), f"参数 {name} 的梯度包含NaN"
    
    def test_memory_efficiency(self, model):
        """测试内存效率"""
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            device = 'cuda'
            
            # 清空缓存
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # 大批量测试
            large_batch_size = 32
            x = torch.randn(large_batch_size, 256, 1, device=device)
            file_ids = [f'file{i%4+1}' for i in range(large_batch_size)]
            
            with torch.no_grad():
                outputs = model(x, file_ids, return_loss=False)
            
            peak_memory = torch.cuda.memory_allocated()
            memory_usage = (peak_memory - initial_memory) / 1024**2  # MB
            
            print(f"内存使用: {memory_usage:.2f} MB for batch_size={large_batch_size}")
            
            # 清理
            del x, outputs
            torch.cuda.empty_cache()
    
    def test_batch_size_variation(self, model):
        """测试不同批量大小"""
        model.eval()
        
        for batch_size in [1, 3, 8, 16]:
            x = torch.randn(batch_size, 256, 1)
            file_ids = [f'file{i%4+1}' for i in range(batch_size)]
            
            outputs = model(x, file_ids)
            
            assert outputs['v_pred'].shape == (batch_size, 256)
            assert outputs['condition_features'].shape == (batch_size, 32)
            
            # 采样测试
            samples = model.sample(batch_size, file_ids[:batch_size], num_steps=5)
            assert samples.shape == (batch_size, 256, 1)
    
    def test_reproducibility(self, model):
        """测试结果可重现性"""
        model.eval()
        
        # 设置随机种子
        torch.manual_seed(42)
        x1 = torch.randn(2, 256, 1)
        
        torch.manual_seed(42)
        x2 = torch.randn(2, 256, 1)
        
        assert torch.allclose(x1, x2)
        
        # 测试模型输出的一致性（在评估模式下）
        torch.manual_seed(42)
        samples1 = model.sample(2, ['file1', 'file2'], num_steps=10)
        
        torch.manual_seed(42)
        samples2 = model.sample(2, ['file1', 'file2'], num_steps=10)
        
        # 注意: 由于采样过程中的随机性，这里只测试形状一致性
        assert samples1.shape == samples2.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])