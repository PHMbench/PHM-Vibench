"""
Flow模型基础功能测试
测试核心组件的基本功能
"""

import pytest
import torch
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from model_factory.ISFM.layers.flow_model import RectifiedFlow
from model_factory.ISFM.layers.condition_encoder import ConditionalEncoder
from model_factory.ISFM.layers.utils.flow_utils import (
    DimensionAdapter, TimeEmbedding, MetadataExtractor
)


class TestDimensionAdapter:
    """测试维度适配器"""
    
    def test_3d_to_1d_encoding(self):
        """测试3D到1D编码"""
        x = torch.randn(4, 100, 3)  # (B, L, C)
        x_flat = DimensionAdapter.encode_3d_to_1d(x)
        
        assert x_flat.shape == (4, 300)  # (B, L*C)
        assert torch.allclose(x.view(4, 300), x_flat)
    
    def test_1d_to_3d_decoding(self):
        """测试1D到3D解码"""
        x_flat = torch.randn(4, 300)  # (B, L*C)
        x = DimensionAdapter.decode_1d_to_3d(x_flat, seq_len=100, channels=3)
        
        assert x.shape == (4, 100, 3)  # (B, L, C)
        assert torch.allclose(x_flat.view(4, 100, 3), x)
    
    def test_round_trip(self):
        """测试往返编码"""
        original = torch.randn(2, 50, 2)
        encoded = DimensionAdapter.encode_3d_to_1d(original)
        decoded = DimensionAdapter.decode_1d_to_3d(encoded, 50, 2)
        
        assert torch.allclose(original, decoded)


class TestTimeEmbedding:
    """测试时间编码"""
    
    def test_time_embedding_shape(self):
        """测试时间编码形状"""
        time_emb = TimeEmbedding(dim=64)
        t = torch.rand(8)  # 8个时间步
        
        emb = time_emb(t)
        assert emb.shape == (8, 64)
    
    def test_time_embedding_range(self):
        """测试时间编码值域"""
        time_emb = TimeEmbedding(dim=32)
        t = torch.tensor([0.0, 0.5, 1.0])
        
        emb = time_emb(t)
        
        # 检查没有NaN或Inf
        assert not torch.isnan(emb).any()
        assert not torch.isinf(emb).any()
        
        # 不同时间步应该产生不同的编码
        assert not torch.allclose(emb[0], emb[1])
        assert not torch.allclose(emb[1], emb[2])


class TestMetadataExtractor:
    """测试metadata提取器"""
    
    def test_extract_condition_ids_normal(self):
        """测试正常的条件ID提取"""
        metadata = {
            'Domain_id': 5,
            'Dataset_id': 12,
            'Name': 'CWRU'
        }
        
        domain_id, system_id = MetadataExtractor.extract_condition_ids(metadata)
        assert domain_id == 5
        assert system_id == 12
    
    def test_extract_condition_ids_missing(self):
        """测试缺失值的处理"""
        metadata = {
            'Domain_id': None,
            'Dataset_id': pd.NA,
            'Name': 'Unknown'
        }
        
        domain_id, system_id = MetadataExtractor.extract_condition_ids(metadata)
        assert domain_id == 0  # 未知值应该返回0
        assert system_id == 0
    
    def test_get_max_ids(self):
        """测试获取最大ID"""
        df = pd.DataFrame({
            'Domain_id': [1, 2, 5, 3],
            'Dataset_id': [10, 15, 8, 20],
            'Name': ['A', 'B', 'C', 'D']
        })
        
        max_domain, max_system = MetadataExtractor.get_max_ids(df)
        assert max_domain == 5
        assert max_system == 20


class TestRectifiedFlow:
    """测试RectifiedFlow模型"""
    
    @pytest.fixture
    def flow_model(self):
        return RectifiedFlow(
            latent_dim=128,
            hidden_dim=64,
            condition_dim=32
        )
    
    def test_forward_pass(self, flow_model):
        """测试前向传播"""
        batch_size = 4
        x = torch.randn(batch_size, 128)
        condition = torch.randn(batch_size, 32)
        
        outputs = flow_model(x, condition)
        
        # 检查输出键
        expected_keys = ['v_pred', 'v_true', 'x_t', 'noise', 't', 't_emb']
        for key in expected_keys:
            assert key in outputs
        
        # 检查形状
        assert outputs['v_pred'].shape == (batch_size, 128)
        assert outputs['v_true'].shape == (batch_size, 128)
        assert outputs['x_t'].shape == (batch_size, 128)
    
    def test_forward_no_condition(self, flow_model):
        """测试无条件前向传播"""
        x = torch.randn(2, 128)
        
        outputs = flow_model(x)  # 不传入condition
        
        assert outputs['v_pred'].shape == (2, 128)
        assert outputs['v_true'].shape == (2, 128)
    
    def test_loss_computation(self, flow_model):
        """测试损失计算"""
        x = torch.randn(3, 128)
        outputs = flow_model(x)
        losses = flow_model.compute_loss(outputs)
        
        # 检查损失项
        assert 'flow_loss' in losses
        assert 'velocity_reg' in losses
        assert 'total_loss' in losses
        
        # 检查损失值
        assert losses['total_loss'].item() >= 0
        assert not torch.isnan(losses['total_loss'])
    
    def test_sampling(self, flow_model):
        """测试采样"""
        flow_model.eval()
        
        samples = flow_model.sample(
            batch_size=3,
            num_steps=10,
            device='cpu'
        )
        
        assert samples.shape == (3, 128)
        assert not torch.isnan(samples).any()
    
    def test_conditional_sampling(self, flow_model):
        """测试条件采样"""
        flow_model.eval()
        condition = torch.randn(2, 32)
        
        samples = flow_model.sample(
            batch_size=2,
            condition=condition,
            num_steps=10,
            device='cpu'
        )
        
        assert samples.shape == (2, 128)
        assert not torch.isnan(samples).any()
    
    def test_encode_to_noise(self, flow_model):
        """测试编码到噪声"""
        flow_model.eval()
        x = torch.randn(2, 128)
        
        noise = flow_model.encode_to_noise(x, num_steps=10)
        
        assert noise.shape == (2, 128)
        assert not torch.isnan(noise).any()


class TestConditionalEncoder:
    """测试条件编码器"""
    
    @pytest.fixture
    def encoder(self):
        return ConditionalEncoder(
            embed_dim=32,
            num_domains=5,
            num_systems=10
        )
    
    def test_forward_pass(self, encoder):
        """测试前向传播"""
        metadata_batch = [
            {'Domain_id': 1, 'Dataset_id': 3, 'Name': 'Test1'},
            {'Domain_id': 2, 'Dataset_id': 5, 'Name': 'Test2'},
        ]
        
        features = encoder(metadata_batch)
        
        assert features.shape == (2, 32)
        assert not torch.isnan(features).any()
    
    def test_missing_values(self, encoder):
        """测试缺失值处理"""
        metadata_batch = [
            {'Domain_id': None, 'Dataset_id': 3, 'Name': 'Test1'},
            {'Domain_id': 2, 'Dataset_id': None, 'Name': 'Test2'},
        ]
        
        features = encoder(metadata_batch)
        
        assert features.shape == (2, 32)
        assert not torch.isnan(features).any()
    
    def test_out_of_range_ids(self, encoder):
        """测试超出范围的ID"""
        metadata_batch = [
            {'Domain_id': 100, 'Dataset_id': 200, 'Name': 'Test'},  # 超出范围
        ]
        
        features = encoder(metadata_batch)
        
        assert features.shape == (1, 32)
        assert not torch.isnan(features).any()
    
    def test_prototype_retrieval(self, encoder):
        """测试原型检索"""
        domain_proto = encoder.get_domain_prototype(1)
        system_proto = encoder.get_system_prototype(3)
        
        assert domain_proto.shape == (32,)
        assert system_proto.shape == (32,)
        assert not torch.isnan(domain_proto).any()
        assert not torch.isnan(system_proto).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])