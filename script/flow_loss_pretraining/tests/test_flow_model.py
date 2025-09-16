#!/usr/bin/env python3
"""
Flow模型测试脚本
用于验证Flow模型的基本功能和集成
"""

import pytest
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.model_factory.ISFM.M_04_ISFM_Flow import Model as FlowModel


class TestFlowModel:
    """Flow模型测试类"""

    @pytest.fixture
    def model_config(self):
        """模型配置"""
        class Args:
            def __init__(self):
                self.sequence_length = 1024
                self.channels = 1
                self.hidden_dim = 128
                self.condition_dim = 32
                self.use_conditional = True
                self.num_steps = 20
                self.sigma = 0.001

        return Args()

    @pytest.fixture
    def sample_metadata(self):
        """样本元数据"""
        import pandas as pd

        metadata = pd.DataFrame({
            'Id': ['test_1', 'test_2', 'test_3'],
            'Dataset_id': [1, 1, 1],
            'Domain_id': [1, 1, 1],
            'Label': [0, 1, 2]
        })
        return metadata

    def test_model_initialization(self, model_config, sample_metadata):
        """测试模型初始化"""
        model = FlowModel(model_config, sample_metadata)

        assert model is not None
        assert hasattr(model, 'flow_model')
        assert hasattr(model, 'condition_encoder')

    def test_forward_pass(self, model_config, sample_metadata):
        """测试前向传播"""
        model = FlowModel(model_config, sample_metadata)

        # 创建输入数据
        batch_size = 4
        x = torch.randn(batch_size, 1024, 1)
        file_ids = ['test_1', 'test_2', 'test_1', 'test_3']

        # 前向传播
        output = model(x, file_ids)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size

    def test_sampling(self, model_config, sample_metadata):
        """测试采样生成"""
        model = FlowModel(model_config, sample_metadata)

        # 生成样本
        samples = model.sample(
            batch_size=3,
            file_ids=['test_1', 'test_2', 'test_3'],
            num_steps=10
        )

        assert samples is not None
        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (3, 1024, 1)

    def test_anomaly_detection(self, model_config, sample_metadata):
        """测试异常检测"""
        model = FlowModel(model_config, sample_metadata)

        # 创建测试数据
        x = torch.randn(2, 1024, 1)
        file_ids = ['test_1', 'test_2']

        # 计算异常分数
        anomaly_scores = model.compute_anomaly_score(x, file_ids)

        assert anomaly_scores is not None
        assert isinstance(anomaly_scores, torch.Tensor)
        assert anomaly_scores.shape[0] == 2

    def test_device_consistency(self, model_config, sample_metadata):
        """测试设备一致性"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FlowModel(model_config, sample_metadata).to(device)

        x = torch.randn(2, 1024, 1).to(device)
        file_ids = ['test_1', 'test_2']

        output = model(x, file_ids)

        assert output.device == device

    def test_gradient_flow(self, model_config, sample_metadata):
        """测试梯度流"""
        model = FlowModel(model_config, sample_metadata)

        x = torch.randn(2, 1024, 1, requires_grad=True)
        file_ids = ['test_1', 'test_2']

        output = model(x, file_ids)
        loss = output.mean()
        loss.backward()

        # 检查梯度
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestFlowModelIntegration:
    """Flow模型集成测试"""

    def test_config_loading(self):
        """测试配置加载"""
        config_path = os.path.join(
            os.path.dirname(__file__),
            '../experiments/configs/quick_validation.yaml'
        )

        # 如果配置文件存在，测试加载
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            assert 'model' in config
            assert config['model']['name'] == 'M_04_ISFM_Flow'

    def test_training_compatibility(self, model_config, sample_metadata):
        """测试训练兼容性"""
        model = FlowModel(model_config, sample_metadata)

        # 模拟训练步骤
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        x = torch.randn(4, 1024, 1)
        file_ids = ['test_1', 'test_2', 'test_1', 'test_3']

        # 前向传播
        output = model(x, file_ids)
        loss = output.mean()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证参数更新
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


def test_environment_setup():
    """测试环境设置"""
    # 检查必要的包
    try:
        import torch
        import numpy as np
        import pandas as pd
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        pytest.fail(f"Missing required package: {e}")


if __name__ == "__main__":
    # 运行基本测试
    print("🧪 运行Flow模型基础测试...")

    # 环境测试
    test_environment_setup()

    # 创建测试实例
    class Args:
        def __init__(self):
            self.sequence_length = 1024
            self.channels = 1
            self.hidden_dim = 128
            self.condition_dim = 32
            self.use_conditional = True
            self.num_steps = 10
            self.sigma = 0.001

    # 创建元数据
    import pandas as pd
    metadata = pd.DataFrame({
        'Id': ['test_1', 'test_2', 'test_3'],
        'Dataset_id': [1, 1, 1],
        'Domain_id': [1, 1, 1],
        'Label': [0, 1, 2]
    })

    try:
        # 测试模型创建
        print("📝 测试模型初始化...")
        model = FlowModel(Args(), metadata)
        print("✅ 模型初始化成功")

        # 测试前向传播
        print("📝 测试前向传播...")
        x = torch.randn(2, 1024, 1)
        file_ids = ['test_1', 'test_2']
        output = model(x, file_ids)
        print(f"✅ 前向传播成功，输出形状: {output.shape}")

        # 测试采样
        print("📝 测试采样生成...")
        samples = model.sample(batch_size=2, file_ids=['test_1', 'test_2'], num_steps=5)
        print(f"✅ 采样成功，样本形状: {samples.shape}")

        # 测试异常检测
        print("📝 测试异常检测...")
        anomaly_scores = model.compute_anomaly_score(x, file_ids)
        print(f"✅ 异常检测成功，分数形状: {anomaly_scores.shape}")

        print("🎉 所有基础测试通过！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise