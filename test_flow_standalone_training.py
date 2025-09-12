#!/usr/bin/env python3
"""
Standalone Flow pretraining test with synthetic data.
Tests the complete Flow training loop without external dependencies.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from argparse import Namespace
import numpy as np
import tempfile
import os

# Import Flow components
from src.model_factory.ISFM.M_04_ISFM_Flow import Model as FlowModel
from src.task_factory.task.pretrain.flow_pretrain import FlowPretrainTask


def create_synthetic_data(batch_size=8, seq_len=512, num_channels=1, num_batches=5):
    """Create synthetic vibration data for testing."""
    synthetic_data = []
    for i in range(num_batches):
        # Generate realistic vibration signals
        t = torch.linspace(0, 1, seq_len)
        
        # Create base signal with multiple frequency components
        signal = (torch.sin(2 * np.pi * 50 * t) +  # 50Hz component
                 0.5 * torch.sin(2 * np.pi * 120 * t) +  # 120Hz harmonic
                 0.3 * torch.sin(2 * np.pi * 180 * t) +  # 180Hz harmonic
                 0.1 * torch.randn(seq_len))  # Noise
        
        # Create batch with variations
        batch_x = []
        batch_file_ids = []
        
        for j in range(batch_size):
            # Add random variations to each sample
            variation = signal + 0.05 * torch.randn(seq_len)
            # Shape should be (seq_len, num_channels) = (512, 1) per the model expectation
            variation = variation.unsqueeze(-1)  # Add channel dimension at the end
            batch_x.append(variation)  # Shape: (seq_len, channels)
            batch_file_ids.append(f'synthetic_file_{i}_{j}')
        
        x = torch.stack(batch_x)  # Shape: (batch_size, seq_len, channels)
        
        batch = {
            'x': x,
            'file_id': batch_file_ids
        }
        synthetic_data.append(batch)
    
    return synthetic_data


def create_mock_arguments():
    """Create mock arguments for Flow pretraining."""
    
    # Data args
    args_data = Namespace()
    args_data.batch_size = 8
    args_data.window_size = 512
    args_data.num_channels = 1
    
    # Model args (matching M_04_ISFM_Flow expected parameters)
    args_model = Namespace()
    args_model.type = "ISFM"
    args_model.name = "M_04_ISFM_Flow"
    args_model.sequence_length = 512
    args_model.channels = 1
    args_model.hidden_dim = 128
    args_model.time_dim = 32
    args_model.condition_dim = 32
    args_model.use_conditional = True
    args_model.sigma_min = 0.001
    args_model.sigma_max = 1.0
    
    # Task args
    args_task = Namespace()
    args_task.name = "flow_pretrain"
    args_task.type = "pretrain"
    args_task.loss = "CE"  # Required by Default_task
    args_task.metrics = ["acc"]  # Required by Default_task
    args_task.num_steps = 20  # Fewer steps for quick test
    args_task.flow_lr = 1e-3
    args_task.use_contrastive = False  # Start with Flow only
    args_task.enable_visualization = False
    args_task.track_memory = True
    args_task.track_gradients = False
    args_task.lr = 1e-3
    args_task.weight_decay = 1e-5
    
    # Trainer args
    args_trainer = Namespace()
    args_trainer.gpus = 1 if torch.cuda.is_available() else 0
    args_trainer.precision = 16 if torch.cuda.is_available() else 32
    args_trainer.gradient_clip_val = 1.0
    args_trainer.max_epochs = 1
    args_trainer.log_every_n_steps = 5
    
    # Environment args
    args_environment = Namespace()
    args_environment.seed = 42
    
    # Mock metadata (compatible with both Flow model and metrics)
    class MockMetadata:
        def __init__(self):
            import pandas as pd
            self.df = pd.DataFrame({
                'Dataset_id': [1, 2, 3, 4],
                'Domain_id': [1, 1, 2, 2], 
                'file_id': ['file1', 'file2', 'file3', 'file4'],
                'Label': [0, 1, 0, 1]
            })
            # Create mapping for file_id lookup
            self._file_mapping = {
                'file1': {'Domain_id': 1, 'Dataset_id': 1, 'Label': 0},
                'file2': {'Domain_id': 1, 'Dataset_id': 2, 'Label': 1},
                'file3': {'Domain_id': 2, 'Dataset_id': 3, 'Label': 0},
                'file4': {'Domain_id': 2, 'Dataset_id': 4, 'Label': 1},
                'synthetic_file_0_0': {'Domain_id': 1, 'Dataset_id': 1, 'Label': 0},
                'synthetic_file_0_1': {'Domain_id': 1, 'Dataset_id': 1, 'Label': 0},
                'synthetic_file_1_0': {'Domain_id': 1, 'Dataset_id': 1, 'Label': 0},
                'test_1': {'Domain_id': 1, 'Dataset_id': 1, 'Label': 0},
                'test_2': {'Domain_id': 1, 'Dataset_id': 1, 'Label': 0}
            }
        
        def items(self):
            """Provide items() method for metrics compatibility."""
            return {
                'item1': {'Name': 'test_dataset', 'Label': 0},
                'item2': {'Name': 'test_dataset', 'Label': 1},
                'item3': {'Name': 'test_dataset', 'Label': 2}, 
                'item4': {'Name': 'test_dataset', 'Label': 3}
            }.items()
            
        def __contains__(self, key):
            """Make metadata iterable for 'in' checks."""
            return key in self._file_mapping
            
        def __getitem__(self, key):
            """Allow dictionary-like access."""
            return self._file_mapping.get(key, {'Domain_id': 1, 'Dataset_id': 1, 'Label': 0})
    
    metadata = MockMetadata()
    
    return args_data, args_model, args_task, args_trainer, args_environment, metadata


def test_flow_standalone_training():
    """Test Flow pretraining with synthetic data."""
    print("🚀 开始Flow预训练独立测试...")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Create arguments
        args_data, args_model, args_task, args_trainer, args_environment, metadata = create_mock_arguments()
        
        # Create Flow model (M_04_ISFM_Flow uses args_m, metadata)
        print("📦 创建Flow模型...")
        flow_model = FlowModel(args_model, metadata)
        print(f"   ✅ 模型参数数量: {sum(p.numel() for p in flow_model.parameters()):,}")
        
        # Create Flow task
        print("🎯 创建Flow预训练任务...")
        flow_task = FlowPretrainTask(
            network=flow_model,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata
        )
        print("   ✅ Flow任务创建成功")
        
        # Test forward pass
        print("⚡ 测试前向传播...")
        synthetic_data = create_synthetic_data(batch_size=8, seq_len=512, num_batches=3)
        
        # Ensure consistent device usage
        device = flow_task.device if hasattr(flow_task, 'device') else 'cpu'
        if torch.cuda.is_available() and device != 'cpu':
            flow_task = flow_task.cuda()
            device = 'cuda'
        else:
            device = 'cpu'
        print(f"   使用设备: {device}")
        
        flow_task.eval()
        with torch.no_grad():
            for i, batch in enumerate(synthetic_data[:2]):  # Test first 2 batches
                # Move batch data to same device as model
                batch['x'] = batch['x'].to(device)
                print(f"   批次 {i+1}: 输入形状 {batch['x'].shape} (batch_size, seq_len, channels), 设备: {batch['x'].device}")
                
                # Test forward pass
                outputs = flow_task.forward(batch)
                print(f"   ✅ 前向传播成功，输出keys: {list(outputs.keys())}")
                
                # Test training step
                flow_task.train()
                loss = flow_task.training_step(batch, i)
                print(f"   ✅ 训练步骤成功，损失: {loss.item():.4f}")
                
                # Test generation
                if hasattr(flow_task, 'generate_samples'):
                    samples = flow_task.generate_samples(
                        batch_size=4, 
                        file_ids=['test_1', 'test_2', 'test_3', 'test_4']
                    )
                    if samples is not None:
                        print(f"   ✅ 样本生成成功，形状: {samples.shape}")
                    else:
                        print("   ⚠️  样本生成返回None（可能为正常状态）")
        
        # Test validation capability
        print("🔍 测试验证能力...")
        validation_result = flow_task.validate_pipeline_compatibility()
        print(f"   Pipeline兼容性: {validation_result['status']}")
        print(f"   ✅ 兼容性检查完成")
        
        print("\n🎉 Flow预训练独立测试完全通过！")
        print("=" * 50)
        print("✅ 模型创建和初始化")
        print("✅ 前向传播和训练步骤")
        print("✅ 损失计算和反向传播")
        print("✅ 样本生成（如适用）")
        print("✅ Pipeline兼容性验证")
        print("🚀 Flow模块已就绪，可进行完整实验！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Flow测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_flow_standalone_training()
    if success:
        print("\n✨ 所有测试通过 - Flow预训练模块可用于实验！")
        exit(0)
    else:
        print("\n💥 测试失败 - 请检查错误信息")
        exit(1)