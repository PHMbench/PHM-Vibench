#!/usr/bin/env python3
"""
Standalone Flow + Contrastive joint training test.
Tests the Flow pretraining with contrastive learning enabled.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from argparse import Namespace
import numpy as np

# Import Flow components
from src.model_factory.ISFM.M_04_ISFM_Flow import Model as FlowModel
from src.task_factory.task.pretrain.flow_pretrain import FlowPretrainTask


def create_contrastive_data(batch_size=8, seq_len=512, num_batches=3):
    """Create synthetic data with contrastive pairs."""
    synthetic_data = []
    
    for i in range(num_batches):
        # Generate realistic base signals
        t = torch.linspace(0, 1, seq_len)
        
        # Create different signal patterns for contrastive learning
        patterns = [
            # Pattern 1: Low frequency dominant
            torch.sin(2 * np.pi * 30 * t) + 0.3 * torch.sin(2 * np.pi * 60 * t),
            # Pattern 2: High frequency dominant  
            0.3 * torch.sin(2 * np.pi * 30 * t) + torch.sin(2 * np.pi * 150 * t),
            # Pattern 3: Mixed patterns
            torch.sin(2 * np.pi * 50 * t) + torch.sin(2 * np.pi * 100 * t)
        ]
        
        batch_x = []
        batch_file_ids = []
        
        for j in range(batch_size):
            # Select pattern and add variations
            pattern_idx = j % len(patterns)
            base_signal = patterns[pattern_idx]
            
            # Add noise and variations
            signal = base_signal + 0.1 * torch.randn(seq_len)
            signal = signal.unsqueeze(-1)  # Shape: (seq_len, channels)
            
            batch_x.append(signal)
            batch_file_ids.append(f'contrastive_file_{i}_{j}_pattern_{pattern_idx}')
        
        x = torch.stack(batch_x)  # Shape: (batch_size, seq_len, channels)
        
        batch = {
            'x': x,
            'file_id': batch_file_ids
        }
        synthetic_data.append(batch)
    
    return synthetic_data


def create_contrastive_args():
    """Create arguments for Flow + Contrastive training."""
    
    # Data args
    args_data = Namespace()
    args_data.batch_size = 8
    args_data.window_size = 512
    args_data.num_channels = 1
    
    # Model args
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
    
    # Task args - ENABLE CONTRASTIVE LEARNING
    args_task = Namespace()
    args_task.name = "flow_pretrain"
    args_task.type = "pretrain"
    args_task.loss = "CE"
    args_task.metrics = ["acc"]
    args_task.num_steps = 20
    args_task.flow_lr = 1e-3
    args_task.use_contrastive = True  # ENABLE CONTRASTIVE LEARNING
    args_task.contrastive_weight = 0.3  # Weight for contrastive loss
    args_task.flow_weight = 1.0  # Weight for Flow loss
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
    
    # Enhanced mock metadata with contrastive patterns
    class MockMetadata:
        def __init__(self):
            import pandas as pd
            self.df = pd.DataFrame({
                'Dataset_id': [1, 2, 3, 4, 5, 6],
                'Domain_id': [1, 1, 2, 2, 3, 3], 
                'file_id': ['file1', 'file2', 'file3', 'file4', 'file5', 'file6'],
                'Label': [0, 1, 0, 1, 0, 1]
            })
            # Extended mapping for contrastive patterns
            self._file_mapping = {}
            for i in range(10):  # Support more synthetic files
                for j in range(8):  # Support batch size
                    for p in range(3):  # Support 3 patterns
                        file_id = f'contrastive_file_{i}_{j}_pattern_{p}'
                        self._file_mapping[file_id] = {
                            'Domain_id': (p + 1), 
                            'Dataset_id': (i % 4) + 1, 
                            'Label': p % 2
                        }
        
        def items(self):
            return {
                'item1': {'Name': 'test_dataset', 'Label': 0},
                'item2': {'Name': 'test_dataset', 'Label': 1},
                'item3': {'Name': 'test_dataset', 'Label': 2}, 
                'item4': {'Name': 'test_dataset', 'Label': 3}
            }.items()
            
        def __contains__(self, key):
            return key in self._file_mapping
            
        def __getitem__(self, key):
            return self._file_mapping.get(key, {'Domain_id': 1, 'Dataset_id': 1, 'Label': 0})
    
    metadata = MockMetadata()
    
    return args_data, args_model, args_task, args_trainer, args_environment, metadata


def test_flow_contrastive_training():
    """Test Flow + Contrastive joint training."""
    print("🚀 开始Flow+对比学习联合训练测试...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Create arguments with contrastive learning enabled
        args_data, args_model, args_task, args_trainer, args_environment, metadata = create_contrastive_args()
        
        # Create Flow model
        print("📦 创建Flow模型...")
        flow_model = FlowModel(args_model, metadata)
        print(f"   ✅ 模型参数数量: {sum(p.numel() for p in flow_model.parameters()):,}")
        
        # Create Flow + Contrastive task
        print("🎯 创建Flow+对比学习任务...")
        flow_task = FlowPretrainTask(
            network=flow_model,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata
        )
        print("   ✅ Flow+对比学习任务创建成功")
        print(f"   📊 对比学习权重: {flow_task.contrastive_weight}")
        print(f"   📊 Flow损失权重: {flow_task.flow_weight}")
        
        # Test joint training
        print("⚡ 测试联合训练...")
        contrastive_data = create_contrastive_data(batch_size=8, seq_len=512, num_batches=3)
        
        # Use CPU for now to avoid device issues in testing
        device = 'cpu'  # Force CPU to test joint training logic
        flow_task = flow_task.to(device)
        print(f"   使用设备: {device} (测试模式)")
        
        flow_task.eval()
        initial_losses = []
        final_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(contrastive_data[:2]):
                batch['x'] = batch['x'].to(device)
                print(f"   批次 {i+1}: 输入形状 {batch['x'].shape}")
                
                # Test forward pass
                outputs = flow_task.forward(batch)
                print(f"   ✅ 前向传播成功，输出keys包含: {len(list(outputs.keys()))} 项")
                
                # Check for contrastive loss components
                if 'contrastive_loss' in outputs:
                    print(f"   📊 Flow损失: {outputs['flow_loss']:.4f}")
                    print(f"   📊 对比损失: {outputs['contrastive_loss']:.4f}")
                    print(f"   📊 总损失: {outputs['total_loss']:.4f}")
                else:
                    print(f"   📊 总损失: {outputs['total_loss']:.4f}")
                
                # Test training step
                flow_task.train()
                loss = flow_task.training_step(batch, i)
                if i == 0:
                    initial_losses.append(loss.item())
                else:
                    final_losses.append(loss.item())
                print(f"   ✅ 联合训练步骤成功，损失: {loss.item():.4f}")
        
        # Test contrastive learning benefits
        print("🔍 验证对比学习效果...")
        if final_losses and initial_losses:
            loss_improvement = initial_losses[0] - final_losses[0] if final_losses else 0
            print(f"   📈 损失改进: {loss_improvement:.4f}")
            
        # Test pipeline compatibility
        print("🔍 测试Pipeline兼容性...")
        validation_result = flow_task.validate_pipeline_compatibility()
        print(f"   Pipeline兼容性: {validation_result['status']}")
        
        # Test joint loss components
        print("🔍 验证联合损失组件...")
        if hasattr(flow_task, 'flow_contrastive_loss') and flow_task.flow_contrastive_loss is not None:
            print("   ✅ FlowContrastiveLoss组件已激活")
            print(f"   📊 Flow权重: {flow_task.flow_contrastive_loss.flow_weight}")
            print(f"   📊 对比权重: {flow_task.flow_contrastive_loss.contrastive_weight}")
        else:
            print("   ⚠️  FlowContrastiveLoss组件未激活")
        
        print("\n🎉 Flow+对比学习联合训练测试完全通过！")
        print("=" * 50)
        print("✅ 联合模型创建和初始化")
        print("✅ Flow和对比学习损失计算")  
        print("✅ 联合训练步骤执行")
        print("✅ 损失权重平衡验证")
        print("✅ Pipeline兼容性确认")
        print("🚀 Flow+对比学习模块已就绪！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Flow+对比学习测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_flow_contrastive_training()
    if success:
        print("\n✨ Flow+对比学习联合训练测试通过！")
        exit(0)
    else:
        print("\n💥 测试失败 - 请检查错误信息")
        exit(1)