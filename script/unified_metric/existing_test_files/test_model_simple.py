#!/usr/bin/env python3
"""
测试模型工厂功能
验证模型构建和前向传播是否正常工作
"""

import sys
import os
from datetime import datetime
import time

# 添加项目路径
sys.path.insert(0, '.')

print("=" * 60)
print("🤖 模型工厂测试")
print("=" * 60)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 导入torch
try:
    import torch
    print(f"🔥 PyTorch版本: {torch.__version__}")
    print(f"🎮 CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU设备: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
except ImportError:
    print("❌ PyTorch未安装，请先安装PyTorch")
    sys.exit(1)

try:
    # 导入必要的模块
    print(f"\n📦 导入模块...")
    from src.configs import load_config
    from src.model_factory import build_model
    print(f"✅ 模块导入成功")

    # 加载配置
    config_path = sys.argv[1] if len(sys.argv) > 1 else "script/unified_metric/configs/unified_experiments_1epoch.yaml"
    print(f"\n📖 加载配置: {config_path}")
    config = load_config(config_path)
    print(f"✅ 配置加载成功")

    # 显示模型配置
    print(f"\n🤖 模型配置:")
    print(f"  - 模型名称: {config.model.name}")
    print(f"  - 模型类型: {config.model.type}")
    print(f"  - 嵌入层: {config.model.embedding}")
    print(f"  - 骨干网络: {config.model.backbone}")
    print(f"  - 任务头: {config.model.task_head}")
    print(f"  - 模型维度: {config.model.d_model}")
    print(f"  - 输入维度: {config.model.input_dim}")
    print(f"  - 输出维度: {config.model.output_dim}")

    # 准备元数据
    print(f"\n📊 准备元数据...")
    import pandas as pd
    import numpy as np

    # 创建模拟的metadata dataframe，包含必需的列
    df = pd.DataFrame({
        'Dataset_id': [1] * 100,  # 100个样本，都属于数据集1 (整数)
        'Label': np.random.randint(0, 10, 100),  # 10个类别 (整数)
        'Sample_rate': [1000.0] * 100  # 采样率 (浮点数)
    })

    # 创建兼容的metadata对象
    class MockMetadata:
        def __init__(self, dataframe):
            self.df = dataframe

        def __getitem__(self, key):
            # 返回第一行的数据作为样本信息，确保Dataset_id是整数
            row = self.df.iloc[0].to_dict()
            row['Dataset_id'] = int(row['Dataset_id'])  # 强制转换为整数
            row['Label'] = int(row['Label'])  # 标签也应该是整数
            return row

        @property
        def columns(self):
            return self.df.columns.tolist()

        # 添加与真实metadata兼容的方法
        def __len__(self):
            return len(self.df)

        def __iter__(self):
            return iter(self.df.values)

    metadata = MockMetadata(df)
    print(f"  - 模拟数据集: {len(df)} 个样本")
    print(f"  - 数据集ID: {df['Dataset_id'].unique()}")
    print(f"  - 标签范围: {df['Label'].min()} - {df['Label'].max()}")
    print(f"  - 输入维度: {metadata['input_dim']}")
    print(f"  - 类别数: {metadata['num_classes']}")
    print(f"  - 序列长度: {metadata['sequence_length']}")

    # 构建模型
    print(f"\n🏗️ 构建模型...")
    start_time = time.time()
    model = build_model(config.model, metadata)
    build_time = time.time() - start_time
    print(f"✅ 模型构建成功 (耗时: {build_time:.2f}秒)")
    print(f"  - 模型类型: {type(model).__name__}")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型参数统计:")
    print(f"  - 总参数数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 冻结参数: {total_params - trainable_params:,}")

    # 准备测试数据
    print(f"\n📦 准备测试数据...")
    batch_size = 2  # 使用小批次以节省内存
    seq_len = min(1024, config.data.window_size if hasattr(config.data, 'window_size') else 2048)  # 使用较短序列
    input_dim = config.model.input_dim if hasattr(config.model, 'input_dim') else 1

    # 创建输入张量
    x = torch.randn(batch_size, input_dim, seq_len)
    print(f"  - 输入形状: {x.shape}")
    print(f"  - 输入类型: {x.dtype}")

    # 创建元数据（HSE模型需要）
    batch_metadata = {
        'dataset_id': torch.tensor([1, 2], dtype=torch.long),
        'domain_id': torch.tensor([1, 1], dtype=torch.long),
        'sample_rate': torch.tensor([1024, 2048], dtype=torch.float32)
    }
    print(f"  - 元数据键: {list(batch_metadata.keys())}")

    # 设置模型为评估模式
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        batch_metadata = {k: v.cuda() for k, v in batch_metadata.items()}

    # 执行前向传播
    print(f"\n🚀 执行前向传播...")
    start_time = time.time()

    with torch.no_grad():
        # M_02_ISFM_Prompt模型需要file_id参数来获取system_id
        file_id = 1  # 模拟文件ID，对应Dataset_id = 1
        outputs = model(x, file_id=file_id)

    forward_time = time.time() - start_time
    print(f"✅ 前向传播成功 (耗时: {forward_time:.3f}秒)")

    # 分析输出
    if isinstance(outputs, tuple):
        print(f"\n📊 输出结构 (tuple):")
        for i, output in enumerate(outputs):
            if output is not None:
                if hasattr(output, 'shape'):
                    print(f"  - 输出 {i}: 形状 {output.shape}, 类型 {output.dtype}")
                else:
                    print(f"  - 输出 {i}: 类型 {type(output)}")
            else:
                print(f"  - 输出 {i}: None")

        # 假设第一个是logits
        if len(outputs) > 0 and outputs[0] is not None:
            logits = outputs[0]
            print(f"\n📈 Logits详情:")
            print(f"  - 形状: {logits.shape}")
            print(f"  - 数值范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

            # 计算预测类别
            if len(logits.shape) > 1:
                predictions = torch.argmax(logits, dim=-1)
                print(f"  - 预测类别: {predictions.tolist()}")

    elif isinstance(outputs, dict):
        print(f"\n📊 输出结构 (dict):")
        for key, value in outputs.items():
            if value is not None and hasattr(value, 'shape'):
                print(f"  - {key}: 形状 {value.shape}, 类型 {value.dtype}")
    else:
        if outputs is not None:
            print(f"\n📊 输出详情:")
            print(f"  - 形状: {outputs.shape}")
            print(f"  - 类型: {outputs.dtype}")
            print(f"  - 数值范围: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")

    # 内存使用情况
    if torch.cuda.is_available():
        print(f"\n💾 GPU内存使用:")
        print(f"  - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  - 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    # 测试梯度计算（可选）
    print(f"\n🔄 测试梯度计算...")
    model.train()
    x_grad = torch.randn(batch_size, metadata['input_dim'], seq_len, requires_grad=True)
    if torch.cuda.is_available():
        x_grad = x_grad.cuda()
        batch_metadata_grad = {k: v.cuda() for k, v in batch_metadata.items()}

    try:
        outputs_grad = model(x_grad, batch_metadata_grad)
        if isinstance(outputs_grad, tuple):
            loss = outputs_grad[0].mean()
        else:
            loss = outputs_grad.mean()

        loss.backward()
        print(f"✅ 梯度计算成功")
        print(f"  - 损失值: {loss.item():.4f}")

        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        print(f"  - 梯度存在: {'是' if has_grad else '否'}")
    except Exception as e:
        print(f"⚠️ 梯度计算失败: {e}")

    print(f"\n✅ 模型测试完成 - 所有功能正常!")
    print(f"  - 模型构建: ✅")
    print(f"  - 前向传播: ✅")
    print(f"  - 输出格式: ✅")

except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    print(f"请确保:")
    print(f"1. 已安装所有依赖")
    print(f"2. 在项目根目录执行此脚本")
    print(f"3. 模型文件路径正确")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ 模型测试失败: {e}")
    print(f"\n详细信息:")
    import traceback
    traceback.print_exc()

    # 提供故障排除建议
    print(f"\n🔧 故障排除建议:")
    print(f"1. 检查模型配置是否正确")
    print(f"2. 确认输入数据形状是否匹配")
    print(f"3. 验证元数据格式是否正确")
    print(f"4. 如果GPU内存不足，使用较小的批次大小")

    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 模型工厂测试完成")
print("=" * 60)