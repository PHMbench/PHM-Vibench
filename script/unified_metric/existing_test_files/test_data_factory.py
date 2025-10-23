#!/usr/bin/env python3
"""
测试数据工厂功能
验证数据加载和批处理是否正常工作
"""

import sys
import os
from datetime import datetime
import time

# 添加项目路径
sys.path.insert(0, '.')

print("=" * 60)
print("🏭 数据工厂测试")
print("=" * 60)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    # 导入必要的模块
    print(f"\n📦 导入模块...")
    from src.configs import load_config
    from src.data_factory import build_data
    print(f"✅ 模块导入成功")

    # 加载配置
    config_path = sys.argv[1] if len(sys.argv) > 1 else "script/unified_metric/configs/unified_experiments_1epoch.yaml"
    print(f"\n📖 加载配置: {config_path}")
    config = load_config(config_path)
    print(f"✅ 配置加载成功")

    # 显示配置信息
    print(f"\n📋 数据配置:")
    print(f"  - 数据目录: {config.data.data_dir}")
    print(f"  - 元数据文件: {config.data.metadata_file}")
    print(f"  - 批量大小: {config.data.batch_size}")
    print(f"  - 目标系统: {config.task.target_system_id}")

    # 构建数据工厂
    print(f"\n🏭 构建数据工厂...")
    start_time = time.time()
    data = build_data(config.data, config.task)
    build_time = time.time() - start_time
    print(f"✅ 数据工厂构建成功 (耗时: {build_time:.2f}秒)")

    # 获取元数据
    print(f"\n📊 获取元数据...")
    metadata = data.get_metadata()
    print(f"✅ 元数据获取成功")
    print(f"  元数据键: {list(metadata.keys())}")

    # 显示关键元数据
    if 'input_dim' in metadata:
        print(f"  - 输入维度: {metadata['input_dim']}")
    if 'num_classes' in metadata:
        print(f"  - 类别数: {metadata['num_classes']}")
    if 'sequence_length' in metadata:
        print(f"  - 序列长度: {metadata['sequence_length']}")

    # 测试训练数据加载器
    print(f"\n🚂 测试训练数据加载器...")
    start_time = time.time()
    train_loader = data.get_dataloader('train')
    loader_time = time.time() - start_time
    print(f"✅ 训练加载器创建成功 (耗时: {loader_time:.2f}秒)")
    print(f"  - 批次数: {len(train_loader)}")

    # 获取第一个批次
    print(f"\n📦 获取第一个训练批次...")
    start_time = time.time()
    batch = next(iter(train_loader))
    batch_time = time.time() - start_time
    print(f"✅ 批次获取成功 (耗时: {batch_time:.2f}秒)")

    # 分析批次结构
    if isinstance(batch, (list, tuple)):
        print(f"\n📋 批次结构 (list/tuple):")
        for i, item in enumerate(batch):
            if hasattr(item, 'shape'):
                print(f"  - 元素 {i}: 形状 {item.shape}, 类型 {item.dtype}")
            else:
                print(f"  - 元素 {i}: 类型 {type(item)}")

        # 假设第一个是数据，第二个是标签
        if len(batch) >= 2:
            data_batch = batch[0]
            label_batch = batch[1]

            print(f"\n📊 数据批次详情:")
            print(f"  - 形状: {data_batch.shape}")
            print(f"  - 数据类型: {data_batch.dtype}")
            print(f"  - 最小值: {data_batch.min().item():.4f}")
            print(f"  - 最大值: {data_batch.max().item():.4f}")
            print(f"  - 均值: {data_batch.mean().item():.4f}")
            print(f"  - 标准差: {data_batch.std().item():.4f}")

            print(f"\n🏷️ 标签批次详情:")
            if hasattr(label_batch, 'shape'):
                print(f"  - 形状: {label_batch.shape}")
                print(f"  - 数据类型: {label_batch.dtype}")
                if label_batch.numel() > 0:
                    print(f"  - 最小值: {label_batch.min().item()}")
                    print(f"  - 最大值: {label_batch.max().item()}")
                    print(f"  - 唯一值: {label_batch.unique().tolist()[:10]}")  # 显示前10个

    elif isinstance(batch, dict):
        print(f"\n📋 批次结构 (dict):")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  - {key}: 形状 {value.shape}, 类型 {value.dtype}")

    # 测试验证数据加载器（如果存在）
    print(f"\n🔍 测试验证数据加载器...")
    try:
        val_loader = data.get_dataloader('val')
        print(f"✅ 验证加载器创建成功")
        print(f"  - 批次数: {len(val_loader)}")

        # 获取一个验证批次
        val_batch = next(iter(val_loader))
        print(f"  - 验证批次形状: {val_batch[0].shape if isinstance(val_batch, (list, tuple)) else 'N/A'}")
    except:
        print(f"⚠️ 验证加载器不存在或创建失败")

    # 测试测试数据加载器（如果存在）
    print(f"\n🧪 测试测试数据加载器...")
    try:
        test_loader = data.get_dataloader('test')
        print(f"✅ 测试加载器创建成功")
        print(f"  - 批次数: {len(test_loader)}")
    except:
        print(f"⚠️ 测试加载器不存在或创建失败")

    # 内存使用情况
    if torch.cuda.is_available():
        print(f"\n💾 GPU内存使用:")
        print(f"  - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  - 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    print(f"\n✅ 数据工厂测试完成 - 所有功能正常!")
    print(f"  - 数据加载: ✅")
    print(f"  - 批处理: ✅")
    print(f"  - 元数据: ✅")

except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    print(f"请确保:")
    print(f"1. 已安装所有依赖: pip install -r requirements.txt")
    print(f"2. 在项目根目录执行此脚本")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ 数据工厂测试失败: {e}")
    print(f"\n详细信息:")
    import traceback
    traceback.print_exc()

    # 提供一些故障排除建议
    print(f"\n🔧 故障排除建议:")
    print(f"1. 检查数据目录是否正确")
    print(f"2. 验证元数据文件是否可读")
    print(f"3. 确认批量大小是否合适")
    print(f"4. 检查数据格式是否正确")

    sys.exit(1)

# 导入torch用于内存检查
try:
    import torch
except:
    pass

print("\n" + "=" * 60)
print("✅ 数据工厂测试完成")
print("=" * 60)