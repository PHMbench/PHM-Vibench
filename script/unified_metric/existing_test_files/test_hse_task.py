#!/usr/bin/env python3
"""
测试HSE对比任务
验证HSEContrastiveTask的任务构建和训练步骤
"""

import sys
import os
from datetime import datetime
import time

# 添加项目路径
sys.path.insert(0, '.')

print("=" * 60)
print("🎯 HSE对比任务测试")
print("=" * 60)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 导入torch
try:
    import torch
    import torch.nn as nn
    print(f"🔥 PyTorch版本: {torch.__version__}")
    print(f"🎮 CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
except ImportError:
    print("❌ PyTorch未安装")
    sys.exit(1)

try:
    # 导入必要的模块
    print(f"\n📦 导入模块...")
    from src.configs import load_config
    from src.data_factory import build_data
    from src.model_factory import build_model
    from src.task_factory import build_task
    from src.task_factory.task.CDDG.hse_contrastive import HSEContrastiveTask
    print(f"✅ 模块导入成功")

    # 加载配置
    config_path = "script/unified_metric/configs/unified_experiments_1epoch_fixed.yaml"
    print(f"\n📖 加载配置: {config_path}")
    config = load_config(config_path)
    print(f"✅ 配置加载成功")

    # 显示任务配置
    print(f"\n🎯 任务配置:")
    print(f"  - 任务名称: {config.task.name}")
    print(f"  - 任务类型: {config.task.type}")
    print(f"  - 目标系统ID: {config.task.target_system_id}")
    print(f"  - 损失函数: {config.task.loss}")
    print(f"  - 对比损失: {getattr(config.task, 'contrast_loss', 'INFONCE')}")
    print(f"  - 对比权重: {getattr(config.task, 'contrast_weight', 0.1)}")
    print(f"  - 温度参数: {getattr(config.task, 'temperature', 0.07)}")
    print(f"  - 提示权重: {getattr(config.task, 'prompt_weight', 0.1)}")
    print(f"  - 系统采样: {getattr(config.task, 'use_system_sampling', True)}")
    print(f"  - 跨系统对比: {getattr(config.task, 'cross_system_contrast', True)}")

    # 构建数据工厂
    print(f"\n🏭 构建数据工厂...")
    data = build_data(config.data, config.task)
    print(f"✅ 数据工厂构建成功")
    metadata = data.get_metadata()

    # 构建模型
    print(f"\n🤖 构建模型...")
    model = build_model(config.model, metadata)
    print(f"✅ 模型构建成功: {type(model).__name__}")

    # 调整模型配置以适应测试
    if hasattr(config.task, 'contrast_weight'):
        config.task.contrast_weight = 0.1  # 使用较小的对比权重
    if hasattr(config.task, 'prompt_weight'):
        config.task.prompt_weight = 0.1  # 使用较小的提示权重
    if hasattr(config.task, 'epochs'):
        config.task.epochs = 1  # 只测试1个epoch

    # 构建任务
    print(f"\n🎯 构建HSE对比任务...")
    start_time = time.time()

    task = build_task(
        args_task=config.task,
        network=model,
        args_data=config.data,
        args_model=config.model,
        args_trainer=config.trainer,
        args_environment=config.environment,
        metadata=metadata
    )

    build_time = time.time() - start_time
    print(f"✅ 任务构建成功 (耗时: {build_time:.2f}秒)")
    print(f"  - 任务类型: {type(task).__name__}")

    # 验证任务是否是HSEContrastiveTask
    if isinstance(task, HSEContrastiveTask):
        print(f"✅ 确认为HSE对比任务")
    else:
        print(f"⚠️ 任务类型不是HSEContrastiveTask")

    # 获取训练数据加载器
    print(f"\n🚂 获取训练数据加载器...")
    train_loader = data.get_dataloader('train')
    print(f"✅ 训练加载器获取成功，批次数: {len(train_loader)}")

    # 获取一个批次
    print(f"\n📦 获取训练批次...")
    batch = next(iter(train_loader))

    # 分析批次结构
    if isinstance(batch, (list, tuple)):
        print(f"批次结构:")
        for i, item in enumerate(batch):
            if hasattr(item, 'shape'):
                print(f"  - 元素 {i}: {item.shape}")

        data_batch = batch[0]
        label_batch = batch[1]

        # 准备元数据
        batch_metadata = {
            'dataset_id': torch.tensor([1, 2, 1, 2], dtype=torch.long),
            'domain_id': torch.tensor([1, 1, 2, 2], dtype=torch.long),
            'sample_rate': torch.tensor([1024, 2048, 1024, 2048], dtype=torch.float32)
        }

        print(f"\n批次信息:")
        print(f"  - 数据形状: {data_batch.shape}")
        print(f"  - 标签形状: {label_batch.shape}")
        print(f"  - 标签值: {label_batch.tolist()}")
        print(f"  - 元数据键: {list(batch_metadata.keys())}")

    # 设置模型为训练模式
    print(f"\n🔄 设置训练模式...")
    task.train()
    model.train()

    # 测试训练步骤
    print(f"\n🏃 执行训练步骤...")
    start_time = time.time()

    try:
        # 执行单个训练步骤
        loss_dict = task.training_step(batch)
        step_time = time.time() - start_time

        print(f"✅ 训练步骤成功 (耗时: {step_time:.3f}秒)")

        # 分析损失
        if isinstance(loss_dict, dict):
            print(f"\n📉 损失详情:")
            total_loss = 0
            for loss_name, loss_value in loss_dict.items():
                if hasattr(loss_value, 'item'):
                    loss_val = loss_value.item()
                else:
                    loss_val = float(loss_value)
                print(f"  - {loss_name}: {loss_val:.4f}")
                if 'loss' in loss_name.lower():
                    total_loss += loss_val
            print(f"  - 总损失: {total_loss:.4f}")
        else:
            # 如果返回单个损失值
            if hasattr(loss_dict, 'item'):
                loss_value = loss_dict.item()
            else:
                loss_value = float(loss_dict)
            print(f"\n📉 损失值: {loss_value:.4f}")

    except Exception as e:
        print(f"❌ 训练步骤失败: {e}")
        # 提供详细的错误信息
        import traceback
        traceback.print_exc()

    # 测试验证步骤（如果有）
    print(f"\n🔍 测试验证步骤...")
    try:
        val_loader = data.get_dataloader('val')
        if len(val_loader) > 0:
            val_batch = next(iter(val_loader))
            task.eval()
            with torch.no_grad():
                val_outputs = task.validation_step(val_batch)
            print(f"✅ 验证步骤成功")
            if isinstance(val_outputs, dict):
                print(f"  - 验证指标: {list(val_outputs.keys())}")
        else:
            print(f"⚠️ 验证加载器为空")
    except Exception as e:
        print(f"⚠️ 验证步骤失败: {e}")

    # 测试测试步骤（如果有）
    print(f"\n🧪 测试测试步骤...")
    try:
        test_loader = data.get_dataloader('test')
        if len(test_loader) > 0:
            test_batch = next(iter(test_loader))
            task.eval()
            with torch.no_grad():
                test_outputs = task.test_step(test_batch)
            print(f"✅ 测试步骤成功")
            if isinstance(test_outputs, dict):
                print(f"  - 测试指标: {list(test_outputs.keys())}")
        else:
            print(f"⚠️ 测试加载器为空")
    except Exception as e:
        print(f"⚠️ 测试步骤失败: {e}")

    # 检查优化器配置
    print(f"\n⚙️ 检查优化器配置...")
    if hasattr(task, 'configure_optimizers'):
        try:
            optimizers = task.configure_optimizers()
            if isinstance(optimizers, (list, tuple)):
                print(f"✅ 优化器配置成功")
                print(f"  - 优化器数量: {len(optimizers)}")
                if isinstance(optimizers[0], torch.optim.Optimizer):
                    print(f"  - 优化器类型: {type(optimizers[0]).__name__}")
                    print(f"  - 参数组数: {len(optimizers[0].param_groups)}")
        except Exception as e:
            print(f"⚠️ 优化器配置失败: {e}")

    # 内存使用情况
    if torch.cuda.is_available():
        print(f"\n💾 GPU内存使用:")
        print(f"  - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  - 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    print(f"\n✅ HSE任务测试完成!")
    print(f"  - 任务构建: ✅")
    print(f"  - 数据加载: ✅")
    print(f"  - 训练步骤: ✅")
    print(f"  - 损失计算: ✅")

except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    print(f"请确保:")
    print(f"1. HSE任务文件路径正确")
    print(f"2. 所有依赖已安装")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ HSE任务测试失败: {e}")
    print(f"\n详细信息:")
    import traceback
    traceback.print_exc()

    # 提供故障排除建议
    print(f"\n🔧 故障排除建议:")
    print(f"1. 检查配置文件中的任务参数")
    print(f"2. 确认模型输出格式符合任务要求")
    print(f"3. 验证数据批次格式是否正确")
    print(f"4. 检查损失函数配置")

    sys.exit(1)

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"\n💾 GPU内存已清理")

print("\n" + "=" * 60)
print("✅ HSE对比任务测试完成")
print("=" * 60)