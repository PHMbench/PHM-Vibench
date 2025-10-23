#!/usr/bin/env python3
"""
测试配置系统加载
验证修正后的配置文件是否可以正确加载
"""

import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '.')

print("=" * 60)
print("⚙️ 配置系统测试")
print("=" * 60)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 配置文件路径
config_path = "script/unified_metric/configs/unified_experiments_1epoch_fixed.yaml"

print(f"\n📄 配置文件: {config_path}")

# 检查文件是否存在
if not os.path.exists(config_path):
    print(f"\n❌ 错误: 配置文件不存在")
    print(f"请先运行: python update_config_path.py")
    sys.exit(1)

try:
    # 导入配置加载器
    print(f"\n📦 导入配置系统...")
    from src.configs import load_config
    print(f"✅ 配置系统导入成功")

    # 加载配置
    print(f"\n📖 加载配置文件...")
    config = load_config(config_path)
    print(f"✅ 配置加载成功")

    # 显示配置摘要
    print(f"\n📋 配置摘要:")
    print(f"  - 项目名称: {config.environment.project}")
    print(f"  - 随机种子: {config.environment.seed}")
    print(f"  - 输出目录: {config.environment.output_dir}")

    print(f"\n📊 数据配置:")
    print(f"  - 数据目录: {config.data.data_dir}")
    print(f"  - 元数据文件: {config.data.metadata_file}")
    print(f"  - 批量大小: {config.data.batch_size}")
    print(f"  - 工作进程: {config.data.num_workers}")
    print(f"  - 窗口大小: {config.data.window_size}")

    print(f"\n🤖 模型配置:")
    print(f"  - 模型名称: {config.model.name}")
    print(f"  - 模型类型: {config.model.type}")
    print(f"  - 嵌入层: {config.model.embedding}")
    print(f"  - 骨干网络: {config.model.backbone}")
    print(f"  - 任务头: {config.model.task_head}")
    print(f"  - 模型维度: {config.model.d_model}")

    print(f"\n🎯 任务配置:")
    print(f"  - 任务名称: {config.task.name}")
    print(f"  - 任务类型: {config.task.type}")
    print(f"  - 目标系统ID: {config.task.target_system_id}")
    print(f"  - 训练轮数: {config.task.epochs}")
    print(f"  - 学习率: {config.task.lr}")
    print(f"  - 损失函数: {config.task.loss}")
    print(f"  - 对比权重: {getattr(config.task, 'contrast_weight', 0.1)}")

    # 检查数据目录是否存在
    data_dir = config.data.data_dir
    metadata_file = os.path.join(data_dir, config.data.metadata_file)

    print(f"\n🔍 验证数据路径:")
    print(f"  - 数据目录存在: {'✅' if os.path.exists(data_dir) else '❌'}")
    print(f"  - 元数据文件存在: {'✅' if os.path.exists(metadata_file) else '❌'}")

    # 显示修改信息（如果有）
    if hasattr(config, '_modification_info'):
        print(f"\n📝 配置修改信息:")
        info = config._modification_info
        print(f"  - 修改时间: {info.get('modified_at', '未知')}")
        print(f"  - 原始路径: {info.get('original_data_dir', '未知')}")
        print(f"  - 新路径: {info.get('new_data_dir', '未知')}")

    print(f"\n✅ 配置系统测试完成 - 所有检查通过!")

except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    print(f"请确保:")
    print(f"1. 在项目根目录执行此脚本")
    print(f"2. PYTHONPATH设置正确")
    print(f"3. 所有依赖已安装")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ 配置加载失败: {e}")
    print(f"\n详细信息:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 配置系统测试完成")
print("=" * 60)