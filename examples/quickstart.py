#!/usr/bin/env python3
"""
PHM-Vibench 快速开始示例
============================

这个示例演示了如何在5分钟内上手PHM-Vibench，从最简配置开始运行第一个实验。

对PHM基础模型开发者的建议：
- 首先运行这个示例了解基本流程
- 然后参考 QUICKSTART.md 了解更多细节
- 最后查看 MODEL_INTERFACE.md 学习如何开发自定义模型

运行方式：
    python examples/quickstart.py

作者: PHM-Vibench Team
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.Pipeline_01_Fault_Diagnosis import pipeline
from src.utils.config_utils import save_config
import tempfile
import yaml


def create_minimal_config():
    """创建最简配置 - PHM基础模型开发者的起点"""
    config = {
        # 环境配置 - 控制实验行为
        'environment': {
            'WANDB_MODE': 'disabled',  # 禁用wandb，简化输出
            'VBENCH_HOME': str(project_root),
            'project': 'quickstart_demo',
            'seed': 42,  # 固定随机种子保证可复现
            'iterations': 1,  # 只运行1次迭代
            'wandb': False,
            'swanlab': False
        },
        
        # 数据配置 - 定义输入数据
        'data': {
            # 使用dummy数据进行快速演示
            'data_dir': str(project_root / 'data'),
            'metadata_file': 'metadata_dummy.csv',
            
            # 数据加载参数
            'batch_size': 16,        # 小批次用于快速训练
            'num_workers': 2,        # 较少worker避免内存问题
            'train_ratio': 0.7,
            'normalization': True,   # 标准化输入数据
            
            # 信号处理参数
            'window_size': 1024,     # 信号窗口长度
            'stride': 512,           # 窗口滑动步长
            'truncate_lenth': 2048,  # 最大信号长度
        },
        
        # 模型配置 - 选择模型架构
        'model': {
            # 使用简单的ResNet1D作为起点
            'name': 'ResNet1D',
            'type': 'CNN',
            
            # 模型超参数
            'depth': 18,
            'in_channels': 1,
            'num_classes': 4,        # CWRU数据集有4个类别
            'dropout': 0.1
        },
        
        # 任务配置 - 定义学习任务
        'task': {
            'name': 'classification',  # 故障分类任务
            'type': 'DG',             # Domain Generalization任务类型
            
            # 数据划分
            'target_system_id': [1],   # 目标系统ID
            'source_domain_id': [0, 2, 3],  # 源域ID
            'target_domain_id': [1],   # 目标域ID
            
            # 训练参数
            'loss': 'CE',             # 交叉熵损失
            'metrics': ['acc', 'f1'], # 评估指标
            'optimizer': 'adam',
            'lr': 0.001,              # 学习率
            'weight_decay': 0.0001,
            'epochs': 10,             # 只训练10个epoch用于演示
            
            # 早停参数
            'early_stopping': True,
            'es_patience': 5,
        },
        
        # 训练器配置 - 控制训练过程
        'trainer': {
            'name': 'Default_trainer',
            'num_epochs': 10,         # 快速训练
            'gpus': 1 if sys.platform != 'darwin' else 0,  # Mac使用CPU
            'device': 'cuda' if sys.platform != 'darwin' else 'cpu',
            'early_stopping': True,
            'patience': 5,
            'wandb': False,
            'pruning': False
        }
    }
    return config


def create_advanced_config():
    """创建进阶配置 - 展示ISFM基础模型的使用"""
    config = create_minimal_config()
    
    # 使用ISFM基础模型
    config['model'] = {
        'name': 'M_01_ISFM',
        'type': 'ISFM',
        
        # ISFM架构组件
        'embedding': 'E_01_HSE',      # 层次信号嵌入
        'backbone': 'B_08_PatchTST',  # Patch-based Transformer
        'task_head': 'H_01_Linear_cla', # 线性分类头
        
        # 模型超参数
        'input_dim': 1,
        'd_model': 64,                # 较小的模型用于快速演示
        'num_heads': 4,
        'num_layers': 2,
        'd_ff': 128,
        'dropout': 0.1,
        
        # Patch参数
        'patch_size_L': 32,
        'num_patches': 32,
        'output_dim': 64,
    }
    
    return config


def run_experiment(config, experiment_name="quickstart"):
    """运行实验并返回结果"""
    print(f"\n{'='*60}")
    print(f"🚀 开始运行 {experiment_name} 实验")
    print(f"{'='*60}")
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, allow_unicode=True)
        temp_config_path = f.name
    
    try:
        # 创建参数对象
        args = argparse.Namespace(
            config_path=temp_config_path,
            fs_config_path=None,
            notes=f"{experiment_name} 快速开始示例"
        )
        
        # 运行pipeline
        results = pipeline(args)
        
        print(f"\n✅ {experiment_name} 实验完成!")
        if results:
            print(f"实验结果: {results}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 实验出现错误: {e}")
        print("💡 常见解决方案:")
        print("   1. 确保数据文件存在: data/metadata_dummy.csv")
        print("   2. 检查CUDA环境（如果使用GPU）")
        print("   3. 确保所有依赖已安装: pip install -r requirements.txt")
        raise
    finally:
        # 清理临时文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def main():
    """主函数 - 运行快速开始示例"""
    print("🎯 PHM-Vibench 快速开始")
    print("=" * 60)
    print("""
这个示例将演示：
1. 📝 创建最简配置
2. 🏃‍♂️ 运行第一个实验（ResNet1D + 分类任务）
3. 🚀 运行进阶实验（ISFM基础模型）
4. 📊 理解实验结果

估计耗时: 5-10分钟
""")
    
    input("按回车键开始实验...")
    
    try:
        # 实验1: 基础实验
        print("\n🔰 实验1: 基础实验 (ResNet1D)")
        print("=" * 40)
        basic_config = create_minimal_config()
        
        print("📋 配置概览:")
        print(f"   - 模型: {basic_config['model']['name']} ({basic_config['model']['type']})")
        print(f"   - 任务: {basic_config['task']['name']}")
        print(f"   - 训练轮数: {basic_config['trainer']['num_epochs']}")
        print(f"   - 批次大小: {basic_config['data']['batch_size']}")
        
        basic_results = run_experiment(basic_config, "基础ResNet1D")
        
        # 实验2: 进阶实验
        print(f"\n🚀 实验2: 进阶实验 (ISFM基础模型)")
        print("=" * 40)
        advanced_config = create_advanced_config()
        
        print("📋 配置概览:")
        print(f"   - 模型: {advanced_config['model']['name']} (ISFM基础模型)")
        print(f"   - 嵌入: {advanced_config['model']['embedding']}")
        print(f"   - 骨干网络: {advanced_config['model']['backbone']}")
        print(f"   - 任务头: {advanced_config['model']['task_head']}")
        
        advanced_results = run_experiment(advanced_config, "进阶ISFM")
        
        # 总结
        print(f"\n🎉 所有实验完成!")
        print("=" * 60)
        print("📈 结果总结:")
        print(f"   - 基础实验 (ResNet1D): {basic_results if basic_results else '请查看训练日志'}")
        print(f"   - 进阶实验 (ISFM): {advanced_results if advanced_results else '请查看训练日志'}")
        
        print(f"\n📚 下一步学习:")
        print("   1. 查看实验结果保存在: save/ 目录下")
        print("   2. 阅读 docs/QUICKSTART.md 了解更多配置选项")
        print("   3. 参考 docs/MODEL_INTERFACE.md 学习自定义模型开发")
        print("   4. 运行其他示例: examples/basic_classification.py")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("\n💡 解决方案:")
        print("   1. 确保在项目根目录运行: cd PHM-Vibench")
        print("   2. 创建dummy数据: python examples/create_dummy_data.py")
        print("   3. 或使用真实数据集，修改配置中的 metadata_file")
        
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("\n💡 解决方案:")
        print("   1. 安装依赖: pip install -r requirements.txt")
        print("   2. 检查Python路径设置")
        print("   3. 确保在正确的虚拟环境中")
        
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        print("\n💡 获取帮助:")
        print("   1. 查看完整错误日志")
        print("   2. 参考 docs/FAQ.md")
        print("   3. 在GitHub提issue: https://github.com/your-repo/issues")


if __name__ == '__main__':
    main()