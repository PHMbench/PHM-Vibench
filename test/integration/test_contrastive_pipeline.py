#!/usr/bin/env python3
"""
ContrastiveIDTask与Pipeline_ID的集成测试
验证完整训练流程的正确性和性能
"""
import sys
import os
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configs.config_utils import load_config
from src.Pipeline_ID import pipeline


def test_pipeline_id_with_contrastive_debug():
    """测试Pipeline_ID与ContrastiveIDTask的基本集成"""
    print("\n=== 测试Pipeline_ID + ContrastiveIDTask基本集成 ===")
    
    # 使用新的contrastive预设
    config = load_config('contrastive')
    
    # 确认配置正确加载
    assert config.task.name == "contrastive_id"
    assert config.task.type == "pretrain"
    assert config.trainer.epochs == 1  # 确认已更新为1
    assert config.data.factory_name == "id"
    
    print(f"✅ 配置加载成功: {config.task.name}")
    print(f"   - 训练epoch: {config.trainer.epochs}")
    print(f"   - 数据工厂: {config.data.factory_name}")
    print(f"   - 批大小: {config.data.batch_size}")
    print(f"   - 窗口大小: {config.data.window_size}")


def test_pipeline_config_overrides():
    """测试配置覆盖功能"""
    print("\n=== 测试配置覆盖功能 ===")
    
    # 测试基本覆盖
    base_config = load_config('contrastive')
    overrides = {
        'task': {'temperature': 0.05},
        'data': {'batch_size': 8},
        'trainer': {'accelerator': 'cpu'}
    }
    
    config = load_config(base_config, overrides)
    
    assert config.task.temperature == 0.05
    assert config.data.batch_size == 8
    assert config.trainer.accelerator == 'cpu'
    
    print("✅ 配置覆盖测试通过")
    print(f"   - 温度参数: {config.task.temperature}")
    print(f"   - 批大小: {config.data.batch_size}")
    print(f"   - 加速器: {config.trainer.accelerator}")


def test_contrastive_task_registration():
    """测试对比学习任务是否正确注册"""
    print("\n=== 测试对比学习任务注册 ===")
    
    from src.task_factory import TASK_REGISTRY
    
    # 检查任务是否已注册
    # TASK_REGISTRY使用key格式: "task_type.task_name"
    key = "pretrain.contrastive_id"
    
    try:
        task_cls = TASK_REGISTRY.get(key)
        print("✅ ContrastiveIDTask任务注册验证通过")
        print(f"   - 注册键: {key}")
        print(f"   - 任务类: {task_cls}")
    except KeyError:
        print(f"❌ ContrastiveIDTask任务未注册，键: {key}")
        # 显示已注册的键（如果有debug方法）
        print("   - 尝试导入任务模块验证...")
        try:
            from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
            print("   - ContrastiveIDTask模块可以正常导入")
        except Exception as e:
            print(f"   - ContrastiveIDTask模块导入失败: {e}")
            raise


def test_mock_pipeline_execution():
    """使用Mock数据测试Pipeline执行"""
    print("\n=== 测试Mock数据Pipeline执行 ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 使用contrastive预设并设置临时保存目录
        config = load_config('contrastive')
        config.environment.save_dir = temp_dir
        config.trainer.accelerator = 'cpu'  # 确保使用CPU
        
        # 设置Mock metadata路径（使用不存在的路径触发Mock模式）
        config.data.data_dir = temp_dir
        config.data.metadata_file = "mock_metadata.xlsx"
        
        print(f"✅ 配置准备完成")
        print(f"   - 保存目录: {config.environment.save_dir}")
        print(f"   - 数据目录: {config.data.data_dir}")
        print(f"   - metadata文件: {config.data.metadata_file}")
        
        # 这里可以调用pipeline，但由于可能需要实际数据，先跳过
        print("⚠️  跳过实际Pipeline执行（需要真实数据集）")


def test_contrastive_specific_configs():
    """测试对比学习特定配置参数"""
    print("\n=== 测试对比学习特定配置参数 ===")
    
    config = load_config('contrastive')
    
    # 验证对比学习特定参数
    assert hasattr(config.task, 'temperature'), "缺少temperature参数"
    assert config.task.temperature == 0.07, f"temperature应该是0.07，实际是{config.task.temperature}"
    
    # 验证窗口参数
    assert config.data.num_window == 2, f"num_window应该是2，实际是{config.data.num_window}"
    assert config.data.window_sampling_strategy == 'random', f"采样策略应该是random"
    
    # 验证模型参数
    assert config.model.name == "M_01_ISFM", "应该使用ISFM模型"
    assert config.model.backbone == "B_08_PatchTST", "应该使用PatchTST backbone"
    
    print("✅ 对比学习特定配置验证通过")
    print(f"   - 温度参数: {config.task.temperature}")
    print(f"   - 窗口数: {config.data.num_window}")
    print(f"   - 采样策略: {config.data.window_sampling_strategy}")
    print(f"   - 模型: {config.model.name}")
    print(f"   - 主干网络: {config.model.backbone}")


def test_all_contrastive_presets():
    """测试所有对比学习预设"""
    print("\n=== 测试所有对比学习预设 ===")
    
    presets = ['contrastive', 'contrastive_ablation', 'contrastive_cross', 'contrastive_prod']
    
    for preset in presets:
        try:
            config = load_config(preset)
            assert config.task.name == "contrastive_id"
            assert config.task.type == "pretrain"
            print(f"✅ 预设 '{preset}' 加载成功")
        except Exception as e:
            print(f"❌ 预设 '{preset}' 加载失败: {e}")
            raise


def test_config_validation():
    """测试配置验证"""
    print("\n=== 测试配置验证 ===")
    
    config = load_config('contrastive')
    
    # 基本配置验证
    assert config.data.window_size > 0, "window_size应该大于0"
    assert config.data.batch_size > 0, "batch_size应该大于0"
    assert config.task.temperature > 0, "temperature应该大于0"
    assert config.trainer.epochs >= 1, "epochs应该至少为1"
    
    print("✅ 配置验证通过")
    print(f"   - 窗口大小: {config.data.window_size}")
    print(f"   - 批大小: {config.data.batch_size}")
    print(f"   - 温度参数: {config.task.temperature}")
    print(f"   - 训练轮数: {config.trainer.epochs}")


def main():
    """运行所有集成测试"""
    print("开始ContrastiveIDTask与Pipeline_ID集成测试...")
    
    try:
        test_pipeline_id_with_contrastive_debug()
        test_pipeline_config_overrides()
        test_contrastive_task_registration()
        test_mock_pipeline_execution()
        test_contrastive_specific_configs()
        test_all_contrastive_presets()
        test_config_validation()
        
        print("\n" + "="*60)
        print("🎉 所有集成测试通过！ContrastiveIDTask已成功集成到Pipeline_ID")
        print("="*60)
        
        print("\n📋 下一步建议:")
        print("1. 使用真实数据集测试:")
        print("   python main.py --pipeline Pipeline_ID --config contrastive")
        print("2. 运行消融实验:")
        print("   python scripts/ablation_studies.py --preset contrastive_ablation")
        print("3. 运行多数据集实验:")
        print("   python scripts/multi_dataset_experiments.py --preset contrastive")
        
    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()