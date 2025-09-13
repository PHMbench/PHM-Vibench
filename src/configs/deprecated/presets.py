"""
PHM-Vibench配置预设集合
======================

提供面向不同使用场景的配置模板：
- 🚀 quickstart: 5分钟快速上手
- 🏗️ basic: 基础研究配置
- 🧠 isfm: ISFM基础模型配置 
- 🔬 research: 深度研究配置
- 🏭 production: 生产环境配置
- 📊 benchmark: 基准测试配置

使用方式：
    from src.configs.presets import get_preset_config
    
    config = get_preset_config("quickstart", 
                              model__d_model=256,
                              trainer__num_epochs=100)

作者: PHM-Vibench Team
"""

from typing import Dict, Any, Optional
from .config_schema import (
    PHMConfig, 
    EnvironmentConfig,
    DataConfig, 
    ModelConfig,
    TaskConfig,
    TrainerConfig
)


# ==================== 预设配置定义 ====================

def get_quickstart_config(**overrides) -> PHMConfig:
    """
    快速开始配置 - 5分钟上手PHM-Vibench
    
    特点：
    - 使用简单的ResNet1D模型
    - 小数据集和短训练时间
    - 禁用复杂功能，专注核心流程
    """
    return PHMConfig(
        environment=EnvironmentConfig(
            experiment_name="quickstart_demo",
            project="phm_quickstart", 
            notes="快速开始示例，展示基本功能",
            seed=42,
            iterations=1,
            wandb=False,
            swanlab=False,
            WANDB_MODE="disabled"
        ),
        data=DataConfig(
            data_dir="./data",
            metadata_file="metadata_dummy.csv",  # 使用dummy数据
            batch_size=16,  # 小批次用于快速运行
            num_workers=2,
            pin_memory=True,
            train_ratio=0.7,
            normalization=True,
            window_size=512,  # 较小窗口
            stride=256
        ),
        model=ModelConfig(
            name="ResNet1D",
            type="CNN",
            input_dim=1,
            num_classes=4,
            depth=18,
            in_channels=1,
            dropout=0.1
        ),
        task=TaskConfig(
            name="classification",
            type="DG",
            target_system_id=[1],
            epochs=10,  # 快速训练
            lr=0.001,
            optimizer="adam",
            loss="CE",
            metrics=["acc"],
            early_stopping=True,
            es_patience=5
        ),
        trainer=TrainerConfig(
            name="Default_trainer",
            num_epochs=10,
            gpus=1,
            device="auto",
            early_stopping=True,
            patience=5,
            wandb=False,
            mixed_precision=False  # 简化设置
        ),
        **overrides
    )


def get_basic_config(**overrides) -> PHMConfig:
    """
    基础配置 - 标准研究设置
    
    特点：
    - 平衡的参数设置
    - 适中的训练时间
    - 包含常用功能
    """
    return PHMConfig(
        environment=EnvironmentConfig(
            experiment_name="basic_experiment",
            project="phm_basic",
            notes="基础实验配置",
            seed=42,
            iterations=1,
            wandb=True,  # 启用实验跟踪
            WANDB_MODE="online"
        ),
        data=DataConfig(
            data_dir="./data",
            metadata_file="metadata.xlsx",
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            train_ratio=0.7,
            normalization=True,
            window_size=1024,
            stride=512
        ),
        model=ModelConfig(
            name="ResNet1D", 
            type="CNN",
            input_dim=1,
            num_classes=10,
            depth=18,
            in_channels=1,
            dropout=0.1
        ),
        task=TaskConfig(
            name="classification",
            type="DG", 
            epochs=50,
            lr=0.001,
            weight_decay=0.0001,
            optimizer="adam",
            loss="CE",
            metrics=["acc", "f1"],
            scheduler=True,
            scheduler_type="step",
            step_size=20,
            gamma=0.5,
            early_stopping=True,
            es_patience=10
        ),
        trainer=TrainerConfig(
            name="Default_trainer",
            num_epochs=50,
            gpus=1,
            device="auto",
            mixed_precision=True,  # 启用混合精度
            gradient_clip_val=1.0,
            early_stopping=True,
            patience=10,
            wandb=True,
            save_top_k=3
        ),
        **overrides
    )


def get_isfm_config(**overrides) -> PHMConfig:
    """
    ISFM基础模型配置 - 工业信号基础模型
    
    特点：
    - 使用ISFM架构
    - Transformer骨干网络
    - 多任务头支持
    """
    return PHMConfig(
        environment=EnvironmentConfig(
            experiment_name="isfm_experiment",
            project="phm_isfm",
            notes="ISFM基础模型实验",
            seed=42,
            iterations=1,
            wandb=True,
            WANDB_MODE="online"
        ),
        data=DataConfig(
            data_dir="./data",
            metadata_file="metadata.xlsx", 
            batch_size=32,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            train_ratio=0.7,
            normalization=True,
            window_size=1024,
            stride=512
        ),
        model=ModelConfig(
            name="M_01_ISFM",  # 推荐的ISFM版本
            type="ISFM",
            
            # ISFM组件
            embedding="E_01_HSE",
            backbone="B_08_PatchTST", 
            task_head="H_01_Linear_cla",
            
            # 模型参数
            input_dim=1,
            d_model=128,
            num_heads=8,
            num_layers=6,
            d_ff=512,
            dropout=0.1,
            
            # Patch参数
            patch_size_L=16,
            patch_size_C=1,
            num_patches=64,
            output_dim=128
        ),
        task=TaskConfig(
            name="classification",
            type="DG",
            epochs=100,  # ISFM通常需要更多训练
            lr=0.0001,   # 较小学习率
            weight_decay=0.0001,
            optimizer="adam",
            loss="CE",
            metrics=["acc", "f1", "precision", "recall"],
            scheduler=True,
            scheduler_type="cosine",
            early_stopping=True,
            es_patience=20
        ),
        trainer=TrainerConfig(
            name="Default_trainer",
            num_epochs=100,
            gpus=1,
            device="auto",
            mixed_precision=True,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            early_stopping=True,
            patience=20,
            wandb=True,
            save_top_k=5,
            monitor_metric="val_acc",
            mode="max"
        ),
        **overrides
    )


def get_research_config(**overrides) -> PHMConfig:
    """
    研究配置 - 深度研究设置
    
    特点：
    - 多次运行用于统计
    - 完整的监控和日志
    - 高级功能启用
    """
    return PHMConfig(
        environment=EnvironmentConfig(
            experiment_name="research_experiment",
            project="phm_research",
            notes="深度研究实验，多次运行统计结果",
            seed=42,
            iterations=5,  # 多次运行
            wandb=True,
            swanlab=True,  # 双重监控
            WANDB_MODE="online"
        ),
        data=DataConfig(
            data_dir="./data",
            metadata_file="metadata.xlsx",
            batch_size=64,  # 更大批次
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            train_ratio=0.7,
            val_ratio=0.15,
            normalization="standardization",
            window_size=2048,
            stride=1024
        ),
        model=ModelConfig(
            name="M_02_ISFM",  # 使用增强版ISFM
            type="ISFM",
            
            embedding="E_02_HSE_v2", 
            backbone="B_08_PatchTST",
            task_head="H_09_multiple_task",  # 多任务头
            
            d_model=256,    # 更大模型
            num_heads=16,
            num_layers=12,
            d_ff=1024,
            dropout=0.1,
            
            patch_size_L=32,
            num_patches=64,
            output_dim=256
        ),
        task=TaskConfig(
            name="classification",
            type="CDDG",  # 跨数据集泛化
            epochs=200,
            lr=0.0001,
            weight_decay=0.0001,
            optimizer="adamw",  # 更好的优化器
            loss="CE",
            metrics=["acc", "f1", "precision", "recall", "auc"],
            scheduler=True,
            scheduler_type="cosine",
            early_stopping=True,
            es_patience=30
        ),
        trainer=TrainerConfig(
            name="Default_trainer",
            num_epochs=200,
            gpus=1,
            device="auto",
            mixed_precision=True,
            gradient_clip_val=0.5,
            accumulate_grad_batches=2,
            early_stopping=True,
            patience=30,
            wandb=True,
            save_top_k=10,
            log_every_n_steps=20,
            val_check_interval=0.5,  # 更频繁验证
            profiler="simple"  # 性能分析
        ),
        **overrides
    )


def get_production_config(**overrides) -> PHMConfig:
    """
    生产配置 - 生产环境优化
    
    特点：
    - 性能优化设置
    - 稳定性优先
    - 资源高效利用
    """
    return PHMConfig(
        environment=EnvironmentConfig(
            experiment_name="production_experiment",
            project="phm_production",
            notes="生产环境配置，优化性能和稳定性",
            seed=42,
            iterations=1,
            wandb=False,  # 生产环境通常不需要
            WANDB_MODE="disabled"
        ),
        data=DataConfig(
            data_dir="./data",
            metadata_file="metadata.xlsx",
            batch_size=128,  # 大批次提高效率
            num_workers=16,  # 多进程加载
            pin_memory=True,
            persistent_workers=True,
            train_ratio=0.8,  # 更多训练数据
            normalization=True,
            window_size=1024,
            stride=512
        ),
        model=ModelConfig(
            name="M_01_ISFM",  # 稳定版本
            type="ISFM",
            
            embedding="E_01_HSE",
            backbone="B_04_Dlinear",  # 高效骨干
            task_head="H_01_Linear_cla",
            
            d_model=128,  # 平衡性能和效率
            num_heads=8,
            num_layers=6,
            dropout=0.0,  # 生产环境不需要dropout
            
            patch_size_L=16,
            num_patches=32,
            output_dim=128
        ),
        task=TaskConfig(
            name="classification",
            type="DG",
            epochs=50,
            lr=0.001,
            weight_decay=0.0001,
            optimizer="adam",
            loss="CE",
            metrics=["acc"],  # 简化指标
            scheduler=False,  # 简化调度
            early_stopping=False,  # 完整训练
            shuffle=True
        ),
        trainer=TrainerConfig(
            name="Default_trainer",
            num_epochs=50,
            gpus=1,
            device="auto",
            mixed_precision=True,  # 性能优化
            gradient_clip_val=None,  # 简化设置
            accumulate_grad_batches=1,
            early_stopping=False,
            wandb=False,
            save_top_k=1,  # 只保存最佳模型
            enable_progress_bar=False,  # 减少输出
            log_every_n_steps=100
        ),
        **overrides
    )


def get_benchmark_config(**overrides) -> PHMConfig:
    """
    基准测试配置 - 标准评估设置
    
    特点：
    - 标准化参数
    - 公平比较设置
    - 多指标评估
    """
    return PHMConfig(
        environment=EnvironmentConfig(
            experiment_name="benchmark_experiment", 
            project="phm_benchmark",
            notes="基准测试配置，用于模型对比",
            seed=42,
            iterations=3,  # 多次运行求均值
            wandb=True,
            WANDB_MODE="online"
        ),
        data=DataConfig(
            data_dir="./data",
            metadata_file="metadata.xlsx",
            batch_size=32,  # 标准批次
            num_workers=4,
            pin_memory=True,
            train_ratio=0.7,
            val_ratio=0.15,
            normalization="standardization",  # 标准归一化
            window_size=1024,
            stride=512
        ),
        model=ModelConfig(
            # 模型参数将被覆盖，这里提供默认值
            name="ResNet1D",
            type="CNN",
            input_dim=1,
            dropout=0.1
        ),
        task=TaskConfig(
            name="classification",
            type="DG",
            epochs=100,  # 充分训练
            lr=0.001,
            weight_decay=0.0001,
            optimizer="adam",
            loss="CE",
            metrics=["acc", "f1", "precision", "recall", "auc"],  # 完整指标
            scheduler=True,
            scheduler_type="step",
            step_size=30,
            gamma=0.1,
            early_stopping=True,
            es_patience=20
        ),
        trainer=TrainerConfig(
            name="Default_trainer",
            num_epochs=100,
            gpus=1,
            device="auto",
            mixed_precision=True,
            gradient_clip_val=1.0,
            early_stopping=True,
            patience=20,
            wandb=True,
            save_top_k=5,
            monitor_metric="val_f1",
            mode="max"
        ),
        **overrides
    )


# ==================== 多任务和特殊配置 ====================

def get_multitask_config(**overrides) -> PHMConfig:
    """多任务学习配置"""
    base_config = get_isfm_config()
    
    # 更新为多任务设置
    multitask_overrides = {
        'environment__experiment_name': 'multitask_experiment',
        'environment__notes': '多任务学习实验',
        'model__task_head': 'H_09_multiple_task',
        'task__name': 'multitask',
        'task__type': 'Multitask', 
        'task__task_list': ['classification', 'prediction'],
        'task__loss_weights': {'classification': 1.0, 'prediction': 0.5},
        'trainer__num_epochs': 150
    }
    multitask_overrides.update(overrides)
    
    return PHMConfig(**{**base_config.dict(), **_flatten_dict(multitask_overrides)})


def get_fewshot_config(**overrides) -> PHMConfig:
    """少样本学习配置"""  
    base_config = get_basic_config()
    
    fewshot_overrides = {
        'environment__experiment_name': 'fewshot_experiment',
        'environment__notes': '少样本学习实验',
        'model__name': 'ProtoNet',
        'model__type': 'FewShot',
        'task__name': 'classification',
        'task__type': 'FS',
        'task__num_support': 5,
        'task__num_query': 15,
        'task__num_episodes': 1000,
        'task__epochs': 200,
        'trainer__num_epochs': 200
    }
    fewshot_overrides.update(overrides)
    
    return PHMConfig(**{**base_config.dict(), **_flatten_dict(fewshot_overrides)})


# ==================== 预设管理 ====================

PRESET_CONFIGS = {
    'quickstart': get_quickstart_config,
    'basic': get_basic_config,
    'isfm': get_isfm_config,
    'research': get_research_config,
    'production': get_production_config,
    'benchmark': get_benchmark_config,
    'multitask': get_multitask_config,
    'fewshot': get_fewshot_config
}


def get_preset_config(preset_name: str, **overrides) -> PHMConfig:
    """
    获取预设配置
    
    Args:
        preset_name: 预设名称
        **overrides: 覆盖参数（支持双下划线语法）
        
    Returns:
        PHMConfig: 配置对象
    """
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"未知预设 '{preset_name}'，可用预设: {available}")
    
    config_func = PRESET_CONFIGS[preset_name]
    return config_func(**overrides)


def list_presets() -> Dict[str, str]:
    """列出所有可用预设及其描述"""
    descriptions = {
        'quickstart': '🚀 5分钟快速上手，使用简单模型和小数据集',
        'basic': '🏗️ 基础研究配置，平衡的参数设置',
        'isfm': '🧠 ISFM基础模型配置，使用Transformer架构',
        'research': '🔬 深度研究配置，多次运行和完整监控', 
        'production': '🏭 生产环境配置，性能和稳定性优化',
        'benchmark': '📊 基准测试配置，标准化评估设置',
        'multitask': '🎯 多任务学习配置，同时训练多个任务',
        'fewshot': '🎪 少样本学习配置，原型网络架构'
    }
    return descriptions


def create_custom_preset(name: str, base_preset: str = 'basic', **overrides) -> PHMConfig:
    """
    创建自定义预设
    
    Args:
        name: 自定义预设名称
        base_preset: 基础预设名称
        **overrides: 覆盖参数
        
    Returns:
        PHMConfig: 自定义配置对象
    """
    base_config = get_preset_config(base_preset)
    custom_config = PHMConfig(**{**base_config.dict(), **_flatten_dict(overrides)})
    
    # 更新实验名称
    custom_config.environment.experiment_name = name
    custom_config.environment.notes = f"基于 {base_preset} 的自定义配置"
    
    return custom_config


# ==================== 辅助函数 ====================

def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """将嵌套字典扁平化为双下划线格式"""
    flattened = {}
    for key, value in d.items():
        new_key = f"{prefix}__{key}" if prefix else key
        if isinstance(value, dict) and not key.endswith('_'):
            flattened.update(_flatten_dict(value, new_key))
        else:
            flattened[new_key] = value
    return flattened


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 列出所有预设
    print("📋 可用配置预设:")
    for name, desc in list_presets().items():
        print(f"  {name}: {desc}")
    
    # 创建快速开始配置
    print(f"\n🚀 创建快速开始配置:")
    quickstart = get_preset_config("quickstart")
    print(f"  实验名: {quickstart.environment.experiment_name}")
    print(f"  模型: {quickstart.model.type}.{quickstart.model.name}")
    print(f"  批次大小: {quickstart.data.batch_size}")
    
    # 创建自定义ISFM配置
    print(f"\n🧠 创建自定义ISFM配置:")
    custom_isfm = get_preset_config("isfm", 
                                   model__d_model=256,
                                   trainer__num_epochs=150)
    print(f"  模型维度: {custom_isfm.model.d_model}")
    print(f"  训练轮数: {custom_isfm.trainer.num_epochs}")
    
    # 保存配置文件
    print(f"\n💾 保存配置文件:")
    quickstart.save_yaml("quickstart_config.yaml", minimal=True)
    print("  已保存: quickstart_config.yaml")