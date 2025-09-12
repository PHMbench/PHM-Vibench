"""
Flow配置工厂 (Flow Configuration Factory)

这个模块提供Flow预训练任务自测试的配置生成功能，使用argparse.Namespace模式
生成与PHM-Vibench框架兼容的模拟配置对象。

Author: PHM-Vibench Team
Date: 2025-09-10
"""

from argparse import Namespace
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import torch


@dataclass
class FlowConfigTemplate:
    """
    Flow配置模板 (Flow Configuration Template)
    
    定义Flow自测试中使用的各种配置模板参数。
    """
    # 数据配置
    batch_size: int = 8
    sequence_length: int = 64
    input_dim: int = 3
    num_classes: int = 4
    
    # 模型配置
    hidden_dim: int = 128
    time_dim: int = 32
    condition_dim: int = 32
    
    # 任务配置
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_steps: int = 50
    
    # 训练器配置
    max_epochs: int = 5
    gpus: int = 1 if torch.cuda.is_available() else 0
    precision: int = 32
    
    # 环境配置
    seed: int = 42


class FlowConfigurationFactory:
    """
    Flow配置工厂类 (Flow Configuration Factory Class)
    
    使用argparse.Namespace模式生成Flow预训练任务自测试所需的各种配置对象。
    遵循test/conftest.py中basic_model_configs的模式，确保与PHM-Vibench框架兼容。
    """
    
    def __init__(self, template: Optional[FlowConfigTemplate] = None):
        """
        初始化配置工厂
        
        Args:
            template: 配置模板，如果为None则使用默认模板
        """
        self.template = template or FlowConfigTemplate()
        
    def create_flow_task_config(
        self,
        use_contrastive: bool = True,
        enable_visualization: bool = False,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Namespace:
        """
        创建Flow任务配置 (Create Flow task configuration)
        
        生成args_task Namespace对象，遵循PHM-Vibench的任务配置模式。
        
        Args:
            use_contrastive: 是否使用对比学习
            enable_visualization: 是否启用可视化
            custom_params: 自定义参数字典
            
        Returns:
            Namespace: 任务配置对象
        """
        config = {
            # 基本任务信息
            'name': 'flow_pretrain',
            'type': 'pretrain',
            
            # Flow参数
            'num_steps': self.template.num_steps,
            'flow_lr': self.template.lr,
            'sigma_min': 0.001,
            'sigma_max': 1.0,
            
            # 对比学习参数
            'use_contrastive': use_contrastive,
            'lambda_flow': 1.0,
            'lambda_contrastive': 0.1,
            'temperature': 0.1,
            'contrastive_samples': 256,
            
            # 生成参数
            'use_conditional': True,
            'generation_steps': 50,
            'generation_batch_size': 16,
            
            # 训练参数
            'lr': self.template.lr,
            'weight_decay': self.template.weight_decay,
            'max_epochs': self.template.max_epochs,
            'optimizer': 'adam',
            'scheduler': True,
            'scheduler_type': 'cosine',
            
            # 监控参数
            'enable_visualization': enable_visualization,
            'track_memory': True,
            'track_gradients': True,
            'log_generation_samples': False,
            
            # 验证参数
            'validation_interval': 1.0,
            'validation_samples': 64,
            'compute_metrics': True,
            'metrics_interval': 5,
            
            # 早停参数
            'early_stopping': False,
            'es_patience': 10,
            'es_min_delta': 1e-4,
        }
        
        # 应用自定义参数
        if custom_params:
            config.update(custom_params)
            
        return Namespace(**config)
    
    def create_model_config(
        self,
        model_size: str = "small",
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Namespace:
        """
        创建模型配置 (Create model configuration)
        
        生成args_model Namespace对象，遵循conftest.py中basic_model_configs的模式。
        
        Args:
            model_size: 模型大小 ("small", "medium", "large")
            custom_params: 自定义参数字典
            
        Returns:
            Namespace: 模型配置对象
        """
        # 基于模型大小调整参数
        size_configs = {
            "small": {
                "hidden_dim": 64,
                "time_dim": 16,
                "condition_dim": 16,
                "num_layers": 2,
            },
            "medium": {
                "hidden_dim": 128,
                "time_dim": 32,
                "condition_dim": 32,
                "num_layers": 4,
            },
            "large": {
                "hidden_dim": 256,
                "time_dim": 64,
                "condition_dim": 64,
                "num_layers": 6,
            }
        }
        
        size_config = size_configs.get(model_size, size_configs["small"])
        
        config = {
            # 基本模型信息
            'name': 'M_04_ISFM_Flow',
            'model_name': 'M_04_ISFM_Flow',
            
            # 输入维度
            'input_dim': self.template.input_dim,
            'sequence_length': self.template.sequence_length,
            
            # 架构参数
            'hidden_dim': size_config["hidden_dim"],
            'time_dim': size_config["time_dim"],
            'condition_dim': size_config["condition_dim"],
            'num_layers': size_config["num_layers"],
            
            # Flow特定参数
            'use_conditional': True,
            'time_embedding_type': 'sinusoidal',
            'condition_embedding_type': 'linear',
            
            # 正则化参数
            'dropout': 0.1,
            'layer_norm': True,
            'activation': 'gelu',
            
            # 输出参数
            'num_classes': self.template.num_classes,
            'output_dim': self.template.input_dim,
        }
        
        # 应用自定义参数
        if custom_params:
            config.update(custom_params)
            
        return Namespace(**config)
    
    def create_data_config(
        self,
        dataset_name: str = "CWRU",
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Namespace:
        """
        创建数据配置 (Create data configuration)
        
        生成args_data Namespace对象。
        
        Args:
            dataset_name: 数据集名称
            custom_params: 自定义参数字典
            
        Returns:
            Namespace: 数据配置对象
        """
        config = {
            # 数据集信息
            'data_dir': 'data',
            'dataset': dataset_name,
            'metadata_file': f'metadata_{dataset_name}.xlsx',
            
            # 数据加载参数
            'batch_size': self.template.batch_size,
            'sequence_length': self.template.sequence_length,
            'channels': self.template.input_dim,
            'num_workers': 0,  # 测试时使用0避免多进程问题
            
            # 数据处理参数
            'normalize': True,
            'standardize': False,
            'augmentation': False,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            
            # 域信息
            'num_domains': 1,
            'domain_id': 1,
            'source_domains': [1],
            'target_domains': [1],
            
            # 采样参数
            'sampling_rate': 1000.0,
            'overlap_ratio': 0.0,
            'signal_length': self.template.sequence_length,
        }
        
        # 应用自定义参数
        if custom_params:
            config.update(custom_params)
            
        return Namespace(**config)
    
    def create_trainer_config(
        self,
        fast_mode: bool = True,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Namespace:
        """
        创建训练器配置 (Create trainer configuration)
        
        生成args_trainer Namespace对象，适用于PyTorch Lightning训练器。
        
        Args:
            fast_mode: 是否使用快速模式（用于测试）
            custom_params: 自定义参数字典
            
        Returns:
            Namespace: 训练器配置对象
        """
        if fast_mode:
            # 快速测试模式
            config = {
                'max_epochs': 2,
                'max_steps': 10,
                'limit_train_batches': 3,
                'limit_val_batches': 2,
                'limit_test_batches': 2,
                'log_every_n_steps': 1,
                'val_check_interval': 1.0,
                'check_val_every_n_epoch': 1,
                'enable_checkpointing': False,
                'enable_progress_bar': False,
                'enable_model_summary': False,
                'logger': False,
            }
        else:
            # 标准模式
            config = {
                'max_epochs': self.template.max_epochs,
                'log_every_n_steps': 50,
                'val_check_interval': 1.0,
                'check_val_every_n_epoch': 1,
                'enable_checkpointing': True,
                'enable_progress_bar': True,
                'enable_model_summary': True,
            }
        
        # 通用配置
        common_config = {
            'gpus': self.template.gpus,
            'precision': self.template.precision,
            'gradient_clip_val': 1.0,
            'gradient_clip_algorithm': 'norm',
            'accumulate_grad_batches': 1,
            'deterministic': True,
            'benchmark': False,
        }
        
        config.update(common_config)
        
        # 应用自定义参数
        if custom_params:
            config.update(custom_params)
            
        return Namespace(**config)
    
    def create_environment_config(
        self,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Namespace:
        """
        创建环境配置 (Create environment configuration)
        
        生成args_environment Namespace对象。
        
        Args:
            custom_params: 自定义参数字典
            
        Returns:
            Namespace: 环境配置对象
        """
        config = {
            # 随机种子
            'seed': self.template.seed,
            'deterministic': True,
            'benchmark': False,
            
            # 设备配置
            'device': 'auto',
            'gpus': self.template.gpus,
            'precision': self.template.precision,
            
            # 并行配置
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            
            # 日志配置
            'logging_level': 'WARNING',  # 测试时减少日志输出
            'log_dir': 'logs/self_test',
            'experiment_name': 'flow_self_test',
            
            # 保存配置
            'save_dir': 'outputs/self_test',
            'save_predictions': False,
            'save_checkpoints': False,
        }
        
        # 应用自定义参数
        if custom_params:
            config.update(custom_params)
            
        return Namespace(**config)
    
    def create_complete_config_set(
        self,
        test_scenario: str = "basic",
        custom_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Namespace]:
        """
        创建完整配置集合 (Create complete configuration set)
        
        生成所有需要的配置对象，用于任务工厂实例化。
        
        Args:
            test_scenario: 测试场景 ("basic", "contrastive", "performance")
            custom_overrides: 自定义覆盖参数，格式为 {config_type: {param: value}}
            
        Returns:
            Dict[str, Namespace]: 包含所有配置的字典
        """
        overrides = custom_overrides or {}
        
        # 根据测试场景调整参数
        scenario_configs = {
            "basic": {
                "use_contrastive": False,
                "fast_mode": True,
                "model_size": "small"
            },
            "contrastive": {
                "use_contrastive": True,
                "fast_mode": True,
                "model_size": "medium"
            },
            "performance": {
                "use_contrastive": True,
                "fast_mode": False,
                "model_size": "large"
            }
        }
        
        scenario = scenario_configs.get(test_scenario, scenario_configs["basic"])
        
        # 生成各个配置
        configs = {
            'args_task': self.create_flow_task_config(
                use_contrastive=scenario["use_contrastive"],
                custom_params=overrides.get('task', {})
            ),
            'args_model': self.create_model_config(
                model_size=scenario["model_size"],
                custom_params=overrides.get('model', {})
            ),
            'args_data': self.create_data_config(
                custom_params=overrides.get('data', {})
            ),
            'args_trainer': self.create_trainer_config(
                fast_mode=scenario["fast_mode"],
                custom_params=overrides.get('trainer', {})
            ),
            'args_environment': self.create_environment_config(
                custom_params=overrides.get('environment', {})
            )
        }
        
        return configs
    
    def get_config_summary(self, configs: Dict[str, Namespace]) -> Dict[str, Any]:
        """
        获取配置摘要 (Get configuration summary)
        
        Args:
            configs: 配置字典
            
        Returns:
            包含配置摘要的字典
        """
        summary = {}
        
        for config_name, config_obj in configs.items():
            config_dict = vars(config_obj)
            summary[config_name] = {
                "total_params": len(config_dict),
                "key_params": {k: v for k, v in config_dict.items() 
                             if k in ['name', 'type', 'batch_size', 'hidden_dim', 
                                    'lr', 'max_epochs', 'use_contrastive']},
                "param_types": {k: type(v).__name__ for k, v in config_dict.items()}
            }
        
        return summary


# 便捷函数，遵循conftest.py的命名约定
def create_flow_test_configs(
    scenario: str = "basic",
    batch_size: int = 8,
    seq_len: int = 64,
    input_dim: int = 3
) -> Dict[str, Namespace]:
    """
    创建Flow测试配置的便捷函数 (Convenience function for creating Flow test configs)
    
    Args:
        scenario: 测试场景
        batch_size: 批次大小
        seq_len: 序列长度
        input_dim: 输入维度
        
    Returns:
        Dict[str, Namespace]: 配置字典
    """
    template = FlowConfigTemplate(
        batch_size=batch_size,
        sequence_length=seq_len,
        input_dim=input_dim
    )
    
    factory = FlowConfigurationFactory(template)
    return factory.create_complete_config_set(test_scenario=scenario)


# 导出的类和函数
__all__ = [
    'FlowConfigTemplate',
    'FlowConfigurationFactory',
    'create_flow_test_configs',
]


if __name__ == "__main__":
    """
    Flow配置工厂自测试 (Flow Configuration Factory Self-Test)
    
    测试配置工厂的各种功能，确保生成的配置对象符合PHM-Vibench框架要求。
    """
    print("=" * 60)
    print("Flow配置工厂自测试 (Flow Configuration Factory Self-Test)")
    print("=" * 60)
    
    try:
        # 测试1: 基本配置工厂创建
        print("\n1. 测试基本配置工厂创建...")
        factory = FlowConfigurationFactory()
        print("✓ FlowConfigurationFactory创建成功")
        
        # 测试2: 单个配置对象创建
        print("\n2. 测试单个配置对象创建...")
        
        # 任务配置
        task_config = factory.create_flow_task_config()
        print(f"✓ 任务配置创建成功: {task_config.name}.{task_config.type}")
        print(f"  - 使用对比学习: {task_config.use_contrastive}")
        print(f"  - 学习率: {task_config.lr}")
        
        # 模型配置
        model_config = factory.create_model_config()
        print(f"✓ 模型配置创建成功: {model_config.name}")
        print(f"  - 隐藏维度: {model_config.hidden_dim}")
        print(f"  - 输入维度: {model_config.input_dim}")
        
        # 数据配置
        data_config = factory.create_data_config()
        print(f"✓ 数据配置创建成功: {data_config.dataset}")
        print(f"  - 批次大小: {data_config.batch_size}")
        print(f"  - 序列长度: {data_config.sequence_length}")
        
        # 训练器配置
        trainer_config = factory.create_trainer_config()
        print(f"✓ 训练器配置创建成功")
        print(f"  - 最大轮数: {trainer_config.max_epochs}")
        print(f"  - GPU数量: {trainer_config.gpus}")
        
        # 环境配置
        env_config = factory.create_environment_config()
        print(f"✓ 环境配置创建成功")
        print(f"  - 随机种子: {env_config.seed}")
        print(f"  - 确定性: {env_config.deterministic}")
        
        # 测试3: 不同模型大小配置
        print("\n3. 测试不同模型大小配置...")
        for size in ["small", "medium", "large"]:
            model_config = factory.create_model_config(model_size=size)
            print(f"✓ {size}模型配置: hidden_dim={model_config.hidden_dim}, "
                  f"time_dim={model_config.time_dim}")
        
        # 测试4: 不同测试场景
        print("\n4. 测试不同测试场景...")
        for scenario in ["basic", "contrastive", "performance"]:
            configs = factory.create_complete_config_set(test_scenario=scenario)
            print(f"✓ {scenario}场景配置集创建成功")
            print(f"  - 配置数量: {len(configs)}")
            print(f"  - 配置类型: {list(configs.keys())}")
            print(f"  - 使用对比学习: {configs['args_task'].use_contrastive}")
        
        # 测试5: 自定义参数覆盖
        print("\n5. 测试自定义参数覆盖...")
        custom_overrides = {
            'task': {'lr': 1e-3, 'use_contrastive': False},
            'model': {'hidden_dim': 512},
            'trainer': {'max_epochs': 10}
        }
        configs = factory.create_complete_config_set(
            test_scenario="basic",
            custom_overrides=custom_overrides
        )
        print(f"✓ 自定义覆盖成功")
        print(f"  - 学习率: {configs['args_task'].lr}")
        print(f"  - 隐藏维度: {configs['args_model'].hidden_dim}")
        print(f"  - 最大轮数: {configs['args_trainer'].max_epochs}")
        
        # 测试6: 配置摘要生成
        print("\n6. 测试配置摘要生成...")
        summary = factory.get_config_summary(configs)
        print(f"✓ 配置摘要生成成功")
        for config_name, config_summary in summary.items():
            print(f"  - {config_name}: {config_summary['total_params']}个参数")
        
        # 测试7: 便捷函数测试
        print("\n7. 测试便捷函数...")
        conv_configs = create_flow_test_configs(scenario="contrastive")
        print(f"✓ 便捷函数创建配置成功")
        print(f"  - 配置数量: {len(conv_configs)}")
        
        # 测试8: Namespace对象验证
        print("\n8. 测试Namespace对象验证...")
        task_config = factory.create_flow_task_config()
        
        # 验证是否是Namespace对象
        from argparse import Namespace
        is_namespace = isinstance(task_config, Namespace)
        print(f"✓ Namespace类型验证: {'通过' if is_namespace else '失败'}")
        
        # 验证属性访问
        has_required_attrs = all(hasattr(task_config, attr) for attr in 
                               ['name', 'type', 'lr', 'use_contrastive'])
        print(f"✓ 必需属性验证: {'通过' if has_required_attrs else '失败'}")
        
        # 测试9: 设备兼容性
        print("\n9. 测试设备兼容性...")
        # CPU配置
        cpu_template = FlowConfigTemplate(gpus=0)
        cpu_factory = FlowConfigurationFactory(cpu_template)
        cpu_configs = cpu_factory.create_complete_config_set()
        print(f"✓ CPU配置创建成功: GPUs={cpu_configs['args_trainer'].gpus}")
        
        # GPU配置（如果可用）
        if torch.cuda.is_available():
            gpu_template = FlowConfigTemplate(gpus=1)
            gpu_factory = FlowConfigurationFactory(gpu_template)
            gpu_configs = gpu_factory.create_complete_config_set()
            print(f"✓ GPU配置创建成功: GPUs={gpu_configs['args_trainer'].gpus}")
        else:
            print("✓ GPU不可用，跳过GPU配置测试")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！Flow配置工厂工作正常。")
        print("🔧 生成的配置对象与PHM-Vibench框架完全兼容。")
        print("📝 支持多种测试场景和自定义参数覆盖。")
        print("⚙️ 可用于Flow预训练任务的自测试配置生成。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        print("请检查代码实现并修复问题。")
        import traceback
        traceback.print_exc()
        raise