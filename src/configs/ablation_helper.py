"""
消融实验辅助工具
================

提供简化的参数网格搜索和消融实验功能：
- 🔄 参数组合生成
- 📊 单参数消融实验  
- 🔍 网格搜索支持
- ⚡ 与SimpleNamespace配置系统集成

使用方式:
    from src.configs.ablation_helper import AblationHelper
    
    # 单参数消融
    configs = AblationHelper.single_param_ablation(
        'configs/base.yaml', 
        'model.d_model', 
        [128, 256, 512]
    )
    
    # 网格搜索
    param_grid = {
        'model.d_model': [128, 256],
        'task.lr': [0.001, 0.01]
    }
    configs_with_overrides = AblationHelper.grid_search('configs/base.yaml', param_grid)

作者: PHM-Vibench Team
"""

from itertools import product
from typing import Dict, List, Any, Tuple, Union, Optional
from types import SimpleNamespace

from .config_utils import load_config


class AblationHelper:
    """消融实验辅助工具"""
    
    @staticmethod
    def generate_overrides(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """生成参数组合
        
        Args:
            param_grid: 参数网格，格式如：
                {
                    'model.d_model': [128, 256, 512],
                    'task.lr': [0.001, 0.01],
                    'data.batch_size': [32, 64]
                }
                
        Returns:
            List[Dict[str, Any]]: 覆盖参数列表
            例如: [
                {'model.d_model': 128, 'task.lr': 0.001, 'data.batch_size': 32},
                {'model.d_model': 128, 'task.lr': 0.001, 'data.batch_size': 64},
                ...
            ]
        """
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        
        overrides_list = []
        for combo in product(*values):
            overrides = dict(zip(keys, combo))
            overrides_list.append(overrides)
        return overrides_list
    
    @staticmethod
    def single_param_ablation(base_config_path: str, 
                             param_name: str, 
                             values: List[Any]) -> List[SimpleNamespace]:
        """单参数消融实验
        
        Args:
            base_config_path: 基础配置文件路径
            param_name: 要变化的参数名（支持嵌套，如 'model.d_model'）
            values: 参数值列表
            
        Returns:
            List[SimpleNamespace]: 配置对象列表
        """
        configs = []
        for value in values:
            config = load_config(base_config_path, {param_name: value})
            configs.append(config)
        return configs
    
    @staticmethod
    def grid_search(base_config_path: str, 
                   param_grid: Dict[str, List[Any]]) -> List[Tuple[SimpleNamespace, Dict[str, Any]]]:
        """网格搜索配置生成
        
        Args:
            base_config_path: 基础配置文件路径
            param_grid: 参数网格
            
        Returns:
            List[Tuple[SimpleNamespace, Dict[str, Any]]]: (配置对象, 覆盖参数) 元组列表
        """
        overrides_list = AblationHelper.generate_overrides(param_grid)
        configs_with_overrides = []
        
        for overrides in overrides_list:
            config = load_config(base_config_path, overrides)
            configs_with_overrides.append((config, overrides))
        
        return configs_with_overrides
    
    @staticmethod
    def compare_param_values(configs: List[SimpleNamespace], 
                           param_path: str) -> List[Any]:
        """比较多个配置中某个参数的值
        
        Args:
            configs: 配置对象列表
            param_path: 参数路径，如 'model.d_model'
            
        Returns:
            List[Any]: 参数值列表
        """
        values = []
        for config in configs:
            # 解析参数路径
            keys = param_path.split('.')
            obj = config
            for key in keys:
                obj = getattr(obj, key)
            values.append(obj)
        return values
    
    @staticmethod
    def create_experiment_name(base_name: str, overrides: Dict[str, Any]) -> str:
        """根据覆盖参数创建实验名称
        
        Args:
            base_name: 基础实验名称
            overrides: 覆盖参数字典
            
        Returns:
            str: 实验名称
        """
        override_str = "_".join([f"{k.replace('.', '_')}{v}" for k, v in overrides.items()])
        return f"{base_name}_{override_str}"
    
    @staticmethod
    def validate_config(config: SimpleNamespace) -> bool:
        """简单的配置验证
        
        Args:
            config: 配置对象
            
        Returns:
            bool: 是否有效
        """
        try:
            # 检查必需的顶级配置节
            required_sections = ['environment', 'data', 'model', 'task', 'trainer']
            for section in required_sections:
                if not hasattr(config, section):
                    print(f"⚠️  缺少配置节: {section}")
                    return False
            
            # 检查数据配置的必需字段
            if not hasattr(config.data, 'data_dir') or not hasattr(config.data, 'metadata_file'):
                print("⚠️  数据配置缺少必需字段: data_dir 或 metadata_file")
                return False
            
            # 检查模型配置的必需字段
            if not hasattr(config.model, 'name') or not hasattr(config.model, 'type'):
                print("⚠️  模型配置缺少必需字段: name 或 type")
                return False
            
            # 检查任务配置的必需字段
            if not hasattr(config.task, 'name') or not hasattr(config.task, 'type'):
                print("⚠️  任务配置缺少必需字段: name 或 type")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  配置验证错误: {e}")
            return False


# 便捷函数
def quick_ablation(base_config_path: str, 
                  param_name: str, 
                  values: List[Any]) -> List[SimpleNamespace]:
    """快速单参数消融的便捷函数"""
    return AblationHelper.single_param_ablation(base_config_path, param_name, values)


def quick_grid_search(base_config_path: str, 
                     param_grid: Optional[Dict[str, List[Any]]] = None,
                     **param_kwargs) -> List[Tuple[SimpleNamespace, Dict[str, Any]]]:
    """快速网格搜索的便捷函数 - 支持双模式API
    
    支持两种参数传递方式：
    
    方式1 - 字典传参（推荐，支持点号）:
        configs = quick_grid_search(
            'quickstart',
            {'model.dropout': [0.1, 0.2], 'task.lr': [0.001, 0.01]}
        )
    
    方式2 - kwargs传参（便捷，IDE友好）:
        configs = quick_grid_search(
            'quickstart',
            model__dropout=[0.1, 0.2],  # 双下划线自动转为点号
            task__lr=[0.001, 0.01]
        )
    
    注意：Python语法不允许在关键字参数中使用点号，因此方式2需要使用双下划线。
    """
    if param_grid is None:
        param_grid = {}
    
    # 合并kwargs参数（将双下划线转换为点号）
    for key, values in param_kwargs.items():
        param_key = key.replace('__', '.')
        param_grid[param_key] = values
    
    if not param_grid:
        raise ValueError("必须提供参数网格，使用param_grid字典或**kwargs参数")
    
    return AblationHelper.grid_search(base_config_path, param_grid)


__all__ = [
    "AblationHelper",
    "quick_ablation", 
    "quick_grid_search"
]


# 使用示例和测试
if __name__ == "__main__":
    print("🔬 消融实验工具测试")
    print("=" * 40)
    
    # 示例配置路径（需要根据实际情况调整）
    config_path = "configs/demo/Single_DG/CWRU.yaml"
    
    try:
        # 示例1: 单参数消融
        print("\n📊 示例1: 单参数消融")
        configs = AblationHelper.single_param_ablation(
            config_path, 
            'model.d_model', 
            [128, 256, 512]
        )
        
        values = AblationHelper.compare_param_values(configs, 'model.d_model')
        print(f"  生成配置数量: {len(configs)}")
        print(f"  d_model值: {values}")
        
    except Exception as e:
        print(f"  ❌ 单参数消融测试失败: {e}")
    
    try:
        # 示例2: 网格搜索
        print("\n🔍 示例2: 网格搜索")
        param_grid = {
            'model.d_model': [128, 256],
            'task.lr': [0.001, 0.01]
        }
        
        configs_with_overrides = AblationHelper.grid_search(config_path, param_grid)
        print(f"  生成配置组合数量: {len(configs_with_overrides)}")
        
        for i, (config, overrides) in enumerate(configs_with_overrides[:2]):  # 只显示前两个
            print(f"  组合{i+1}: {overrides}")
        
    except Exception as e:
        print(f"  ❌ 网格搜索测试失败: {e}")
    
    try:
        # 示例3: 双模式API测试
        print("\n⚡ 示例3: 双模式API测试")
        
        # 方式1：字典传参（推荐）
        configs1 = quick_grid_search(
            config_path,
            {'model.d_model': [64, 128], 'task.epochs': [10, 20]}
        )
        print(f"  方式1 - 字典传参: {len(configs1)} 个配置")
        
        # 方式2：kwargs传参（便捷）
        configs2 = quick_grid_search(
            config_path,
            model__d_model=[64, 128],  # 双下划线转点号
            task__epochs=[10, 20]
        )
        print(f"  方式2 - kwargs传参: {len(configs2)} 个配置")
        print(f"  两种方式生成的配置数相同: {len(configs1) == len(configs2)}")
        
    except Exception as e:
        print(f"  ❌ 便捷函数测试失败: {e}")
    
    print(f"\n🎉 消融实验工具测试完成！")