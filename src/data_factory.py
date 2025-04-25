"""
数据工厂模块，用于构建和初始化数据集

此模块负责创建和配置数据集对象，实现数据集的注册和构建功能。
所有数据集类应在此处注册，以便通过配置文件引用。
"""

import os
import logging
import importlib
from typing import Dict, Any, Optional, Type

# 数据集注册表
DATA_REGISTRY = {}

def register_dataset(name: str, dataset_cls: Type):
    """注册数据集类
    
    Args:
        name: 数据集名称
        dataset_cls: 数据集类
    """
    if name in DATA_REGISTRY:
        logging.warning(f"数据集 '{name}' 已存在，将被覆盖")
    DATA_REGISTRY[name] = dataset_cls
    
def build_dataset(config: Dict[str, Any]):
    """根据配置构建数据集
    
    Args:
        config: 数据集配置字典
        
    Returns:
        构建好的数据集对象
    
    Raises:
        ValueError: 若数据集名称未注册或配置无效
    """
    dataset_name = config.get('name')
    if not dataset_name:
        raise ValueError("配置中未指定数据集名称")
    
    # 检查数据集是否已注册
    if dataset_name not in DATA_REGISTRY:
        try:
            # 尝试动态导入数据集模块
            module_path = f"src.datasets.{dataset_name.lower()}"
            module = importlib.import_module(module_path)
            if hasattr(module, dataset_name):
                dataset_cls = getattr(module, dataset_name)
                register_dataset(dataset_name, dataset_cls)
            else:
                raise ValueError(f"找不到数据集类：{dataset_name}")
        except (ImportError, AttributeError):
            raise ValueError(f"数据集 '{dataset_name}' 未注册，且无法动态导入")
    
    # 获取数据集类和参数
    dataset_cls = DATA_REGISTRY[dataset_name]
    dataset_args = config.get('args', {})
    
    # 构建数据集
    try:
        dataset = dataset_cls(**dataset_args)
        logging.info(f"成功构建数据集: {dataset_name}")
        return dataset
    except Exception as e:
        raise ValueError(f"构建数据集 {dataset_name} 失败: {str(e)}")

# 默认数据集类，可用于测试或示例
class DefaultDataset:
    """默认数据集类，用于测试或示范"""
    
    def __init__(self, data_path="data/default", **kwargs):
        """初始化默认数据集
        
        Args:
            data_path: 数据路径
            **kwargs: 其他参数
        """
        self.data_path = data_path
        self.kwargs = kwargs
        logging.info(f"初始化默认数据集，路径: {data_path}")
        
    def __len__(self):
        return 0
        
    def __getitem__(self, idx):
        return None

# 注册默认数据集
register_dataset("DefaultDataset", DefaultDataset)

# 导入特定数据集并注册
# 示例：从各个模块导入数据集类并注册
try:
    from src.datasets.bearing_dataset import BearingDataset
    register_dataset("BearingDataset", BearingDataset)
except ImportError:
    logging.debug("BearingDataset 未导入")
    
try:
    from src.datasets.dummy_dataset import DummyDataset
    register_dataset("DummyDataset", DummyDataset)
except ImportError:
    logging.debug("DummyDataset 未导入")