"""
模型工厂模块，用于构建各种机器学习模型

此模块负责创建和配置模型对象，实现模型的注册和构建功能。
所有模型类应在此处注册，以便通过配置文件引用。
"""

import logging
import importlib
from typing import Dict, Any, Optional, Type

# 模型注册表
MODEL_REGISTRY = {}

def register_model(name: str, model_cls: Type):
    """注册模型类
    
    Args:
        name: 模型名称
        model_cls: 模型类
    """
    if name in MODEL_REGISTRY:
        logging.warning(f"模型 '{name}' 已存在，将被覆盖")
    MODEL_REGISTRY[name] = model_cls

def build_model(config: Dict[str, Any]):
    """根据配置构建模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        构建好的模型对象
    
    Raises:
        ValueError: 若模型名称未注册或配置无效
    """
    model_name = config.get('name')
    if not model_name:
        raise ValueError("配置中未指定模型名称")
    
    # 检查模型是否已注册
    if model_name not in MODEL_REGISTRY:
        try:
            # 尝试动态导入模型模块
            module_path = f"src.models.{model_name.lower()}"
            module = importlib.import_module(module_path)
            if hasattr(module, model_name):
                model_cls = getattr(module, model_name)
                register_model(model_name, model_cls)
            else:
                raise ValueError(f"找不到模型类：{model_name}")
        except (ImportError, AttributeError):
            raise ValueError(f"模型 '{model_name}' 未注册，且无法动态导入")
    
    # 获取模型类和参数
    model_cls = MODEL_REGISTRY[model_name]
    model_args = config.get('args', {})
    
    # 构建模型
    try:
        model = model_cls(**model_args)
        logging.info(f"成功构建模型: {model_name}")
        return model
    except Exception as e:
        raise ValueError(f"构建模型 {model_name} 失败: {str(e)}")

# 默认模型类，可用于测试或示例
class DefaultModel:
    """默认模型类，用于测试或示范"""
    
    def __init__(self, input_dim=10, output_dim=2, hidden_dims=None, **kwargs):
        """初始化默认模型
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度列表
            **kwargs: 其他参数
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.kwargs = kwargs
        logging.info(f"初始化默认模型，输入维度: {input_dim}, 输出维度: {output_dim}")
    
    def forward(self, x):
        """前向计算
        
        Args:
            x: 输入数据
            
        Returns:
            模型输出
        """
        return None

# 注册默认模型
register_model("DefaultModel", DefaultModel)

# 导入特定模型并注册
# 示例：从各个模块导入模型类并注册
try:
    from src.models.transformer import Transformer
    register_model("Transformer", Transformer)
except ImportError:
    logging.debug("Transformer 模型未导入")
    
try:
    from src.models.cnn import CNN
    register_model("CNN", CNN)
except ImportError:
    logging.debug("CNN 模型未导入")
    
try:
    from src.models.rnn import RNN
    register_model("RNN", RNN)
except ImportError:
    logging.debug("RNN 模型未导入")