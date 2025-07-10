"""Model factory
================

This module dynamically imports model definitions under
``src.model_factory`` and instantiates them using configuration
objects. It also handles optional checkpoint loading.
"""
import os
import importlib
import torch
from ..utils.utils import get_num_classes


def model_factory(args_model, metadata):
    """Instantiate a model by name.

    Parameters
    ----------
    args_model : Namespace
        Configuration namespace with at least ``name`` and ``type``
        fields. Other attributes are passed to the model's ``Model``
        constructor.
    metadata : Any
        Dataset metadata, used here only to compute ``num_classes``.

    Returns
    -------
    nn.Module
        Instantiated model ready for training.
    """
    # 获取模型名称
    model_name = args_model.name
    model_type = args_model.type
    args_model.num_classes = get_num_classes(metadata)
    # 直接导入模型模块
    try:
        model_module = importlib.import_module(f"src.model_factory.{model_type}.{model_name}")
        print(f"成功导入模型模块: {model_name}")
    except ImportError:
        raise ValueError(f"未找到名为 {model_name} 的模型模块")
    
    
    # 创建模型实例
    try:
        # 将配置传递给模型构造函数
        model = model_module.Model(args_model, metadata)
        
        # 如果指定了预训练权重路径，加载权重
        if hasattr(args_model, 'weights_path') and args_model.weights_path:
            weights_path = args_model.weights_path
            if os.path.exists(weights_path):
                try:
                    # 尝试加载模型权重
                    load_ckpt(model, weights_path)
                    print(f"加载权重成功: {weights_path}")
                except Exception as e:
                    print(f"加载权重时出错: {str(e)},初始化模型时使用默认权重")
                    # 权重加载失败但不阻止模型使用
                    pass
        
        return model
    
    except Exception as e:
        raise RuntimeError(f"创建模型实例时出错: {str(e)}")
    

def load_ckpt(model, ckpt_path):
    """Load weights from ``ckpt_path`` into ``model``.

    Parameters
    ----------
    model : nn.Module
        Model instance to be updated.
    ckpt_path : str
        Path to a PyTorch checkpoint file.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_dict = model.state_dict()
    matched_dict = {}
    skipped = []
    for name, param in state_dict.items():
        if name in model_dict:
            matched_dict[name] = param
        else:
            skipped.append((name, "not in model"))
    # 加载匹配的权重
    model.load_state_dict(matched_dict, strict=False)
    # 打印跳过的参数
    if skipped:
        print("跳过以下不匹配的参数：")
        for name, model_sz in skipped:
            print(f"  {name}: checkpoint vs model {model_sz}")
    print(f"已加载匹配的权重: {ckpt_path}")
