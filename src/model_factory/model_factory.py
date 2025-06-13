"""
模型读取模块
负责加载和构建模型实例
"""
import os
import importlib
import torch

def model_factory(args_model,metadata):
    """
    简化版模型读取器，直接加载单个模型
    
    Args:
        args_model: 包含模型配置的命名空间或字典
            必须包含:
            - model_name: 模型名称
            - model_config: 模型配置字典
            
    Returns:
        model: 初始化好的模型实例
    """
    # 获取模型名称
    model_name = args_model.name
    model_type = args_model.type
    # 直接导入模型模块
    try:
        model_module = importlib.import_module(f"src.model_factory.{model_type}.{model_name}")
        print(f"成功导入模型模块: {model_name}")
    except ImportError:
        raise ValueError(f"未找到名为 {model_name} 的模型模块")
    
    
    # 创建模型实例
    try:
        # 如果args_model.model_config存在，使用它作为参数
        model = model_module.Model(args_model,metadata) # TODO metadata 
        
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
    """
    加载模型权重，匹配shape后加载
    Args:
        ckpt_path (str): 模型权重文件路径
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
    state_dict = torch.load(ckpt_path, map_location='cpu')
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