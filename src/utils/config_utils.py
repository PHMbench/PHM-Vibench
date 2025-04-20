import os
import yaml
import types
from datetime import datetime

def load_config(config_path):
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def makedir(path):
    """创建目录（如果不存在）
    
    Args:
        path: 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def path_name(configs, iteration=0):
    """根据配置和迭代次数生成路径和名称
    
    Args:
        configs: 配置字典
        iteration: 迭代次数
        
    Returns:
        路径和实验名称
    """
    # 获取各组件名称
    dataset_name = configs['dataset']['name']
    model_name = configs['model']['name']
    task_name = configs['task']['name']
    trainer_name = configs['trainer']['name']
    
    # 组建实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_name}_{dataset_name}_{task_name}_{timestamp}"
    
    # 创建结果保存路径
    result_dir = f"results/{exp_name}/iter_{iteration}"
    makedir(result_dir)
    
    return result_dir, exp_name

def transfer_namespace(raw_arg_dict):
    """将字典转换为命名空间对象
    
    Args:
        raw_arg_dict: 原始参数字典
        
    Returns:
        命名空间对象
    """
    namespace = types.SimpleNamespace()
    for k, v in raw_arg_dict.items():
        setattr(namespace, k, v)
    return namespace