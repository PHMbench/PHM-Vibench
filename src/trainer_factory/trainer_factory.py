import importlib
from typing import Any, Optional
import pytorch_lightning as pl
import os
from argparse import Namespace

def trainer_factory(
        args_environment: Namespace,  # 环境参数 (Namespace)
    args_trainer: Namespace,  # 训练参数 (Namespace)
    args_data: Namespace,     # 数据参数 (Namespace)
    path: str                 # 存储路径
) -> Optional[pl.Trainer]:    # 返回PyTorch Lightning Trainer或None
    """
    训练器工厂函数 (调试简化版)

    根据配置动态加载并实例化训练器 (PyTorch Lightning Trainer)。
    错误将打印到控制台而不是引发异常。

    Args:
        args_trainer: 训练配置对象 (Namespace)。可以包含 'name' 属性指定训练器类型。
        args_data: 数据参数 (Namespace)。
        path: 存储路径，用于保存日志、检查点等。

    Returns:
        实例化的 PyTorch Lightning 训练器，如果发生错误则返回 None。
    """
    # 1. 获取训练器名称，默认为 'Default_trainer'
    trainer_name = getattr(args_trainer, 'trainer_name', 'Default_trainer')

    try:
        # 2. 导入指定的训练器模块
        # 模块路径: src.trainer_factory.<trainer_name>
        # 例如: src.trainer_factory.Default_trainer
        full_module_path = f"src.trainer_factory.{trainer_name}"
        trainer_module = importlib.import_module(full_module_path)

        print(f"成功导入训练器模块: {full_module_path}")

        # 3. 实例化训练器
        trainer_instance = trainer_module.trainer(
            args_e=args_environment,
            args_t=args_trainer,
            args_d=args_data,
            path=path
        )
        
        print(f"成功实例化训练器: {trainer_name}")
        return trainer_instance
        
    except ImportError as e:
        print(f"错误: 无法导入训练器模块 '{full_module_path}': {str(e)}")
        return None
    except AttributeError as e:
        print(f"错误: 训练器模块 '{full_module_path}' 不包含 'trainer' 函数: {str(e)}")
        return None
    except Exception as e:
        print(f"错误: 实例化训练器时出现未知错误: {str(e)}")
        return None


# 测试代码
if __name__ == '__main__':
    from argparse import Namespace
    import os

    # 模拟参数
    trainer_args = Namespace(
        trainer_name='Default_trainer',  # 指定使用默认训练器
        device='cuda',
        gpus=1,
        n_epochs=100,
        patience=10,
        wandb=False,
        pruning=False
    )

    data_args = Namespace(
        task_name='test_task'
    )

    # 创建临时路径用于测试
    test_path = os.path.join(os.path.dirname(__file__), 'test_save')
    os.makedirs(test_path, exist_ok=True)

    # 运行工厂函数
    trainer = trainer_factory(
        args_trainer=trainer_args,
        args_data=data_args,
        path=test_path
    )

    if trainer:
        print(f"训练器创建成功: {type(trainer)}")
    else:
        print("训练器创建失败。")