import importlib
from typing import Any, Type, Optional
import torch.nn as nn
import pytorch_lightning as pl
import os
from argparse import Namespace

def task_factory(
    args_task: Namespace,      # Task config (Namespace)
    network: nn.Module,
    args_data: Namespace,      # Data args (Namespace)
    args_model: Namespace,     # Model args (Namespace)
    args_trainer: Namespace,   # Training args (Namespace) - Renamed from args_t
    args_environment: Namespace, # Environment args (Namespace)
    metadata: Any              # Metadata object/dict
) -> Optional[pl.LightningModule]: # Return Optional since we removed raises
    """
    任务模块工厂函数 (调试简化版)

    根据配置动态加载并实例化任务模块 (PyTorch Lightning Module)。
    错误将打印到控制台而不是引发异常。

    Args:
        args_task: 任务配置对象 (Namespace)。应包含 'type' 和 'name'。
                   可选 'class_name'。
        network: 实例化的主干网络。
        args_data: 数据参数 (Namespace)。
        args_model: 模型参数 (Namespace)。
        args_trainer: 训练参数 (Namespace)。
        metadata: 数据元信息。

    Returns:
        实例化的 PyTorch Lightning 任务模块，如果发生错误则返回 None。
    """
    # 1. 检查必要的配置属性
    task_name = args_task.name
    task_type = args_task.type

    # Module path construction:
    # Default_task.py is assumed to be directly under src/task_factory/
    # Other tasks are under src/task_factory/task/<task_type>/<task_name>.py
    if task_type == 'Default_task' or task_name == 'Default_task':
        # Handles Default_task.py at src/task_factory/Default_task.py
        # Module path example: src.task_factory.Default_task
        full_module_path = f"src.task_factory.{task_name}"
    else:
        # Handles tasks in the new structure: src/task_factory/task/<type>/<name>.py
        # Module path example: src.task_factory.task.DG.Classification
        full_module_path = f"src.task_factory.task.{task_type}.{task_name}"
    
    task_module = importlib.import_module(full_module_path)



    print(f"成功导入模块: {full_module_path}")


    # 6. 实例化任务类

    # 注意: 确保 TaskClass 的 __init__ 接收这些参数名
    task_instance = task_module.task(
    network=network,
    args_data=args_data,  # Data args (Namespace)
    args_model=args_model,  # Model args (Namespace)
    args_task=args_task,  # Training args (Namespace)
    args_trainer=args_trainer,  # Trainer args (Namespace)
    args_environment=args_environment,  # Environment args (Namespace)
    metadata=metadata)
    print(f"成功实例化任务类: {full_module_path}")
    return task_instance


# 更新 __main__ 测试用例
if __name__ == '__main__':
    from argparse import Namespace
    import torch.nn as nn
    from typing import Dict, List, Any, Optional

    # 使用正确的变量名 args_trainer
    trainer_args = Namespace( # Renamed from train_args
        optimizer='adam',
        lr=1e-3,
        weight_decay=0.0,
        monitor="val_total_loss",
        patience=10,
        cla_loss="CE",
        metrics=["acc", "f1"],
        regularization={'flag': True, 'method': {'l2': 0.001}},
        scheduler={'name': 'reduceonplateau', 'options': {'patience': 5}},
        max_epochs=50
    )

    model_args = Namespace(
        input_dim=128,
        name='DummyModel',
        type='SimpleFC'
    )

    data_args = Namespace(
        task={'mydataset': {'n_classes': 10, 'path': '/path/to/data'}},
        batch_size=32
    )

    metadata = {'info': 'some metadata', 'data_names': ['mydataset']}

    # 模拟网络
    class DummyModel(nn.Module):
        def __init__(self, args_m, args_d):
            super().__init__()
            n_classes = 2 # Default
            if isinstance(args_d.task, dict) and args_d.task:
                 first_task_name = list(args_d.task.keys())[0]
                 n_classes = args_d.task[first_task_name].get('n_classes', 2)
            else:
                 print("警告 (__main__): data_args.task 不是有效字典，n_classes 默认为 2")
            self.fc = nn.Linear(args_m.input_dim, n_classes)
        def forward(self, x): return self.fc(x)

    dummy_network = DummyModel(model_args, data_args)

    print("\n--- 测试 Task Factory (使用 Default_task) ---")
    # 假设 Default_task.py 在 src/task_factory/ 下
    # 并且类名是 Default_task
    task_args_default = Namespace(
        type='Default_task', # 假设 type 和 name 相同，指向文件
        name='Default_task',
        class_name='Default_task' # 明确指定类名
    )

    # 运行工厂函数
    task_instance = task_factory(
        args_task=task_args_default,
        network=dummy_network,
        args_data=data_args,
        args_model=model_args,
        args_trainer=trainer_args, # 使用正确的变量名
        metadata=metadata
    )

    if task_instance:
        print(f"Task Factory 创建成功: {type(task_instance)}")
        # 检查 metadata 是否正确传递 (需要 Default_task 有 self.metadata)
        if hasattr(task_instance, 'metadata'):
             print(f"Task Metadata: {task_instance.metadata}")
        else:
             print("Task instance does not have 'metadata' attribute.")
    else:
        print("Task Factory 创建失败。")


    print("\n--- 测试不存在的任务模块 ---")
    task_args_nonexistent = Namespace(
        type='NonExistent',
        name='NonExistentTask',
        class_name='NonExistentTask'
        )
    task_instance_bad = task_factory(
        args_task=task_args_nonexistent,
        network=dummy_network,
        args_data=data_args,
        args_model=model_args,
        args_trainer=trainer_args,
        metadata=metadata
        )

    if not task_instance_bad:
        print("成功捕获到预期的错误 (模块不存在)。")
    else:
        print("错误：未捕获到预期的错误 (模块不存在)。")

    print("\n--- 测试存在的模块但类名错误 ---")
    # 假设 Default_task.py 存在，但类名不是 WrongClassName
    task_args_wrong_class = Namespace(
        type='Default_task',
        name='Default_task',
        class_name='WrongClassName' # 错误的类名
    )
    task_instance_wrong_class = task_factory(
        args_task=task_args_wrong_class,
        network=dummy_network,
        args_data=data_args,
        args_model=model_args,
        args_trainer=trainer_args,
        metadata=metadata
    )
    if not task_instance_wrong_class:
        print("成功捕获到预期的错误 (类名错误)。")
    else:
        print("错误：未捕获到预期的错误 (类名错误)。")

