import argparse
import os
import sys
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.config_utils import load_config, path_name, transfer_namespace
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab, get_num_classes
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer



def pipeline(args):
    """领域泛化(Domain Generalization)任务的流水线
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        所有迭代的实验结果列表
    """
    # -----------------------
    # 1. 加载配置文件
    # -----------------------
    config_path = args.config_path
    print(f"[INFO] 加载配置文件: {config_path}")
    configs = load_config(config_path)
    
    # 确保配置中包含必要的部分
    required_sections = ['data', 'model', 'task', 'trainer', 'environment']
    for section in required_sections:
        if section not in configs:
            print(f"[ERROR] 配置文件中缺少 {section} 部分")
            return
    
    # 设置环境变量和命名空间
    args_environment = transfer_namespace(configs.get('environment', {}))

    args_data = transfer_namespace(configs.get('data', {}))

    args_model = transfer_namespace(configs.get('model', {}))

    args_task = transfer_namespace(configs.get('task', {}))

    args_trainer = transfer_namespace(configs.get('trainer', {}))
    if args_task.name == 'Multitask':
        args_data.task_list = args_task.task_list
        args_model.task_list = args_task.task_list    
    for key, value in configs['environment'].items():
        if key.isupper():
            os.environ[key] = str(value)
            print(f"[INFO] 设置环境变量: {key}={value}")
    
    # 创建实验目录
    print("[INFO] 创建实验目录...")
    VBENCH_HOME = args_environment.VBENCH_HOME
    VBENCH_DATA = args_data.data_dir
    sys.path.append(VBENCH_HOME)
    sys.path.append(VBENCH_DATA)
    
    # -----------------------
    # 2. 多次迭代训练与测试
    # -----------------------
    all_results = []
    
    for it in range(args_environment.iterations):
        print(f"\n{'='*50}\n[INFO] 开始实验迭代 {it+1}/{args_environment.iterations}\n{'='*50}")
        
        # 设置路径和名称
        path, name = path_name(configs, it)
        # 把name 加到args_trainer中
        args_trainer.logger_name = name
        # 设置随机种子
        current_seed = args_environment.seed + it
        seed_everything(current_seed)
        print(f"[INFO] 设置随机种子: {current_seed}")
        init_lab(args_environment, args, name)


        # 构建数据工厂
        print("[INFO] 构建数据工厂...")
        data_factory = build_data(args_data, args_task)
        # 构建模型
        print("[INFO] 构建模型...")
        model = build_model(args_model,metadata=data_factory.get_metadata())
        
        # 构建任务
        print("[INFO] 构建任务...")
        task = build_task(
            args_task=args_task,
            network=model,
            args_data=args_data,
            args_model=args_model,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=data_factory.get_metadata()
        )
        
        # 构建训练器
        print("[INFO] 构建训练器...")
        trainer = build_trainer(
            args_environment,
            args_trainer,
            args_data,
            path
        )
        
        # 执行训练
        print("[INFO] 开始训练...")
        trainer.fit(
            task,
            data_factory.get_dataloader('train'),
            data_factory.get_dataloader('val')
        )
        
        # 加载最佳模型并测试
        print("[INFO] 加载最佳模型并测试...")
        task = load_best_model_checkpoint(task, trainer)
        result = trainer.test(task, data_factory.get_dataloader('test'))
        data_factory.data.close()  # 关闭数据工厂，释放资源
        all_results.append(result[0])  # Lightning返回的是包含字典的列表
        
        # 保存结果
        print("[INFO] 保存测试结果...")
        result_df = pd.DataFrame([result[0]])
        result_df.to_csv(os.path.join(path, f'test_result_{it}.csv'), index=False)

        # 关闭wandb和swanlab
        close_lab()

    print(f"\n{'='*50}\n[INFO] 所有实验已完成\n{'='*50}")
    pd.DataFrame(all_results).to_csv(os.path.join(path, 'all_results.csv'), index=False)
    return all_results


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="领域泛化(DG)任务流水线")
    
    parser.add_argument('--config_path', 
                        type=str, 
                        default='/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/Single_DG/CWRU.yaml',
                        help='配置文件路径')
    parser.add_argument('--notes', 
                        type=str, 
                        default='',
                        help='实验备注')

    
    args = parser.parse_args()
    
    # 执行DG流水线
    results = pipeline(args)
    print(f"完成所有实验！")