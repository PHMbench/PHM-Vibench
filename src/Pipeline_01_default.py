import argparse
import os
import pandas as pd
import wandb
from typing import Dict, Any, List, Optional
from pytorch_lightning import seed_everything

from src.utils.config_utils import load_config, path_name, transfer_namespace
from src.data_factory import build_dataset
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer

def pipeline(config_path='configs/demo/basic.yaml', 
         iterations=1, 
         use_wandb=False, 
         notes='', 
         seed=42):
    """默认流水线执行入口，使用工厂模式调用各个组件
    
    Args:
        config_path: 配置文件路径
        iterations: 实验重复次数
        use_wandb: 是否使用 WandB
        notes: 实验备注
        seed: 随机种子
        
    Returns:
        所有迭代的实验结果列表
    """
    # -----------------------
    # 1. 加载配置文件
    # -----------------------
    print(f"[INFO] 加载配置文件: {config_path}")
    configs = load_config(config_path)
    
    # 确保配置中包含必要的部分
    required_sections = ['dataset', 'model', 'task', 'trainer']
    for section in required_sections:
        if section not in configs:
            print(f"[ERROR] 配置文件中缺少 {section} 部分")
            return
    
    # 准备命名空间参数
    args_trainer = transfer_namespace(configs['trainer'].get('args', {}))
    args_model = transfer_namespace(configs['model'].get('args', {}))
    args_dataset = transfer_namespace(configs['dataset'].get('args', {}))
    args_task = transfer_namespace(configs['task'].get('args', {}))
    
    # -----------------------
    # 2. 多次迭代训练与测试
    # -----------------------
    all_results = []
    
    for it in range(iterations):
        print(f"\n{'='*50}\n[INFO] 开始实验迭代 {it+1}/{iterations}\n{'='*50}")
        
        # 设置路径和名称
        path, name = path_name(configs, it)
        
        # 设置随机种子
        current_seed = seed + it
        seed_everything(current_seed)
        print(f"[INFO] 设置随机种子: {current_seed}")
        
        # 初始化 WandB
        if use_wandb:
            project_name = getattr(args_trainer, 'project', 'vbench')
            wandb.init(project=project_name, name=name, notes=notes)
        else:
            wandb.init(mode='disabled')  # 避免 wandb 报错
            
        try:
            # 构建数据集
            print(f"[INFO] 构建数据集: {configs['dataset']['name']}")
            dataset = build_dataset(configs['dataset'])
            
            # 构建模型
            print(f"[INFO] 构建模型: {configs['model']['name']}")
            model = build_model(configs['model'])
            
            # 构建任务（将模型和数据集都传递给任务，实现任务特定的数据包装）
            print(f"[INFO] 构建任务: {configs['task']['name']}")
            task_config = configs['task'].copy()
            task_config['args'] = task_config.get('args', {})
            task_config['args']['model'] = model
            task_config['args']['dataset'] = dataset  # 将原始数据集传递给任务
            
            task = build_task(task_config)
            
            # 从任务中获取适配特定任务的数据加载器
            print(f"[INFO] 从任务中获取数据加载器")
            train_loader = task.get_train_loader()
            val_loader = task.get_val_loader()
            test_loader = task.get_test_loader()
            
            # 从任务中获取损失函数和其他评估指标
            loss_fn = task.get_loss_function()
            metrics = task.get_metrics()
            
            # 构建训练器
            print(f"[INFO] 构建训练器: {configs['trainer']['name']}")
            trainer_config = configs['trainer'].copy()
            trainer = build_trainer(trainer_config)
            
            # 执行训练和评估
            print(f"[INFO] 开始训练 (迭代 {it+1})")
            result = trainer(
                task=task,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                loss_function=loss_fn,
                metrics=metrics,
                use_wandb=use_wandb,  # 直接使用函数参数
                notes=notes,  # 直接使用函数参数
                save_path=path,
                configs=configs,
                args_trainer=args_trainer,
                args_model=args_model,
                args_dataset=args_dataset,
                args_task=args_task,
                iteration=it
            )
            all_results.append(result)
            
            # 保存结果
            result_df = pd.DataFrame([result])
            result_df.to_csv(os.path.join(path, f'test_result_{it}.csv'), index=False)
            
        finally:
            # 确保 wandb 正确关闭
            if use_wandb:
                wandb.finish()
    
    print(f"\n{'='*50}\n[INFO] 所有实验已完成\n{'='*50}")
    return all_results


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Vbench 默认流水线')
    
    parser.add_argument('--config_path', 
                        type=str,
                        default='configs/demo/basic.yaml',
                        help='配置文件路径')
    parser.add_argument('--iterations', 
                        type=int, 
                        default=1,
                        help='实验重复次数')
    parser.add_argument('--notes', 
                        type=str, 
                        default='实验备注',
                        help='实验备注')
    parser.add_argument('--use_wandb', 
                        action='store_true',
                        help='是否使用 WandB 记录实验')
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 执行流水线
    pipeline(
        config_path=args.config_path,
        iterations=args.iterations,
        use_wandb=args.use_wandb,
        notes=args.notes,
        seed=args.seed
    )