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

def main(config_path='configs/demo/basic.yaml', 
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
    args_t = transfer_namespace(configs['trainer'].get('args', {}))
    args_m = transfer_namespace(configs['model'].get('args', {}))
    args_d = transfer_namespace(configs['dataset'].get('args', {}))
    args_task = transfer_namespace(configs['task'].get('args', {}))
    
    # -----------------------
    # 2. 多次迭代训练与测试
    # -----------------------
    all_results = []
    
    # 将 WandB 选项和实验备注添加到训练器配置中
    configs['trainer']['args'] = configs['trainer'].get('args', {})
    configs['trainer']['args']['wandb'] = use_wandb
    configs['trainer']['args']['notes'] = notes
    
    for it in range(iterations):
        print(f"\n{'='*50}\n[INFO] 开始实验迭代 {it+1}/{iterations}\n{'='*50}")
        
        # 设置路径和名称
        path, name = path_name(configs, it)
        
        # 设置随机种子
        curr_seed = seed + it
        seed_everything(curr_seed)
        print(f"[INFO] 设置随机种子: {curr_seed}")
        
        # 初始化 WandB
        if use_wandb:
            project_name = getattr(args_t, 'project', 'vbench')
            wandb.init(project=project_name, name=name, notes=notes)
        else:
            wandb.init(mode='disabled')  # 避免 wandb 报错
            
        try:
            # 构建数据集
            print(f"[INFO] 构建数据集: {configs['dataset']['name']}")
            dataset = build_dataset(configs['dataset'])
            
            # 获取数据加载器
            train_loader = dataset.get_train_loader()
            val_loader = dataset.get_val_loader()
            test_loader = dataset.get_test_loader()
            
            # 构建模型
            print(f"[INFO] 构建模型: {configs['model']['name']}")
            model = build_model(configs['model'])
            
            # 构建任务
            print(f"[INFO] 构建任务: {configs['task']['name']}")
            # 将模型实例传递给任务
            task_config = configs['task'].copy()
            task_config['args'] = task_config.get('args', {})
            task_config['args']['model'] = model
            task = build_task(task_config)
            
            # 构建训练器（传递所有必要的参数和数据加载器）
            print(f"[INFO] 构建训练器: {configs['trainer']['name']}")
            trainer_config = configs['trainer'].copy()
            trainer = build_trainer(trainer_config)
            
            # 执行训练和评估，直接传递所有必要的组件
            print(f"[INFO] 开始训练 (迭代 {it+1})")
            result = trainer(
                dataset=dataset,
                model=model,
                task=task,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                configs=configs,
                args_t=args_t,
                args_m=args_m,
                args_d=args_d,
                args_task=args_task,
                save_path=path,
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
    main(
        config_path=args.config_path,
        iterations=args.iterations,
        use_wandb=args.use_wandb,
        notes=args.notes,
        seed=args.seed
    )