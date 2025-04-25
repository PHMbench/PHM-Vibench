import argparse
import os
import pandas as pd
import wandb
from pytorch_lightning import seed_everything

from src.utils.config_utils import load_config, path_name, transfer_namespace
from src.data_factory import build_dataset
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer

def pipeline(config_path='configs/demo/basic.yaml'):
    """默认流水线执行入口，使用工厂模式调用各个组件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        所有迭代的实验结果列表
    """
    # -----------------------
    # 1. 加载配置文件
    # -----------------------
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
    args_data.name = configs['data'].get('name', 'default')

    args_model = transfer_namespace(configs.get('model', {}).get('args', {}))
    args_model.name = configs['model'].get('name', 'default')

    args_task = transfer_namespace(configs.get('task', {}).get('args', {}))
    args_task.name = configs['task'].get('name', 'default')

    args_trainer = transfer_namespace(configs.get('trainer', {}).get('args', {}))
    args_trainer.name = configs['trainer'].get('name', 'default')
    
    for key, value in configs['environment'].items():
        if key.isupper():
            os.environ[key] = str(value)
            print(f"[INFO] 设置环境变量: {key}={value}")
    
    # -----------------------
    # 2. 多次迭代训练与测试
    # -----------------------
    all_results = []
    
    for it in range(args_environment.iterations):
        print(f"\n{'='*50}\n[INFO] 开始实验迭代 {it+1}/{args_environment.iterations}\n{'='*50}")
        
        # 设置路径和名称
        path, name = path_name(configs, it)
        
        # 设置随机种子
        current_seed = args_environment.seed + it
        seed_everything(current_seed)
        print(f"[INFO] 设置随机种子: {current_seed}")
        
        # 初始化 WandB
        if args_environment.WANDB_MODE:
            project_name = getattr(args_trainer, 'project', 'vbench')
            wandb.init(project=project_name, name=name, notes=args_environment.notes)
        else:
            wandb.init(mode='disabled')  # 避免 wandb 报错
        
        # 按照 data -> model -> task -> trainer 的顺序构建组件
        
        # 1. 构建数据集
        print(f"[INFO] 构建数据集")
        dataset = build_dataset(args_data)
        
        # 2. 构建模型
        print(f"[INFO] 构建模型: {args_model.name}")
        model = build_model(configs['model'])
        
        # 3. 构建任务
        print(f"[INFO] 构建任务: {configs['task']['name']}")
        task_config = configs['task'].copy()
        task_config['args'] = task_config.get('args', {})
        task_config['args']['data_config'] = configs['data']
        task = build_task(task_config)
        
        # 设置数据集和模型到任务中
        task.set_dataset(dataset)
        task.set_model(model)
        
        # 从任务中获取适配特定任务的数据加载器
        print(f"[INFO] 从任务中获取数据加载器")
        train_loader = task.get_train_loader()
        val_loader = task.get_val_loader()
        test_loader = task.get_test_loader()
        
        # 从任务中获取损失函数和其他评估指标
        loss_fn = task.get_loss_function()
        metrics = task.get_metrics()
        
        # 4. 构建训练器
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
            use_wandb=args_environment.use_wandb,
            notes=args_environment.notes,
            save_path=path,
            configs=configs,
            args_trainer=args_trainer,
            args_model=args_model,
            args_data=args_data,
            args_task=args_task,
            iteration=it
        )
        all_results.append(result)
        
        # 保存结果
        result_df = pd.DataFrame([result])
        result_df.to_csv(os.path.join(path, f'test_result_{it}.csv'), index=False)
        # 确保 wandb 正确关闭
        if args_environment.WANDB_MODE:
            wandb.finish()
    
    print(f"\n{'='*50}\n[INFO] 所有实验已完成\n{'='*50}")
    pd.DataFrame(all_results).to_csv(os.path.join(path, 'all_results.csv'), index=False)
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