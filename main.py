import argparse
import os
from dotenv import dotenv_values
from typing import Dict, Any

# 调用默认 pipeline
from src.Pipeline_01_default import main as default_pipeline

def main():
    """
    Vbench 主入口，配置环境变量并调用实验流水线
    """
    # 加载环境变量配置
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    env_configs = dotenv_values(env_path)
    
    # 将环境变量设置到全局环境中
    for key, value in env_configs.items():
        os.environ[key] = value
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Vbench Framework')
    
    parser.add_argument('--config_path', 
                      type=str, 
                      default='configs/demo/basic.yaml', 
                      help='配置文件路径')
    parser.add_argument('--iterations', 
                      type=int, 
                      default=1, 
                      help='实验重复次数')
    parser.add_argument('--use_wandb', 
                      action='store_true', 
                      help='是否使用WandB记录实验')
    parser.add_argument('--notes', 
                      type=str, 
                      default='', 
                      help='实验备注')
    parser.add_argument('--seed', 
                      type=int, 
                      default=42, 
                      help='随机种子')
    
    args = parser.parse_args()
    
    # 调用默认流水线
    results = default_pipeline(
        config_path=args.config_path,
        iterations=args.iterations,
        use_wandb=args.use_wandb,
        notes=args.notes,
        seed=args.seed
    )
    
    return results

if __name__ == "__main__":
    main()


