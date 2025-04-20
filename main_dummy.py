#!/usr/bin/env python
"""
Vbench 框架测试入口点
用于测试框架各个模块的集成与功能
"""
import os
import argparse
from dotenv import dotenv_values
from datetime import datetime
from typing import Dict, Any

from src.Pipeline_01_default import main as pipeline_main

def setup_env():
    """
    设置环境变量
    创建必要的目录结构
    """
    # 加载环境变量
    env_config = dotenv_values(".env")
    if not env_config:
        print("[INFO] 未找到 .env 文件，使用默认环境配置")
        # 设置一些默认配置
        os.environ["WANDB_MODE"] = "disabled"  # 默认禁用 wandb
        os.environ["VBENCH_HOME"] = os.path.abspath(".")
    else:
        # 将环境变量写入系统环境
        for key, value in env_config.items():
            os.environ[key] = value
    
    # 创建必要的目录
    dirs_to_create = ["results", "data/processed", "data/raw", "save"]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    # 打印环境信息
    print(f"[INFO] VBENCH_HOME: {os.environ.get('VBENCH_HOME', os.path.abspath('.'))}")
    print(f"[INFO] WANDB_MODE: {os.environ.get('WANDB_MODE', 'disabled')}")


def test_framework(config_path, iterations=1):
    """
    测试框架功能
    
    Args:
        config_path: 配置文件路径
        iterations: 实验重复次数
    
    Returns:
        测试结果
    """
    print(f"\n{'='*60}")
    print(f" Vbench 框架测试 - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 调用流水线
    results = pipeline_main(
        config_path=config_path,
        iterations=iterations,
        use_wandb=False,  # 测试时禁用 wandb
        notes="框架测试运行",
        seed=42
    )
    
    print(f"\n{'='*60}")
    print(f" 框架测试 {'成功' if results else '失败'} - 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Vbench 框架测试程序')
    
    parser.add_argument('--config', 
                       type=str, 
                       default='configs/demo/dummy_test.yaml',
                       help='测试配置文件路径')
    parser.add_argument('--iterations', 
                       type=int, 
                       default=1,
                       help='测试迭代次数')
    parser.add_argument('--setup_only', 
                       action='store_true',
                       help='仅设置环境，不运行测试')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_env()
    
    if not args.setup_only:
        # 运行测试
        test_results = test_framework(
            config_path=args.config,
            iterations=args.iterations
        )
        
        if test_results:
            print("\n[SUCCESS] 框架测试成功完成!")
            
            # 打印部分结果
            print("\n测试指标:")
            for key, value in test_results[0].items():
                print(f"  {key}: {value}")
        else:
            print("\n[ERROR] 框架测试未返回预期结果")
    else:
        print("\n[INFO] 环境设置完成，跳过测试运行")


if __name__ == "__main__":
    main()