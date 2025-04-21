#!/usr/bin/env python
"""
Vbench 框架测试入口点
用于测试框架各个模块的集成与功能
"""
import os
import argparse
import importlib
import sys
from dotenv import dotenv_values
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

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
    dirs_to_create = ["results", "data/processed", "data/raw", "save", "test/results"]
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


def test_module(module_name: str, test_function: Optional[str] = None, **kwargs) -> Dict:
    """
    测试特定模块的功能
    
    Args:
        module_name: 模块名称，如'data_factory', 'model_factory'等
        test_function: 要测试的特定函数，如果为None则测试整个模块
        **kwargs: 传递给测试函数的参数
    
    Returns:
        测试结果
    """
    print(f"\n{'='*60}")
    print(f" 测试模块: {module_name} - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        # 动态导入模块
        module_path = f"src.{module_name}"
        module = importlib.import_module(module_path)
        
        # 如果指定了特定函数，则测试该函数
        if test_function:
            if hasattr(module, test_function):
                test_func = getattr(module, test_function)
                result = test_func(**kwargs)
                success = True
            else:
                print(f"[ERROR] 模块 {module_name} 中没有找到函数 {test_function}")
                result = None
                success = False
        # 否则尝试调用模块的test函数
        elif hasattr(module, "test"):
            result = module.test(**kwargs)
            success = True
        # 最后尝试导入test子模块
        else:
            try:
                test_module = importlib.import_module(f"{module_path}.test")
                result = test_module.run(**kwargs)
                success = True
            except (ImportError, AttributeError) as e:
                print(f"[ERROR] 模块 {module_name} 没有提供测试接口: {str(e)}")
                result = None
                success = False
        
        print(f"\n{'='*60}")
        print(f" 模块 {module_name} 测试 {'成功' if success else '失败'} - 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        return {"success": success, "result": result}
    
    except Exception as e:
        print(f"[ERROR] 测试模块 {module_name} 时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def list_available_modules() -> List[str]:
    """
    列出所有可用的模块
    
    Returns:
        可用模块列表
    """
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    modules = []
    
    # 遍历src目录下的所有文件夹
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        # 如果是目录且包含__init__.py文件，认为是模块
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "__init__.py")):
            modules.append(item)
    
    return modules


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
    parser.add_argument('--module',
                       type=str,
                       help='要测试的特定模块，如data_factory、model_factory等')
    parser.add_argument('--function',
                       type=str,
                       help='要测试的特定函数')
    parser.add_argument('--list_modules',
                       action='store_true',
                       help='列出所有可用的模块')
    parser.add_argument('--all_modules',
                       action='store_true',
                       help='测试所有可用的模块')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_env()
    
    if args.list_modules:
        modules = list_available_modules()
        print("\n可用模块:")
        for module in modules:
            print(f"  - {module}")
        return
    
    if args.module:
        # 测试特定模块
        test_module(args.module, args.function)
        return
    
    if args.all_modules:
        # 测试所有模块
        modules = list_available_modules()
        results = {}
        for module in modules:
            results[module] = test_module(module)
        
        # 打印汇总结果
        print("\n模块测试汇总:")
        for module, result in results.items():
            status = "成功" if result["success"] else "失败"
            print(f"  - {module}: {status}")
        
        return
    
    if not args.setup_only:
        # 运行框架测试
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