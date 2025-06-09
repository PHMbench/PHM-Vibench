#!/usr/bin/env python3
"""
测试改进后的元数据检查和下载功能
"""
import os
import sys
import tempfile
import shutil

# 添加项目路径到 Python 路径
sys.path.append('/home/lq/LQcode/2_project/PHMBench/PHM-Vibench')

from src.data_factory.data_utils import check_and_download_metadata


def test_with_existing_files():
    """测试使用项目中现有的元数据文件"""
    
    data_dir = "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/data"
    
    # 测试1: 查找不存在的文件，应该返回可用的替代文件
    print("=== 测试查找不存在的文件 ===")
    try:
        result_path = check_and_download_metadata(data_dir, "metadata_6_1.xlsx")
        print(f"✅ 成功返回路径: {result_path}")
        
        # 验证文件确实存在
        if os.path.exists(result_path):
            print(f"✅ 确认文件存在，大小: {os.path.getsize(result_path)} bytes")
        else:
            print("❌ 返回的文件路径不存在")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    # 测试2: 查找存在的文件
    print("\n=== 测试查找存在的文件 ===")
    try:
        result_path = check_and_download_metadata(data_dir, "metadata_5_29.xlsx")
        print(f"✅ 成功返回路径: {result_path}")
        
        if os.path.exists(result_path):
            print(f"✅ 确认文件存在，大小: {os.path.getsize(result_path)} bytes")
        else:
            print("❌ 返回的文件路径不存在")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def test_data_factory_integration():
    """测试与 data_factory 的集成"""
    
    print("\n=== 测试 data_factory 集成 ===")
    
    # 创建一个简单的命名空间类来模拟 args_data
    class SimpleNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    try:
        # 模拟 args_data
        args_data = SimpleNamespace(
            data_dir="/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/data",
            metadata_file="metadata_6_1.xlsx"  # 不存在的文件，应该找到替代品
        )
        
        # 导入并测试 data_factory 的元数据初始化
        from src.data_factory.data_factory import data_factory
        
        # 只测试元数据初始化部分
        factory = data_factory.__new__(data_factory)  # 创建实例但不调用完整的 __init__
        metadata = factory._init_metadata(args_data)
        
        print(f"✅ 成功初始化元数据，记录数: {len(metadata)}")
        print(f"✅ 元数据键示例: {list(metadata.keys())[:5]}")
        
    except Exception as e:
        print(f"❌ data_factory 集成测试失败: {e}")


if __name__ == "__main__":
    test_with_existing_files()
    test_data_factory_integration()
