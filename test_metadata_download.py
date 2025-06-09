#!/usr/bin/env python3
"""
测试元数据自动下载功能
"""
import os
import sys
import tempfile
import shutil

# 添加项目路径到 Python 路径
sys.path.append('/home/lq/LQcode/2_project/PHMBench/PHM-Vibench')

from src.data_factory.data_utils import check_and_download_metadata, download_metadata


def test_download_functions():
    """测试下载功能"""
    
    # 创建临时目录进行测试
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"临时测试目录: {temp_dir}")
        
        # 测试1: 直接下载功能
        print("\n=== 测试 download_metadata 函数 ===")
        success = download_metadata(
            metadata_file="metadata_6_1.xlsx",
            save_path=temp_dir,
            source='auto'
        )
        print(f"下载结果: {'成功' if success else '失败'}")
        
        # 检查下载的文件
        downloaded_file = os.path.join(temp_dir, "metadata_6_1.xlsx")
        if os.path.exists(downloaded_file):
            print(f"文件已下载到: {downloaded_file}")
            print(f"文件大小: {os.path.getsize(downloaded_file)} bytes")
        else:
            print("文件下载失败或不存在")
        
        # 测试2: 检查并下载功能（文件已存在的情况）
        print("\n=== 测试 check_and_download_metadata 函数（文件已存在）===")
        try:
            result_path = check_and_download_metadata(temp_dir, "metadata_6_1.xlsx")
            print(f"函数返回路径: {result_path}")
        except Exception as e:
            print(f"检查和下载失败: {e}")
        
        # 删除文件，测试自动下载
        if os.path.exists(downloaded_file):
            os.remove(downloaded_file)
            print(f"已删除文件: {downloaded_file}")
        
        # 测试3: 检查并下载功能（文件不存在的情况）
        print("\n=== 测试 check_and_download_metadata 函数（文件不存在）===")
        try:
            result_path = check_and_download_metadata(temp_dir, "metadata_6_1.xlsx")
            print(f"自动下载成功，文件路径: {result_path}")
        except Exception as e:
            print(f"自动下载失败: {e}")


if __name__ == "__main__":
    test_download_functions()
