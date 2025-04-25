"""
测试数据工厂模块

用于验证数据工厂的功能
"""
import os
import sys
import argparse
import logging
import numpy as np
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_factory.data_factory import DataFactory
from src.data_factory.data_reader import DataReader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_reader():
    """测试数据读取器"""
    logger.info("=== 开始测试数据读取器 ===")
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # 创建数据读取器
    reader = DataReader(
        metadata_file='metadata_dummy.csv',
        data_dir=data_dir
    )
    
    # 测试元数据读取
    metadata = reader.get_metadata()
    logger.info(f"元数据读取成功，共有 {len(metadata)} 条记录")
    
    # 测试数据集加载
    datasets = reader.load_dataset()
    logger.info(f"数据集加载成功，共有 {len(datasets)} 个数据集")
    
    # 打印数据集信息
    for dataset_id, dataset in datasets.items():
        meta = reader.get_metadata(dataset_id)
        logger.info(f"数据集 {dataset_id}:")
        logger.info(f"  名称: {meta.get('Name', 'N/A')}")
        logger.info(f"  描述: {meta.get('Description', 'N/A')}")
        logger.info(f"  特征形状: {dataset['features'].shape}")
        logger.info(f"  标签形状: {dataset['labels'].shape}")
        logger.info(f"  样本率: {dataset.get('sample_rate', 'N/A')}")
        logger.info(f"  样本长度: {dataset.get('sample_length', 'N/A')}")
        logger.info(f"  通道数: {dataset.get('channels', 'N/A')}")
        logger.info("")
    
    logger.info("=== 数据读取器测试完成 ===\n")

def test_data_factory():
    """测试数据集工厂"""
    logger.info("=== 开始测试数据集工厂 ===")
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # 创建数据工厂
    factory = DataFactory(
        metadata_file='metadata_dummy.csv',
        data_dir=data_dir,
        batch_size=16,
        train_val_rate=0.8,
        seed=42
    )
    
    # 获取数据集信息
    info = factory.get_datasets_info()
    logger.info(f"数据集工厂初始化成功，共有 {len(info)} 个数据集")
    
    # 打印数据集信息
    for dataset_id, dataset_info in info.items():
        logger.info(f"数据集 {dataset_id}:")
        logger.info(f"  名称: {dataset_info.get('name', 'N/A')}")
        logger.info(f"  描述: {dataset_info.get('description', 'N/A')}")
        logger.info(f"  训练集大小: {dataset_info.get('train_size', 'N/A')}")
        logger.info(f"  验证集大小: {dataset_info.get('val_size', 'N/A')}")
        logger.info(f"  测试集大小: {dataset_info.get('test_size', 'N/A')}")
        logger.info(f"  特征维度: {dataset_info.get('feature_dim', 'N/A')}")
        logger.info(f"  标签类型: {dataset_info.get('label_type', 'N/A')}")
        logger.info("")
    
    # 测试数据加载器
    for dataset_id in info.keys():
        logger.info(f"测试数据集 {dataset_id} 的数据加载器")
        
        # 获取数据加载器
        train_loader, val_loader, test_loader = factory.get_data_loaders(dataset_id)
        
        # 查看一个批次
        for x_batch, y_batch in train_loader:
            logger.info(f"  训练批次特征形状: {x_batch.shape}")
            logger.info(f"  训练批次标签形状: {y_batch.shape}")
            logger.info(f"  训练批次特征类型: {x_batch.dtype}")
            logger.info(f"  训练批次标签类型: {y_batch.dtype}")
            logger.info(f"  训练批次特征样例: {x_batch[0, :5]}")
            logger.info(f"  训练批次标签样例: {y_batch[0]}")
            break
        
        for x_batch, y_batch in val_loader:
            logger.info(f"  验证批次特征形状: {x_batch.shape}")
            logger.info(f"  验证批次标签形状: {y_batch.shape}")
            break
            
        logger.info("")
    
    logger.info("=== 数据集工厂测试完成 ===")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试数据工厂模块')
    parser.add_argument('--test-reader', action='store_true', help='测试数据读取器')
    parser.add_argument('--test-factory', action='store_true', help='测试数据集工厂')
    parser.add_argument('--test-all', action='store_true', help='测试所有组件')
    
    args = parser.parse_args()
    
    # 如果没有指定测试项，则默认测试所有组件
    if not (args.test_reader or args.test_factory):
        args.test_all = True
    
    # 执行测试
    if args.test_reader or args.test_all:
        test_data_reader()
    
    if args.test_factory or args.test_all:
        test_data_factory()