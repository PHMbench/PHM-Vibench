import numpy as np
import pandas as pd
import os
from utils import fix_byte_order,load_data

def read(file_path, *args):
    """
    读取RM_017_Ottawa19数据集（Ottawa19）
    Args:
        file_path (str): 数据文件路径
    Returns:
        numpy.ndarray: 数据数组，形状为 length x channel
    """
 
    data = load_data(file_path, file_type='csv')  # 修复拼写错误
    data = data.iloc[:, [4, 10]].values
    # 转换为浮点数
    data = data.astype(float)
    
    # 修复字节序问题
    data = fix_byte_order(data)
    
    # 确保是二维数组
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        
    return data



if __name__ == "__main__":
    from utils import test_reader
    test_reader(metadata_path = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/metadata_5_data.csv',
                 data_dir = '/home/user/data/PHMbenchdata/raw/',
                 name = 'RM_018_THU24',
                 output_dir = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/src/data_factory/reader/output',
                 read=read)
