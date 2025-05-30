import numpy as np
import pandas as pd
import os
from utils import fix_byte_order,load_data

def read(file_path, *args):
    """
    读取RM_023_HIT23数据集
    Args:
        file_path (str): 数据文件路径
    Returns:
        numpy.ndarray: 数据数组，形状为 length x channel
    """
 
    mat_data = load_data(file_path, file_type='mat')
    
    # 自动识别并读取数组数据，排除matlab文件头信息
    data_arrays = []
    for key, value in mat_data.items():
        # 跳过matlab文件的元数据
        if not key.startswith('__') and isinstance(value, np.ndarray):
            data_arrays.append(value)
    
    # 如果只有一个数组，直接使用
    if len(data_arrays) == 1:
        data = data_arrays[0]
    # 如果有多个数组，按列拼接
    elif len(data_arrays) > 1:
        data = np.concatenate(data_arrays, axis=1)
    else:
        raise ValueError(f"未找到有效的数组数据在文件 {file_path}")
    
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
                 name = 'RM_023_HIT23',
                 output_dir = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/src/data_factory/reader/output',
                 read=read)
