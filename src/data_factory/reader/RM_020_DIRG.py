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
 
    data = load_data(file_path, file_type='mat')  # 修复拼写错误
    file_name = file_path.split('/')[-1].split('.')[0]
    data = data[file_name]
    # data = pd.DataFrame(data)
    # data.columns = ['A1_x', 'A1_y', 'A1_z', 'A2_x', 'A2_y', 'A2_z']
    
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
                 name = 'RM_020_DIRG',
                 output_dir = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/src/data_factory/reader/output',
                 read=read)
