import numpy as np
import pandas as pd
from .utils import fix_byte_order

def read(file_path, *args):
    """
    读取RM_017_Ottawa19数据集（Ottawa19）
    Args:
        file_path (str): 数据文件路径
    Returns:
        numpy.ndarray: 数据数组，形状为 length x channel
    """
    try:
        # 读取CSV数据，无表头
        data = pd.read_csv(file_path, header=None, low_memory=False).values
        
        # 如果第一行是字符串标题，则跳过
        if data.dtype == 'object':
            data = data[1:]
        
        # 转换为浮点数
        data = data.astype(float)
        
        # 修复字节序问题
        data = fix_byte_order(data)
        
        # 确保是二维数组
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        return data
        
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

if __name__ == "__main__":
    file_path = "/user/data/PHMbenchdata/raw/RM_017_Ottawa19/sample.csv"
    data = read(file_path)
    if data is not None:
        print(f"Data shape: {data.shape}")
        print(f"Data sample: {data[:5]}")
    else:
        print("Failed to read data")
