
import os
import numpy as np
import pandas as pd
from .utils import load_data, fix_byte_order

def read(file_path, *args):
    """
    读取RM_011_pump数据集，支持.xlsx和.csv格式
    """
    try:
        # 根据文件扩展名选择读取方式
        if file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path, header=None).values
        elif file_path.endswith('.csv'):
            data = pd.read_csv(file_path, header=None).values
        else:
            # 尝试作为csv读取
            data = pd.read_csv(file_path, header=None).values
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    # 修复字节序问题
    data = fix_byte_order(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data

if __name__ == "__main__":
    # 测试数据读取
    file_path = '/home/user/data/PHMbenchdata/raw/RM_011_pump/NU205/Data mentioned in Table 2/Defect free/Acoustic/Book1.xlsx'
    data = read(file_path)
    print(f"Data shape: {data.shape}")
    print(f"Data sample: {data[:5]}")