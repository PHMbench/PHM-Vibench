import pandas as pd
import os
from .utils import fix_byte_order


def read(file_path,*args):
    """
    Reads data from a .mat file specified by file_path.
    
    Args:
        file_path (str): Path to the .mat data file (e.g., Vbench/data/RM_002_XJTU/1.mat).
    
    Returns:
        numpyarray: dimension as length \times channel
    """
    # 读取数据
    if '1st_test' in file_path:
        raw_data = pd.read_csv(file_path, sep='\t',header=None).loc[:]
    else:
        raw_data = pd.read_csv(file_path, sep='\t',header=None).loc[:]
    data = fix_byte_order(raw_data.values)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    # 整合数据
    return data

if __name__ == "__main__":
    file_path = "/home/user/data/PHMbenchdata/PHM-Vibench/raw/RM_004_IMS/1st_test/2003.10.22.12.06.24"
    data = read(file_path)
    print(data)
    
    file_path = "/home/user/data/PHMbenchdata/RM_004_IMS/2nd_test/2004.02.12.10.32.39"
    data = read(file_path)
    print(data)