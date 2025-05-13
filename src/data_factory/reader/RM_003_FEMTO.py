import os
import pandas as pd

def read(file_path):
    """
    Reads data from a .mat file specified by file_path.
    
    Args:
        file_path (str): Path to the .mat data file (e.g., Vbench/data/RM_002_XJTU/1.mat).
    
    Returns:
        numpyarray: dimension as length \times channel
    """
    # 读取数据
    if 'Bearing1_4' in file_path: # solve a BUG from dataset
        raw_data = pd.read_csv(file_path,header=None)[0].str.split(';', expand=True).loc[:,4:6]
    else:
        
        raw_data = pd.read_csv(file_path,header=None).loc[:,4:6] # NO sep='\t' for FEMTO
    
    # 整合数据
    return raw_data.values

if __name__ == "__main__":
    file_path = "/home/user/data/PHMbenchdata/RM_003_FEMTO/Learning_set/Bearing1_2/acc_00001.csv"
    data = read(file_path)
    print(data)
    
    file_path = "/home/user/data/PHMbenchdata/RM_003_FEMTO/Test_Set/Bearing1_4/acc_00001.csv"
    data = read(file_path)
    print(data)