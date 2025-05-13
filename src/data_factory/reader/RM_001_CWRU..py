from utils import load_data
import os
import numpy as np


def read(file_path):
    """
    读取CWRU数据集，提取DE_time和FE_time数据并在C维度拼接
    """
    # 获取文件名（不包括扩展名）
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"处理文件: {file_name}")
    
    # 读取数据
    matdata = load_data(file_path, file_type='mat')
    if matdata is None:
        print(f"无法读取文件: {file_path}")
        return None, None
    
    # 查找包含DE_time的变量
    de_vars = [key for key in matdata.keys() if 'DE_time' in key]
    fe_vars = [key for key in matdata.keys() if 'FE_time' in key]
    
    if not de_vars:
        print(f"文件中未找到DE_time变量")
        return None, None
    
    # 如果存在多个DE_time变量，选择包含文件名的那个
    selected_de = None
    for var in de_vars:
        if file_name in var:
            selected_de = var
            break
    
    # 如果没有找到包含文件名的变量，使用第一个
    if not selected_de:
        selected_de = de_vars[0]
        print(f"未找到包含文件名'{file_name}'的DE_time变量，使用{selected_de}")
    
    de_data = matdata[selected_de][:, 0]
    print(f'DE_time: {selected_de} 包含 {de_data.shape[0]} 个数据')
    
    # 同样处理FE_time
    fe_data = None
    if fe_vars:
        selected_fe = None
        for var in fe_vars:
            if file_name in var:
                selected_fe = var
                break
        
        if not selected_fe:
            selected_fe = fe_vars[0]
            print(f"未找到包含文件名'{file_name}'的FE_time变量，使用{selected_fe}")
        
        fe_data = matdata[selected_fe][:, 0]
        
        
        print(f'FE_time: {selected_fe} 包含 {fe_data.shape[0]} 个数据')
    else:
        print(f"文件中未找到FE_time变量")
    
    # 整合数据
    if fe_data is not None:
        # 确保两个数组长度相同
        min_length = min(len(de_data), len(fe_data))
        signal_data = np.column_stack((de_data[:min_length], fe_data[:min_length]))
        print(f"拼接后数据形状: {signal_data.shape}")
    else:
        # 确保输出是二维数组
        signal_data = de_data.reshape(-1, 1)
        print(f"仅DE_time数据形状: {signal_data.shape}")
    
    return signal_data

if __name__ == "__main__":
    file_path = "/home/user/data/PHMbenchdata/RM_001_CWRU/98.mat"
    data = read(file_path)
    print(data)