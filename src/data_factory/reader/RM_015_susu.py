
import os
import numpy as np
import scipy.io

def read(file_path,*args):
    # 从file_path中获取文件名
    file_name = os.path.basename(file_path)
    frequency = file_name.split('_')[-1].split('.')[0]  # 获取频率部分
    fault_type = file_name.split('_')[0]  # 获取条件部分
    if frequency == '18Hz':
        struct_name = f"Device_Bearing{fault_type}_{frequency}_Time_Axis"  # 构造结构体名称
        data = scipy.io.loadmat(file_path)[struct_name]
    elif frequency == '20Hz':
        data = scipy.io.loadmat(file_path)['Device_Time_Axis']
    data = data.T
            
    return data

if __name__ == "__main__":
    file_path = "D:/Bench_dataset/RM_015_susu/Data_18Hz/Normal_1_inch_18Hz.mat"
    data = read(file_path)
    print(data.shape)
    