import numpy as np
import csv
import pandas as pd
def read(file_path):
    """
    Reads data from a csv file specified by file_path.
    Args:
        file_path (str): Path to the csv data file (e.g., Vbench/data/RM_005_Ottawa23/1.csv).
    Returns:
        numpy.ndarray: dimension as length x channel
    """    
    # 读取数据
    data = pd.read_csv(file_path, header=None,low_memory=False).values
    data = data[1:]
    # print('speed:', data[0,2], 'load:', data[0,3])
    # 保留第1，2，5列数据
    data = data[:, [0, 1, 4]]
    # 将数据转换为浮点数
    data = data.astype(float)
    return data

if __name__ == "__main__":
    path_ori = 'D:/Bench_dataset/RM_005_Ottawa23/1_CSV_Raw_Data_Files (.csv)/'
    # 测试正常类型数据
    for i in range(20):
        path = path_ori + '1_Healthy/' + 'H_' + str(i+1) + '_0.csv'
        print(path)
        data = read(path)
        print(data.shape)
    # 测试故障数据
    fault_name_list = ['2_Inner_Race_Faults', '3_Outer_Race_Faults', '4_Ball_Faults', '5_Cage_Faults']
    for i in range(4):
        fault_name = fault_name_list[i]
        for j in range(5):
            for k in range(2):
                path = path_ori + fault_name + '/' + fault_name[2] + '_' + str(i*5+j+1) + '_' + str(k+1) + '.csv'
                # print(path)
                data = read(path)
                # print(data.shape)