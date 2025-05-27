import os
import numpy as np
import scipy.io as io

def read(file_path):
    matdata = io.loadmat(file_path)  # 从.mat中加载所有列表数据，返回为字典类型
    matdata = matdata[file_path[37:-4]]  # 访问第一层结构体
    matdata = matdata['Y'][0]  # 访问第二层结构体
    matdata = matdata[0].T  # 访问第三层结构体
    matdata_current_1 = matdata[1]
    matdata_current_1 = matdata_current_1['Data'][0]
    matdata_current_2 = matdata[2]
    matdata_current_2 = matdata_current_2['Data'][0]
    matdata_vibration = matdata[6]
    matdata_vibration = matdata_vibration['Data'][0]
    matdata = np.concatenate((matdata_current_1, matdata_current_2, matdata_vibration), axis=0)
    return matdata.T  # 返回数据


if __name__ == "__main__":
    path_ori = 'D:/Bench_dataset/RM_027_KAT(PU)/'
    # 使用循环遍历数据集
    working_condition = ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']
    fault_name = ['K001', 'KA04', 'KI04']
    datasets_list = []
    for i in range(4):
        for j in range(3):
            for k in range(20):
                datasets_list.append(path_ori + fault_name[j] + '/' + working_condition[i] + '_' + fault_name[j] + '_' + str(k+1) + '.mat')
    for path in datasets_list:
        print(path)
        data = read(path)
        print(data.shape)
    