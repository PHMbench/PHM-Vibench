import os
import numpy as np
import scipy.io as io

def read(file_path):
    matdata = io.loadmat(file_path)  # 从.mat中加载所有列表数据，返回为字典类型
    matdata = matdata['bearing']  # 访问第一层结构体
    # print(matdata['load'][0][0].astype('float').item())
    matdata = matdata['gs'][0]  # 访问第二层结构体
    matdata = matdata[0] # 访问第三层结构体
    matdata = matdata.astype(float)
    return matdata  # 返回数据


if __name__ == "__main__":
    file_path = 'D:/Bench_dataset/RM_007_MFPT/2 - Three Outer Race Fault Conditions/OuterRaceFault_1.mat'
    data = read(file_path)
    print(data.shape)

    