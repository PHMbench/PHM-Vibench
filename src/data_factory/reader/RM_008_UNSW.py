import os
import numpy as np
import scipy.io as io

def read(file_path):
    matdata = io.loadmat(file_path)  # 从.mat中加载所有列表数据，返回为字典类型
    accH = matdata['accH']
    accV = matdata['accV']
    enc1 = matdata['enc1']
    enc2 = matdata['enc2']
    loadcell = matdata['loadCell']
    tacho = matdata['tacho']
    # 将所有数据按列拼接
    matdata = np.concatenate((accH, accV, enc1, enc2, loadcell, tacho), axis=1)
    return matdata  # 返回数据


if __name__ == "__main__":
    file_path = 'D:/Bench_dataset/RM_008_UNSW/Test 3/6Hz/vib_000182497_06.mat'
    data = read(file_path)
    print(data.shape)

    