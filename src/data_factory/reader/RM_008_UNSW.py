<<<<<<< HEAD
from utils import load_data
import os
import numpy as np
from utils import fix_byte_order

def read(file_path,*args):
    data = load_data(file_path, file_type='mat')
    # Define keys in specific order
    keys = ["accH", "accV", "enc1", "enc2", "loadCell", "tacho"]
    # Extract arrays from the dictionary
    arrays = [data[key] for key in keys if key in data]
    # Concatenate along the last dimension (C)
    data = np.concatenate(arrays, axis=1)
    
    data = fix_byte_order(data)
    

    return data
if __name__ == "__main__":
    file_path = "/home/user/data/a_bearing/a_008_UNSW_raw/Test 2/6Hz/vib_000005664_06.mat"
    data = read(file_path)
    print(data)
    
    file_path = "/home/user/data/a_bearing/a_008_UNSW_raw/Test 3/Multiple speeds/vib_000334687_15.mat"
    data = read(file_path)
    print(data.shape)
    
=======
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

    
>>>>>>> 242639c1139e19fe4d875f9b51427781d600c8e0
