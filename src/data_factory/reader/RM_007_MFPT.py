from utils import load_data
import os
import numpy as np
from utils import fix_byte_order

def read(file_path,*args):
    data = load_data(file_path, file_type='mat')
    # if 'baseline' in file_path:
    #     data = data["bearing"][0][0]
    # else:
    #     data = data["bearing"][0][0][2]
    data = data['bearing']['gs'][0][0]  # 访问第三层结构体
    data = fix_byte_order(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data
if __name__ == "__main__":
    file_path = "/home/user/data/a_bearing/a_007_MFPT_raw/1 - Three Baseline Conditions/baseline_1.mat"
    data = read(file_path)
    print(data)
    
    file_path = "/home/user/data/a_bearing/a_007_MFPT_raw/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_1.mat"
    data = read(file_path)
    print(data.shape)
    
    file_path = "/home/user/data/a_bearing/a_007_MFPT_raw/6 - Real World Examples/IntermediateSpeedBearing.mat"
    data = read(file_path)
    print(data.shape)

# def read(file_path):
#     matdata = io.loadmat(file_path)  # 从.mat中加载所有列表数据，返回为字典类型
#     matdata = matdata['bearing']  # 访问第一层结构体
#     # print(matdata['load'][0][0].astype('float').item())
#     matdata = matdata['gs'][0]  # 访问第二层结构体
#     matdata = matdata[0] # 访问第三层结构体
#     matdata = matdata.astype(float)
#     return matdata  # 返回数据

    
