
import os
import numpy as np
import scipy.io
from .utils import load_data
from .utils import fix_byte_order

def read(file_path,*args):
    # ends = file_path.split('.')[-1]
    # data = load_data(file_path, file_type=ends)
    if file_path.endswith('.mat'):
        data = scipy.io.loadmat(file_path)['hz_1'][:,0]
            

    elif file_path.endswith('.txt'):
        data = np.loadtxt(file_path)[:,1]

    # 修复字节序问题
    data = fix_byte_order(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data

if __name__ == "__main__":
    file_path = "/home/user/data/a_bearing/a_006_THU_raw/vibration/health_bearing/1hz_1.mat"
    data = read(file_path)
    print(data.shape)
    
    file_path = "/home/user/data/a_bearing/a_006_THU_raw/voltage/health_bearing/10hz_1.txt"
    data = read(file_path)
    print(data.shape)