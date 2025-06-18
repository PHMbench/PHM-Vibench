
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
    from utils import test_reader
    test_reader(metadata_path = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/metadata_5_data.csv',
                 data_dir = '/home/user/data/PHMbenchdata/PHM-Vibench/raw',
                 name = 'RM_006_THU',
                 output_dir = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/src/data_factory/reader/output',
                 read=read)