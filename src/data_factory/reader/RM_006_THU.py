
import os
import numpy as np
import scipy.io

def read(file_path):

    if file_path.endswith('.mat'):
        data = scipy.io.loadmat(file_path)['hz_1'][:,0]
            

    elif file_path.endswith('.txt'):
        data = np.loadtxt(file_path)[:,1]
    return data

if __name__ == "__main__":
    file_path = "/home/user/data/a_bearing/a_006_THU_raw/vibration/health_bearing/1hz_1.mat"
    data = read(file_path)
    print(data)
    
    file_path = "/home/user/data/a_bearing/a_006_THU_raw/voltage/health_bearing/10hz_1.txt"
    data = read(file_path)
    print(data)