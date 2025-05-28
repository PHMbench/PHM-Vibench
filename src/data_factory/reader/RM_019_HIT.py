import numpy as np
import pandas as pd

def read(file_path):
    """
    Reads data from a npy file specified by file_path.
    Args:
        file_path (str): Path to the npy data file.
    Returns:
        numpy.ndarray: dimension as length x channel
    """
    # 读取数据
    data = np.load(file_path, allow_pickle=True)
   
    return data

if __name__ == "__main__":
    file_path = "D:/Bench_dataset/RM_019_HIT/data1.npy"
    data = read(file_path)
    print(data.shape)
