import numpy as np
import csv
import pandas as pd

def read(file_path):
    """
    Reads data from a csv file specified by file_path.
    Args:
        file_path (str): Path to the csv data file.
    Returns:
        numpy.ndarray: dimension as length x channel
    """    
    # 读取数据
    data = pd.read_csv(file_path, header=None).values
    return data

if __name__ == "__main__":
    file_list = ['ib600_2.csv', 'ib800_2.csv', 'ib1000_2.csv', 'n600_3_2.csv', 'n800_3_2.csv', 'n1000_3_2.csv',
                 'ob600_2.csv', 'ob800_2.csv', 'ob1000_2.csv', 'tb600_2.csv', 'tb800_2.csv', 'tb1000_2.csv']
    for file_name in file_list:
        file_path = f"D:/Bench_dataset/RM_016_JNU/{file_name}"
        print(file_path)
        data = read(file_path)
        print(data.shape)