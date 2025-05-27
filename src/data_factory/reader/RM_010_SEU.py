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
    data = pd.read_csv(file_path, header=None,low_memory=False).values
    if data.shape[1] == 9:
        data = data[16:, [0,1,2,3,4,5,6,7]]
    elif data.shape[1] == 1:
        # 根据\t进行分割data12行及之后的内容
        data = data[12:].astype(str)
        data = np.array([line.split('\t') for line in data.flatten()])
        data = data[:, [0,1,2,3,4,5,6,7]]

    data = data.astype(float)
    return data

if __name__ == "__main__":
    # # 测试bearingset数据
    # file_list = ['ball_20_0.csv', 'ball_30_2.csv','comb_20_0.csv', 'comb_30_2.csv', 'inner_20_0.csv', 'inner_30_2.csv', 'outer_20_0.csv', 'outer_30_2.csv','health_20_0.csv', 'health_30_2.csv']
    # for file_name in file_list:
    #     file_path = f"D:/Bench_dataset/RM_010_SEU/gearbox/bearingset/{file_name}"
    #     print(file_path)
    #     data = read(file_path)
    #     print(data.shape)

    # 测试gearset数据
    file_list = ['Chipped_20_0.csv', 'Chipped_30_2.csv', 'Miss_20_0.csv', 'Miss_30_2.csv', 'Health_20_0.csv', 'Health_30_2.csv','Root_20_0.csv', 'Root_30_2.csv', 'Surface_20_0.csv', 'Surface_30_2.csv']
    for file_name in file_list:
        file_path = f"D:/Bench_dataset/RM_010_SEU/gearbox/gearset/{file_name}"
        print(file_path)
        data = read(file_path)
        print(data.shape)