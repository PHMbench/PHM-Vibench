## 需要将数据文件中文元素删去
import numpy as np
import csv

def read(file_path):
    """
    Reads data from a xls file specified by file_path.
    Args:
        file_path (str): Path to the xls data file (e.g., Vbench/data/RM_031_HUST24/1.xls).
    Returns:
        numpyarray: dimension as length \times channel
    """

    # 读取数据
    with open(file_path, 'r') as f:
        reader = csv.reader((line.replace('\0','') for line in f))
        data = list(reader)
        # 提取data[22]及之后的数据
        data = data[22:]
        # 将字符串按照\t分割
        data = np.array([i[0].split('\t') for i in data])
        # 将data由一维ndarray中包含list转为二维ndarray
        data = np.array([i[2:5] for i in data])
    return data

if __name__ == "__main__":
    path_ori = 'D:/Bench_dataset/RM_031_HUST24/Raw data/'
    # 使用循环创建数据集
    datasets_list_path = [[] for i in range(11)]
    frequency = ['20Hz', '25Hz', '30Hz', '35Hz', '40Hz', '60Hz', '65Hz', '70Hz', '75Hz', '80Hz', 'VS_0_40_0Hz']
    for i in range(len(frequency)):
        datasets_list_path[i] = [path_ori + 'H_' + frequency[i] + '.xls',
                            path_ori + '0.5X_I_' + frequency[i] + '.xls',
                            path_ori + 'I_' + frequency[i] + '.xls',
                            path_ori + '0.5X_O_' + frequency[i] + '.xls',
                            path_ori + 'O_' + frequency[i] + '.xls',
                            path_ori + '0.5X_B_' + frequency[i] + '.xls',
                            path_ori + 'B_' + frequency[i] + '.xls',
                            path_ori + '0.5X_C_' + frequency[i] + '.xls',
                            path_ori + 'C_' + frequency[i] + '.xls']
    for path_fre in datasets_list_path:
        for path in path_fre:
            print(path)
            data = read(path)
            print(data.shape)
