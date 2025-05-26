import pandas as pd
from .utils import fix_byte_order

def read(file_path,*args):
    raw_data = pd.read_csv(file_path)
    data = fix_byte_order(raw_data.values)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data

if __name__ == "__main__":
    file_path = "/home/user/data/PHMbenchdata/RM_002_XJTU/35Hz12kN/Bearing1_1/1.csv"
    data = read(file_path)
    print(data)