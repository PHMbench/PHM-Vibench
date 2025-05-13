import pandas as pd

def read(file_path,*args):
    raw_data = pd.read_csv(file_path)
    return raw_data.values

if __name__ == "__main__":
    file_path = "/home/user/data/PHMbenchdata/RM_002_XJTU/35Hz12kN/Bearing1_1/1.csv"
    data = read(file_path)
    print(data)